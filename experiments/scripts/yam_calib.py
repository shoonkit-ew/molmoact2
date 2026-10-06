#!/usr/bin/env python3
"""Estimate the head camera's pose relative to each YAM arm base from existing recordings.

The dataset stores camera intrinsics (meta/episode_provenance.json) but no extrinsics. We recover
them by PnP: for frames where both arms are visible, forward kinematics on the recorded joints
gives the gripper's 3D position in each arm's base frame, and you click where that gripper tip is
in the head-camera image. ~20 frames per arm, spread over the workspace, is plenty.

  # 1. click gripper tips (opens a window; progress is saved after every frame, rerun to resume)
  .venv/bin/python scripts/yam_calib.py label --repo fold_handkerchief_20260918
  # 2. fit + validate -> outputs/calib/head_cam_<repo>.json and a validation image
  .venv/bin/python scripts/yam_calib.py solve --repo fold_handkerchief_20260918

Then `scripts/yam_replay.py` picks the calibration up automatically and overlays predicted vs
recorded gripper paths on the head camera.

Kinematics come from sim_eval's YAM MuJoCo model (sim_eval/assets, via
sim_eval/scripts/download_assets.py). The tracked point is `grasp_site`: on link_6, between the
fingertips — click there, not on the finger you happen to see best.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

# av, mujoco, pyarrow and lerobot are imported where used: PyAV (also pulled in by lerobot) bundles
# FFmpeg libraries that clash with opencv's and make cv2's GUI hang without ever showing a window.
# The clicking UI therefore runs in a fresh process (`click`) that only imports cv2/numpy.

REPO_ROOT = Path(__file__).resolve().parents[2]
MJCF = REPO_ROOT / "sim_eval/assets/yam/yam_mujoco/yam.xml"
HEAD_CAM = "observation.images.head_cam"
LINKS = ["arm", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6"]
ARMS = {"left": slice(0, 6), "right": slice(7, 13)}  # 14-D state: 6 joints + gripper per arm
OUT_DIR = Path("outputs/calib")


class YamFK:
    """Forward kinematics for one YAM arm, in that arm's base frame (meters)."""

    def __init__(self, path: Path = MJCF):
        import mujoco

        self._mj = mujoco
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run: uv run python sim_eval/scripts/download_assets.py")
        self.m = mujoco.MjModel.from_xml_path(str(path))
        self.d = mujoco.MjData(self.m)
        self.qadr = [self.m.jnt_qposadr[self.m.joint(f"joint{i}").id] for i in range(1, 7)]

    def _set(self, q6):
        self.d.qpos[self.qadr] = q6
        self._mj.mj_kinematics(self.m, self.d)

    def grasp(self, q6) -> np.ndarray:
        self._set(q6)
        return self.d.site("grasp_site").xpos.copy()

    def grasp_batch(self, Q) -> np.ndarray:
        return np.stack([self.grasp(q) for q in np.asarray(Q)])

    def skeleton(self, q6) -> np.ndarray:
        """Base -> link_1..link_6 origins -> grasp point, for drawing the arm."""
        self._set(q6)
        pts = [self.d.body(name).xpos.copy() for name in LINKS]
        pts.append(self.d.site("grasp_site").xpos.copy())
        return np.stack(pts)


def head_cam_intrinsics(root: Path, episode: int):
    prov = json.loads((root / "meta/episode_provenance.json").read_text())["episodes"]
    entry = prov[episode] if isinstance(prov, list) else prov[str(episode)]
    cal = entry["recording_metadata"].get("camera_calibration", {}).get(HEAD_CAM)
    if cal is None:
        return None
    c = cal["streams"]["color"]["intrinsics"]
    return {"fx": c["fx"], "fy": c["fy"], "cx": c["cx"], "cy": c["cy"], "dist": c["dist_coeffs"],
            "serial": cal.get("serial")}


def k_matrix(intr) -> np.ndarray:
    return np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], dtype=np.float64)


def project(points_base, calib_arm, K, dist) -> np.ndarray:
    """Pixel coords; NaN for points behind the camera or well outside its view (lens distortion
    polynomials fold those back into the image as garbage)."""
    pts = np.asarray(points_base, np.float64).reshape(-1, 3)
    rvec, tvec = np.array(calib_arm["rvec"]), np.array(calib_arm["tvec"])
    uv, _ = cv2.projectPoints(pts, rvec, tvec, K, np.array(dist))
    uv = uv.reshape(-1, 2)
    cam = pts @ cv2.Rodrigues(rvec)[0].T + tvec.reshape(1, 3)
    z = np.maximum(cam[:, 2], 1e-9)
    r_max = 1.3 * np.hypot(K[0, 2] / K[0, 0], K[1, 2] / K[1, 1])  # a bit past the image corners
    uv[(cam[:, 2] < 0.05) | (np.hypot(cam[:, 0] / z, cam[:, 1] / z) > r_max)] = np.nan
    return uv


def strips(uv) -> list:
    """Split a projected polyline at NaN points into drawable runs of >= 2 points."""
    runs, cur = [], []
    for p in uv:
        if np.isfinite(p).all():
            cur.append(p)
        else:
            runs, cur = runs + [np.array(cur)] if len(cur) >= 2 else runs, []
    return runs + [np.array(cur)] if len(cur) >= 2 else runs


def load_calibration(repo: str, out_dir: Path = OUT_DIR):
    path = out_dir / f"head_cam_{repo}.json"
    if not path.exists():
        return None
    calib = json.loads(path.read_text())
    calib["K"] = np.array(calib["K"])
    return calib


def decode_frame(path: Path, t: float) -> np.ndarray:
    import av

    with av.open(str(path)) as c:
        s = c.streams.video[0]
        c.seek(int(max(t - 0.5, 0) / s.time_base), stream=s, backward=True)
        for f in c.decode(s):
            if f.time is not None and f.time >= t - 1e-3:
                return f.to_ndarray(format="rgb24")
    raise RuntimeError(f"no frame at t={t} in {path}")


def farthest_points(feats: np.ndarray, n: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(feats)))]
    dist = np.linalg.norm(feats - feats[chosen[0]], axis=1)
    for _ in range(n - 1):
        chosen.append(int(dist.argmax()))
        dist = np.minimum(dist, np.linalg.norm(feats - feats[chosen[-1]], axis=1))
    return chosen


# ---------------------------------------------------------------------------------------------- label

def cmd_label(args):
    root = Path(args.data_root) / args.repo
    out_dir = Path(args.out)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    labels_path = out_dir / f"head_cam_{args.repo}_labels.json"

    if labels_path.exists():
        labels = json.loads(labels_path.read_text())
        print(f"resuming {labels_path}: {sum(f['done'] for f in labels['frames'])}/{len(labels['frames'])} done")
    else:
        import pyarrow.dataset as pads
        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        print("selecting diverse frames (reads joint states for the whole task from the mount)...")
        fk = YamFK()
        meta = LeRobotDatasetMetadata(args.repo, root=root)
        table = pads.dataset(str(root / "data"), format="parquet").to_table(
            columns=["episode_index", "frame_index", "timestamp", "observation.state"])
        rng = np.random.default_rng(args.seed)
        rows = rng.choice(table.num_rows, size=min(args.candidates, table.num_rows), replace=False)
        ep = np.asarray(table["episode_index"].to_numpy())[rows]
        fr = np.asarray(table["frame_index"].to_numpy())[rows]
        ts = np.asarray(table["timestamp"].to_numpy())[rows]
        state = np.stack(table["observation.state"].to_numpy(zero_copy_only=False)[rows]).astype(np.float64)
        feats = np.concatenate([fk.grasp_batch(state[:, ARMS["left"]]), fk.grasp_batch(state[:, ARMS["right"]])], 1)
        picks = farthest_points(feats, args.frames, args.seed)
        labels = {"repo": args.repo, "camera": HEAD_CAM, "frames": []}
        for i, k in enumerate(picks):
            e, f = int(ep[k]), int(fr[k])
            intr = head_cam_intrinsics(root, e)
            if intr is None:
                continue
            em = meta.episodes[e]
            img = decode_frame(root / meta.get_video_file_path(e, HEAD_CAM),
                               float(em[f"videos/{HEAD_CAM}/from_timestamp"]) + float(ts[k]))
            img_path = frames_dir / f"{args.repo}_e{e}_f{f}.jpg"
            cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            labels["frames"].append({
                "episode": e, "frame": f, "image": str(img_path), "intrinsics": intr,
                "q_left": state[k, ARMS["left"]].tolist(), "q_right": state[k, ARMS["right"]].tolist(),
                "gripper_left": float(state[k, 6]), "gripper_right": float(state[k, 13]),
                "uv_left": None, "uv_right": None, "done": False,
            })
            print(f"  prepared frame {i + 1}/{len(picks)} (episode {e}, frame {f})")
        labels_path.write_text(json.dumps(labels, indent=1))

    # Fresh interpreter for the GUI: this process has PyAV loaded, which hangs cv2's window.
    print("opening the clicking window...")
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "click",
                    "--labels", str(labels_path), "--scale", str(args.scale)], check=True)
    labels = json.loads(labels_path.read_text())
    n_l = sum(f["uv_left"] is not None for f in labels["frames"])
    n_r = sum(f["uv_right"] is not None for f in labels["frames"])
    n_done = sum(f["done"] for f in labels["frames"])
    print(f"{n_done}/{len(labels['frames'])} frames done ({n_l} left / {n_r} right clicks) -> {labels_path}")
    if n_done == len(labels["frames"]):
        print(f"next: yam_calib.py solve --repo {args.repo}")


def cmd_click(args):
    labels_path = Path(args.labels)
    labels = json.loads(labels_path.read_text())
    scale = args.scale
    win = "yam head-cam calibration"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    click = {}
    cv2.setMouseCallback(win, lambda ev, x, y, *_: click.update(uv=(x / scale, y / scale))
                         if ev == cv2.EVENT_LBUTTONDOWN else None)

    todo = [f for f in labels["frames"] if not f["done"]]
    for n, item in enumerate(todo):
        base = cv2.imread(item["image"])
        cv2.resizeWindow(win, int(base.shape[1] * scale), int(base.shape[0] * scale))
        for arm in ("left", "right"):
            click.clear()
            while True:
                view = cv2.resize(base, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                for side, color in (("left", (255, 200, 0)), ("right", (255, 0, 255))):
                    if item[f"uv_{side}"] is not None:
                        u, v = (np.array(item[f"uv_{side}"]) * scale).astype(int)
                        cv2.drawMarker(view, (u, v), color, cv2.MARKER_CROSS, 24, 2)
                        cv2.putText(view, side[0].upper(), (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                grip = item[f"gripper_{arm}"]
                lines = [
                    f"frame {len(labels['frames']) - len(todo) + n + 1}/{len(labels['frames'])}"
                    f"  (episode {item['episode']}, frame {item['frame']})",
                    f"click the {arm.upper()} gripper: midpoint between its fingertips"
                    f"  [{arm} gripper is {'OPEN' if grip > 0.5 else 'CLOSED'} ({grip:.2f})]",
                    "s = not visible / skip this arm    q = save and quit",
                ]
                for i, text in enumerate(lines):
                    cv2.putText(view, text, (12, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                    cv2.putText(view, text, (12, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow(win, view)
                key = cv2.waitKey(30) & 0xFF
                if "uv" in click:
                    item[f"uv_{arm}"] = list(click["uv"])
                    break
                if key == ord("s"):
                    break
                if key == ord("q"):
                    labels_path.write_text(json.dumps(labels, indent=1))
                    cv2.destroyAllWindows()
                    print(f"saved progress to {labels_path}")
                    return
        item["done"] = True
        labels_path.write_text(json.dumps(labels, indent=1))
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------------------------- solve

def solve_arm(obj, img, K, dist):
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj, img, K, dist, iterationsCount=2000, reprojectionError=12.0, flags=cv2.SOLVEPNP_SQPNP)
    if not ok or inliers is None or len(inliers) < 6:
        raise RuntimeError("PnP failed — need >= 6 consistent clicks spread over the workspace")
    inl = inliers.ravel()
    rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K, dist, rvec, tvec)
    err = np.linalg.norm(cv2.projectPoints(obj, rvec, tvec, K, dist)[0].reshape(-1, 2) - img, axis=1)
    loo = []
    for i in inl:  # leave-one-out: how well a fit WITHOUT this click predicts it
        keep = np.array([j for j in inl if j != i])
        _, r, t = cv2.solvePnP(obj[keep], img[keep], K, dist, rvec.copy(), tvec.copy(),
                               useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        loo.append(np.linalg.norm(cv2.projectPoints(obj[i:i + 1], r, t, K, dist)[0].ravel() - img[i]))
    return rvec, tvec, inl, err, np.array(loo)


def cmd_solve(args):
    out_dir = Path(args.out)
    labels = json.loads((out_dir / f"head_cam_{args.repo}_labels.json").read_text())
    frames = [f for f in labels["frames"] if f["done"]]
    fk = YamFK()
    intr = [f["intrinsics"] for f in frames]
    K = np.mean([k_matrix(i) for i in intr], axis=0)
    dist = np.mean([i["dist"] for i in intr], axis=0)
    result = {"repo": args.repo, "camera": HEAD_CAM, "serial": intr[0].get("serial"),
              "K": K.tolist(), "dist": dist.tolist(), "arms": {}}
    poses = {}
    for arm in ("left", "right"):
        use = [f for f in frames if f[f"uv_{arm}"] is not None]
        obj = fk.grasp_batch([f[f"q_{arm}"] for f in use]).astype(np.float64)
        img = np.array([f[f"uv_{arm}"] for f in use], dtype=np.float64)
        rvec, tvec, inl, err, loo = solve_arm(obj, img, K, dist)
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tvec.ravel()
        poses[arm] = T
        cam_in_base = -R.T @ tvec.ravel()
        outliers = [(use[i]["episode"], use[i]["frame"], round(float(err[i]), 1)) for i in range(len(use)) if i not in set(inl)]
        result["arms"][arm] = {"rvec": rvec.ravel().tolist(), "tvec": tvec.ravel().tolist(), "T_cam_base": T.tolist(),
                               "n_clicks": len(use), "n_inliers": int(len(inl)),
                               "rmse_px": float(np.sqrt(np.mean(err[inl] ** 2))),
                               "loo_rmse_px": float(np.sqrt(np.mean(loo ** 2))),
                               "camera_position_in_base_m": cam_in_base.tolist()}
        print(f"{arm:5s}: {len(inl)}/{len(use)} clicks used | reprojection RMSE {result['arms'][arm]['rmse_px']:.1f}px | "
              f"leave-one-out {result['arms'][arm]['loo_rmse_px']:.1f}px | camera at {np.round(cam_in_base, 3)} m in {arm} base frame")
        if outliers:
            print(f"       rejected as misclicks (episode, frame, px error): {outliers}")
    rel = np.linalg.inv(poses["left"]) @ poses["right"]
    result["right_base_in_left_base_m"] = rel[:3, 3].tolist()
    print(f"implied right-arm base position in left-arm base frame: {np.round(rel[:3, 3], 3)} m "
          f"(distance {np.linalg.norm(rel[:3, 3]):.3f} m) — should match the physical mounting")
    path = out_dir / f"head_cam_{args.repo}.json"
    path.write_text(json.dumps(result, indent=1))

    # Validation: arm skeletons + reprojected grasp points over the labelled frames.
    tiles = []
    for f in frames[: args.preview]:
        img = cv2.imread(f["image"])
        for arm, color in (("left", (255, 200, 0)), ("right", (255, 0, 255))):
            cal = result["arms"][arm]
            sk = project(fk.skeleton(f[f"q_{arm}"]), cal, K, dist)
            cv2.polylines(img, [s.astype(np.int32) for s in strips(sk)], False, color, 2)
            if np.isfinite(sk[-1]).all():
                cv2.circle(img, tuple(sk[-1].astype(int)), 6, color, -1)
            if f[f"uv_{arm}"] is not None:
                cv2.drawMarker(img, tuple(np.array(f[f"uv_{arm}"]).astype(int)), (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
        tiles.append(cv2.resize(img, (480, 360)))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    preview = out_dir / f"head_cam_{args.repo}_validation.jpg"
    cv2.imwrite(str(preview), grid)
    print(f"wrote {path}\nwrote {preview}  (FK skeleton should sit on the real arm; green + = your click)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    lab = sub.add_parser("label", help="pick diverse frames and click gripper tips")
    lab.add_argument("--repo", required=True)
    lab.add_argument("--frames", type=int, default=24)
    lab.add_argument("--candidates", type=int, default=4000, help="random frames to choose the diverse set from")
    lab.add_argument("--scale", type=float, default=2.0, help="display zoom for precise clicking")
    lab.add_argument("--seed", type=int, default=0)
    lab.add_argument("--data-root", default=str(Path.home() / "cos-mount/data-collection/yam"))
    lab.add_argument("--out", default=str(OUT_DIR))
    sol = sub.add_parser("solve", help="fit head-camera pose per arm and write a validation image")
    sol.add_argument("--repo", required=True)
    sol.add_argument("--preview", type=int, default=12)
    sol.add_argument("--out", default=str(OUT_DIR))
    clk = sub.add_parser("click", help="(internal) clicking window; launched by label")
    clk.add_argument("--labels", required=True)
    clk.add_argument("--scale", type=float, default=2.0)
    args = p.parse_args()
    {"label": cmd_label, "solve": cmd_solve, "click": cmd_click}[args.cmd](args)


if __name__ == "__main__":
    main()
