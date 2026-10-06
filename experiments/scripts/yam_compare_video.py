#!/usr/bin/env python3
"""Side-by-side comparison video of saved yam_replay.py runs (no server needed).

Grid, one column per camera (head, left wrist, right wrist):
  row 1      trajectories: GT + every --models run drawn on the head cam (as in yam_replay --video),
             wrist cams for context
  rows 2..   one row per --attn-rows run: its attention heatmap + per-group rings on each camera
             (needs that run made with --attn, which saves the maps in its .npz)

Make the runs first, one server at a time, into one directory:
  .venv/bin/python scripts/yam_replay.py --repo R --episode N --label molmo_15k --attn --out outputs/replays_attn
  ... same for molmo_5k (step-5000 server) and evo_20k (--policy openpi, scripts/evo_attn_server.py)
Then:
  .venv/bin/python scripts/yam_compare_video.py --repo R --episode N --dir outputs/replays_attn \\
      --models molmo_5k molmo_15k evo_20k --attn-rows evo_20k molmo_5k molmo_15k
Writes <dir>/<repo>_ep<N>_compare.mp4.
"""

import argparse
from pathlib import Path

import av
import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from yam_calib import ARMS, YamFK, load_calibration, project, strips
from yam_replay import (CAMS, GT_COLOR, MODEL_COLORS, NOW_COLOR, RING_COLORS, FrameStream, attention_group_maps,
                        attention_rings, parse_layers)

PANEL = (480, 360)  # per-camera panel size (w, h)


def text(img, s, org, color=(255, 255, 255), scale=0.5):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def heat_overlay(img, m, valid):
    """Blend a group map (rows, cols; sums to 1) over the image: per-camera colour scale, alpha by heat."""
    m = np.minimum(m, np.percentile(m[valid], 97))
    m = np.where(valid, m, m[valid].min())
    m = (m - m.min()) / (m.max() - m.min() + 1e-12)
    heat = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
    a = (0.1 + 0.6 * heat)[..., None]
    return (img * (1 - a) + color * a).astype(np.uint8)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--dir", default="outputs/replays_attn", help="where the runs' .npz live; video goes here too")
    p.add_argument("--models", nargs="+", required=True, help="run labels whose trajectories go in row 1")
    p.add_argument("--attn-rows", nargs="*", default=[], help="run labels to show attention for, one row each")
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--calib", default="fold_handkerchief_20260918", help="fallback head-cam calibration")
    p.add_argument("--data-root", default=str(Path.home() / "cos-mount/data-collection/yam"))
    args = p.parse_args()

    out_dir = Path(args.dir)
    stem = f"{args.repo}_ep{args.episode}"
    runs = {}
    for label in dict.fromkeys(args.models + args.attn_rows):
        z = np.load(out_dir / f"{stem}_{label}.npz")
        runs[label] = {k: z[k] for k in z.files}
        if label in args.attn_rows and "attn_maps" not in z.files:
            p.error(f"{label}: no attention saved; rerun yam_replay.py for it with --attn")
    strides = {int(r["stride"]) for r in runs.values()}
    if len(strides) != 1:
        p.error(f"runs use different --stride values: {strides}")
    stride = strides.pop()

    root = Path(args.data_root) / args.repo
    ds = LeRobotDataset(args.repo, root=root, episodes=[args.episode], video_backend="pyav")
    rows = ds.hf_dataset.with_format("numpy")
    actions = np.stack(rows["action"]).astype(np.float32)
    states = np.stack(rows["observation.state"]).astype(np.float32)
    ep_meta = ds.meta.episodes[args.episode]
    instruction = str(next(iter(runs.values()))["instruction"])
    n_frames = len(actions) if args.max_frames is None else min(len(actions), args.max_frames)
    last_query = min(int(r["t"][-1]) for r in runs.values())
    n_frames = min(n_frames, last_query + stride)
    streams = {cam: FrameStream(root / ds.meta.get_video_file_path(args.episode, key),
                                float(ep_meta[f"videos/{key}/from_timestamp"]), ds.meta.fps)
               for cam, key in CAMS.items()}
    calib = load_calibration(args.repo) or load_calibration(args.calib)
    fk = YamFK() if calib else None
    colors = dict(zip(sorted(args.models), MODEL_COLORS))
    index = {label: {int(t): i for i, t in enumerate(r["t"])} for label, r in runs.items()}

    pw, ph = PANEL
    width, height = pw * 3, ph * (1 + len(args.attn_rows)) + 30
    path = out_dir / f"{stem}_compare.mp4"
    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=round(ds.meta.fps))
    stream.width, stream.height, stream.pix_fmt, stream.options = width, height, "yuv420p", {"crf": "21"}

    cache_t, paths, attn = None, [], {}
    for k in range(n_frames):
        frames = {cam: s.get(k) for cam, s in streams.items()}
        t = k // stride * stride
        if t != cache_t:  # new query: recompute everything drawn from it
            cache_t = t
            gt = actions[t:t + args.horizon]
            preds = {m: runs[m]["pred"][index[m][t]][: len(gt), : actions.shape[1]] for m in args.models}
            errs = {m: float(np.abs(preds[m] - gt).mean()) for m in args.models}
            hold = float(np.abs(states[t][None] - gt).mean())
            paths = []
            if calib:
                K, dist = calib["K"], calib["dist"]
                for arm in ("left", "right"):
                    cal = calib["arms"][arm]
                    for name, chunk, color in [("GT", gt, GT_COLOR)] + [(m, preds[m], colors[m]) for m in args.models]:
                        uv = project(fk.grasp_batch(chunk[:, ARMS[arm]]), cal, K, dist)
                        head = uv[-1] if np.isfinite(uv[-1]).all() else None
                        paths.append((strips(uv), head, color, f"{arm[0].upper()} {name}"))
                    now = project(fk.grasp(states[t, ARMS[arm]]), cal, K, dist)[0]
                    if np.isfinite(now).all():
                        paths.append(([], now, NOW_COLOR, ""))
            attn = {}
            for m in args.attn_rows:
                r, i = runs[m], index[m][t]
                a = {"maps_by_layer": r["attn_maps"][i].astype(np.float32)}
                if "attn_token_norm" in r:
                    a["token_norm"] = r["attn_token_norm"][i]
                layers = [int(x) for x in r["attn_layers"]]
                groups = [(g.split(":")[0], parse_layers(g.split(":")[1])) for g in str(r["attn_groups"]).split(",") if g]
                maps, valid = attention_group_maps(a, layers)
                attn[m] = (maps, valid, attention_rings(a, frames, groups), layers, groups)

        # Row 1: trajectories on the head cam; wrist cams as context.
        head = frames["top"].copy()
        for runs_uv, hd, color, label in paths:
            cv2.polylines(head, [r.astype(np.int32) for r in runs_uv], False, color, 2, cv2.LINE_AA)
            if hd is not None:
                cv2.circle(head, tuple(hd.astype(int)), 5, color, -1, cv2.LINE_AA)
                if label:
                    text(head, label, (int(hd[0]) + 7, int(hd[1]) - 7), color, 0.45)
        legend = [("now", NOW_COLOR), ("GT", GT_COLOR)] + [(m, colors[m]) for m in args.models]
        for i, (name, color) in enumerate(legend):
            cv2.rectangle(head, (10, 10 + 20 * i), (24, 24 + 20 * i), color, -1)
            text(head, name + (f"  {errs[name]:.3f}" if name in errs else ""), (30, 22 + 20 * i))
        grid = [[cv2.resize(head, PANEL)] + [cv2.resize(frames[c], PANEL) for c in ("left", "right")]]
        text(grid[0][1], "left wrist", (10, 22)); text(grid[0][2], "right wrist", (10, 22))

        # Rows 2..: attention per model.
        for m in args.attn_rows:
            maps, valid, rings, layers, groups = attn[m]
            row = []
            for ci, cam in enumerate(("top", "left", "right")):
                img = heat_overlay(frames[cam], maps[ci], valid[ci])
                for name, color, ring, centre in rings[cam]:  # no labels on the rings: the row legend names them
                    cv2.polylines(img, [ring.astype(np.int32)], True, color, 2, cv2.LINE_AA)
                    cv2.drawMarker(img, tuple(centre.astype(int)), color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
                row.append(cv2.resize(img, PANEL))
            # Row legend on a dark strip (top-left panel) so rings/heat underneath can't hide it.
            legend = [(f"{m} attention | heatmap L{layers[0]}-{layers[-1]}", (255, 255, 255))] + \
                     [(f"ring {name}: L{gl[0]}-{gl[-1]}", c) for (name, gl), c in zip(groups, RING_COLORS)]
            row[0][: 10 + 20 * len(legend)] = (row[0][: 10 + 20 * len(legend)] * 0.35).astype(np.uint8)
            for li, (s_, c) in enumerate(legend):
                cv2.putText(row[0], s_, (10, 20 + 20 * li), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
            grid.append(row)

        canvas = np.vstack([np.hstack(r) for r in grid] + [np.zeros((30, width, 3), np.uint8)])
        text(canvas, f"{instruction} | {stem} | frame {k}/{n_frames} | chunk |pred-GT| shown in legend, hold-still {hold:.3f}",
             (10, height - 10))
        for packet in stream.encode(av.VideoFrame.from_ndarray(canvas, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    for s in streams.values():
        s.close()
    print(f"wrote {path} ({n_frames} frames, {width}x{height})")


if __name__ == "__main__":
    main()
