#!/usr/bin/env python3
"""Replay a held-out bimanual-YAM episode through one or more policies and compare them in Rerun.

Open-loop: every `--stride` frames the policy gets the *recorded* cameras + joint state and predicts
an action chunk. Supported servers:
  --policy molmo    examples/yam/host_server_yam.py (HTTP /act, 30-step chunks)
  --policy openpi   evo's scripts/serve_policy.py (pi0.5, websocket, 50-step chunks)
Every run saves its predictions to <out>/<repo>_ep<N>_<label>.npz; `--compare <label> ...` overlays
earlier runs on the same episode, so models that can't share the GPU run one after another.

The viewer shows, on one scrubbable timeline (compared over the first `--horizon` steps, 30 = 1 s):
  cams/{top,left,right}        the three camera frames sent to the model
  joints/<joint>/GT            what the operator actually commanded (every frame)
  joints/<joint>/<label>       each model's chunk, first `--stride` steps (what a controller
                               executing `stride` actions per query would run)
  chunk                        GT vs every model's chunk at each query, per joint
  error/<label>, error/hold    mean |pred - GT| over the chunk vs a hold-still baseline
  cams/top/...                 with a head-cam calibration (scripts/yam_calib.py): gripper paths drawn
                               on the head camera, per arm. White dot = gripper now (tail); GREEN = GT;
                               one colour per model; the labelled dot is each path's head (~1 s later).
                               --skeleton also draws the arm.

  .venv/bin/python scripts/yam_replay.py --repo fold_handkerchief_20260918 --episode 150 --label molmo_5k
  # stop the molmo server, start evo's, then:
  .venv/bin/python scripts/yam_replay.py --repo fold_handkerchief_20260918 --episode 150 \
      --policy openpi --label evo_20k --compare molmo_5k --live
Writes <out>/<repo>_ep<N>_<labels>.rrd (reopen with `.venv/bin/rerun <file>`) and a per-joint PNG;
--video also writes <...>_top.mp4: the head camera at full frame rate with the overlay burned in.
Tasks without their own calibration reuse --calib (same head camera, check the overlay sits right).
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import av
import cv2
import json_numpy
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import rerun as rr
import rerun.blueprint as rrb
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from yam_calib import ARMS, OUT_DIR as CALIB_DIR, YamFK, load_calibration, project, strips

GT_COLOR, NOW_COLOR = (0, 230, 120), (255, 255, 255)
MODEL_COLORS = [(255, 140, 0), (235, 60, 235), (80, 150, 255), (255, 230, 0)]  # by sorted --label
SPLIT = Path(__file__).resolve().parents[1] / "launch_scripts" / "yam_data_collection_split.json"
CAMS = {
    "top": "observation.images.head_cam",
    "left": "observation.images.wrist_left",
    "right": "observation.images.wrist_right",
}


class FrameStream:
    """Sequentially decodes one camera's frames for an episode; returns frame k on request."""

    def __init__(self, path: Path, start_ts: float, fps: float):
        self.container = av.open(str(path))
        stream = self.container.streams.video[0]
        stream.thread_type = "AUTO"
        self.container.seek(int(start_ts / stream.time_base), stream=stream, backward=True)
        self.frames = self.container.decode(stream)
        self.start_ts, self.fps, self.current = start_ts, fps, None

    def get(self, k: int) -> np.ndarray:
        while self.current is None or self.current[0] < k:
            frame = next(self.frames)
            if frame.time is None:
                continue
            idx = int(round((frame.time - self.start_ts) * self.fps))
            if idx >= 0:
                self.current = (idx, frame)
        return self.current[1].to_ndarray(format="rgb24")

    def close(self):
        self.container.close()


class MolmoClient:
    """examples/yam/host_server_yam.py: json_numpy over HTTP POST."""

    def __init__(self, url, attention=False):
        self.url, self.attention, self.last_attention = url, attention, None

    def __call__(self, images, state, instruction):
        req = {f"{cam}_cam": img for cam, img in images.items()}
        req.update(instruction=instruction, state=state, return_attention=self.attention)
        resp = requests.post(self.url, data=json_numpy.dumps(req), timeout=60)
        resp.raise_for_status()
        out = json_numpy.loads(resp.text)
        self.last_attention = out.get("attention")
        return np.asarray(out["actions"], dtype=np.float32)


# Default layers, chosen on fold_handkerchief ep150 by attention on the cloth in the head cam vs its
# area share. molmo (36 action-expert layers): 10-15 favour the object (1.3-1.9x); 16-24 change
# most with the scene but sit on the arms/grippers and avoid the cloth (0.4-0.9x).
# pi0.5 (18 Gemma layers via scripts/evo_attn_server.py): 12-15 (3.3x). Override with --attn-layers.
ATTN_LAYERS = {"molmo": list(range(10, 16)), "openpi": [12, 13, 14, 15]}
ATTN_DOWNSCALE = 4


# Ring per layer group on each camera (--attn): where that group's attention peaks.
ATTN_GROUPS = {
    "molmo": "object:10-15,robot:1-7,late:19-23",  # bands measured on the head cam, fold ep150 (15k)
    "openpi": "attn:12-15",
}
RING_COLORS = [(0, 220, 255), (255, 215, 0), (255, 60, 60), (180, 120, 255)]


def parse_layers(spec):
    """'10-15' or '12,13' or '1-3,9' -> sorted layer list."""
    out = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return sorted(out)


def attention_group_maps(attention, layers):
    """(cams, rows, cols) attention map for a group of layers, each camera summing to 1.

    Each layer is normalized per camera before averaging, so layers that put more mass on images
    don't outvote the rest. molmo: SigLIP2's fixed high-norm sink tokens (>10x median norm at the
    LLM input) and the bottom row / right column (27 patches padded to 28, half-cell pooling) are
    zeroed first.
    """
    maps = np.asarray(attention["maps_by_layer"], np.float32)[layers]  # (layers, cams, rows, cols)
    valid = np.ones(maps.shape[1:], bool)
    if attention.get("token_norm") is not None:
        norm = np.asarray(attention["token_norm"], np.float32)
        valid &= norm <= 10 * np.median(norm, axis=(1, 2), keepdims=True)
        valid[:, -1, :] = False
        valid[:, :, -1] = False
    maps = np.where(valid, maps, 0)
    maps = maps / (maps.sum((-1, -2), keepdims=True) + 1e-12)
    return maps.mean(0), valid


def attention_overlays(attention, images, layers):
    """Per-camera heatmap of where the action tokens read each image (--attn).

    Colour scale per camera (clip at the 97th percentile, then min-max): the raw image share is
    small and differs by camera and layer (pi0.5: 1-6%), so absolute values aren't comparable.
    Coarse grids (molmo 14x14, pi0.5 12x16 after letterbox crop): read patterns, not single cells.
    """
    raw = np.asarray(attention["maps_by_layer"], np.float32)[layers].mean(0)
    share = raw.sum((-1, -2)) / raw.sum()
    maps, valid = attention_group_maps(attention, layers)
    out = {}
    for i, (cam, img) in enumerate(images.items()):
        m, v = maps[i], valid[i]
        m = np.minimum(m, np.percentile(m[v], 97))
        m = np.where(v, m, m[v].min())
        m = (m - m.min()) / (m.max() - m.min() + 1e-12)
        # Quarter resolution RGBA (alpha follows heat), drawn over the camera via a 4x transform:
        # keeps the viewer's memory ~16x lower than a full-size layer.
        h, w = img.shape[0] // ATTN_DOWNSCALE, img.shape[1] // ATTN_DOWNSCALE
        heat = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_TURBO)[..., ::-1]
        out[cam] = np.dstack([color, (heat * 190).astype(np.uint8)])
    return out, dict(zip(images, share))


RING_CORE = 0.7       # ring encloses the connected cells above this fraction of the group's peak
RING_MAX_CELLS = 1.5  # radius cap (grid cells): marks where attention concentrates, not its full spread


def attention_rings(attention, images, groups):
    """{cam: [(group, colour, ring polyline, centre)]} in image pixels: one small ring per group.

    Peak = maximum of the lightly smoothed group map. The ring is the equal-area ellipse of the
    connected core above RING_CORE x peak, centred on its attention-weighted centroid, radius capped
    at RING_MAX_CELLS and clipped to the frame. Cells aren't square (4:3 image on a square grid),
    hence ellipses. The centre is drawn as a crosshair.
    """
    out = {cam: [] for cam in images}
    for (name, layers), color in zip(groups, RING_COLORS):
        maps, valid = attention_group_maps(attention, layers)
        for i, (cam, img) in enumerate(images.items()):
            m = cv2.GaussianBlur(maps[i], (3, 3), 0.8) * valid[i]
            region = (m >= RING_CORE * m.max()).astype(np.uint8)
            _, labels = cv2.connectedComponents(region)
            region = labels == labels[np.unravel_index(m.argmax(), m.shape)]
            w = m * region
            rows, cols = m.shape
            cy, cx = (np.indices(m.shape) * w).sum((1, 2)) / w.sum()
            r = float(np.clip(np.sqrt(region.sum() / np.pi), 0.6, RING_MAX_CELLS))
            sx, sy = img.shape[1] / cols, img.shape[0] / rows
            centre = np.array([(cx + 0.5) * sx, (cy + 0.5) * sy])
            t = np.linspace(0, 2 * np.pi, 49)
            ring = centre + np.stack([r * sx * np.cos(t), r * sy * np.sin(t)], 1)
            ring = np.clip(ring, [0, 0], [img.shape[1] - 1, img.shape[0] - 1])
            out[cam].append((name, color, ring, centre))
    return out


def crosshair(centre, size=10.0):
    """Two short strips forming a + at centre (for rr.LineStrips2D)."""
    x, y = centre
    return [np.array([[x - size, y], [x + size, y]]), np.array([[x, y - size], [x, y + size]])]


def viewer_layout(attn):
    """Camera panels with every overlay (paths, attention) drawn in place; plots on the right."""
    cams = [rrb.Spatial2DView(origin=f"cams/{c}", name=n) for c, n in
            (("top", "head cam"), ("left", "left wrist"), ("right", "right wrist"))]
    plots = [rrb.TextDocumentView(origin="status", name="status"),
             rrb.Tabs(rrb.Spatial2DView(origin="chunk", name="chunk"),
                      rrb.TimeSeriesView(origin="joints", name="joints")),
             rrb.TimeSeriesView(origin="error", name="chunk error")]
    if attn:
        plots.append(rrb.TimeSeriesView(origin="attn_share", name="attention share per camera"))
    return rrb.Blueprint(
        rrb.Horizontal(rrb.Vertical(cams[0], rrb.Horizontal(cams[1], cams[2]), row_shares=[3, 2]),
                       rrb.Vertical(*plots, row_shares=[1, 3, 2] + ([2] if attn else [])),
                       column_shares=[3, 2]),
        collapse_panels=True)


class OpenpiClient:
    """evo/openpi scripts/serve_policy.py: msgpack over websocket (same wire format as openpi_client,
    inlined because that package pins numpy<2). Returns absolute joint positions."""

    def __init__(self, url, attention=False):
        import msgpack
        import websockets.sync.client

        self._msgpack = msgpack
        self.ws = websockets.sync.client.connect(url, compression=None, max_size=None, ping_interval=None)
        self.ws.recv()  # server metadata
        self.attention, self.last_attention = attention, None

    def _pack(self, obj):
        if isinstance(obj, np.ndarray):
            return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
        return obj

    @staticmethod
    def _unpack(obj):
        if b"__ndarray__" in obj:
            return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
        return obj

    def __call__(self, images, state, instruction):
        obs = {CAMS[cam]: np.ascontiguousarray(img) for cam, img in images.items()}
        obs.update({"observation.state": state, "prompt": instruction})
        if self.attention:  # needs scripts/evo_attn_server.py, not plain serve_policy.py
            obs["return_attention"] = True
        self.ws.send(self._msgpack.packb(obs, default=self._pack))
        resp = self.ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"policy server error:\n{resp}")
        out = self._msgpack.unpackb(resp, object_hook=self._unpack)
        self.last_attention = out.get("attention")
        return np.asarray(out["actions"], dtype=np.float32)


def chunk_figure(names, models):
    fig, axes = plt.subplots(2, 7, figsize=(16, 4.2), sharex=True)
    lines = []
    for ax, name in zip(axes.flat, names):
        gt, = ax.plot([], [], color="0.4", lw=2, label="GT")
        preds = [ax.plot([], [], color=np.array(c) / 255, lw=1.5, label=m)[0] for m, c in models]
        ax.set_title(name.replace(".pos", ""), fontsize=8)
        ax.tick_params(labelsize=6)
        lines.append((ax, gt, preds))
    axes.flat[0].legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    return fig, lines


def render_chunk(fig, lines, gt, preds):
    for j, (ax, gt_line, pred_lines) in enumerate(lines):
        gt_line.set_data(np.arange(len(gt)), gt[:, j])
        for line, pred in zip(pred_lines, preds):
            line.set_data(np.arange(len(pred)), pred[:, j])
        vals = np.concatenate([gt[:, j]] + [p[:, j] for p in preds])
        lo, hi = vals.min(), vals.max()
        pad = max(0.02, 0.1 * (hi - lo))
        ax.set_xlim(0, len(gt) - 1)
        ax.set_ylim(lo - pad, hi + pad)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


class OverlayVideo:
    """Head-camera mp4 with the latest query's paths drawn on every frame (H.264 via PyAV)."""

    def __init__(self, path, fps, legend):
        self.path, self.fps, self.legend = path, fps, legend
        self.out = self.stream = None

    def write(self, img, paths, caption):
        """paths: [(polyline runs, head point or None, RGB colour, head label)], in drawing order."""
        img = np.ascontiguousarray(img)
        if self.out is None:
            self.out = av.open(str(self.path), "w")
            self.stream = self.out.add_stream("libx264", rate=round(self.fps))
            self.stream.width, self.stream.height = img.shape[1], img.shape[0]
            self.stream.pix_fmt, self.stream.options = "yuv420p", {"crf": "20"}
        for runs, head, color, label in paths:
            cv2.polylines(img, [r.astype(np.int32) for r in runs], False, color, 2, cv2.LINE_AA)
            if head is not None:
                hx, hy = head.astype(int)
                cv2.circle(img, (hx, hy), 5, color, -1, cv2.LINE_AA)
                if label:
                    _text(img, label, (hx + 7, hy - 7), color, 0.45)
        for i, (name, color) in enumerate(self.legend):
            cv2.rectangle(img, (10, 10 + 20 * i), (24, 24 + 20 * i), color, -1)
            _text(img, name, (30, 22 + 20 * i), (255, 255, 255), 0.5)
        h = img.shape[0]
        img[h - 30:] = (img[h - 30:] * 0.35).astype(np.uint8)  # dark strip so the caption stays readable
        cv2.putText(img, caption, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for pkt in self.stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
            self.out.mux(pkt)

    def close(self):
        if self.out is not None:
            for pkt in self.stream.encode():
                self.out.mux(pkt)
            self.out.close()


def _text(img, text, org, color, scale):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="list tasks and their held-out episodes, then exit")
    p.add_argument("--repo", help="task dataset folder, e.g. fold_handkerchief_20260918")
    p.add_argument("--episode", type=int, default=None, help="default: first held-out (val) episode")
    p.add_argument("--instruction", default=None, help="default: the episode's own task string")
    p.add_argument("--policy", choices=["molmo", "openpi"], default="molmo", help="which server type to query")
    p.add_argument("--server", default=None,
                   help="default: http://127.0.0.1:8202/act (molmo) or ws://127.0.0.1:8000 (openpi)")
    p.add_argument("--label", default=None, help="name for this model in the viewer and the saved .npz "
                   "(e.g. molmo_5k, evo_20k); default: the --policy value")
    p.add_argument("--compare", nargs="*", default=[], metavar="LABEL",
                   help="overlay saved predictions from earlier runs on this episode (same --stride)")
    p.add_argument("--horizon", type=int, default=30, help="chunk steps compared and drawn (30 = 1 s)")
    p.add_argument("--stride", type=int, default=10, help="query every N frames (30fps data)")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--data-root", default=str(Path.home() / "cos-mount/data-collection/yam"))
    p.add_argument("--out", default="outputs/replays")
    p.add_argument("--live", action="store_true", help="open the Rerun viewer and stream into it")
    p.add_argument("--no-overlay", action="store_true",
                   help="skip the FK overlay even if outputs/calib/head_cam_<repo>.json exists")
    p.add_argument("--skeleton", action="store_true", help="also draw each arm's FK skeleton (white)")
    p.add_argument("--calib", default="fold_handkerchief_20260918",
                   help="calibration to use when the task has no head_cam_<repo>.json of its own")
    p.add_argument("--video", action="store_true", help="also write the head camera + overlay as an mp4")
    p.add_argument("--attn-layers", default=None,
                   help="layers to average for --attn, e.g. 16-24 or 12,13 (default: per policy, see ATTN_LAYERS)")
    p.add_argument("--attn-groups", default=None,
                   help='rings per layer group for --attn, "name:layers,..." e.g. "object:10-15,late:19-23"; '
                        '"" for none (default: per policy, see ATTN_GROUPS)')
    p.add_argument("--attn", action="store_true",
                   help="overlay where the action tokens attend in each camera; openpi needs "
                        "scripts/evo_attn_server.py instead of serve_policy.py")
    args = p.parse_args()

    if args.list:
        for repo, s in json.loads(SPLIT.read_text())["repos"].items():
            val, n = s["val"], s["total_episodes"]
            # evo's pi0.5 split (openpi data_loader._split_episodes, seed 42, 5%) — assumes its _v21
            # copy of the task keeps this episode order.
            evo_val = set(np.random.default_rng(42).permutation(n)[: round(n * 0.05)].tolist())
            both = [e for e in val if e in evo_val]
            print(f"{repo:32s} {len(val):3d} held-out of {n:4d}  held out by evo too: {both or '-'}  "
                  f"val: {', '.join(map(str, val[:10]))}{' ...' if len(val) > 10 else ''}")
        return
    if not args.repo:
        p.error("--repo is required (see --list)")

    root = Path(args.data_root) / args.repo
    split = json.loads(SPLIT.read_text())["repos"].get(args.repo)
    episode = args.episode if args.episode is not None else split["val"][0]
    if split and episode in split["train"]:
        print(f"WARNING: episode {episode} was in the TRAINING split — this measures fit, not generalization")

    ds = LeRobotDataset(args.repo, root=root, episodes=[episode], video_backend="pyav")
    rows = ds.hf_dataset.with_format("numpy")
    actions = np.stack(rows["action"]).astype(np.float32)
    states = np.stack(rows["observation.state"]).astype(np.float32)
    names = ds.meta.features["action"]["names"]
    ep_meta = ds.meta.episodes[episode]
    instruction = args.instruction or ep_meta["tasks"][0]
    n_frames = len(actions) if args.max_frames is None else min(len(actions), args.max_frames)
    fps = ds.meta.fps

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or args.policy
    stem = f"{args.repo}_ep{episode}"
    cached = {}
    for other in args.compare:
        path = out_dir / f"{stem}_{other}.npz"
        if not path.exists():
            p.error(f"no saved run {path} — replay that model first with --label {other}")
        z = np.load(path)
        if int(z["stride"]) != args.stride:
            p.error(f"{path} was recorded with --stride {int(z['stride'])}; use the same stride")
        cached[other] = dict(zip(z["t"].tolist(), z["pred"]))
    # Sorted, so a model keeps its colour whichever order the runs were made in.
    labels = sorted({*args.compare, label})
    models = list(zip(labels, MODEL_COLORS))
    run_name = f"{stem}_{'_vs_'.join(labels)}"
    rr.init("yam_replay", recording_id=run_name)
    rrd_path = str(out_dir / f"{run_name}.rrd")
    if args.live:
        # The viewer ships in this venv's bin/ (pip rerun-sdk), which isn't on PATH unless the
        # venv is activated; rr.spawn() only searches PATH.
        os.environ["PATH"] = f"{Path(sys.executable).parent}{os.pathsep}{os.environ.get('PATH', '')}"
        rr.spawn(connect=False)
        # Each sink call (spawn/save/connect) *replaces* the previous one; set_sinks sends to both.
        rr.set_sinks(rr.GrpcSink(), rr.FileSink(rrd_path))
    else:
        rr.save(rrd_path)
    rr.send_blueprint(viewer_layout(args.attn))
    if args.attn:
        for cam in CAMS:  # attention layers are logged small; scale them onto the camera pixels
            rr.log(f"cams/{cam}/attention", rr.Transform3D(scale=[ATTN_DOWNSCALE, ATTN_DOWNSCALE, 1]), static=True)

    for k in range(n_frames):  # full recorded trajectory, every frame
        rr.set_time("frame", sequence=k)
        for j, name in enumerate(names):
            rr.log(f"joints/{name}/GT", rr.Scalars(actions[k, j]))

    streams = {
        cam: FrameStream(root / ds.meta.get_video_file_path(episode, key),
                         float(ep_meta[f"videos/{key}/from_timestamp"]), fps)
        for cam, key in CAMS.items()
    }
    server = args.server or ("http://127.0.0.1:8202/act" if args.policy == "molmo" else "ws://127.0.0.1:8000")
    attn_layers = parse_layers(args.attn_layers) if args.attn_layers else ATTN_LAYERS[args.policy]
    attn_groups = [(g.split(":")[0], parse_layers(g.split(":")[1]))
                   for g in (ATTN_GROUPS[args.policy] if args.attn_groups is None else args.attn_groups).split(",") if g]
    if args.policy == "molmo":
        policy = MolmoClient(server, attention=args.attn)
    else:
        policy = OpenpiClient(server, attention=args.attn)
    fig, lines = chunk_figure(names, models)
    executed = {m: np.full_like(actions, np.nan) for m in labels}
    err = {m: [] for m in ["hold", *labels]}
    raw_t, raw_pred, latencies = [], [], []
    raw_attn, raw_norm = [], []  # --attn: per-query maps_by_layer / token_norm, saved for yam_compare_video.py
    calib, calib_note = None, "off (--no-overlay)"
    if not args.no_overlay:
        calib, calib_note = load_calibration(args.repo), "on"
        if calib is None and args.calib:
            calib, calib_note = load_calibration(args.calib), f"on, borrowed from {args.calib}"
        if calib is None:
            calib_note = f"off (no {CALIB_DIR}/head_cam_<repo>.json)"
    fk = YamFK() if calib else None
    video = OverlayVideo(out_dir / f"{run_name}_top.mp4", fps,
                         [("now", NOW_COLOR), ("GT", GT_COLOR), *models]) if args.video else None
    print(f"replaying {args.repo} episode {episode} ({n_frames} frames, query every {args.stride}) "
          f"instruction={instruction!r} | {label} via {server}"
          f"{' | comparing ' + ', '.join(args.compare) if args.compare else ''} | "
          f"head-cam overlay: {calib_note}")

    for t in range(0, n_frames, args.stride):
        images = {cam: st.get(t) for cam, st in streams.items()}
        t0 = time.perf_counter()
        pred = policy(images, states[t], instruction)[:, : actions.shape[1]]
        latencies.append(time.perf_counter() - t0)
        raw_t.append(t)
        if args.attn and policy.last_attention is not None:
            raw_attn.append(np.asarray(policy.last_attention["maps_by_layer"], np.float16))
            if policy.last_attention.get("token_norm") is not None:
                raw_norm.append(np.asarray(policy.last_attention["token_norm"], np.float32))
        raw_pred.append(pred)
        gt = actions[t : t + args.horizon]
        preds = {}
        for m in labels:
            chunk = pred if m == label else cached[m].get(t)
            if chunk is None:
                p.error(f"saved run {m} has no query at frame {t} (was it shorter? check --max-frames)")
            preds[m] = chunk[: len(gt)]
        # A model with a shorter chunk than --horizon is compared over its own length only.
        for m in labels:
            err[m].append(float(np.abs(preds[m] - gt[: len(preds[m])]).mean()))
        err["hold"].append(float(np.abs(states[t][None] - gt).mean()))

        rr.set_time("frame", sequence=t)
        for cam, img in images.items():
            rr.log(f"cams/{cam}", rr.Image(img).compress(jpeg_quality=80))
        if args.attn and policy.last_attention is not None:
            overlays, share = attention_overlays(policy.last_attention, images, attn_layers)
            for cam, rgba in overlays.items():
                rr.log(f"cams/{cam}/attention", rr.Image(rgba, draw_order=5.0))
                rr.log(f"attn_share/{cam}", rr.Scalars(float(share[cam])))
            for cam, rings in attention_rings(policy.last_attention, images, attn_groups).items():
                for name, color, ring, centre in rings:
                    rr.log(f"cams/{cam}/rings/{name}",
                           rr.LineStrips2D([ring] + crosshair(centre), colors=[color], radii=2.0, draw_order=10.0))
        video_paths = []  # same overlay, for the mp4
        if calib:
            K, dist = calib["K"], calib["dist"]
            for arm in ("left", "right"):
                cal = calib["arms"][arm]
                now_uv = project(fk.grasp(states[t, ARMS[arm]]), cal, K, dist)
                if args.skeleton:
                    arm_runs = strips(project(fk.skeleton(states[t, ARMS[arm]]), cal, K, dist))
                    rr.log(f"cams/top/{arm}/arm", rr.LineStrips2D(arm_runs, colors=[NOW_COLOR], radii=1.5))
                    video_paths.append((arm_runs, None, NOW_COLOR, ""))
                # Empty point lists when the gripper is out of view (NaN projection).
                now = now_uv[np.isfinite(now_uv).all(1)]
                rr.log(f"cams/top/{arm}/now", rr.Points2D(now, colors=[NOW_COLOR], radii=4))
                for m, chunk, color in [("GT", gt, GT_COLOR)] + [(m, preds[m], c) for m, c in models]:
                    uv = project(fk.grasp_batch(chunk[:, ARMS[arm]]), cal, K, dist)
                    runs = strips(uv)
                    rr.log(f"cams/top/{arm}/{m}", rr.LineStrips2D(runs, colors=[color], radii=2.5))
                    head = uv[-1:][np.isfinite(uv[-1:]).all(1)]
                    rr.log(f"cams/top/{arm}/{m}_head", rr.Points2D(head, colors=[color], radii=6,
                                                                   labels=[f"{arm[0].upper()} {m}"] * len(head)))
                    video_paths.append((runs, head[0] if len(head) else None, color, f"{arm[0].upper()} {m}"))
                if len(now):
                    video_paths.append(([], now[0], NOW_COLOR, ""))
        if video:
            caption = (f"{instruction} | frame {{}} | " + "  ".join(f"{m} {err[m][-1]:.3f}" for m in labels)
                       + f"  hold {err['hold'][-1]:.3f}")
            for k in range(t, min(t + args.stride, n_frames)):
                frame = images["top"] if k == t else streams["top"].get(k)
                video.write(frame.copy(), video_paths, caption.format(k))
        rr.log("chunk", rr.Image(render_chunk(fig, lines, gt, [preds[m] for m in labels])))
        for m in ["hold", *labels]:
            rr.log(f"error/{m}", rr.Scalars(err[m][-1]))
        rr.log("status", rr.TextDocument(
            f"**{instruction}** — frame {t}/{n_frames}\n\n"
            + "   ".join(f"{m} {err[m][-1]:.4f}" for m in labels)
            + f"   hold-still {err['hold'][-1]:.4f}   (mean |pred−GT|)   latency {latencies[-1]*1e3:.0f} ms",
            media_type=rr.MediaType.MARKDOWN))
        for m in labels:
            for k in range(min(args.stride, len(preds[m]))):
                executed[m][t + k] = preds[m][k]
                rr.set_time("frame", sequence=t + k)
                for j, name in enumerate(names):
                    rr.log(f"joints/{name}/{m}", rr.Scalars(preds[m][k, j]))

    if video:
        video.close()
    for st in streams.values():
        st.close()
    plt.close(fig)
    extra = {}
    if raw_attn:
        extra["attn_maps"] = np.stack(raw_attn)        # (queries, layers, cams, rows, cols)
        extra["attn_layers"] = np.array(attn_layers)   # heatmap group used in this run
        extra["attn_groups"] = np.array(ATTN_GROUPS[args.policy] if args.attn_groups is None else args.attn_groups)
    if raw_norm:
        extra["attn_token_norm"] = np.stack(raw_norm)  # (queries, cams, rows, cols)
    np.savez(out_dir / f"{stem}_{label}.npz", t=np.array(raw_t), pred=np.stack(raw_pred), stride=args.stride,
             latency=np.array(latencies), instruction=instruction, server=server, **extra)

    # Static summary: GT vs each model's executed actions per joint over the whole episode.
    fig, axes = plt.subplots(7, 2, figsize=(14, 16), sharex=True)
    for j, name in enumerate(names):
        ax = axes[j % 7, j // 7]
        ax.plot(actions[:n_frames, j], color="0.4", lw=1.5, label="GT")
        for m, c in models:
            ax.plot(executed[m][:n_frames, j], color=np.array(c) / 255, lw=1, label=m)
        ax.set_title(name, fontsize=9)
    axes[0, 0].legend(fontsize=8)
    scores = "  ".join(f"{m} {np.mean(err[m]):.4f}" for m in labels)
    fig.suptitle(f"{args.repo} ep{episode} — {instruction!r} — mean chunk |pred−GT|: {scores}  "
                 f"hold {np.mean(err['hold']):.4f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"{run_name}.png", dpi=110)

    print(f"done: {len(raw_t)} queries, {label} median latency {np.median(latencies)*1e3:.0f} ms, "
          f"chunk {raw_pred[0].shape[0]} steps (compared over {args.horizon})")
    hold = np.array(err["hold"])
    for m in labels:
        e = np.array(err[m])
        print(f"  {m:12s} mean |pred-GT| {e.mean():.4f}  beats hold-still ({hold.mean():.4f}) on {np.mean(e < hold):.0%} of queries")
    print(f"wrote {rrd_path}, {out_dir / (run_name + '.png')}, {out_dir / (stem + '_' + label + '.npz')}"
          + (f", {video.path}" if video else ""))

if __name__ == "__main__":
    main()
