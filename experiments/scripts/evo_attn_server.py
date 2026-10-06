#!/usr/bin/env python3
"""evo's pi0.5 policy server, plus opt-in attention capture for yam_replay.py --attn.

Runs evo's own scripts/serve_policy.py unchanged (same CLI, same websocket protocol). A request
with "return_attention": True also gets back "attention": {"maps_by_layer": (layers, cams, rows,
cols)} = how much the action tokens attend to each image patch, averaged over heads, action
tokens and denoising steps (sums to <1 per layer: the rest goes to text/state/action tokens).

How: pi0.5's Gemma attention computes `probs = jax.nn.softmax(...)` explicitly (evo
src/openpi/models/gemma.py). We hand gemma.py a softmax that, for denoising calls (action tokens as
queries, T != S), also streams the per-key mean to the host with jax.debug.callback (works under
jit and scan). The first 3 x 256 prefix tokens are the cameras (SigLIP 16x16 each) in IMAGE_KEYS
order: base (head), left wrist, right wrist. Images are letterboxed to 224x224, so padding rows
are cropped from the returned grids.

Run with evo's environment, from the evo repo (same arguments as serve_policy.py):
  cd /home/robo/Desktop/EW/evo
  .venv/bin/python /home/robo/Desktop/EW/molmoact2/experiments/scripts/evo_attn_server.py --port 8000 \\
      policy:checkpoint --policy.config=pi05_evo_yam_v1_bi_yam_relative_full \\
      --policy.dir=yam_evo_v1/checkpoints/pi05_evo_yam_v1_bi_yam_relative_full/yam_pi05_v1_9tasks/20000
"""

import math
import os
import sys
import types

import jax
import numpy as np

sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))  # evo/scripts/serve_policy.py
import serve_policy  # noqa: E402
import tyro  # noqa: E402
from openpi.models import gemma  # noqa: E402

CAM_KEYS = ("observation.images.head_cam", "observation.images.wrist_left", "observation.images.wrist_right")
TOKENS_PER_IMAGE, GRID, PATCH, RES = 256, 16, 14, 224
_calls: list[np.ndarray] = []


def _record(x):
    _calls.append(np.asarray(x, dtype=np.float32))


def _softmax(logits, axis=-1):
    probs = jax.nn.softmax(logits, axis=axis)
    # Gemma logits are (B, K, G, T, S). Denoising steps query with the 50 action tokens against the
    # cached prefix + themselves (T != S); the prefix pass (T == S) is skipped.
    if logits.ndim == 5 and logits.shape[3] != logits.shape[4]:
        jax.debug.callback(_record, probs[0].astype(jax.numpy.float32).mean(axis=(0, 1, 2)), ordered=True)
    return probs


# gemma.py calls `jax.nn.softmax`; give that module a `jax` whose nn.softmax is ours.
gemma.jax = types.SimpleNamespace(**{k: getattr(jax, k) for k in dir(jax) if not k.startswith("__")})
gemma.jax.nn = types.SimpleNamespace(**{k: getattr(jax.nn, k) for k in dir(jax.nn) if not k.startswith("__")})
gemma.jax.nn.softmax = _softmax


def _valid_rows_cols(h, w):
    """Patch rows/cols of the 16x16 grid that hold image (not letterbox padding)."""
    ratio = max(w / RES, h / RES)
    rh, rw = int(h / ratio), int(w / ratio)
    top, left = (RES - rh) // 2, (RES - rw) // 2
    return (slice(top // PATCH, math.ceil((top + rh) / PATCH)), slice(left // PATCH, math.ceil((left + rw) / PATCH)))


class AttentionPolicy:
    def __init__(self, policy):
        self._policy = policy

    def __getattr__(self, name):
        return getattr(self._policy, name)

    def infer(self, obs, **kwargs):
        want = bool(obs.pop("return_attention", False))
        shapes = [np.asarray(obs[k]).shape[:2] for k in CAM_KEYS]
        _calls.clear()
        out = self._policy.infer(obs, **kwargs)
        jax.effects_barrier()
        if want and _calls:
            layers = gemma.get_config("gemma_2b").depth if hasattr(gemma, "get_config") else 18
            calls = np.stack(_calls)  # (steps * layers, S), call order
            calls = calls[: len(calls) // layers * layers].reshape(-1, layers, calls.shape[-1]).mean(0)
            maps = []
            for i, (h, w) in enumerate(shapes):
                grid = calls[:, i * TOKENS_PER_IMAGE:(i + 1) * TOKENS_PER_IMAGE].reshape(layers, GRID, GRID)
                rows, cols = _valid_rows_cols(h, w)
                maps.append(grid[:, rows, cols])
            out["attention"] = {"maps_by_layer": np.stack(maps, axis=1).astype(np.float32)}
        return out


_create = serve_policy._policy_config.create_trained_policy
serve_policy._policy_config.create_trained_policy = lambda *a, **k: AttentionPolicy(_create(*a, **k))

if __name__ == "__main__":
    serve_policy.main(tyro.cli(serve_policy.Args))
