"""MolmoAct2-BimanualYAM inference server.

Mirrors `host_server.py` but for the bimanual YAM checkpoint:

  * 3 cameras in fixed order [top, left, right]
  * raw robot state is shape (14,)  (per-arm 7-D, two arms)
  * norm_tag = "yam_dual_molmoact2"

Wire protocol:

    GET  /act        -> health check, returns {"status": "ok", ...}
    POST /act        -> action inference
        request body  (json_numpy):
            {
              "top_cam":     ndarray(H, W, 3) uint8 RGB,
              "left_cam":    ndarray(H, W, 3) uint8 RGB,
              "right_cam":   ndarray(H, W, 3) uint8 RGB,
              "instruction": str,
              "state":       ndarray(14,) float32,
              "timestamp":   float (optional),
              "num_steps":   int   (optional, default 10),
              "enable_cuda_graph": bool (optional),
            }
        response body (json_numpy):
            {"actions": ndarray(N, D) float32, "dt_ms": float}

Run:

    uv run python host_server_yam.py --host 0.0.0.0 --port 8202
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from typing import Any

import json_numpy
import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from huggingface_hub import snapshot_download
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

# Patches the stdlib `json` module so np.ndarray round-trips through JSON.
# Must be called before any json.dumps/loads we rely on.
json_numpy.patch()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("molmoact2.yam.server")


REPO_ID = "allenai/MolmoAct2-BimanualYAM"
NORM_TAG = "yam_dual_molmoact2"
STATE_DIM = 14
NUM_CAMERAS = 3
DEFAULT_NUM_STEPS = 10


def _patch_modeling_for_bf16(local_dir: str) -> None:
    """Same idempotent patches as the DROID server. The dtype needle may no
    longer match newer revisions of `modeling_molmoact2.py` (and will warn
    rather than fail); `_to_array` is still required for bf16.
    """
    patches = [
        (
            "device=device,\n            dtype=torch.float32,\n            generator=generator,",
            "device=device,\n"
            "            dtype=source_tensor.dtype,  # patched_bf16_dtype\n"
            "            generator=generator,",
            "patched_bf16_dtype",
        ),
        (
            "return value.detach().cpu().numpy().astype(np.float32, copy=False)",
            "return value.detach().cpu().float().numpy().astype(np.float32, copy=False)  # patched_bf16_to_array",
            "patched_bf16_to_array",
        ),
    ]
    candidates = [os.path.join(local_dir, "modeling_molmoact2.py")]
    modules_root = os.path.expanduser(
        "~/.cache/huggingface/modules/transformers_modules"
    )
    if os.path.isdir(modules_root):
        for sub in os.listdir(modules_root):
            p = os.path.join(modules_root, sub, "modeling_molmoact2.py")
            if os.path.isfile(p):
                candidates.append(p)
    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                src = f.read()
        except OSError:
            continue
        new_src = src
        applied: list[str] = []
        for needle, replacement, marker in patches:
            if marker in new_src:
                continue
            if needle not in new_src:
                log.warning("patch %s: needle not found in %s", marker, path)
                continue
            new_src = new_src.replace(needle, replacement, 1)
            applied.append(marker)
        if new_src != src:
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_src)
            log.info("Applied patches %s in %s", applied, path)


class AttentionCapture:
    """Opt-in: what the action expert's cross-attention reads from the VLM's image tokens.

    The action expert attends to the VLM's per-layer keys/values (prompt = 3 camera images + text +
    state). `_attention` uses a fused SDPA kernel that doesn't return weights, so while `active` we
    recompute softmax(q.k) for the same inputs, average over heads and action positions, and keep
    one (S,) vector per (denoising step, layer). `input_ids` is grabbed from the call that feeds the
    expert so image-patch positions (config.image_patch_id, in camera order) can be sliced out.
    """

    def __init__(self, model: Any, num_cameras: int) -> None:
        self.active = False
        self.num_cameras = num_cameras
        self.image_patch_id = int(model.config.image_patch_id)
        self.calls: list[torch.Tensor] = []
        self.calls_vw: list[torch.Tensor] = []
        self.input_ids: torch.Tensor | None = None
        cross = [m for m in model.modules() if type(m).__name__ == "ActionExpertCrossAttention"]
        if not cross:
            raise RuntimeError("no ActionExpertCrossAttention found; attention capture unsupported")
        self.num_layers = len(cross)
        self.layer_of = {id(m): i for i, m in enumerate(cross)}  # module -> block index (model order)
        self.layers: list[int] = []
        cls, orig, cap = type(cross[0]), type(cross[0])._attention, self

        def attention(self_, q, k, v, *, attn_mask=None):  # q: (B,T,H,D)  k: (B,S,H,D)
            if cap.active:
                scores = torch.einsum("bthd,bshd->bhts", q.float(), k.float()) * q.shape[-1] ** -0.5
                if attn_mask is not None:
                    scores = scores + attn_mask.float()
                w = scores.softmax(-1)  # (B,H,T,S)
                cap.layers.append(cap.layer_of[id(self_)])
                cap.calls.append(w.mean(dim=(1, 2))[0])  # (S,)
                # Value-norm weighting: sink tokens soak up attention but carry ~zero value, so
                # attention x |v| shows what the output is actually made of.
                wv = w * v.float().norm(dim=-1).transpose(1, 2)[:, :, None, :]  # (B,H,T,S)
                cap.calls_vw.append((wv / wv.sum(-1, keepdim=True)).mean(dim=(1, 2))[0])
            return orig(self_, q, k, v, attn_mask=attn_mask)

        cls._attention = attention
        # Norm of each image token as it enters the LLM: SigLIP2 forms a few fixed-position
        # high-norm "register" patches (left edge, ~3500x the median after the projector) that act
        # as attention sinks; the replay masks cells flagged here before drawing.
        def vision_hook(mod, inp, out):
            if cap.active:
                feats = out[0] if isinstance(out, tuple) else out
                cap.token_norm = feats.detach().float().reshape(-1, feats.shape[-1]).norm(dim=-1)

        model.model.vision_backbone.register_forward_hook(vision_hook)
        self.token_norm: torch.Tensor | None = None
        inner, orig_gen = model.model, model.model.generate_actions_from_inputs

        def generate(*args, **kwargs):
            if cap.active:
                cap.input_ids = kwargs["input_ids"].detach()
            return orig_gen(*args, **kwargs)

        inner.generate_actions_from_inputs = generate

    def begin(self) -> None:
        self.calls, self.calls_vw, self.layers, self.input_ids, self.active = [], [], [], None, True
        self.token_norm = None

    def finish(self) -> dict[str, Any]:
        """Image attention per camera: maps (cams, g, g) of attention mass (all tokens sum to 1,
        so a map's total is that camera's share), averaged over denoising steps and layers, plus
        share_by_layer (layers, cams)."""
        self.active = False
        pos = (self.input_ids[0] == self.image_patch_id).nonzero().squeeze(-1)
        per_cam = pos.numel() // self.num_cameras
        grid = int(round(per_cam**0.5))
        assert grid * grid * self.num_cameras == pos.numel(), f"image tokens {pos.numel()} not cams x square"

        layer_idx = torch.tensor(self.layers, device=self.calls[0].device)

        def image_part(calls):
            # Average each block's calls over denoising steps, grouped by which block made them.
            calls = torch.stack(calls)[:, pos]  # (n_calls, image tokens)
            per_layer = torch.stack([calls[layer_idx == i].mean(0) for i in range(self.num_layers)])
            return per_layer.view(self.num_layers, self.num_cameras, grid, grid)

        raw, vw = image_part(self.calls), image_part(self.calls_vw)
        return {
            "maps": raw.mean(0).cpu().numpy().astype(np.float32),                  # (cams, g, g)
            "maps_vw": vw.mean(0).cpu().numpy().astype(np.float32),                # value-norm weighted
            "share_by_layer": raw.sum((-1, -2)).cpu().numpy().astype(np.float32),  # (layers, cams)
            "maps_by_layer": raw.cpu().numpy().astype(np.float16),                 # (layers, cams, g, g)
            "calls_per_layer": np.bincount(self.layers, minlength=self.num_layers),  # = denoising steps
            "token_norm": (None if self.token_norm is None or self.token_norm.numel() != pos.numel() else
                           self.token_norm.view(self.num_cameras, grid, grid).cpu().numpy().astype(np.float32)),
        }


class Policy:
    """Holds the loaded model + processor and serializes inference calls."""

    def __init__(
        self,
        repo_id: str,
        device: str,
        dtype: torch.dtype,
        enable_cuda_graph: bool = False,
        norm_tag: str = NORM_TAG,
    ) -> None:
        self.default_cuda_graph = enable_cuda_graph
        self.repo_id = repo_id
        self.norm_tag = norm_tag
        # `predict_action` reads `norm_stats.json` from `config._name_or_path`.
        # Always resolve to a local dir so that lookup works: either a converted
        # fine-tune checkpoint already on disk, or the HF snapshot.
        local_dir = repo_id if os.path.isdir(repo_id) else snapshot_download(repo_id=repo_id)
        log.info("Resolved model dir: %s (norm_tag=%s)", local_dir, norm_tag)

        _patch_modeling_for_bf16(local_dir)

        log.info("Loading processor")
        # `tokenizer_config.json` ships `extra_special_tokens` as a list, which
        # transformers >=4.46 rejects. The model code only uses these via
        # `convert_tokens_to_ids`, so an empty dict is safe.
        self.processor = AutoProcessor.from_pretrained(
            local_dir, trust_remote_code=True, extra_special_tokens={}
        )

        log.info("Loading model (dtype=%s, device=%s)", dtype, device)
        self.model = (
            AutoModelForImageTextToText.from_pretrained(
                local_dir,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
            .to(device)
            .eval()
        )
        self.device = device

        # Upstream `_move_inputs_to_device` only moves tensors; it does not
        # cast floats to the model dtype. With bf16 weights the processor's
        # fp32 `pixel_values` then trips `mat1 and mat2 must have the same
        # dtype`. Replace the bound method per-instance.
        target_dtype = next(self.model.parameters()).dtype

        def _move_and_cast(
            inputs: Any, dev: Any, _target: torch.dtype = target_dtype
        ) -> dict[str, Any]:
            out: dict[str, Any] = {}
            for key, value in inputs.items():
                if torch.is_tensor(value):
                    value = value.to(dev)
                    if value.is_floating_point() and value.dtype != _target:
                        value = value.to(_target)
                out[key] = value
            return out

        self.model._move_inputs_to_device = _move_and_cast
        # CUDA-graph capture in the action expert is not safe under concurrent
        # calls; coarse-grained serialization is fine at ~5 Hz robot poll.
        self._lock = threading.Lock()
        self._capture: AttentionCapture | None = None

    @torch.inference_mode()
    def predict(
        self,
        top_cam: np.ndarray,
        left_cam: np.ndarray,
        right_cam: np.ndarray,
        instruction: str,
        state: np.ndarray,
        num_steps: int = DEFAULT_NUM_STEPS,
        enable_cuda_graph: bool = False,
        return_attention: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
        # Camera order must match training: [top, left, right].
        images = [_to_pil(top_cam), _to_pil(left_cam), _to_pil(right_cam)]
        state_f32 = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_f32.shape != (STATE_DIM,):
            raise ValueError(
                f"state must be shape ({STATE_DIM},), got {state_f32.shape}"
            )

        with self._lock:
            if return_attention:
                if self._capture is None:
                    self._capture = AttentionCapture(self.model, NUM_CAMERAS)
                self._capture.begin()
                enable_cuda_graph = False  # graphs replay a captured kernel; the hook would not run
            try:
                out = self._predict_action(images, instruction, state_f32, num_steps, enable_cuda_graph)
            finally:
                attention = self._capture.finish() if return_attention and self._capture.calls else None
        raw = out.actions
        if torch.is_tensor(raw):
            raw = raw.detach().to(dtype=torch.float32, device="cpu").numpy()
        actions = np.asarray(raw, dtype=np.float32)
        if actions.ndim == 3 and actions.shape[0] == 1:
            actions = actions[0]
        return (actions, attention) if return_attention else actions

    def _predict_action(self, images, instruction, state_f32, num_steps, enable_cuda_graph):
        return self.model.predict_action(
            processor=self.processor,
            images=images,
            task=instruction,
            state=state_f32,
            norm_tag=self.norm_tag,
            inference_action_mode="continuous",
            enable_depth_reasoning=False,
            num_steps=num_steps,
            normalize_language=True,
            enable_cuda_graph=enable_cuda_graph,
        )


def _to_pil(arr: Any) -> Image.Image:
    if isinstance(arr, Image.Image):
        return arr.convert("RGB")
    a = np.asarray(arr)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"image must be HxWx3, got shape {a.shape}")
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return Image.fromarray(a, mode="RGB")


def build_app(policy: Policy) -> FastAPI:
    app = FastAPI(title="MolmoAct2-BimanualYAM server", version="0.1.0")

    @app.get("/act")
    async def health() -> JSONResponse:
        return JSONResponse(
            {
                "status": "ok",
                "repo_id": policy.repo_id,
                "norm_tag": policy.norm_tag,
                "device": policy.device,
                "dtype": str(policy.model.dtype),
                "num_cameras": NUM_CAMERAS,
                "state_dim": STATE_DIM,
            }
        )

    @app.get("/healthz")
    async def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @app.post("/act")
    async def act(request: Request) -> Response:
        raw = await request.body()
        try:
            payload = json_numpy.loads(raw.decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            return _error_response(400, f"failed to decode json_numpy body: {e}")

        try:
            top_cam = payload["top_cam"]
            left_cam = payload["left_cam"]
            right_cam = payload["right_cam"]
            instruction = str(payload["instruction"])
            state = payload["state"]
        except KeyError as e:
            return _error_response(400, f"missing required field: {e}")

        num_steps = int(payload.get("num_steps", DEFAULT_NUM_STEPS))
        enable_cuda_graph = bool(
            payload.get("enable_cuda_graph", policy.default_cuda_graph)
        )

        return_attention = bool(payload.get("return_attention", False))
        t0 = time.perf_counter()
        try:
            result = policy.predict(
                top_cam=top_cam,
                left_cam=left_cam,
                right_cam=right_cam,
                instruction=instruction,
                state=state,
                num_steps=num_steps,
                enable_cuda_graph=enable_cuda_graph,
                return_attention=return_attention,
            )
        except Exception as e:  # noqa: BLE001
            log.exception("inference failed")
            return _error_response(500, f"inference failed: {e}")
        dt_ms = (time.perf_counter() - t0) * 1000.0

        actions, attention = result if return_attention else (result, None)
        reply = {"actions": actions, "dt_ms": dt_ms}
        if attention is not None:
            reply["attention"] = attention
        body = json_numpy.dumps(reply)
        return Response(content=body, media_type="application/json")

    return app


def _error_response(status: int, message: str) -> Response:
    body = json_numpy.dumps({"error": message})
    return Response(content=body, status_code=status, media_type="application/json")


def warmup(policy: Policy) -> None:
    log.info("Warming up model with dummy frames (cuda_graph=%s) ...",
             policy.default_cuda_graph)
    dummy_img = np.zeros((180, 320, 3), dtype=np.uint8)
    dummy_state = np.zeros(STATE_DIM, dtype=np.float32)
    t0 = time.perf_counter()
    try:
        policy.predict(
            top_cam=dummy_img,
            left_cam=dummy_img,
            right_cam=dummy_img,
            instruction="warmup",
            state=dummy_state,
            num_steps=DEFAULT_NUM_STEPS,
            enable_cuda_graph=policy.default_cuda_graph,
        )
    except Exception:  # noqa: BLE001
        log.exception("warmup inference failed (server will still start)")
        return
    log.info("Warmup OK (%.1f ms)", (time.perf_counter() - t0) * 1000.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MolmoAct2-BimanualYAM inference server")
    p.add_argument("--host", default="0.0.0.0", help="bind address (default: 0.0.0.0)")
    p.add_argument("--port", type=int, default=8202, help="bind port (default: 8202)")
    p.add_argument(
        "--repo-id", default=REPO_ID,
        help=f"HF repo id, or a local converted checkpoint dir (default: {REPO_ID})",
    )
    p.add_argument(
        "--norm-tag", default=NORM_TAG,
        help=f"normalization tag in the checkpoint's norm_stats.json (default: {NORM_TAG})",
    )
    p.add_argument("--device", default="cuda:0", help="torch device (default: cuda:0)")
    p.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="model dtype (default: bfloat16; fp32 needs ~26 GB)",
    )
    p.add_argument("--no-warmup", action="store_true", help="skip warmup pass")
    p.add_argument(
        "--cuda-graph",
        action="store_true",
        help="enable CUDA graph capture for action expert (faster but ~2 GB more VRAM)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    policy = Policy(
        repo_id=args.repo_id,
        device=args.device,
        dtype=dtype,
        enable_cuda_graph=args.cuda_graph,
        norm_tag=args.norm_tag,
    )
    if not args.no_warmup:
        warmup(policy)

    app = build_app(policy)

    import uvicorn

    log.info("Listening on %s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
