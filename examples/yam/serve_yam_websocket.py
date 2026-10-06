"""MolmoAct2-BimanualYAM policy server speaking the openpi websocket protocol.

`host_server_yam.py` serves the same model over HTTP + json_numpy. robot_class's
`inference/yam/run.py` talks to an openpi-style websocket server instead (msgpack
frames, server metadata sent on connect). This script wraps the same `Policy` so that
runner can drive MolmoAct2 unchanged:

    python -m inference.yam.run \\
        --left-channel can_follower_l --right-channel can_follower_r \\
        --remote-host <this machine> --remote-port 8203 --prompt "pick up the block"

Wire protocol (client = openpi_client.WebsocketClientPolicy):

    on connect  server -> client   msgpack metadata dict (see build_metadata)
    per step    client -> server   msgpack obs dict:
                    observation.state                  (14,) float  left j1..6, gripper, right j1..6, gripper
                    observation.images.head_cam        (H, W, 3) uint8 RGB   -> MolmoAct2 `top_cam`
                    observation.images.wrist_left      (H, W, 3) uint8 RGB   -> `left_cam`
                    observation.images.wrist_right     (H, W, 3) uint8 RGB   -> `right_cam`
                    prompt                             str                   -> `instruction`
                server -> client   {"actions": (N, 14) float32, "server_timing": {...}}
                                   or, on failure, a plain-text traceback frame and a close.
    GET /healthz   -> 200 "OK"

Chunked execution only: MolmoAct2 has no real-time-chunking path, so metadata says
rtc_enabled=False and the runner must not be started with --rtc.

Needs `msgpack` in addition to the repo's own dependencies.

Run:

    uv run python examples/yam/serve_yam_websocket.py --host 0.0.0.0 --port 8203
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import http
import logging
import os
import sys
import time
import traceback
from typing import Any, Callable

import numpy as np

try:
    import msgpack
except ImportError as exc:  # pragma: no cover
    raise SystemExit("serve_yam_websocket.py needs msgpack: `uv pip install msgpack`") from exc
import websockets.asyncio.server as ws_server
import websockets.exceptions
import websockets.frames

log = logging.getLogger("molmoact2.yam.ws")

YAM_ACTION_DIM = 14
# Left-first, matching robot_class's BIMANUAL_JOINT_ACTION_NAMES and MolmoAct2's own
# gello_min env (concat(left 7, right 7)).
YAM_ACTION_NAMES: tuple[str, ...] = (
    *(f"left_joint_{i}" for i in range(1, 7)), "left_gripper",
    *(f"right_joint_{i}" for i in range(1, 7)), "right_gripper",
)
# robot_class camera role -> MolmoAct2 predict() argument. Order here is not the model's
# order; Policy.predict() fixes that to [top, left, right].
IMAGE_KEYS: dict[str, str] = {
    "observation.images.head_cam": "top_cam",
    "observation.images.wrist_left": "left_cam",
    "observation.images.wrist_right": "right_cam",
}
POLICY_CAMERAS = ("head_cam", "wrist_left", "wrist_right")


# --- msgpack + numpy: byte-for-byte the openpi_client.msgpack_numpy wire format -------------
def _pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj: Any) -> Any:
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


_packer = functools.partial(msgpack.Packer, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


def build_metadata(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """What robot_class's `_validate_policy_contract` checks before any arm connects."""
    metadata: dict[str, Any] = {
        "robot": "yam",
        "policy": "molmoact2",
        "action_dim": YAM_ACTION_DIM,
        "action_names": list(YAM_ACTION_NAMES),
        "cameras": list(POLICY_CAMERAS),
        "model_action_representation": "absolute joint positions (6/arm) + gripper, left-first",
        "rtc_enabled": False,
    }
    metadata.update(extra or {})
    return metadata


def obs_to_predict_kwargs(obs: dict) -> dict[str, Any]:
    """Validate one runner observation and rename it to Policy.predict()'s arguments."""
    required = ("observation.state", "prompt", *IMAGE_KEYS)
    missing = [key for key in required if key not in obs]
    if missing:
        raise KeyError(f"observation is missing {missing}; got {sorted(obs)}")

    state = np.asarray(obs["observation.state"], dtype=np.float32).reshape(-1)
    if state.shape != (YAM_ACTION_DIM,):
        raise ValueError(f"observation.state must be ({YAM_ACTION_DIM},), got {state.shape}")
    if not np.isfinite(state).all():
        raise ValueError("observation.state contains NaN or Inf")

    instruction = str(obs["prompt"]).strip()
    if not instruction:
        raise ValueError("prompt is empty")

    kwargs: dict[str, Any] = {"state": state, "instruction": instruction}
    for obs_key, arg in IMAGE_KEYS.items():
        image = np.asarray(obs[obs_key])
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{obs_key} must be HxWx3, got {image.shape}")
        kwargs[arg] = image
    return kwargs


class YamWebsocketServer:
    """openpi-protocol websocket server around `predict(**kwargs) -> (N, 14) actions`."""

    def __init__(self, predict: Callable[..., np.ndarray], metadata: dict, host: str, port: int) -> None:
        self._predict = predict
        self._metadata = metadata
        self._host = host
        self._port = port

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self) -> None:
        async with ws_server.serve(
            self._handler, self._host, self._port,
            compression=None, max_size=None,
            # Inference runs for hundreds of ms and the client sends no pings
            # (ping_interval=None); don't let our keepalive kill a quiet connection.
            ping_interval=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    def infer(self, obs: dict) -> np.ndarray:
        obs = {k: v for k, v in obs.items() if not k.startswith("_rtc_")}  # RTC fields: unsupported, ignored
        actions = np.asarray(self._predict(**obs_to_predict_kwargs(obs)), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != YAM_ACTION_DIM:
            raise ValueError(
                f"model returned actions of shape {actions.shape}; the YAM runner needs (N, {YAM_ACTION_DIM}). "
                "Check the checkpoint's norm_stats.json action dimension."
            )
        if not np.isfinite(actions).all():
            raise ValueError("model returned NaN or Inf actions")
        return actions

    async def _handler(self, websocket: ws_server.ServerConnection) -> None:
        log.info("connection from %s opened", websocket.remote_address)
        packer = _packer()
        await websocket.send(packer.pack(self._metadata))
        loop = asyncio.get_running_loop()
        while True:
            try:
                obs = _unpackb(await websocket.recv())
                t0 = time.monotonic()
                # Off the event loop: a blocked loop can't answer a close or a new connection.
                actions = await loop.run_in_executor(None, self.infer, obs)
                infer_ms = (time.monotonic() - t0) * 1000
                log.debug("infer %.0f ms, %s", infer_ms, actions.shape)
                await websocket.send(packer.pack({"actions": actions, "server_timing": {"infer_ms": infer_ms}}))
            except websockets.exceptions.ConnectionClosed:
                log.info("connection from %s closed", websocket.remote_address)
                return
            except Exception:
                # Same contract as openpi's server: a text frame is an error, then close.
                # The runner turns it into "policy unavailable", pauses and holds the arms.
                log.exception("inference failed")
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                return


def _health_check(connection: ws_server.ServerConnection, request: ws_server.Request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MolmoAct2-BimanualYAM websocket (openpi protocol) server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8203, help="default 8203 (8202 is the HTTP server, 8000 is evo's pi0.5)")
    p.add_argument("--repo-id", default=None, help="HF repo id or local converted checkpoint dir (default: host_server_yam.REPO_ID)")
    p.add_argument("--norm-tag", default=None, help="norm_stats.json tag (default: host_server_yam.NORM_TAG)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--cuda-graph", action="store_true", help="faster action expert, ~2 GB more VRAM")
    p.add_argument("--num-steps", type=int, default=None, help="flow-matching steps (default: host_server_yam.DEFAULT_NUM_STEPS)")
    p.add_argument("--no-warmup", action="store_true")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    # Heavy imports (torch, transformers) live here so the protocol layer above can be
    # imported and tested without a GPU. host_server_yam.py is imported, not edited.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    import host_server_yam as hs

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    repo_id = args.repo_id or hs.REPO_ID
    norm_tag = args.norm_tag or hs.NORM_TAG
    num_steps = args.num_steps or hs.DEFAULT_NUM_STEPS

    policy = hs.Policy(repo_id=repo_id, device=args.device, dtype=dtype,
                       enable_cuda_graph=args.cuda_graph, norm_tag=norm_tag)
    if not args.no_warmup:
        hs.warmup(policy)

    def predict(**kwargs: Any) -> np.ndarray:
        return policy.predict(num_steps=num_steps, enable_cuda_graph=policy.default_cuda_graph, **kwargs)

    metadata = build_metadata({"repo_id": repo_id, "norm_tag": norm_tag, "num_steps": num_steps})
    log.info("Listening on ws://%s:%d (metadata: %s)", args.host, args.port, metadata)
    YamWebsocketServer(predict, metadata, args.host, args.port).serve_forever()


if __name__ == "__main__":
    main()
