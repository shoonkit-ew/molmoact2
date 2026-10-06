# Changes in this fork (bimanual YAM fine-tuning, serving and evaluation)

Summary of what was added or changed relative to upstream MolmoAct2. Deployment-specific values
(hosts, keys, buckets) are placeholders: `<...>`.

## Serving (`examples/yam/`)

- **`host_server_yam.py`** (HTTP `/act`, default port 8202)
  - `--repo-id` accepts a local converted checkpoint directory; `--norm-tag` selects the tag in that
    checkpoint's `norm_stats.json` (e.g. `eastworlds_yam_data_collection` for our fine-tunes).
  - Opt-in attention capture: send `"return_attention": true` in the `/act` payload. The reply then
    also carries `attention` (per-layer, per-camera maps of how the action expert's cross-attention
    reads the image tokens) and `token_norm` (per-cell image-token norms, used to mask ViT
    high-norm artifact cells). Off by default, so normal latency is unchanged.
- **`serve_yam_websocket.py`** (new, default port 8203): serves the same `Policy` over the openpi
  websocket protocol (msgpack frames, metadata sent on connect, `GET /healthz`), so an openpi-style
  YAM robot runner can drive MolmoAct2 unchanged. Chunked execution only (`rtc_enabled=False`).
  Needs `msgpack`. It does **not** return attention maps; use the HTTP server for those.
  ```bash
  uv run python examples/yam/serve_yam_websocket.py --host 0.0.0.0 --port 8203 \
    --repo-id <checkpoint dir or HF repo> --norm-tag <norm tag>
  ```

## Training (`experiments/`)

- **`finetune.sh`** (new): one launcher for `smoke`, `probe`, `lora`, `full` and `depth` modes.
  Handles the data sync (rclone to local NVMe), the environment workarounds, run naming, GPU pinning
  (`GPUS=0,1,2,3`) and resume (`RESUME=<run>`). Header comments document every mode.
  Reads `WANDB_API_KEY` / `WANDB_ENTITY` from the environment or from the gitignored
  `experiments/.wandb_api_key` and `experiments/.wandb_entity`; `.hf_access_token` is optional.
  Force-added past the `*.sh` ignore rule.
- **`launch_scripts/data_mixtures.py`**: new mixture `yam_data_collection` (multi-task, one tag) and
  `MOLMOACT2_LEROBOT_VAL_DATASETS` for held-out validation. Episodes come from
  `launch_scripts/yam_data_collection_split.json`: a per-task 80/20 split by whole episode,
  plus `excluded_episodes` with reasons. `*_depth` companion folders are excluded because
  `olmo/data/lerobot_wrapper.py` can't resolve their tag metadata.
- **`launch_scripts/train_lerobot.py`**: `--action_val_interval` and
  `--action_val_examples_per_dataset` flags that wire in the validator below.
- **`olmo/eval/action_validator.py`** (new): held-out action validation. Before training and every N
  steps it samples action chunks on val episodes and logs per-task `val/<task>/mse`,
  `mse_vs_hold` (model MSE ÷ MSE of holding the first action; < 1 beats standing still) and
  `accuracy@{tau}`, plus means over tasks. Open-loop only.
- **`olmo/train/{trainer,trainer_config,run_trainer}.py`**: run the validator in the training loop;
  wandb 0.30 compatibility (`wandb.run.get_url()` → `wandb.run.url`, `wandb.finish` without `quiet`).
- **`.gitignore` / `experiments/.gitignore`**: ignore per-user secret files and the local
  `tencent_cos.local.yaml`.

## Offline evaluation tools (`experiments/scripts/`)

- **`yam_replay.py`**: replays a held-out episode open-loop through a server (`--policy molmo` or
  `--policy openpi`), compares predicted chunks with the recording, and logs everything to Rerun
  (`.rrd`) plus per-joint plots. `--compare` overlays earlier runs of other models, `--attn` adds
  attention heatmaps and rings, `--video` burns the overlay into an mp4.
- **`yam_calib.py`**: one-off head-camera pose estimate per camera placement (click gripper tips,
  then PnP against forward kinematics) so predicted paths can be projected onto the head cam.
- **`yam_compare_video.py`**: builds a comparison grid video from saved replay runs, no server needed.
- **`evo_attn_server.py`**: wraps a pi0.5 (openpi) policy server and adds opt-in attention capture,
  so both models can be compared with the same viewer.

## Workflow

1. Fine-tune: `cd experiments && ./finetune.sh full 8` (checkpoints in `experiments/checkpoints/<run>/`).
2. Convert a checkpoint to HF format (`python -m olmo.hf_model.convert_molmoact2_to_hf <step dir> <out dir>`).
3. Check it offline with `yam_replay.py` before robot trials (needs the HTTP server from step 4a).
4. Serve it:
   ```bash
   # a) HTTP, for replay / attention maps (port 8202)
   uv run python examples/yam/host_server_yam.py --host 127.0.0.1 --port 8202 \
     --repo-id <checkpoint dir> --norm-tag <norm tag>
   # b) websocket, for the robot runner (port 8203)
   uv run python examples/yam/serve_yam_websocket.py --host 0.0.0.0 --port 8203 \
     --repo-id <checkpoint dir> --norm-tag <norm tag>
   curl http://<server>:8203/healthz        # -> OK
   ```
5. Run it on the robot with the `robot_class` client (separate repo, on the machine wired to the arms):
   ```bash
   git clone git@github.com:Eastworld-Labs/robot_class.git && cd robot_class
   source setup_env.sh --yam                 # creates/activates .venv, installs the i2rt SDK, brings up CAN
   python -m inference.yam.run \
     --left-channel <can_follower_l> --right-channel <can_follower_r> \
     --remote-host <server> --remote-port 8203 \
     --prompt "<task prompt>"                # see the list below
   ```
   Do not pass `--rtc` (MolmoAct2 is chunked only). The runner starts paused (`--interactive`); unpause
   from its dashboard. Camera serials come from `configs/yam_cameras.yaml`.

## Task prompts

Use the exact training string. The model was fine-tuned on these, and a slug like `stack_cup` or a
paraphrase may behave worse. Dataset folders are under `data-collection/yam/`.

| Prompt | Dataset folder |
|---|---|
| `Build Cup Pyramid` | `build_cup_pyramid_20260928` |
| `Fold Handkerchief` | `fold_handkerchief_20260918` |
| `Peg In Hole` | `peg_in_hole_20260924` |
| `Pick Up Bottle` | `pick_up_bottle_20260917` |
| `Sort Block` | `sort_block_20260922` |
| `Sort Column Toy` | `sort_column_toy_20260922` |
| `Sort Wall Plug` | `sort_wall_plug_20260922` |
| `Stack Cup` | `stack_cup_20260911`, `yam_stack_cup_20260928` |
| `T - Pack Bolt M12` | `t_pack_bolt_m12_20260924` |
| `T - Pack Nut M12` | `t_pack_nut_m12_20260923` |
| `T - Pack Washer M16` | `yam_t_pack_washer_m16_20260923` |
| `T - Packing` | `yam_t_packing_20260924` |
| `Collect Pen` | `yam_collect_pen_20260904` |
| `Handover` | `yam_handover_20260904` |
