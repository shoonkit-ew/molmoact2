#!/usr/bin/env bash
set -euo pipefail

# Common finetune launcher — safe on a shared machine/box: every variable
# here lives only in this script's own process tree. Running it (./finetune.sh)
# never exports anything into your shell or anyone else's session; `source`ing
# it would, so don't.
#
# Usage:
#   ./finetune.sh <smoke|lora|full> [nproc] [mixture]
#   ./finetune.sh smoke                    # 1 GPU, 20 steps, validates config/data
#   ./finetune.sh lora 8                   # 8 GPUs, real LoRA fine-tune
#   ./finetune.sh full 8                   # 8 GPUs, real full fine-tune
#
#   ./finetune.sh probe <device_batch_size> [nproc] [preset=lora|full]
#   ./finetune.sh probe 4 8 lora           # 10 steps at device_batch_size=4 on
#                                          # 8 GPUs, lora-shaped — find the real
#                                          # VRAM ceiling before committing to a
#                                          # long run. Watch for OOM, and for the
#                                          # "Peak GPU Memory (MB)" line in the
#                                          # log / wandb run (same run, logged
#                                          # under System/Peak GPU Memory (MB)).
#                                          # No checkpoints are written.
#
# To pin specific physical GPUs on a shared box (e.g. only GPUs 0-3 so other
# users' runs on 4-7 aren't affected), set GPUS instead of relying on [nproc]
# alone — it both masks the devices and sets nproc to match automatically:
#   GPUS=0,1,2,3 ./finetune.sh lora
#
# Depth reasoning (MolmoAct2-Think) — optional, off by default:
#   ./finetune.sh depth 4                  # step 1: write depth labels for the head cam of every
#                                          # task in the split, one shard per GPU (resumable per file)
#   DEPTH=1 ./finetune.sh probe 2 4 full   # step 2: same modes, now starting from allenai/MolmoAct2-Think
#   DEPTH=1 GPUS=0,1,2,3 ./finetune.sh full   # with depth tokens in ~half the samples (the rest stay
#                                          # action-only, so depth can be switched off at inference)
# Labels land in DEPTH_DATA_DIR (default /opt/dlami/nvme/yam_depth), separate from the raw recordings
# and from the COS *_depth sensor folders. Run `probe` first: sequences grow by ~100 tokens/sample.
#
# Data: lora/full/depth auto-sync the full dataset collection to local disk first
# (idempotent rclone copy — skips files already present and matching, safe
# and cheap to re-run across many invocations; never auto-deleted, so it's
# only paid once per machine). smoke/probe read straight off the live rclone
# mount instead, since they're quick one-off checks not worth a bulk download.
# Set LEROBOT_DATA_ROOT explicitly to override either behavior.
#
# Per-user wandb config — each user sets these in their own shell before
# running, OR drops a personal file (gitignored, never committed):
#   WANDB_API_KEY   from wandb.ai/authorize           -> experiments/.wandb_api_key
#   WANDB_ENTITY    your wandb username/team          -> experiments/.wandb_entity
# The team's "CW Forge" page (forge.coreweave.com) is just a branded UI over
# the same standard wandb.ai account/API — no custom WANDB_BASE_URL needed.
# Optional overrides: WANDB_PROJECT (default SK-ablation), CHECKPOINT,
# RCLONE_CONFIG_PATH (default ~/sk_vla/rclone.conf).

cd "$(dirname "$0")"

MODE="${1:?Usage: ./finetune.sh <smoke|lora|full|probe|depth> ...}"

if [[ "$MODE" == "probe" ]]; then
    PROBE_DBS="${2:?Usage: ./finetune.sh probe <device_batch_size> [nproc] [preset=lora|full]}"
    NPROC="${3:-8}"
    PROBE_PRESET="${4:-lora}"
    MIXTURE="yam_data_collection"
else
    NPROC="${2:-1}"
    MIXTURE="${3:-yam_data_collection}"
fi
DEPTH="${DEPTH:-0}"
if [[ "$DEPTH" == "1" ]]; then
    CHECKPOINT="${CHECKPOINT:-allenai/MolmoAct2-Think}"
else
    CHECKPOINT="${CHECKPOINT:-allenai/MolmoAct2-BimanualYAM}"
fi
WANDB_PROJECT="${WANDB_PROJECT:-SK-ablation}"

if [[ -n "${GPUS:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPUS"
    NPROC=$(( $(grep -o ',' <<<"$GPUS" | wc -l) + 1 ))
fi


RUN_NAME="${MODE}-$(whoami)-$(date +%Y%m%d-%H%M%S)"
[[ "$DEPTH" == "1" ]] && RUN_NAME="think-${RUN_NAME}"
# RESUME=<run name> continues that run from its latest checkpoint (same save folder, wandb run and
# config; --max_duration is the TOTAL step count). Works with a different GPU count than the original.
if [[ -n "${RESUME:-}" ]]; then
    [[ -d "checkpoints/$RESUME" ]] || { echo "No checkpoints/$RESUME to resume" >&2; exit 1; }
    RUN_NAME="$RESUME"
fi

[[ -z "${WANDB_API_KEY:-}" && -f .wandb_api_key ]] && WANDB_API_KEY="$(cat .wandb_api_key)"
[[ -z "${WANDB_ENTITY:-}" && -f .wandb_entity ]] && WANDB_ENTITY="$(cat .wandb_entity)"
: "${WANDB_API_KEY:?Set WANDB_API_KEY, or put your key in experiments/.wandb_api_key}"
: "${WANDB_ENTITY:?Set WANDB_ENTITY, or put it in experiments/.wandb_entity}"

# Optional but recommended with nproc>1: every rank independently hits the HF
# Hub to resolve the checkpoint; unauthenticated + concurrent risks 429s. Not
# fatal if unset (read by olmo/util.py's get_hf_access_token(), which also
# accepts HF_TOKEN).
[[ -z "${HF_ACCESS_TOKEN:-}" && -f .hf_access_token ]] && HF_ACCESS_TOKEN="$(cat .hf_access_token)"
[[ -n "${HF_ACCESS_TOKEN:-}" ]] && export HF_ACCESS_TOKEN

export WANDB_API_KEY

if [[ -z "${LEROBOT_DATA_ROOT:-}" ]]; then
    # Shared ephemeral NVMe scratch (AWS Deep Learning AMI convention), not
    # $HOME — much more space, and already world-writable (sticky bit, like
    # /tmp) so anyone else on this box benefits from the same cache without
    # permission changes. Does NOT survive an instance *stop* (only a reboot
    # while running) — re-syncs from scratch after a stop.
    LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-/opt/dlami/nvme/yam_data_local}"
    if [[ "$MODE" == "lora" || "$MODE" == "full" || "$MODE" == "depth" ]]; then
        RCLONE_CONFIG_PATH="${RCLONE_CONFIG_PATH:-$HOME/sk_vla/rclone.conf}"
        echo ">> syncing dataset collection to local disk (idempotent — skips files already present and matching): $LOCAL_DATA_DIR"
        mkdir -p "$LOCAL_DATA_DIR"
        rclone --config "$RCLONE_CONFIG_PATH" copy \
            tencentcos:eastworlds-data-1428669724/data-collection/yam "$LOCAL_DATA_DIR" \
            --transfers=16 --progress
        LEROBOT_DATA_ROOT="$LOCAL_DATA_DIR"
    elif [[ -d "$LOCAL_DATA_DIR" && -n "$(ls -A "$LOCAL_DATA_DIR" 2>/dev/null)" ]]; then
        # smoke/probe: reuse an already-synced local copy for free (no re-sync
        # triggered here), rather than needlessly reading over the slow mount.
        echo ">> reusing already-synced local data at $LOCAL_DATA_DIR"
        LEROBOT_DATA_ROOT="$LOCAL_DATA_DIR"
    else
        LEROBOT_DATA_ROOT="$HOME/sk_vla/cos-mount/data-collection/yam"
    fi
fi
export LEROBOT_DATA_ROOT
export PYTHONPATH="$PWD:$PWD/lerobot/src:${PYTHONPATH:-}"

# The Deep Learning AMI puts system CUDA (/usr/local/cuda-12.9, -13.2) on
# LD_LIBRARY_PATH. That makes the venv's libcublas.so.12 (12.8, from the torch
# wheel) load the *system's* libcublasLt.so.12 (12.9) — a mismatched pair that
# fails every bf16 GEMM with CUBLAS_STATUS_INVALID_VALUE. The torch wheel finds
# its own CUDA libs via rpath, so drop the system CUDA entries for this run.
LD_LIBRARY_PATH="$(tr ':' '\n' <<<"${LD_LIBRARY_PATH:-}" | { grep -v '^/usr/local/cuda' || true; } | paste -sd: -)"
export LD_LIBRARY_PATH

# Default (1ms, lerobot_wrapper.py:2910) is far tighter than a 30fps frame
# interval (~33ms) — real video timestamp quantization routinely exceeds it,
# so examples get silently skipped/retried (lerobot_wrapper.py:2520-2547) far
# more than intended. 0.04s comfortably covers one frame at <=30fps.
export LEROBOT_TOLERANCE_S="${LEROBOT_TOLERANCE_S:-0.04}"

# Required even for LeRobot-only training: olmo/data/get_dataset.py reads this
# unconditionally (os.environ["MOLMO_DATA_DIR"], no fallback) once real data
# loading starts, regardless of whether any VLM/academic datasets are mixed
# in. Not used for anything in our mixture — just needs to resolve to a path.
export MOLMO_DATA_DIR="${MOLMO_DATA_DIR:-$HOME/sk_vla/molmo_data}"
mkdir -p "$MOLMO_DATA_DIR"

# MolmoAct2 paper's fine-tuning recipe (Sec. 4.3.1): 8 sampled flow times per action
# chunk (vs 4 in post-training; trainer default is 1), and no knowledge insulation
# (flow loss updates the VLM; train_lerobot.py's --action_expert_detach_vlm=false default).
RECIPE_ARGS=(--num_flow_timesteps=8 --action_expert_detach_vlm=false)

# Depth reasoning (DEPTH=1): the model writes ~100 depth tokens for the head cam before acting.
# robot_action + robot_depth_action at 1.0 each = roughly half the samples with depth, half
# without, so inference can run with or without it. Labels come from `./finetune.sh depth`.
if [[ "$DEPTH" == "1" ]]; then
    export LEROBOT_DEPTH_DATA_ROOT="${DEPTH_DATA_DIR:-/opt/dlami/nvme/yam_depth}"
    RECIPE_ARGS+=(--enable_depth_reasoning=true --num_depth_tokens=128 --num_depth_tokens_per_image=100
                  --depth_code_input_noise_rate=0.1
                  --style_robot_action=1.0 --style_robot_depth=0.0 --style_robot_depth_action=1.0)
    if [[ "$MODE" != "depth" && ! -d "$LEROBOT_DEPTH_DATA_ROOT" ]]; then
        echo "DEPTH=1 but no depth labels at $LEROBOT_DEPTH_DATA_ROOT — run ./finetune.sh depth first" >&2
        exit 1
    fi
fi

if [[ "$MODE" == "depth" ]]; then
    export LEROBOT_DEPTH_DATA_ROOT="${DEPTH_DATA_DIR:-/opt/dlami/nvme/yam_depth}"
    mkdir -p "$LEROBOT_DEPTH_DATA_ROOT"
    echo ">> depth labels -> $LEROBOT_DEPTH_DATA_ROOT  ($NPROC shard(s); rerun to resume)"
    for repo in $(.venv/bin/python -c "import json; print(' '.join(json.load(open('launch_scripts/yam_data_collection_split.json'))['repos']))"); do
        echo ">> $repo"
        pids=()
        for ((i = 0; i < NPROC; i++)); do
            gpu=$i
            [[ -n "${GPUS:-}" ]] && gpu=$(cut -d, -f$((i + 1)) <<<"$GPUS")
            CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python scripts/generate_depth_annotation.py \
                "$LEROBOT_DATA_ROOT/$repo" --camera-key observation.images.head_cam \
                --num-shards "$NPROC" --shard-index "$i" ${DEPTH_ARGS:-} &
            pids+=($!)
        done
        for pid in "${pids[@]}"; do wait "$pid"; done
    done
    echo ">> done. Train with: DEPTH=1 ./finetune.sh <probe|full> ..."
    exit 0
fi

case "$MODE" in
  smoke)
    ARGS=(--max_duration=20 --device_batch_size=1 --global_batch_size=1
          --num_workers=0 --pin_memory=false
          --ft_vlm=false --ft_action_expert=true --ft_embedding=none)
    ;;
  lora)
    ARGS=(--max_duration=25000 --device_batch_size=2 --global_batch_size=64
          --num_workers=4 --pin_memory=true
          --save_interval=5000 --save_num_checkpoints_to_keep=5
          --ft_vlm=true --ft_action_expert=true --ft_embedding=lm_head
          --lora_enable=true --lora_rank=64
          --llm_learning_rate=5e-5 --vit_learning_rate=5e-5
          --connector_learning_rate=5e-5 --action_expert_learning_rate=5e-5
          "${RECIPE_ARGS[@]}" --action_val_interval=1000)
    ;;
  full)
    ARGS=(--max_duration=25000 --device_batch_size=2 --global_batch_size=64
          --num_workers=4 --pin_memory=true
          --save_interval=5000 --save_num_checkpoints_to_keep=5
          --ft_vlm=true --ft_action_expert=true --ft_embedding=lm_head --lora_enable=false
          --llm_learning_rate=1e-5 --vit_learning_rate=5e-6
          --connector_learning_rate=5e-6 --action_expert_learning_rate=5e-5
          "${RECIPE_ARGS[@]}" --action_val_interval=1000)
    ;;
  probe)
    case "$PROBE_PRESET" in
      lora) PRESET_ARGS=(--ft_vlm=true --ft_action_expert=true --ft_embedding=lm_head
                          --lora_enable=true --lora_rank=64) ;;
      full) PRESET_ARGS=(--ft_vlm=true --ft_action_expert=true --ft_embedding=lm_head --lora_enable=false) ;;
      *)
        echo "Unknown probe preset '$PROBE_PRESET' (expected: lora | full)" >&2
        exit 1
        ;;
    esac
    PROBE_GBS=$(( PROBE_DBS * NPROC ))
    RUN_NAME="probe-dbs${PROBE_DBS}-${PROBE_PRESET}-$(whoami)-$(date +%Y%m%d-%H%M%S)"
    # Same recipe as the real run so memory matches; tiny validation (baseline + step 10)
    # so the probe also exercises the held-out validation path end to end.
    ARGS=(--max_duration=10 --device_batch_size="$PROBE_DBS" --global_batch_size="$PROBE_GBS"
          --num_workers=0 --pin_memory=false
          --save_interval=999999 --save_num_checkpoints_to_keep=1
          "${PRESET_ARGS[@]}" "${RECIPE_ARGS[@]}"
          --action_val_interval=10 --action_val_examples_per_dataset=8)
    ;;
  *)
    echo "Unknown mode '$MODE' (expected: smoke | lora | full | probe | depth)" >&2
    exit 1
    ;;
esac

echo ">> mode=$MODE nproc=$NPROC mixture=$MIXTURE run=$RUN_NAME entity=$WANDB_ENTITY project=$WANDB_PROJECT"

.venv/bin/torchrun --standalone --nproc-per-node="$NPROC" launch_scripts/train_lerobot.py \
  "$CHECKPOINT" "$MIXTURE" \
  --wandb.name="$RUN_NAME" --wandb.entity="$WANDB_ENTITY" --wandb.project="$WANDB_PROJECT" \
  --save_folder="checkpoints/${RUN_NAME}" \
  --packing=false --dynamic_seq_len=true \
  --norm_mode=min_max --frame_loading_backend=av \
  "${ARGS[@]}"
