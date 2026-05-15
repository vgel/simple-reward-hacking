#!/usr/bin/env bash
set -euo pipefail

RESUME_DIR=""
if [[ "${1:-}" == "--resume" ]]; then
    RESUME_DIR="${2:?--resume requires a log dir path}"
    shift 2
fi

ENV_DIR="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$ENV_DIR/.env" ] && set -a && . "$ENV_DIR/.env" && set +a

: "${TINKER_API_KEY:?TINKER_API_KEY must be set}"

VF_ENV_ARGS='{"sandbox": "bwrap", "dataset_name": "vgel/SYNTHETIC-2-RL-fn_name-Qwen3-32B-Hard-5x", "shuffle_seed": null}'

if [[ -n "$RESUME_DIR" ]]; then
    LOG_PATH="$RESUME_DIR"
    echo "resuming from $LOG_PATH"
else
    LOG_PATH="$ENV_DIR/misc/runs/qwen3-8b-$(date +%Y%m%d-%H%M)"
fi

cd "$ENV_DIR"
uv run python -m tinker_cookbook.recipes.verifiers_rl.train \
    model_name="Qwen/Qwen3-8B" \
    lora_rank=64 \
    learning_rate=2e-4 \
    vf_env_id="simple-reward-hacking" \
    vf_env_args="$VF_ENV_ARGS" \
    group_size=16 \
    groups_per_batch=32 \
    max_tokens=2048 \
    temperature=1.0 \
    max_steps=102 \
    save_every=1 \
    eval_every=0 \
    log_path="$LOG_PATH" \
    behavior_if_log_dir_exists="$([ -n "$RESUME_DIR" ] && echo resume || echo ask)" \
    wandb_project="reward-hacking-qwen3" \
    wandb_name="qwen3-8b"
