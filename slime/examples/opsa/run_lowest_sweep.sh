#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

MODEL="qwen3-1.7b"
FRACTIONS="0.1,0.2,0.3,0.4"
SAVE_INTERVAL_OVERRIDE="${SAVE_INTERVAL_OVERRIDE:-}"
HF_CHECKPOINT="${HF_CHECKPOINT:-}"
ACTOR_CHECKPOINT="${ACTOR_CHECKPOINT:-${REF_LOAD:-}}"
SAVE_ROOT="${SAVE_ROOT:-}"
PROMPT_DATA="${PROMPT_DATA:-}"
EVAL_DATA="${EVAL_DATA:-}"
MEGATRON_PATH="${MEGATRON_PATH:-}"
RAY_ADDRESS="${RAY_ADDRESS:-}"
RAY_PORT="${RAY_PORT:-6379}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
WANDB_PROJECT="${WANDB_PROJECT:-}"
WANDB_GROUP="${WANDB_GROUP:-}"
WANDB_TEAM="${WANDB_TEAM:-}"
WANDB_MODE="${WANDB_MODE:-}"
WANDB_DIR="${WANDB_DIR:-}"
WANDB_LOG_ALL_METRICS="${WANDB_LOG_ALL_METRICS:-false}"
WANDB_OPEN_METRICS="${WANDB_OPEN_METRICS:-false}"
LIGHT_CHECKPOINT=false
COLOCATE=false
DRY_RUN=false

usage() {
   cat <<'EOF'
Usage:
  bash examples/opsa/run_lowest_sweep.sh [options]

Runs OPSA sequentially for lowest 10%, 20%, 30%, and 40% by default.

Options:
  --model NAME                qwen3-1.7b, qwen3-4b, qwen3-14b, or qwen3.5-9b
  --fractions LIST            Comma-separated fractions (default: 0.1,0.2,0.3,0.4)
  --save-interval INTEGER     Override each run's checkpoint interval
  --hf-checkpoint PATH        Hugging Face checkpoint used by rollout
  --actor-checkpoint PATH     Megatron checkpoint used to initialize the actor
  --save-root PATH            Parent directory for fraction-specific runs
  --prompt-data FILE          Training JSON/JSONL file
  --eval-data FILE            Evaluation JSON/JSONL file
  --megatron-path DIR         Megatron-LM checkout
  --ray-address URL           Submit every run to an existing Ray dashboard
  --ray-port PORT             Local Ray GCS port (default: 6379)
  --dashboard-port PORT       Local Ray dashboard port (default: 8265)
  --wandb-project NAME        Enable W&B and log every run to this project
  --wandb-group NAME          Optional shared group/run name for the sweep
  --wandb-team NAME           W&B entity/team
  --wandb-mode MODE           online, offline, or disabled
  --wandb-dir PATH            Directory for local W&B files
  --wandb-log-all-metrics     Log all Slime metrics instead of the compact set
  --wandb-open-metrics        Add SGLang OpenMetrics to online W&B runs
  --light-checkpoint          Produce non-resumable weight-only checkpoints
  --colocate                  Share all physical GPUs between actor and rollout
  --dry-run                   Print all resolved commands without running them
  -h, --help                  Show this help
EOF
}

die() {
   echo "error: $*" >&2
   exit 2
}

require_value() {
   if [ "$#" -lt 2 ] || [ -z "$2" ]; then
      die "$1 requires a value"
   fi
}

normalize_boolean() {
   local name="$1"
   local value="$2"
   case "$value" in
      1|true|TRUE|yes|YES|on|ON) echo true ;;
      0|false|FALSE|no|NO|off|OFF|"") echo false ;;
      *) die "$name must be a boolean (true/false, 1/0, yes/no, or on/off)" ;;
   esac
}

WANDB_LOG_ALL_METRICS="$(normalize_boolean WANDB_LOG_ALL_METRICS "$WANDB_LOG_ALL_METRICS")"
WANDB_OPEN_METRICS="$(normalize_boolean WANDB_OPEN_METRICS "$WANDB_OPEN_METRICS")"

while [ "$#" -gt 0 ]; do
   case "$1" in
      --model)
         require_value "$@"
         MODEL="$2"
         shift 2
         ;;
      --fractions)
         require_value "$@"
         FRACTIONS="$2"
         shift 2
         ;;
      --save-interval)
         require_value "$@"
         SAVE_INTERVAL_OVERRIDE="$2"
         shift 2
         ;;
      --hf-checkpoint)
         require_value "$@"
         HF_CHECKPOINT="$2"
         shift 2
         ;;
      --actor-checkpoint)
         require_value "$@"
         ACTOR_CHECKPOINT="$2"
         shift 2
         ;;
      --save-root)
         require_value "$@"
         SAVE_ROOT="$2"
         shift 2
         ;;
      --prompt-data)
         require_value "$@"
         PROMPT_DATA="$2"
         shift 2
         ;;
      --eval-data)
         require_value "$@"
         EVAL_DATA="$2"
         shift 2
         ;;
      --megatron-path)
         require_value "$@"
         MEGATRON_PATH="$2"
         shift 2
         ;;
      --ray-address)
         require_value "$@"
         RAY_ADDRESS="$2"
         shift 2
         ;;
      --ray-port)
         require_value "$@"
         RAY_PORT="$2"
         shift 2
         ;;
      --dashboard-port)
         require_value "$@"
         DASHBOARD_PORT="$2"
         shift 2
         ;;
      --wandb-project)
         require_value "$@"
         WANDB_PROJECT="$2"
         shift 2
         ;;
      --wandb-group)
         require_value "$@"
         WANDB_GROUP="$2"
         shift 2
         ;;
      --wandb-team)
         require_value "$@"
         WANDB_TEAM="$2"
         shift 2
         ;;
      --wandb-mode)
         require_value "$@"
         WANDB_MODE="$2"
         shift 2
         ;;
      --wandb-dir)
         require_value "$@"
         WANDB_DIR="$2"
         shift 2
         ;;
      --wandb-log-all-metrics)
         WANDB_LOG_ALL_METRICS=true
         shift
         ;;
      --wandb-open-metrics)
         WANDB_OPEN_METRICS=true
         shift
         ;;
      --light-checkpoint)
         LIGHT_CHECKPOINT=true
         shift
         ;;
      --colocate)
         COLOCATE=true
         shift
         ;;
      --dry-run)
         DRY_RUN=true
         shift
         ;;
      -h|--help)
         usage
         exit 0
         ;;
      *)
         die "unknown option: $1"
         ;;
   esac
done

case "$MODEL" in
   qwen3-1.7b|qwen3-4b|qwen3-14b|qwen3.5-9b) ;;
   *) die "unsupported model '$MODEL'" ;;
esac

if [ -z "$FRACTIONS" ]; then
   die "--fractions must contain at least one value"
fi
if [ -n "$SAVE_INTERVAL_OVERRIDE" ] && ! [[ "$SAVE_INTERVAL_OVERRIDE" =~ ^[1-9][0-9]*$ ]]; then
   die "--save-interval/SAVE_INTERVAL_OVERRIDE must be a positive integer"
fi
if [ "$DRY_RUN" = false ] && [ -z "$SAVE_ROOT" ]; then
   die "--save-root is required"
fi
if [ -n "$WANDB_PROJECT" ]; then
   case "$WANDB_MODE" in
      ""|online|offline|disabled) ;;
      *) die "--wandb-mode must be online, offline, or disabled" ;;
   esac
   if [ "$WANDB_OPEN_METRICS" = true ] && [ "$WANDB_MODE" != "" ] && [ "$WANDB_MODE" != online ]; then
      die "--wandb-open-metrics requires online W&B mode"
   fi
fi

IFS=',' read -r -a FRACTION_VALUES <<< "$FRACTIONS"
for fraction in "${FRACTION_VALUES[@]}"; do
   [ -n "$fraction" ] || die "--fractions contains an empty value"
   percentage="$(awk -v value="$fraction" 'BEGIN { printf "%g", value * 100 }')"
   percentage="${percentage//./p}"

   RUN_ARGS=(
      bash "${SCRIPT_DIR}/run_opsa.sh"
      --model "$MODEL"
      --preset opsa
      --fraction "$fraction"
      --ray-port "$RAY_PORT"
      --dashboard-port "$DASHBOARD_PORT"
   )

   if [ -n "$HF_CHECKPOINT" ]; then
      RUN_ARGS+=(--hf-checkpoint "$HF_CHECKPOINT")
   fi
   if [ -n "$ACTOR_CHECKPOINT" ]; then
      RUN_ARGS+=(--actor-checkpoint "$ACTOR_CHECKPOINT")
   fi
   if [ -n "$SAVE_ROOT" ]; then
      RUN_ARGS+=(--save-dir "${SAVE_ROOT}/${MODEL}/opsa-lowest${percentage}")
   fi
   if [ -n "$PROMPT_DATA" ]; then
      RUN_ARGS+=(--prompt-data "$PROMPT_DATA")
   fi
   if [ -n "$EVAL_DATA" ]; then
      RUN_ARGS+=(--eval-data "$EVAL_DATA")
   fi
   if [ -n "$MEGATRON_PATH" ]; then
      RUN_ARGS+=(--megatron-path "$MEGATRON_PATH")
   fi
   if [ -n "$RAY_ADDRESS" ]; then
      RUN_ARGS+=(--ray-address "$RAY_ADDRESS")
   fi
   if [ -n "$WANDB_PROJECT" ]; then
      RUN_ARGS+=(--wandb-project "$WANDB_PROJECT")
      if [ -n "$WANDB_GROUP" ]; then
         RUN_ARGS+=(--wandb-group "$WANDB_GROUP")
      fi
      if [ -n "$WANDB_TEAM" ]; then
         RUN_ARGS+=(--wandb-team "$WANDB_TEAM")
      fi
      if [ -n "$WANDB_MODE" ]; then
         RUN_ARGS+=(--wandb-mode "$WANDB_MODE")
      fi
      if [ -n "$WANDB_DIR" ]; then
         RUN_ARGS+=(--wandb-dir "$WANDB_DIR")
      fi
      if [ "$WANDB_LOG_ALL_METRICS" = true ]; then
         RUN_ARGS+=(--wandb-log-all-metrics)
      fi
      if [ "$WANDB_OPEN_METRICS" = true ]; then
         RUN_ARGS+=(--wandb-open-metrics)
      fi
   fi
   if [ -n "$SAVE_INTERVAL_OVERRIDE" ]; then
      RUN_ARGS+=(--save-interval "$SAVE_INTERVAL_OVERRIDE")
   fi
   if [ "$LIGHT_CHECKPOINT" = true ]; then
      RUN_ARGS+=(--light-checkpoint)
   fi
   if [ "$COLOCATE" = true ]; then
      RUN_ARGS+=(--colocate)
   fi
   if [ "$DRY_RUN" = true ]; then
      RUN_ARGS+=(--dry-run)
   fi

   echo
   echo "=== OPSA lowest ${percentage}%: ${MODEL} ==="
   "${RUN_ARGS[@]}"
done
