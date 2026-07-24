#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

MODEL="qwen3-1.7b"
FRACTIONS="0.1,0.2,0.3,0.4"
HF_CHECKPOINT="${HF_CHECKPOINT:-}"
ACTOR_CHECKPOINT="${ACTOR_CHECKPOINT:-${REF_LOAD:-}}"
SAVE_ROOT="${SAVE_ROOT:-}"
PROMPT_DATA="${PROMPT_DATA:-}"
EVAL_DATA="${EVAL_DATA:-}"
MEGATRON_PATH="${MEGATRON_PATH:-}"
RAY_ADDRESS="${RAY_ADDRESS:-}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
LIGHT_CHECKPOINT=false
DRY_RUN=false

usage() {
   cat <<'EOF'
Usage:
  bash examples/opsa/run_lowest_sweep.sh [options]

Runs OPSA sequentially for lowest 10%, 20%, 30%, and 40% by default.

Options:
  --model NAME                qwen3-1.7b, qwen3-4b, or qwen3.5-9b
  --fractions LIST            Comma-separated fractions (default: 0.1,0.2,0.3,0.4)
  --hf-checkpoint PATH        Hugging Face checkpoint used by rollout
  --actor-checkpoint PATH     Megatron checkpoint used to initialize the actor
  --save-root PATH            Parent directory for fraction-specific runs
  --prompt-data FILE          Training JSON/JSONL file
  --eval-data FILE            Evaluation JSON/JSONL file
  --megatron-path DIR         Megatron-LM checkout
  --ray-address URL           Submit every run to an existing Ray dashboard
  --dashboard-port PORT       Local Ray dashboard port (default: 8265)
  --light-checkpoint          Produce non-resumable weight-only checkpoints
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
      --dashboard-port)
         require_value "$@"
         DASHBOARD_PORT="$2"
         shift 2
         ;;
      --light-checkpoint)
         LIGHT_CHECKPOINT=true
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
   qwen3-1.7b|qwen3-4b|qwen3.5-9b) ;;
   *) die "unsupported model '$MODEL'" ;;
esac

if [ -z "$FRACTIONS" ]; then
   die "--fractions must contain at least one value"
fi
if [ "$DRY_RUN" = false ] && [ -z "$SAVE_ROOT" ]; then
   die "--save-root is required"
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
   if [ "$LIGHT_CHECKPOINT" = true ]; then
      RUN_ARGS+=(--light-checkpoint)
   fi
   if [ "$DRY_RUN" = true ]; then
      RUN_ARGS+=(--dry-run)
   fi

   echo
   echo "=== OPSA lowest ${percentage}%: ${MODEL} ==="
   "${RUN_ARGS[@]}"
done
