#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
SLIME_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"

MODEL="qwen3-1.7b"
PRESET="opsa"
TOKEN_FRACTION="0.2"
HF_CHECKPOINT="${HF_CHECKPOINT:-}"
ACTOR_CHECKPOINT="${ACTOR_CHECKPOINT:-${REF_LOAD:-}}"
RESUME_FROM="${RESUME_FROM:-}"
SAVE_DIR="${SAVE_DIR:-}"
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
  bash examples/opsa/run_opsa.sh [options]

Method:
  --model NAME                qwen3-1.7b, qwen3-4b, or qwen3.5-9b
  --preset NAME               opsa, fixed-negative, or fixed-positive
  --fraction FLOAT            Lowest-token fraction in (0, 1] (default: 0.2)

Paths:
  --hf-checkpoint PATH        Hugging Face checkpoint used by rollout
  --actor-checkpoint PATH     Megatron checkpoint used to initialize the actor
  --resume-from PATH          Resume a prior full training checkpoint
  --save-dir PATH             Output checkpoint directory
  --prompt-data FILE          Training JSON/JSONL file
  --eval-data FILE            Evaluation JSON/JSONL file
  --megatron-path DIR         Megatron-LM checkout

Runtime:
  --ray-address URL           Submit to an existing Ray dashboard
  --dashboard-port PORT       Local Ray dashboard port (default: 8265)
  --light-checkpoint          Omit optimizer and RNG state (not resumable)
  --dry-run                   Print the resolved command without checking paths
  -h, --help                  Show this help

The corresponding uppercase environment variables may be used for path and Ray
options. Command-line values take precedence.
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
      --preset)
         require_value "$@"
         PRESET="$2"
         shift 2
         ;;
      --fraction)
         require_value "$@"
         TOKEN_FRACTION="$2"
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
      --resume-from)
         require_value "$@"
         RESUME_FROM="$2"
         shift 2
         ;;
      --save-dir)
         require_value "$@"
         SAVE_DIR="$2"
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

case "$PRESET" in
   opsa|fixed-negative|fixed-positive) ;;
   *) die "unsupported preset '$PRESET'" ;;
esac

if ! [[ "$TOKEN_FRACTION" =~ ^(0([.][0-9]+)?|1([.]0*)?)$ ]]; then
   die "--fraction must be a number in (0, 1]"
fi
if ! awk -v fraction="$TOKEN_FRACTION" 'BEGIN { exit !(fraction > 0 && fraction <= 1) }'; then
   die "--fraction must be a number in (0, 1]"
fi

case "$DASHBOARD_PORT" in
   ""|*[!0-9]*) die "--dashboard-port must be an integer from 1 to 65535" ;;
esac
if [ "$DASHBOARD_PORT" -lt 1 ] || [ "$DASHBOARD_PORT" -gt 65535 ]; then
   die "--dashboard-port must be an integer from 1 to 65535"
fi
if [ "$LIGHT_CHECKPOINT" = true ] && [ -n "$RESUME_FROM" ]; then
   die "--light-checkpoint cannot be combined with --resume-from"
fi

source "${SCRIPT_DIR}/models/${MODEL}.sh"
MODEL_CONFIG="${SLIME_ROOT}/${MODEL_CONFIG_RELATIVE}"
if [ ! -f "$MODEL_CONFIG" ]; then
   die "model definition not found: $MODEL_CONFIG"
fi
source "$MODEL_CONFIG"

TOTAL_GPUS=$((ACTOR_GPUS + ROLLOUT_GPUS))
if [ $((ACTOR_GPUS % TENSOR_MODEL_PARALLEL_SIZE)) -ne 0 ]; then
   die "actor GPUs must be divisible by tensor model parallel size"
fi
if [ $((ROLLOUT_GPUS % ROLLOUT_GPUS_PER_ENGINE)) -ne 0 ]; then
   die "rollout GPUs must be divisible by rollout GPUs per engine"
fi

if [ "$DRY_RUN" = true ]; then
   HF_CHECKPOINT="${HF_CHECKPOINT:-<HF_CHECKPOINT>}"
   ACTOR_CHECKPOINT="${ACTOR_CHECKPOINT:-<ACTOR_MEGATRON_CHECKPOINT>}"
   SAVE_DIR="${SAVE_DIR:-<SAVE_DIR>}"
   PROMPT_DATA="${PROMPT_DATA:-<TRAIN_DATA>}"
   EVAL_DATA="${EVAL_DATA:-<EVAL_DATA>}"
   MEGATRON_PATH="${MEGATRON_PATH:-<MEGATRON_LM>}"
else
   [ -n "$HF_CHECKPOINT" ] || die "--hf-checkpoint is required"
   [ -n "$ACTOR_CHECKPOINT" ] || die "--actor-checkpoint is required"
   [ -n "$SAVE_DIR" ] || die "--save-dir is required"
   [ -n "$PROMPT_DATA" ] || die "--prompt-data is required"
   [ -n "$EVAL_DATA" ] || die "--eval-data is required"
   [ -n "$MEGATRON_PATH" ] || die "--megatron-path is required"

   [ -d "$HF_CHECKPOINT" ] || die "Hugging Face checkpoint is not a directory: $HF_CHECKPOINT"
   [ -d "$ACTOR_CHECKPOINT" ] || die "actor checkpoint is not a directory: $ACTOR_CHECKPOINT"
   [ -f "$PROMPT_DATA" ] || die "training data is not a file: $PROMPT_DATA"
   [ -f "$EVAL_DATA" ] || die "evaluation data is not a file: $EVAL_DATA"
   [ -d "$MEGATRON_PATH" ] || die "Megatron-LM path is not a directory: $MEGATRON_PATH"
   if [ -n "$RESUME_FROM" ] && [ ! -d "$RESUME_FROM" ]; then
      die "resume checkpoint is not a directory: $RESUME_FROM"
   fi

   command -v python3 >/dev/null 2>&1 || die "python3 is required"
   command -v ray >/dev/null 2>&1 || die "ray is required"

   if [ -z "$RAY_ADDRESS" ]; then
      command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is required when starting local Ray"
      GPU_LIST="$(nvidia-smi --query-gpu=index --format=csv,noheader)" ||
         die "failed to query local GPUs"
      DETECTED_GPUS=0
      while IFS= read -r gpu_index; do
         if [ -n "$gpu_index" ]; then
            DETECTED_GPUS=$((DETECTED_GPUS + 1))
         fi
      done <<< "$GPU_LIST"
      if [ "$DETECTED_GPUS" -lt "$TOTAL_GPUS" ]; then
         die "$MODEL requires $TOTAL_GPUS local GPUs, but only $DETECTED_GPUS were detected"
      fi
      if ! python3 -c 'import socket, sys; sock = socket.socket(); sock.bind(("127.0.0.1", int(sys.argv[1]))); sock.close()' "$DASHBOARD_PORT" 2>/dev/null; then
         die "local Ray dashboard port is already in use: $DASHBOARD_PORT"
      fi
   fi
   mkdir -p "$SAVE_DIR"
fi

case "$PRESET" in
   opsa)
      OPSA_ARGS=(
         --opsa-mode entropy
         --opsa-token-fraction "$TOKEN_FRACTION"
         --opsa-advantage-min -1.0
         --opsa-advantage-max -0.5
      )
      ;;
   fixed-negative)
      OPSA_ARGS=(
         --opsa-mode fixed
         --opsa-token-fraction "$TOKEN_FRACTION"
         --opsa-fixed-advantage -0.5
      )
      ;;
   fixed-positive)
      OPSA_ARGS=(
         --opsa-mode fixed
         --opsa-token-fraction "$TOKEN_FRACTION"
         --opsa-fixed-advantage 0.2
      )
      ;;
esac

CKPT_ARGS=(
   --hf-checkpoint "$HF_CHECKPOINT"
   --ref-load "$ACTOR_CHECKPOINT"
   --save "$SAVE_DIR"
   --save-interval 20
)
if [ -n "$RESUME_FROM" ]; then
   CKPT_ARGS+=(--load "$RESUME_FROM")
fi
if [ "$LIGHT_CHECKPOINT" = true ]; then
   CKPT_ARGS+=(--no-save-optim --no-save-rng --no-load-optim --no-load-rng)
fi

ROLLOUT_ARGS=(
   --prompt-data "$PROMPT_DATA"
   --input-key prompt
   --apply-chat-template
   --disable-thinking
   --rollout-shuffle
   --num-rollout "$NUM_ROLLOUT"
   --rollout-batch-size 64
   --n-samples-per-prompt 1
   --rollout-max-response-len "$ROLLOUT_MAX_RESPONSE_LEN"
   --rollout-temperature 1
   --num-steps-per-rollout 1
   --global-batch-size 64
   --balance-data
   --custom-rm-path slime.rollout.opsa.reward_func
   --custom-reward-post-process-path slime.rollout.opsa.post_process_rewards
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data aime "$EVAL_DATA"
   --eval-input-key prompt
   --eval-label-key label
   --n-samples-per-eval-prompt 4
   --eval-max-response-len "$EVAL_MAX_RESPONSE_LEN"
   --eval-top-p 0.8
   --eval-temperature 0.7
   --eval-top-k 20
   --eval-rm-type math
   --log-passrate
)

PERF_ARGS=(
   --tensor-model-parallel-size "$TENSOR_MODEL_PARALLEL_SIZE"
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU"
)

ALGORITHM_ARGS=(
   --advantage-estimator opsa
   "${OPSA_ARGS[@]}"
   --entropy-coef 0.0
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)
if [ "$OPTIMIZER_CPU_OFFLOAD" = true ]; then
   OPTIMIZER_ARGS+=(
      --optimizer-cpu-offload
      --overlap-cpu-optimizer-d2h-h2d
      --use-precision-aware-optimizer
   )
fi

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine "$ROLLOUT_GPUS_PER_ENGINE"
   --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC"
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

TRAIN_COMMAND=(
   python3 train.py
   --actor-num-nodes 1
   --actor-num-gpus-per-node "$ACTOR_GPUS"
   --rollout-num-gpus "$ROLLOUT_GPUS"
   "${MODEL_ARGS[@]}"
   "${CKPT_ARGS[@]}"
   "${ROLLOUT_ARGS[@]}"
   "${OPTIMIZER_ARGS[@]}"
   "${ALGORITHM_ARGS[@]}"
   "${PERF_ARGS[@]}"
   "${EVAL_ARGS[@]}"
   "${SGLANG_ARGS[@]}"
   "${MISC_ARGS[@]}"
)

HAS_NVLINK=0
if [ "$DRY_RUN" = false ] && [ -z "$RAY_ADDRESS" ]; then
   GPU_TOPOLOGY="$(nvidia-smi topo -m 2>/dev/null || true)"
   if [[ "$GPU_TOPOLOGY" == *"NV"* ]]; then
      HAS_NVLINK=1
   fi
fi

RUNTIME_PYTHONPATH="${MEGATRON_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export RUNTIME_PYTHONPATH HAS_NVLINK
RUNTIME_ENV_JSON="$(
   python3 -c 'import json, os; print(json.dumps({"env_vars": {"PYTHONPATH": os.environ["RUNTIME_PYTHONPATH"], "CUDA_DEVICE_MAX_CONNECTIONS": "1", "NCCL_NVLS_ENABLE": os.environ["HAS_NVLINK"], "SGLANG_DISABLE_CUDNN_CHECK": "1"}}))'
)"

echo "Model:               $MODEL_DISPLAY_NAME"
echo "Preset:              $PRESET"
echo "Lowest fraction:     $TOKEN_FRACTION"
echo "Actor/Rollout GPUs:  ${ACTOR_GPUS}/${ROLLOUT_GPUS}"
echo "Training TP:         $TENSOR_MODEL_PARALLEL_SIZE"
echo "Rollout/Eval length: ${ROLLOUT_MAX_RESPONSE_LEN}/${EVAL_MAX_RESPONSE_LEN}"
echo "Checkpoint mode:     $([ "$LIGHT_CHECKPOINT" = true ] && echo light || echo resumable)"

if [ "$DRY_RUN" = true ]; then
   if [ -n "$RAY_ADDRESS" ]; then
      DRY_RAY_ADDRESS="$RAY_ADDRESS"
      echo "Ray:                 existing cluster at $RAY_ADDRESS"
   else
      DRY_RAY_ADDRESS="http://127.0.0.1:${DASHBOARD_PORT}"
      echo "Ray:                 start local cluster with $TOTAL_GPUS GPUs"
   fi
   printf '\n[dry-run]'
   printf ' %q' ray job submit --address "$DRY_RAY_ADDRESS" --working-dir "$SLIME_ROOT" --runtime-env-json "$RUNTIME_ENV_JSON" -- "${TRAIN_COMMAND[@]}"
   printf '\n'
   exit 0
fi

RAY_STARTED_BY_SCRIPT=false
cleanup() {
   if [ "$RAY_STARTED_BY_SCRIPT" = true ]; then
      ray stop --force >/dev/null 2>&1 || true
   fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [ -z "$RAY_ADDRESS" ]; then
   ray start \
      --head \
      --node-ip-address 127.0.0.1 \
      --num-gpus "$TOTAL_GPUS" \
      --disable-usage-stats \
      --dashboard-host 127.0.0.1 \
      --dashboard-port "$DASHBOARD_PORT"
   RAY_STARTED_BY_SCRIPT=true
   RAY_ADDRESS="http://127.0.0.1:${DASHBOARD_PORT}"
else
   echo "Using existing Ray cluster: $RAY_ADDRESS"
fi

cd "$SLIME_ROOT"
ray job submit \
   --address "$RAY_ADDRESS" \
   --working-dir "$SLIME_ROOT" \
   --runtime-env-json "$RUNTIME_ENV_JSON" \
   -- "${TRAIN_COMMAND[@]}"
