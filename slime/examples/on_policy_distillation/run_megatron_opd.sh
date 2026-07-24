#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

STUDENT_HF=""
STUDENT_MEGATRON=""
TEACHER_HF=""
TEACHER_MEGATRON=""
TEACHER_CKPT_STEP=""
SAVE_DIR=""
PROMPT_DATA=""
MEGATRON_LM=""
TRAIN_GPUS=""
RAY_ADDRESS=""
START_LOCAL_RAY=0
PYTHON_BIN="python3"
INPUT_KEY="prompt"
STEPS=700
ACTOR_GPUS=4
ROLLOUT_GPUS=4
RAY_DASHBOARD_PORT=8265
RAY_GCS_PORT=6379
DRY_RUN=0
EXTRA_TRAIN_ARGS=()
STARTED_RAY=0

usage() {
    cat <<'EOF'
Run pure sampled-token OPD with a same-architecture Megatron teacher.

Required:
  --student-hf PATH          student Hugging Face checkpoint/config
  --student-megatron PATH    initial student Megatron checkpoint
  --teacher-hf PATH          teacher HF config used for architecture validation
  --teacher-megatron PATH    teacher Megatron checkpoint
  --save-dir PATH            resumable output checkpoint directory
  --prompt-data PATH         training JSON/JSONL/Parquet file
  --megatron-lm PATH         Megatron-LM checkout visible to Ray workers

Ray:
  --ray-address URL          submit to an existing Ray dashboard
  --train-gpus LIST          GPU IDs/UUIDs for a locally launched Ray head
  --actor-gpus N             actor GPUs (default: 4)
  --rollout-gpus N           rollout GPUs (default: 4)

Other:
  --teacher-ckpt-step N      load an explicit teacher checkpoint step
  --steps N                  rollout/train steps (default: 700)
  --input-key KEY            prompt field (default: prompt)
  --python PATH              Python executable (default: python3)
  --extra-train-arg ARG      append one train.py argument; repeat as needed
  --dry-run                  validate options and print commands only
  -h, --help                 show this help
EOF
}

die() {
    echo "error: $*" >&2
    exit 2
}

need_value() {
    [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"
}

positive_integer() {
    [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

VALIDATED_GPU_UUIDS=()

validate_gpu_list_exists() {
    local csv=$1
    local label=$2
    local inventory requested index uuid matched_uuid
    local -A seen_uuids=()
    VALIDATED_GPU_UUIDS=()
    inventory="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits)" || die "failed to query GPUs with nvidia-smi"

    IFS=',' read -r -a requested_gpus <<< "$csv"
    for requested in "${requested_gpus[@]}"; do
        matched_uuid=""
        while IFS=',' read -r index uuid; do
            index=${index//[[:space:]]/}
            uuid=${uuid//[[:space:]]/}
            if [[ "$requested" == "$index" || "$requested" == "$uuid" ]]; then
                matched_uuid=$uuid
                break
            fi
        done <<< "$inventory"
        [[ -n "$matched_uuid" ]] || die "$label GPU does not exist: $requested"
        [[ -z "${seen_uuids[$matched_uuid]+present}" ]] || die "$label selects GPU $matched_uuid more than once"
        seen_uuids[$matched_uuid]=1
        VALIDATED_GPU_UUIDS+=("$matched_uuid")
    done
}

require_bindable_port() {
    local host=$1
    local port=$2
    local label=$3
    "$PYTHON_BIN" - "$host" "$port" "$label" <<'PY'
import socket
import sys

host, port, label = sys.argv[1], int(sys.argv[2]), sys.argv[3]
errors = []
for family, socktype, proto, _, address in socket.getaddrinfo(
    host, port, type=socket.SOCK_STREAM, flags=socket.AI_PASSIVE
):
    sock = socket.socket(family, socktype, proto)
    try:
        sock.bind(address)
    except OSError as exc:
        errors.append(str(exc))
    else:
        sock.close()
        raise SystemExit(0)
    finally:
        sock.close()
message = "; ".join(errors) if errors else "no bindable address"
print(f"{label} port {host}:{port} is unavailable: {message}", file=sys.stderr)
raise SystemExit(2)
PY
}
print_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    if [[ "$STARTED_RAY" -eq 1 ]]; then
        ray stop --force >/dev/null 2>&1 || true
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

validate_same_architecture() {
    "$PYTHON_BIN" - "$STUDENT_HF/config.json" "$TEACHER_HF/config.json" <<'PY'
import json
import sys

keys = (
    "model_type",
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
)


def load_text_config(path):
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    return config.get("text_config", config)


student = load_text_config(sys.argv[1])
teacher = load_text_config(sys.argv[2])
mismatches = [(key, student.get(key), teacher.get(key)) for key in keys if student.get(key) != teacher.get(key)]
if mismatches:
    print("Megatron OPD requires identical student and teacher architectures:", file=sys.stderr)
    for key, student_value, teacher_value in mismatches:
        print(f"  {key}: student={student_value!r}, teacher={teacher_value!r}", file=sys.stderr)
    raise SystemExit(2)
PY
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --student-hf) need_value "$@"; STUDENT_HF=$2; shift 2 ;;
        --student-megatron) need_value "$@"; STUDENT_MEGATRON=$2; shift 2 ;;
        --teacher-hf) need_value "$@"; TEACHER_HF=$2; shift 2 ;;
        --teacher-megatron) need_value "$@"; TEACHER_MEGATRON=$2; shift 2 ;;
        --teacher-ckpt-step) need_value "$@"; TEACHER_CKPT_STEP=$2; shift 2 ;;
        --save-dir) need_value "$@"; SAVE_DIR=$2; shift 2 ;;
        --prompt-data) need_value "$@"; PROMPT_DATA=$2; shift 2 ;;
        --megatron-lm) need_value "$@"; MEGATRON_LM=$2; shift 2 ;;
        --train-gpus) need_value "$@"; TRAIN_GPUS=$2; shift 2 ;;
        --ray-address) need_value "$@"; RAY_ADDRESS=${2%/}; shift 2 ;;
        --python) need_value "$@"; PYTHON_BIN=$2; shift 2 ;;
        --input-key) need_value "$@"; INPUT_KEY=$2; shift 2 ;;
        --steps) need_value "$@"; STEPS=$2; shift 2 ;;
        --actor-gpus) need_value "$@"; ACTOR_GPUS=$2; shift 2 ;;
        --rollout-gpus) need_value "$@"; ROLLOUT_GPUS=$2; shift 2 ;;
        --ray-dashboard-port) need_value "$@"; RAY_DASHBOARD_PORT=$2; shift 2 ;;
        --ray-gcs-port) need_value "$@"; RAY_GCS_PORT=$2; shift 2 ;;
        --extra-train-arg) need_value "$@"; EXTRA_TRAIN_ARGS+=("$2"); shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown option: $1" ;;
    esac
done

[[ -n "$STUDENT_HF" ]] || die "--student-hf is required"
[[ -n "$STUDENT_MEGATRON" ]] || die "--student-megatron is required"
[[ -n "$TEACHER_HF" ]] || die "--teacher-hf is required for architecture validation"
[[ -n "$TEACHER_MEGATRON" ]] || die "--teacher-megatron is required"
[[ -n "$SAVE_DIR" ]] || die "--save-dir is required"
[[ -n "$PROMPT_DATA" ]] || die "--prompt-data is required"
[[ -n "$MEGATRON_LM" ]] || die "--megatron-lm is required"
for value in "$STEPS" "$ACTOR_GPUS" "$ROLLOUT_GPUS" "$RAY_DASHBOARD_PORT" "$RAY_GCS_PORT"; do
    positive_integer "$value" || die "expected a positive integer, got: $value"
done
if [[ -n "$TEACHER_CKPT_STEP" ]]; then positive_integer "$TEACHER_CKPT_STEP" || die "--teacher-ckpt-step must be positive"; fi
(( RAY_DASHBOARD_PORT <= 65535 )) || die "--ray-dashboard-port must be <= 65535"
(( RAY_GCS_PORT <= 65535 )) || die "--ray-gcs-port must be <= 65535"
case "$MEGATRON_LM" in *[\"\\]*) die "--megatron-lm cannot contain a quote or backslash" ;; esac

TOTAL_TRAIN_GPUS=$((ACTOR_GPUS + ROLLOUT_GPUS))
if [[ -z "$RAY_ADDRESS" ]]; then
    START_LOCAL_RAY=1
    [[ -n "$TRAIN_GPUS" ]] || die "--train-gpus is required when launching Ray locally"
    IFS=',' read -r -a train_gpu_array <<< "$TRAIN_GPUS"
    [[ ${#train_gpu_array[@]} -ge "$TOTAL_TRAIN_GPUS" ]] || die "--train-gpus provides ${#train_gpu_array[@]} GPUs; ${TOTAL_TRAIN_GPUS} are required"
    for gpu in "${train_gpu_array[@]}"; do [[ -n "$gpu" ]] || die "--train-gpus contains an empty item"; done
    RAY_ADDRESS="http://127.0.0.1:${RAY_DASHBOARD_PORT}"
else
    [[ "$RAY_ADDRESS" =~ ^https?:// ]] || die "--ray-address must be an http(s) Ray dashboard URL"
fi

if [[ "$START_LOCAL_RAY" -eq 1 ]]; then
    [[ "$RAY_GCS_PORT" != "$RAY_DASHBOARD_PORT" ]] || die "Ray GCS and dashboard ports must differ"
fi

source "${SLIME_DIR}/scripts/models/qwen3-1.7B.sh"
LOAD_PATH="$STUDENT_MEGATRON"
if [[ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
    LOAD_PATH="$SAVE_DIR"
fi

RAY_START_COMMAND=(
    env "CUDA_VISIBLE_DEVICES=${TRAIN_GPUS}" ray start --head
    --node-ip-address 127.0.0.1
    --port "$RAY_GCS_PORT"
    --num-gpus "$TOTAL_TRAIN_GPUS"
    --disable-usage-stats
    --dashboard-host 127.0.0.1
    --dashboard-port "$RAY_DASHBOARD_PORT"
)
OPD_TEACHER_ARGS=(--opd-teacher-load "$TEACHER_MEGATRON")
if [[ -n "$TEACHER_CKPT_STEP" ]]; then OPD_TEACHER_ARGS+=(--opd-teacher-ckpt-step "$TEACHER_CKPT_STEP"); fi
TRAIN_COMMAND=(
    "$PYTHON_BIN" train.py
    --actor-num-nodes 1
    --actor-num-gpus-per-node "$ACTOR_GPUS"
    --rollout-num-gpus "$ROLLOUT_GPUS"
    --num-gpus-per-node "$TOTAL_TRAIN_GPUS"
    "${MODEL_ARGS[@]}"
    --hf-checkpoint "$STUDENT_HF"
    --load "$LOAD_PATH"
    --save "$SAVE_DIR"
    --save-interval 20
    --prompt-data "$PROMPT_DATA"
    --input-key "$INPUT_KEY"
    --apply-chat-template
    --disable-thinking
    --rollout-shuffle
    --num-rollout "$STEPS"
    --rollout-batch-size 64
    --n-samples-per-prompt 1
    --rollout-max-response-len 12000
    --rollout-temperature 1
    --num-steps-per-rollout 1
    --global-batch-size 64
    --balance-data
    --custom-rm-path slime.rollout.on_policy_distillation.zero_reward_func
    --advantage-estimator grpo
    --use-opd
    --opd-type megatron
    --opd-kl-coef 1.0
    "${OPD_TEACHER_ARGS[@]}"
    --optimizer adam
    --lr 1e-6
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --tensor-model-parallel-size 1
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu 16384
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.4
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    "${EXTRA_TRAIN_ARGS[@]}"
)
RUNTIME_ENV_JSON="{\"env_vars\":{\"PYTHONPATH\":\"${MEGATRON_LM}\",\"CUDA_DEVICE_MAX_CONNECTIONS\":\"1\"}}"
RAY_SUBMIT_COMMAND=(
    ray job submit
    --address "$RAY_ADDRESS"
    --working-dir "$SLIME_DIR"
    --runtime-env-json "$RUNTIME_ENV_JSON"
    -- "${TRAIN_COMMAND[@]}"
)

if [[ -f "$STUDENT_HF/config.json" && -f "$TEACHER_HF/config.json" ]] && command -v "$PYTHON_BIN" >/dev/null; then
    validate_same_architecture
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Megatron OPD dry run (task reward is fixed at zero; teacher architecture must match):"
    if [[ "$START_LOCAL_RAY" -eq 1 ]]; then print_command "${RAY_START_COMMAND[@]}"; fi
    print_command "${RAY_SUBMIT_COMMAND[@]}"
    exit 0
fi

command -v "$PYTHON_BIN" >/dev/null || die "Python executable not found: $PYTHON_BIN"
command -v ray >/dev/null || die "ray executable not found"
if [[ "$START_LOCAL_RAY" -eq 1 ]]; then
    command -v nvidia-smi >/dev/null || die "nvidia-smi is required to validate local GPU assignments"
    validate_gpu_list_exists "$TRAIN_GPUS" "training"
    require_bindable_port 127.0.0.1 "$RAY_GCS_PORT" "Ray GCS"
    require_bindable_port 127.0.0.1 "$RAY_DASHBOARD_PORT" "Ray dashboard"
fi
[[ -d "$STUDENT_HF" && -f "$STUDENT_HF/config.json" ]] || die "invalid student Hugging Face checkpoint: $STUDENT_HF"
[[ -d "$TEACHER_HF" && -f "$TEACHER_HF/config.json" ]] || die "invalid teacher Hugging Face config: $TEACHER_HF"
[[ -d "$STUDENT_MEGATRON" && -f "$STUDENT_MEGATRON/latest_checkpointed_iteration.txt" ]] || die "invalid student Megatron checkpoint: $STUDENT_MEGATRON"
[[ -d "$TEACHER_MEGATRON" && -f "$TEACHER_MEGATRON/latest_checkpointed_iteration.txt" ]] || die "invalid teacher Megatron checkpoint: $TEACHER_MEGATRON"
[[ -f "$PROMPT_DATA" ]] || die "prompt data not found: $PROMPT_DATA"
[[ -d "$MEGATRON_LM" ]] || die "Megatron-LM directory not found: $MEGATRON_LM"
validate_same_architecture
mkdir -p "$SAVE_DIR"

if [[ "$START_LOCAL_RAY" -eq 1 ]]; then
    if ray status >/dev/null 2>&1; then
        die "a Ray cluster is already active; pass its dashboard with --ray-address"
    fi
    "${RAY_START_COMMAND[@]}"
    STARTED_RAY=1
fi

cd "$SLIME_DIR"
"${RAY_SUBMIT_COMMAND[@]}"
