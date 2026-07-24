#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SLIME_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

STUDENT_HF=""
STUDENT_MEGATRON=""
SAVE_DIR=""
PROMPT_DATA=""
MEGATRON_LM=""
TEACHER_MODEL=""
TEACHER_URL=""
TRAIN_GPUS=""
TEACHER_GPUS=""
RAY_ADDRESS=""
START_LOCAL_RAY=0
PYTHON_BIN="python3"
INPUT_KEY="prompt"
STEPS=700
ACTOR_GPUS=4
ROLLOUT_GPUS=3
TEACHER_TP=1
TEACHER_HOST="127.0.0.1"
TEACHER_PORT=13141
TEACHER_MEM_FRACTION="0.6"
RAY_DASHBOARD_PORT=8265
RAY_GCS_PORT=6379
DRY_RUN=0
EXTRA_TRAIN_ARGS=()
STARTED_TEACHER=0
STARTED_RAY=0
TEACHER_PID=""
TEACHER_LOG=""

usage() {
    cat <<'EOF'
Run pure sampled-token OPD with a Qwen3-1.7B student and an SGLang teacher.

Required:
  --student-hf PATH          Qwen3-1.7B Hugging Face checkpoint
  --student-megatron PATH    Qwen3-1.7B Megatron checkpoint
  --save-dir PATH            resumable output checkpoint directory
  --prompt-data PATH         training JSON/JSONL/Parquet file
  --megatron-lm PATH         Megatron-LM checkout visible to Ray workers

Teacher (choose one):
  --teacher-url URL          existing SGLang /generate endpoint
  --teacher-model PATH       launch a local Qwen3-4B-Instruct-2507 teacher
  --teacher-gpus LIST        GPU IDs/UUIDs for a locally launched teacher
  --teacher-tp N             local teacher tensor parallelism (default: 1)

Ray:
  --ray-address URL          submit to an existing Ray dashboard
  --train-gpus LIST          GPU IDs/UUIDs for a locally launched Ray head
  --actor-gpus N             actor GPUs (default: 4)
  --rollout-gpus N           rollout GPUs (default: 3)

Other:
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
    if [[ "$STARTED_TEACHER" -eq 1 ]] && kill -0 "$TEACHER_PID" 2>/dev/null; then
        kill "$TEACHER_PID" 2>/dev/null || true
        wait "$TEACHER_PID" 2>/dev/null || true
    fi
    if [[ "$STARTED_RAY" -eq 1 ]]; then
        ray stop --force >/dev/null 2>&1 || true
    fi
    if [[ -n "$TEACHER_LOG" ]]; then
        echo "teacher log: ${TEACHER_LOG}" >&2
    fi
    exit "$status"
}
trap cleanup EXIT INT TERM

while [[ $# -gt 0 ]]; do
    case "$1" in
        --student-hf) need_value "$@"; STUDENT_HF=$2; shift 2 ;;
        --student-megatron) need_value "$@"; STUDENT_MEGATRON=$2; shift 2 ;;
        --save-dir) need_value "$@"; SAVE_DIR=$2; shift 2 ;;
        --prompt-data) need_value "$@"; PROMPT_DATA=$2; shift 2 ;;
        --megatron-lm) need_value "$@"; MEGATRON_LM=$2; shift 2 ;;
        --teacher-model) need_value "$@"; TEACHER_MODEL=$2; shift 2 ;;
        --teacher-url) need_value "$@"; TEACHER_URL=${2%/}; shift 2 ;;
        --teacher-gpus) need_value "$@"; TEACHER_GPUS=$2; shift 2 ;;
        --train-gpus) need_value "$@"; TRAIN_GPUS=$2; shift 2 ;;
        --ray-address) need_value "$@"; RAY_ADDRESS=${2%/}; shift 2 ;;
        --python) need_value "$@"; PYTHON_BIN=$2; shift 2 ;;
        --input-key) need_value "$@"; INPUT_KEY=$2; shift 2 ;;
        --steps) need_value "$@"; STEPS=$2; shift 2 ;;
        --actor-gpus) need_value "$@"; ACTOR_GPUS=$2; shift 2 ;;
        --rollout-gpus) need_value "$@"; ROLLOUT_GPUS=$2; shift 2 ;;
        --teacher-tp) need_value "$@"; TEACHER_TP=$2; shift 2 ;;
        --teacher-host) need_value "$@"; TEACHER_HOST=$2; shift 2 ;;
        --teacher-port) need_value "$@"; TEACHER_PORT=$2; shift 2 ;;
        --teacher-mem-fraction) need_value "$@"; TEACHER_MEM_FRACTION=$2; shift 2 ;;
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
[[ -n "$SAVE_DIR" ]] || die "--save-dir is required"
[[ -n "$PROMPT_DATA" ]] || die "--prompt-data is required"
[[ -n "$MEGATRON_LM" ]] || die "--megatron-lm is required"
for value in "$STEPS" "$ACTOR_GPUS" "$ROLLOUT_GPUS" "$TEACHER_TP" "$TEACHER_PORT" "$RAY_DASHBOARD_PORT" "$RAY_GCS_PORT"; do
    positive_integer "$value" || die "expected a positive integer, got: $value"
done
(( TEACHER_PORT <= 65535 )) || die "--teacher-port must be <= 65535"
(( RAY_DASHBOARD_PORT <= 65535 )) || die "--ray-dashboard-port must be <= 65535"
(( RAY_GCS_PORT <= 65535 )) || die "--ray-gcs-port must be <= 65535"
case "$MEGATRON_LM" in *[\"\\]*) die "--megatron-lm cannot contain a quote or backslash" ;; esac

LOCAL_TEACHER=0
if [[ -n "$TEACHER_URL" ]]; then
    [[ -z "$TEACHER_MODEL" ]] || die "--teacher-url and --teacher-model are mutually exclusive"
    [[ -z "$TEACHER_GPUS" ]] || die "--teacher-gpus is only valid with --teacher-model"
    [[ "$TEACHER_URL" =~ ^https?://.+/generate$ ]] || die "--teacher-url must be a full http(s) /generate endpoint"
else
    LOCAL_TEACHER=1
    [[ -n "$TEACHER_MODEL" ]] || die "provide --teacher-url or --teacher-model"
    [[ -n "$TEACHER_GPUS" ]] || die "--teacher-gpus is required when launching the teacher"
    IFS=',' read -r -a teacher_gpu_array <<< "$TEACHER_GPUS"
    [[ ${#teacher_gpu_array[@]} -eq "$TEACHER_TP" ]] || die "--teacher-gpus count must equal --teacher-tp"
    for gpu in "${teacher_gpu_array[@]}"; do [[ -n "$gpu" ]] || die "--teacher-gpus contains an empty item"; done
    TEACHER_URL="http://${TEACHER_HOST}:${TEACHER_PORT}/generate"
fi

TOTAL_TRAIN_GPUS=$((ACTOR_GPUS + ROLLOUT_GPUS))
if [[ -z "$RAY_ADDRESS" ]]; then
    START_LOCAL_RAY=1
    [[ -n "$TRAIN_GPUS" ]] || die "--train-gpus is required when launching Ray locally"
    IFS=',' read -r -a train_gpu_array <<< "$TRAIN_GPUS"
    [[ ${#train_gpu_array[@]} -ge "$TOTAL_TRAIN_GPUS" ]] || die "--train-gpus provides ${#train_gpu_array[@]} GPUs; ${TOTAL_TRAIN_GPUS} are required"
    for gpu in "${train_gpu_array[@]}"; do [[ -n "$gpu" ]] || die "--train-gpus contains an empty item"; done
    if [[ "$LOCAL_TEACHER" -eq 1 ]]; then
        for train_gpu in "${train_gpu_array[@]}"; do
            for teacher_gpu in "${teacher_gpu_array[@]}"; do
                [[ "$train_gpu" != "$teacher_gpu" ]] || die "teacher and training GPU lists overlap at $train_gpu"
            done
        done
    fi
    RAY_ADDRESS="http://127.0.0.1:${RAY_DASHBOARD_PORT}"
else
    [[ "$RAY_ADDRESS" =~ ^https?:// ]] || die "--ray-address must be an http(s) Ray dashboard URL"
fi

if [[ "$START_LOCAL_RAY" -eq 1 ]]; then
    [[ "$RAY_GCS_PORT" != "$RAY_DASHBOARD_PORT" ]] || die "Ray GCS and dashboard ports must differ"
    if [[ "$LOCAL_TEACHER" -eq 1 ]]; then
        [[ "$TEACHER_PORT" != "$RAY_GCS_PORT" ]] || die "teacher and Ray GCS ports must differ"
        [[ "$TEACHER_PORT" != "$RAY_DASHBOARD_PORT" ]] || die "teacher and Ray dashboard ports must differ"
    fi
fi

source "${SLIME_DIR}/scripts/models/qwen3-1.7B.sh"
LOAD_PATH="$STUDENT_MEGATRON"
if [[ -f "${SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
    LOAD_PATH="$SAVE_DIR"
fi

TEACHER_HEALTH_URL="${TEACHER_URL%/generate}/health_generate"
TEACHER_COMMAND=(
    env "CUDA_VISIBLE_DEVICES=${TEACHER_GPUS}" "$PYTHON_BIN" -m sglang.launch_server
    --model-path "$TEACHER_MODEL"
    --host "$TEACHER_HOST"
    --port "$TEACHER_PORT"
    --tp "$TEACHER_TP"
    --chunked-prefill-size 4096
    --mem-fraction-static "$TEACHER_MEM_FRACTION"
)
RAY_START_COMMAND=(
    env "CUDA_VISIBLE_DEVICES=${TRAIN_GPUS}" ray start --head
    --node-ip-address 127.0.0.1
    --port "$RAY_GCS_PORT"
    --num-gpus "$TOTAL_TRAIN_GPUS"
    --disable-usage-stats
    --dashboard-host 127.0.0.1
    --dashboard-port "$RAY_DASHBOARD_PORT"
)
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
    --custom-rm-path slime.rollout.on_policy_distillation.reward_func
    --custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
    --rm-url "$TEACHER_URL"
    --advantage-estimator grpo
    --use-opd
    --opd-type sglang
    --opd-kl-coef 1.0
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

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "SGLang OPD dry run (task reward is fixed at zero):"
    if [[ "$LOCAL_TEACHER" -eq 1 ]]; then print_command "${TEACHER_COMMAND[@]}"; fi
    if [[ $START_LOCAL_RAY -eq 1 ]]; then print_command "${RAY_START_COMMAND[@]}"; fi
    print_command "${RAY_SUBMIT_COMMAND[@]}"
    exit 0
fi

command -v "$PYTHON_BIN" >/dev/null || die "Python executable not found: $PYTHON_BIN"
command -v ray >/dev/null || die "ray executable not found"
command -v curl >/dev/null || die "curl executable not found"
if [[ "$LOCAL_TEACHER" -eq 1 || "$START_LOCAL_RAY" -eq 1 ]]; then
    command -v nvidia-smi >/dev/null || die "nvidia-smi is required to validate local GPU assignments"
fi
if [[ "$LOCAL_TEACHER" -eq 1 ]]; then
    validate_gpu_list_exists "$TEACHER_GPUS" "teacher"
    teacher_gpu_uuid_array=("${VALIDATED_GPU_UUIDS[@]}")
    require_bindable_port "$TEACHER_HOST" "$TEACHER_PORT" "teacher"
fi
if [[ "$START_LOCAL_RAY" -eq 1 ]]; then
    validate_gpu_list_exists "$TRAIN_GPUS" "training"
    train_gpu_uuid_array=("${VALIDATED_GPU_UUIDS[@]}")
    require_bindable_port 127.0.0.1 "$RAY_GCS_PORT" "Ray GCS"
    require_bindable_port 127.0.0.1 "$RAY_DASHBOARD_PORT" "Ray dashboard"
fi
if [[ "$LOCAL_TEACHER" -eq 1 && "$START_LOCAL_RAY" -eq 1 ]]; then
    for teacher_uuid in "${teacher_gpu_uuid_array[@]}"; do
        for train_uuid in "${train_gpu_uuid_array[@]}"; do
            [[ "$teacher_uuid" != "$train_uuid" ]] || die "teacher and training GPU assignments resolve to the same GPU: $teacher_uuid"
        done
    done
fi
[[ -d "$STUDENT_HF" && -f "$STUDENT_HF/config.json" ]] || die "invalid student Hugging Face checkpoint: $STUDENT_HF"
[[ -d "$STUDENT_MEGATRON" && -f "$STUDENT_MEGATRON/latest_checkpointed_iteration.txt" ]] || die "invalid student Megatron checkpoint: $STUDENT_MEGATRON"
[[ -f "$PROMPT_DATA" ]] || die "prompt data not found: $PROMPT_DATA"
[[ -d "$MEGATRON_LM" ]] || die "Megatron-LM directory not found: $MEGATRON_LM"
if [[ "$LOCAL_TEACHER" -eq 1 ]]; then
    [[ -d "$TEACHER_MODEL" && -f "$TEACHER_MODEL/config.json" ]] || die "invalid teacher model: $TEACHER_MODEL"
fi
mkdir -p "$SAVE_DIR"

if [[ "$LOCAL_TEACHER" -eq 1 ]]; then
    if curl --fail --silent --show-error --max-time 2 "$TEACHER_HEALTH_URL" >/dev/null 2>&1; then
        die "teacher port already serves an SGLang health endpoint: $TEACHER_HEALTH_URL"
    fi
    TEACHER_LOG="$(mktemp "${TMPDIR:-/tmp}/slime-opd-teacher.XXXXXX.log")"
    "${TEACHER_COMMAND[@]}" >"$TEACHER_LOG" 2>&1 &
    TEACHER_PID=$!
    STARTED_TEACHER=1
    deadline=$((SECONDS + 600))
    until curl --fail --silent --show-error --max-time 5 "$TEACHER_HEALTH_URL" >/dev/null 2>&1; do
        kill -0 "$TEACHER_PID" 2>/dev/null || die "teacher exited before becoming ready; see $TEACHER_LOG"
        (( SECONDS < deadline )) || die "teacher did not become ready within 600 seconds; see $TEACHER_LOG"
        sleep 5
    done
else
    curl --fail --silent --show-error --max-time 10 "$TEACHER_HEALTH_URL" >/dev/null || die "teacher health check failed: $TEACHER_HEALTH_URL"
fi

if [[ $START_LOCAL_RAY -eq 1 ]]; then
    if ray status >/dev/null 2>&1; then
        die "a Ray cluster is already active; pass its dashboard with --ray-address"
    fi
    "${RAY_START_COMMAND[@]}"
    STARTED_RAY=1
fi

cd "$SLIME_DIR"
"${RAY_SUBMIT_COMMAND[@]}"
