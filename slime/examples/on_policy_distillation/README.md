# Standard On-Policy Distillation

These examples keep only sampled-token OPD. The student generates each response, and the teacher scores those exact token IDs. Pure OPD uses zero task reward; the only training signal is the weighted teacher/student log-probability difference.

OPD exposes five public options:

- `--use-opd`
- `--opd-type {sglang,megatron}`
- `--opd-kl-coef FLOAT`
- `--opd-teacher-load PATH` (Megatron only)
- `--opd-teacher-ckpt-step N` (optional, Megatron only)

OPD is separate from the base advantage estimator, but it is mutually exclusive with OPSA (`--advantage-estimator opsa`). The launchers use GRPO with one response per prompt and a zero task reward, so the base task advantage is zero.

All commands below assume the modified Slime project is the current working
directory:

```bash
export SLIME_DIR=/path/to/On-Policy-Self-Adaptation/slime
cd "$SLIME_DIR"
```

## Prepare the student checkpoint

Both examples train Qwen3-1.7B. Convert its Hugging Face checkpoint once; replace every placeholder with a path in your environment.

```bash
source scripts/models/qwen3-1.7B.sh
PYTHONPATH="$MEGATRON_LM" python3 tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "$QWEN3_1_7B_HF" \
  --save "$QWEN3_1_7B_MEGATRON"
```

The launchers resume from `--save-dir` when it contains `latest_checkpointed_iteration.txt`; otherwise they initialize from `--student-megatron`. Optimizer and RNG state are saved by default.

## SGLang teacher: Qwen3-1.7B to Qwen3-4B-Instruct-2507

SGLang mode supports a different teacher architecture because the teacher is an independent server. It does **not** permit arbitrary tokenizers: the teacher must assign the same token IDs the same meaning as the student. The post-processor validates the returned sequence length and every response token ID before attaching teacher log-probabilities, and rejects malformed or misaligned responses.

Launch a local teacher and a local Ray head (the GPU lists are deliberately user supplied):

```bash
bash examples/on_policy_distillation/run_sglang_opd.sh \
  --student-hf "$QWEN3_1_7B_HF" \
  --student-megatron "$QWEN3_1_7B_MEGATRON" \
  --teacher-model "$QWEN3_4B_INSTRUCT_HF" \
  --teacher-gpus "$TEACHER_GPU_IDS" \
  --train-gpus "$TRAIN_GPU_IDS" \
  --prompt-data "$TRAIN_DATA" \
  --save-dir "$OUTPUT_DIR" \
  --megatron-lm "$MEGATRON_LM"
```

To use an already running teacher and Ray cluster, replace the local teacher/GPU options with:

```bash
bash examples/on_policy_distillation/run_sglang_opd.sh \
  --student-hf "$QWEN3_1_7B_HF" \
  --student-megatron "$QWEN3_1_7B_MEGATRON" \
  --teacher-url "$TEACHER_GENERATE_URL" \
  --ray-address "$RAY_DASHBOARD_URL" \
  --prompt-data "$TRAIN_DATA" \
  --save-dir "$OUTPUT_DIR" \
  --megatron-lm "$MEGATRON_LM"
```

`TEACHER_GENERATE_URL` must be the complete `http(s)://.../generate` endpoint. The launcher checks the corresponding `/health_generate` endpoint before submitting the job.

## Megatron teacher: same architecture only

Megatron mode swaps the actor weights in place to run the teacher forward pass. Student and teacher must therefore have the same model architecture. This example fixes the model arguments to Qwen3-1.7B and compares the student and teacher Hugging Face `config.json` files before launch; a Qwen3-4B teacher is rejected.

Convert the same-architecture teacher with the Qwen3-1.7B model arguments, then run:

```bash
bash examples/on_policy_distillation/run_megatron_opd.sh \
  --student-hf "$QWEN3_1_7B_HF" \
  --student-megatron "$QWEN3_1_7B_MEGATRON" \
  --teacher-hf "$QWEN3_1_7B_TEACHER_HF" \
  --teacher-megatron "$QWEN3_1_7B_TEACHER_MEGATRON" \
  --train-gpus "$TRAIN_GPU_IDS" \
  --prompt-data "$TRAIN_DATA" \
  --save-dir "$OUTPUT_DIR" \
  --megatron-lm "$MEGATRON_LM"
```

Pass `--ray-address "$RAY_DASHBOARD_URL"` instead of `--train-gpus` to use an existing cluster. Use `--teacher-ckpt-step N` to select a teacher step explicitly.

## Validation and lifecycle

Both launchers default to 700 steps, batch size 64, non-thinking rollouts, learning rate `1e-6`, and save interval 20. Add `--dry-run` to print the exact commands without checking paths, starting services, creating directories, or using GPUs. A dry run still validates option combinations and GPU counts.

For a real run, paths, checkpoint trackers, and teacher health are checked first. Local GPU IDs or UUIDs are resolved with `nvidia-smi`, compared by canonical UUID to prevent overlap, and every local teacher/Ray port must pass a socket-bind preflight. A launcher stops only the local teacher and Ray head that it started; it never uses `pkill`, and it does not stop externally supplied services.
