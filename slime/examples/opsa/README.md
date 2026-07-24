# OPSA

OPSA (On-Policy Self-Adaptation) trains only the actor's lowest-log-probability
response tokens. Selection is local to each data-parallel rank's packed batch.
For a fraction \(f\), OPSA selects
\(K=\max(1,\lfloor fN\rfloor)\) valid response tokens by ascending actor log
probability, ranks the selected tokens by entropy, and assigns

\[
A = A_{\max} + (A_{\min} - A_{\max})r.
\]

The canonical preset uses \(f=0.2\), \(A_{\min}=-1.0\), and
\(A_{\max}=-0.5\). Equal selected-token entropies receive \(-1.0\). Unselected
tokens have zero advantage and are excluded from both the policy-loss mask and
the loss denominator.

All commands below assume the modified Slime project is the current working
directory:

```bash
cd /path/to/On-Policy-Self-Adaptation/slime
```

This directory contains only the final method and two targeted ablations:

- `opsa`: entropy-ranked advantages in `[-1.0, -0.5]`.
- `fixed-negative`: fixed advantage `-0.5` on the same selected tokens.
- `fixed-positive`: fixed advantage `+0.2` on the same selected tokens.

There is no entropy threshold, token/EOS/think mask, position reweighting,
clipping, forced token value, top-1 branch, or random branch. Training task
reward is zero. OPSA is reference-free: `--ref-load` initializes the actor but
no reference-model forward pass or KL loss is used.

## Model presets

| `--model` | Steps | Actor / rollout GPUs | Training TP | Rollout / eval length | Max tokens/GPU |
|---|---:|---:|---:|---:|---:|
| `qwen3-1.7b` | 700 | 4 / 4 | 1 | 12,000 / 32,768 | 16,384 |
| `qwen3-4b` (Base) | 1,000 | 4 / 4 | 2 | 12,000 / 32,768 | 24,576 |
| `qwen3.5-9b` | 1,000 | 2 / 6 | 2 | 16,384 / 32,768 | 32,768 |

All presets use non-thinking rollout, batch size 64, learning rate `1e-6`, and
save/evaluation interval 20. Qwen3.5-9B additionally enables optimizer CPU
offload. Install its optional linear-attention dependency with:

```bash
pip install -e '.[qwen35]'
```

## Prepare checkpoints

The Hugging Face checkpoint is used by SGLang rollout. Training is initialized
from a separately converted Megatron checkpoint. From the modified Slime
project directory, select the model definition and convert it:

```bash
export PYTHONPATH="${MEGATRON_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
source scripts/models/qwen3-1.7B.sh  # or qwen3-4B.sh / qwen3.5-9B.sh

torchrun --nproc-per-node 1 tools/convert_hf_to_torch_dist.py \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --save "${ACTOR_CHECKPOINT}"
```

Set `HF_CHECKPOINT`, `ACTOR_CHECKPOINT`, and `MEGATRON_PATH` to external paths;
models and Megatron-LM are intentionally not bundled in this repository.

## Run

Inspect a fully resolved configuration without requiring checkpoints, datasets,
GPUs, or Ray:

```bash
bash examples/opsa/run_opsa.sh \
  --model qwen3-1.7b \
  --preset opsa \
  --fraction 0.2 \
  --dry-run
```

Start a real single-node run:

```bash
bash examples/opsa/run_opsa.sh \
  --model qwen3-1.7b \
  --preset opsa \
  --fraction 0.2 \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --actor-checkpoint "${ACTOR_CHECKPOINT}" \
  --save-dir "${OUTPUT_ROOT}/qwen3-1.7b/opsa-lowest20" \
  --prompt-data "${TRAIN_DATA}" \
  --eval-data "${EVAL_DATA}" \
  --megatron-path "${MEGATRON_PATH}"
```

Choose `--preset fixed-negative` or `--preset fixed-positive` for the two fixed
advantage ablations. The launcher starts a local Ray cluster when
`--ray-address` is omitted, validates both its GCS and dashboard ports, and
stops only that cluster on exit. Override the local ports with `--ray-port` and
`--dashboard-port`. To use a cluster you already manage, pass its dashboard URL:

```bash
bash examples/opsa/run_opsa.sh \
  --ray-address "${RAY_DASHBOARD_URL}" \
  --model qwen3-4b \
  --preset fixed-negative \
  --fraction 0.2 \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --actor-checkpoint "${ACTOR_CHECKPOINT}" \
  --save-dir "${OUTPUT_ROOT}/qwen3-4b/fixed-negative-lowest20" \
  --prompt-data "${TRAIN_DATA}" \
  --eval-data "${EVAL_DATA}" \
  --megatron-path "${MEGATRON_PATH}"
```

Checkpoints include optimizer and RNG state by default and can be resumed with
`--resume-from`. `--light-checkpoint` explicitly creates smaller,
non-resumable checkpoints.

## Lowest-token sweep

The sweep launcher runs 10%, 20%, 30%, and 40% sequentially and creates a
separate directory below `--save-root` for each fraction:

```bash
bash examples/opsa/run_lowest_sweep.sh \
  --model qwen3.5-9b \
  --hf-checkpoint "${HF_CHECKPOINT}" \
  --actor-checkpoint "${ACTOR_CHECKPOINT}" \
  --save-root "${OUTPUT_ROOT}" \
  --prompt-data "${TRAIN_DATA}" \
  --eval-data "${EVAL_DATA}" \
  --megatron-path "${MEGATRON_PATH}"
```

Use `--fractions 0.1,0.3` to run a subset. Add `--dry-run` to validate and print
every generated command without starting Ray or training.

## Validation status

This open-source configuration is CPU/CLI validated: selector and argument
behavior is covered by CPU tests, and every model/method launcher is exercised
with `--dry-run`. The three model presets have not been re-run end to end on GPUs
as part of this cleanup, so the validation status must not be interpreted as a
new reproduction of their training results.
