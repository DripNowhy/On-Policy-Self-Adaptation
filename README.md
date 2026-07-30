# On-Policy Self-Adaptation (OPSA)

[中文](README_zh.md)

OPSA is a teacher-free on-policy training method. It improves a policy using
only the policy's own token probabilities and entropies, without a teacher
model, reward model, reference-model forward pass, or task reward.

![Overview of On-Policy Self-Adaptation](assets/opsa-overview.png)

## Method

On each data-parallel rank, OPSA ranks the valid response tokens in the local
packed batch by log probability recomputed by the current actor. For \(N\)
valid tokens and selection fraction \(f\), it selects

$$
K = \max(1, \lfloor fN \rfloor)
$$

tokens with the lowest log probabilities. Entropy is min-max normalized within
the selected set, and the token-level advantage is

$$
r_i = \frac{H_i-H_{\min}}{H_{\max}-H_{\min}},
\qquad
A_i = A_{\max} + (A_{\min}-A_{\max})r_i.
$$

The default configuration uses \(f=0.2\), \(A_{\min}=-1.0\), and
\(A_{\max}=-0.5\). If all selected entropies are equal, every selected token
receives \(-1.0\). Unselected tokens are excluded from both the policy-loss
numerator and denominator.

## Results

Each benchmark cell reports **Avg@32 / Pass@32** in percent. All results are
from the OPSA manuscript.

| Model | Variant | AIME24 | AIME25 | HMMT25 | MBPP+ |
|---|---|---:|---:|---:|---:|
| Qwen3-1.7B | Base | 13.44 / 40.00 | 9.69 / 30.00 | 5.73 / 23.33 | 58.24 / 70.10 |
| Qwen3-1.7B | **+ OPSA** | **48.85 / 80.00** | **35.31 / 66.67** | **23.33 / 50.00** | **59.44 / 73.02** |
| Qwen3-4B | Base | 23.33 / 56.67 | 20.52 / 56.67 | 13.13 / 33.33 | 66.93 / 74.34 |
| Qwen3-4B | **+ OPSA** | **62.08 / 83.33** | **58.44 / 83.33** | **37.40 / 60.00** | **68.35 / 75.67** |
| Qwen3.5-9B | Base | 76.35 / 93.33 | 56.04 / 93.33 | 44.48 / 86.67 | 77.33 / 91.27 |
| Qwen3.5-9B | **+ OPSA** | **87.81 / 96.67** | **76.98 / 96.67** | **67.40 / 93.33** | **79.27 / 92.53** |

The models are trained and evaluated in non-thinking mode. Training uses only
the questions from DAPO-17k. Evaluation samples 32 responses per prompt with
temperature `0.7`, top-k `20`, top-p `0.8`, and maximum response length
`32,768`.

## Reproduce OPSA

### Installation

```bash
git clone https://github.com/DripNowhy/On-Policy-Self-Adaptation.git
cd On-Policy-Self-Adaptation/slime
pip install -e .
```

### CPU dry run

The dry run resolves and prints the complete training command. It does not
require checkpoints, datasets, GPUs, or Ray, and it does not start training.

```bash
bash examples/opsa/run_opsa.sh \
  --model qwen3-1.7b \
  --preset opsa \
  --fraction 0.2 \
  --dry-run
```

### GPU training

Before training, prepare a Hugging Face checkpoint, its converted Megatron
actor checkpoint, training and evaluation JSON/JSONL files, and a Megatron-LM
checkout. See the
[checkpoint conversion instructions](slime/examples/opsa/README.md#prepare-checkpoints).

After replacing the paths below, this command starts the canonical Qwen3-1.7B
OPSA run on a single machine with 8 GPUs:

```bash
bash examples/opsa/run_opsa.sh \
  --model qwen3-1.7b \
  --preset opsa \
  --fraction 0.2 \
  --hf-checkpoint /path/to/Qwen3-1.7B \
  --actor-checkpoint /path/to/qwen3-1.7b-megatron \
  --save-dir /path/to/outputs/opsa-qwen3-1.7b-lowest20 \
  --prompt-data /path/to/train.jsonl \
  --eval-data /path/to/eval.jsonl \
  --megatron-path /path/to/Megatron-LM
```

This preset uses 4 actor GPUs and 4 rollout GPUs, runs for 700 steps, and saves
and evaluates every 20 steps. The launcher starts a local Ray cluster and stops
only the cluster it started. For an existing Ray cluster or the other supported
models, see the [launcher guide](slime/examples/opsa/README.md).

## Training logs

Reference training curves and metrics are available in the
[public W&B workspace](https://wandb.ai/whywhyyy0731-purdue-university/opsa/workspace).

## License

This project is built on [slime](https://github.com/THUDM/slime) and released
under the [Apache License 2.0](LICENSE). See [UPSTREAM.md](UPSTREAM.md) for
snapshot provenance.
