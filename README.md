<div align="center">

# OPSA: On-Policy Self-Adaptation

**A teacher-free RL objective that learns from the tokens the current policy least expects.**

[![Code](https://img.shields.io/badge/CODE-000000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/DripNowhy/On-Policy-Self-Adaptation)
[![W&B Logs](https://img.shields.io/badge/W%26B%20TRAINING%20LOGS-%2300B4AB?style=for-the-badge&logo=weightsandbiases&logoColor=white&labelColor=000000)](https://wandb.ai/whywhyyy0731-purdue-university/opsa/workspace)
[![Reproduce](https://img.shields.io/badge/REPRODUCE-%23FFD14D?style=for-the-badge&logo=gnubash&logoColor=black)](slime/examples/opsa/README.md)
[![Slime](https://img.shields.io/badge/BUILT%20ON%20SLIME-6F42C1?style=for-the-badge&logo=python&logoColor=white)](https://github.com/THUDM/slime)
[![License](https://img.shields.io/badge/APACHE--2.0-A42C25?style=for-the-badge&logo=apache&logoColor=white)](LICENSE)

**English** · [中文](README_zh.md)

<p>
  <a href="#introduction">📖 Introduction</a> •
  <a href="#method">🧠 Method</a> •
  <a href="#main-results">📊 Main Results</a> •
  <a href="#getting-started">✨ Getting Started</a>
</p>
<p>
  <a href="#ablations">🧪 Ablations</a> •
  <a href="#model-presets">📦 Model Presets</a> •
  <a href="#reference">🔧 Reference</a>
</p>

</div>

> Adapt where the policy is least confident, using only signals produced by the
> current policy.

# 📖Introduction

**OPSA studies whether a student policy can improve without a teacher model,
reward model, reference model, or task reward.**

The current actor first identifies the response tokens it considers least likely.
Only those tokens are trained. Their negative advantages are shaped by the actor's
own entropy, giving stronger suppression to selected tokens with higher uncertainty.

<p align="center">
  <img src="assets/opsa-overview.png" alt="OPSA paper overview: method, training dynamics, and Qwen3-1.7B results" width="96%">
</p>

<p align="center"><sub>Overview from the OPSA manuscript.</sub></p>

|  | |
|---|---|
| 🚫 **Teacher-free** | No teacher model, reward model, reference forward, or KL term. Task reward is zero. |
| 🎯 **Token-sparse** | Gradients flow only through the lowest-logp fraction of valid response tokens—20% by default. |
| 🌡️ **Entropy-shaped** | Selected-token entropy is mapped to negative advantage from `-0.5` to `-1.0`. |

# 🧠Method

For each data-parallel rank, OPSA performs three operations on the local packed
training batch:

1. **Score.** The current Megatron actor recomputes sampled-token log probabilities
   and entropies for every valid response token.
2. **Select.** After concatenating the local valid tokens, OPSA chooses the lowest
   actor-logp fraction:

   $$K = \max(1, \lfloor fN \rfloor).$$

3. **Shape and update.** Entropy is min-max normalized inside the selected set:

   $$
   r_i = \frac{H_i-H_{\min}}{H_{\max}-H_{\min}},
   \qquad
   A_i = A_{\max} + (A_{\min}-A_{\max}) r_i.
   $$

The canonical configuration uses $f=0.2$, $A_{\min}=-1.0$, and
$A_{\max}=-0.5$. If all selected entropies are equal, every selected token
receives `-1.0`. Unselected tokens receive zero advantage and are removed from
both the policy-loss numerator and denominator.

> [!NOTE]
> Selection is **DP-local**: tokens are never ranked across data-parallel workers.
> OPSA uses zero task reward and performs no reference-model forward pass or KL
> loss.

# 📊Main Results

The table below reports the main results from the OPSA manuscript. Each
benchmark cell is **Avg@32 / Pass@32**; all values are percentages.

| Model | Variant | AIME24 | AIME25 | HMMT25 | MBPP+ |
|---|---|---:|---:|---:|---:|
| Qwen3-1.7B | Base | 13.44 / 40.00 | 9.69 / 30.00 | 5.73 / 23.33 | 58.24 / 70.10 |
| Qwen3-1.7B | **+ OPSA** | **48.85 / 80.00** | **35.31 / 66.67** | **23.33 / 50.00** | **59.44 / 73.02** |
| Qwen3-4B | Base | 23.33 / 56.67 | 20.52 / 56.67 | 13.13 / 33.33 | 66.93 / 74.34 |
| Qwen3-4B | **+ OPSA** | **62.08 / 83.33** | **58.44 / 83.33** | **37.40 / 60.00** | **68.35 / 75.67** |
| Qwen3.5-9B | Base | 76.35 / 93.33 | 56.04 / 93.33 | 44.48 / 86.67 | 77.33 / 91.27 |
| Qwen3.5-9B | **+ OPSA** | **87.81 / 96.67** | **76.98 / 96.67** | **67.40 / 93.33** | **79.27 / 92.53** |

All models are trained and evaluated in non-thinking mode. Training uses only
questions from DAPO-17k, without labels or ground-truth answers. Evaluation uses
SGLang with 32 responses per prompt, temperature `0.7`, top-k `20`, top-p `0.8`,
and a maximum generation length of `32,768`. Checkpoints are selected by
validation Avg@4.

Selected Qwen3-1.7B and Qwen3-4B training curves are available in the
[public W&B workspace](https://wandb.ai/whywhyyy0731-purdue-university/opsa/workspace).

# ✨Getting Started

## Environment setup

```bash
git clone https://github.com/DripNowhy/On-Policy-Self-Adaptation.git
cd On-Policy-Self-Adaptation/slime
pip install -e .
```

## Reproduce OPSA

Inspect the fully expanded canonical configuration without allocating GPUs or
requiring local datasets and checkpoints:

```bash
bash examples/opsa/run_opsa.sh \
  --model qwen3-1.7b \
  --preset opsa \
  --fraction 0.2 \
  --dry-run
```

> [!NOTE]
> Before removing `--dry-run`, provide the external Hugging Face checkpoint,
> Megatron checkpoint, prompt/evaluation data, and Megatron-LM paths described in
> the [reproducibility guide](slime/examples/opsa/README.md).

# 🧪Ablations

OPSA keeps a compact set of controlled variants. All fixed-advantage variants use
the exact same lowest-token selector as canonical OPSA.

| Configuration | Selected tokens | Advantage |
|---|---|---|
| **Canonical OPSA** | Lowest 20% actor logp by default | Entropy-mapped from `-0.5` to `-1.0` |
| **Fixed negative** | Same lowest-token selector | `-0.5` |
| **Fixed positive** | Same lowest-token selector | `+0.2` |
| **Fraction sweep** | Lowest 10 / 20 / 30 / 40% | Canonical entropy mapping |

<p align="center">
  <img src="assets/lowest-token-fraction-ablation.png" alt="AIME24 Avg@4 during OPSA training with the lowest 10, 20, 30, or 40 percent of tokens" width="58%">
</p>

<p align="center">
  <sub>Paper ablation of the selected-token fraction. The lowest 20%, 30%, and 40% settings all reach above 45 Avg@4 on AIME24.</sub>
</p>

```bash
# Fixed-advantage controls
bash examples/opsa/run_opsa.sh --model qwen3-1.7b --preset fixed-negative --dry-run
bash examples/opsa/run_opsa.sh --model qwen3-1.7b --preset fixed-positive  --dry-run

# Sequential lowest-token fraction sweep
bash examples/opsa/run_lowest_sweep.sh --model qwen3-1.7b --dry-run
```

Historical entropy thresholds, EOS/think masks, position reweighting, clipping,
forced values, top-1 branches, random branches, trace-only experiment code,
experiment outputs, and machine-specific paths are intentionally excluded.

# 📦Model Presets

| Model | Steps | Actor / rollout GPUs | TP | Rollout / eval length | Max tokens/GPU |
|---|---:|---:|---:|---:|---:|
| **Qwen3-1.7B** | 700 | 4 / 4 | 1 | 12k / 32k | 16,384 |
| **Qwen3-4B Base** | 1,000 | 4 / 4 | 2 | 12k / 32k | 24,576 |
| **Qwen3.5-9B** | 1,000 | 2 / 6 | 2 | 16k / 32k | 32,768 |

All presets use non-thinking generation, batch size 64, learning rate `1e-6`,
and save/evaluation intervals of 20. Qwen3.5-9B retains optimizer CPU offload
and has an optional `flash-linear-attention==0.4.1` dependency.

# 🔧Reference

<details>
<summary><b>Public OPSA interface</b></summary>

<br>

| Flag | Value | Default | Notes |
|---|---|---|---|
| `--advantage-estimator` | `opsa` | — | Enables OPSA and bypasses reward-based estimators |
| `--opsa-mode` | `entropy` \| `fixed` | `entropy` | Entropy mode requests actor entropy; fixed mode does not |
| `--opsa-token-fraction` | float in `(0, 1]` | `0.2` | Fraction of valid response tokens selected |
| `--opsa-advantage-min` | float | `-1.0` | Most negative entropy-ranked advantage |
| `--opsa-advantage-max` | float | `-0.5` | Least negative; must satisfy `min < max < 0` |
| `--opsa-fixed-advantage` | non-zero float | `None` | Required by—and only valid with—fixed mode |

OPSA is mutually exclusive with standard OPD and advantage normalization.

</details>

<details>
<summary><b>Optional W&B tracking</b></summary>

<br>

W&B is off by default and is enabled only when `--wandb-project` or
`WANDB_PROJECT` is provided. If no group is specified, the launcher derives one
from the model, preset, and selected-token fraction:

```bash
bash examples/opsa/run_opsa.sh \
  --model qwen3-1.7b --preset opsa --fraction 0.2 \
  --wandb-project opsa --wandb-mode online --dry-run
```

The compact default logs:

- `train/{loss,entropy_loss,grad_norm,lr-pg_*}`
- `rollout/opsa/{selected_fraction,advantage_mean}`
- `rollout/{log_probs,entropy,truncated_ratio,repetition_frac}`
- `rollout/response_len/{min,mean,max}`
- `eval/<dataset>-{avg@4,pass@4}`
- `eval/<dataset>/{repetition_frac,truncated_ratio}`
- `eval/<dataset>/response_len/{min,mean,max}`
- `perf/{rollout_time,actor_train_tok_per_s}`

The compact stream does **not** publish `train/opsa/*`. Use
`--wandb-log-all-metrics` for the full slime stream and `--wandb-open-metrics`
for SGLang OpenMetrics. Both are explicit opt-ins. The launcher never accepts or
prints an API key; inject `WANDB_API_KEY` through the local environment or the
existing Ray cluster's secret mechanism. See the
[tracking guide](slime/examples/opsa/README.md#optional-wb-tracking) for every CLI
and environment option.

</details>

<details>
<summary><b>Repository layout</b></summary>

<br>

| Path | Contents |
|---|---|
| [`slime/`](slime/) | Installable modified slime project and native runtime |
| [`slime/examples/opsa/`](slime/examples/opsa/) | OPSA launchers, presets, conversion commands, ablations, and sweeps |
| [`slime/examples/on_policy_distillation/`](slime/examples/on_policy_distillation/) | Standard OPD baselines using an SGLang or same-architecture Megatron teacher |
| [`slime/tests/test_opsa.py`](slime/tests/test_opsa.py) and [`slime/tests/test_opsa_loss_mask.py`](slime/tests/test_opsa_loss_mask.py) | CPU selector, argument, mask, and loss tests |
| [`UPSTREAM.md`](UPSTREAM.md) | Snapshot provenance and upstream attribution |

Datasets, Hugging Face checkpoints, Megatron checkpoints, and Megatron-LM are
external dependencies and are not included.

</details>

## Validation status

> [!IMPORTANT]
> The implementation and launchers are CPU- and CLI-validated. The paper results
> above were produced on 8 NVIDIA H100 or H200 GPUs. End-to-end training of all
> three public presets was not repeated as part of the open-source cleanup.

Tests cover DP-local selection, 10/20/30/40% fractions, the at-least-one-token
rule, equal entropy, fixed positive and negative advantages, empty/malformed
inputs, loss masking, argument validation, reference-free startup, and standard
OPD regression behavior. Launchers are syntax checked and exercised with
`--dry-run`.

## Upstream and license

The implementation is based on local slime commit `594c562`, with public
reference point
[`THUDM/slime@0988f0f`](https://github.com/THUDM/slime/commit/0988f0f4a0ab55d1bb3ce6285a597d912144fa80).
See [`UPSTREAM.md`](UPSTREAM.md) for details.

This repository is released under the [Apache License 2.0](LICENSE).
