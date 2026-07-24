# OPSA：On-Policy Self-Adaptation

[English](README.md)

OPSA 是一种无需 teacher 的训练方法。它聚焦当前 student actor 最不确定的
response token：先选择 actor log probability 最低的 token，再根据这些 token
的 actor entropy 分配 token-level negative advantage。

![OPSA 方法概览](assets/opsa-overview.svg)

这个仓库的根目录代表 OPSA 项目本身。完整的修改版
[Slime](https://github.com/THUDM/slime) 被整理为独立的
[`slime/`](slime/) 子项目；根目录 README 和 `assets/` 只用于介绍 OPSA。

## 方法

在每个 data-parallel rank 上，OPSA 会拼接该 rank 本地 packed training
batch 中的所有有效 response token。设有效 token 数量为 \(N\)，选择比例为
\(f\)，OPSA 使用当前 Megatron actor 重算的 log probability，选择其中最低的：

$$
K = \max(1, \lfloor fN \rfloor).
$$

随后仅在被选中的 token 内对 entropy 做 min-max normalization：

$$
r_i = \frac{H_i-H_{\min}}{H_{\max}-H_{\min}},
\qquad
A_i = A_{\max} + (A_{\min}-A_{\max})r_i.
$$

Canonical OPSA 使用 \(f=0.2\)、\(A_{\min}=-1.0\) 和
\(A_{\max}=-0.5\)。如果选中 token 的 entropy 全部相同，它们都获得
\(-1.0\)。未选中的 token advantage 为零，并且同时从 policy-loss 的分子和
分母中排除。

选择过程是 DP-local 的，不会跨 data-parallel worker 排序 token。OPSA 使用
zero task reward，不执行 reference-model forward，也不计算 KL loss。

## 保留的配置

| 配置 | 选择的 token | Advantage |
|---|---|---|
| Canonical OPSA | 默认 actor logp 最低 20% | 按 entropy 从 `-0.5` 映射到 `-1.0` |
| Fixed negative | 相同的 lowest-token selector | `-0.5` |
| Fixed positive | 相同的 lowest-token selector | `+0.2` |
| Fraction sweep | 最低 10/20/30/40% | Canonical entropy mapping |

OPSA 和 standard OPD 的公开接口有意删除了历史实验中的 entropy threshold、
EOS/think 特殊 mask、position reweighting、clipping、forced value、
top-1/random 分支、实验产物以及机器相关路径。

## 仓库结构

- [`slime/`](slime/)：可独立安装的修改版 Slime 工程及原生 runtime。
- [`slime/examples/opsa/`](slime/examples/opsa/)：OPSA launcher、模型 preset、
  checkpoint 转换、fixed-advantage 消融和 fraction sweep。
- [`slime/examples/on_policy_distillation/`](slime/examples/on_policy_distillation/)：
  使用 SGLang teacher 或同架构 Megatron teacher 的 standard OPD baseline。
- [`slime/tests/test_opsa.py`](slime/tests/test_opsa.py) 和
  [`slime/tests/test_opsa_loss_mask.py`](slime/tests/test_opsa_loss_mask.py)：
  selector、参数、mask 与 loss normalization 的 CPU 测试。
- [`UPSTREAM.md`](UPSTREAM.md)：本地快照来源及上游 attribution。

数据集、Hugging Face checkpoint、Megatron checkpoint 和 Megatron-LM 都是外部
依赖，不会复制进本仓库。

## 快速开始

```bash
git clone https://github.com/DripNowhy/On-Policy-Self-Adaptation.git
cd On-Policy-Self-Adaptation/slime
pip install -e .
```

不分配 GPU、也不要求本地 checkpoint 和数据集，先检查完整展开后的配置：

```bash
bash examples/opsa/run_opsa.sh \
  --model qwen3-1.7b \
  --preset opsa \
  --fraction 0.2 \
  --dry-run
```

检查两个 fixed-advantage 消融：

```bash
bash examples/opsa/run_opsa.sh --model qwen3-1.7b --preset fixed-negative --dry-run
bash examples/opsa/run_opsa.sh --model qwen3-1.7b --preset fixed-positive --dry-run
```

展开顺序执行的 10/20/30/40% sweep：

```bash
bash examples/opsa/run_lowest_sweep.sh --model qwen3-1.7b --dry-run
```

在移除 `--dry-run` 之前，请阅读[完整复现指南](slime/examples/opsa/README.md)。

## OPSA 公开参数

```text
--advantage-estimator opsa
--opsa-mode {entropy,fixed}
--opsa-token-fraction FLOAT
--opsa-advantage-min FLOAT
--opsa-advantage-max FLOAT
--opsa-fixed-advantage FLOAT
```

Entropy mode 会自动请求 actor entropy；fixed mode 不计算 entropy。OPSA 与
standard OPD、advantage normalization 互斥。

## 模型 preset

| 模型 | Steps | Actor / rollout GPU | TP | Rollout / eval length | Max tokens/GPU |
|---|---:|---:|---:|---:|---:|
| Qwen3-1.7B | 700 | 4 / 4 | 1 | 12k / 32k | 16,384 |
| Qwen3-4B Base | 1,000 | 4 / 4 | 2 | 12k / 32k | 24,576 |
| Qwen3.5-9B | 1,000 | 2 / 6 | 2 | 16k / 32k | 32,768 |

三个 preset 均使用 non-thinking generation、batch size 64、learning rate
`1e-6`、save/evaluation interval 20。Qwen3.5-9B 保留 optimizer CPU offload，
并提供可选依赖 `flash-linear-attention==0.4.1`。

## 验证范围

当前版本完成了 CPU 和 CLI 验证。测试覆盖 DP-local selection、10/20/30/40%
比例、至少选择一个 token、entropy 相同时的行为、fixed 正负 advantage、空或
非法输入、loss mask、参数校验、reference-free startup 和 standard OPD 回归。
所有 launcher 均执行 shell syntax check 和 `--dry-run`。

本次整理没有重新执行三个公开模型的端到端 GPU 训练，因此不把上述验证表述为新的
训练复现结果。

## 上游与许可证

实现基于本地 Slime commit `594c562`，对应的公开参考点为
[`THUDM/slime@0988f0f`](https://github.com/THUDM/slime/commit/0988f0f4a0ab55d1bb3ce6285a597d912144fa80)。
详细信息见 [`UPSTREAM.md`](UPSTREAM.md)。

本仓库使用 [Apache License 2.0](LICENSE)。
