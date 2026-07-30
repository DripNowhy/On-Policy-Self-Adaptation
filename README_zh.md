# On-Policy Self-Adaptation (OPSA)

[English](README.md)

OPSA 是一种无需 teacher 的 on-policy 训练方法。它只使用当前策略自身的 token
probability 和 entropy 完成改进，不需要 teacher model、reward model、
reference-model forward 或 task reward。

![On-Policy Self-Adaptation 方法总览](assets/opsa-overview.png)

## 方法

在每个 data-parallel rank 上，OPSA 使用当前 actor 重算的 log probability，
对本地 packed batch 中的有效 response token 排序。设有效 token 数量为 \(N\)，
选择比例为 \(f\)，OPSA 选择 log probability 最低的

$$
K = \max(1, \lfloor fN \rfloor)
$$

个 token。随后只在这些 token 内对 entropy 做 min-max normalization，并分配
token-level advantage：

$$
r_i = \frac{H_i-H_{\min}}{H_{\max}-H_{\min}},
\qquad
A_i = A_{\max} + (A_{\min}-A_{\max})r_i.
$$

默认配置使用 \(f=0.2\)、\(A_{\min}=-1.0\) 和
\(A_{\max}=-0.5\)。如果已选 token 的 entropy 全部相同，它们都获得
\(-1.0\)。未选 token 同时从 policy loss 的分子和分母中排除。

## 结果

每个 benchmark 单元格均为 **Avg@32 / Pass@32**，单位为百分比。所有数值均来自
OPSA 论文。

| 模型 | 版本 | AIME24 | AIME25 | HMMT25 | MBPP+ |
|---|---|---:|---:|---:|---:|
| Qwen3-1.7B | Base | 13.44 / 40.00 | 9.69 / 30.00 | 5.73 / 23.33 | 58.24 / 70.10 |
| Qwen3-1.7B | **+ OPSA** | **48.85 / 80.00** | **35.31 / 66.67** | **23.33 / 50.00** | **59.44 / 73.02** |
| Qwen3-4B | Base | 23.33 / 56.67 | 20.52 / 56.67 | 13.13 / 33.33 | 66.93 / 74.34 |
| Qwen3-4B | **+ OPSA** | **62.08 / 83.33** | **58.44 / 83.33** | **37.40 / 60.00** | **68.35 / 75.67** |
| Qwen3.5-9B | Base | 76.35 / 93.33 | 56.04 / 93.33 | 44.48 / 86.67 | 77.33 / 91.27 |
| Qwen3.5-9B | **+ OPSA** | **87.81 / 96.67** | **76.98 / 96.67** | **67.40 / 93.33** | **79.27 / 92.53** |

所有模型均使用 non-thinking 模式训练和评测。训练只使用 DAPO-17k 中的问题。
评测时每道题采样 32 个 response，temperature 为 `0.7`、top-k 为 `20`、
top-p 为 `0.8`、最大 response length 为 `32,768`。

## 复现 OPSA

### 安装

```bash
git clone https://github.com/DripNowhy/On-Policy-Self-Adaptation.git
cd On-Policy-Self-Adaptation/slime
pip install -e .
```

### CPU dry run

Dry run 会展开并打印完整训练命令，不需要 checkpoint、数据集、GPU 或 Ray，
也不会启动训练。

```bash
bash examples/opsa/run_opsa.sh \
  --model qwen3-1.7b \
  --preset opsa \
  --fraction 0.2 \
  --dry-run
```

### GPU 训练

训练前需要准备 Hugging Face checkpoint、由它转换得到的 Megatron actor
checkpoint、训练和评测 JSON/JSONL 文件，以及 Megatron-LM。转换方法见
[checkpoint 转换说明](slime/examples/opsa/README.md#prepare-checkpoints)。

替换下面的路径后，该命令会在单机 8 张 GPU 上启动 canonical Qwen3-1.7B OPSA：

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

该 preset 使用 4 张 actor GPU 和 4 张 rollout GPU，共训练 700 steps，每
20 steps 保存和评测一次。Launcher 会自动启动本地 Ray，并且退出时只停止自己
启动的集群。已有 Ray 集群或其他模型的用法见
[launcher 说明](slime/examples/opsa/README.md)。

## 训练日志

训练曲线和指标可以参考
[公开 W&B workspace](https://wandb.ai/whywhyyy0731-purdue-university/opsa/workspace)。

## 许可证

本项目基于 [slime](https://github.com/THUDM/slime)，使用
[Apache License 2.0](LICENSE)。快照来源见 [UPSTREAM.md](UPSTREAM.md)。
