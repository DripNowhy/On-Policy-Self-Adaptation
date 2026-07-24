# 在策略蒸馏

在策略蒸馏（OPD）使用 student 自己生成的 rollout，并在每个被采样的 response token 上匹配 teacher 的 log-probability。它是在基础 advantage estimator 之上附加的训练信号。

## 公开参数

| 参数 | 含义 |
| --- | --- |
| `--use-opd` | 启用标准 sampled-token OPD。 |
| `--opd-type {sglang,megatron}` | 选择 teacher 后端。 |
| `--opd-kl-coef FLOAT` | 缩放 OPD 训练信号。 |
| `--opd-teacher-load PATH` | Megatron teacher checkpoint；仅 `megatron` 模式必需。 |
| `--opd-teacher-ckpt-step N` | 可选的 Megatron teacher checkpoint step。 |

标准 OPD 与 OPSA（`--advantage-estimator opsa`）互斥。

## Teacher 后端

### SGLang

Student 会把完整 token 序列发送给独立的 SGLang teacher。由于 teacher 是独立服务，它可以使用不同模型架构；但它的 tokenizer 必须对输入序列使用相同的 token ID 映射。训练前，Slime 会校验返回序列长度、每个 response token ID 以及 log-probability shape；任何畸形或错位的返回都会立即报错。

使用完整的 SGLang `/generate` URL 配置 rollout hook：

```bash
--custom-rm-path slime.rollout.on_policy_distillation.reward_func
--custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
--rm-url http://teacher-host:port/generate
--use-opd
--opd-type sglang
--opd-kl-coef 1.0
```

Post-processor 返回零 task reward。空 response 会得到空 list，不会因为 `[-0:]` 而错误选中整段 teacher 序列。

### Megatron

Megatron 模式加载第二个 checkpoint，并在训练 forward 中计算 teacher log-probability。运行时会在同一个 Megatron model 上切换权重，因此 teacher 与 student 必须具有完全相同的模型架构。更大或结构不同的 teacher 必须使用 SGLang 模式。

```bash
--custom-rm-path slime.rollout.on_policy_distillation.zero_reward_func
--use-opd
--opd-type megatron
--opd-kl-coef 1.0
--opd-teacher-load /path/to/same-architecture-teacher
```

零奖励 hook 使其成为纯蒸馏训练。只有在明确要把 OPD 与 task reward 结合时才应替换它。

## 可复现实例

参见 [`examples/on_policy_distillation/`](../../../examples/on_policy_distillation/)：

- Qwen3-1.7B student 到 Qwen3-4B-Instruct-2507 的本地或外部 SGLang teacher 示例。
- 带启动前架构检查的同架构 Megatron teacher 示例。

两个 launcher 都支持 `--dry-run`、可续训 checkpoint、已有 Ray 集群，以及只清理自身启动进程的安全生命周期。
