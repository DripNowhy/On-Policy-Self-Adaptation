# On-Policy Distillation

On-policy distillation (OPD) trains on the student's own rollouts while matching the teacher's log-probability for each sampled response token. It is an additive training signal on top of a base advantage estimator.

## Public arguments

| Argument | Meaning |
| --- | --- |
| `--use-opd` | Enable standard sampled-token OPD. |
| `--opd-type {sglang,megatron}` | Select the teacher backend. |
| `--opd-kl-coef FLOAT` | Scale the OPD training signal. |
| `--opd-teacher-load PATH` | Megatron teacher checkpoint; required only for `megatron`. |
| `--opd-teacher-ckpt-step N` | Optional explicit Megatron teacher checkpoint step. |

Standard OPD and OPSA (`--advantage-estimator opsa`) are mutually exclusive.

## Teacher backends

### SGLang

The student sends its complete token sequence to an independent SGLang teacher. Because the teacher is a separate server, its architecture may differ from the student's. Its tokenizer must still use the same token-ID mapping for the submitted sequence. Slime validates the returned sequence length, response token IDs, and log-probability shape before training; a malformed or misaligned response fails immediately.

Use these rollout hooks with a full SGLang `/generate` URL:

```bash
--custom-rm-path slime.rollout.on_policy_distillation.reward_func
--custom-reward-post-process-path slime.rollout.on_policy_distillation.post_process_rewards
--rm-url http://teacher-host:port/generate
--use-opd
--opd-type sglang
--opd-kl-coef 1.0
```

The post-processor returns zero task reward. It handles an empty response as an empty list rather than accidentally selecting the full teacher sequence.

### Megatron

Megatron mode loads a second checkpoint and computes teacher log-probabilities during the training forward pass. The teacher and student must have identical model architecture because the runtime swaps their weights into the same Megatron model. A larger or otherwise structurally different teacher must use SGLang mode.

```bash
--custom-rm-path slime.rollout.on_policy_distillation.zero_reward_func
--use-opd
--opd-type megatron
--opd-kl-coef 1.0
--opd-teacher-load /path/to/same-architecture-teacher
```

The zero-reward hook makes this a pure distillation run. Omit it only when deliberately combining OPD with a task reward.

## Reproducible examples

See [`examples/on_policy_distillation/`](../../../examples/on_policy_distillation/) for:

- Qwen3-1.7B student to Qwen3-4B-Instruct-2507 using a local or external SGLang teacher.
- A same-architecture Megatron teacher with a preflight architecture check.

Both launchers support `--dry-run`, resumable checkpoints, an existing Ray cluster, and scoped cleanup of only the processes they start.
