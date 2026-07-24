from argparse import Namespace
from collections.abc import Callable, Iterator
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from megatron.core import mpu
from torch.utils.checkpoint import checkpoint

from slime.utils.distributed_utils import distributed_masked_whiten
from slime.utils.misc import load_function
from slime.utils.ppo_utils import (
    calculate_log_probs_and_entropy,
    compute_approx_kl,
    compute_gspo_kl,
    compute_opsm_mask,
    compute_policy_loss,
    get_advantages_and_returns_batch,
    get_grpo_returns,
    get_reinforce_plus_plus_baseline_advantages,
    get_reinforce_plus_plus_returns,
)
from slime.utils.types import RolloutBatch

from .cp_utils import (
    all_gather_with_cp,
    get_logits_and_tokens_offset_with_cp,
    get_sum_of_sample_mean,
    slice_log_prob_with_cp,
)
from .opd import apply_opd_kl_to_advantages
from .opsa import compute_opsa


def get_responses(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield response-aligned `(logits_chunk, tokens_chunk)` pairs per sample.

    After squeezing batch dimension and applying temperature scaling, this
    function extracts the logits and tokens corresponding to response segments
    for each sample. When context parallelism is disabled, it slices directly
    from the concatenated sequence. With context parallelism enabled, it
    handles split sequences across ranks.

    Args:
        logits: Model outputs with shape `[1, T, V]` (policy) or `[1, T, 1]`
            (value). Must be float32.
        args: Configuration containing `rollout_temperature` for scaling.
        unconcat_tokens: List of token tensors (prompt+response) per sample.
        total_lengths: Total sequence lengths (prompt+response) per sample.
        response_lengths: Response segment lengths per sample.

    Yields:
        Tuple of `(logits_chunk, tokens_chunk)` where `logits_chunk` is shape
        `[R, V]` (policy) or `[R, 1]` (value) and `tokens_chunk` is shape `[R]`
        (1D int64), both aligned to response tokens for one sample.
    """
    qkv_format = args.qkv_format

    assert logits.dtype == torch.float32, f"{logits.dtype}"
    assert len(logits.shape) == 3, f"{logits.shape}"

    if qkv_format == "thd":
        assert logits.size(0) == 1, f"{logits.shape}"
        logits = logits.squeeze(0)
    else:
        assert max_seq_lens is not None
        logits = logits.view(-1, logits.size(-1))

    if args.rollout_temperature != 1.0:
        logits = logits.div(args.rollout_temperature)

    cp_size = mpu.get_context_parallel_world_size()
    end = 0
    seq_start = 0
    for i, (tokens, total_length, response_length) in enumerate(
        zip(unconcat_tokens, total_lengths, response_lengths, strict=False)
    ):
        max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None

        if cp_size == 1:
            if qkv_format == "bshd":
                end = max_seq_len * i + total_length
                start = end - response_length
            else:
                end += total_length
                start = end - response_length
            logits_chunk = logits[start - 1 : end - 1]
            tokens_chunk = tokens[-response_length:]
        elif args.allgather_cp:
            # DSA: global concat then contiguous CP split. Each rank owns logits for
            # global positions [chunk_start, chunk_end).
            logits_local_len = logits.size(0)
            cp_rank = mpu.get_context_parallel_rank()
            chunk_start = cp_rank * logits_local_len
            chunk_end = chunk_start + logits_local_len

            prompt_length = total_length - response_length
            resp_token_start = seq_start + prompt_length
            resp_token_end = seq_start + total_length
            logit_global_start = resp_token_start - 1
            logit_global_end = resp_token_end - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)
            if e <= s:
                logits_chunk = logits[0:0]
                tokens_chunk = tokens[0:0]
            else:
                logits_chunk = logits[s - chunk_start : e - chunk_start]
                tokens_chunk = tokens[(s + 1) - seq_start : (e + 1) - seq_start]
            assert logits_chunk.size(0) == tokens_chunk.size(0), f"{logits_chunk.size(0)} vs {tokens_chunk.size(0)}"
        else:
            # TODO: this is super ugly... do better abstraction.
            chunk_size, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )

            logits_0, logits_1 = logits[end : end + chunk_size], logits[end + chunk_size : end + 2 * chunk_size]
            end += 2 * chunk_size

            logits_0 = logits_0[logits_offset[0][0] - chunks_offset[0][0] : logits_offset[0][1] - chunks_offset[0][0]]
            tokens_0 = tokens[tokens_offset[0][0] : tokens_offset[0][1]]

            logits_1 = logits_1[logits_offset[1][0] - chunks_offset[1][0] : logits_offset[1][1] - chunks_offset[1][0]]
            tokens_1 = tokens[tokens_offset[1][0] : tokens_offset[1][1]]

            assert logits_0.size(0) == tokens_0.size(0), f"{logits_0.size(0)} vs {tokens_0.size(0)}"
            assert logits_1.size(0) == tokens_1.size(0), f"{logits_1.size(0)} vs {tokens_1.size(0)}"

            logits_chunk = torch.cat([logits_0, logits_1], dim=0)
            tokens_chunk = torch.cat([tokens_0, tokens_1], dim=0)

        seq_start += total_length

        yield logits_chunk, tokens_chunk


def _allgather_cp_redistribute(
    res: dict[str, list[torch.Tensor]],
    *,
    logits_local_len: int,
    args: Namespace,
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None = None,
) -> None:
    """Redistribute response tensors from allgather-CP layout to zigzag ring-attn layout.

    After allgather context parallelism, each rank holds a contiguous chunk of
    the global sequence.  This helper reconstructs per-sample full response
    tensors via a differentiable all-reduce and re-slices them into the zigzag
    CP pattern expected by downstream code.

    The *res* dict is modified **in-place**.

    Args:
        res: Dict mapping metric names to lists of per-sample tensors.
        logits_local_len: Local sequence length on this rank.
        args: Configuration (needs ``qkv_format``).
        total_lengths: Total sequence lengths (prompt + response) per sample.
        response_lengths: Response segment lengths per sample.
        max_seq_lens: Optional padded max sequence lengths per sample.
    """
    cp_group = mpu.get_context_parallel_group()
    cp_rank = mpu.get_context_parallel_rank()
    chunk_start = cp_rank * logits_local_len
    chunk_end = chunk_start + logits_local_len

    for key, values in res.items():
        # Skip keys where all values are None (e.g. entropy when not computed)
        if all(v is None for v in values):
            continue

        # Determine reference dtype/device from first non-None value
        ref_value = next(v for v in values if v is not None)
        ref_dtype = ref_value.dtype
        ref_device = ref_value.device

        # Reconstruct full response tensors with each rank's contiguous contribution
        full_resps = []
        seq_start = 0
        for value, total_length, response_length in zip(values, total_lengths, response_lengths, strict=False):
            prompt_length = total_length - response_length
            logit_global_start = seq_start + prompt_length - 1
            logit_global_end = seq_start + total_length - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)

            if value is None or e <= s:
                # This rank has no response logprobs for this sample
                full_resp = torch.zeros(
                    [response_length] + list(ref_value.shape[1:]),
                    dtype=ref_dtype,
                    device=ref_device,
                    requires_grad=True,
                )
            else:
                resp_start = s - logit_global_start
                resp_end = e - logit_global_start
                pad = (0, 0) * (value.dim() - 1) + (resp_start, response_length - resp_end)
                full_resp = F.pad(value, pad)

            assert full_resp.size(0) == response_length, f"Expected {response_length}, got {full_resp.size(0)}"
            full_resps.append(full_resp)
            seq_start += total_length

        # Single differentiable all-reduce to gather full response from all CP ranks
        all_cat = torch.cat(full_resps, dim=0)
        all_cat = dist.nn.all_reduce(all_cat, group=cp_group)

        # Re-slice each sample into zigzag CP pattern
        new_values = []
        for idx, (full_resp, total_length, response_length) in enumerate(
            zip(all_cat.split(response_lengths, dim=0), total_lengths, response_lengths, strict=False)
        ):
            max_seq_len = max_seq_lens[idx] if max_seq_lens is not None else None
            new_values.append(
                slice_log_prob_with_cp(full_resp, total_length, response_length, args.qkv_format, max_seq_len)
            )

        res[key] = new_values


def _build_shifted_tokens(
    T: int,
    device: torch.device,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    qkv_format: str,
    max_seq_lens: list[int] | None,
    allgather_cp: bool,
) -> torch.Tensor:
    """Build shifted target tokens for the full packed/padded logits."""
    cp_size = mpu.get_context_parallel_world_size()

    # --- zigzag CP: completely different layout ---
    if cp_size > 1 and not allgather_cp:
        full_tokens = torch.zeros(T, dtype=torch.long, device=device)
        end = 0
        for i, (tokens, total_length, response_length) in enumerate(
            zip(unconcat_tokens, total_lengths, response_lengths, strict=False)
        ):
            max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
            chunk_size_cp, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )
            for half, base in ((0, end), (1, end + chunk_size_cp)):
                lo = logits_offset[half][0] - chunks_offset[half][0]
                hi = logits_offset[half][1] - chunks_offset[half][0]
                full_tokens[base + lo : base + hi] = tokens[tokens_offset[half][0] : tokens_offset[half][1]]
            end += 2 * chunk_size_cp
        return full_tokens

    # --- cp1 and allgather-CP both build global shifted tokens the same way ---
    T_global = sum(total_lengths) if allgather_cp else T
    full_tokens = torch.zeros(T_global, dtype=torch.long, device=device)

    if qkv_format == "thd" or allgather_cp:
        offset = 0
        for tokens, total_length in zip(unconcat_tokens, total_lengths, strict=False):
            full_tokens[offset : offset + total_length - 1] = tokens[1:total_length]
            offset += total_length
    else:  # bshd, cp1
        for i, (tokens, total_length) in enumerate(zip(unconcat_tokens, total_lengths, strict=False)):
            seq_start = max_seq_lens[i] * i
            full_tokens[seq_start : seq_start + total_length - 1] = tokens[1:total_length]

    # allgather-CP: slice to local chunk
    if allgather_cp:
        cp_rank = mpu.get_context_parallel_rank()
        chunk_start = cp_rank * T
        chunk_end = chunk_start + T
        if chunk_end <= T_global:
            return full_tokens[chunk_start:chunk_end].contiguous()
        local = torch.zeros(T, dtype=torch.long, device=device)
        valid = T_global - chunk_start
        if valid > 0:
            local[:valid] = full_tokens[chunk_start:]
        return local

    return full_tokens


def _extract_per_sample(
    log_prob_full: torch.Tensor,
    entropy_full: torch.Tensor | None,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    qkv_format: str,
    max_seq_lens: list[int] | None,
    allgather_cp: bool,
) -> tuple[list[torch.Tensor], list[torch.Tensor | None]]:
    """Slice per-sample response log-probs/entropy from full-length 1-D tensors."""
    cp_size = mpu.get_context_parallel_world_size()
    log_probs_list: list[torch.Tensor] = []
    entropy_list: list[torch.Tensor | None] = []

    def _append(lp: torch.Tensor) -> None:
        log_probs_list.append(lp)
        entropy_list.append(None)

    def _append_with_entropy(lp: torch.Tensor, start: int, end: int) -> None:
        log_probs_list.append(lp)
        entropy_list.append(entropy_full[start:end] if entropy_full is not None else None)

    if cp_size > 1 and not allgather_cp:
        # zigzag CP
        pos = 0
        for i, (_tokens, total_length, response_length) in enumerate(
            zip(unconcat_tokens, total_lengths, response_lengths, strict=False)
        ):
            max_seq_len = max_seq_lens[i] if max_seq_lens is not None else None
            chunk_size_cp, chunks_offset, logits_offset, _tokens_offset = get_logits_and_tokens_offset_with_cp(
                total_length, response_length, qkv_format, max_seq_len
            )
            lo0 = logits_offset[0][0] - chunks_offset[0][0]
            hi0 = logits_offset[0][1] - chunks_offset[0][0]
            lo1 = logits_offset[1][0] - chunks_offset[1][0]
            hi1 = logits_offset[1][1] - chunks_offset[1][0]

            lp = torch.cat(
                [
                    log_prob_full[pos + lo0 : pos + hi0],
                    log_prob_full[pos + chunk_size_cp + lo1 : pos + chunk_size_cp + hi1],
                ],
                dim=0,
            )
            log_probs_list.append(lp)
            if entropy_full is not None:
                ent = torch.cat(
                    [
                        entropy_full[pos + lo0 : pos + hi0],
                        entropy_full[pos + chunk_size_cp + lo1 : pos + chunk_size_cp + hi1],
                    ],
                    dim=0,
                )
                entropy_list.append(ent)
            else:
                entropy_list.append(None)
            pos += 2 * chunk_size_cp

    elif allgather_cp:
        cp_rank = mpu.get_context_parallel_rank()
        local_len = log_prob_full.size(0)
        chunk_start = cp_rank * local_len
        chunk_end = chunk_start + local_len

        seq_start = 0
        for total_length, response_length in zip(total_lengths, response_lengths, strict=False):
            prompt_length = total_length - response_length
            logit_global_start = seq_start + prompt_length - 1
            logit_global_end = seq_start + total_length - 1

            s = max(logit_global_start, chunk_start)
            e = min(logit_global_end, chunk_end)
            if e <= s:
                _append(log_prob_full[0:0])
            else:
                _append_with_entropy(
                    log_prob_full[s - chunk_start : e - chunk_start], s - chunk_start, e - chunk_start
                )
            seq_start += total_length

    else:
        # cp1
        if qkv_format == "thd":
            offset = 0
            for total_length, response_length in zip(total_lengths, response_lengths, strict=False):
                end = offset + total_length
                start = end - response_length
                _append_with_entropy(log_prob_full[start - 1 : end - 1], start - 1, end - 1)
                offset += total_length
        else:  # bshd
            for i, (total_length, response_length) in enumerate(zip(total_lengths, response_lengths, strict=False)):
                end = max_seq_lens[i] * i + total_length
                start = end - response_length
                _append_with_entropy(log_prob_full[start - 1 : end - 1], start - 1, end - 1)

    return log_probs_list, entropy_list


def get_log_probs_and_entropy(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Compute per-token log-probabilities (and optionally entropy) on responses.

    Computes on the **full** logits ``[T, V]`` tensor at once (instead of
    per-sample slicing) so backward traverses ``[T, V]`` only once, then
    extracts per-sample response portions.

    When ``entropy_coef == 0``, entropy is computed under ``torch.no_grad()``
    to avoid retaining the computation graph and to skip cloning.
    """
    assert non_loss_data
    qkv_format = args.qkv_format

    assert logits.dtype == torch.float32, f"{logits.dtype}"
    assert len(logits.shape) == 3, f"{logits.shape}"

    if qkv_format == "thd":
        assert logits.size(0) == 1, f"{logits.shape}"
        logits = logits.squeeze(0)
    else:
        assert max_seq_lens is not None
        logits = logits.view(-1, logits.size(-1))

    # Apply rollout temperature scaling to logits to match rollout-time log-probs.
    rollout_temperature = getattr(args, "rollout_temperature", 1.0)
    if rollout_temperature != 1.0:
        logits = logits / rollout_temperature
    logits = logits.contiguous()
    T = logits.size(0)
    device = logits.device
    tp_group = mpu.get_tensor_model_parallel_group()
    chunk_size = args.log_probs_chunk_size

    # --- build full shifted-token target tensor ---
    full_tokens = _build_shifted_tokens(
        T, device, unconcat_tokens, total_lengths, response_lengths, qkv_format, max_seq_lens, args.allgather_cp
    )

    # --- compute on full [T,V] logits at once via calculate_log_probs_and_entropy ---
    log_prob_full, entropy_full = calculate_log_probs_and_entropy(
        logits,
        full_tokens,
        tp_group,
        with_entropy=with_entropy,
        chunk_size=chunk_size,
    )
    log_prob_full = log_prob_full.squeeze(-1)  # [T, 1] -> [T]

    # --- extract per-sample response portions ---
    log_probs_list, entropy_list = _extract_per_sample(
        log_prob_full,
        entropy_full,
        unconcat_tokens,
        total_lengths,
        response_lengths,
        qkv_format,
        max_seq_lens,
        args.allgather_cp,
    )

    res = {"log_probs": log_probs_list}
    if with_entropy:
        res["entropy"] = entropy_list

    # we need to turn the all gather kv into zigzag ring attn kv
    if args.allgather_cp:
        _allgather_cp_redistribute(
            res,
            logits_local_len=T,
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return torch.empty((0,), device=device), res


def get_values(
    logits: torch.Tensor,
    *,
    args: Namespace,
    unconcat_tokens: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    with_entropy: bool = False,
    non_loss_data: bool = True,
    max_seq_lens: list[int] | None = None,
) -> dict[str, list[torch.Tensor]]:
    """Extract per-token value predictions over response tokens.

    For each sample, extracts response-aligned chunks from the value head
    output and squeezes the final dimension from `[R, 1]` to `[R]`.

    Args:
        logits: Value head output with shape `[1, T, 1]`.
        args: Configuration (passed to `get_responses` which uses
            `rollout_temperature` even though values don't need temperature).
        unconcat_tokens: List of token tensors per sample.
        total_lengths: Total sequence lengths per sample.
        response_lengths: Response segment lengths per sample.
        with_entropy: Unused; kept for signature compatibility.
        non_loss_data: Unused; kept for signature compatibility.

    Returns:
        Dict with key "values" mapping to a list of `[R]` value tensors
        per sample.
    """
    value_list = []
    for logits_chunk, _ in get_responses(
        logits,
        args=args,
        unconcat_tokens=unconcat_tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        max_seq_lens=max_seq_lens,
    ):
        assert logits_chunk.size(-1) == 1, f"{logits_chunk.shape}"
        value_list.append(logits_chunk.squeeze(-1))

    res = {
        "values": value_list,
    }

    if args.allgather_cp:
        _allgather_cp_redistribute(
            res,
            logits_local_len=logits.size(1),
            args=args,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            max_seq_lens=max_seq_lens,
        )

    return torch.empty((0,), device=logits.device), res


def _compute_opsa_for_rollout(
    args: Namespace,
    rollout_data: RolloutBatch,
    student_log_probs: list[torch.Tensor] | None,
    loss_masks: list[torch.Tensor],
    total_lengths: list[int],
    response_lengths: list[int],
    max_seq_lens: list[int] | None,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Run one OPSA selection over the complete DP-local packed batch."""
    if student_log_probs is None:
        raise ValueError("OPSA requires log-probs recomputed by the current Megatron actor.")

    entropies = rollout_data.get("entropy")
    if args.opsa_mode == "entropy" and entropies is None:
        raise ValueError("OPSA entropy mode requires actor entropy from the log-prob forward pass.")

    cp_size = mpu.get_context_parallel_world_size()
    if cp_size > 1:
        full_log_probs = [
            all_gather_with_cp(log_prob, total_length, response_length)
            for log_prob, total_length, response_length in zip(
                student_log_probs, total_lengths, response_lengths, strict=True
            )
        ]
        full_entropies = None
        if entropies is not None:
            full_entropies = [
                all_gather_with_cp(entropy, total_length, response_length)
                for entropy, total_length, response_length in zip(
                    entropies, total_lengths, response_lengths, strict=True
                )
            ]
    else:
        full_log_probs = student_log_probs
        full_entropies = entropies

    output = compute_opsa(
        full_log_probs,
        loss_masks,
        token_fraction=args.opsa_token_fraction,
        mode=args.opsa_mode,
        entropies=full_entropies,
        advantage_min=args.opsa_advantage_min,
        advantage_max=args.opsa_advantage_max,
        fixed_advantage=args.opsa_fixed_advantage,
    )

    # Keep the full response mask so the standard CP-aware reducer can slice it
    # once, exactly like the original response loss mask.
    rollout_data["opsa_loss_mask"] = output.loss_masks
    rollout_data.update({key: value.detach() for key, value in output.metrics.items()})

    if cp_size == 1:
        advantages = output.advantages
    else:
        advantages = [
            slice_log_prob_with_cp(
                advantage,
                total_length,
                response_length,
                args.qkv_format,
                max_seq_lens[index] if max_seq_lens is not None else None,
            )
            for index, (advantage, total_length, response_length) in enumerate(
                zip(output.advantages, total_lengths, response_lengths, strict=True)
            )
        ]
    return advantages, [advantage.clone() for advantage in advantages]


def compute_advantages_and_returns(args: Namespace, rollout_data: RolloutBatch) -> None:
    """Compute the configured advantages and returns on the pipeline last stage."""
    if not mpu.is_pipeline_last_stage():
        return

    response_lengths: list[int] = rollout_data.get("response_lengths")
    loss_masks: list[torch.Tensor] = rollout_data.get("loss_masks")
    total_lengths: list[int] = rollout_data.get("total_lengths")
    max_seq_lens: list[int] | None = rollout_data.get("max_seq_lens")

    if args.advantage_estimator == "opsa":
        advantages, returns = _compute_opsa_for_rollout(
            args,
            rollout_data,
            rollout_data.get("log_probs"),
            loss_masks,
            total_lengths,
            response_lengths,
            max_seq_lens,
        )
        rollout_data["advantages"] = advantages
        rollout_data["returns"] = returns
        return

    log_probs: list[torch.Tensor] | None = rollout_data.get(
        "rollout_log_probs" if args.use_rollout_logprobs else "log_probs"
    )
    ref_log_probs: list[torch.Tensor] | None = rollout_data.get("ref_log_probs")
    rewards: list[float] = rollout_data.get("rewards")
    values: list[torch.Tensor] | None = rollout_data.get("values")

    if args.kl_coef == 0 or not log_probs:
        xs = log_probs if log_probs is not None else values
        if xs is None:
            raise ValueError("The selected advantage estimator requires actor log-probs or values.")
        kl = [torch.zeros_like(value, dtype=torch.float32, device=value.device) for value in xs]
    else:
        if ref_log_probs is None:
            raise ValueError("A non-zero --kl-coef requires reference log-probs.")
        kl = [
            compute_approx_kl(log_prob, ref_log_prob, kl_loss_type=args.kl_loss_type)
            for log_prob, ref_log_prob in zip(log_probs, ref_log_probs, strict=True)
        ]

    if args.advantage_estimator in {"grpo", "gspo"}:
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_grpo_returns(reward_tensor, kl)
        advantages = list(returns)
    elif args.advantage_estimator == "ppo":
        shaped_rewards = []
        cp_rank = mpu.get_context_parallel_rank()
        for reward, kl_value in zip(rewards, kl, strict=True):
            shaped_reward = kl_value * -args.kl_coef
            if cp_rank == 0 and shaped_reward.numel() > 0:
                shaped_reward[-1] += reward
            shaped_rewards.append(shaped_reward)
        advantages, returns = get_advantages_and_returns_batch(
            total_lengths, response_lengths, values, shaped_rewards, args.gamma, args.lambd
        )
    elif args.advantage_estimator == "reinforce_plus_plus":
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_reinforce_plus_plus_returns(
            rewards=reward_tensor,
            kl=kl,
            loss_masks=loss_masks,
            response_lengths=response_lengths,
            total_lengths=total_lengths,
            kl_coef=args.kl_coef,
            gamma=args.gamma,
        )
        advantages = list(returns)
    elif args.advantage_estimator == "reinforce_plus_plus_baseline":
        reward_tensor = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        advantages = get_reinforce_plus_plus_baseline_advantages(
            rewards=reward_tensor,
            kl=kl,
            loss_masks=loss_masks,
            kl_coef=args.kl_coef,
        )
        returns = advantages
    else:
        raise NotImplementedError(f"advantage_estimator {args.advantage_estimator} is not supported.")

    if args.use_opd:
        advantages, reverse_kls = apply_opd_kl_to_advantages(
            advantages,
            log_probs,
            rollout_data.get("teacher_log_probs"),
            kl_coef=args.opd_kl_coef,
        )
        rollout_data["opd_reverse_kl"] = reverse_kls

    if args.normalize_advantages:
        all_advs = torch.cat(advantages)
        cp_size = mpu.get_context_parallel_world_size()
        if cp_size == 1:
            all_masks = torch.cat(loss_masks)
        else:
            mask_chunks = []
            for index, (total_len, response_len, loss_mask) in enumerate(
                zip(total_lengths, response_lengths, loss_masks, strict=True)
            ):
                prompt_len = total_len - response_len
                max_seq_len = max_seq_lens[index] if max_seq_lens is not None else None
                _, _, _, token_offsets = get_logits_and_tokens_offset_with_cp(
                    total_len, response_len, args.qkv_format, max_seq_len
                )
                local_parts = []
                for start, end in token_offsets:
                    response_start = max(0, start - prompt_len)
                    response_end = max(0, end - prompt_len)
                    if response_end > response_start:
                        local_parts.append(loss_mask[response_start:response_end])
                mask_chunks.append(
                    torch.cat(local_parts)
                    if local_parts
                    else torch.empty(0, device=all_advs.device, dtype=loss_mask.dtype)
                )
            all_masks = torch.cat(mask_chunks)

        if all_masks.numel() > 0:
            if all_advs.shape != all_masks.shape:
                raise ValueError(
                    f"Advantage/mask shape mismatch before whitening: {all_advs.shape} vs {all_masks.shape}."
                )
            whitened = distributed_masked_whiten(
                all_advs,
                all_masks,
                process_group=mpu.get_data_parallel_group(),
                shift_mean=True,
            )
            advantages = list(torch.split(whitened, [value.numel() for value in advantages]))

    rollout_data["advantages"] = advantages
    rollout_data["returns"] = returns


def vanilla_tis_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    tis = torch.exp(old_log_probs - rollout_log_probs)
    tis_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
    tis_clipfrac = (tis_weights != tis).float()
    metrics = {
        "tis": tis.clone().detach(),
        "tis_clipfrac": tis_clipfrac.clone().detach(),
        "tis_abs": tis_abs.clone().detach(),
    }
    pg_loss = pg_loss * tis_weights
    return pg_loss, loss_masks, metrics


def icepop_function(
    args,
    *,
    pg_loss: torch.Tensor,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    **kwargs: Any,
) -> tuple[torch.Tensor, list[torch.Tensor], dict[str, torch.Tensor]]:
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    ice_ratio = torch.exp(old_log_probs - rollout_log_probs)
    ice_abs = (torch.exp(old_log_probs - rollout_log_probs) - 1).abs()
    ice_weight = torch.where(
        (ice_ratio >= args.tis_clip_low) & (ice_ratio <= args.tis_clip), ice_ratio, torch.zeros_like(ice_ratio)
    )
    ice_clipfrac = (ice_weight != ice_ratio).float()
    metrics = {
        "tis": ice_ratio.clone().detach(),
        "tis_clipfrac": ice_clipfrac.clone().detach(),
        "tis_abs": ice_abs.clone().detach(),
    }
    pg_loss = pg_loss * ice_weight
    return pg_loss, loss_masks, metrics


def policy_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute policy loss, using opsa_loss_mask as the OPSA denominator mask."""
    advantages = torch.cat(batch["advantages"], dim=0)
    old_log_prob_chunks = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]
    opsa_loss_mask = batch.get("opsa_loss_mask")
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens")

    # Standard Slime keeps entropy metrics for its original estimators. OPSA
    # fixed mode never computes entropy; validation also requires entropy_coef=0.
    with_entropy = args.advantage_estimator != "opsa" or args.opsa_mode == "entropy"
    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=with_entropy,
        max_seq_lens=max_seq_lens,
    )
    log_prob_chunks = log_probs_and_entropy["log_probs"]

    need_full_log_probs = args.use_opsm or args.advantage_estimator == "gspo"
    full_log_probs = None
    full_old_log_probs = None
    if need_full_log_probs:
        full_log_probs = [
            all_gather_with_cp(value, total_length, response_length)
            for value, total_length, response_length in zip(
                log_prob_chunks, total_lengths, response_lengths, strict=True
            )
        ]
        full_old_log_probs = [
            all_gather_with_cp(value, total_length, response_length)
            for value, total_length, response_length in zip(
                old_log_prob_chunks, total_lengths, response_lengths, strict=True
            )
        ]

    if args.use_opsm:
        opsm_mask, opsm_clipfrac = compute_opsm_mask(
            args=args,
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            advantages=batch["advantages"],
            loss_masks=batch["loss_masks"],
        )

    if args.advantage_estimator == "gspo":
        ppo_kl = compute_gspo_kl(
            full_log_probs=full_log_probs,
            full_old_log_probs=full_old_log_probs,
            local_log_probs=log_prob_chunks,
            loss_masks=batch["loss_masks"],
        )
    else:
        ppo_kl = torch.cat(old_log_prob_chunks, dim=0) - torch.cat(log_prob_chunks, dim=0)

    old_log_probs = torch.cat(old_log_prob_chunks, dim=0)
    log_probs = torch.cat(log_prob_chunks, dim=0)
    pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, args.eps_clip, args.eps_clip_high)
    if args.use_opsm:
        pg_loss = pg_loss * opsm_mask

    modified_response_masks = batch["loss_masks"]
    tis_metrics = {}
    if args.get_mismatch_metrics or args.use_tis:
        sum_of_sample_mean_for_mismatch_metrics = sum_of_sample_mean
        if "rollout_log_probs" not in batch:
            raise ValueError("rollout_log_probs must be provided for TIS.")
        ois = (-ppo_kl).exp()
        tis_kwargs = {
            "args": args,
            "pg_loss": pg_loss,
            "train_log_probs": batch["log_probs"],
            "rollout_log_probs": batch["rollout_log_probs"],
            "loss_masks": batch["loss_masks"],
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
        }
        tis_func = (
            load_function(args.custom_tis_function_path)
            if args.custom_tis_function_path is not None
            else vanilla_tis_function
        )
        pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)
        sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            modified_response_masks,
            args.calculate_per_token_loss,
            args.qkv_format,
            max_seq_lens,
        )

    pg_loss_masks = modified_response_masks
    filtered_sum_of_sample_mean = None
    if opsa_loss_mask is not None:
        pg_loss_masks = [
            response_mask.to(device=opsa_mask.device, dtype=opsa_mask.dtype) * opsa_mask
            for response_mask, opsa_mask in zip(modified_response_masks, opsa_loss_mask, strict=True)
        ]
        filtered_sum_of_sample_mean = get_sum_of_sample_mean(
            total_lengths,
            response_lengths,
            pg_loss_masks,
            args.calculate_per_token_loss,
            args.qkv_format,
            max_seq_lens,
        )

    if getattr(args, "custom_pg_loss_reducer_function_path", None) is not None:
        pg_loss_reducer = load_function(args.custom_pg_loss_reducer_function_path)(
            total_lengths, response_lengths, pg_loss_masks, args.calculate_per_token_loss
        )
    else:
        pg_loss_reducer = filtered_sum_of_sample_mean or sum_of_sample_mean

    pg_loss = pg_loss_reducer(pg_loss)
    pg_clipfrac = pg_loss_reducer(pg_clipfrac) if filtered_sum_of_sample_mean else sum_of_sample_mean(pg_clipfrac)
    ppo_kl = sum_of_sample_mean(ppo_kl)

    if with_entropy:
        entropy = torch.cat(log_probs_and_entropy["entropy"], dim=0)
        entropy_loss = sum_of_sample_mean(entropy)
    else:
        entropy_loss = logits.new_zeros(())
    loss = pg_loss - args.entropy_coef * entropy_loss

    if args.use_kl_loss:
        ref_log_probs = torch.cat(batch["ref_log_probs"], dim=0)
        importance_ratio = torch.exp(log_probs - old_log_probs) if args.use_unbiased_kl else None
        kl = compute_approx_kl(
            log_probs,
            ref_log_probs,
            kl_loss_type=args.kl_loss_type,
            importance_ratio=importance_ratio,
        )
        kl_loss = sum_of_sample_mean(kl)
        loss = loss + args.kl_loss_coef * kl_loss

    if log_probs.numel() == 0:
        loss = loss + 0 * logits.sum()

    reported_loss = {
        "loss": loss.clone().detach(),
        "pg_loss": pg_loss.clone().detach(),
        "entropy_loss": entropy_loss.clone().detach(),
        "pg_clipfrac": pg_clipfrac.clone().detach(),
        "ppo_kl": ppo_kl.clone().detach(),
    }
    if "rollout_log_probs" in batch and batch["rollout_log_probs"]:
        rollout_log_probs = torch.cat(batch["rollout_log_probs"], dim=0)
        reported_loss["train_rollout_logprob_abs_diff"] = (
            sum_of_sample_mean((old_log_probs - rollout_log_probs).abs()).clone().detach()
        )
    if args.use_kl_loss:
        reported_loss["kl_loss"] = kl_loss.clone().detach()
    if opsa_loss_mask is not None:
        selected = (advantages != 0).to(dtype=advantages.dtype)
        reported_loss["opsa/selected_fraction"] = sum_of_sample_mean(selected).clone().detach()
        reported_loss["opsa/advantage_mean"] = pg_loss_reducer(advantages).clone().detach()
        reported_loss["opsa/negative_fraction"] = (
            pg_loss_reducer((advantages < 0).to(dtype=advantages.dtype)).clone().detach()
        )
        reported_loss["opsa/positive_fraction"] = (
            pg_loss_reducer((advantages > 0).to(dtype=advantages.dtype)).clone().detach()
        )
    if args.get_mismatch_metrics or args.use_tis:
        reported_loss["ois"] = sum_of_sample_mean_for_mismatch_metrics(ois).clone().detach()
        for key, value in tis_metrics.items():
            reported_loss[key] = sum_of_sample_mean_for_mismatch_metrics(value)
    if args.use_opsm:
        reported_loss["opsm_clipfrac"] = opsm_clipfrac
    if "opd_reverse_kl" in batch:
        reverse_kl = torch.cat(batch["opd_reverse_kl"], dim=0)
        reported_loss["opd_reverse_kl"] = sum_of_sample_mean(reverse_kl).clone().detach()
    return loss, reported_loss


def value_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute clipped value loss and metrics.

    Extracts current value predictions from `logits`, compares them against
    stored old values with clipping, and computes the maximum of clipped and
    unclipped squared errors (PPO-style value clipping).

    Args:
        args: Configuration containing `value_clip` threshold.
        batch: Mini-batch with "values" (old predictions), "returns",
            "unconcat_tokens", "total_lengths", and "response_lengths".
        logits: Value head output with shape `[1, T, 1]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `loss` is a scalar tensor and
        `metrics` contains detached scalars "value_loss" and "value_clipfrac".
    """
    old_values = torch.cat(batch["values"], dim=0)

    _, values = get_values(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=batch["total_lengths"],
        response_lengths=batch["response_lengths"],
        max_seq_lens=batch.get("max_seq_lens", None),
    )
    values = torch.cat([value.flatten() for value in values["values"]], dim=0)

    returns = torch.cat(batch["returns"], dim=0)

    values_clipfrac = torch.abs(values - old_values) > args.value_clip
    values_clipped = old_values + (values - old_values).clamp(-args.value_clip, args.value_clip)
    surr1 = (values_clipped - returns) ** 2
    surr2 = (values - returns) ** 2
    loss = torch.max(surr1, surr2)

    loss = sum_of_sample_mean(loss)
    values_clipfrac = sum_of_sample_mean(values_clipfrac.float())

    # make sure the gradient could backprop correctly.
    if values.numel() == 0:
        loss += 0 * values.sum()

    reported_loss = {
        "value_loss": loss.clone().detach(),
        "value_clipfrac": values_clipfrac.clone().detach(),
    }

    return loss, reported_loss


def sft_loss_function(
    args: Namespace,
    batch: RolloutBatch,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute supervised fine-tuning loss over response tokens.

    Computes log-probabilities of the ground-truth tokens in the response
    segments and returns the negative log-likelihood as the loss.

    Args:
        args: Configuration (passed through to helpers).
        batch: Mini-batch with "unconcat_tokens", "response_lengths", and
            "total_lengths".
        logits: Policy logits with shape `[1, T, V]`.
        sum_of_sample_mean: Reduction function that averages per-sample values.

    Returns:
        Tuple of `(loss, metrics)` where `metrics` contains a single detached
        scalar "loss".
    """
    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]

    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=False,
        max_seq_lens=batch.get("max_seq_lens", None),
    )

    log_probs = log_probs_and_entropy["log_probs"]
    log_probs = torch.cat(log_probs, dim=0)
    loss = -sum_of_sample_mean(log_probs)

    # make sure the gradient could backprop correctly.
    if log_probs.numel() == 0:
        loss += 0 * logits.sum()

    return (
        loss,
        {
            "loss": loss.clone().detach(),
        },
    )


def loss_function(
    args: Namespace,
    batch: RolloutBatch,
    num_microbatches: int,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, int | torch.Tensor, dict[str, list[str] | torch.Tensor]]:
    """Dispatch to the configured loss and rescale for Megatron integration.

    Selects one of "policy_loss", "value_loss", "sft_loss", or a custom loss
    function based on `args.loss_type`, computes the loss and metrics, then
    rescales the loss by micro-batch and parallelism factors to integrate with
    Megatron's gradient accumulation.

    Args:
        args: Configuration specifying `loss_type`, `calculate_per_token_loss`,
            `global_batch_size`, and optionally `custom_loss_function_path`.
        batch: Mini-batch with "loss_masks", "response_lengths", and other
            keys required by the selected loss function.
        num_microbatches: Number of gradient accumulation steps.
        logits: Model outputs (policy or value head).

    Returns:
        Tuple of `(scaled_loss, normalizer, logging_dict)` where:
        - `scaled_loss` is the loss tensor (scalar) rescaled for Megatron.
        - `normalizer` is `num_tokens` (scalar tensor) if
          `args.calculate_per_token_loss` is True, else `1` (int).
        - `logging_dict` has keys "keys" (list of str metric names) and
          "values" (1D tensor: [count, metric1, metric2, ...]).
    """
    opsa_loss_mask = batch.get("opsa_loss_mask")
    normalizer_masks = opsa_loss_mask if opsa_loss_mask is not None else batch["loss_masks"]
    if opsa_loss_mask is not None:
        num_tokens = torch.clamp_min(sum(mask.sum() for mask in normalizer_masks), 1)
    else:
        num_tokens = sum(torch.clamp_min(mask.sum(), 1) for mask in normalizer_masks)
    num_samples = len(batch["response_lengths"])

    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        args.calculate_per_token_loss,
        args.qkv_format,
        batch.get("max_seq_lens", None),
    )

    match args.loss_type:
        case "policy_loss":
            func = policy_loss_function
        case "value_loss":
            func = value_loss_function
        case "sft_loss":
            func = sft_loss_function
        case "custom_loss":
            func = load_function(args.custom_loss_function_path)
        case _:
            raise ValueError(f"Unknown loss type: {args.loss_type}")

    if args.recompute_loss_function:
        loss, log = checkpoint(func, args, batch, logits, sum_of_sample_mean, use_reentrant=False)
    else:
        loss, log = func(args, batch, logits, sum_of_sample_mean)

    # With allgather-CP, some CP ranks may have no loss-contributing tokens (e.g., all
    # padding). Without this, gradient doesn't flow through their attention path, so
    # the CP gather's backward (reduce-scatter) is not called, deadlocking other CP
    # ranks that call it. Adding this zero loss forces autograd to traverse the full
    # graph on every rank without changing gradient values.
    if args.allgather_cp and mpu.get_context_parallel_world_size() > 1:
        loss = loss + 0 * logits.sum()

    # Here we need to divide by cp_size because to cancel the multiply in Megatron.
    global_batch_size = batch.get("dynamic_global_batch_size", args.global_batch_size)
    if not args.calculate_per_token_loss:
        loss = (
            loss * num_microbatches / global_batch_size * mpu.get_data_parallel_world_size(with_context_parallel=True)
        )
    else:
        loss = loss * mpu.get_context_parallel_world_size()

    return (
        loss,
        (num_tokens if args.calculate_per_token_loss else torch.tensor(1, device=logits.device)),
        {
            "keys": list(log.keys()),
            "values": torch.tensor(
                [
                    num_samples if not args.calculate_per_token_loss else num_tokens,
                ]
                + list(log.values()),
                device=logits.device,
            ),
        },
    )
