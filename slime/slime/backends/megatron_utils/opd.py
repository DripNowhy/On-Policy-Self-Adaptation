"""Pure PyTorch sampled-token on-policy distillation utilities."""

import torch


def apply_opd_kl_to_advantages(
    advantages: list[torch.Tensor],
    actor_log_probs: list[torch.Tensor] | None,
    teacher_log_probs: list[torch.Tensor] | None,
    *,
    kl_coef: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Apply sampled-token reverse KL to base advantages.

    Returns a new advantage list and the per-token reverse-KL tensors. Inputs
    are not modified.
    """

    if actor_log_probs is None:
        raise ValueError("Standard OPD requires actor log-probs.")
    if teacher_log_probs is None:
        raise ValueError("Standard OPD requires teacher log-probs.")
    if not (len(advantages) == len(actor_log_probs) == len(teacher_log_probs)):
        raise ValueError("OPD advantages, actor log-probs, and teacher log-probs must have equal sample counts.")

    updated_advantages = []
    reverse_kls = []
    for index, (advantage, actor_log_prob, teacher_log_prob) in enumerate(
        zip(advantages, actor_log_probs, teacher_log_probs, strict=True)
    ):
        teacher_log_prob = teacher_log_prob.to(device=actor_log_prob.device, dtype=actor_log_prob.dtype)
        if actor_log_prob.shape != teacher_log_prob.shape or advantage.shape != actor_log_prob.shape:
            raise ValueError(
                f"OPD tensor shape mismatch for sample {index}: advantage={tuple(advantage.shape)}, actor={tuple(actor_log_prob.shape)}, teacher={tuple(teacher_log_prob.shape)}."
            )
        reverse_kl = actor_log_prob - teacher_log_prob
        updated_advantages.append(advantage - kl_coef * reverse_kl)
        reverse_kls.append(reverse_kl)

    return updated_advantages, reverse_kls
