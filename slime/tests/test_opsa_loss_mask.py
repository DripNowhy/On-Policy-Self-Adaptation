from types import SimpleNamespace

import pytest
import torch

from slime.backends.megatron_utils import loss as loss_module


def _policy_args(advantage_estimator: str, *, opsa_mode: str = "fixed") -> SimpleNamespace:
    return SimpleNamespace(
        advantage_estimator=advantage_estimator,
        allgather_cp=False,
        calculate_per_token_loss=False,
        custom_pg_loss_reducer_function_path=None,
        entropy_coef=0.0,
        eps_clip=0.2,
        eps_clip_high=0.2,
        get_mismatch_metrics=False,
        global_batch_size=1,
        loss_type="policy_loss",
        opsa_mode=opsa_mode,
        qkv_format="thd",
        recompute_loss_function=False,
        use_kl_loss=False,
        use_opsm=False,
        use_rollout_logprobs=False,
        use_tis=False,
    )


def _patch_cpu_policy_dependencies(monkeypatch, token_losses: torch.Tensor) -> None:
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        loss_module.mpu,
        "get_data_parallel_world_size",
        lambda **_kwargs: 1,
    )

    def fake_log_probs_and_entropy(logits, *, with_entropy, **_kwargs):
        values = {"log_probs": [torch.zeros_like(token_losses)]}
        if with_entropy:
            values["entropy"] = [torch.zeros_like(token_losses)]
        return logits.new_empty((0,)), values

    monkeypatch.setattr(loss_module, "get_log_probs_and_entropy", fake_log_probs_and_entropy)
    monkeypatch.setattr(
        loss_module,
        "compute_policy_loss",
        lambda *_args, **_kwargs: (token_losses, torch.zeros_like(token_losses)),
    )


def test_none_opsa_loss_mask_falls_back_to_standard_response_mask(monkeypatch):
    token_losses = torch.tensor([2.0, 100.0])
    _patch_cpu_policy_dependencies(monkeypatch, token_losses)
    batch = {
        "advantages": [torch.ones(2)],
        "log_probs": [torch.zeros(2)],
        "loss_masks": [torch.tensor([1.0, 0.0])],
        "opsa_loss_mask": None,
        "response_lengths": [2],
        "total_lengths": [3],
        "unconcat_tokens": [torch.tensor([10, 11, 12])],
    }

    loss, _, _ = loss_module.loss_function(
        _policy_args("grpo"),
        batch,
        num_microbatches=1,
        logits=torch.zeros(1, 3, 4),
    )

    assert loss.item() == pytest.approx(2.0)


def test_opsa_loss_mask_controls_policy_numerator_and_denominator(monkeypatch):
    token_losses = torch.tensor([2.0, 100.0, 4.0])
    _patch_cpu_policy_dependencies(monkeypatch, token_losses)
    batch = {
        "advantages": [torch.tensor([1.0, 0.0, 1.0])],
        "log_probs": [torch.zeros(3)],
        "loss_masks": [torch.ones(3)],
        "opsa_loss_mask": [torch.tensor([1.0, 0.0, 1.0])],
        "response_lengths": [3],
        "total_lengths": [4],
        "unconcat_tokens": [torch.tensor([10, 11, 12, 13])],
    }

    loss, _, log = loss_module.loss_function(
        _policy_args("opsa"),
        batch,
        num_microbatches=1,
        logits=torch.zeros(1, 4, 4),
    )

    # The masked token (100) contributes neither to the numerator nor to the
    # denominator: (2 + 4) / 2 = 3, rather than 106 / 3 or 6 / 3.
    assert loss.item() == pytest.approx(3.0)
    metric_index = log["keys"].index("pg_loss") + 1
    assert log["values"][metric_index].item() == pytest.approx(3.0)
