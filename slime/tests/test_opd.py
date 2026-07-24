import pytest
import torch

from slime.backends.megatron_utils.opd import apply_opd_kl_to_advantages


def test_sampled_token_opd_reverse_kl_updates_base_advantages():
    advantages, reverse_kls = apply_opd_kl_to_advantages(
        [torch.tensor([1.0, 2.0])],
        [torch.tensor([-1.0, -3.0])],
        [torch.tensor([-2.0, -1.0])],
        kl_coef=0.5,
    )

    assert torch.allclose(reverse_kls[0], torch.tensor([1.0, -2.0]))
    assert torch.allclose(advantages[0], torch.tensor([0.5, 3.0]))


def test_opd_rejects_sample_count_mismatch():
    with pytest.raises(ValueError, match="equal sample counts"):
        apply_opd_kl_to_advantages(
            [torch.zeros(1)],
            [torch.zeros(1), torch.zeros(1)],
            [torch.zeros(1)],
            kl_coef=1.0,
        )


@pytest.mark.parametrize(
    ("advantages", "actor", "teacher"),
    [
        ([torch.zeros(2)], [torch.zeros(1)], [torch.zeros(1)]),
        ([torch.zeros(1)], [torch.zeros(1)], [torch.zeros(2)]),
    ],
)
def test_opd_rejects_tensor_shape_mismatch(advantages, actor, teacher):
    with pytest.raises(ValueError, match="shape mismatch"):
        apply_opd_kl_to_advantages(advantages, actor, teacher, kl_coef=1.0)


@pytest.mark.parametrize(
    ("actor", "teacher", "message"),
    [
        (None, [torch.zeros(1)], "actor log-probs"),
        ([torch.zeros(1)], None, "teacher log-probs"),
    ],
)
def test_opd_rejects_missing_log_probs(actor, teacher, message):
    with pytest.raises(ValueError, match=message):
        apply_opd_kl_to_advantages([torch.zeros(1)], actor, teacher, kl_coef=1.0)


def test_opd_accepts_zero_length_response():
    advantages, reverse_kls = apply_opd_kl_to_advantages(
        [torch.empty(0)],
        [torch.empty(0)],
        [torch.empty(0)],
        kl_coef=1.0,
    )
    assert advantages[0].numel() == 0
    assert reverse_kls[0].numel() == 0
