from types import SimpleNamespace

import pytest
import torch

from slime.backends.megatron_utils.opsa import compute_opsa, requires_reference_model, validate_opsa_args


def _fixed(log_probs, loss_masks, fraction=0.2, value=-0.5):
    return compute_opsa(
        log_probs,
        loss_masks,
        token_fraction=fraction,
        mode="fixed",
        fixed_advantage=value,
    )


def _args(**overrides):
    values = {
        "advantage_estimator": "opsa",
        "use_opd": False,
        "normalize_advantages": False,
        "use_rollout_logprobs": False,
        "use_rollout_entropy": False,
        "kl_coef": 0.0,
        "use_kl_loss": False,
        "kl_loss_coef": 0.0,
        "entropy_coef": 0.0,
        "opsa_token_fraction": 0.2,
        "opsa_advantage_min": -1.0,
        "opsa_advantage_max": -0.5,
        "opsa_mode": "entropy",
        "opsa_fixed_advantage": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_entropy_opsa_selects_lowest_valid_tokens_across_packed_samples():
    output = compute_opsa(
        [torch.tensor([-10.0, -9.0, -8.0]), torch.tensor([-7.0, -6.0])],
        [torch.tensor([1, 0, 1]), torch.tensor([1, 1])],
        token_fraction=0.5,
        mode="entropy",
        entropies=[torch.tensor([2.0, 99.0, 4.0]), torch.tensor([3.0, 1.0])],
        advantage_min=-1.0,
        advantage_max=-0.5,
    )

    assert torch.equal(output.loss_masks[0], torch.tensor([1, 0, 1]))
    assert torch.equal(output.loss_masks[1], torch.tensor([0, 0]))
    assert torch.allclose(output.advantages[0], torch.tensor([-0.5, 0.0, -1.0]))
    assert torch.equal(output.advantages[1], torch.zeros(2))
    assert output.metrics["opsa/selected_fraction"].item() == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("fraction", "expected_count"),
    [(0.1, 1), (0.2, 2), (0.3, 3), (0.4, 4)],
)
def test_lowest_fraction_ablations(fraction, expected_count):
    output = _fixed(
        [torch.arange(10, dtype=torch.float32)],
        [torch.ones(10)],
        fraction=fraction,
    )
    assert int(output.loss_masks[0].sum().item()) == expected_count


def test_nonempty_batch_always_selects_at_least_one_token():
    output = _fixed([torch.tensor([-1.0, -0.5])], [torch.ones(2)], fraction=0.1)
    assert output.loss_masks[0].sum().item() == 1


def test_equal_selected_entropies_receive_advantage_min():
    output = compute_opsa(
        [torch.tensor([-4.0, -3.0, -2.0, -1.0])],
        [torch.ones(4)],
        token_fraction=0.5,
        mode="entropy",
        entropies=[torch.full((4,), 2.0)],
        advantage_min=-1.0,
        advantage_max=-0.5,
    )
    assert torch.equal(output.advantages[0], torch.tensor([-1.0, -1.0, 0.0, 0.0]))


@pytest.mark.parametrize("value", [-0.5, 0.2])
def test_fixed_negative_and_positive_advantages(value):
    output = _fixed(
        [torch.tensor([-3.0, -2.0, -1.0])],
        [torch.ones(3)],
        fraction=2 / 3,
        value=value,
    )
    assert torch.allclose(output.advantages[0], torch.tensor([value, value, 0.0]))


def test_empty_and_all_masked_batches_select_nothing():
    empty = _fixed([], [], fraction=0.2)
    assert empty.advantages == []
    assert empty.loss_masks == []
    assert empty.metrics["opsa/selected_tokens"].item() == 0

    masked = _fixed([torch.tensor([-3.0, -2.0])], [torch.zeros(2)], fraction=0.2)
    assert torch.equal(masked.advantages[0], torch.zeros(2))
    assert torch.equal(masked.loss_masks[0], torch.zeros(2))


def test_selection_is_dp_local_not_global():
    # A global selection would choose both rank-0 tokens. Independent calls
    # still select one token on each simulated DP rank.
    rank0 = _fixed([torch.tensor([-100.0, -90.0])], [torch.ones(2)], fraction=0.5)
    rank1 = _fixed([torch.tensor([-2.0, -1.0])], [torch.ones(2)], fraction=0.5)
    assert rank0.loss_masks[0].sum().item() == 1
    assert rank1.loss_masks[0].sum().item() == 1
    assert rank1.loss_masks[0][0].item() == 1


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError, match="shape mismatch"):
        _fixed([torch.ones(2)], [torch.ones(3)])


def test_opsa_startup_never_allocates_a_reference_model():
    args = _args(kl_loss_coef=1.0)
    validate_opsa_args(args)
    assert requires_reference_model(args) is False


def test_reference_model_follows_enabled_kl_features_for_other_estimators():
    assert requires_reference_model(_args(advantage_estimator="grpo", kl_coef=0.1)) is True
    assert requires_reference_model(_args(advantage_estimator="grpo", use_kl_loss=True)) is True
    assert requires_reference_model(_args(advantage_estimator="grpo", kl_loss_coef=1.0)) is False


def test_entropy_mode_enables_actor_entropy():
    args = _args()
    validate_opsa_args(args)
    assert args.use_rollout_entropy is True


@pytest.mark.parametrize("value", [-0.5, 0.2])
def test_fixed_mode_requires_no_entropy(value):
    args = _args(opsa_mode="fixed", opsa_fixed_advantage=value)
    validate_opsa_args(args)
    assert args.use_rollout_entropy is False


@pytest.mark.parametrize(
    "overrides",
    [
        {"use_opd": True},
        {"normalize_advantages": True},
        {"use_rollout_logprobs": True},
        {"kl_coef": 0.1},
        {"use_kl_loss": True},
        {"opsa_token_fraction": 0.0},
        {"opsa_token_fraction": 1.1},
        {"opsa_advantage_min": -0.5, "opsa_advantage_max": -1.0},
        {"opsa_mode": "fixed", "opsa_fixed_advantage": None},
        {"opsa_mode": "fixed", "opsa_fixed_advantage": 0.0},
        {"opsa_mode": "fixed", "opsa_fixed_advantage": -0.5, "use_rollout_entropy": True},
        {"opsa_mode": "fixed", "opsa_fixed_advantage": -0.5, "entropy_coef": 0.01},
    ],
)
def test_invalid_opsa_parameter_combinations(overrides):
    with pytest.raises(ValueError):
        validate_opsa_args(_args(**overrides))
