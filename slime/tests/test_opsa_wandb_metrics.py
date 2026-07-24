from types import SimpleNamespace

import slime.utils.logging_utils as logging_utils


def _args(**overrides):
    values = {
        "advantage_estimator": "opsa",
        "use_wandb": True,
        "use_tensorboard": False,
        "wandb_log_all_metrics": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _capture_wandb(monkeypatch):
    logged = []
    monkeypatch.setattr(logging_utils.wandb, "log", lambda metrics: logged.append(metrics))
    return logged


def test_compact_opsa_train_metrics_follow_allowlist(monkeypatch):
    logged = _capture_wandb(monkeypatch)
    metrics = {
        "train/step": 7,
        "train/loss": 1.25,
        "train/entropy_loss": 0.42,
        "train/grad_norm": 3.0,
        "train/lr-pg_0": 1e-6,
        "train/lr-pg_1": 2e-6,
        "train/pg_loss": 1.1,
        "train/opsa/selected_fraction": 0.2,
        "train/opsa/advantage_mean": -0.75,
    }

    logging_utils.log(_args(), metrics, step_key="train/step")

    assert logged == [
        {
            "train/loss": 1.25,
            "train/entropy_loss": 0.42,
            "train/grad_norm": 3.0,
            "train/lr-pg_0": 1e-6,
            "train/lr-pg_1": 2e-6,
            "train/step": 7,
        }
    ]
    assert metrics["train/loss"] == 1.25


def test_compact_opsa_keeps_rollout_and_eval_response_min_mean_max(monkeypatch):
    logged = _capture_wandb(monkeypatch)

    logging_utils.log(
        _args(),
        {
            "rollout/step": 3,
            "rollout/response_len/min": 10,
            "rollout/response_len/mean": 20,
            "rollout/response_len/median": 18,
            "rollout/response_len/max": 40,
            "rollout/opsa/selected_fraction": 0.2,
            "rollout/opsa/advantage_mean": -0.75,
            "rollout/opsa/selected_tokens": 128,
            "rollout/log_probs": -1.5,
            "rollout/ref_log_probs": -1.4,
            "rollout/entropy": 0.6,
            "rollout/truncated_ratio": 0.1,
            "rollout/repetition_frac": 0.05,
            "perf/rollout_time": 12.0,
            "perf/actor_train_tok_per_s": 1234.0,
            "perf/tokens_per_gpu_per_sec": 999.0,
        },
        step_key="rollout/step",
    )
    logging_utils.log(
        _args(),
        {
            "eval/step": 6,
            "eval/aime/response_len/min": 100,
            "eval/aime/response_len/mean": 200,
            "eval/aime/response_len/median": 180,
            "eval/aime/response_len/max": 400,
            "eval/aime": 0.5,
            "eval/aime-pass@1": 0.4,
            "eval/aime-pass@2": 0.5,
            "eval/aime-pass@4": 0.6,
        },
        step_key="eval/step",
    )

    assert logged == [
        {
            "rollout/response_len/min": 10,
            "rollout/response_len/mean": 20,
            "rollout/response_len/max": 40,
            "rollout/opsa/selected_fraction": 0.2,
            "rollout/opsa/advantage_mean": -0.75,
            "rollout/log_probs": -1.5,
            "rollout/entropy": 0.6,
            "rollout/truncated_ratio": 0.1,
            "rollout/repetition_frac": 0.05,
            "perf/rollout_time": 12.0,
            "perf/actor_train_tok_per_s": 1234.0,
            "rollout/step": 3,
        },
        {
            "eval/aime/response_len/min": 100,
            "eval/aime/response_len/mean": 200,
            "eval/aime/response_len/max": 400,
            "eval/aime-pass@1": 0.4,
            "eval/aime-pass@4": 0.6,
            "eval/step": 6,
        },
    ]


def test_opsa_all_metrics_opt_in_bypasses_filter(monkeypatch):
    logged = _capture_wandb(monkeypatch)
    metrics = {
        "train/step": 1,
        "train/loss": 2.0,
        "train/entropy_loss": 0.5,
        "train/pg_loss": 1.8,
        "train/opsa/selected_fraction": 0.2,
    }

    logging_utils.log(_args(wandb_log_all_metrics=True), metrics, step_key="train/step")

    assert logged == [metrics]


def test_non_opsa_metrics_are_unchanged(monkeypatch):
    logged = _capture_wandb(monkeypatch)
    metrics = {"rollout/step": 2, "perf/rollout_time": 3.0, "rollout/reward": 0.5}

    logging_utils.log(_args(advantage_estimator="grpo"), metrics, step_key="rollout/step")

    assert logged == [metrics]


def test_compact_opsa_skips_wandb_when_only_step_remains(monkeypatch):
    logged = _capture_wandb(monkeypatch)

    logging_utils.log(
        _args(),
        {"rollout/step": 8, "passrate/pass@1": 0.75},
        step_key="rollout/step",
    )

    assert logged == []


def test_tensorboard_receives_complete_metrics(monkeypatch):
    logged = _capture_wandb(monkeypatch)
    tensorboard_calls = []

    class FakeTensorboardAdapter:
        def __init__(self, args):
            self.args = args

        def log(self, data, step):
            tensorboard_calls.append((data, step))

    monkeypatch.setattr(logging_utils, "_TensorboardAdapter", FakeTensorboardAdapter)
    metrics = {
        "train/step": 9,
        "train/loss": 1.0,
        "train/entropy_loss": 0.25,
        "train/pg_loss": 0.8,
        "train/opsa/selected_fraction": 0.2,
    }

    logging_utils.log(_args(use_tensorboard=True), metrics, step_key="train/step")

    assert logged == [{"train/loss": 1.0, "train/entropy_loss": 0.25, "train/step": 9}]
    assert tensorboard_calls == [
        (
            {
                "train/loss": 1.0,
                "train/entropy_loss": 0.25,
                "train/pg_loss": 0.8,
                "train/opsa/selected_fraction": 0.2,
            },
            9,
        )
    ]
