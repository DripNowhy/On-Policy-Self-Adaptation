from types import SimpleNamespace

import slime.utils.wandb_utils as wandb_utils


def _args(**overrides):
    values = {
        "advantage_estimator": "opsa",
        "opsa_mode": "entropy",
        "opsa_token_fraction": 0.2,
        "opsa_advantage_min": -1.0,
        "opsa_advantage_max": -0.5,
        "opsa_fixed_advantage": None,
        "model_name": "qwen3",
        "num_layers": 28,
        "hidden_size": 2048,
        "train_backend": "megatron",
        "global_batch_size": 64,
        "micro_batch_size": 1,
        "lr": 1e-6,
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 4,
        "rollout_num_gpus": 4,
        "rollout_batch_size": 64,
        "rollout_max_response_len": 12000,
        "eval_interval": 20,
        "wandb_key": "secret",
        "wandb_dir": "/private/wandb",
        "wandb_host": "https://private.example",
        "hf_checkpoint": "/private/model",
        "load": "/private/checkpoint",
        "prompt_data": "/private/data.jsonl",
        "ray_address": "10.0.0.1:6379",
        "arbitrary_internal_option": "not portable",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_opsa_config_is_compact_and_path_free(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    config = wandb_utils._compute_config_for_logging(_args())

    assert config["advantage_estimator"] == "opsa"
    assert config["opsa_token_fraction"] == 0.2
    assert config["model_name"] == "qwen3"
    assert config["global_batch_size"] == 64
    assert config["actor_num_gpus_per_node"] == 4
    assert config["rollout_max_response_len"] == 12000
    assert config["eval_interval"] == 20
    assert config["env_vars"] == {"SLURM_JOB_ID": "123"}

    assert "wandb_key" not in config
    assert "wandb_dir" not in config
    assert "wandb_host" not in config
    assert "hf_checkpoint" not in config
    assert "load" not in config
    assert "prompt_data" not in config
    assert "ray_address" not in config
    assert "arbitrary_internal_option" not in config


def test_non_opsa_config_keeps_existing_fields_but_removes_wandb_key():
    config = wandb_utils._compute_config_for_logging(
        _args(advantage_estimator="grpo", arbitrary_internal_option="preserved")
    )

    assert config["arbitrary_internal_option"] == "preserved"
    assert config["hf_checkpoint"] == "/private/model"
    assert "wandb_key" not in config


def test_secondary_uses_sanitized_config(monkeypatch):
    args = _args(
        wandb_run_id="run-id",
        wandb_mode="offline",
        use_wandb=True,
        wandb_team="team",
        wandb_project="project",
        wandb_dir=None,
    )
    captured = {}

    monkeypatch.setattr(wandb_utils.wandb, "init", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(wandb_utils.wandb, "define_metric", lambda *args, **kwargs: None)
    monkeypatch.setattr(wandb_utils.wandb, "Settings", lambda **kwargs: kwargs)

    wandb_utils.init_wandb_secondary(args)

    assert captured["config"]["advantage_estimator"] == "opsa"
    assert "wandb_key" not in captured["config"]
    assert "load" not in captured["config"]


def test_opsa_open_metrics_is_opt_in(monkeypatch):
    args = _args(
        use_wandb=True,
        wandb_mode="online",
        wandb_run_id="run-id",
        wandb_open_metrics=False,
    )
    finish_calls = []
    monkeypatch.setattr(wandb_utils.wandb, "finish", lambda: finish_calls.append(True))

    wandb_utils.reinit_wandb_primary_with_open_metrics(args, "http://router")

    assert finish_calls == []


def _assert_open_metrics_reinitialized(monkeypatch, args):
    import sys

    fake_router = SimpleNamespace(__version__="slime-test")
    captured = {}
    finish_calls = []
    monkeypatch.setitem(sys.modules, "sglang_router", fake_router)
    monkeypatch.setattr(wandb_utils.wandb, "finish", lambda: finish_calls.append(True))
    monkeypatch.setattr(wandb_utils.wandb, "init", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(wandb_utils.wandb, "Settings", lambda **kwargs: kwargs)
    monkeypatch.setattr(wandb_utils, "_init_wandb_common", lambda: None)

    wandb_utils.reinit_wandb_primary_with_open_metrics(args, "http://router")

    assert finish_calls == [True]
    assert captured["settings"]["x_stats_open_metrics_endpoints"] == {"sgl_engine": "http://router/engine_metrics"}


def test_opsa_open_metrics_can_be_explicitly_enabled(monkeypatch):
    args = _args(
        use_wandb=True,
        wandb_mode="online",
        wandb_run_id="run-id",
        wandb_open_metrics=True,
        wandb_team="team",
        wandb_project="project",
        wandb_dir=None,
    )
    _assert_open_metrics_reinitialized(monkeypatch, args)


def test_non_opsa_open_metrics_preserves_existing_behavior(monkeypatch):
    args = _args(
        advantage_estimator="grpo",
        use_wandb=True,
        wandb_mode="online",
        wandb_run_id="run-id",
        wandb_open_metrics=False,
        wandb_team="team",
        wandb_project="project",
        wandb_dir=None,
    )
    _assert_open_metrics_reinitialized(monkeypatch, args)
