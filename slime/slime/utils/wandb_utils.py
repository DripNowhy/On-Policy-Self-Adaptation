import logging
import os
from copy import deepcopy

import wandb

logger = logging.getLogger(__name__)

# OPSA runs intentionally publish only stable, portable experiment metadata.
# In particular, paths, service addresses, credentials, and arbitrary environment
# variables must not become part of the public run config.
_OPSA_CONFIG_KEYS = (
    # Method.
    "advantage_estimator",
    "opsa_mode",
    "opsa_token_fraction",
    "opsa_advantage_min",
    "opsa_advantage_max",
    "opsa_fixed_advantage",
    # Model architecture.
    "model_name",
    "num_layers",
    "hidden_size",
    "ffn_hidden_size",
    "num_attention_heads",
    "num_query_groups",
    "kv_channels",
    "seq_length",
    "max_position_embeddings",
    "position_embedding_type",
    "rotary_percent",
    "rotary_base",
    "normalization",
    "norm_epsilon",
    "swiglu",
    "untie_embeddings_and_output_weights",
    "qk_layernorm",
    "vocab_size",
    "padded_vocab_size",
    # Training and optimizer.
    "train_backend",
    "train_iters",
    "num_epoch",
    "num_rollout",
    "global_batch_size",
    "micro_batch_size",
    "num_steps_per_rollout",
    "use_dynamic_batch_size",
    "max_tokens_per_gpu",
    "optimizer",
    "optimizer_cpu_offload",
    "overlap_cpu_optimizer_d2h_h2d",
    "use_precision_aware_optimizer",
    "recompute_granularity",
    "recompute_method",
    "recompute_num_layers",
    "lr",
    "min_lr",
    "lr_decay_style",
    "lr_decay_iters",
    "lr_warmup_iters",
    "lr_warmup_fraction",
    "weight_decay",
    "adam_beta1",
    "adam_beta2",
    "adam_eps",
    "clip_grad",
    "bf16",
    "fp16",
    "seed",
    "entropy_coef",
    "save_interval",
    # Resource layout.
    "actor_num_nodes",
    "actor_num_gpus_per_node",
    "rollout_num_gpus",
    "rollout_num_gpus_per_engine",
    "num_gpus_per_node",
    "tensor_model_parallel_size",
    "pipeline_model_parallel_size",
    "expert_model_parallel_size",
    "context_parallel_size",
    "sequence_parallel",
    "colocate",
    "offload_train",
    "offload_rollout",
    # Rollout and evaluation.
    "rollout_batch_size",
    "n_samples_per_prompt",
    "rollout_temperature",
    "rollout_top_p",
    "rollout_top_k",
    "rollout_max_context_len",
    "rollout_max_prompt_len",
    "rollout_max_response_len",
    "rollout_seed",
    "disable_thinking",
    "eval_interval",
    "n_samples_per_eval_prompt",
    "eval_temperature",
    "eval_top_p",
    "eval_top_k",
    "eval_max_context_len",
    "eval_max_prompt_len",
    "eval_max_response_len",
)


def _is_opsa(args) -> bool:
    return getattr(args, "advantage_estimator", None) == "opsa"


def _is_offline_mode(args) -> bool:
    """Detect whether W&B should run in offline mode.

    Priority order:
    1) args.wandb_mode if provided
    2) WANDB_MODE environment variable
    """
    if args.wandb_mode:
        return args.wandb_mode == "offline"
    return os.environ.get("WANDB_MODE") == "offline"


def init_wandb_primary(args):
    if not args.use_wandb:
        args.wandb_run_id = None
        return

    # Set W&B mode if specified (overrides WANDB_MODE env var)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_mode == "offline":
            logger.info("W&B offline mode enabled. Data will be saved locally.")
        elif args.wandb_mode == "disabled":
            logger.info("W&B disabled mode enabled. No data will be logged.")
        elif args.wandb_mode == "online":
            logger.info("W&B online mode enabled. Data will be uploaded to cloud.")

    offline = _is_offline_mode(args)

    # Only perform explicit login when NOT offline
    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Prepare wandb init parameters
    # add random 6 length string with characters
    if args.wandb_random_suffix:
        group = args.wandb_group + "_" + wandb.util.generate_id()
        run_name = f"{group}-RANK_{args.rank}"
    else:
        group = args.wandb_group
        run_name = args.wandb_group

    # Prepare wandb init parameters
    init_kwargs = {
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "group": group,
        "name": run_name,
        "config": _compute_config_for_logging(args),
    }

    # Configure settings based on offline/online mode
    if offline:
        init_kwargs["settings"] = wandb.Settings(mode="offline")
    else:
        init_kwargs["settings"] = wandb.Settings(mode="shared", x_primary=True)

    # Add custom directory if specified
    if args.wandb_dir:
        # Ensure directory exists to avoid backend crashes
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir
        logger.info(f"W&B logs will be stored in: {args.wandb_dir}")

    wandb.init(**init_kwargs)

    _init_wandb_common()

    # Set wandb_run_id in args for easy access throughout the training process
    args.wandb_run_id = wandb.run.id


def reinit_wandb_primary_with_open_metrics(args, router_addr):
    """Re-initialize the primary W&B run with open metrics endpoints.

    The primary wandb init happens before rollout servers start (to obtain
    ``wandb_run_id`` for secondary processes).  This function is called
    *after* servers are up so the router address is available for scraping
    SGLang Prometheus metrics via the primary process's stats monitor.
    """
    if not args.use_wandb or _is_offline_mode(args):
        return
    if getattr(args, "wandb_mode", None) == "disabled":
        return
    if router_addr is None:
        return
    if _is_opsa(args) and not getattr(args, "wandb_open_metrics", False):
        logger.info("Skipping SGLang OpenMetrics for OPSA. Pass --wandb-open-metrics to upload sgl_engine.* metrics.")
        return
    if os.environ.get("SLIME_DISABLE_WANDB_OPEN_METRICS_REINIT", "").lower() in {"1", "true", "yes", "on"}:
        logger.info("Skipping W&B open-metrics reinit because SLIME_DISABLE_WANDB_OPEN_METRICS_REINIT is set.")
        return
    wandb_run_id = getattr(args, "wandb_run_id", None)
    if wandb_run_id is None:
        return

    import sglang_router

    if "slime" not in sglang_router.__version__:
        logger.warning(
            "Only customized sglang_router from https://github.com/zhuzilin/sgl-router supports uploading metrics."
        )
        return

    logger.info(f"Re-initializing primary W&B with SGLang metrics at {router_addr}.")

    wandb.finish()

    init_kwargs = {
        "id": wandb_run_id,
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "resume": "allow",
        "reinit": True,
        "settings": wandb.Settings(
            mode="shared",
            x_primary=True,
            x_stats_open_metrics_endpoints={
                "sgl_engine": f"{router_addr}/engine_metrics",
            },
            x_stats_open_metrics_filters={
                "sgl_engine.*": {},
            },
        ),
    }

    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir

    wandb.init(**init_kwargs)
    _init_wandb_common()


def _compute_config_for_logging(args):
    output = deepcopy(args.__dict__)
    output.pop("wandb_key", None)

    if _is_opsa(args):
        output = {key: output[key] for key in _OPSA_CONFIG_KEYS if key in output}

    whitelist_env_vars = [
        "SLURM_JOB_ID",
        # We may insert more default values here, and may also allow users to configure a whitelist
    ]
    output["env_vars"] = {k: v for k, v in os.environ.items() if k in whitelist_env_vars}

    return output


# https://docs.wandb.ai/guides/track/log/distributed-training/#track-all-processes-to-a-single-run
def init_wandb_secondary(args):
    wandb_run_id = getattr(args, "wandb_run_id", None)
    if wandb_run_id is None:
        return

    # Set W&B mode if specified (same as primary)
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode

    offline = _is_offline_mode(args)

    if (not offline) and args.wandb_key is not None:
        wandb.login(key=args.wandb_key, host=args.wandb_host)

    # Configure settings based on offline/online mode
    if offline:
        settings_kwargs = dict(mode="offline")
    else:
        settings_kwargs = dict(
            mode="shared",
            x_primary=False,
            x_update_finish_state=False,
        )

    init_kwargs = {
        "id": wandb_run_id,
        "entity": args.wandb_team,
        "project": args.wandb_project,
        "config": _compute_config_for_logging(args),
        "resume": "allow",
        "reinit": True,
        "settings": wandb.Settings(**settings_kwargs),
    }

    # Add custom directory if specified
    if args.wandb_dir:
        os.makedirs(args.wandb_dir, exist_ok=True)
        init_kwargs["dir"] = args.wandb_dir

    wandb.init(**init_kwargs)

    _init_wandb_common()


def _init_wandb_common():
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("rollout/step")
    wandb.define_metric("rollout/*", step_metric="rollout/step")
    wandb.define_metric("multi_turn/*", step_metric="rollout/step")
    wandb.define_metric("passrate/*", step_metric="rollout/step")
    wandb.define_metric("eval/step")
    wandb.define_metric("eval/*", step_metric="eval/step")
    wandb.define_metric("perf/*", step_metric="rollout/step")
