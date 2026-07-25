import logging

import wandb

from . import wandb_utils
from .tensorboard_utils import _TensorboardAdapter

_OPSA_COMPACT_EXACT_METRICS = {
    "train/loss",
    "train/entropy_loss",
    "train/grad_norm",
    "rollout/opsa/selected_fraction",
    "rollout/opsa/advantage_mean",
    "rollout/log_probs",
    "rollout/entropy",
    "rollout/truncated_ratio",
    "rollout/repetition_frac",
    "perf/rollout_time",
    "perf/actor_train_tok_per_s",
}
_OPSA_RESPONSE_LENGTH_STATS = {"min", "mean", "max"}
_OPSA_EVAL_SCORE_SUFFIXES = ("-avg@4", "-pass@4")
_OPSA_EVAL_SAMPLE_METRIC_SUFFIXES = ("/repetition_frac", "/truncated_ratio")


_LOGGER_CONFIGURED = False


# ref: SGLang
def configure_logger(prefix: str = ""):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def init_tracking(args, primary: bool = True, **kwargs):
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)


def update_tracking_open_metrics(args, router_addr):
    wandb_utils.reinit_wandb_primary_with_open_metrics(args, router_addr)


def finish_tracking(args):
    if not args.use_wandb:
        return
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        logging.getLogger(__name__).exception("Failed to finish wandb run")


def _is_compact_opsa_metric(metric_name: str) -> bool:
    if metric_name in _OPSA_COMPACT_EXACT_METRICS:
        return True
    if metric_name.startswith("train/lr-pg_"):
        return True
    if metric_name.startswith("eval/") and metric_name.endswith(_OPSA_EVAL_SCORE_SUFFIXES):
        return True
    if metric_name.startswith("eval/") and metric_name.endswith(_OPSA_EVAL_SAMPLE_METRIC_SUFFIXES):
        return True

    prefix, separator, statistic = metric_name.rpartition("/")
    if separator == "" or statistic not in _OPSA_RESPONSE_LENGTH_STATS:
        return False
    if prefix == "rollout/response_len":
        return True
    return prefix.startswith("eval/") and prefix.endswith("/response_len")


def _filter_wandb_metrics(args, metrics, step_key: str):
    if getattr(args, "advantage_estimator", None) != "opsa" or getattr(args, "wandb_log_all_metrics", False):
        return metrics

    filtered = {key: value for key, value in metrics.items() if _is_compact_opsa_metric(key)}
    if step_key in metrics:
        filtered[step_key] = metrics[step_key]
    return filtered


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    if args.use_wandb:
        wandb_metrics = _filter_wandb_metrics(args, metrics, step_key)
        if any(key != step_key for key in wandb_metrics):
            wandb.log(wandb_metrics)

    if args.use_tensorboard:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])
