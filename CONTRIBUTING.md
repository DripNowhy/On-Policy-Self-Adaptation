# Contributing

Contributions should keep this repository focused on OPSA, its documented
ablations, and the standard OPD baselines used for comparison.

Please do not commit checkpoints, datasets, W&B runs, logs, debug rollouts,
backup files, machine-specific paths, or one-off experimental branches.

Before opening a pull request:

```bash
pre-commit run --all-files

cd slime
export MEGATRON_PATH=/path/to/Megatron-LM
PYTHONPATH=.:"${MEGATRON_PATH}" python -m pytest -q \
  tests/test_opsa.py \
  tests/test_opsa_loss_mask.py \
  tests/test_opd.py \
  tests/test_on_policy_distillation_cpu.py

bash -n examples/opsa/run_opsa.sh
bash -n examples/opsa/run_lowest_sweep.sh
```

GPU training is not required for documentation-only changes. For runtime
changes, state the model, data, resource layout, command, and validation scope
in the pull request.
