# OPSA

OPSA (On-Policy Self-Advantage) applies policy loss only to the current actor's
lowest-log-probability response tokens. The selector is **DP-local**: each data
parallel rank concatenates the valid response tokens in its own packed training
batch, selects

\[
K = \max(1, \lfloor fN \rfloor)
\]

tokens by ascending recomputed actor log probability, and does not gather token
statistics across DP ranks.

Within the selected tokens, entropy is min-max ranked and the canonical method
assigns

\[
A = A_{\max} + (A_{\min} - A_{\max})r,
\]

with `f=0.2`, `A_min=-1.0`, and `A_max=-0.5`. If all selected entropies are
equal, every selected token receives `-1.0`. Unselected tokens receive zero
advantage and are excluded from the policy-loss mask and denominator.

OPSA is reference-free. The checkpoint supplied through `--ref-load` is used
only to initialize the actor when `--load` is absent; OPSA performs neither a
reference forward pass nor a KL loss. Training uses zero task reward.

## Public interface

```text
--advantage-estimator opsa
--opsa-mode {entropy,fixed}
--opsa-token-fraction FLOAT
--opsa-advantage-min FLOAT
--opsa-advantage-max FLOAT
--opsa-fixed-advantage FLOAT
```

Entropy mode automatically requests actor entropy. Fixed mode uses the same
lowest-token selector without computing entropy. OPSA is mutually exclusive
with standard OPD and advantage normalization.

The focused presets are:

- `opsa`: entropy-ranked advantage from `-0.5` to `-1.0`.
- `fixed-negative`: fixed `-0.5` advantage.
- `fixed-positive`: fixed `+0.2` advantage.

No entropy threshold, token/EOS/think mask, position reweighting, clipping,
forced-token override, top-1 branch, or random branch is part of OPSA.

## Reproduction launchers

The launcher supports Qwen3-1.7B, Qwen3-4B Base, and Qwen3.5-9B. It validates
paths and GPU/resource combinations before a real run, can submit to an existing
Ray dashboard, and starts/stops only its own local Ray cluster otherwise.

```bash
bash examples/opsa/run_opsa.sh \
  --model qwen3-1.7b \
  --preset opsa \
  --fraction 0.2 \
  --dry-run

bash examples/opsa/run_lowest_sweep.sh \
  --model qwen3-1.7b \
  --dry-run
```

See `examples/opsa/README.md` for checkpoint conversion, real-run commands,
resource defaults, fixed-advantage ablations, resumption, and the default
10/20/30/40% sweep.

## Validation boundary

This cleaned release is CPU/CLI validated: CPU unit tests cover the OPSA
selector and arguments, and launchers are checked using shell syntax validation
and dry runs. No end-to-end GPU training of the three model presets was repeated
during the cleanup.
