# Upstream provenance

This repository was prepared from the tracked `slime/` contents at local
repository commit `594c562`. The existing working tree was not copied
wholesale; the OPSA implementation and its focused ablations were selected
explicitly.

The modified framework is contained in the [`slime/`](slime/) subdirectory so
the repository root can document On-Policy Self-Adaptation itself. The nested
directory is a normal source directory, not a second Git repository.

The corresponding public Slime reference point is
[THUDM/slime commit `0988f0f4a0ab55d1bb3ce6285a597d912144fa80`](https://github.com/THUDM/slime/commit/0988f0f4a0ab55d1bb3ce6285a597d912144fa80).
The local snapshot may contain project-specific changes relative to that public
commit; this file records provenance rather than asserting a byte-for-byte
match.

Generated experiment outputs, checkpoints, logs, caches, debug rollouts, and
machine-local paths are intentionally excluded. Datasets, Hugging Face
checkpoints, Megatron checkpoints, and Megatron-LM remain external. The source
remains under the Apache License 2.0 in [LICENSE](LICENSE).
