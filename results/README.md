# R2R-CE val-unseen results

- [ranking.csv](ranking.csv): all 41 checkpoints, ranked by SR and then SPL.
- [r2rce_val_unseen.json](r2rce_val_unseen.json): metrics for the complete checkpoint window.
- [checkpoint.json](checkpoint.json): exported iter12800 model metadata and SHA256.
- [iter12800/](iter12800/): per-episode results and aggregate metrics from the release validation.

The release code and exported checkpoint were evaluated on all 1839 episodes
on November 8, 2026. Every per-episode metric matched the original iter12800
evaluation exactly. The release test suite also passed all 135 tests.

See the [evaluation configuration](../docs/EVALUATION.md) and
[installation guide](../docs/INSTALL.md) for the recorded setup.
