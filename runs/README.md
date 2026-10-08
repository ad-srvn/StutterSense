# Experiment runs

This directory contains curated outputs produced by the current notebook methodology. Each run has its own configuration, immutable dataset/model revisions, provenance, splits, metrics, figures, and limitations.

| Run | Scope | Status |
|---|---|---|
| [`2026-10-08-full-stream-seed42`](2026-10-08-full-stream-seed42/README.md) | 2,000 samples selected from a complete 21,856-row scan; seed 42; four model variants | Validated |

The earlier methodology is retained separately under [`results/`](../results/README.md). Generated checkpoints and embedding caches are excluded from Git; their hashes and metadata are recorded with each applicable run.
