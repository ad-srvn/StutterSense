# StutterSense v2 results

This directory contains the curated, lightweight record of the completed three-seed Google Colab experiment. Model checkpoints and the frozen WavLM embedding cache remain local because they are reproducible generated artifacts.

## Run configuration

| Item | Value |
|---|---|
| Dataset | `psusac/sep28k@67630930030046543194dc7c701b87b2cdc7680f` |
| Encoder | `microsoft/wavlm-base-plus@4c66d4806a428f2e922ccfa1a962776e232d487b` |
| Sampling | 2,000 clips from a complete 21,856-row reservoir scan |
| Decoding | 2,000 successful; 0 failed |
| Split | 1,400 train / 300 validation / 300 test |
| Seeds | 42, 123, 456 |
| Epochs | 12 |
| Hardware | NVIDIA A100-SXM4-40GB |
| Bootstrap | 1,000 primary-seed dependence-group replicates |

No verified speaker or episode identifiers were available. Exact duplicate components were kept within a single split, but speaker independence is not verified.

## Multi-seed test results

Validation-selected metrics are mean ± sample standard deviation across the three optimization seeds.

| Model | Macro F1 | Micro F1 | Macro AUROC | Macro AP | Hard ECE ↓ | Soft ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Hard BCE | 0.414 ± 0.006 | 0.441 ± 0.015 | 0.801 ± 0.002 | 0.411 ± 0.002 | **0.0236 ± 0.0026** | 0.0447 ± 0.0145 |
| Soft BCE | 0.421 ± 0.017 | 0.448 ± 0.009 | 0.803 ± 0.002 | 0.422 ± 0.001 | 0.0418 ± 0.0052 | 0.0192 ± 0.0028 |
| BCE + Brier ablation | **0.432 ± 0.025** | **0.458 ± 0.016** | **0.803 ± 0.002** | **0.423 ± 0.003** | 0.0417 ± 0.0053 | 0.0183 ± 0.0023 |
| Ambiguity multi-task | 0.423 ± 0.013 | 0.447 ± 0.005 | 0.797 ± 0.002 | 0.417 ± 0.005 | 0.0410 ± 0.0067 | **0.0161 ± 0.0038** |

The Brier ablation won on seeds 123 and 456, while hard BCE won on seed 42. Its higher mean F1 therefore should not be interpreted as a uniform improvement.

## Paired bootstrap

Primary-seed macro-F1 differences relative to hard BCE were:

| Candidate | Difference | 95% interval |
|---|---:|---:|
| Soft BCE | -0.0126 | [-0.0651, +0.0386] |
| BCE + Brier | -0.0111 | [-0.0604, +0.0398] |
| Ambiguity multi-task | -0.0061 | [-0.0549, +0.0433] |

Every interval includes zero. All three soft-target variants improved primary-seed soft Brier score relative to hard BCE, with intervals excluding zero.

## Event summary

Averaged across all models and seeds:

| Event | F1 | AUROC | AP |
|---|---:|---:|---:|
| Interjection | 0.608 | 0.863 | 0.684 |
| Prolongation | 0.448 | 0.854 | 0.422 |
| Sound repetition | 0.447 | 0.838 | 0.449 |
| Block | 0.324 | 0.707 | 0.243 |
| Word repetition | 0.285 | 0.743 | 0.292 |

Validation-selected thresholds increased mean macro F1 by 0.153–0.178 compared with the fixed 0.5 policy. The ambiguity head achieved mean per-event AUROC 0.705 and F1 0.462 for predicting intermediate vote counts; its ranking discrimination was similar to event entropy.

## Artifact guide

- `metrics_by_seed.csv`: fixed and tuned aggregate metrics for every seed and model.
- `per_class_metrics.csv`: event-level classification, ranking, calibration, and confusion statistics.
- `calibration_by_event.csv`: majority-vote and annotator-proportion calibration.
- `thresholds.csv`: fixed and validation-selected per-event thresholds.
- `paired_group_bootstrap.csv`: primary-seed paired confidence intervals.
- `ambiguity_detection.csv`: ambiguity-head, entropy, and prevalence baselines.
- `risk_coverage.csv`: entropy, rarity, predicted-ambiguity, and random rankings.
- `quality_control.csv`: diagnostic-annotation subgroup results.
- `sample_manifest.csv` and `split_indices.json`: sample and split audit trail.
- `config.json` and `dataset_provenance.json`: pinned run configuration and environment metadata.

Primary-seed figures are stored once in [`../figures/`](../figures/) because the completed v2 run reproduced the original seed-42 metrics and figures exactly. External FluencyBank evaluation and the quality-filter retraining ablation were not executed.
