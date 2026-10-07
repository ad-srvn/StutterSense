# Reference experiment summary

The fixed held-out test split contained 300 clips. The split prevents exact waveform duplicates from crossing partitions, but it is not verified speaker-disjoint because the hosted schema exposes no speaker or episode identifier.

## Measured results

| Model | Macro F1 | Micro F1 | Macro AUROC | Soft Brier ↓ | Soft ECE ↓ | Hard ECE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Hard-label BCE | 0.1853 | 0.3651 | 0.8091 | 0.0595 | 0.0279 | **0.0297** |
| Soft-label BCE | 0.2645 | 0.4045 | 0.8281 | 0.0549 | 0.0160 | 0.0417 |
| Soft BCE + Brier | **0.2713** | **0.4148** | **0.8287** | **0.0547** | **0.0157** | 0.0441 |

- Both soft-target variants improved macro F1, micro F1, ranking metrics, and calibration to annotator proportions relative to the hard-label baseline in this run.
- Calibration depends on the target definition: soft-target calibration improved while majority-label ECE worsened.
- The Brier-augmented model had the strongest point estimates, but its advantage over ordinary soft BCE was small.
- Prediction entropy was positively associated with the human-disagreement proxy (Spearman ρ from 0.316 to 0.346 across models).
- Mean Hamming error was roughly 0.052–0.054 on high-agreement clips and 0.125–0.126 on clips containing at least one ambiguous event label.
- Retaining only the least-uncertain 10% of test clips reduced Hamming risk from approximately 0.105–0.107 at full coverage to 0.013–0.020.
- Fixed 0.5 thresholds produced low recall for several rare classes. The hard-label model, for example, predicted no positive `block` examples despite a block AUROC of 0.695.

These are descriptive measurements from one run. They do not establish statistical significance, speaker-independent generalization, epistemic uncertainty, or clinical validity.

## Limitations and threats to validity

- The hosted default subset is not assumed to equal the complete original SEP-28k corpus.
- No verified speaker or episode IDs means potential speaker or content leakage across the clip-level split.
- Aggregate counts reveal ambiguity but not annotator identities, dependencies, or individual reliability.
- A 2-of-3 majority threshold discards information and can be unstable near the boundary.
- Frozen embeddings and one MLP family limit the scope of architectural conclusions.
- One random split and seed do not quantify sampling or optimization variability.
- Predictive entropy mixes data ambiguity, model fit, and class prevalence; it is not a pure measure of epistemic uncertainty.
- Class imbalance makes thresholded F1 unstable, especially for events with few positive examples.
- These outputs are research predictions, not clinical diagnoses.

## Suggested follow-up experiments

- Obtain verified speaker or episode IDs and perform grouped speaker-disjoint cross-validation.
- Repeat the experiment across seeds and bootstrap clips or speakers for confidence intervals.
- Compare frozen features with carefully regularized end-to-end fine-tuning.
- Evaluate ensembles or Bayesian approximations for epistemic uncertainty.
- Model individual annotators when rater-level labels become available.
- Tune thresholds on validation data and report both fixed and tuned policies.
- Pre-register primary calibration metrics and test on an external corpus.

