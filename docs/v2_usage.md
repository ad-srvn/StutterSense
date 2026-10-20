# StutterSense 2.0 usage

StutterSense 2.0 provides a self-contained Colab notebook and a modular Python runner. The completed three-seed notebook results are summarized in `results/v2/`; the earlier seed-42 report remains in `results/`.

## Configuration

`configs/v2_full.json` pins the dataset and WavLM revisions and uses optimization seeds 42, 123, and 456. `configs/v2_smoke.json` limits the hosted stream and is only an executable integration check. Every run receives a new timestamped directory. To resume, set `output.resume_from` to an existing run whose configuration fingerprint, split, cache, and checkpoints match.

The default SEP-28k evaluation is marked `NOT_VERIFIED` for speaker/episode independence. See `docs/identity_manifest.md` before enabling a grouped identity split.

## Local CPU development

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-v2.txt -r requirements-dev.txt
python run_v2.py --config configs/v2_smoke.json --device cpu
```

The last command is executable and downloads the pinned dataset sample and WavLM weights. It is not part of offline validation.

## Offline validation

```bash
python scripts/validate_repo.py
python scripts/validate_v2.py
python -m unittest discover -s tests -v
```

These commands use synthetic inputs and do not download SEP-28k or WavLM.

## Google Colab GPU experiment

Upload `StutterSense_v2.ipynb` to a fresh Colab runtime, select **Runtime → Change runtime type → GPU**, and choose **Runtime → Run all**. The notebook is self-contained, uses the pinned full configuration, and does not require Google Drive or a repository clone. The final cell creates a downloadable output ZIP.

The equivalent terminal command on a configured GPU machine is:

```bash
python run_v2.py --config configs/v2_full.json --device cuda
```

To resume after interruption, use the identical experiment configuration and an existing run directory:

```bash
python run_v2.py --config configs/v2_full.json --device cuda --resume-from v2_outputs/<run-directory>
```

## Selection policy

All four objectives share one split. The runner trains on `train`, performs early stopping and per-event threshold selection on `validation`, and touches `test` labels only afterward. Fixed 0.5 thresholds remain an explicit comparison. Undefined event metrics are stored as blank values with status fields rather than replaced with zero.

## Modular runner outputs

Runs are written below `v2_outputs/<UTC timestamp>-<config hash>/` and labeled `scientific_validation: UNVALIDATED`. Expected files include configuration, environment and provenance JSON; sample and split manifests; per-seed and aggregate metrics; per-event calibration and error tables; thresholds; grouped bootstrap intervals; risk-coverage tables; figures; embedding cache; and metadata-rich checkpoints.

The concrete top-level files are `run_metadata.json`, `config.json`, `environment.json`, `provenance.json`, `sample_manifest.csv`, `splits.json`, `metrics_by_seed.csv`, `metrics_aggregate.csv`, `per_event_metrics.csv`, `thresholds.csv`, `training_history.csv`, `group_bootstrap_intervals.csv`, `paired_group_bootstrap.csv`, `ambiguity_metrics.csv`, `false_positive_negative.csv`, `error_stratification.csv`, `risk_coverage.csv`, and `artifact_manifest.json`. Full runs also create `baseline_vs_v2.csv` when the validated baseline table is available. Checkpoints, cache files, and plots are placed in `checkpoints/`, `cache/`, and `figures/` within the run directory.

The bootstrap resamples whole dependence groups. When only duplicate groups are known, interval rows state `NOT_VERIFIED_IDENTITY_INDEPENDENCE` and are descriptive, not evidence of statistical significance. Seed means likewise describe optimization variability and are not independent experimental replicates.

## Notebook outputs

The self-contained notebook writes to `/content/stuttersense_v2_outputs` in Colab. Its curated CSV and JSON artifacts are committed in `results/v2/`. Checkpoints and the WavLM embedding cache are excluded from Git because they are reproducible generated artifacts.

## Inference scope

`StutterSensePredictor` accepts a user-provided v2 checkpoint, waveform, and sample rate. It verifies label order, preprocessing, encoder revision, dataset revision, and classifier dimensions. Outputs are five clip-level event probabilities and ambiguity probabilities only for a multi-task checkpoint. It does not produce timestamps, streaming decisions, or end-of-turn probabilities.
