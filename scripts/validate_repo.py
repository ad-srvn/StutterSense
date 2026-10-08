#!/usr/bin/env python3
"""Fast, offline integrity checks for the notebook and committed result runs."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "StutterSense.ipynb"
REVISED_RUN = ROOT / "runs" / "2026-10-08-full-stream-seed42"
EVENTS = {"prolongation", "block", "sound_rep", "word_rep", "interjection"}
REVISED_MODELS = {"hard", "soft", "brier_ablation", "ambiguity_multitask"}
REVISED_FIGURES = {
    "A_dataset_label_frequency.png",
    "B_annotator_disagreement.png",
    "C_training_validation_curves.png",
    "D_per_class_f1.png",
    "E_reliability_diagrams.png",
    "F_risk_coverage.png",
    "G_disagreement_vs_uncertainty.png",
    "H_multilabel_error_analysis.png",
    "I_precision_recall_curves.png",
    "J_per_event_calibration.png",
    "annotation_count_distribution.png",
    "audio_duration_distribution.png",
    "event_cooccurrence.png",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_number(value: str) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def png_metadata(path: Path) -> tuple[int, int, float | None]:
    data = path.read_bytes()
    require(data[:8] == b"\x89PNG\r\n\x1a\n", f"Invalid PNG signature: {path}")
    width, height = struct.unpack(">II", data[16:24])
    offset = 8
    dpi = None
    while offset + 12 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        payload = data[offset + 8 : offset + 8 + length]
        if kind == b"pHYs" and len(payload) == 9:
            pixels_per_meter_x, _, unit = struct.unpack(">IIB", payload)
            if unit == 1:
                dpi = pixels_per_meter_x * 0.0254
            break
        offset += 12 + length
    return width, height, dpi


def validate_notebook() -> None:
    require(NOTEBOOK.exists(), "StutterSense.ipynb is missing")
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    require(notebook.nbformat == 4, "Notebook must use nbformat 4")
    require(len(notebook.cells) == 38, f"Expected 38 notebook cells, found {len(notebook.cells)}")

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    require(len(code_cells) == 23, f"Expected 23 code cells, found {len(code_cells)}")
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"{NOTEBOOK.name}:cell-{index}")
            require(cell.execution_count is None, f"Cell {index} has a stored execution count")
            require(not cell.outputs, f"Cell {index} contains stored outputs")

    source = "\n".join(cell.source for cell in notebook.cells)
    environment_source = notebook.cells[2].source
    require('"numpy>=1.26,<3"' not in environment_source, "Colab setup must not replace preloaded NumPy")
    require('"-U"' not in environment_source, "Colab setup must not upgrade its preloaded scientific stack")
    require("changed_core" in environment_source, "Colab setup needs an in-memory package-change guard")
    for required in (
        'DATASET_ID = "psusac/sep28k"',
        'MODEL_NAME = "microsoft/wavlm-base-plus"',
        'IN_COLAB = "google.colab" in sys.modules',
        '"/content/stuttersense_outputs" if IN_COLAB else "outputs"',
        "RESOLVED_DATASET_REVISION",
        "RESOLVED_MODEL_REVISION",
        "streaming=True",
        "truncation=False",
        "reservoir_sample",
        "build_dependence_groups",
        "hard_targets = (counts_matrix >= 2)",
        "soft_targets = counts_matrix / 3.0",
        "ambiguity_targets",
        "brier_ablation",
        "ambiguity_multitask",
        "select_f1_thresholds",
        "paired_group_bootstrap_difference",
        "validate_fluencybank_manifest",
    ):
        require(required in source, f"Required notebook content is missing: {required}")
    require((ROOT / "tests" / "test_synthetic.py").exists(), "Synthetic test suite is missing")


def validate_revised_run() -> None:
    required_files = {
        "README.md", "artifact_manifest.json", "ambiguity_detection.csv",
        "ambiguity_prevalence.csv", "calibration_by_event.csv", "config.json",
        "dataset_provenance.json", "evaluation_metrics.json", "experiment_summary.md",
        "external_evaluation_status.json", "metrics_by_seed.csv",
        "paired_group_bootstrap.csv", "per_class_metrics.csv", "quality_control.csv",
        "risk_coverage.csv", "sample_manifest.csv", "split_indices.json",
        "thresholds.csv", "thresholds.json", "training_histories.json",
    }
    missing = sorted(name for name in required_files if not (REVISED_RUN / name).exists())
    require(not missing, f"Missing revised-run files: {missing}")

    config = json.loads((REVISED_RUN / "config.json").read_text(encoding="utf-8"))
    provenance = json.loads((REVISED_RUN / "dataset_provenance.json").read_text(encoding="utf-8"))
    require(config["experiment_seeds"] == [42], "Unexpected revised-run seeds")
    require(config["full_stream_sampling"], "Revised run was not full-stream sampled")
    require(not config["fast_smoke_test"] and not config["cpu_smoke_test"], "Revised run used smoke mode")
    require(config["run_max_samples"] == 2000, "Unexpected revised-run sample target")
    require(config["dataset_revision_resolved"] == provenance["resolved_dataset_revision"], "Dataset revision mismatch")
    require(config["model_revision_resolved"] == provenance["resolved_model_revision"], "Model revision mismatch")
    sampling = provenance["sampling"]
    require(sampling == {
        "algorithm": "Algorithm R reservoir sampling without replacement",
        "seed": 42,
        "stream_examples_scanned": 21856,
        "scan_limit": None,
        "requested_samples": 2000,
        "decoded_samples": 2000,
        "decode_failures": 0,
        "full_stream": True,
    }, "Unexpected revised-run sampling provenance")
    require(provenance["verified_identifier_columns"] == [], "Unexpected identity columns")
    require(config["verified_group_column"] is None, "Unverified group column was used")

    manifest = read_csv(REVISED_RUN / "sample_manifest.csv")
    require(len(manifest) == 2000, "Unexpected revised-run manifest size")
    hashes = [row["audio_sha256"] for row in manifest]
    require(len(hashes) == len(set(hashes)), "Revised run contains duplicate fingerprints")
    for row in manifest:
        require(all(row[event] in {"0", "1", "2", "3"} for event in EVENTS), "Invalid revised-run label count")
        require(finite_number(row["duration_seconds"]), "Invalid revised-run duration")

    splits = json.loads((REVISED_RUN / "split_indices.json").read_text(encoding="utf-8"))
    require({name: len(values) for name, values in splits.items()} == {
        "train": 1400, "validation": 300, "test": 300,
    }, "Unexpected revised-run split sizes")
    split_sets = {name: set(values) for name, values in splits.items()}
    require(set().union(*split_sets.values()) == set(range(2000)), "Revised splits do not partition the manifest")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        require(split_sets[left].isdisjoint(split_sets[right]), f"Revised {left}/{right} overlap")
        left_groups = {manifest[index]["dependence_group"] for index in split_sets[left]}
        right_groups = {manifest[index]["dependence_group"] for index in split_sets[right]}
        require(left_groups.isdisjoint(right_groups), f"Revised {left}/{right} dependence-group overlap")

    metrics = read_csv(REVISED_RUN / "metrics_by_seed.csv")
    require(len(metrics) == 8, "Expected eight revised summary rows")
    require({row["model"] for row in metrics} == REVISED_MODELS, "Unexpected revised model set")
    require({row["threshold_policy"] for row in metrics} == {"fixed_0.5", "validation_tuned"}, "Unexpected threshold policies")
    metric_columns = {
        "macro_f1", "micro_f1", "macro_auroc", "macro_average_precision",
        "hard_brier", "soft_brier", "hard_ece", "soft_ece",
        "mean_confidence", "mean_prediction_entropy", "uncertainty_ambiguity_spearman",
    }
    require(all(all(finite_number(row[column]) for column in metric_columns) for row in metrics), "Non-finite revised summary metric")

    per_class = read_csv(REVISED_RUN / "per_class_metrics.csv")
    require(len(per_class) == 40, "Expected 40 revised per-class rows")
    expected_combinations = {
        (model, policy, event)
        for model in REVISED_MODELS
        for policy in ("fixed_0.5", "validation_tuned")
        for event in EVENTS
    }
    require({(row["model"], row["threshold_policy"], row["event"]) for row in per_class} == expected_combinations,
            "Revised per-class combinations are incomplete")
    for row in per_class:
        require(all(finite_number(row[column]) for column in (
            "threshold", "precision", "recall", "f1", "support", "auroc",
            "average_precision", "hard_brier", "soft_brier", "hard_ece", "soft_ece",
            "tp", "tn", "fp", "fn", "specificity", "npv",
            "false_positive_rate", "false_negative_rate",
        )), "Non-finite revised per-class metric")

    thresholds = read_csv(REVISED_RUN / "thresholds.csv")
    require(len(thresholds) == 40, "Expected 40 threshold rows")
    for row in thresholds:
        threshold = float(row["threshold"])
        require(0 <= threshold <= 1, "Threshold outside [0, 1]")
        if row["policy"] == "fixed_0.5":
            require(threshold == 0.5, "Fixed threshold is not 0.5")

    require(len(read_csv(REVISED_RUN / "calibration_by_event.csv")) == 40, "Unexpected calibration table size")
    require(len(read_csv(REVISED_RUN / "paired_group_bootstrap.csv")) == 6, "Unexpected bootstrap table size")
    require(len(read_csv(REVISED_RUN / "ambiguity_detection.csv")) == 45, "Unexpected ambiguity table size")
    require(len(read_csv(REVISED_RUN / "ambiguity_prevalence.csv")) == 20, "Unexpected ambiguity-prevalence size")
    require(len(read_csv(REVISED_RUN / "quality_control.csv")) == 56, "Unexpected quality-control table size")
    require(len(read_csv(REVISED_RUN / "risk_coverage.csv")) == 130, "Unexpected risk-coverage table size")

    external = json.loads((REVISED_RUN / "external_evaluation_status.json").read_text(encoding="utf-8"))
    require(not external["enabled"] and not external["executed"], "Unexpected external evaluation claim")
    histories = json.loads((REVISED_RUN / "training_histories.json").read_text(encoding="utf-8"))
    require(set(histories) == {f"seed_42/{model}" for model in REVISED_MODELS}, "Training histories are incomplete")

    artifact_manifest = json.loads((REVISED_RUN / "artifact_manifest.json").read_text(encoding="utf-8"))
    require(len(artifact_manifest["local_artifacts"]) == 5, "Unexpected local-artifact manifest")
    for artifact in artifact_manifest["local_artifacts"]:
        require(len(artifact["sha256"]) == 64, "Invalid artifact SHA-256")
        artifact_path = REVISED_RUN / artifact["path"]
        if artifact_path.exists():
            require(artifact_path.stat().st_size == artifact["bytes"], f"Artifact size mismatch: {artifact_path}")
            observed_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            require(observed_hash == artifact["sha256"], f"Artifact hash mismatch: {artifact_path}")

    figure_names = {path.name for path in (REVISED_RUN / "figures").glob("*.png")}
    require(figure_names == REVISED_FIGURES, "Revised figure set is incomplete")
    for figure in sorted((REVISED_RUN / "figures").glob("*.png")):
        width, height, dpi = png_metadata(figure)
        require(width >= 1500 and height >= 1000, f"Revised figure resolution is too small: {figure}")
        require(dpi is not None and dpi >= 299, f"Revised figure is not stored at 300 DPI: {figure}")


def main() -> int:
    validate_notebook()
    validate_revised_run()
    print("Repository validation passed.")
    print("- Notebook: nbformat, syntax, Colab/local setup, and revised pipeline checks")
    print("- Revised run: full-stream provenance, metrics, grouped splits, and 13 figures")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Validation failed: {error}", file=sys.stderr)
        raise
