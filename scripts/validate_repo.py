#!/usr/bin/env python3
"""Fast, offline integrity checks for the notebook and curated reference run."""

from __future__ import annotations

import ast
import csv
import json
import math
import struct
import sys
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "StutterSense.ipynb"
RESULTS = ROOT / "results"
EVENTS = {"prolongation", "block", "sound_rep", "word_rep", "interjection"}
MODELS = {"hard", "soft", "calibrated"}
EXPECTED_FIGURES = {
    "A_dataset_label_frequency.png",
    "B_annotator_disagreement.png",
    "C_training_validation_curves.png",
    "D_per_class_f1.png",
    "E_reliability_diagrams.png",
    "F_risk_coverage.png",
    "G_disagreement_vs_uncertainty.png",
    "H_multilabel_error_analysis.png",
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
    require(len(notebook.cells) == 36, f"Expected 36 notebook cells, found {len(notebook.cells)}")

    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    require(len(code_cells) == 22, f"Expected 22 code cells, found {len(code_cells)}")
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code":
            ast.parse(cell.source, filename=f"{NOTEBOOK.name}:cell-{index}")
            require(cell.execution_count is None, f"Cell {index} has a stored execution count")
            require(not cell.outputs, f"Cell {index} contains stored outputs")

    source = "\n".join(cell.source for cell in notebook.cells)
    lowered = source.lower()
    for forbidden in ("google.colab", "/content/", "stuttersense_colab", "download_output_zip"):
        require(forbidden not in lowered, f"Platform-specific notebook content remains: {forbidden}")
    for required in (
        'DATASET_ID = "psusac/sep28k"',
        'MODEL_NAME = "microsoft/wavlm-base-plus"',
        'OUTPUT_DIR = os.environ.get("STUTTERSENSE_OUTPUT_DIR", "outputs")',
        "RESOLVED_DATASET_REVISION",
        "streaming=True",
        "truncation=False",
        "hard_targets = (counts_matrix >= 2)",
        "soft_targets = counts_matrix / 3.0",
    ):
        require(required in source, f"Required notebook content is missing: {required}")


def validate_results() -> None:
    required_files = {
        "README.md",
        "ambiguity_uncertainty.csv",
        "comparison_table.csv",
        "config.json",
        "dataset_provenance.json",
        "evaluation_metrics.json",
        "experiment_summary.md",
        "materialized_clip_metadata.csv",
        "per_class_metrics.csv",
        "risk_coverage.csv",
        "split_indices.json",
        "thresholds.json",
        "training_histories.json",
    }
    missing = sorted(name for name in required_files if not (RESULTS / name).exists())
    require(not missing, f"Missing curated result files: {missing}")

    comparison = read_csv(RESULTS / "comparison_table.csv")
    require({row["model"] for row in comparison} == MODELS, "Comparison table model set is incorrect")
    numeric_columns = {
        "macro_f1", "micro_f1", "macro_auroc", "macro_average_precision",
        "hard_brier", "soft_brier", "hard_ece", "soft_ece",
        "mean_confidence", "mean_prediction_entropy",
        "uncertainty_ambiguity_spearman", "high_agreement_hamming_error",
        "low_agreement_hamming_error",
    }
    for row in comparison:
        require(row["threshold_policy"] == "fixed_0.5", "Unexpected threshold policy")
        require(all(finite_number(row[column]) for column in numeric_columns), "Non-finite comparison metric")

    per_class = read_csv(RESULTS / "per_class_metrics.csv")
    require(len(per_class) == 15, "Expected 15 per-class result rows")
    require({row["model"] for row in per_class} == MODELS, "Per-class model set is incorrect")
    require({row["event"] for row in per_class} == EVENTS, "Per-class event set is incorrect")

    metadata = read_csv(RESULTS / "materialized_clip_metadata.csv")
    require(len(metadata) == 2000, f"Expected 2,000 metadata rows, found {len(metadata)}")
    hashes = [row["audio_sha256"] for row in metadata]
    require(len(hashes) == len(set(hashes)), "Duplicate audio fingerprints found in reference run")
    for row in metadata:
        for event in EVENTS:
            require(row[event] in {"0", "1", "2", "3"}, f"Invalid annotation count for {event}")

    splits = json.loads((RESULTS / "split_indices.json").read_text(encoding="utf-8"))
    require({name: len(values) for name, values in splits.items()} == {
        "train": 1400, "validation": 300, "test": 300
    }, "Unexpected reference split sizes")
    split_sets = {name: set(values) for name, values in splits.items()}
    require(split_sets["train"].isdisjoint(split_sets["validation"]), "Train/validation overlap")
    require(split_sets["train"].isdisjoint(split_sets["test"]), "Train/test overlap")
    require(split_sets["validation"].isdisjoint(split_sets["test"]), "Validation/test overlap")
    require(set().union(*split_sets.values()) == set(range(2000)), "Splits do not partition all clips")

    provenance = json.loads((RESULTS / "dataset_provenance.json").read_text(encoding="utf-8"))
    require(provenance["materialized_successful_clips"] == 2000, "Incorrect provenance sample count")
    require(provenance["decode_failures"] == 0, "Reference run contains decode failures")

    figure_names = {path.name for path in (RESULTS / "figures").glob("*.png")}
    require(figure_names == EXPECTED_FIGURES, "Figure set does not match the expected research outputs")
    for figure in sorted((RESULTS / "figures").glob("*.png")):
        width, height, dpi = png_metadata(figure)
        require(width >= 1500 and height >= 1000, f"Figure resolution is too small: {figure}")
        require(dpi is not None and dpi >= 299, f"Figure is not stored at 300 DPI: {figure}")


def main() -> int:
    validate_notebook()
    validate_results()
    print("Repository validation passed.")
    print("- Notebook: nbformat, syntax, portability, and required pipeline checks")
    print("- Results: schemas, finite metrics, 2,000-row partition, and 11 high-resolution figures")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"Validation failed: {error}", file=sys.stderr)
        raise
