"""Reusable tables and plots for v2 failure, calibration, and coverage analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from .evaluation import calibration_points, precision_recall_rows, risk_coverage_curve


def false_positive_negative_rows(
    targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: Sequence[float],
    events: Sequence[str],
    sample_ids: Sequence[str],
) -> list[dict]:
    y = np.asarray(targets, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    predicted = p >= np.asarray(thresholds)
    rows = []
    for event_index, event in enumerate(events):
        for index, sample_id in enumerate(sample_ids):
            error_type = None
            if y[index, event_index] == 0 and predicted[index, event_index]:
                error_type = "false_positive"
            elif y[index, event_index] == 1 and not predicted[index, event_index]:
                error_type = "false_negative"
            if error_type:
                rows.append({
                    "sample_id": sample_id,
                    "event": event,
                    "error_type": error_type,
                    "probability": float(p[index, event_index]),
                    "threshold": float(thresholds[event_index]),
                })
    return rows


def stratified_error_rows(
    sample_errors: Sequence[float], metadata: pd.DataFrame, columns: Sequence[str]
) -> list[dict]:
    errors = np.asarray(sample_errors, dtype=float)
    if len(errors) != len(metadata):
        raise ValueError("sample_errors and metadata must align")
    rows = []
    for column in columns:
        if column not in metadata:
            continue
        values = metadata[column]
        if pd.api.types.is_numeric_dtype(values) and values.nunique(dropna=True) > 8:
            strata = pd.qcut(values, q=min(4, values.nunique()), duplicates="drop")
        else:
            strata = values.fillna("missing").astype(str)
        for stratum in pd.unique(strata):
            mask = np.asarray(strata == stratum)
            if mask.any():
                rows.append({
                    "field": column,
                    "stratum": str(stratum),
                    "count": int(mask.sum()),
                    "mean_sample_error": float(errors[mask].mean()),
                })
    return rows


def ambiguity_disagreement_rows(
    annotation_counts: np.ndarray,
    ambiguity_probabilities: np.ndarray | None,
    events: Sequence[str],
) -> list[dict]:
    counts = np.asarray(annotation_counts, dtype=int)
    if counts.ndim != 2 or counts.shape[1] != len(events):
        raise ValueError("annotation counts and events must align")
    ambiguity = np.isin(counts, (1, 2)).astype(int)
    rows = []
    for index, event in enumerate(events):
        row = {
            "event": event,
            "ambiguous_count": int(ambiguity[:, index].sum()),
            "ambiguous_prevalence": float(ambiguity[:, index].mean()),
            "mean_vote_entropy": float(np.mean(_vote_entropy(counts[:, index] / 3.0))),
        }
        if ambiguity_probabilities is not None:
            values = np.asarray(ambiguity_probabilities, dtype=float)
            if values.shape != counts.shape:
                raise ValueError("ambiguity probabilities and counts must align")
            row["mean_predicted_ambiguity"] = float(values[:, index].mean())
        rows.append(row)
    return rows


def _vote_entropy(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def save_precision_recall_plot(
    targets: np.ndarray,
    probabilities: np.ndarray,
    events: Sequence[str],
    destination: str | Path,
) -> None:
    import matplotlib.pyplot as plt

    rows = pd.DataFrame(precision_recall_rows(targets, probabilities, events))
    figure, axis = plt.subplots(figsize=(8, 6))
    for event in events:
        selected = rows[(rows.event == event) & (rows.status == "OK")]
        if not selected.empty:
            axis.plot(selected.recall, selected.precision, label=event)
    axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1))
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(destination, dpi=300)
    plt.close(figure)


def save_calibration_plot(
    hard_targets: np.ndarray,
    soft_targets: np.ndarray,
    probabilities: np.ndarray,
    events: Sequence[str],
    destination: str | Path,
    bins: int = 10,
) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(len(events), 2, figsize=(10, 3 * len(events)), squeeze=False)
    for event_index, event in enumerate(events):
        for target_index, (target_name, targets) in enumerate(
            (("majority vote", hard_targets), ("soft target", soft_targets))
        ):
            points = calibration_points(probabilities[:, event_index], targets[:, event_index], bins)
            axis = axes[event_index, target_index]
            axis.plot([0, 1], [0, 1], "--", color="0.6")
            if points:
                axis.plot(
                    [row["mean_probability"] for row in points],
                    [row["mean_target"] for row in points],
                    marker="o",
                )
            axis.set(title=f"{event} — {target_name}", xlabel="Predicted", ylabel="Observed")
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1)
    figure.tight_layout()
    figure.savefig(destination, dpi=300)
    plt.close(figure)


def save_risk_coverage_plot(
    curves: dict[str, list[dict]], destination: str | Path
) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 6))
    for name, rows in curves.items():
        axis.plot(
            [row["coverage"] for row in rows],
            [row["risk"] for row in rows],
            marker="o",
            label=name,
        )
    axis.set(
        xlabel="Coverage (fraction retained)",
        ylabel="Risk (mean per-sample label error)",
        xlim=(0, 1),
    )
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(destination, dpi=300)
    plt.close(figure)


def build_risk_curves(
    sample_errors: np.ndarray,
    prediction_entropy: np.ndarray,
    ambiguity_scores: np.ndarray | None,
    prevalence_scores: np.ndarray,
    coverages: Sequence[float],
) -> dict[str, list[dict]]:
    curves = {
        "prediction entropy": risk_coverage_curve(sample_errors, prediction_entropy, coverages),
        "prevalence baseline": risk_coverage_curve(sample_errors, prevalence_scores, coverages),
    }
    if ambiguity_scores is not None:
        curves["predicted ambiguity"] = risk_coverage_curve(
            sample_errors, ambiguity_scores, coverages
        )
    return curves
