"""Thresholding, metrics, calibration, risk coverage, and paired uncertainty."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


@dataclass(frozen=True)
class MetricValue:
    value: float | None
    status: str

    def to_dict(self) -> dict[str, float | str | None]:
        return asdict(self)


@dataclass(frozen=True)
class ThresholdSelection:
    thresholds: np.ndarray
    fit_split: str
    objective: str


def _arrays(targets: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(targets)
    p = np.asarray(probabilities, dtype=float)
    if y.shape != p.shape or y.ndim != 2:
        raise ValueError("Targets and probabilities must be aligned two-dimensional arrays")
    if np.any((p < 0) | (p > 1)) or not np.isfinite(p).all():
        raise ValueError("Probabilities must be finite and lie in [0, 1]")
    return y, p


def safe_rank_metric(function: Callable, y_true: np.ndarray, scores: np.ndarray) -> MetricValue:
    if np.unique(y_true).size < 2:
        return MetricValue(None, "UNDEFINED_ABSENT_CLASS")
    try:
        return MetricValue(float(function(y_true, scores)), "OK")
    except ValueError as error:
        return MetricValue(None, f"UNDEFINED:{error}")


def binary_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int | None]:
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y == 1) & (pred == 1)))
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))

    def divide(numerator: int, denominator: int) -> float | None:
        return float(numerator / denominator) if denominator else None

    precision = divide(tp, tp + fp)
    recall = divide(tp, tp + fn)
    f1 = None if precision is None or recall is None else (
        0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    )
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
        "specificity": divide(tn, tn + fp), "npv": divide(tn, tn + fn),
        "false_positive_rate": divide(fp, fp + tn),
        "false_negative_rate": divide(fn, fn + tp),
    }


def expected_calibration_error(
    probabilities: np.ndarray, targets: np.ndarray, n_bins: int = 10
) -> float:
    p = np.asarray(probabilities, dtype=float).ravel()
    y = np.asarray(targets, dtype=float).ravel()
    if p.shape != y.shape or n_bins < 1:
        raise ValueError("Probabilities and targets must align; n_bins must be positive")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.minimum(np.digitize(p, edges[1:-1]), n_bins - 1)
    total = 0.0
    for index in range(n_bins):
        selected = bin_ids == index
        if selected.any():
            total += selected.mean() * abs(p[selected].mean() - y[selected].mean())
    return float(total)


def calibration_points(
    probabilities: np.ndarray, targets: np.ndarray, n_bins: int = 10
) -> list[dict[str, float | int]]:
    p = np.asarray(probabilities, dtype=float).ravel()
    y = np.asarray(targets, dtype=float).ravel()
    if p.shape != y.shape:
        raise ValueError("Probabilities and targets must align")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.minimum(np.digitize(p, edges[1:-1]), n_bins - 1)
    rows = []
    for index in range(n_bins):
        selected = bin_ids == index
        if selected.any():
            rows.append({
                "bin": index,
                "count": int(selected.sum()),
                "mean_probability": float(p[selected].mean()),
                "mean_target": float(y[selected].mean()),
            })
    return rows


def select_f1_thresholds(
    validation_targets: np.ndarray,
    validation_probabilities: np.ndarray,
    grid_size: int = 201,
    fit_split: str = "validation",
) -> ThresholdSelection:
    if fit_split != "validation":
        raise ValueError("Thresholds may only be selected on the validation split")
    y, p = _arrays(validation_targets, validation_probabilities)
    grid = np.linspace(0.0, 1.0, grid_size)
    selected = []
    for event_index in range(y.shape[1]):
        if np.unique(y[:, event_index]).size < 2:
            selected.append(0.5)
            continue
        candidates = np.unique(np.concatenate((grid, p[:, event_index], [0.5])))
        scores = []
        for threshold in candidates:
            value = binary_confusion(y[:, event_index], p[:, event_index] >= threshold)["f1"]
            scores.append(-np.inf if value is None else value)
        best = candidates[np.flatnonzero(np.isclose(scores, np.max(scores)))]
        selected.append(float(sorted(best, key=lambda value: (abs(value - 0.5), -value))[0]))
    return ThresholdSelection(np.asarray(selected), fit_split, "maximum per-event F1")


def evaluate_multilabel(
    hard_targets: np.ndarray,
    soft_targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: Sequence[float],
    events: Sequence[str],
    calibration_bins: int = 10,
) -> tuple[dict, list[dict]]:
    hard, probabilities = _arrays(hard_targets, probabilities)
    soft = np.asarray(soft_targets, dtype=float)
    if soft.shape != hard.shape or len(events) != hard.shape[1]:
        raise ValueError("Targets, events, and probabilities must align")
    threshold_array = np.asarray(thresholds, dtype=float)
    if threshold_array.shape != (hard.shape[1],):
        raise ValueError("One threshold is required per event")
    predicted = probabilities >= threshold_array
    rows = []
    for index, event in enumerate(events):
        confusion = binary_confusion(hard[:, index], predicted[:, index])
        auroc = safe_rank_metric(roc_auc_score, hard[:, index], probabilities[:, index])
        ap = safe_rank_metric(average_precision_score, hard[:, index], probabilities[:, index])
        rows.append({
            "event": event,
            "threshold": float(threshold_array[index]),
            **confusion,
            "support": int(hard[:, index].sum()),
            "auroc": auroc.value,
            "auroc_status": auroc.status,
            "average_precision": ap.value,
            "average_precision_status": ap.status,
            "hard_brier": float(np.mean((probabilities[:, index] - hard[:, index]) ** 2)),
            "soft_brier": float(np.mean((probabilities[:, index] - soft[:, index]) ** 2)),
            "hard_ece": expected_calibration_error(
                probabilities[:, index], hard[:, index], calibration_bins
            ),
            "soft_ece": expected_calibration_error(
                probabilities[:, index], soft[:, index], calibration_bins
            ),
        })
    available_f1 = [row["f1"] for row in rows if row["f1"] is not None]
    available_auroc = [row["auroc"] for row in rows if row["auroc"] is not None]
    available_ap = [row["average_precision"] for row in rows if row["average_precision"] is not None]
    micro = binary_confusion(hard.ravel(), predicted.ravel())
    summary = {
        "macro_f1": float(np.mean(available_f1)) if available_f1 else None,
        "macro_f1_defined_events": len(available_f1),
        "micro_f1": micro["f1"],
        "macro_auroc": float(np.mean(available_auroc)) if available_auroc else None,
        "macro_auroc_defined_events": len(available_auroc),
        "macro_average_precision": float(np.mean(available_ap)) if available_ap else None,
        "macro_ap_defined_events": len(available_ap),
        "hard_brier": float(np.mean((probabilities - hard) ** 2)),
        "soft_brier": float(np.mean((probabilities - soft) ** 2)),
        "hard_ece": expected_calibration_error(probabilities, hard, calibration_bins),
        "soft_ece": expected_calibration_error(probabilities, soft, calibration_bins),
        "sample_count": len(hard),
    }
    return summary, rows


def precision_recall_rows(
    targets: np.ndarray, probabilities: np.ndarray, events: Sequence[str]
) -> list[dict]:
    y, p = _arrays(targets, probabilities)
    rows = []
    for index, event in enumerate(events):
        if np.unique(y[:, index]).size < 2:
            rows.append({"event": event, "status": "UNDEFINED_ABSENT_CLASS"})
            continue
        precision, recall, thresholds = precision_recall_curve(y[:, index], p[:, index])
        padded = np.concatenate((thresholds, [np.nan]))
        rows.extend({
            "event": event,
            "status": "OK",
            "precision": float(precision[position]),
            "recall": float(recall[position]),
            "threshold": None if np.isnan(padded[position]) else float(padded[position]),
        } for position in range(len(precision)))
    return rows


def binary_entropy(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-7, 1 - 1e-7)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def risk_coverage_curve(
    sample_errors: Sequence[float], rejection_scores: Sequence[float], coverages: Sequence[float]
) -> list[dict[str, float | int]]:
    errors = np.asarray(sample_errors, dtype=float)
    scores = np.asarray(rejection_scores, dtype=float)
    if errors.shape != scores.shape or errors.ndim != 1:
        raise ValueError("Errors and rejection scores must be aligned vectors")
    order = np.argsort(scores, kind="stable")
    rows = []
    for coverage in coverages:
        if not 0 < coverage <= 1:
            raise ValueError("Coverage values must lie in (0, 1]")
        kept = max(1, int(np.ceil(coverage * len(errors))))
        rows.append({
            "coverage": float(coverage),
            "retained": kept,
            "risk": float(errors[order[:kept]].mean()),
        })
    return rows


def random_risk_coverage(
    sample_errors: Sequence[float], coverages: Sequence[float], repeats: int, seed: int
) -> list[dict[str, float | int]]:
    errors = np.asarray(sample_errors, dtype=float)
    rng = np.random.default_rng(seed)
    collected = {float(coverage): [] for coverage in coverages}
    for _ in range(repeats):
        scores = rng.random(len(errors))
        for row in risk_coverage_curve(errors, scores, coverages):
            collected[float(row["coverage"])].append(float(row["risk"]))
    return [{
        "coverage": coverage,
        "risk": float(np.mean(values)),
        "risk_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "repeats": repeats,
    } for coverage, values in collected.items()]


def prevalence_rejection_score(probabilities: np.ndarray, training_prevalence: np.ndarray) -> np.ndarray:
    """Baseline score for low-prevalence predictions, separate from ambiguity."""
    p = np.asarray(probabilities, dtype=float)
    prevalence = np.asarray(training_prevalence, dtype=float)
    if p.ndim != 2 or prevalence.shape != (p.shape[1],):
        raise ValueError("Training prevalence must provide one value per event")
    weights = 1.0 - np.clip(prevalence, 0.0, 1.0)
    return np.average(p, axis=1, weights=np.maximum(weights, 1e-8))


def paired_group_bootstrap(
    metric: Callable[[np.ndarray], float],
    reference_values: np.ndarray,
    candidate_values: np.ndarray,
    groups: Sequence[str | int],
    replicates: int,
    seed: int,
    confidence: float,
    identity_status: str,
) -> dict[str, float | str | None]:
    """Paired group bootstrap; labels identity uncertainty instead of hiding it."""
    reference = np.asarray(reference_values)
    candidate = np.asarray(candidate_values)
    group_array = np.asarray(groups)
    if len(reference) != len(candidate) or len(reference) != len(group_array):
        raise ValueError("Paired arrays and groups must align")
    unique = np.unique(group_array)
    members = {group: np.flatnonzero(group_array == group) for group in unique}
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(replicates):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([members[group] for group in sampled])
        deltas.append(float(metric(candidate[indices]) - metric(reference[indices])))
    alpha = (1.0 - confidence) / 2.0
    verified = identity_status.startswith("VERIFIED")
    return {
        "estimate": float(metric(candidate) - metric(reference)),
        "lower": float(np.quantile(deltas, alpha)),
        "upper": float(np.quantile(deltas, 1.0 - alpha)),
        "replicates": replicates,
        "resampling_unit_status": "VERIFIED_DEPENDENCE_GROUP" if verified else "NOT_VERIFIED_IDENTITY_INDEPENDENCE",
        "interpretation": "interval is descriptive only" if not verified else "paired group bootstrap interval",
    }


def paired_prediction_bootstrap(
    hard_targets: np.ndarray,
    soft_targets: np.ndarray,
    reference_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    reference_thresholds: Sequence[float],
    candidate_thresholds: Sequence[float],
    groups: Sequence[str | int],
    replicates: int,
    seed: int,
    confidence: float,
    identity_status: str,
) -> list[dict]:
    """Paired macro-F1 and soft-Brier intervals using whole dependence groups."""
    hard, reference = _arrays(hard_targets, reference_probabilities)
    soft = np.asarray(soft_targets, dtype=float)
    _, candidate = _arrays(hard_targets, candidate_probabilities)
    group_array = np.asarray(groups)
    if soft.shape != hard.shape or len(group_array) != len(hard):
        raise ValueError("Targets, predictions, and groups must align")
    reference_thresholds = np.asarray(reference_thresholds, dtype=float)
    candidate_thresholds = np.asarray(candidate_thresholds, dtype=float)

    def macro_f1(indices: np.ndarray, probabilities: np.ndarray, thresholds: np.ndarray) -> float:
        values = []
        predicted = probabilities[indices] >= thresholds
        for event_index in range(hard.shape[1]):
            value = binary_confusion(hard[indices, event_index], predicted[:, event_index])["f1"]
            if value is not None:
                values.append(value)
        return float(np.mean(values)) if values else np.nan

    def soft_brier(indices: np.ndarray, probabilities: np.ndarray) -> float:
        return float(np.mean((probabilities[indices] - soft[indices]) ** 2))

    unique = np.unique(group_array)
    members = {group: np.flatnonzero(group_array == group) for group in unique}
    rng = np.random.default_rng(seed)
    deltas = {"macro_f1": [], "soft_brier": []}
    for _ in range(replicates):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([members[group] for group in sampled])
        deltas["macro_f1"].append(
            macro_f1(indices, candidate, candidate_thresholds)
            - macro_f1(indices, reference, reference_thresholds)
        )
        deltas["soft_brier"].append(
            soft_brier(indices, candidate) - soft_brier(indices, reference)
        )
    full = np.arange(len(hard))
    estimates = {
        "macro_f1": macro_f1(full, candidate, candidate_thresholds)
        - macro_f1(full, reference, reference_thresholds),
        "soft_brier": soft_brier(full, candidate) - soft_brier(full, reference),
    }
    alpha = (1.0 - confidence) / 2.0
    verified = identity_status.startswith("VERIFIED")
    rows = []
    for metric_name, values in deltas.items():
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        rows.append({
            "metric": metric_name,
            "estimate": estimates[metric_name],
            "lower": float(np.quantile(finite, alpha)) if len(finite) else None,
            "upper": float(np.quantile(finite, 1.0 - alpha)) if len(finite) else None,
            "valid_replicates": int(len(finite)),
            "requested_replicates": replicates,
            "resampling_unit_status": (
                "VERIFIED_DEPENDENCE_GROUP" if verified
                else "NOT_VERIFIED_IDENTITY_INDEPENDENCE"
            ),
            "interpretation": (
                "paired group bootstrap interval" if verified else "descriptive interval only"
            ),
        })
    return rows


def group_bootstrap_metric_intervals(
    hard_targets: np.ndarray,
    soft_targets: np.ndarray,
    probabilities: np.ndarray,
    thresholds: Sequence[float],
    events: Sequence[str],
    groups: Sequence[str | int],
    replicates: int,
    seed: int,
    confidence: float,
    identity_status: str,
    calibration_bins: int = 10,
) -> list[dict]:
    """Absolute metric intervals that resample complete known dependence groups."""
    hard, probabilities = _arrays(hard_targets, probabilities)
    soft = np.asarray(soft_targets, dtype=float)
    group_array = np.asarray(groups)
    if soft.shape != hard.shape or len(group_array) != len(hard):
        raise ValueError("Targets, predictions, and groups must align")
    metric_names = (
        "macro_f1", "micro_f1", "macro_auroc", "macro_average_precision",
        "hard_brier", "soft_brier", "hard_ece", "soft_ece",
    )
    point, _ = evaluate_multilabel(
        hard, soft, probabilities, thresholds, events, calibration_bins
    )
    unique = np.unique(group_array)
    members = {group: np.flatnonzero(group_array == group) for group in unique}
    rng = np.random.default_rng(seed)
    distributions = {name: [] for name in metric_names}
    for _ in range(replicates):
        sampled = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([members[group] for group in sampled])
        summary, _ = evaluate_multilabel(
            hard[indices], soft[indices], probabilities[indices],
            thresholds, events, calibration_bins,
        )
        for name in metric_names:
            if summary[name] is not None and np.isfinite(summary[name]):
                distributions[name].append(float(summary[name]))
    alpha = (1.0 - confidence) / 2.0
    verified = identity_status.startswith("VERIFIED")
    rows = []
    for name in metric_names:
        values = np.asarray(distributions[name], dtype=float)
        rows.append({
            "metric": name,
            "estimate": point[name],
            "lower": float(np.quantile(values, alpha)) if len(values) else None,
            "upper": float(np.quantile(values, 1.0 - alpha)) if len(values) else None,
            "valid_replicates": int(len(values)),
            "requested_replicates": replicates,
            "resampling_unit_status": (
                "VERIFIED_DEPENDENCE_GROUP" if verified
                else "NOT_VERIFIED_IDENTITY_INDEPENDENCE"
            ),
            "interpretation": (
                "group bootstrap interval" if verified else "descriptive interval only"
            ),
        })
    return rows


def aggregate_seed_metrics(rows: Sequence[dict], metric_names: Sequence[str]) -> list[dict]:
    """Summarize variability across seeds without treating seeds as independent samples."""
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["model"], row["threshold_policy"]), []).append(row)
    output = []
    for (model, policy), group in sorted(grouped.items()):
        result = {
            "model": model,
            "threshold_policy": policy,
            "seed_count": len(group),
            "inference": "descriptive seed variability; not a significance test",
        }
        for metric_name in metric_names:
            values = np.asarray([
                row[metric_name] for row in group if row.get(metric_name) is not None
            ], dtype=float)
            result[f"{metric_name}_mean"] = float(values.mean()) if len(values) else None
            result[f"{metric_name}_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
            result[f"{metric_name}_defined_seeds"] = int(len(values))
        output.append(result)
    return output
