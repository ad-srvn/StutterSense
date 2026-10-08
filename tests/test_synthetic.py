from __future__ import annotations

import ast
import hashlib
import math
import tempfile
import unittest
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.metrics import (
    average_precision_score, precision_recall_fscore_support, roc_auc_score,
)
from scipy.stats import spearmanr
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
FUNCTIONS = {
    "reservoir_sample",
    "select_verified_group_column",
    "build_dependence_groups",
    "soft_brier_regularized_loss",
    "ambiguity_multitask_loss",
    "training_objective",
    "binary_entropy",
    "expected_calibration_error",
    "select_f1_thresholds",
    "confusion_statistics",
    "evaluate_predictions",
    "reliability_points",
    "risk_coverage_curve",
    "random_risk_curve",
    "paired_group_bootstrap_difference",
    "safe_rank_metric",
    "validate_fluencybank_manifest",
}
CLASSES = {"StutterMLP", "AmbiguityMultiTaskMLP"}


def load_notebook_symbols():
    notebook = nbformat.read(ROOT / "StutterSense.ipynb", as_version=4)
    namespace = {
        "np": np,
        "pd": pd,
        "torch": torch,
        "nn": nn,
        "math": math,
        "hashlib": hashlib,
        "Path": Path,
        "f1_score": f1_score,
        "average_precision_score": average_precision_score,
        "precision_recall_fscore_support": precision_recall_fscore_support,
        "roc_auc_score": roc_auc_score,
        "spearmanr": spearmanr,
        "DROPOUT": 0.2,
        "BRIER_LAMBDA": 0.25,
        "AMBIGUITY_LOSS_WEIGHT": 0.25,
    }
    found = set()
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        tree = ast.parse(cell.source)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in FUNCTIONS | CLASSES:
                module = ast.Module(body=[node], type_ignores=[])
                exec(compile(module, f"notebook::{node.name}", "exec"), namespace)
                found.add(node.name)
    missing = (FUNCTIONS | CLASSES) - found
    if missing:
        raise AssertionError(f"Notebook symbols missing from tests: {sorted(missing)}")
    return namespace


SYMBOLS = load_notebook_symbols()


class SamplingAndGroupingTests(unittest.TestCase):
    def test_reservoir_is_reproducible_and_scans_every_item(self):
        sample_a, scanned_a = SYMBOLS["reservoir_sample"](range(100), 10, seed=7)
        sample_b, scanned_b = SYMBOLS["reservoir_sample"](range(100), 10, seed=7)
        sample_c, _ = SYMBOLS["reservoir_sample"](range(100), 10, seed=8)
        self.assertEqual(sample_a, sample_b)
        self.assertEqual(scanned_a, 100)
        self.assertEqual(scanned_b, 100)
        self.assertEqual(len(sample_a), 10)
        self.assertNotEqual(sample_a, sample_c)
        self.assertEqual([position for position, _ in sample_a], sorted(position for position, _ in sample_a))

    def test_verified_group_selection_never_infers_generic_ids(self):
        frame = pd.DataFrame({"id": range(6), "speaker_id": ["a", "a", "b", "b", "c", "c"]})
        selected = SYMBOLS["select_verified_group_column"](frame, ["speaker_id", "episode_id"])
        self.assertEqual(selected, "speaker_id")
        self.assertIsNone(SYMBOLS["select_verified_group_column"](frame[["id"]], ["speaker_id", "episode_id"]))

    def test_dependence_groups_join_duplicates_and_speakers(self):
        hashes = np.array(["x", "x", "y", "z", "q"], dtype=object)
        speakers = np.array(["a", "b", "b", "c", "d"], dtype=object)
        groups = SYMBOLS["build_dependence_groups"](hashes, speakers)
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[1], groups[2])
        self.assertNotEqual(groups[2], groups[3])


class LossAndModelTests(unittest.TestCase):
    def test_brier_regularization_matches_manual_objective(self):
        logits = torch.tensor([[0.0, 1.0], [-1.0, 0.5]], requires_grad=True)
        targets = torch.tensor([[0.0, 1 / 3], [2 / 3, 1.0]])
        observed = SYMBOLS["soft_brier_regularized_loss"](logits, targets, 0.25)
        expected = nn.functional.binary_cross_entropy_with_logits(logits, targets) + 0.25 * torch.mean((torch.sigmoid(logits) - targets) ** 2)
        torch.testing.assert_close(observed, expected)
        observed.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_ambiguity_multitask_loss_and_head_train(self):
        torch.manual_seed(4)
        model = SYMBOLS["AmbiguityMultiTaskMLP"](8, hidden_dim=4, output_dim=5, dropout=0.0)
        x = torch.randn(6, 8)
        soft = torch.rand(6, 5)
        ambiguity = torch.randint(0, 2, (6, 5)).float()
        event_logits, ambiguity_logits = model(x)
        loss = SYMBOLS["ambiguity_multitask_loss"](event_logits, ambiguity_logits, soft, ambiguity, 0.25)
        before = model.event_head.weight.detach().clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(torch.equal(before, model.event_head.weight))

    def test_all_training_objectives_are_finite(self):
        logits = torch.randn(4, 5, requires_grad=True)
        hard = torch.randint(0, 2, (4, 5)).float()
        soft = torch.rand(4, 5)
        ambiguity = torch.randint(0, 2, (4, 5)).float()
        for variant in ("hard", "soft", "soft_quality_filtered", "brier_ablation"):
            loss = SYMBOLS["training_objective"](logits, hard, soft, ambiguity, variant)
            self.assertTrue(torch.isfinite(loss), variant)
        torch.testing.assert_close(
            SYMBOLS["training_objective"](logits, hard, soft, ambiguity, "hard"),
            nn.functional.binary_cross_entropy_with_logits(logits, hard),
        )
        torch.testing.assert_close(
            SYMBOLS["training_objective"](logits, hard, soft, ambiguity, "soft"),
            nn.functional.binary_cross_entropy_with_logits(logits, soft),
        )
        pair = (logits, torch.randn(4, 5, requires_grad=True))
        loss = SYMBOLS["training_objective"](pair, hard, soft, ambiguity, "ambiguity_multitask")
        self.assertTrue(torch.isfinite(loss))


class MetricTests(unittest.TestCase):
    def test_calibration_and_entropy(self):
        probabilities = np.array([[0.0, 1.0], [0.25, 0.75]])
        self.assertAlmostEqual(SYMBOLS["expected_calibration_error"](probabilities, probabilities), 0.0)
        entropy = SYMBOLS["binary_entropy"](np.array([0.5, 0.01]))
        self.assertGreater(entropy[0], entropy[1])

    def test_threshold_selection_uses_validation_targets(self):
        targets = np.array([[0, 0], [0, 0], [1, 0], [1, 0]])
        probabilities = np.array([[0.1, 0.2], [0.2, 0.3], [0.4, 0.1], [0.8, 0.2]])
        thresholds = SYMBOLS["select_f1_thresholds"](targets, probabilities)
        self.assertAlmostEqual(thresholds[0], 0.4)
        self.assertAlmostEqual(thresholds[1], 0.5)

    def test_confusion_statistics(self):
        stats = SYMBOLS["confusion_statistics"]([0, 0, 1, 1], [0, 1, 0, 1])
        self.assertEqual((stats["tn"], stats["fp"], stats["fn"], stats["tp"]), (1, 1, 1, 1))
        self.assertAlmostEqual(stats["specificity"], 0.5)
        self.assertAlmostEqual(stats["false_negative_rate"], 0.5)

    def test_complete_evaluation_for_perfect_predictions(self):
        namespace = SYMBOLS["evaluate_predictions"].__globals__
        namespace["EVENTS"] = ["event_a", "event_b"]
        namespace["test_idx"] = np.arange(4)
        namespace["hard_targets"] = np.array([[0, 1], [0, 0], [1, 1], [1, 0]])
        namespace["soft_targets"] = namespace["hard_targets"].astype(float)
        namespace["ambiguity_targets"] = np.array([[0, 0], [0, 1], [1, 1], [1, 0]])
        probabilities = np.array([[0.1, 0.9], [0.2, 0.1], [0.8, 0.8], [0.9, 0.2]])
        summary, rows, _, errors = SYMBOLS["evaluate_predictions"](
            7, "synthetic", probabilities, np.array([0.5, 0.5]), "fixed_0.5"
        )
        for metric in ("macro_f1", "micro_f1", "macro_auroc", "macro_average_precision"):
            self.assertAlmostEqual(summary[metric], 1.0)
        for metric in ("hard_brier", "soft_brier", "hard_ece", "soft_ece"):
            self.assertGreaterEqual(summary[metric], 0.0)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["precision"] == row["recall"] == row["f1"] == 1.0 for row in rows))
        self.assertTrue(all(row["fp"] == row["fn"] == 0 for row in rows))
        np.testing.assert_allclose(errors, 0.0)

    def test_risk_coverage_prefers_low_error_samples(self):
        errors = np.array([0.0, 0.0, 1.0, 1.0])
        uncertainty = np.array([0.1, 0.2, 0.8, 0.9])
        risk = SYMBOLS["risk_coverage_curve"](errors, uncertainty, [0.5, 1.0])
        np.testing.assert_allclose(risk, [0.0, 0.5])
        random_a = SYMBOLS["random_risk_curve"](errors, [0.5, 1.0], repeats=20, seed=9)
        random_b = SYMBOLS["random_risk_curve"](errors, [0.5, 1.0], repeats=20, seed=9)
        np.testing.assert_allclose(random_a, random_b)
        self.assertAlmostEqual(random_a[-1], errors.mean())

    def test_reliability_points_preserve_bin_means_and_counts(self):
        predicted, observed, counts = SYMBOLS["reliability_points"](
            np.array([0.1, 0.2, 0.8, 0.9]), np.array([0.0, 0.5, 0.5, 1.0]), n_bins=2
        )
        np.testing.assert_allclose(predicted, [0.15, 0.85])
        np.testing.assert_allclose(observed, [0.25, 0.75])
        np.testing.assert_array_equal(counts, [2, 2])

    def test_paired_group_bootstrap_is_zero_for_identical_models(self):
        y_hard = np.array([[0], [1], [0], [1]])
        y_soft = y_hard.astype(float)
        probabilities = np.array([[0.1], [0.9], [0.2], [0.8]])
        groups = np.array([0, 0, 1, 1])
        result = SYMBOLS["paired_group_bootstrap_difference"](
            y_hard, y_soft, probabilities, probabilities, groups,
            np.array([0.5]), np.array([0.5]), 50, 3,
        )
        for metric in result.values():
            self.assertEqual(metric, {"estimate": 0.0, "lower": 0.0, "upper": 0.0})

    def test_external_manifest_requires_verified_mapping_checksum(self):
        validator = SYMBOLS["validate_fluencybank_manifest"]
        namespace = validator.__globals__
        namespace["EVENTS"] = ["event_a", "event_b"]
        namespace["FLUENCYBANK_SOURCE_REPOSITORY"] = "https://example.test/source"
        namespace["FLUENCYBANK_REQUIRED_COLUMNS"] = [
            "audio_path", "sample_id", "speaker_id", "event_a", "event_b",
            "source_repository", "source_revision", "mapping_path", "mapping_sha256",
            "mapping_verified", "license_acknowledged", "access_basis",
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "clip.wav").write_bytes(b"synthetic audio placeholder")
            (root / "mapping.csv").write_bytes(b"mapping")
            mapping_hash = hashlib.sha256(b"mapping").hexdigest()
            manifest = pd.DataFrame([{
                "audio_path": "clip.wav", "sample_id": "sample-1", "speaker_id": "speaker-1",
                "event_a": 0, "event_b": 2,
                "source_repository": "https://example.test/source",
                "source_revision": "a" * 40, "mapping_path": "mapping.csv",
                "mapping_sha256": mapping_hash, "mapping_verified": True,
                "license_acknowledged": True, "access_basis": "authorized research access",
            }])
            path = root / "manifest.csv"
            manifest.to_csv(path, index=False)
            validated = validator(path)
            self.assertEqual(Path(validated.loc[0, "audio_path"]), (root / "clip.wav").resolve())
            manifest.loc[0, "mapping_sha256"] = "f" * 64
            manifest.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                validator(path)


if __name__ == "__main__":
    unittest.main()
