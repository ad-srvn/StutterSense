from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from stuttersense_v2.config import EVENTS, ExperimentConfig, OutputConfig, load_config, prepare_run_directory
from stuttersense_v2.data import (
    IdentityAudit,
    audit_partitions,
    build_dependence_groups,
    grouped_partition,
    load_identity_manifest,
    reservoir_sample,
)
from stuttersense_v2.evaluation import (
    aggregate_seed_metrics,
    evaluate_multilabel,
    paired_prediction_bootstrap,
    select_f1_thresholds,
)
from stuttersense_v2.inference import preprocess_waveform
from stuttersense_v2.modeling import (
    CHECKPOINT_SCHEMA_VERSION,
    ambiguity_multitask_loss,
    soft_brier_regularized_loss,
    validate_checkpoint_payload,
)


ROOT = Path(__file__).resolve().parents[1]


class ConfigurationAndProvenanceTests(unittest.TestCase):
    def test_full_config_is_pinned_and_deterministic(self):
        first = load_config(ROOT / "configs" / "v2_full.json")
        second = load_config(ROOT / "configs" / "v2_full.json")
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.training.seeds, (42, 123, 456))
        self.assertTrue(first.dataset.revision)
        self.assertTrue(first.model.encoder_revision)

    def test_unique_directory_and_matching_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(
                ExperimentConfig(),
                output=OutputConfig(root=directory, run_name="synthetic"),
            )
            run_dir, resumed = prepare_run_directory(config)
            self.assertFalse(resumed)
            (run_dir / "run_metadata.json").write_text(
                json.dumps({"config_sha256": config.fingerprint}), encoding="utf-8"
            )
            resumed_config = replace(
                config,
                output=OutputConfig(root=directory, run_name="synthetic", resume_from=str(run_dir)),
            )
            observed, resumed = prepare_run_directory(resumed_config)
            self.assertTrue(resumed)
            self.assertEqual(observed, run_dir)
            with self.assertRaises(FileExistsError):
                prepare_run_directory(config)


class SamplingAndLeakageTests(unittest.TestCase):
    def test_reservoir_reproducibility_and_full_scan(self):
        first, first_count = reservoir_sample(range(100), 12, 9)
        second, second_count = reservoir_sample(range(100), 12, 9)
        third, _ = reservoir_sample(range(100), 12, 10)
        self.assertEqual(first, second)
        self.assertEqual(first_count, second_count)
        self.assertEqual(first_count, 100)
        self.assertNotEqual(first, third)

    def test_duplicates_and_verified_identities_form_connected_groups(self):
        groups = build_dependence_groups(
            ["a", "a", "b", "c", "d"], ["s1", "s2", "s2", "s3", "s4"]
        )
        self.assertEqual(groups[0], groups[1])
        self.assertEqual(groups[1], groups[2])
        self.assertNotEqual(groups[2], groups[3])

    def test_grouped_partitions_have_no_overlap(self):
        hashes = np.array(["a", "a", "b", "c", "d", "e", "f", "g"])
        groups = build_dependence_groups(hashes)
        partitions = grouped_partition(groups, (0.5, 0.25, 0.25), seed=4)
        audit = audit_partitions(
            partitions,
            hashes,
            groups,
            IdentityAudit("NOT_VERIFIED", None, None, "synthetic", tuple(), "unknown identity"),
        )
        self.assertEqual(audit.duplicate_partition_overlap, 0)
        self.assertEqual(audit.group_partition_overlap, 0)
        self.assertIn("NOT VERIFIED", audit.unresolved_leakage_risk)

    def test_identity_manifest_rejects_unverified_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.csv"
            path.write_text(
                "sample_id,group_id,group_kind,identity_source,identity_revision,identity_verified\n"
                "one,speaker-a,speaker,source,rev,false\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "identity_verified=true"):
                load_identity_manifest(path, ["one"])


class ObjectiveAndEvaluationTests(unittest.TestCase):
    def test_loss_definitions_match_manual_values(self):
        logits = torch.tensor([[0.0, 1.0]], requires_grad=True)
        soft = torch.tensor([[1 / 3, 2 / 3]])
        observed = soft_brier_regularized_loss(logits, soft, 0.25)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, soft)
        expected += 0.25 * torch.mean((torch.sigmoid(logits) - soft) ** 2)
        torch.testing.assert_close(observed, expected)
        ambiguity_logits = torch.tensor([[0.5, -0.5]], requires_grad=True)
        ambiguity = torch.tensor([[1.0, 0.0]])
        multi = ambiguity_multitask_loss(logits, ambiguity_logits, soft, ambiguity, 0.25)
        self.assertTrue(torch.isfinite(multi))

    def test_threshold_selection_is_validation_only(self):
        targets = np.array([[0], [0], [1], [1]])
        probabilities = np.array([[0.1], [0.2], [0.4], [0.8]])
        selected = select_f1_thresholds(targets, probabilities, fit_split="validation")
        self.assertGreaterEqual(selected.thresholds[0], 0.2)
        self.assertLessEqual(selected.thresholds[0], 0.4)
        with self.assertRaisesRegex(ValueError, "validation"):
            select_f1_thresholds(targets, probabilities, fit_split="test")

    def test_absent_class_metrics_are_explicitly_undefined(self):
        hard = np.zeros((4, 1))
        soft = np.zeros((4, 1))
        probabilities = np.array([[0.1], [0.2], [0.3], [0.4]])
        summary, rows = evaluate_multilabel(hard, soft, probabilities, [0.5], ["event"])
        self.assertIsNone(rows[0]["auroc"])
        self.assertEqual(rows[0]["auroc_status"], "UNDEFINED_ABSENT_CLASS")
        self.assertEqual(summary["macro_auroc_defined_events"], 0)

    def test_seed_aggregation_is_descriptive(self):
        rows = [
            {"model": "hard", "threshold_policy": "validation_tuned", "macro_f1": 0.4},
            {"model": "hard", "threshold_policy": "validation_tuned", "macro_f1": 0.5},
        ]
        result = aggregate_seed_metrics(rows, ["macro_f1"])[0]
        self.assertAlmostEqual(result["macro_f1_mean"], 0.45)
        self.assertIn("not a significance test", result["inference"])

    def test_paired_bootstrap_retains_unverified_identity_warning(self):
        hard = np.array([[0], [0], [1], [1]])
        soft = hard.astype(float)
        probabilities = np.array([[0.1], [0.2], [0.8], [0.9]])
        rows = paired_prediction_bootstrap(
            hard, soft, probabilities, probabilities, [0.5], [0.5],
            [0, 0, 1, 1], 20, 7, 0.95, "NOT_VERIFIED",
        )
        self.assertTrue(all(row["estimate"] == 0 for row in rows))
        self.assertTrue(all(row["resampling_unit_status"] == "NOT_VERIFIED_IDENTITY_INDEPENDENCE" for row in rows))


class CheckpointAndPreprocessingTests(unittest.TestCase):
    def _payload(self):
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_state_dict": {},
            "objective": "hard",
            "input_dim": 8,
            "hidden_dim": 4,
            "dropout": 0.2,
            "events": list(EVENTS),
            "sample_rate": 16_000,
            "encoder_name": "encoder",
            "encoder_revision": "a" * 40,
            "dataset_id": "dataset",
            "dataset_revision": "b" * 40,
            "split_fingerprint": "c" * 64,
            "seed": 42,
            "ambiguity_supported": False,
            "preprocessing": "mono-float32-resample-v1",
        }

    def test_checkpoint_label_order_and_ambiguity_are_validated(self):
        payload = self._payload()
        self.assertIs(validate_checkpoint_payload(payload, EVENTS), payload)
        payload["events"] = list(reversed(EVENTS))
        with self.assertRaisesRegex(ValueError, "label order"):
            validate_checkpoint_payload(payload, EVENTS)

    def test_preprocessing_mixes_channels_and_resamples(self):
        stereo = np.vstack([np.ones(800), np.zeros(800)])
        output = preprocess_waveform(stereo, 8_000, 16_000)
        self.assertEqual(output.dtype, np.float32)
        self.assertGreaterEqual(len(output), 1_598)
        self.assertLessEqual(len(output), 1_602)
        self.assertAlmostEqual(float(output.mean()), 0.5, places=2)

    def test_preprocessing_rejects_overlong_clips(self):
        with self.assertRaisesRegex(ValueError, "chunk"):
            preprocess_waveform(np.zeros(16_001), 16_000, 16_000, 1.0)


if __name__ == "__main__":
    unittest.main()
