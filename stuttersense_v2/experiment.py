"""StutterSense 2.0 experiment orchestration.

All generated runs remain scientifically unvalidated until independently reviewed.
The validated 2026-10-08 baseline under ``results/`` is read-only.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import os
import platform
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .config import ExperimentConfig, prepare_run_directory
from .data import (
    IdentityAudit,
    audit_dataset_identity,
    audit_partitions,
    build_dependence_groups,
    grouped_partition,
    load_identity_manifest,
    reservoir_sample,
    waveform_sha256,
)
from .evaluation import (
    aggregate_seed_metrics,
    binary_entropy,
    evaluate_multilabel,
    group_bootstrap_metric_intervals,
    paired_prediction_bootstrap,
    prevalence_rejection_score,
    random_risk_coverage,
    risk_coverage_curve,
    safe_rank_metric,
    select_f1_thresholds,
)
from .inference import preprocess_waveform
from .modeling import CheckpointMetadata, build_classifier, training_objective, validate_checkpoint_payload
from .reporting import (
    ambiguity_disagreement_rows,
    build_risk_curves,
    false_positive_negative_rows,
    save_calibration_plot,
    save_precision_recall_plot,
    save_risk_coverage_plot,
    stratified_error_rows,
)


DIAGNOSTIC_COLUMNS = (
    "unsure", "poor_audio_quality", "no_speech", "difficult_to_understand",
    "no_stuttered_words", "natural_pause", "music",
)
SAMPLE_ID_COLUMNS = ("clip_id", "sample_id", "file_id", "utterance_id")
SUMMARY_METRICS = (
    "macro_f1", "micro_f1", "macro_auroc", "macro_average_precision",
    "hard_brier", "soft_brier", "hard_ece", "soft_ece",
)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _hash_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment() -> dict[str, Any]:
    packages = (
        "torch", "transformers", "datasets", "huggingface-hub", "numpy", "scipy",
        "scikit-learn", "pandas", "soundfile", "matplotlib",
    )
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": versions,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def _set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not (
        getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")
    return device


def _decode_audio(value: Any, target_rate: int) -> tuple[np.ndarray, int]:
    if hasattr(value, "get_all_samples"):
        decoded = value.get_all_samples()
        values = decoded.data.detach().cpu().numpy() if torch.is_tensor(decoded.data) else decoded.data
        source_rate = int(decoded.sample_rate)
    elif isinstance(value, dict) and value.get("array") is not None:
        values, source_rate = value["array"], int(value["sampling_rate"])
    else:
        source = None
        if isinstance(value, dict):
            source = io.BytesIO(value["bytes"]) if value.get("bytes") is not None else value.get("path")
        elif isinstance(value, (str, os.PathLike, bytes, bytearray)):
            source = io.BytesIO(value) if isinstance(value, (bytes, bytearray)) else value
        if source is None:
            raise TypeError(f"Unsupported audio value: {type(value)!r}")
        values, source_rate = sf.read(source, dtype="float32", always_2d=False)
    return preprocess_waveform(values, source_rate, target_rate, None), source_rate


def _load_sample(config: ExperimentConfig) -> tuple[pd.DataFrame, list[np.ndarray], dict, Any]:
    from datasets import Audio, load_dataset
    from huggingface_hub import HfApi

    api = HfApi()
    dataset_info = api.dataset_info(config.dataset.dataset_id, revision=config.dataset.revision)
    model_info = api.model_info(config.model.encoder_name, revision=config.model.encoder_revision)
    if dataset_info.sha != config.dataset.revision:
        raise RuntimeError("Resolved dataset revision differs from the configured revision")
    if model_info.sha != config.model.encoder_revision:
        raise RuntimeError("Resolved model revision differs from the configured revision")

    stream = load_dataset(
        config.dataset.dataset_id,
        config.dataset.subset,
        split=config.dataset.split,
        revision=config.dataset.revision,
        streaming=True,
    )
    if "audio" not in stream.column_names:
        raise ValueError("Dataset schema does not contain an audio column")
    stream = stream.cast_column("audio", Audio(decode=False))
    smoke = config.mode == "smoke"
    sample_size = min(config.sampling.sample_size or 64, 64) if smoke else config.sampling.sample_size
    scan_limit = config.sampling.scan_limit
    if smoke and scan_limit is None:
        scan_limit = 512
    sampled, scanned = reservoir_sample(
        stream, sample_size, config.sampling.reservoir_seed, scan_limit
    )
    rows: list[dict] = []
    waveforms: list[np.ndarray] = []
    failures: list[dict] = []
    available_columns = tuple(stream.column_names)
    for stream_position, example in sampled:
        try:
            counts = np.asarray([example[event] for event in config.events], dtype=float)
            if not np.isfinite(counts).all() or not np.allclose(counts, np.round(counts)):
                raise ValueError("Event counts must be finite integers")
            if np.any((counts < 0) | (counts > 3)):
                raise ValueError("Event counts must lie in [0, 3]")
            waveform, original_rate = _decode_audio(example["audio"], config.dataset.sample_rate)
            audio_hash = waveform_sha256(waveform)
            source_id_column = next(
                (column for column in SAMPLE_ID_COLUMNS if column in available_columns and example.get(column) is not None),
                None,
            )
            row = {
                "sample_id": str(example[source_id_column]) if source_id_column else f"sha256:{audio_hash}",
                "sample_id_source": source_id_column or "derived audio content key; not a source identity",
                "stream_position": int(stream_position),
                "audio_sha256": audio_hash,
                "original_sample_rate": int(original_rate),
                "num_samples": int(waveform.size),
                "duration_seconds": float(waveform.size / config.dataset.sample_rate),
                "rms_amplitude": float(np.sqrt(np.mean(np.square(waveform, dtype=np.float64)))),
                "peak_amplitude": float(np.max(np.abs(waveform))),
                "clipping_fraction": float(np.mean(np.abs(waveform) >= 0.999)),
            }
            row.update({event: int(example[event]) for event in config.events})
            for column in DIAGNOSTIC_COLUMNS:
                if column in available_columns and example.get(column) is not None:
                    row[column] = int(example[column])
            if config.dataset.verified_group_column in available_columns:
                row[config.dataset.verified_group_column] = str(example[config.dataset.verified_group_column])
            rows.append(row)
            waveforms.append(waveform)
        except Exception as error:
            failures.append({"stream_position": int(stream_position), "error": repr(error)})
    if not rows:
        raise RuntimeError("No sampled audio decoded successfully")
    metadata = pd.DataFrame(rows)
    sampling = {
        "algorithm": "Algorithm R reservoir sampling without replacement",
        "seed": config.sampling.reservoir_seed,
        "examples_scanned": scanned,
        "scan_limit": scan_limit,
        "requested_samples": sample_size,
        "decoded_samples": len(rows),
        "decode_failures": len(failures),
        "full_stream": scan_limit is None,
    }
    provenance = {
        "dataset_id": config.dataset.dataset_id,
        "dataset_revision": dataset_info.sha,
        "model_id": config.model.encoder_name,
        "model_revision": model_info.sha,
        "available_columns": list(available_columns),
        "sampling": sampling,
        "decode_failures": failures,
    }
    return metadata, waveforms, provenance, stream.features


def _identity_and_groups(
    config: ExperimentConfig, metadata: pd.DataFrame, available_columns: Sequence[str]
) -> tuple[np.ndarray, IdentityAudit]:
    if config.dataset.metadata_manifest:
        identities, audit = load_identity_manifest(
            config.dataset.metadata_manifest, metadata["sample_id"].astype(str)
        )
    else:
        audit = audit_dataset_identity(
            available_columns,
            config.dataset.verified_group_kind,
            config.dataset.verified_group_column,
            config.dataset.identity_source,
            config.dataset.identity_revision,
        )
        identities = (
            metadata[audit.group_column].to_numpy(dtype=str)
            if audit.status.startswith("VERIFIED") and audit.group_column else None
        )
    groups = build_dependence_groups(metadata["audio_sha256"], identities)
    return groups, audit


def _embedding_cache_key(config: ExperimentConfig, metadata: pd.DataFrame) -> str:
    return _hash_json({
        "encoder": config.model.encoder_name,
        "revision": config.model.encoder_revision,
        "sample_rate": config.dataset.sample_rate,
        "fingerprints": metadata["audio_sha256"].tolist(),
    })


def _extract_embeddings(
    config: ExperimentConfig,
    waveforms: list[np.ndarray],
    metadata: pd.DataFrame,
    run_dir: Path,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    cache_key = _embedding_cache_key(config, metadata)
    cache_path = run_dir / "cache" / f"wavlm-{cache_key[:16]}.npz"
    cache_meta = cache_path.with_suffix(".json")
    if config.model.reuse_embeddings and cache_path.is_file() and cache_meta.is_file():
        recorded = json.loads(cache_meta.read_text(encoding="utf-8"))
        if recorded.get("cache_key") != cache_key:
            raise ValueError("Embedding cache metadata mismatch")
        if recorded.get("sha256") != _file_sha256(cache_path):
            raise ValueError("Embedding cache checksum mismatch")
        values = np.load(cache_path)["embeddings"]
        if len(values) != len(metadata):
            raise ValueError("Embedding cache sample count mismatch")
        return np.asarray(values, dtype=np.float32)

    from transformers import AutoFeatureExtractor, WavLMModel

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        config.model.encoder_name, revision=config.model.encoder_revision
    )
    encoder = WavLMModel.from_pretrained(
        config.model.encoder_name, revision=config.model.encoder_revision
    )
    encoder.requires_grad_(False).eval().to(device)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(waveforms), batch_size):
            encoded = feature_extractor(
                waveforms[start : start + batch_size],
                sampling_rate=config.dataset.sample_rate,
                padding=True,
                truncation=False,
                return_attention_mask=True,
                return_tensors="pt",
            )
            values = encoded["input_values"].to(device)
            mask = encoded.get("attention_mask", torch.ones_like(values, dtype=torch.long)).to(device)
            hidden = encoder(input_values=values, attention_mask=mask).last_hidden_state
            frame_mask = encoder._get_feature_vector_attention_mask(hidden.shape[1], mask)
            weights = frame_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)
            chunks.append(pooled.float().cpu().numpy())
    embeddings = np.concatenate(chunks).astype(np.float32)
    if config.model.reuse_embeddings:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, embeddings=embeddings)
        _write_json(cache_meta, {
            "cache_key": cache_key,
            "sha256": _file_sha256(cache_path),
            "shape": list(embeddings.shape),
            "dtype": str(embeddings.dtype),
            "scientific_validation": "UNVALIDATED",
        })
    encoder.cpu()
    return embeddings


def _split_fingerprint(partitions: dict[str, np.ndarray], metadata: pd.DataFrame) -> str:
    return _hash_json({
        name: metadata.iloc[indices]["audio_sha256"].tolist()
        for name, indices in partitions.items()
    })


def _load_checkpoint_model(path: Path, config: ExperimentConfig, device: torch.device) -> nn.Module:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_checkpoint_payload(payload, tuple(config.events))
    if payload["dataset_revision"] != config.dataset.revision:
        raise ValueError("Checkpoint dataset revision mismatch")
    if payload["encoder_revision"] != config.model.encoder_revision:
        raise ValueError("Checkpoint encoder revision mismatch")
    model = build_classifier(
        payload["objective"], payload["input_dim"], payload["hidden_dim"],
        len(config.events), payload["dropout"],
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return model.eval().to(device)


def _train_model(
    config: ExperimentConfig,
    variant: str,
    seed: int,
    embeddings: np.ndarray,
    hard: np.ndarray,
    soft: np.ndarray,
    ambiguity: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    split_fingerprint: str,
    checkpoint_path: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
) -> tuple[nn.Module, list[dict]]:
    if checkpoint_path.is_file():
        model = _load_checkpoint_model(checkpoint_path, config, device)
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if payload["seed"] != seed or payload["split_fingerprint"] != split_fingerprint:
            raise ValueError("Resume refused: checkpoint seed or split fingerprint mismatch")
        return model, payload.get("history", [])

    _set_seed(seed)
    model = build_classifier(
        variant, embeddings.shape[1], config.model.hidden_dim,
        len(config.events), config.model.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)

    def loader(indices: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(embeddings[indices]).float(),
            torch.from_numpy(hard[indices]).float(),
            torch.from_numpy(soft[indices]).float(),
            torch.from_numpy(ambiguity[indices]).float(),
        )
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          generator=generator if shuffle else None, num_workers=0)

    train_loader = loader(train_indices, True)
    validation_loader = loader(validation_indices, False)
    best_state = None
    best_validation = float("inf")
    stale = 0
    history = []
    for epoch in range(epochs):
        losses = {}
        for phase, phase_loader in (("train", train_loader), ("validation", validation_loader)):
            model.train(phase == "train")
            total, count = 0.0, 0
            context = torch.enable_grad() if phase == "train" else torch.inference_mode()
            with context:
                for x, hard_y, soft_y, ambiguity_y in phase_loader:
                    x, hard_y = x.to(device), hard_y.to(device)
                    soft_y, ambiguity_y = soft_y.to(device), ambiguity_y.to(device)
                    if phase == "train":
                        optimizer.zero_grad(set_to_none=True)
                    loss = training_objective(
                        model(x), hard_y, soft_y, ambiguity_y, variant,
                        config.training.brier_weight, config.training.ambiguity_weight,
                    )
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    total += float(loss.detach()) * len(x)
                    count += len(x)
            losses[phase] = total / count
        history.append({"epoch": epoch + 1, **losses})
        if losses["validation"] < best_validation - 1e-6:
            best_validation = losses["validation"]
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= config.training.early_stopping_patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    metadata = CheckpointMetadata(
        objective=variant,
        input_dim=embeddings.shape[1],
        hidden_dim=config.model.hidden_dim,
        dropout=config.model.dropout,
        events=tuple(config.events),
        sample_rate=config.dataset.sample_rate,
        encoder_name=config.model.encoder_name,
        encoder_revision=config.model.encoder_revision,
        dataset_id=config.dataset.dataset_id,
        dataset_revision=config.dataset.revision,
        split_fingerprint=split_fingerprint,
        seed=seed,
        ambiguity_supported=variant == "ambiguity_multitask",
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**metadata.to_dict(), "model_state_dict": best_state,
                "best_validation_loss": best_validation, "history": history}, checkpoint_path)
    model.load_state_dict(best_state)
    return model.eval().to(device), history


def _predict(model: nn.Module, embeddings: np.ndarray, indices: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray | None]:
    event_chunks, ambiguity_chunks = [], []
    loader = DataLoader(torch.from_numpy(embeddings[indices]).float(), batch_size=64)
    model.eval()
    with torch.inference_mode():
        for values in loader:
            outputs = model(values.to(device))
            if isinstance(outputs, tuple):
                event_logits, ambiguity_logits = outputs
                ambiguity_chunks.append(torch.sigmoid(ambiguity_logits).cpu().numpy())
            else:
                event_logits = outputs
            event_chunks.append(torch.sigmoid(event_logits).cpu().numpy())
    return (
        np.concatenate(event_chunks),
        np.concatenate(ambiguity_chunks) if ambiguity_chunks else None,
    )


def _baseline_comparison(run_dir: Path, aggregate_rows: list[dict]) -> None:
    baseline_path = Path("results/metrics_by_seed.csv")
    if not baseline_path.is_file():
        return
    baseline = pd.read_csv(baseline_path)
    rows = []
    for row in aggregate_rows:
        selected = baseline[
            (baseline.model == row["model"])
            & (baseline.threshold_policy == row["threshold_policy"])
        ]
        if selected.empty:
            continue
        rows.append({
            "model": row["model"],
            "threshold_policy": row["threshold_policy"],
            "baseline_2026_10_08_macro_f1": float(selected.iloc[0].macro_f1),
            "v2_macro_f1_mean": row.get("macro_f1_mean"),
            "v2_seed_count": row["seed_count"],
            "comparison_status": "DESCRIPTIVE_ONLY; protocols may differ",
        })
    if rows:
        pd.DataFrame(rows).to_csv(run_dir / "baseline_vs_v2.csv", index=False)


def run_experiment(config: ExperimentConfig) -> Path:
    """Execute one configured v2 run and return its collision-safe directory."""
    config.validate()
    run_dir, resumed = prepare_run_directory(config)
    device = _device(config.training.device)
    epochs = min(config.training.epochs, 2) if config.mode == "smoke" else config.training.epochs
    batch_size = min(config.training.batch_size, 4) if config.mode == "smoke" else config.training.batch_size
    bootstrap_replicates = min(config.evaluation.bootstrap_replicates, 100) if config.mode == "smoke" else config.evaluation.bootstrap_replicates
    initial_metadata = {
        "schema_version": 2,
        "config_sha256": config.fingerprint,
        "run_status": "IN_PROGRESS",
        "scientific_validation": "UNVALIDATED",
        "resumed": resumed,
        "warning": "Do not cite this run until its provenance and outputs have been reviewed.",
    }
    _write_json(run_dir / "run_metadata.json", initial_metadata)
    _write_json(run_dir / "config.json", config.to_dict())
    _write_json(run_dir / "environment.json", _environment())

    metadata, waveforms, provenance, features = _load_sample(config)
    groups, identity_audit = _identity_and_groups(config, metadata, provenance["available_columns"])
    partitions = grouped_partition(
        groups,
        (config.split.train_fraction, config.split.validation_fraction, config.split.test_fraction),
        config.split.seed,
    )
    split_audit = audit_partitions(
        partitions, metadata["audio_sha256"], groups, identity_audit
    )
    metadata["dependence_group"] = groups
    metadata["partition"] = ""
    for name, indices in partitions.items():
        metadata.loc[indices, "partition"] = name
    provenance.update({
        "dataset_feature_schema": str(features),
        "identity_audit": identity_audit.to_dict(),
        "split_audit": split_audit.to_dict(),
    })
    _write_json(run_dir / "provenance.json", provenance)
    metadata.to_csv(run_dir / "sample_manifest.csv", index=False)
    _write_json(run_dir / "splits.json", {
        "split_fingerprint": _split_fingerprint(partitions, metadata),
        "partitions": {name: values.tolist() for name, values in partitions.items()},
        "audit": split_audit.to_dict(),
    })

    counts = metadata[list(config.events)].to_numpy(dtype=np.float32)
    hard = (counts >= 2).astype(np.float32)
    soft = counts / 3.0
    ambiguity = np.isin(counts, (1, 2)).astype(np.float32)
    embeddings = _extract_embeddings(config, waveforms, metadata, run_dir, device, batch_size)
    split_fingerprint = _split_fingerprint(partitions, metadata)
    train_indices = partitions["train"]
    validation_indices = partitions["validation"]
    test_indices = partitions["test"]

    summary_rows: list[dict] = []
    per_event_rows: list[dict] = []
    threshold_rows: list[dict] = []
    history_rows: list[dict] = []
    bootstrap_rows: list[dict] = []
    interval_rows: list[dict] = []
    ambiguity_rows: list[dict] = []
    failure_rows: list[dict] = []
    stratification_rows: list[dict] = []
    risk_rows: list[dict] = []
    predictions: dict[tuple[int, str], dict[str, Any]] = {}

    for seed in config.training.seeds:
        for variant in config.training.objectives:
            checkpoint = run_dir / "checkpoints" / f"seed_{seed}" / f"{variant}.pt"
            model, history = _train_model(
                config, variant, seed, embeddings, hard, soft, ambiguity,
                train_indices, validation_indices, split_fingerprint,
                checkpoint, device, epochs, batch_size,
            )
            history_rows.extend({"seed": seed, "model": variant, **row} for row in history)
            validation_probabilities, _ = _predict(model, embeddings, validation_indices, device)
            tuned = select_f1_thresholds(
                hard[validation_indices], validation_probabilities,
                config.evaluation.threshold_grid_size, "validation",
            )
            test_probabilities, test_ambiguity = _predict(model, embeddings, test_indices, device)
            policies = {
                "fixed_0.5": np.full(len(config.events), 0.5),
                "validation_tuned": tuned.thresholds,
            }
            predictions[(seed, variant)] = {
                "probabilities": test_probabilities,
                "ambiguity": test_ambiguity,
                "thresholds": policies,
            }
            for policy, thresholds in policies.items():
                summary, event_rows = evaluate_multilabel(
                    hard[test_indices], soft[test_indices], test_probabilities,
                    thresholds, config.events, config.evaluation.calibration_bins,
                )
                summary_rows.append({"seed": seed, "model": variant, "threshold_policy": policy, **summary})
                per_event_rows.extend({
                    "seed": seed, "model": variant, "threshold_policy": policy, **row
                } for row in event_rows)
                threshold_rows.extend({
                    "seed": seed, "model": variant, "policy": policy,
                    "event": event, "threshold": float(thresholds[index]),
                    "selected_on": "fixed" if policy == "fixed_0.5" else "validation",
                } for index, event in enumerate(config.events))

            selected_thresholds = policies["validation_tuned"]
            interval_rows.extend({
                "seed": seed, "model": variant, "threshold_policy": "validation_tuned", **row
            } for row in group_bootstrap_metric_intervals(
                hard[test_indices], soft[test_indices], test_probabilities,
                selected_thresholds, config.events, groups[test_indices],
                bootstrap_replicates, seed, config.evaluation.bootstrap_confidence,
                identity_audit.status, config.evaluation.calibration_bins,
            ))
            predicted = test_probabilities >= selected_thresholds
            sample_errors = np.not_equal(predicted, hard[test_indices]).mean(axis=1)
            entropy = binary_entropy(test_probabilities).mean(axis=1)
            prevalence_score = prevalence_rejection_score(
                test_probabilities, hard[train_indices].mean(axis=0)
            )
            ambiguity_score = test_ambiguity.mean(axis=1) if test_ambiguity is not None else None
            curves = build_risk_curves(
                sample_errors, entropy, ambiguity_score, prevalence_score,
                config.evaluation.risk_coverages,
            )
            curves["random baseline"] = random_risk_coverage(
                sample_errors, config.evaluation.risk_coverages, 200, seed
            )
            for ranking, rows in curves.items():
                risk_rows.extend({"seed": seed, "model": variant, "ranking": ranking, **row} for row in rows)
            failure_rows.extend({"seed": seed, "model": variant, **row} for row in false_positive_negative_rows(
                hard[test_indices], test_probabilities, selected_thresholds,
                config.events, metadata.iloc[test_indices]["sample_id"].astype(str),
            ))
            disagreement = ambiguity[test_indices].mean(axis=1)
            test_meta = metadata.iloc[test_indices].copy()
            test_meta["annotation_disagreement"] = disagreement
            stratification_rows.extend({"seed": seed, "model": variant, **row} for row in stratified_error_rows(
                sample_errors, test_meta,
                (
                    "duration_seconds", "rms_amplitude", "peak_amplitude",
                    "clipping_fraction", "annotation_disagreement", *DIAGNOSTIC_COLUMNS,
                ),
            ))
            if test_ambiguity is not None:
                rows = ambiguity_disagreement_rows(counts[test_indices], test_ambiguity, config.events)
                for event_index, row in enumerate(rows):
                    auroc = safe_rank_metric(
                        roc_auc_score,
                        ambiguity[test_indices, event_index], test_ambiguity[:, event_index],
                    )
                    ambiguity_rows.append({
                        "seed": seed, "model": variant, **row,
                        "auroc": auroc.value, "auroc_status": auroc.status,
                    })
            model.cpu()

        reference = predictions[(seed, "hard")]
        for candidate in config.training.objectives:
            if candidate == "hard":
                continue
            compared = predictions[(seed, candidate)]
            rows = paired_prediction_bootstrap(
                hard[test_indices], soft[test_indices],
                reference["probabilities"], compared["probabilities"],
                reference["thresholds"]["validation_tuned"],
                compared["thresholds"]["validation_tuned"],
                groups[test_indices], bootstrap_replicates, seed,
                config.evaluation.bootstrap_confidence, identity_audit.status,
            )
            bootstrap_rows.extend({
                "seed": seed, "reference": "hard", "candidate": candidate, **row
            } for row in rows)

    aggregate_rows = aggregate_seed_metrics(summary_rows, SUMMARY_METRICS)
    pd.DataFrame(summary_rows).to_csv(run_dir / "metrics_by_seed.csv", index=False)
    pd.DataFrame(aggregate_rows).to_csv(run_dir / "metrics_aggregate.csv", index=False)
    pd.DataFrame(per_event_rows).to_csv(run_dir / "per_event_metrics.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(run_dir / "thresholds.csv", index=False)
    pd.DataFrame(history_rows).to_csv(run_dir / "training_history.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(run_dir / "paired_group_bootstrap.csv", index=False)
    pd.DataFrame(interval_rows).to_csv(run_dir / "group_bootstrap_intervals.csv", index=False)
    pd.DataFrame(ambiguity_rows).to_csv(run_dir / "ambiguity_metrics.csv", index=False)
    pd.DataFrame(failure_rows).to_csv(run_dir / "false_positive_negative.csv", index=False)
    pd.DataFrame(stratification_rows).to_csv(run_dir / "error_stratification.csv", index=False)
    pd.DataFrame(risk_rows).to_csv(run_dir / "risk_coverage.csv", index=False)
    if config.mode == "full":
        _baseline_comparison(run_dir, aggregate_rows)

    primary = config.training.seeds[0]
    figures = run_dir / "figures"
    figures.mkdir(exist_ok=True)
    for variant in config.training.objectives:
        current = predictions[(primary, variant)]
        save_precision_recall_plot(
            hard[test_indices], current["probabilities"], config.events,
            figures / f"{variant}_precision_recall.png",
        )
        save_calibration_plot(
            hard[test_indices], soft[test_indices], current["probabilities"], config.events,
            figures / f"{variant}_calibration.png", config.evaluation.calibration_bins,
        )
    primary_values = predictions[(primary, "hard")]
    primary_predicted = primary_values["probabilities"] >= primary_values["thresholds"]["validation_tuned"]
    primary_errors = np.not_equal(primary_predicted, hard[test_indices]).mean(axis=1)
    plot_curves = {
        "prediction entropy": risk_coverage_curve(
            primary_errors, binary_entropy(primary_values["probabilities"]).mean(axis=1),
            config.evaluation.risk_coverages,
        ),
        "prevalence baseline": risk_coverage_curve(
            primary_errors,
            prevalence_rejection_score(primary_values["probabilities"], hard[train_indices].mean(axis=0)),
            config.evaluation.risk_coverages,
        ),
        "random baseline": random_risk_coverage(
            primary_errors, config.evaluation.risk_coverages, 200, primary,
        ),
    }
    save_risk_coverage_plot(plot_curves, figures / "hard_risk_coverage.png")

    final_metadata = {
        **initial_metadata,
        "run_status": "COMPLETED",
        "scientific_validation": "UNVALIDATED",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "split_fingerprint": split_fingerprint,
        "identity_status": identity_audit.status,
        "test_access_policy": "test labels used only after training and validation threshold selection",
    }
    _write_json(run_dir / "run_metadata.json", final_metadata)
    artifact_rows = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "artifact_manifest.json":
            artifact_rows.append({
                "path": str(path.relative_to(run_dir)),
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            })
    _write_json(run_dir / "artifact_manifest.json", {
        "scientific_validation": "UNVALIDATED",
        "files": artifact_rows,
    })
    return run_dir
