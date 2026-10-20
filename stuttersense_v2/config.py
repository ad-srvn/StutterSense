"""Typed experiment configuration and collision-safe run directories."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVENTS = ("prolongation", "block", "sound_rep", "word_rep", "interjection")
OBJECTIVES = ("hard", "soft", "brier_ablation", "ambiguity_multitask")


@dataclass(frozen=True)
class DatasetConfig:
    dataset_id: str = "psusac/sep28k"
    subset: str = "default"
    split: str = "train"
    revision: str = "67630930030046543194dc7c701b87b2cdc7680f"
    sample_rate: int = 16_000
    metadata_manifest: str | None = None
    verified_group_kind: str | None = None
    verified_group_column: str | None = None
    identity_source: str | None = None
    identity_revision: str | None = None


@dataclass(frozen=True)
class SamplingConfig:
    sample_size: int | None = 2_000
    reservoir_seed: int = 42
    scan_limit: int | None = None


@dataclass(frozen=True)
class SplitConfig:
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42


@dataclass(frozen=True)
class ModelConfig:
    encoder_name: str = "microsoft/wavlm-base-plus"
    encoder_revision: str = "4c66d4806a428f2e922ccfa1a962776e232d487b"
    frozen_encoder: bool = True
    reuse_embeddings: bool = True
    hidden_dim: int = 256
    dropout: float = 0.20


@dataclass(frozen=True)
class TrainingConfig:
    seeds: tuple[int, ...] = (42, 123, 456)
    objectives: tuple[str, ...] = OBJECTIVES
    epochs: int = 12
    batch_size: int = 8
    learning_rate: float = 1e-3
    early_stopping_patience: int = 3
    brier_weight: float = 0.25
    ambiguity_weight: float = 0.25
    device: str = "auto"


@dataclass(frozen=True)
class EvaluationConfig:
    threshold_grid_size: int = 201
    calibration_bins: int = 10
    bootstrap_replicates: int = 1_000
    bootstrap_confidence: float = 0.95
    risk_coverages: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 1.0)


@dataclass(frozen=True)
class OutputConfig:
    root: str = "v2_outputs"
    run_name: str | None = None
    resume_from: str | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str = "full"
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    events: tuple[str, ...] = EVENTS

    def validate(self) -> None:
        if self.mode not in {"smoke", "full"}:
            raise ValueError("mode must be 'smoke' or 'full'")
        if not self.dataset.revision or not self.model.encoder_revision:
            raise ValueError("Dataset and encoder revisions must be pinned")
        if self.dataset.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.sampling.sample_size is not None and self.sampling.sample_size < 1:
            raise ValueError("sample_size must be positive or null")
        if self.sampling.scan_limit is not None and self.sampling.scan_limit < 1:
            raise ValueError("scan_limit must be positive or null")
        fractions = (
            self.split.train_fraction,
            self.split.validation_fraction,
            self.split.test_fraction,
        )
        if any(value <= 0 for value in fractions) or abs(sum(fractions) - 1.0) > 1e-9:
            raise ValueError("Split fractions must be positive and sum to 1")
        if not self.training.seeds or len(set(self.training.seeds)) != len(self.training.seeds):
            raise ValueError("Training seeds must be non-empty and unique")
        unknown = set(self.training.objectives) - set(OBJECTIVES)
        if unknown:
            raise ValueError(f"Unknown objectives: {sorted(unknown)}")
        if self.training.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps")
        if not self.model.frozen_encoder:
            raise ValueError("StutterSense v2 currently supports only a frozen WavLM encoder")
        if self.training.epochs < 1 or self.training.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive")
        if self.evaluation.threshold_grid_size < 3:
            raise ValueError("threshold_grid_size must be at least 3")
        if self.evaluation.bootstrap_replicates < 1:
            raise ValueError("bootstrap_replicates must be positive")
        if not 0 < self.evaluation.bootstrap_confidence < 1:
            raise ValueError("bootstrap_confidence must be between 0 and 1")
        if tuple(self.events) != EVENTS:
            raise ValueError(f"Event order must be {EVENTS}")
        if self.output.run_name and Path(self.output.run_name).name != self.output.run_name:
            raise ValueError("run_name must be a single directory name")
        group_fields = (
            self.dataset.verified_group_kind,
            self.dataset.verified_group_column,
            self.dataset.identity_source,
            self.dataset.identity_revision,
        )
        if any(group_fields) and not all(group_fields):
            raise ValueError(
                "verified group kind, column, identity source, and identity revision must be set together"
            )
        if self.dataset.metadata_manifest and any(group_fields):
            raise ValueError("Use either a metadata manifest or a native verified group field, not both")
        if self.dataset.verified_group_kind not in {None, "speaker", "episode", "recording"}:
            raise ValueError("verified_group_kind must be speaker, episode, recording, or null")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        values = self.to_dict()
        # The resume location is operational state, not part of the experiment definition.
        values["output"]["resume_from"] = None
        payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _section(cls: type, payload: dict[str, Any], name: str):
    values = dict(payload.get(name, {}))
    if cls is TrainingConfig:
        values["seeds"] = tuple(values.get("seeds", TrainingConfig.seeds))
        values["objectives"] = tuple(values.get("objectives", TrainingConfig.objectives))
    if cls is EvaluationConfig:
        values["risk_coverages"] = tuple(values.get("risk_coverages", EvaluationConfig.risk_coverages))
    return cls(**values)


def load_config(path: str | Path) -> ExperimentConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    config = ExperimentConfig(
        mode=payload.get("mode", "full"),
        dataset=_section(DatasetConfig, payload, "dataset"),
        sampling=_section(SamplingConfig, payload, "sampling"),
        split=_section(SplitConfig, payload, "split"),
        model=_section(ModelConfig, payload, "model"),
        training=_section(TrainingConfig, payload, "training"),
        evaluation=_section(EvaluationConfig, payload, "evaluation"),
        output=_section(OutputConfig, payload, "output"),
        events=tuple(payload.get("events", EVENTS)),
    )
    config.validate()
    return config


def prepare_run_directory(config: ExperimentConfig) -> tuple[Path, bool]:
    """Create a unique run directory or validate an explicitly resumed run."""
    root = Path(config.output.root)
    if config.output.resume_from:
        run_dir = Path(config.output.resume_from)
        metadata_path = run_dir / "run_metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Cannot resume without {metadata_path}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("config_sha256") != config.fingerprint:
            raise ValueError("Resume refused: configuration fingerprint does not match")
        if metadata.get("run_status") == "COMPLETED":
            raise ValueError("Resume refused: completed runs are immutable")
        return run_dir, True

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = config.output.run_name or f"{stamp}-{config.fingerprint[:10]}"
    run_dir = root / name
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir, False
