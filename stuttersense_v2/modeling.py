"""Baseline-compatible classifier heads, objectives, and checkpoint metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


CHECKPOINT_SCHEMA_VERSION = 2


class StutterMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 5, dropout: float = 0.20):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class AmbiguityMultiTaskMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 5, dropout: float = 0.20):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.event_head = nn.Linear(hidden_dim, output_dim)
        self.ambiguity_head = nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.shared(inputs)
        return self.event_head(hidden), self.ambiguity_head(hidden)


def soft_brier_regularized_loss(
    logits: torch.Tensor, soft_targets: torch.Tensor, weight: float
) -> torch.Tensor:
    """Baseline definition: soft BCE plus a Brier regularizer."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, soft_targets)
    brier = torch.mean((torch.sigmoid(logits) - soft_targets) ** 2)
    return bce + weight * brier


def ambiguity_multitask_loss(
    event_logits: torch.Tensor,
    ambiguity_logits: torch.Tensor,
    soft_targets: torch.Tensor,
    ambiguity_targets: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    event_loss = nn.functional.binary_cross_entropy_with_logits(event_logits, soft_targets)
    ambiguity_loss = nn.functional.binary_cross_entropy_with_logits(
        ambiguity_logits, ambiguity_targets
    )
    return event_loss + weight * ambiguity_loss


def training_objective(
    outputs: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    hard_targets: torch.Tensor,
    soft_targets: torch.Tensor,
    ambiguity_targets: torch.Tensor,
    variant: str,
    brier_weight: float = 0.25,
    ambiguity_weight: float = 0.25,
) -> torch.Tensor:
    """Preserve the four objective definitions from the validated baseline."""
    if variant == "ambiguity_multitask":
        if not isinstance(outputs, tuple):
            raise TypeError("ambiguity_multitask requires event and ambiguity logits")
        return ambiguity_multitask_loss(
            outputs[0], outputs[1], soft_targets, ambiguity_targets, ambiguity_weight
        )
    if isinstance(outputs, tuple):
        raise TypeError(f"{variant} expects event logits only")
    if variant == "hard":
        return nn.functional.binary_cross_entropy_with_logits(outputs, hard_targets)
    if variant == "soft":
        return nn.functional.binary_cross_entropy_with_logits(outputs, soft_targets)
    if variant == "brier_ablation":
        return soft_brier_regularized_loss(outputs, soft_targets, brier_weight)
    raise ValueError(f"Unknown objective: {variant}")


def build_classifier(
    variant: str,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
) -> nn.Module:
    cls = AmbiguityMultiTaskMLP if variant == "ambiguity_multitask" else StutterMLP
    return cls(input_dim, hidden_dim, output_dim, dropout)


@dataclass(frozen=True)
class CheckpointMetadata:
    objective: str
    input_dim: int
    hidden_dim: int
    dropout: float
    events: tuple[str, ...]
    sample_rate: int
    encoder_name: str
    encoder_revision: str
    dataset_id: str
    dataset_revision: str
    split_fingerprint: str
    seed: int
    ambiguity_supported: bool
    preprocessing: str = "mono-float32-resample-v1"
    schema_version: int = CHECKPOINT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "objective": self.objective,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "events": list(self.events),
            "sample_rate": self.sample_rate,
            "encoder_name": self.encoder_name,
            "encoder_revision": self.encoder_revision,
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "split_fingerprint": self.split_fingerprint,
            "seed": self.seed,
            "ambiguity_supported": self.ambiguity_supported,
            "preprocessing": self.preprocessing,
        }


def validate_checkpoint_payload(payload: dict[str, Any], expected_events: tuple[str, ...]) -> dict[str, Any]:
    required = {
        "schema_version", "model_state_dict", "objective", "input_dim", "hidden_dim",
        "dropout", "events", "sample_rate", "encoder_name", "encoder_revision",
        "dataset_id", "dataset_revision", "split_fingerprint", "seed",
        "ambiguity_supported", "preprocessing",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Checkpoint metadata is incomplete: {sorted(missing)}")
    if payload["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported checkpoint schema version")
    if tuple(payload["events"]) != tuple(expected_events):
        raise ValueError("Checkpoint label order is incompatible")
    if payload["objective"] not in {"hard", "soft", "brier_ablation", "ambiguity_multitask"}:
        raise ValueError("Checkpoint objective is unsupported")
    if bool(payload["ambiguity_supported"]) != (payload["objective"] == "ambiguity_multitask"):
        raise ValueError("Checkpoint ambiguity metadata is inconsistent")
    if not payload["encoder_revision"] or not payload["dataset_revision"]:
        raise ValueError("Checkpoint revisions must be pinned")
    if payload["preprocessing"] != "mono-float32-resample-v1":
        raise ValueError("Checkpoint preprocessing contract is unsupported")
    if int(payload["sample_rate"]) <= 0 or int(payload["input_dim"]) <= 0:
        raise ValueError("Checkpoint dimensions and sample rate must be positive")
    return payload
