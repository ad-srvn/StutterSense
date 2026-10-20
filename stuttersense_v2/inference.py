"""Checkpoint-validated clip-level inference for downstream research reuse."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.signal import resample_poly

from .config import EVENTS
from .modeling import build_classifier, validate_checkpoint_payload


@dataclass(frozen=True)
class ClipPrediction:
    events: dict[str, float]
    ambiguity: dict[str, float] | None
    duration_seconds: float
    sample_rate: int
    scope: str = "clip-level probabilities; no timestamps or endpointing probabilities"


def preprocess_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    target_rate: int = 16_000,
    max_duration_seconds: float | None = None,
) -> np.ndarray:
    values = np.asarray(waveform)
    if values.ndim == 2:
        if min(values.shape) > 8:
            raise ValueError("Two-dimensional audio must have a recognizable channel axis")
        channel_axis = 0 if values.shape[0] <= values.shape[1] else 1
        values = values.mean(axis=channel_axis)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("waveform must be a non-empty mono or multi-channel array")
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("waveform contains non-finite values")
    if sample_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if max_duration_seconds is not None and values.size / sample_rate > max_duration_seconds:
        raise ValueError(
            "Audio exceeds the configured clip duration; chunk it explicitly before inference"
        )
    if sample_rate != target_rate:
        divisor = int(np.gcd(sample_rate, target_rate))
        values = resample_poly(values, target_rate // divisor, sample_rate // divisor)
    return np.ascontiguousarray(values, dtype=np.float32)


class StutterSensePredictor:
    """Load a v2 checkpoint and return only supported clip-level outputs."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        local_files_only: bool = True,
        expected_events: Sequence[str] = EVENTS,
    ):
        payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=True)
        self.metadata = validate_checkpoint_payload(payload, tuple(expected_events))
        self.events = tuple(self.metadata["events"])
        self.device = torch.device(device)
        self.classifier = build_classifier(
            self.metadata["objective"],
            int(self.metadata["input_dim"]),
            int(self.metadata["hidden_dim"]),
            len(self.events),
            float(self.metadata["dropout"]),
        )
        self.classifier.load_state_dict(self.metadata["model_state_dict"], strict=True)
        self.classifier.eval().to(self.device)

        from transformers import AutoFeatureExtractor, WavLMModel

        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.metadata["encoder_name"],
            revision=self.metadata["encoder_revision"],
            local_files_only=local_files_only,
        )
        self.encoder = WavLMModel.from_pretrained(
            self.metadata["encoder_name"],
            revision=self.metadata["encoder_revision"],
            local_files_only=local_files_only,
        )
        self.encoder.requires_grad_(False).eval().to(self.device)
        if int(self.encoder.config.hidden_size) != int(self.metadata["input_dim"]):
            raise ValueError("Checkpoint classifier input does not match the pinned encoder")

    @staticmethod
    def _masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor, encoder) -> torch.Tensor:
        if hasattr(encoder, "_get_feature_vector_attention_mask"):
            mask = encoder._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
        else:
            lengths = encoder._get_feat_extract_output_lengths(attention_mask.sum(dim=-1))
            positions = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
            mask = positions < lengths[:, None]
        weights = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.0)

    def predict(
        self,
        waveform: np.ndarray,
        sample_rate: int,
        max_duration_seconds: float | None = 30.0,
    ) -> ClipPrediction:
        expected_rate = int(self.metadata["sample_rate"])
        audio = preprocess_waveform(
            waveform, sample_rate, expected_rate, max_duration_seconds
        )
        encoded = self.feature_extractor(
            audio,
            sampling_rate=expected_rate,
            padding=True,
            truncation=False,
            return_attention_mask=True,
            return_tensors="pt",
        )
        values = encoded["input_values"].to(self.device)
        mask = encoded.get("attention_mask", torch.ones_like(values, dtype=torch.long)).to(self.device)
        with torch.inference_mode():
            hidden = self.encoder(input_values=values, attention_mask=mask).last_hidden_state
            embedding = self._masked_mean(hidden, mask, self.encoder)
            outputs = self.classifier(embedding)
            if isinstance(outputs, tuple):
                event_logits, ambiguity_logits = outputs
                ambiguity_values = torch.sigmoid(ambiguity_logits)[0].cpu().numpy()
            else:
                event_logits = outputs
                ambiguity_values = None
            event_values = torch.sigmoid(event_logits)[0].cpu().numpy()
        return ClipPrediction(
            events=dict(zip(self.events, map(float, event_values))),
            ambiguity=(
                dict(zip(self.events, map(float, ambiguity_values)))
                if ambiguity_values is not None else None
            ),
            duration_seconds=float(audio.size / expected_rate),
            sample_rate=expected_rate,
        )
