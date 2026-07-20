"""Dependency-light data structures shared by the audio analysis modules."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
ProgressCallback = Callable[..., None]
Band = Literal["low_hit", "mid_hit", "high_hit", "full_band_accent"]
StemKind = Literal["mix", "vocals", "drums", "bass", "other"]


@dataclass(slots=True)
class AudioData:
    path: Path | None
    original: FloatArray
    mono: FloatArray
    analysis_channels: FloatArray
    analysis_mono: FloatArray
    original_sample_rate: int
    analysis_sample_rate: int
    channels: int
    sample_count: int
    duration_sec: float
    leading_silence_samples: int
    normalization_gain: float
    dc_offset: float

    def analysis_to_original_sample(self, sample: int | float) -> int:
        if self.sample_count <= 0:
            return 0
        mapped = int(round(float(sample) * self.original_sample_rate / self.analysis_sample_rate))
        return min(max(mapped, 0), self.sample_count - 1)

    def original_to_analysis_sample(self, sample: int | float) -> int:
        if self.analysis_mono.size <= 0:
            return 0
        mapped = int(round(float(sample) * self.analysis_sample_rate / self.original_sample_rate))
        return min(max(mapped, 0), int(self.analysis_mono.size) - 1)


@dataclass(slots=True)
class FeatureBundle:
    hop_length: int
    sample_rate: int
    frame_samples: NDArray[np.int64]
    curves: dict[str, FloatArray]
    normalized_curves: dict[str, FloatArray]
    fused: FloatArray
    band_energy: dict[str, FloatArray]
    scale_votes: NDArray[np.int16]
    percussive: FloatArray
    harmonic: FloatArray


@dataclass(slots=True)
class OnsetCandidate:
    detected_sample: int
    refined_sample: int
    sample: int
    band: Band = "mid_hit"
    confidence: float = 0.0
    salience: float = 0.0
    source: str = "fused"
    detector_votes: list[str] = field(default_factory=list)
    band_evidence: dict[str, float] = field(default_factory=dict)
    peak_value: float = 0.0
    prominence: float = 0.0
    loudness: float = 0.0
    primary_stem: StemKind = "mix"
    stem_evidence: dict[str, float] = field(default_factory=dict)
    semantic_evidence: dict[str, float] = field(default_factory=dict)
    candidate_id: str | None = None


@dataclass(slots=True)
class TempoEstimate:
    bpm: float
    confidence: float
    beat_offset_sample: int
    score: float = 0.0
    candidates: list[dict[str, float]] = field(default_factory=list)


@dataclass(slots=True)
class AnalysisResult:
    original_sample_rate: int
    sample_count: int
    channels: int
    duration_sec: float
    leading_silence_samples: int
    bpm: float
    bpm_confidence: float
    beat_offset_sample: int
    hit_points: list[dict[str, Any]]
    metadata: dict[str, Any]
    warnings: list[str]
    stage_timings_ms: dict[str, int]
    candidate_events: list[dict[str, Any]] = field(default_factory=list)
    stem_audio: dict[str, FloatArray] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        # Stem PCM is a local worker artifact, not part of the JSON API payload.
        value.pop("stem_audio", None)
        return value
