from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Literal

from pydantic import Field, model_validator

from ...schemas import ApiModel

AlignmentMethodId = Literal["qwen", "mfa", "ctc", "singing", "hybrid"]
AlignmentStatus = Literal[
    "empty",
    "queued",
    "processing",
    "completed",
    "failed",
    "unavailable",
]


class AlignmentToken(ApiModel):
    """The one and only token shape returned by every alignment adapter."""

    id: str
    text: str
    phoneme: str | None = None
    start_sample: int = Field(ge=0)
    end_sample: int = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    method: AlignmentMethodId

    @model_validator(mode="after")
    def validate_span(self) -> AlignmentToken:
        if self.end_sample <= self.start_sample:
            raise ValueError("end_sample must be greater than start_sample")
        if not math.isfinite(self.confidence):
            raise ValueError("confidence must be finite")
        return self


AlignmentLayer = Literal["phoneme", "mora", "character"]


class AlignmentAcousticEvidence(ApiModel):
    """Normalized local acoustic evidence used to refine one observed boundary."""

    energy: float = Field(ge=0, le=1)
    spectral_change: float = Field(ge=0, le=1)
    pitch_change: float = Field(ge=0, le=1)


class AlignmentHierarchyUnit(ApiModel):
    """One DP-mapped lyric unit with raw CTC and acoustically refined intervals."""

    id: str
    index: int = Field(ge=0)
    level: AlignmentLayer
    text: str
    kana: str | None = None
    mora: str | None = None
    phoneme: str | None = None
    kind: str | None = None
    character_indices: list[int] = Field(default_factory=list)
    mora_indices: list[int] = Field(default_factory=list)
    phoneme_indices: list[int] = Field(default_factory=list)
    aligned_start_sample: int = Field(ge=0)
    aligned_end_sample: int = Field(ge=0)
    refined_start_sample: int = Field(ge=0)
    refined_end_sample: int = Field(ge=0)
    aligned_sample: int = Field(ge=0)
    refined_sample: int = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    observed_token_index: int | None = Field(default=None, ge=0)
    match_operation: str | None = None
    evidence: AlignmentAcousticEvidence | None = None

    @model_validator(mode="after")
    def validate_intervals(self) -> AlignmentHierarchyUnit:
        if self.aligned_end_sample <= self.aligned_start_sample:
            raise ValueError("aligned interval must have positive duration")
        if self.refined_end_sample <= self.refined_start_sample:
            raise ValueError("refined interval must have positive duration")
        if self.aligned_sample != self.aligned_start_sample:
            raise ValueError("aligned_sample must preserve the raw CTC start boundary")
        if self.refined_sample != self.refined_start_sample:
            raise ValueError("refined_sample must equal the refined start boundary")
        for indices in (
            self.character_indices,
            self.mora_indices,
            self.phoneme_indices,
        ):
            if any(index < 0 for index in indices):
                raise ValueError("hierarchy indices must be non-negative")
            if len(indices) != len(set(indices)) or indices != sorted(indices):
                raise ValueError("hierarchy indices must be sorted and unique")
        return self


class AlignmentHierarchy(ApiModel):
    phonemes: list[AlignmentHierarchyUnit] = Field(default_factory=list)
    moras: list[AlignmentHierarchyUnit] = Field(default_factory=list)
    characters: list[AlignmentHierarchyUnit] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_levels(self) -> AlignmentHierarchy:
        populated_layers = tuple(
            bool(units) for units in (self.phonemes, self.moras, self.characters)
        )
        if any(populated_layers) and not all(populated_layers):
            raise ValueError(
                "a populated alignment hierarchy requires phoneme, mora, and character layers"
            )
        for expected, units in (
            ("phoneme", self.phonemes),
            ("mora", self.moras),
            ("character", self.characters),
        ):
            if any(unit.level != expected for unit in units):
                raise ValueError(f"{expected} hierarchy contains a different unit level")
            if len({unit.id for unit in units}) != len(units):
                raise ValueError(f"{expected} hierarchy ids must be unique")
            if [unit.index for unit in units] != list(range(len(units))):
                raise ValueError(f"{expected} hierarchy indices must be sequential")
            for previous, current in zip(units, units[1:], strict=False):
                if current.refined_start_sample < previous.refined_start_sample:
                    raise ValueError(f"{expected} refined intervals must be monotonic")
                if current.aligned_start_sample < previous.aligned_start_sample:
                    raise ValueError(f"{expected} aligned intervals must be monotonic")
        for unit in self.phonemes:
            if unit.observed_token_index is None:
                raise ValueError(
                    "every phoneme must reference an observed CTC token index"
                )
            if str(unit.match_operation or "").casefold() in {
                "delete",
                "deletion",
                "unmatched",
                "missing",
            }:
                raise ValueError("phoneme hierarchy cannot contain an unobserved match")
            if not unit.mora_indices or not unit.character_indices:
                raise ValueError(
                    "every phoneme must map to at least one mora and character"
                )
            if any(index >= len(self.moras) for index in unit.mora_indices) or any(
                index >= len(self.characters) for index in unit.character_indices
            ):
                raise ValueError("phoneme hierarchy relation is outside its target layer")
            if any(
                unit.index not in self.moras[index].phoneme_indices
                for index in unit.mora_indices
            ) or any(
                unit.index not in self.characters[index].phoneme_indices
                for index in unit.character_indices
            ):
                raise ValueError("phoneme hierarchy relations must be reciprocal")
        for unit in self.moras:
            if not unit.phoneme_indices or not unit.character_indices:
                raise ValueError(
                    "every mora must map to at least one phoneme and character"
                )
            if any(index >= len(self.phonemes) for index in unit.phoneme_indices) or any(
                index >= len(self.characters) for index in unit.character_indices
            ):
                raise ValueError("mora hierarchy relation is outside its target layer")
            if any(
                unit.index not in self.phonemes[index].mora_indices
                for index in unit.phoneme_indices
            ) or any(
                unit.index not in self.characters[index].mora_indices
                for index in unit.character_indices
            ):
                raise ValueError("mora hierarchy relations must be reciprocal")
        for unit in self.characters:
            if not unit.phoneme_indices or not unit.mora_indices:
                raise ValueError(
                    "every character must map to at least one phoneme and mora"
                )
            if any(index >= len(self.phonemes) for index in unit.phoneme_indices) or any(
                index >= len(self.moras) for index in unit.mora_indices
            ):
                raise ValueError("character hierarchy relation is outside its target layer")
            if any(
                unit.index not in self.phonemes[index].character_indices
                for index in unit.phoneme_indices
            ) or any(
                unit.index not in self.moras[index].character_indices
                for index in unit.mora_indices
            ):
                raise ValueError("character hierarchy relations must be reciprocal")
        return self


class AlignmentErrorInfo(ApiModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class AlignmentMethod(ApiModel):
    id: AlignmentMethodId
    name: str
    available: bool
    reason: str | None = None
    model: str | None = None
    automatic_downloads_enabled: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class AlignmentRunRequest(ApiModel):
    method: AlignmentMethodId


class AlignmentResult(ApiModel):
    run_id: str
    track_id: str
    method: AlignmentMethodId
    status: AlignmentStatus
    sample_rate: int = Field(gt=0)
    sample_count: int = Field(gt=0)
    tokens: list[AlignmentToken] = Field(default_factory=list)
    hierarchy: AlignmentHierarchy | None = None
    warnings: list[str] = Field(default_factory=list)
    error: AlignmentErrorInfo | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    @model_validator(mode="after")
    def validate_tokens(self) -> AlignmentResult:
        if self.status != "completed" and self.tokens:
            raise ValueError("non-completed alignment results cannot contain tokens")
        if self.status != "completed" and self.hierarchy is not None:
            raise ValueError("non-completed alignment results cannot contain a hierarchy")
        for token in self.tokens:
            if token.method != self.method:
                raise ValueError("every token method must match the result method")
            if token.start_sample >= self.sample_count or token.end_sample > self.sample_count:
                raise ValueError("alignment token is outside the original sample range")
        if self.hierarchy is not None:
            if self.method != "ctc":
                raise ValueError("only the HuBERT CTC result may contain a lyric hierarchy")
            for unit in (
                *self.hierarchy.phonemes,
                *self.hierarchy.moras,
                *self.hierarchy.characters,
            ):
                if (
                    unit.aligned_start_sample >= self.sample_count
                    or unit.aligned_end_sample > self.sample_count
                    or unit.refined_start_sample >= self.sample_count
                    or unit.refined_end_sample > self.sample_count
                ):
                    raise ValueError(
                        "alignment hierarchy unit is outside the original sample range"
                    )
            if len(self.tokens) != len(self.hierarchy.phonemes):
                raise ValueError("flat CTC tokens must mirror the refined phoneme hierarchy")
            for token, unit in zip(self.tokens, self.hierarchy.phonemes, strict=True):
                if (
                    token.start_sample != unit.refined_start_sample
                    or token.end_sample != unit.refined_end_sample
                    or token.phoneme != unit.phoneme
                ):
                    raise ValueError("flat CTC tokens must preserve refined phoneme spans")
        return self


class AlignmentReport(ApiModel):
    run_id: str
    track_id: str
    method: AlignmentMethodId
    score: float = Field(ge=0, le=1)
    coverage: float = Field(ge=0, le=1)
    acoustic: float = Field(ge=0, le=1)
    rhythm: float = Field(ge=0, le=1)
    stability: float = Field(ge=0, le=1)
    lyric_token_count: int = Field(ge=0)
    aligned_token_count: int = Field(ge=0)
    details: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
