from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


def to_camel(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


class ApiModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
        extra="forbid",
    )


ProjectStatus = Literal["unprocessed", "processing", "completed", "edited", "failed"]
AnalysisMode = Literal["recall", "balanced", "clean", "accurate"]
LyricsInputFormat = Literal["japanese", "kana", "romaji", "lrc"]
VocalAlignmentStatus = Literal[
    "empty", "draft", "ready", "queued", "processing", "completed", "failed", "unavailable"
]
HitBand = Literal["low_hit", "mid_hit", "high_hit", "full_band_accent"]
HitSource = Literal["mix", "percussive", "stems", "fused", "manual"]
StemKind = Literal["mix", "vocals", "drums", "bass", "other"]
CandidateLane = Literal["vocals", "melody", "drums", "mix"]
CandidateStatus = Literal["accepted", "rejected", "uncertain"]
FocusReason = Literal["vocal_presence", "drum_solo", "melodic_lead", "mixed", "manual"]


class StemDescriptor(ApiModel):
    source: StemKind
    available: bool
    waveform_url: str
    audio_url: str | None = None


class FocusAlternative(ApiModel):
    source: StemKind
    score: float = Field(ge=0, le=1)


class FocusSegment(ApiModel):
    id: str
    start_sample: int = Field(ge=0)
    end_sample: int = Field(gt=0)
    focus_source: StemKind
    confidence: float = Field(ge=0, le=1)
    reason: FocusReason
    evidence: dict[StemKind, float] = Field(default_factory=dict)
    alternatives: list[FocusAlternative] = Field(default_factory=list)
    manually_edited: bool = False


class ProjectCreate(ApiModel):
    title: str = Field(min_length=1, max_length=300)
    artist: str = Field(default="未知艺术家", max_length=300)
    genre: str = Field(default="Unknown", max_length=120)
    cover_url: str = Field(default="", max_length=1000)


class ProjectPatch(ApiModel):
    title: str | None = Field(default=None, min_length=1, max_length=300)
    artist: str | None = Field(default=None, max_length=300)
    genre: str | None = Field(default=None, max_length=120)
    cover_url: str | None = Field(default=None, max_length=1000)
    status: ProjectStatus | None = None


class ProjectResponse(ApiModel):
    id: str
    title: str
    artist: str
    genre: str
    cover_url: str
    status: ProjectStatus
    created_at: datetime
    updated_at: datetime
    track_id: str | None = None
    bpm: float | None = None
    duration_sec: float | None = None
    hit_point_count: int = 0
    analysis_mode: AnalysisMode | None = None


class TempoSegment(ApiModel):
    id: str
    start_sample: int = Field(ge=0)
    bpm: float = Field(gt=0, le=500)
    time_signature_numerator: int = Field(default=4, ge=1, le=32)
    time_signature_denominator: int = Field(default=4)
    beat_offset_sample: int = 0
    confidence: float = Field(default=0.0, ge=0, le=1)
    manually_edited: bool = False

    @field_validator("time_signature_denominator")
    @classmethod
    def validate_denominator(cls, value: int) -> int:
        if value not in {1, 2, 4, 8, 16, 32}:
            raise ValueError("time signature denominator must be a power of two")
        return value


class TempoSegmentInput(ApiModel):
    id: str | None = None
    start_sample: int = Field(default=0, ge=0)
    bpm: float = Field(gt=0, le=500)
    time_signature_numerator: int = Field(default=4, ge=1, le=32)
    time_signature_denominator: int = Field(default=4)
    beat_offset_sample: int = 0
    confidence: float = Field(default=0.0, ge=0, le=1)
    manually_edited: bool = True

    @field_validator("time_signature_denominator")
    @classmethod
    def validate_denominator(cls, value: int) -> int:
        if value not in {1, 2, 4, 8, 16, 32}:
            raise ValueError("time signature denominator must be a power of two")
        return value


class TempoMapUpdate(ApiModel):
    tempo_map: list[TempoSegmentInput] = Field(min_length=1)


class HitPoint(ApiModel):
    id: str
    sample: int = Field(ge=0)
    time_sec: float
    acoustic_sample: int = Field(ge=0)
    chart_sample: int = Field(ge=0)
    detected_sample: int = Field(ge=0)
    refined_sample: int = Field(ge=0)
    snapped_sample: int = Field(ge=0)
    snap_error_ms: float
    band: HitBand
    confidence: float = Field(ge=0, le=1)
    salience: float = Field(ge=0, le=1)
    source: HitSource
    detector_votes: list[str]
    primary_stem: StemKind = "mix"
    stem_evidence: dict[StemKind, float] = Field(default_factory=dict)
    manually_edited: bool
    locked: bool
    created_at: datetime
    updated_at: datetime


class HitPointCreate(ApiModel):
    id: str | None = None
    sample: int = Field(ge=0)
    time_sec: float | None = Field(default=None, ge=0)
    acoustic_sample: int | None = Field(default=None, ge=0)
    chart_sample: int | None = Field(default=None, ge=0)
    detected_sample: int | None = Field(default=None, ge=0)
    refined_sample: int | None = Field(default=None, ge=0)
    snapped_sample: int | None = Field(default=None, ge=0)
    snap_error_ms: float = 0.0
    band: HitBand = "mid_hit"
    confidence: float = Field(default=1.0, ge=0, le=1)
    salience: float = Field(default=1.0, ge=0, le=1)
    source: HitSource = "manual"
    detector_votes: list[str] = Field(default_factory=list)
    primary_stem: StemKind = "mix"
    stem_evidence: dict[StemKind, float] = Field(default_factory=dict)
    manually_edited: bool = True
    locked: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None


class HitPointPatch(ApiModel):
    sample: int | None = Field(default=None, ge=0)
    acoustic_sample: int | None = Field(default=None, ge=0)
    chart_sample: int | None = Field(default=None, ge=0)
    detected_sample: int | None = Field(default=None, ge=0)
    refined_sample: int | None = Field(default=None, ge=0)
    snapped_sample: int | None = Field(default=None, ge=0)
    snap_error_ms: float | None = None
    band: HitBand | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    salience: float | None = Field(default=None, ge=0, le=1)
    source: HitSource | None = None
    detector_votes: list[str] | None = None
    primary_stem: StemKind | None = None
    stem_evidence: dict[StemKind, float] | None = None
    manually_edited: bool | None = None
    locked: bool | None = None


class HitPointBulkUpdate(ApiModel):
    hit_points: list[HitPointCreate]


class CandidateEvent(ApiModel):
    id: str
    sample: int = Field(ge=0)
    time_sec: float = Field(ge=0)
    acoustic_sample: int = Field(ge=0)
    chart_sample: int = Field(ge=0)
    snap_error_ms: float
    lane: CandidateLane
    source_evidence: dict[str, float] = Field(default_factory=dict)
    semantic_evidence: dict[str, float] = Field(default_factory=dict)
    confidence: float = Field(ge=0, le=1)
    status: CandidateStatus
    grid_type: str
    grid_confidence: float = Field(ge=0, le=1)
    source: str = "mix"
    generator: str = "analysis"
    character: str | None = None
    mora: str | None = None
    phoneme: str | None = None
    event_level: str = "analysis"
    event_policy: str | None = None
    alignment_unit_id: str | None = None
    alignment_unit_index: int | None = Field(default=None, ge=0)
    alignment_run_id: str | None = None
    character_indices: list[int] = Field(default_factory=list)
    phonemes: list[str] = Field(default_factory=list)
    aligned_sample: int = Field(ge=0)
    refined_sample: int = Field(ge=0)
    evidence: dict[str, float] = Field(default_factory=dict)
    hit_point_id: str | None = None
    created_at: datetime
    updated_at: datetime


class AnalysisMetadata(ApiModel):
    version: str
    mode: AnalysisMode
    parameters: dict[str, Any]
    elapsed_ms: float
    bpm_confidence: float = Field(ge=0, le=1)
    warnings: list[str]
    created_at: datetime


class JobError(ApiModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


VocalLyricsStage = Literal[
    "idle",
    "queued",
    "separating_vocals",
    "detecting_vocal_activity",
    "transcribing",
    "normalizing_pronunciation",
    "aligning_lyrics",
    "refining_samples",
    "saving_results",
    "completed",
]
VocalLyricsStatus = Literal[
    "empty", "draft", "saved", "queued", "processing", "completed", "failed"
]


class VocalLyricsAnchor(ApiModel):
    id: str
    index: int = Field(ge=0)
    original_text: str
    kana: str = ""
    romaji: str = ""
    aligned_sample: int = Field(ge=0)
    refined_sample: int = Field(ge=0)
    grid_sample: int | None = Field(default=None, ge=0)
    confidence: float = Field(ge=0, le=1)
    word_start: bool = True
    active: bool = True
    chart_candidate: bool = False
    activity_score: float = Field(default=0.0, ge=0, le=1)
    attack_score: float = Field(default=0.0, ge=0, le=1)
    pitch_score: float = Field(default=0.0, ge=0, le=1)
    transition_score: float = Field(default=0.0, ge=0, le=1)
    acoustic_confidence: float = Field(default=0.0, ge=0, le=1)
    alignment_shift_ms: float = Field(default=0.0, ge=0)
    chunk_match_confidence: float = Field(default=1.0, ge=0, le=1)
    chunk_index: int = -1
    semantic_unit: Literal["phrase"] = "phrase"


class VocalAlignmentChunk(ApiModel):
    index: int = Field(ge=0)
    start_sample: int = Field(ge=0)
    end_sample: int = Field(gt=0)
    status: Literal[
        "success",
        "silent",
        "asr_failed",
        "unassigned",
        "alignment_failed",
        "alignment_collapse",
        "insufficient_anchors",
        "low_confidence",
    ]
    confidence: float = Field(ge=0, le=1)
    anchor_count: int = Field(ge=0)
    raw_timestamp_count: int = Field(default=0, ge=0)


class VocalLyricsResponse(ApiModel):
    track_id: str
    text: str = ""
    input_format: LyricsInputFormat = "japanese"
    status: VocalLyricsStatus = "empty"
    stage: VocalLyricsStage = "idle"
    progress: float = Field(default=0.0, ge=0, le=1)
    anchors: list[VocalLyricsAnchor] = Field(default_factory=list)
    coverage_chunks: list[VocalAlignmentChunk] = Field(default_factory=list)
    error: JobError | None = None
    updated_at: datetime | None = None


class VocalLyricsUpdate(ApiModel):
    text: str = Field(max_length=100_000)
    input_format: LyricsInputFormat = "japanese"


class VocalLyricsDraftRequest(ApiModel):
    input_format: LyricsInputFormat = "japanese"


class VocalLyricsJobResponse(ApiModel):
    job_id: str
    status: Literal["queued", "processing", "completed", "failed"]


class VocalLyricsJob(ApiModel):
    id: str
    track_id: str
    kind: Literal["alignment", "asr_draft"]
    status: Literal["queued", "processing", "completed", "failed"]
    stage: VocalLyricsStage
    progress: float = Field(ge=0, le=1)
    stage_timings: dict[str, float]
    warnings: list[str] = Field(default_factory=list)
    error: JobError | None
    result: VocalLyricsResponse | None = None
    created_at: datetime
    updated_at: datetime


class TrackResponse(ApiModel):
    id: str
    project_id: str
    original_file_name: str
    audio_url: str
    waveform_url: str
    format: str
    original_sample_rate: int
    channels: int
    sample_count: int
    duration_sec: float
    leading_silence_samples: int
    analysis: dict[str, Any]
    tempo_map: list[TempoSegment]
    hit_points: list[HitPoint]
    candidate_events: list[CandidateEvent] = Field(default_factory=list)
    stems: list[StemDescriptor] = Field(default_factory=list)
    focus_map: list[FocusSegment] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class ProjectDetail(ProjectResponse):
    track: TrackResponse | None = None


class ProjectList(ApiModel):
    items: list[ProjectResponse]
    total: int


class UploadResponse(ApiModel):
    project: ProjectResponse
    track: TrackResponse


class AnalyzeRequest(ApiModel):
    mode: AnalysisMode = "balanced"
    sensitivity: float = Field(default=0.5, ge=0, le=1)


class AnalyzeResponse(ApiModel):
    job_id: str
    status: Literal["queued", "processing", "completed", "failed"]


class AnalysisJob(ApiModel):
    id: str
    track_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    stage: str
    progress: float = Field(ge=0, le=1)
    mode: AnalysisMode
    sensitivity: float
    stage_timings: dict[str, float]
    warnings: list[str]
    error: JobError | None
    created_at: datetime
    updated_at: datetime


class WaveformResponse(ApiModel):
    track_id: str
    sample_rate: int
    sample_count: int
    source: StemKind = "mix"
    level: int
    window_size: int
    mins: list[float]
    maxs: list[float]


class ErrorBody(ApiModel):
    code: str
    message: str
    details: Any | None = None


class ErrorResponse(ApiModel):
    error: ErrorBody
