from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from ..schemas import ApiModel

ChartMode = Literal["pump-single", "pump-double"]
NoteType = Literal["tap", "hold", "mine"]
IssueSeverity = Literal["info", "warning", "error"]


class TempoPoint(ApiModel):
    beat: float = Field(ge=0)
    # Some real-world SM files use extreme BPM values to encode stops/warps.
    bpm: float = Field(gt=0, le=10_000_000)
    time_sec: float


class ChartNote(ApiModel):
    lane: int = Field(ge=0, le=9)
    type: NoteType = "tap"
    end_time_sec: float | None = None
    end_beat: float | None = None
    source: str = "sm"
    confidence: float = Field(default=1.0, ge=0, le=1)
    foot: Literal["left", "right"] | None = None

    @model_validator(mode="after")
    def validate_hold(self) -> ChartNote:
        if self.type == "hold":
            if self.end_time_sec is None or self.end_beat is None:
                raise ValueError("hold notes require end_time_sec and end_beat")
        return self


class ChartEvent(ApiModel):
    time_sec: float
    beat: float = Field(ge=0)
    measure: int = Field(ge=0)
    subdivision: int = Field(default=4, ge=1)
    row_index: int | None = Field(default=None, ge=0)
    notes: list[ChartNote] = Field(min_length=1)
    source_event_id: str | None = None
    # A quantized chart row can merge several BeatForge inputs. Keep the
    # legacy primary id above for existing consumers, while retaining the
    # complete provenance needed to audit anchor coverage.
    source_event_ids: list[str] = Field(default_factory=list)
    source_hit_point_ids: list[str] = Field(default_factory=list)
    # 0 = optional/model-selected, 1 = BeatForge rhythm marker (accepted
    # candidates and non-rejected aligned vocal morae), 2 = confirmed hit point.
    anchor_priority: int = Field(default=0, ge=0, le=2)
    pattern: str | None = None


class ChartStatistics(ApiModel):
    note_count: int = Field(ge=0)
    event_count: int = Field(ge=0)
    hold_count: int = Field(ge=0)
    jump_count: int = Field(ge=0)
    mine_count: int = Field(ge=0)
    duration_sec: float = Field(ge=0)
    nps_average: float = Field(ge=0)
    nps_peak: float = Field(ge=0)
    single_ratio: float = Field(ge=0, le=1)
    jump_ratio: float = Field(ge=0, le=1)
    hold_ratio: float = Field(ge=0, le=1)
    lane_counts: list[int]
    measure_densities: list[float]
    same_foot_runs: list[int] = Field(default_factory=list)
    foot_switch_ratio: float = Field(default=0, ge=0, le=1)
    small_spin_count: int = Field(default=0, ge=0)
    big_spin_count: int = Field(default=0, ge=0)


class ValidationIssue(ApiModel):
    code: str
    severity: IssueSeverity
    message: str
    time_sec: float | None = None
    beat: float | None = None
    penalty: float = Field(default=0, ge=0)


class ValidationResult(ApiModel):
    valid: bool
    score: float = Field(ge=0, le=100)
    issues: list[ValidationIssue]
    metrics: dict[str, Any] = Field(default_factory=dict)


class ChartDocument(ApiModel):
    id: str
    title: str
    artist: str = ""
    music: str = ""
    source_group: str | None = None
    source_path: str | None = None
    mode: ChartMode = "pump-single"
    lane_count: int = Field(default=5, ge=5, le=10)
    difficulty: str = "Hard"
    meter: int = Field(default=1, ge=1, le=99)
    bpm: float = Field(gt=0, le=500)
    offset_sec: float = 0
    duration_sec: float = Field(default=0, ge=0)
    measure_count: int = Field(default=0, ge=0)
    tempo_map: list[TempoPoint] = Field(min_length=1)
    events: list[ChartEvent]
    statistics: ChartStatistics | None = None
    validation: ValidationResult | None = None
    optimization: dict[str, int] | None = None
    model_provenance: dict[str, Any] | None = None
    generator: str = "sm_parser"
    generator_version: str = "1.0"
    seed: int | None = None
    spin_enabled: bool = False

    @model_validator(mode="after")
    def validate_lanes(self) -> ChartDocument:
        if self.mode == "pump-single" and self.lane_count != 5:
            raise ValueError("pump-single charts require five lanes")
        if self.mode == "pump-double" and self.lane_count != 10:
            raise ValueError("pump-double charts require ten lanes")
        for event in self.events:
            if any(note.lane >= self.lane_count for note in event.notes):
                raise ValueError("note lane exceeds chart lane count")
        return self


class ReferenceChartSummary(ApiModel):
    id: str
    title: str
    group: str
    mode: ChartMode
    lane_count: int
    difficulty: str
    meter: int
    bpm: float
    bpm_max: float
    offset_sec: float
    duration_sec: float
    note_count: int
    event_count: int
    nps_average: float
    nps_peak: float
    audio_url: str
    chart_url: str


class CorpusStatistics(ApiModel):
    chart_count: int
    song_count: int
    single_chart_count: int
    single_song_count: int
    double_chart_count: int
    double_song_count: int
    difficulty_min: int
    difficulty_max: int
    total_notes: int
    total_duration_sec: float
    average_nps: float
    groups: dict[str, int]
    lane_transition_probabilities: list[list[float]]
    meter_profiles: dict[str, dict[str, float]]


class GenerateChartRequest(ApiModel):
    difficulty: int = Field(default=8, ge=1, le=15)
    enable_spin: bool = False
    use_local_model: bool = True
    seed: int | None = None


class ChartGenerationResponse(ApiModel):
    generation_id: str
    chart: ChartDocument
    reference_corpus: dict[str, Any]
