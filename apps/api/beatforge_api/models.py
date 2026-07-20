from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import Boolean, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def new_id() -> str:
    return str(uuid.uuid4())


def utc_now() -> datetime:
    return datetime.now(UTC)


class ProjectModel(Base):
    __tablename__ = "projects"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    title: Mapped[str] = mapped_column(String(300), nullable=False, index=True)
    artist: Mapped[str] = mapped_column(String(300), nullable=False, default="未知艺术家")
    genre: Mapped[str] = mapped_column(String(120), nullable=False, default="Unknown")
    cover_url: Mapped[str] = mapped_column(String(1000), nullable=False, default="")
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="unprocessed", index=True
    )
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(default=utc_now, onupdate=utc_now, nullable=False)

    track: Mapped[TrackModel | None] = relationship(
        back_populates="project", cascade="all, delete-orphan", uselist=False
    )


class TrackModel(Base):
    __tablename__ = "tracks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    project_id: Mapped[str] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), unique=True, nullable=False, index=True
    )
    original_file_name: Mapped[str] = mapped_column(String(500), nullable=False)
    stored_file_name: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    file_path: Mapped[str] = mapped_column(Text, nullable=False)
    format: Mapped[str] = mapped_column(String(20), nullable=False)
    original_sample_rate: Mapped[int] = mapped_column(Integer, nullable=False)
    channels: Mapped[int] = mapped_column(Integer, nullable=False)
    sample_count: Mapped[int] = mapped_column(Integer, nullable=False)
    duration_sec: Mapped[float] = mapped_column(Float, nullable=False)
    leading_silence_samples: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    analysis_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    lyrics_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    lyrics_format: Mapped[str] = mapped_column(String(24), nullable=False, default="japanese")
    vocal_alignment_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    waveform_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(default=utc_now, onupdate=utc_now, nullable=False)

    project: Mapped[ProjectModel] = relationship(back_populates="track")
    tempo_segments: Mapped[list[TempoSegmentModel]] = relationship(
        back_populates="track",
        cascade="all, delete-orphan",
        order_by="TempoSegmentModel.start_sample",
    )
    hit_points: Mapped[list[HitPointModel]] = relationship(
        back_populates="track", cascade="all, delete-orphan", order_by="HitPointModel.sample"
    )
    candidate_events: Mapped[list[CandidateEventModel]] = relationship(
        back_populates="track",
        cascade="all, delete-orphan",
        order_by="CandidateEventModel.acoustic_sample",
    )
    jobs: Mapped[list[AnalysisJobModel]] = relationship(
        back_populates="track", cascade="all, delete-orphan"
    )
    vocal_alignment_jobs: Mapped[list[VocalAlignmentJobModel]] = relationship(
        back_populates="track", cascade="all, delete-orphan"
    )


class TempoSegmentModel(Base):
    __tablename__ = "tempo_segments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    track_id: Mapped[str] = mapped_column(
        ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    start_sample: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    bpm: Mapped[float] = mapped_column(Float, nullable=False)
    time_signature_numerator: Mapped[int] = mapped_column(Integer, nullable=False, default=4)
    time_signature_denominator: Mapped[int] = mapped_column(Integer, nullable=False, default=4)
    beat_offset_sample: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    manually_edited: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    track: Mapped[TrackModel] = relationship(back_populates="tempo_segments")


class HitPointModel(Base):
    __tablename__ = "hit_points"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    track_id: Mapped[str] = mapped_column(
        ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sample: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    acoustic_sample: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    chart_sample: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    detected_sample: Mapped[int] = mapped_column(Integer, nullable=False)
    refined_sample: Mapped[int] = mapped_column(Integer, nullable=False)
    snapped_sample: Mapped[int] = mapped_column(Integer, nullable=False)
    snap_error_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    band: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    salience: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="fused")
    detector_votes_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    primary_stem: Mapped[str] = mapped_column(String(24), nullable=False, default="mix")
    stem_evidence_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    manually_edited: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    locked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(default=utc_now, onupdate=utc_now, nullable=False)

    track: Mapped[TrackModel] = relationship(back_populates="hit_points")


class CandidateEventModel(Base):
    __tablename__ = "candidate_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    track_id: Mapped[str] = mapped_column(
        ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    hit_point_id: Mapped[str | None] = mapped_column(
        ForeignKey("hit_points.id", ondelete="SET NULL"), nullable=True, index=True
    )
    sample: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    acoustic_sample: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    chart_sample: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    snap_error_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    lane: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    source_evidence_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    semantic_evidence_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    status: Mapped[str] = mapped_column(String(24), nullable=False, index=True)
    grid_type: Mapped[str] = mapped_column(String(24), nullable=False, default="straight_1_16")
    grid_confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # ``source`` describes the audible source while ``generator`` is stable
    # provenance.  Keeping those concepts separate lets HuBERT candidates say
    # source=vocals while refreshes only replace generator=hubert_ctc rows.
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="mix", index=True)
    generator: Mapped[str] = mapped_column(
        String(32), nullable=False, default="analysis", index=True
    )
    character: Mapped[str | None] = mapped_column(Text, nullable=True)
    mora: Mapped[str | None] = mapped_column(Text, nullable=True)
    phoneme: Mapped[str | None] = mapped_column(Text, nullable=True)
    event_level: Mapped[str] = mapped_column(
        String(24), nullable=False, default="analysis", index=True
    )
    event_policy: Mapped[str | None] = mapped_column(String(48), nullable=True)
    alignment_unit_id: Mapped[str | None] = mapped_column(String(160), nullable=True)
    alignment_unit_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    alignment_run_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    character_indices_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    phonemes_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    aligned_sample: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    refined_sample: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    evidence_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(default=utc_now, onupdate=utc_now, nullable=False)

    track: Mapped[TrackModel] = relationship(back_populates="candidate_events")


class AnalysisJobModel(Base):
    __tablename__ = "analysis_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    track_id: Mapped[str] = mapped_column(
        ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="queued", index=True)
    stage: Mapped[str] = mapped_column(String(80), nullable=False, default="queued")
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    mode: Mapped[str] = mapped_column(String(24), nullable=False, default="balanced")
    sensitivity: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)
    stage_timings_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    warnings_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    error_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(default=utc_now, onupdate=utc_now, nullable=False)

    track: Mapped[TrackModel] = relationship(back_populates="jobs")


class VocalAlignmentJobModel(Base):
    __tablename__ = "vocal_alignment_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    track_id: Mapped[str] = mapped_column(
        ForeignKey("tracks.id", ondelete="CASCADE"), nullable=False, index=True
    )
    operation: Mapped[str] = mapped_column(String(24), nullable=False, default="align")
    replace_vocal_hits: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="queued", index=True)
    stage: Mapped[str] = mapped_column(String(80), nullable=False, default="queued")
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    stage_timings_json: Mapped[str] = mapped_column(Text, nullable=False, default="{}")
    warnings_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    error_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(default=utc_now, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(default=utc_now, onupdate=utc_now, nullable=False)

    track: Mapped[TrackModel] = relationship(back_populates="vocal_alignment_jobs")
