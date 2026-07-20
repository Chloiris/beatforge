from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from .models import (
    AnalysisJobModel,
    CandidateEventModel,
    HitPointModel,
    ProjectModel,
    TempoSegmentModel,
    TrackModel,
    VocalAlignmentJobModel,
)
from .timing import sample_to_time


def load_json(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return fallback


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return json_safe(value.item())
    if hasattr(value, "tolist"):
        return json_safe(value.tolist())
    return str(value)


def dumps(value: Any) -> str:
    return json.dumps(json_safe(value), ensure_ascii=False, separators=(",", ":"))


def _camel_key(value: str) -> str:
    head, *tail = value.split("_")
    return head + "".join(part.capitalize() for part in tail)


def camelize_keys(value: Any) -> Any:
    if isinstance(value, dict):
        return {_camel_key(str(key)): camelize_keys(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [camelize_keys(item) for item in value]
    return value


def project_dict(project: ProjectModel) -> dict[str, Any]:
    track = project.track
    analysis = camelize_keys(load_json(track.analysis_json, {})) if track else {}
    tempo_segments = track.tempo_segments if track else []
    bpm = tempo_segments[0].bpm if tempo_segments else analysis.get("estimatedBpm")
    return {
        "id": project.id,
        "title": project.title,
        "artist": project.artist,
        "genre": project.genre,
        "cover_url": project.cover_url,
        "status": project.status,
        "created_at": project.created_at,
        "updated_at": project.updated_at,
        "track_id": track.id if track else None,
        "bpm": bpm,
        "duration_sec": track.duration_sec if track else None,
        "hit_point_count": len(track.hit_points) if track else 0,
        "analysis_mode": analysis.get("mode") if analysis else None,
    }


def tempo_dict(segment: TempoSegmentModel) -> dict[str, Any]:
    return {
        "id": segment.id,
        "start_sample": segment.start_sample,
        "bpm": segment.bpm,
        "time_signature_numerator": segment.time_signature_numerator,
        "time_signature_denominator": segment.time_signature_denominator,
        "beat_offset_sample": segment.beat_offset_sample,
        "confidence": segment.confidence,
        "manually_edited": segment.manually_edited,
    }


def hit_dict(hit: HitPointModel, sample_rate: int) -> dict[str, Any]:
    acoustic_sample = (
        hit.acoustic_sample if hit.acoustic_sample is not None else hit.refined_sample
    )
    chart_sample = hit.chart_sample if hit.chart_sample is not None else hit.snapped_sample
    return {
        "id": hit.id,
        "sample": acoustic_sample,
        "time_sec": sample_to_time(acoustic_sample, sample_rate),
        "acoustic_sample": acoustic_sample,
        "chart_sample": chart_sample,
        "detected_sample": hit.detected_sample,
        "refined_sample": hit.refined_sample,
        "snapped_sample": hit.snapped_sample,
        "snap_error_ms": hit.snap_error_ms,
        "band": hit.band,
        "confidence": hit.confidence,
        "salience": hit.salience,
        "source": hit.source,
        "detector_votes": load_json(hit.detector_votes_json, []),
        "primary_stem": hit.primary_stem,
        "stem_evidence": load_json(hit.stem_evidence_json, {}),
        "manually_edited": hit.manually_edited,
        "locked": hit.locked,
        "created_at": hit.created_at,
        "updated_at": hit.updated_at,
    }


def candidate_event_dict(candidate: CandidateEventModel, sample_rate: int) -> dict[str, Any]:
    aligned_sample = (
        candidate.aligned_sample
        if candidate.aligned_sample is not None
        else candidate.acoustic_sample
    )
    refined_sample = (
        candidate.refined_sample
        if candidate.refined_sample is not None
        else candidate.acoustic_sample
    )
    return {
        "id": candidate.id,
        "sample": candidate.acoustic_sample,
        "time_sec": sample_to_time(candidate.acoustic_sample, sample_rate),
        "acoustic_sample": candidate.acoustic_sample,
        "chart_sample": candidate.chart_sample,
        "snap_error_ms": candidate.snap_error_ms,
        "lane": candidate.lane,
        "source_evidence": load_json(candidate.source_evidence_json, {}),
        "semantic_evidence": load_json(candidate.semantic_evidence_json, {}),
        "confidence": candidate.confidence,
        "status": candidate.status,
        "grid_type": candidate.grid_type,
        "grid_confidence": candidate.grid_confidence,
        "source": candidate.source,
        "generator": candidate.generator,
        "character": candidate.character,
        "mora": candidate.mora,
        "phoneme": candidate.phoneme,
        "event_level": candidate.event_level,
        "event_policy": candidate.event_policy,
        "alignment_unit_id": candidate.alignment_unit_id,
        "alignment_unit_index": candidate.alignment_unit_index,
        "alignment_run_id": candidate.alignment_run_id,
        "character_indices": load_json(candidate.character_indices_json, []),
        "phonemes": load_json(candidate.phonemes_json, []),
        "aligned_sample": aligned_sample,
        "refined_sample": refined_sample,
        "evidence": load_json(candidate.evidence_json, {}),
        "hit_point_id": candidate.hit_point_id,
        "created_at": candidate.created_at,
        "updated_at": candidate.updated_at,
    }


def track_dict(track: TrackModel, include_hit_points: bool = True) -> dict[str, Any]:
    analysis = camelize_keys(load_json(track.analysis_json, {}))
    return {
        "id": track.id,
        "project_id": track.project_id,
        "original_file_name": track.original_file_name,
        "audio_url": f"/api/tracks/{track.id}/audio",
        "waveform_url": f"/api/tracks/{track.id}/waveform",
        "format": track.format,
        "original_sample_rate": track.original_sample_rate,
        "channels": track.channels,
        "sample_count": track.sample_count,
        "duration_sec": track.duration_sec,
        "leading_silence_samples": track.leading_silence_samples,
        "analysis": analysis,
        "tempo_map": [tempo_dict(segment) for segment in track.tempo_segments],
        "hit_points": (
            [hit_dict(hit, track.original_sample_rate) for hit in track.hit_points]
            if include_hit_points
            else []
        ),
        "candidate_events": (
            [
                candidate_event_dict(candidate, track.original_sample_rate)
                for candidate in track.candidate_events
            ]
            if include_hit_points
            else []
        ),
        "stems": analysis.get("stems", []),
        "focus_map": analysis.get("focusMap", []),
        "created_at": track.created_at,
        "updated_at": track.updated_at,
    }


def job_dict(job: AnalysisJobModel) -> dict[str, Any]:
    return {
        "id": job.id,
        "track_id": job.track_id,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "mode": job.mode,
        "sensitivity": job.sensitivity,
        "stage_timings": load_json(job.stage_timings_json, {}),
        "warnings": load_json(job.warnings_json, []),
        "error": load_json(job.error_json, None),
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }


def vocal_job_dict(job: VocalAlignmentJobModel) -> dict[str, Any]:
    payload = {
        "id": job.id,
        "track_id": job.track_id,
        "kind": job.operation,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "stage_timings": load_json(job.stage_timings_json, {}),
        "warnings": load_json(job.warnings_json, []),
        "error": load_json(job.error_json, None),
        "created_at": job.created_at,
        "updated_at": job.updated_at,
    }
    payload["result"] = (
        vocal_lyrics_dict(job.track)
        if job.status == "completed" and getattr(job, "track", None) is not None
        else None
    )
    return payload


def vocal_lyrics_dict(track: TrackModel) -> dict[str, Any]:
    payload = load_json(track.vocal_alignment_json, {})
    status = str(payload.get("status", "saved" if track.lyrics_text.strip() else "empty"))
    stage = str(payload.get("stage", "idle"))
    progress = float(payload.get("progress", 0.0))
    if status == "completed":
        stage = "completed"
        progress = 1.0
    return {
        "track_id": track.id,
        "text": track.lyrics_text,
        "input_format": track.lyrics_format,
        "status": status,
        "stage": stage,
        "progress": min(1.0, max(0.0, progress)),
        "anchors": payload.get("anchors", []),
        "coverage_chunks": payload.get("coverage_chunks", []),
        "error": payload.get("error"),
        "updated_at": payload.get("updated_at", track.updated_at),
    }
