from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from bisect import bisect_left
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from .config import get_settings
from .database import SessionLocal
from .media import prepare_analysis_source
from .models import (
    AnalysisJobModel,
    CandidateEventModel,
    HitPointModel,
    TempoSegmentModel,
    TrackModel,
    new_id,
)
from .serialization import camelize_keys, dumps, hit_dict, json_safe
from .storage_paths import resolve_storage_path, storage_relative_path
from .timing import nearest_grid_sample
from .waveform_store import write_waveform_lods

# A Demucs analysis peaks near 2 GB on the supported CPU path. Serializing local
# jobs prevents two accurate analyses from exhausting a typical desktop.
_executor = ThreadPoolExecutor(max_workers=1)
_futures: dict[str, Future[None]] = {}
_future_lock = threading.Lock()

PRESERVED_HIT_MERGE_WINDOW_MS = 9.0


def _value(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _update_job(job_id: str, **values: Any) -> None:
    with SessionLocal() as session:
        job = session.get(AnalysisJobModel, job_id)
        if not job:
            return
        for key, value in values.items():
            setattr(job, key, value)
        job.updated_at = datetime.now(UTC)
        session.commit()


def _write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(json_safe(value), handle, ensure_ascii=False, separators=(",", ":"))
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _write_audio_atomic(path: Path, values: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    try:
        sf.write(temporary_name, np.asarray(values, dtype=np.float32), sample_rate)
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _normalize_hit(
    raw: Any,
    *,
    sample_rate: int,
    sample_count: int,
    bpm: float,
    offset: int,
) -> dict[str, Any]:
    refined = int(_value(raw, "refined_sample", _value(raw, "sample", 0)))
    acoustic = min(
        max(0, int(_value(raw, "acoustic_sample", refined))),
        max(0, sample_count - 1),
    )
    detected = min(
        max(0, int(_value(raw, "detected_sample", acoustic))),
        max(0, sample_count - 1),
    )
    refined = min(max(0, refined), max(0, sample_count - 1))
    snapped = int(
        _value(
            raw,
            "snapped_sample",
            nearest_grid_sample(
                acoustic,
                sample_rate=sample_rate,
                bpm=max(1.0, bpm),
                beat_offset_sample=offset,
                subdivisions_per_beat=4,
            ),
        )
    )
    snapped = min(max(0, snapped), max(0, sample_count - 1))
    chart = min(
        max(0, int(_value(raw, "chart_sample", snapped))),
        max(0, sample_count - 1),
    )
    band = str(_value(raw, "band", "mid_hit"))
    if band not in {"low_hit", "mid_hit", "high_hit", "full_band_accent"}:
        band = "mid_hit"
    source = str(_value(raw, "source", "fused"))
    if source not in {"mix", "percussive", "stems", "fused", "manual"}:
        source = "fused"
    votes = _value(raw, "detector_votes", [])
    primary_stem = str(_value(raw, "primary_stem", "mix"))
    if primary_stem not in {"mix", "vocals", "drums", "bass", "other"}:
        primary_stem = "mix"
    raw_stem_evidence = _value(raw, "stem_evidence", {})
    stem_evidence = (
        {
            str(name): min(1.0, max(0.0, float(value)))
            for name, value in raw_stem_evidence.items()
            if str(name) in {"mix", "vocals", "drums", "bass", "other"}
        }
        if isinstance(raw_stem_evidence, dict)
        else {}
    )
    return {
        "id": str(_value(raw, "id", None) or new_id()),
        "candidate_event_id": _value(raw, "candidate_event_id", None),
        "sample": acoustic,
        "acoustic_sample": acoustic,
        "chart_sample": chart,
        "detected_sample": detected,
        "refined_sample": refined,
        "snapped_sample": snapped,
        "snap_error_ms": float(
            _value(raw, "snap_error_ms", (acoustic - chart) * 1000.0 / sample_rate)
        ),
        "band": band,
        "confidence": min(1.0, max(0.0, float(_value(raw, "confidence", 0.0)))),
        "salience": min(1.0, max(0.0, float(_value(raw, "salience", 0.0)))),
        "source": source,
        "detector_votes": [str(item) for item in votes],
        "primary_stem": primary_stem,
        "stem_evidence": stem_evidence,
    }


def _is_near_preserved_sample(
    sample: int, preserved_samples: list[int], tolerance_samples: int
) -> bool:
    """Return whether an analysis hit would duplicate a preserved user hit."""
    if not preserved_samples:
        return False
    index = bisect_left(preserved_samples, sample)
    return any(
        abs(preserved_samples[candidate] - sample) <= tolerance_samples
        for candidate in (index - 1, index)
        if 0 <= candidate < len(preserved_samples)
    )


def _candidate_event_from_hit(hit: dict[str, Any]) -> CandidateEventModel:
    primary_stem = str(hit.get("primary_stem", "mix"))
    lane = {
        "vocals": "vocals",
        "drums": "drums",
        "other": "melody",
    }.get(primary_stem, "mix")
    raw_evidence = dict(hit.get("stem_evidence", {}))
    source_evidence = {
        "vocals": float(raw_evidence.get("vocals", 0.0)),
        "melody": float(raw_evidence.get("other", raw_evidence.get("melody", 0.0))),
        "drums": float(raw_evidence.get("drums", 0.0)),
        "mix": float(raw_evidence.get("mix", 1.0 if lane == "mix" else 0.0)),
    }
    snap_error_ms = float(hit.get("snap_error_ms", 0.0))
    grid_confidence = float(np.exp(-0.5 * (abs(snap_error_ms) / 30.0) ** 2))
    hit_id = str(hit["id"])
    candidate_id = str(
        hit.get("candidate_event_id")
        or uuid.uuid5(uuid.NAMESPACE_URL, f"candidate:{hit_id}")
    )
    return CandidateEventModel(
        id=candidate_id,
        hit_point_id=hit_id,
        sample=int(hit["acoustic_sample"]),
        acoustic_sample=int(hit["acoustic_sample"]),
        chart_sample=int(hit["chart_sample"]),
        snap_error_ms=snap_error_ms,
        lane=lane,
        source_evidence_json=dumps(source_evidence),
        semantic_evidence_json=dumps(
            {
                "lyricAlignment": 0.0,
                "phonemeConfidence": 0.0,
                "pitchConfidence": 0.0,
                "beatConfidence": grid_confidence,
            }
        ),
        confidence=float(hit["confidence"]),
        status="accepted",
        grid_type="straight_1_16",
        grid_confidence=grid_confidence,
    )


def _candidate_event_from_payload(
    payload: dict[str, Any],
    accepted_hit: dict[str, Any] | None,
) -> CandidateEventModel:
    status = str(payload.get("status", "uncertain"))
    if status not in {"accepted", "rejected", "uncertain"}:
        status = "uncertain"
    if status == "accepted" and accepted_hit is None:
        status = "rejected"
    acoustic_sample = int(
        accepted_hit["acoustic_sample"]
        if accepted_hit is not None
        else payload.get("acoustic_sample", payload.get("sample", 0))
    )
    chart_sample = int(
        accepted_hit["chart_sample"]
        if accepted_hit is not None
        else payload.get("chart_sample", payload.get("sample", 0))
    )
    snap_error_ms = float(
        accepted_hit["snap_error_ms"]
        if accepted_hit is not None
        else payload.get("snap_error_ms", 0.0)
    )
    grid_confidence = float(np.exp(-0.5 * (abs(snap_error_ms) / 30.0) ** 2))
    semantic_evidence = dict(payload.get("semantic_evidence", {}))
    semantic_evidence["beatConfidence"] = grid_confidence
    return CandidateEventModel(
        id=str(payload["id"]),
        hit_point_id=str(accepted_hit["id"]) if accepted_hit is not None else None,
        sample=acoustic_sample,
        acoustic_sample=acoustic_sample,
        chart_sample=chart_sample,
        snap_error_ms=snap_error_ms,
        lane=str(payload.get("lane", "mix")),
        source_evidence_json=dumps(dict(payload.get("source_evidence", {}))),
        semantic_evidence_json=dumps(semantic_evidence),
        confidence=float(np.clip(payload.get("confidence", 0.0), 0.0, 1.0)),
        status=status,
        grid_type=str(payload.get("grid_type", "straight_1_16")),
        grid_confidence=grid_confidence,
    )


def _run_analysis(job_id: str) -> None:
    settings = get_settings()
    started = time.perf_counter()
    decoded_path: Path | None = None
    decoded_is_temporary = False
    stage_starts: dict[str, float] = {}
    stage_timings: dict[str, float] = {}
    current_stage: str | None = None
    last_progress = 0.0

    def progress(stage: str, amount: float, _detail: Any = None) -> None:
        nonlocal current_stage, last_progress
        now = time.perf_counter()
        if stage != current_stage:
            if current_stage is not None:
                stage_timings[current_stage] = round(
                    (now - stage_starts[current_stage]) * 1000.0, 3
                )
            current_stage = stage
            stage_starts[stage] = now
        last_progress = max(last_progress, min(0.97, max(0.01, float(amount))))
        _update_job(
            job_id,
            status="processing",
            stage=stage,
            progress=last_progress,
            stage_timings_json=dumps(stage_timings),
        )

    try:
        with SessionLocal() as session:
            job = session.get(AnalysisJobModel, job_id)
            if not job:
                return
            track = session.get(TrackModel, job.track_id)
            if not track:
                raise RuntimeError("track disappeared before analysis started")
            source_path = resolve_storage_path(track.file_path, settings)
            mode = job.mode
            sensitivity = job.sensitivity
        progress("decoding_audio", 0.03)
        decoded_path, decoded_is_temporary = prepare_analysis_source(
            source_path, settings.analyses_dir / "decoded"
        )

        from .audio import (
            analyze_audio,
            build_waveform_lods,
            constrain_hits_to_rhythm_grid,
            waveform_lods_from_samples,
        )

        def analysis_progress(stage: str, amount: float, detail: Any = None) -> None:
            # The core owns decoding through result serialization (3–100%). The
            # worker still has waveform persistence and database writes afterward,
            # so map that real subtask monotonically into 5–80% of the whole job.
            progress(stage, 0.05 + min(1.0, max(0.0, amount)) * 0.75, detail)

        result = analyze_audio(
            decoded_path,
            mode=mode,
            sensitivity=sensitivity,
            progress_callback=analysis_progress,
        )
        progress("extracting_waveform", 0.84)
        waveform_levels = build_waveform_lods(decoded_path)
        stem_audio = dict(_value(result, "stem_audio", {}) or {})

        sample_rate = int(_value(result, "original_sample_rate"))
        sample_count = int(_value(result, "sample_count"))
        channels = int(_value(result, "channels"))
        duration_sec = float(_value(result, "duration_sec", sample_count / sample_rate))
        leading_silence = int(_value(result, "leading_silence_samples", 0))
        bpm = float(_value(result, "bpm", 120.0))
        bpm_confidence = float(_value(result, "bpm_confidence", 0.0))
        beat_offset = int(_value(result, "beat_offset_sample", 0))
        warnings = [str(item) for item in _value(result, "warnings", [])]
        raw_hits = list(_value(result, "hit_points", []))
        raw_candidate_events = list(_value(result, "candidate_events", []))
        hits = [
            _normalize_hit(
                hit,
                sample_rate=sample_rate,
                sample_count=sample_count,
                bpm=bpm,
                offset=beat_offset,
            )
            for hit in raw_hits
        ]
        progress("saving_results", 0.94)
        waveform_payload = {
            "trackId": job.track_id,
            "sampleRate": sample_rate,
            "sampleCount": sample_count,
            "levels": waveform_levels,
            "stems": {
                name: {
                    "sampleRate": sample_rate,
                    "sampleCount": sample_count,
                    "levels": waveform_lods_from_samples(values),
                }
                for name, values in stem_audio.items()
                if name in {"vocals", "drums", "bass", "other"}
            },
        }
        waveform_path = settings.waveform_dir / f"{job.track_id}.json.gz"
        write_waveform_lods(waveform_path, waveform_payload)
        stem_directory = settings.stems_dir / job.track_id
        if stem_audio:
            stem_directory.mkdir(parents=True, exist_ok=True)
            for stem_name, values in stem_audio.items():
                if stem_name not in {"vocals", "drums", "bass", "other"}:
                    continue
                _write_audio_atomic(
                    stem_directory / f"{stem_name}.flac",
                    np.asarray(values, dtype=np.float32),
                    sample_rate,
                )

        elapsed_ms = (time.perf_counter() - started) * 1000.0
        result_timings = _value(result, "stage_timings_ms", {})
        if isinstance(result_timings, dict):
            stage_timings.update({str(key): float(value) for key, value in result_timings.items()})
        metadata = camelize_keys(dict(_value(result, "metadata", {}) or {}))
        metadata.update(
            {
                "version": str(metadata.get("version", "0.1.0")),
                "mode": mode,
                "parameters": metadata.get("parameters", {"sensitivity": sensitivity}),
                "elapsedMs": elapsed_ms,
                "bpmConfidence": bpm_confidence,
                "warnings": warnings,
                "createdAt": datetime.now(UTC).isoformat(),
            }
        )
        if stem_audio:
            metadata["stems"] = [
                {
                    "source": name,
                    "available": True,
                    "waveformUrl": (
                        f"/api/tracks/{job.track_id}/waveform?source={name}"
                    ),
                    "audioUrl": (
                        f"/api/tracks/{job.track_id}/stems/{name}/audio"
                    ),
                }
                for name in ("vocals", "drums", "bass", "other")
                if name in stem_audio
            ]
        else:
            metadata["stems"] = []
        analysis_payload = {
            **metadata,
            "estimatedBpm": bpm,
            "beatOffsetSample": beat_offset,
            "hitPointCount": len(hits),
            "stageTimingsMs": stage_timings,
        }

        with SessionLocal() as session:
            statement = (
                select(TrackModel)
                .where(TrackModel.id == job.track_id)
                .options(
                    selectinload(TrackModel.project),
                    selectinload(TrackModel.hit_points),
                    selectinload(TrackModel.candidate_events),
                    selectinload(TrackModel.tempo_segments),
                )
            )
            track = session.scalar(statement)
            if not track:
                raise RuntimeError("track disappeared while saving analysis")
            track.original_sample_rate = sample_rate
            track.channels = channels
            track.sample_count = sample_count
            track.duration_sec = duration_sec
            track.leading_silence_samples = leading_silence
            track.waveform_path = storage_relative_path(waveform_path, settings)

            preserved_hits = [
                hit
                for hit in track.hit_points
                if hit.manually_edited or hit.locked or hit.source == "manual"
            ]
            preserved_tempos = [
                segment for segment in track.tempo_segments if segment.manually_edited
            ]
            if mode == "accurate" and stem_audio:
                chart_tempo = (
                    min(preserved_tempos, key=lambda segment: segment.start_sample)
                    if preserved_tempos
                    else None
                )
                chart_bpm = float(chart_tempo.bpm) if chart_tempo else bpm
                chart_offset = (
                    int(chart_tempo.beat_offset_sample) if chart_tempo else beat_offset
                )
                chart_confidence = (
                    float(chart_tempo.confidence) if chart_tempo else bpm_confidence
                )
                hits, rhythm_constraint = constrain_hits_to_rhythm_grid(
                    hits,
                    sample_rate=sample_rate,
                    sample_count=sample_count,
                    bpm=chart_bpm,
                    beat_offset_sample=chart_offset,
                    tempo_confidence=chart_confidence,
                    tempo_source="manual" if chart_tempo else "estimated",
                )
                analysis_payload["rhythmConstraint"] = camelize_keys(rhythm_constraint)
                candidate_selection = analysis_payload.get("candidateSelection")
                if isinstance(candidate_selection, dict):
                    candidate_selection["selectedAfterRhythmConstraint"] = len(hits)
                if not rhythm_constraint["applied"]:
                    warning = (
                        "BPM 置信度较低，未自动执行 1/16 节奏量化；请先检查 BPM 和 offset。"
                    )
                    warnings.append(warning)
                    analysis_payload["warnings"] = warnings
            preserved_hit_ids = {hit.id for hit in preserved_hits}
            preserved_samples = sorted(hit.sample for hit in preserved_hits)
            merge_window_samples = max(
                1,
                round(sample_rate * PRESERVED_HIT_MERGE_WINDOW_MS / 1000.0),
            )
            new_hits = [
                hit
                for hit in hits
                if hit["id"] not in preserved_hit_ids
                and not _is_near_preserved_sample(
                    hit["sample"], preserved_samples, merge_window_samples
                )
            ]
            rhythm_metadata = analysis_payload.get("rhythmConstraint")
            if isinstance(rhythm_metadata, dict) and rhythm_metadata.get("applied"):
                rhythm_metadata["suppressedNearPreserved"] = len(hits) - len(new_hits)
                rhythm_metadata["outputCount"] = len(new_hits)
            track.candidate_events.clear()
            session.flush()
            for hit in list(track.hit_points):
                if hit.id not in preserved_hit_ids:
                    track.hit_points.remove(hit)

            preserved_tempo_ids = {segment.id for segment in preserved_tempos}
            for segment in list(track.tempo_segments):
                if segment.id not in preserved_tempo_ids:
                    track.tempo_segments.remove(segment)
            session.flush()
            if not preserved_tempos:
                track.tempo_segments.append(
                    TempoSegmentModel(
                        start_sample=0,
                        bpm=bpm,
                        time_signature_numerator=4,
                        time_signature_denominator=4,
                        beat_offset_sample=beat_offset,
                        confidence=min(1.0, max(0.0, bpm_confidence)),
                        manually_edited=False,
                    )
                )
            for hit in new_hits:
                track.hit_points.append(
                    HitPointModel(
                        id=hit["id"],
                        sample=hit["sample"],
                        acoustic_sample=hit["acoustic_sample"],
                        chart_sample=hit["chart_sample"],
                        detected_sample=hit["detected_sample"],
                        refined_sample=hit["refined_sample"],
                        snapped_sample=hit["snapped_sample"],
                        snap_error_ms=hit["snap_error_ms"],
                        band=hit["band"],
                        confidence=hit["confidence"],
                        salience=hit["salience"],
                        source=hit["source"],
                        detector_votes_json=dumps(hit["detector_votes"]),
                        primary_stem=hit["primary_stem"],
                        stem_evidence_json=dumps(hit["stem_evidence"]),
                        manually_edited=False,
                        locked=False,
                    )
                )
            session.flush()
            accepted_hit_by_candidate = {
                str(hit["candidate_event_id"]): hit
                for hit in new_hits
                if hit.get("candidate_event_id")
            }
            if raw_candidate_events:
                track.candidate_events.extend(
                    _candidate_event_from_payload(
                        candidate,
                        accepted_hit_by_candidate.get(str(candidate["id"])),
                    )
                    for candidate in raw_candidate_events
                )
            else:
                track.candidate_events.extend(
                    _candidate_event_from_hit(hit) for hit in new_hits
                )
            analysis_payload["hitPointCount"] = len(preserved_hits) + len(new_hits)
            track.analysis_json = dumps(analysis_payload)
            track.project.status = (
                "edited" if preserved_hits or preserved_tempos else "completed"
            )
            track.project.updated_at = datetime.now(UTC)
            session.commit()

            if not stem_audio and stem_directory.is_dir():
                shutil.rmtree(stem_directory)

            export_snapshot = {
                "trackId": track.id,
                "audio": {
                    "sampleRate": sample_rate,
                    "sampleCount": sample_count,
                    "durationSec": duration_sec,
                },
                "tempo": {
                    "bpm": bpm,
                    "bpmConfidence": bpm_confidence,
                    "beatOffsetSample": beat_offset,
                },
                "analysisMetadata": analysis_payload,
                "hitPoints": [hit_dict(item, sample_rate) for item in track.hit_points],
                "candidateEvents": [
                    {
                        "id": candidate.id,
                        "acousticSample": candidate.acoustic_sample,
                        "chartSample": candidate.chart_sample,
                        "snapErrorMs": candidate.snap_error_ms,
                        "lane": candidate.lane,
                        "status": candidate.status,
                    }
                    for candidate in track.candidate_events
                ],
            }
        _write_json_atomic(settings.analyses_dir / f"{job.track_id}.json", export_snapshot)
        if current_stage is not None:
            stage_timings[current_stage] = round(
                (time.perf_counter() - stage_starts[current_stage]) * 1000.0, 3
            )
        _update_job(
            job_id,
            status="completed",
            stage="completed",
            progress=1.0,
            stage_timings_json=dumps(stage_timings),
            warnings_json=dumps(warnings),
            error_json=None,
        )
    except Exception as exc:
        error = {
            "code": getattr(exc, "code", "ANALYSIS_FAILED"),
            "message": getattr(exc, "message", str(exc)) or "Audio analysis failed",
            "details": getattr(exc, "details", None),
        }
        _update_job(
            job_id,
            status="failed",
            stage="failed",
            error_json=dumps(error),
            stage_timings_json=dumps(stage_timings),
        )
        with SessionLocal() as session:
            job = session.get(AnalysisJobModel, job_id)
            if job and job.track and job.track.project:
                job.track.project.status = "failed"
                session.commit()
    finally:
        if decoded_is_temporary and decoded_path is not None:
            decoded_path.unlink(missing_ok=True)
        with _future_lock:
            _futures.pop(job_id, None)


def submit_analysis(job_id: str) -> None:
    with _future_lock:
        existing = _futures.get(job_id)
        if existing and not existing.done():
            return
        _futures[job_id] = _executor.submit(_run_analysis, job_id)


def wait_for_job(job_id: str, timeout: float = 60.0) -> None:
    with _future_lock:
        future = _futures.get(job_id)
    if future:
        future.result(timeout=timeout)
