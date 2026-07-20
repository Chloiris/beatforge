"""Public orchestration API for real CPU audio analysis."""

from __future__ import annotations

import inspect
import math
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from .chart_policy import ScoredCandidate, apply_chart_policy
from .config import AnalysisConfig, AnalysisMode, get_config
from .features import extract_features
from .focus import build_focus_analysis, select_focus_candidates
from .io import audio_from_array, load_audio
from .melody import extract_melody_candidates
from .models import AnalysisResult, AudioData, OnsetCandidate, ProgressCallback
from .onsets import detect_onsets, detector_family_count
from .separation import DemucsSeparator, StemSeparator
from .tempo import estimate_tempo

ANALYSIS_VERSION = "0.5.0"


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _emit_progress(
    callback: ProgressCallback | None,
    stage: str,
    progress: float,
    detail: dict[str, Any] | None = None,
) -> None:
    if callback is None:
        return
    # Support the API's simple (stage, progress) callback and richer workers that
    # also persist stage details, without swallowing errors raised by callbacks.
    try:
        parameters = inspect.signature(callback).parameters
        accepts_detail = any(
            parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
            for parameter in parameters.values()
        ) or len(parameters) >= 3
    except (TypeError, ValueError):
        accepts_detail = False
    if accepts_detail:
        callback(stage, float(progress), detail or {})
    else:
        callback(stage, float(progress))


def _fit_stem(stem: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(stem, dtype=np.float32)
    if values.ndim == 2:
        values = np.mean(values, axis=1, dtype=np.float32)
    if values.size < length:
        values = np.pad(values, (0, length - values.size))
    return np.ascontiguousarray(values[:length], dtype=np.float32)


def _stem_to_original_timeline(stem: np.ndarray, audio: AudioData) -> np.ndarray:
    values = np.asarray(stem, dtype=np.float32)
    if audio.analysis_sample_rate != audio.original_sample_rate:
        divisor = math.gcd(audio.analysis_sample_rate, audio.original_sample_rate)
        values = resample_poly(
            values,
            up=audio.original_sample_rate // divisor,
            down=audio.analysis_sample_rate // divisor,
            window=("kaiser", 8.6),
        ).astype(np.float32, copy=False)
    if values.size < audio.sample_count:
        values = np.pad(values, (0, audio.sample_count - values.size))
    return np.ascontiguousarray(values[: audio.sample_count], dtype=np.float32)


def _analysis_boundary_to_original(sample: int, audio: AudioData) -> int:
    mapped = int(
        round(sample * audio.original_sample_rate / audio.analysis_sample_rate)
    )
    return min(max(mapped, 0), audio.sample_count)


def _snap_sample(
    sample: int,
    sample_rate: int,
    bpm: float,
    offset_sample: int,
    subdivisions_per_beat: int,
) -> int:
    if bpm <= 0:
        return sample
    step = sample_rate * 60.0 / (bpm * subdivisions_per_beat)
    index = round((sample - offset_sample) / step)
    return max(0, int(round(offset_sample + index * step)))


def _serialize_hit_points(
    candidates: list[OnsetCandidate],
    audio: AudioData,
    bpm: float,
    offset_sample: int,
    config: AnalysisConfig,
) -> list[dict[str, Any]]:
    created_at = _utc_now()
    output: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        sample = audio.analysis_to_original_sample(candidate.sample)
        detected_sample = audio.analysis_to_original_sample(candidate.detected_sample)
        refined_sample = audio.analysis_to_original_sample(candidate.refined_sample)
        snapped_sample = _snap_sample(
            sample,
            audio.original_sample_rate,
            bpm,
            offset_sample,
            config.snap_subdivisions_per_beat,
        )
        snapped_sample = min(max(snapped_sample, 0), max(0, audio.sample_count - 1))
        stable_key = (
            f"beatforge:{audio.path or 'memory'}:{audio.original_sample_rate}:"
            f"{sample}:{index}:{ANALYSIS_VERSION}"
        )
        output.append(
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key)),
                "sample": sample,
                "acoustic_sample": refined_sample,
                "chart_sample": snapped_sample,
                "time_sec": sample / float(audio.original_sample_rate),
                "detected_sample": detected_sample,
                "refined_sample": refined_sample,
                "snapped_sample": snapped_sample,
                "snap_error_ms": (sample - snapped_sample)
                * 1000.0
                / audio.original_sample_rate,
                "band": candidate.band,
                "confidence": round(float(candidate.confidence), 6),
                "salience": round(float(candidate.salience), 6),
                "source": candidate.source,
                "detector_votes": candidate.detector_votes,
                "primary_stem": candidate.primary_stem,
                "stem_evidence": {
                    str(name): round(float(value), 6)
                    for name, value in candidate.stem_evidence.items()
                },
                "manually_edited": False,
                "locked": False,
                "created_at": created_at,
                "updated_at": created_at,
                "candidate_event_id": candidate.candidate_id,
            }
        )
    return output


def _serialize_candidate_events(
    candidates: list[ScoredCandidate],
    audio: AudioData,
) -> list[dict[str, Any]]:
    created_at = _utc_now()
    output: list[dict[str, Any]] = []
    for index, item in enumerate(candidates):
        acoustic_sample = audio.analysis_to_original_sample(item.acoustic_sample)
        chart_sample = audio.analysis_to_original_sample(item.chart_sample)
        stable_key = (
            f"beatforge-candidate:{audio.path or 'memory'}:{audio.original_sample_rate}:"
            f"{item.candidate.primary_stem}:{acoustic_sample}:{index}:{ANALYSIS_VERSION}"
        )
        candidate_id = str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))
        item.candidate.candidate_id = candidate_id
        output.append(
            {
                "id": candidate_id,
                "sample": acoustic_sample,
                "time_sec": acoustic_sample / float(audio.original_sample_rate),
                "acoustic_sample": acoustic_sample,
                "chart_sample": chart_sample,
                "snap_error_ms": (acoustic_sample - chart_sample)
                * 1000.0
                / audio.original_sample_rate,
                "lane": {
                    "vocals": "vocals",
                    "other": "melody",
                    "drums": "drums",
                }.get(item.candidate.primary_stem, "mix"),
                "source_evidence": item.source_evidence,
                "semantic_evidence": item.semantic_evidence,
                "confidence": round(item.confidence, 6),
                "status": item.status,
                "grid_type": item.grid_type,
                "grid_confidence": round(item.grid_confidence, 6),
                "hit_point_id": None,
                "created_at": created_at,
                "updated_at": created_at,
            }
        )
    return output


def _apply_beat_salience(
    candidates: list[OnsetCandidate], bpm: float, offset: int, sample_rate: int
) -> None:
    if bpm <= 0:
        return
    period = sample_rate * 60.0 / bpm
    tolerance = sample_rate * 0.035
    for candidate in candidates:
        distance = abs(
            (candidate.sample - offset + period / 2.0) % period - period / 2.0
        )
        beat_relation = float(np.exp(-0.5 * (distance / max(tolerance, 1.0)) ** 2))
        candidate.salience = float(
            np.clip(0.90 * candidate.salience + 0.10 * beat_relation, 0.0, 1.0)
        )


def _nearest_rhythm_grid_error(
    sample: int, bpm: float, offset: int, sample_rate: int
) -> float:
    """Return distance to 1/8, triplet, or 1/16 grid without cumulative drift."""

    if bpm <= 0:
        return float("inf")
    samples_per_beat = sample_rate * 60.0 / bpm
    errors: list[float] = []
    for subdivisions_per_beat in (2, 3, 4):
        step = samples_per_beat / subdivisions_per_beat
        index = round((sample - offset) / step)
        errors.append(abs(sample - (offset + index * step)))
    return min(errors)


def _select_chart_candidates(
    candidates: list[OnsetCandidate],
    bpm: float,
    bpm_confidence: float,
    offset: int,
    sample_rate: int,
    config: AnalysisConfig,
) -> tuple[list[OnsetCandidate], dict[str, int]]:
    """Separate real local attacks from chart-worthy candidates.

    Complex mastered songs contain many real micro-attacks that are poor mapping
    anchors.  Independent evidence families establish acoustic anchors; precise
    rhythmic repetition rescues weaker hats and double-kicks.  There is no global
    hit-count or events-per-second cap, so genuinely dense passages remain dense.
    """

    if not candidates or config.mode == "recall":
        count = len(candidates)
        return candidates, {"detected": count, "selected": count, "rhythmRescued": 0}

    ordered = sorted(candidates, key=lambda candidate: candidate.sample)
    local_window = max(1, int(round(config.local_rescue_window_sec * sample_rate)))
    # A highly coherent tempo may safely rescue slightly late/early weak drums.
    # At uncertain tempos the tolerance stays narrow so the grid cannot justify
    # unrelated micro-attacks in a complex mix.
    confidence_bonus_ms = 4.5 * float(
        np.clip((bpm_confidence - 0.72) / 0.14, 0.0, 1.0)
    )
    rhythm_tolerance = (
        config.rhythmic_rescue_tolerance_ms + confidence_bonus_ms
    ) * sample_rate / 1000.0
    local_scores = np.asarray(
        [candidate.confidence + 0.28 * candidate.salience for candidate in ordered],
        dtype=np.float64,
    )
    selected: list[OnsetCandidate] = []
    rhythm_rescued = 0
    reason_counts = {
        "acousticAnchors": 0,
        "strongAcoustic": 0,
        "loudAcoustic": 0,
        "localStandouts": 0,
        "rhythmRescued": 0,
    }

    left = 0
    right = 0
    for index, candidate in enumerate(ordered):
        while ordered[index].sample - ordered[left].sample > local_window:
            left += 1
        while (
            right + 1 < len(ordered)
            and ordered[right + 1].sample - ordered[index].sample <= local_window
        ):
            right += 1
        family_votes = detector_family_count(candidate.detector_votes)
        acoustic_anchor = (
            family_votes >= config.chart_anchor_family_votes
            and candidate.confidence >= config.local_rescue_confidence
        )
        strong_acoustic = (
            candidate.confidence >= config.chart_anchor_confidence
            and candidate.salience >= config.chart_anchor_salience
        )
        loud_acoustic = (
            family_votes >= max(3, config.chart_anchor_family_votes - 1)
            and candidate.salience >= min(0.95, config.chart_anchor_salience + 0.08)
        )
        local_standout = (
            family_votes >= max(3, config.chart_anchor_family_votes - 1)
            and candidate.confidence >= config.local_rescue_confidence
            and local_scores[index] >= float(np.max(local_scores[left : right + 1])) * 0.96
        )
        rhythm_rescue = (
            family_votes >= config.rhythmic_rescue_family_votes
            and candidate.confidence >= config.rhythmic_rescue_confidence
            and _nearest_rhythm_grid_error(
                candidate.sample, bpm, offset, sample_rate
            )
            <= rhythm_tolerance
        )
        if acoustic_anchor or strong_acoustic or loud_acoustic or local_standout or rhythm_rescue:
            selected.append(candidate)
            if acoustic_anchor:
                reason_counts["acousticAnchors"] += 1
            elif strong_acoustic:
                reason_counts["strongAcoustic"] += 1
            elif loud_acoustic:
                reason_counts["loudAcoustic"] += 1
            elif local_standout:
                reason_counts["localStandouts"] += 1
            else:
                rhythm_rescued += 1
                reason_counts["rhythmRescued"] += 1

    return selected, {
        "detected": len(candidates),
        "selected": len(selected),
        "rhythmRescued": rhythm_rescued,
        **reason_counts,
    }


def _run_analysis(
    audio: AudioData,
    requested_mode: AnalysisMode,
    sensitivity: float,
    callback: ProgressCallback | None,
    separator: StemSeparator | None,
    started: float,
    stage_timings: dict[str, int],
) -> AnalysisResult:
    warnings: list[str] = []
    effective_mode: AnalysisMode = requested_mode
    requested_config = get_config(requested_mode, sensitivity)
    config = requested_config
    extra_stems: dict[str, np.ndarray] | None = None
    separation_metadata: dict[str, Any] | None = None

    _emit_progress(callback, "source_separation", 0.14, {"mode": requested_mode})
    stage_started = time.perf_counter()
    if requested_mode == "accurate":
        active_separator = separator or DemucsSeparator()
        if not active_separator.available:
            effective_mode = "balanced"
            config = get_config("balanced", sensitivity)
            warnings.append("精确模式依赖 Demucs；当前环境不可用，已回退到平衡模式。")
        else:
            try:
                separated = active_separator.separate(
                    audio.analysis_channels, audio.analysis_sample_rate
                )
                if separated.warning:
                    warnings.append(separated.warning)
                extra_stems = {
                    name: _fit_stem(stem, audio.analysis_mono.size)
                    for name, stem in separated.stems.items()
                    if name in {"vocals", "drums", "bass", "other"}
                }
                separation_metadata = {
                    "model": separated.model_name or "unknown",
                    "device": separated.device or "unknown",
                    "sources": sorted(extra_stems),
                }
                if not extra_stems:
                    effective_mode = "balanced"
                    config = get_config("balanced", sensitivity)
                    warnings.append("音源分离未返回可用分轨，已回退到平衡模式。")
            except Exception as exc:  # optional plugin failures must not block base mode
                effective_mode = "balanced"
                config = get_config("balanced", sensitivity)
                warnings.append(f"音源分离失败，已回退到平衡模式：{exc}")
    stage_timings["source_separation"] = int(
        round((time.perf_counter() - stage_started) * 1000.0)
    )

    _emit_progress(callback, "computing_multiband_features", 0.22)
    stage_started = time.perf_counter()
    features = extract_features(audio.analysis_mono, audio.analysis_sample_rate, config)
    stage_timings["feature_extraction"] = int(
        round((time.perf_counter() - stage_started) * 1000.0)
    )

    _emit_progress(callback, "detecting_and_refining_candidates", 0.62)
    stage_started = time.perf_counter()
    mix_candidates = detect_onsets(audio.analysis_mono, features, config)
    stage_timings["onset_detection_refinement"] = int(
        round((time.perf_counter() - stage_started) * 1000.0)
    )

    _emit_progress(callback, "estimating_bpm", 0.82)
    stage_started = time.perf_counter()
    leading_analysis = audio.original_to_analysis_sample(audio.leading_silence_samples)
    tempo = estimate_tempo(
        features,
        mix_candidates,
        config,
        default_offset_sample=leading_analysis,
    )
    stage_timings["tempo_estimation"] = int(
        round((time.perf_counter() - stage_started) * 1000.0)
    )
    focus_map: list[dict[str, Any]] = []
    melody_extraction: dict[str, Any] | None = None
    if extra_stems:
        candidates_by_stem: dict[str, list[OnsetCandidate]] = {"mix": mix_candidates}
        stem_feature_started = time.perf_counter()
        for index, stem_name in enumerate(("vocals", "drums", "bass", "other")):
            stem = extra_stems.get(stem_name)
            if stem is None:
                continue
            _emit_progress(
                callback,
                f"analyzing_stem_{stem_name}",
                0.64 + index * 0.04,
                {"source": stem_name},
            )
            if stem_name == "other":
                try:
                    melody = extract_melody_candidates(
                        stem,
                        audio.analysis_sample_rate,
                    )
                    candidates_by_stem[stem_name] = melody.candidates
                    melody_extraction = {
                        "method": melody.method,
                        "voicedFrameCount": melody.voiced_frame_count,
                        "pitchOnsetCount": melody.pitch_onset_count,
                        "energyReattackCount": melody.energy_reattack_count,
                        "local": True,
                        "cloudApi": False,
                    }
                except Exception as exc:
                    candidates_by_stem[stem_name] = []
                    melody_extraction = {
                        "method": "librosa_pyin_local",
                        "error": str(exc),
                        "local": True,
                        "cloudApi": False,
                    }
                    warnings.append(f"本地主旋律音高提取失败，melody lane 保持为空：{exc}")
            else:
                stem_features = extract_features(
                    stem,
                    audio.analysis_sample_rate,
                    config,
                )
                candidates_by_stem[stem_name] = detect_onsets(
                    stem,
                    stem_features,
                    config,
                )
        stage_timings["stem_feature_detection"] = int(
            round((time.perf_counter() - stem_feature_started) * 1000.0)
        )
        focus = build_focus_analysis(
            extra_stems,
            audio.analysis_sample_rate,
            duration_samples=audio.analysis_mono.size,
        )
        lane_candidates, candidate_selection = select_focus_candidates(
            candidates_by_stem,
            focus,
            audio.analysis_sample_rate,
        )
        focus_map = []
        for segment in focus.segments:
            start_sample = _analysis_boundary_to_original(
                int(segment["start_sample"]), audio
            )
            end_sample = _analysis_boundary_to_original(
                int(segment["end_sample"]), audio
            )
            focus_key = (
                f"beatforge-focus:{audio.path or 'memory'}:{start_sample}:"
                f"{end_sample}:{segment['focus_source']}:{ANALYSIS_VERSION}"
            )
            focus_map.append(
                {
                    **segment,
                    "id": str(uuid.uuid5(uuid.NAMESPACE_URL, focus_key)),
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "manually_edited": False,
                }
            )
    else:
        lane_candidates, candidate_selection = _select_chart_candidates(
            mix_candidates,
            tempo.bpm,
            tempo.confidence,
            tempo.beat_offset_sample,
            audio.analysis_sample_rate,
            config,
        )
        for candidate in lane_candidates:
            candidate.primary_stem = "mix"
            candidate.stem_evidence = {"mix": 1.0}
    _apply_beat_salience(
        lane_candidates,
        tempo.bpm,
        tempo.beat_offset_sample,
        audio.analysis_sample_rate,
    )
    policy = apply_chart_policy(
        lane_candidates,
        sample_rate=audio.analysis_sample_rate,
        bpm=tempo.bpm,
        beat_offset_sample=tempo.beat_offset_sample,
        acceptance_threshold=0.0 if not extra_stems else 0.42,
        uncertainty_threshold=0.0 if not extra_stems else 0.28,
        enforce_density=bool(extra_stems),
        difficulty_level=0.5,
    )
    candidates = policy.accepted
    original_offset = audio.analysis_to_original_sample(tempo.beat_offset_sample)

    _emit_progress(callback, "merging_classifying_and_serializing", 0.94)
    stage_started = time.perf_counter()
    candidate_events = _serialize_candidate_events(policy.candidates, audio)
    hit_points = _serialize_hit_points(
        candidates,
        audio,
        tempo.bpm,
        original_offset,
        config,
    )
    hit_id_by_candidate = {
        str(hit["candidate_event_id"]): str(hit["id"])
        for hit in hit_points
        if hit.get("candidate_event_id")
    }
    for candidate_event in candidate_events:
        hit_point_id = hit_id_by_candidate.get(str(candidate_event["id"]))
        candidate_event["hit_point_id"] = hit_point_id
        if candidate_event["status"] == "accepted" and hit_point_id is None:
            candidate_event["status"] = "uncertain"
    stage_timings["result_serialization"] = int(
        round((time.perf_counter() - stage_started) * 1000.0)
    )
    elapsed_ms = int(round((time.perf_counter() - started) * 1000.0))
    created_at = _utc_now()
    metadata = {
        "version": ANALYSIS_VERSION,
        "mode": requested_mode,
        "effective_mode": effective_mode,
        "parameters": asdict(config),
        "elapsed_ms": elapsed_ms,
        "bpm_confidence": tempo.confidence,
        "warnings": list(warnings),
        "tempo_candidates": tempo.candidates,
        "candidate_selection": candidate_selection,
        "chart_policy": {
            "input_count": len(policy.candidates),
            "accepted_count": len(policy.accepted),
            "uncertain_count": sum(item.status == "uncertain" for item in policy.candidates),
            "rejected_count": sum(item.status == "rejected" for item in policy.candidates),
            "score_weights": {
                "sourceEvidence": 0.35,
                "acousticConfidence": 0.25,
                "rhythmAlignment": 0.20,
                "semanticEvidence": 0.20,
            },
            "difficultyLevel": 0.5,
        },
        "melody_extraction": melody_extraction,
        "separator": separation_metadata,
        "focus_map": focus_map,
        "stems": (
            [
                {
                    "source": name,
                    "available": True,
                    "waveform_url": f"?source={name}",
                    "audio_url": f"/stems/{name}/audio",
                }
                for name in ("vocals", "drums", "bass", "other")
                if extra_stems and name in extra_stems
            ]
            if extra_stems
            else []
        ),
        "created_at": created_at,
    }
    result = AnalysisResult(
        original_sample_rate=audio.original_sample_rate,
        sample_count=audio.sample_count,
        channels=audio.channels,
        duration_sec=audio.duration_sec,
        leading_silence_samples=audio.leading_silence_samples,
        bpm=tempo.bpm,
        bpm_confidence=tempo.confidence,
        beat_offset_sample=original_offset,
        hit_points=hit_points,
        metadata=metadata,
        warnings=warnings,
        stage_timings_ms=stage_timings,
        candidate_events=candidate_events,
        stem_audio=(
            {
                name: _stem_to_original_timeline(stem, audio)
                for name, stem in extra_stems.items()
            }
            if extra_stems
            else {}
        ),
    )
    _emit_progress(
        callback,
        "analysis_complete",
        1.0,
        {"elapsed_ms": elapsed_ms, "hit_point_count": len(hit_points)},
    )
    return result


def analyze_audio(
    path: str | Path,
    mode: AnalysisMode = "balanced",
    sensitivity: float = 0.5,
    progress_callback: ProgressCallback | None = None,
    *,
    separator: StemSeparator | None = None,
) -> AnalysisResult:
    """Decode and analyze an audio path with no ground-truth or cloud dependency."""

    started = time.perf_counter()
    stage_timings: dict[str, int] = {}
    _emit_progress(progress_callback, "decoding_audio", 0.03)
    stage_started = time.perf_counter()
    config = get_config(mode, sensitivity)
    audio = load_audio(path, config)
    stage_timings["audio_decoding_preprocessing"] = int(
        round((time.perf_counter() - stage_started) * 1000.0)
    )
    return _run_analysis(
        audio,
        mode,
        sensitivity,
        progress_callback,
        separator,
        started,
        stage_timings,
    )


def analyze_samples(
    samples: np.ndarray,
    sample_rate: int,
    mode: AnalysisMode = "balanced",
    sensitivity: float = 0.5,
    progress_callback: ProgressCallback | None = None,
    *,
    separator: StemSeparator | None = None,
) -> AnalysisResult:
    """Analyze in-memory audio through the identical production feature pipeline."""

    started = time.perf_counter()
    stage_timings: dict[str, int] = {}
    _emit_progress(progress_callback, "decoding_audio", 0.03)
    stage_started = time.perf_counter()
    config = get_config(mode, sensitivity)
    audio = audio_from_array(samples, sample_rate, config)
    stage_timings["audio_decoding_preprocessing"] = int(
        round((time.perf_counter() - stage_started) * 1000.0)
    )
    return _run_analysis(
        audio,
        mode,
        sensitivity,
        progress_callback,
        separator,
        started,
        stage_timings,
    )
