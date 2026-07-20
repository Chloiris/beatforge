from __future__ import annotations

import math
from datetime import UTC, datetime
from difflib import SequenceMatcher
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from .base import AlignmentContext, lyric_units
from .schema import AlignmentReport, AlignmentResult, AlignmentToken

COVERAGE_CONFIDENCE_THRESHOLD = 0.20


def _clamp01(value: float) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def _coverage(
    lyrics: str,
    tokens: list[AlignmentToken],
    metadata: dict[str, Any],
) -> tuple[float, int, dict[str, Any]]:
    expected = lyric_units(lyrics)
    aligned_text = metadata.get("alignedText")
    if isinstance(aligned_text, str):
        observed = lyric_units(aligned_text)
        source = "adapter_aligned_text"
    else:
        # Phoneme adapters can repeat the owning surface token for every phone.
        # Compact only adjacent repeats; no timing is inferred or redistributed.
        compacted: list[str] = []
        for token in tokens:
            if not compacted or compacted[-1] != token.text:
                compacted.append(token.text)
        observed = lyric_units("".join(compacted))
        source = "token_text_sequence"
    if not expected:
        return 0.0, 0, {"matchedLyricUnits": 0, "coverageSource": source}
    matcher = SequenceMatcher(a=expected, b=observed, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return (
        _clamp01(matched / len(expected)),
        len(expected),
        {
            "matchedLyricUnits": matched,
            "observedLyricUnits": len(observed),
            "coverageSource": source,
        },
    )


def _feature_strength(values: np.ndarray) -> tuple[np.ndarray, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float64), 0.0
    reference = max(float(np.quantile(finite, 0.90)), 1e-12)
    return np.clip(np.nan_to_num(values / reference), 0.0, 1.0), reference


def _acoustic(
    context: AlignmentContext,
    tokens: list[AlignmentToken],
) -> tuple[float, dict[str, Any]]:
    audio, stem_rate = sf.read(context.vocals_path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    if audio.size < 32 or not tokens:
        return 0.0, {"evaluatedTokens": 0, "reason": "empty_audio_or_tokens"}

    hop = max(64, round(stem_rate * 0.010))
    frame = max(256, 2 ** math.ceil(math.log2(max(256, round(stem_rate * 0.046)))))
    rms = librosa.feature.rms(y=audio, frame_length=frame, hop_length=hop, center=True)[0]
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=stem_rate,
        n_mfcc=13,
        n_fft=frame,
        hop_length=hop,
    )
    mfcc_change = np.linalg.norm(np.diff(mfcc, axis=1, prepend=mfcc[:, :1]), axis=0)
    try:
        f0 = librosa.yin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=min(librosa.note_to_hz("C7"), stem_rate / 2.2),
            sr=stem_rate,
            frame_length=max(frame, 1024),
            hop_length=hop,
        )
        log_f0 = np.log2(np.maximum(f0, 1e-6))
        pitch_change = np.abs(np.diff(log_f0, prepend=log_f0[:1]))
        if pitch_change.size != rms.size:
            pitch_change = np.interp(
                np.linspace(0, 1, rms.size),
                np.linspace(0, 1, pitch_change.size),
                pitch_change,
            )
    except (ValueError, FloatingPointError):
        pitch_change = np.zeros_like(rms)

    rms_strength, rms_reference = _feature_strength(rms)
    mfcc_strength, mfcc_reference = _feature_strength(mfcc_change)
    pitch_strength, pitch_reference = _feature_strength(pitch_change)
    ratio = stem_rate / context.sample_rate
    token_scores: list[float] = []
    rms_scores: list[float] = []
    mfcc_scores: list[float] = []
    pitch_scores: list[float] = []
    for token in tokens:
        frame_index = int(round(token.start_sample * ratio / hop))
        left = max(0, frame_index - 2)
        right = min(rms.size, frame_index + 3)
        if right <= left:
            continue
        local_rms = float(np.max(rms_strength[left:right]))
        local_mfcc = float(np.max(mfcc_strength[left:right]))
        local_pitch = float(np.max(pitch_strength[left:right]))
        rms_scores.append(local_rms)
        mfcc_scores.append(local_mfcc)
        pitch_scores.append(local_pitch)
        token_scores.append(0.50 * local_rms + 0.30 * local_mfcc + 0.20 * local_pitch)

    return (
        _clamp01(float(np.mean(token_scores)) if token_scores else 0.0),
        {
            "evaluatedTokens": len(token_scores),
            "vocalRms": round(float(np.mean(rms_scores)) if rms_scores else 0.0, 6),
            "mfccChange": round(float(np.mean(mfcc_scores)) if mfcc_scores else 0.0, 6),
            "pitchChange": round(float(np.mean(pitch_scores)) if pitch_scores else 0.0, 6),
            "featureReferences": {
                "rms": rms_reference,
                "mfccChange": mfcc_reference,
                "pitchChangeOctaves": pitch_reference,
            },
            "stemSampleRate": int(stem_rate),
        },
    )


def _rhythm(
    context: AlignmentContext,
    tokens: list[AlignmentToken],
) -> tuple[float, dict[str, Any]]:
    if not context.tempo_map or not tokens:
        return 0.0, {"evaluatedTokens": 0, "grid": "1/16", "available": False}
    segments = sorted(context.tempo_map, key=lambda item: item.start_sample)
    scores: list[float] = []
    errors_ms: list[float] = []
    for token in tokens:
        segment = max(
            (item for item in segments if item.start_sample <= token.start_sample),
            key=lambda item: item.start_sample,
            default=segments[0],
        )
        step = context.sample_rate * 60.0 / segment.bpm / 4.0
        if step <= 0:
            continue
        grid_index = round((token.start_sample - segment.beat_offset_sample) / step)
        nearest = segment.beat_offset_sample + grid_index * step
        distance = abs(token.start_sample - nearest)
        # This is evaluation only. It never changes or snaps a timestamp.
        scores.append(math.exp(-((distance / max(step * 0.5, 1.0)) ** 2)))
        errors_ms.append(distance * 1000.0 / context.sample_rate)
    return (
        _clamp01(float(np.mean(scores)) if scores else 0.0),
        {
            "evaluatedTokens": len(scores),
            "grid": "1/16",
            "available": bool(scores),
            "meanDistanceMs": round(float(np.mean(errors_ms)) if errors_ms else 0.0, 6),
            "forcedSnapApplied": False,
        },
    )


def _stability(
    context: AlignmentContext,
    tokens: list[AlignmentToken],
) -> tuple[float, dict[str, Any]]:
    if not tokens:
        return 0.0, {
            "reverseOrderCount": 0,
            "duplicateTimestampCount": 0,
            "denseTimestampCount": 0,
            "invalidSpanCount": 0,
        }
    starts = [token.start_sample for token in tokens]
    reverse_count = sum(
        current < previous for previous, current in zip(starts, starts[1:], strict=False)
    )
    duplicate_count = len(tokens) - len(
        {(token.start_sample, token.end_sample) for token in tokens}
    )
    dense_threshold = max(1, round(context.sample_rate * 0.005))
    ordered = sorted(starts)
    dense_count = sum(
        0 < current - previous < dense_threshold
        for previous, current in zip(ordered, ordered[1:], strict=False)
    )
    invalid_count = sum(
        token.end_sample < token.start_sample
        or token.start_sample < 0
        or token.end_sample > context.sample_count
        for token in tokens
    )
    denominator = max(1, len(tokens))
    penalty = (
        0.35 * reverse_count / denominator
        + 0.30 * duplicate_count / denominator
        + 0.15 * dense_count / denominator
        + 0.20 * invalid_count / denominator
    )
    return (
        _clamp01(1.0 - penalty),
        {
            "reverseOrderCount": reverse_count,
            "duplicateTimestampCount": duplicate_count,
            "denseTimestampCount": dense_count,
            "denseThresholdMs": dense_threshold * 1000.0 / context.sample_rate,
            "invalidSpanCount": invalid_count,
        },
    )


class AlignmentEvaluator:
    """Proxy evaluator used when no hand-labelled ground truth is available."""

    def evaluate(self, context: AlignmentContext, result: AlignmentResult) -> AlignmentReport:
        if result.hierarchy is not None and result.hierarchy.characters:
            characters = result.hierarchy.characters
            phones = result.hierarchy.phonemes
            forced_aligned_count = 0
            aligned_count = 0
            for unit in characters:
                mapped_phones = [
                    phones[index]
                    for index in unit.phoneme_indices
                    if index < len(phones)
                ]
                observed_phones = [
                    phone
                    for phone in mapped_phones
                    if phone.refined_end_sample > phone.refined_start_sample
                    and phone.observed_token_index is not None
                    and str(phone.match_operation or "").casefold()
                    not in {"delete", "deletion", "unmatched", "missing"}
                ]
                fully_observed = bool(observed_phones) and len(observed_phones) == len(
                    mapped_phones
                )
                if fully_observed:
                    forced_aligned_count += 1
                    mean_confidence = sum(
                        phone.confidence for phone in observed_phones
                    ) / len(observed_phones)
                    if mean_confidence >= COVERAGE_CONFIDENCE_THRESHOLD:
                        aligned_count += 1
            lyric_count = len(characters)
            coverage = _clamp01(aligned_count / lyric_count)
            coverage_details = {
                "matchedLyricUnits": aligned_count,
                "observedLyricUnits": aligned_count,
                "forcedAlignedLyricUnits": forced_aligned_count,
                "forcedTargetCoverage": _clamp01(
                    forced_aligned_count / lyric_count
                ),
                "coverageSource": "confidence_qualified_typed_character_hierarchy",
                "confidenceThreshold": COVERAGE_CONFIDENCE_THRESHOLD,
                "confidenceAggregation": "mean_mapped_observed_phone_confidence",
                "timestampSource": "mapped_observed_ctc_phones",
            }
            report_aligned_count = aligned_count
        else:
            coverage, lyric_count, coverage_details = _coverage(
                context.lyrics,
                result.tokens,
                result.metadata,
            )
            report_aligned_count = len(result.tokens)
        acoustic, acoustic_details = _acoustic(context, result.tokens)
        rhythm, rhythm_details = _rhythm(context, result.tokens)
        stability, stability_details = _stability(context, result.tokens)
        score = _clamp01(
            0.35 * coverage
            + 0.30 * acoustic
            + 0.20 * rhythm
            + 0.15 * stability
        )
        return AlignmentReport(
            run_id=result.run_id,
            track_id=context.track_id,
            method=result.method,
            score=round(score, 6),
            coverage=round(coverage, 6),
            acoustic=round(acoustic, 6),
            rhythm=round(rhythm, 6),
            stability=round(stability, 6),
            lyric_token_count=lyric_count,
            aligned_token_count=report_aligned_count,
            details={
                "weights": {
                    "coverage": 0.35,
                    "acoustic": 0.30,
                    "rhythm": 0.20,
                    "stability": 0.15,
                },
                "coverage": coverage_details,
                "acoustic": acoustic_details,
                "rhythm": rhythm_details,
                "stability": stability_details,
                "groundTruth": "proxy_only",
            },
            created_at=datetime.now(UTC),
        )
