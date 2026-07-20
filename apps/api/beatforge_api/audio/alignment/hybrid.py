from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Any

import librosa
import numpy as np
import soundfile as sf

from .base import (
    AdapterDiagnostics,
    AdapterOutput,
    AlignmentAdapter,
    AlignmentAdapterError,
    AlignmentContext,
    alignment_token_id,
    lyric_units,
)
from .schema import AlignmentResult, AlignmentToken


@dataclass(frozen=True, slots=True)
class _Candidate:
    method: str
    text: str
    phoneme: str | None
    start_sample: int
    end_sample: int
    confidence: float
    lyric_start: int
    lyric_end: int


def _find_units(haystack: list[str], needle: list[str], start: int) -> int | None:
    if not needle:
        return None
    limit = len(haystack) - len(needle)
    for index in range(max(0, start), limit + 1):
        if haystack[index : index + len(needle)] == needle:
            return index
    return None


def _surface_candidates(result: AlignmentResult, expected: list[str]) -> list[_Candidate]:
    grouped: list[list[AlignmentToken]] = []
    for token in result.tokens:
        if grouped and grouped[-1][-1].text == token.text:
            grouped[-1].append(token)
        else:
            grouped.append([token])
    cursor = 0
    candidates: list[_Candidate] = []
    for group in grouped:
        text = group[0].text
        units = lyric_units(text)
        lyric_start = _find_units(expected, units, cursor)
        if lyric_start is None:
            continue
        lyric_end = lyric_start + len(units)
        cursor = lyric_end
        start = min(token.start_sample for token in group)
        end = max(token.end_sample for token in group)
        if end <= start:
            continue
        phones = [token.phoneme for token in group if token.phoneme]
        candidates.append(
            _Candidate(
                method=result.method,
                text=text,
                phoneme=" ".join(phones) if phones else None,
                start_sample=start,
                end_sample=end,
                confidence=float(np.mean([token.confidence for token in group])),
                lyric_start=lyric_start,
                lyric_end=lyric_end,
            )
        )
    return candidates


def _acoustic_profiles(context: AlignmentContext) -> tuple[np.ndarray, np.ndarray, int, int]:
    audio, sample_rate = sf.read(context.vocals_path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    audio = np.ascontiguousarray(audio, dtype=np.float32)
    hop = max(64, round(sample_rate * 0.010))
    frame = max(512, 2 ** math.ceil(math.log2(max(256, round(sample_rate * 0.046)))))
    rms = librosa.feature.rms(y=audio, frame_length=frame, hop_length=hop)[0]
    try:
        f0 = librosa.yin(
            audio,
            fmin=librosa.note_to_hz("C2"),
            fmax=min(librosa.note_to_hz("C7"), sample_rate / 2.2),
            sr=sample_rate,
            frame_length=max(1024, frame),
            hop_length=hop,
        )
        pitch_change = np.abs(np.diff(np.log2(np.maximum(f0, 1e-6)), prepend=0.0))
        if pitch_change.size != rms.size:
            pitch_change = np.interp(
                np.linspace(0, 1, rms.size),
                np.linspace(0, 1, pitch_change.size),
                pitch_change,
            )
    except (ValueError, FloatingPointError):
        pitch_change = np.zeros_like(rms)
    rms_reference = max(float(np.quantile(rms, 0.90)), 1e-12)
    pitch_reference = max(float(np.quantile(pitch_change, 0.90)), 1e-12)
    return (
        np.clip(rms / rms_reference, 0.0, 1.0),
        np.clip(pitch_change / pitch_reference, 0.0, 1.0),
        hop,
        int(sample_rate),
    )


def _acoustic_score(
    context: AlignmentContext,
    sample: int,
    rms: np.ndarray,
    pitch: np.ndarray,
    hop: int,
    audio_rate: int,
) -> float:
    audio_sample = round(sample * audio_rate / context.sample_rate)
    index = round(audio_sample / hop)
    left = max(0, index - 2)
    right = min(rms.size, index + 3)
    if right <= left:
        return 0.0
    return float(0.65 * np.max(rms[left:right]) + 0.35 * np.max(pitch[left:right]))


def _beat_score(context: AlignmentContext, sample: int) -> float:
    if not context.tempo_map:
        return 0.0
    segment = max(
        (item for item in context.tempo_map if item.start_sample <= sample),
        key=lambda item: item.start_sample,
        default=context.tempo_map[0],
    )
    step = context.sample_rate * 60.0 / segment.bpm / 4.0
    nearest_index = round((sample - segment.beat_offset_sample) / step)
    nearest = segment.beat_offset_sample + nearest_index * step
    distance = abs(sample - nearest)
    return float(math.exp(-((distance / max(step * 0.5, 1.0)) ** 2)))


class HybridAlignmentAdapter(AlignmentAdapter):
    method = "hybrid"
    name = "Hybrid Fusion"

    def diagnostics(self, context: AlignmentContext | None = None) -> AdapterDiagnostics:
        return AdapterDiagnostics(
            available=True,
            reason=None,
            model="observed-span consensus",
            automatic_downloads_enabled=False,
            details={"minimumSuccessfulMethods": 2, "timestampAveraging": False},
        )

    def run(self, context: AlignmentContext) -> AdapterOutput:
        raise AlignmentAdapterError(
            "HYBRID_COMPONENTS_REQUIRED",
            "Hybrid must be run through AlignmentRunner with component results.",
        )

    def fuse(
        self,
        context: AlignmentContext,
        results: list[AlignmentResult],
        failures: dict[str, dict[str, Any]],
    ) -> AdapterOutput:
        successful = [
            result
            for result in results
            if result.status == "completed" and result.tokens and result.method != "hybrid"
        ]
        if len(successful) < 2:
            raise AlignmentAdapterError(
                "HYBRID_INSUFFICIENT_METHODS",
                "Hybrid fusion requires at least two successful real alignment methods.",
                status="unavailable",
                details={
                    "successfulMethods": [result.method for result in successful],
                    "componentFailures": failures,
                },
            )
        expected = lyric_units(context.lyrics)
        by_lyric_start: dict[int, list[_Candidate]] = {}
        for result in successful:
            for candidate in _surface_candidates(result, expected):
                by_lyric_start.setdefault(candidate.lyric_start, []).append(candidate)
        if not by_lyric_start:
            raise AlignmentAdapterError(
                "HYBRID_NO_COMMON_SEQUENCE",
                "Successful methods did not expose a common monotonic lyric sequence.",
                details={"successfulMethods": [result.method for result in successful]},
            )

        rms, pitch, hop, audio_rate = _acoustic_profiles(context)
        output: list[AlignmentToken] = []
        support_counts: dict[str, int] = {}
        previous_start = -1
        for lyric_start in sorted(by_lyric_start):
            candidates = by_lyric_start[lyric_start]
            unique_methods = {candidate.method for candidate in candidates}
            start_median = median(candidate.start_sample for candidate in candidates)
            end_median = median(candidate.end_sample for candidate in candidates)
            consensus_scale = max(1.0, context.sample_rate * 0.150)
            scored: list[tuple[float, _Candidate, dict[str, float]]] = []
            for candidate in candidates:
                temporal_distance = (
                    abs(candidate.start_sample - start_median)
                    + abs(candidate.end_sample - end_median)
                ) / 2.0
                consensus = math.exp(-temporal_distance / consensus_scale)
                acoustic = _acoustic_score(
                    context,
                    candidate.start_sample,
                    rms,
                    pitch,
                    hop,
                    audio_rate,
                )
                beat = _beat_score(context, candidate.start_sample)
                score = (
                    0.35 * candidate.confidence
                    + 0.30 * consensus
                    + 0.20 * acoustic
                    + 0.15 * beat
                )
                scored.append(
                    (
                        score,
                        candidate,
                        {"consensus": consensus, "acoustic": acoustic, "beat": beat},
                    )
                )
            # Enforce sequence by rejecting observed spans that would move backwards.
            # No timestamp is shifted, averaged, snapped, or synthesized.
            ordered = [
                item
                for item in sorted(scored, key=lambda item: item[0], reverse=True)
                if item[1].start_sample >= previous_start
            ]
            if not ordered:
                continue
            score, selected, evidence = ordered[0]
            support = len(unique_methods)
            support_counts[str(support)] = support_counts.get(str(support), 0) + 1
            agreement = support / len(successful)
            confidence = min(1.0, max(0.0, score * (0.65 + 0.35 * agreement)))
            index = len(output)
            output.append(
                AlignmentToken(
                    id=alignment_token_id(
                        context.track_id,
                        self.method,
                        index,
                        selected.start_sample,
                        selected.end_sample,
                    ),
                    text=selected.text,
                    phoneme=selected.phoneme,
                    start_sample=selected.start_sample,
                    end_sample=selected.end_sample,
                    confidence=confidence,
                    method=self.method,
                )
            )
            previous_start = selected.start_sample
            # Keep aggregate evidence auditable without changing the unified token shape.
            evidence["support"] = float(support)
        if not output:
            raise AlignmentAdapterError(
                "HYBRID_EMPTY",
                "Hybrid had successful components but could not retain an ordered real span.",
            )
        return AdapterOutput(
            tokens=tuple(output),
            warnings=tuple(
                f"{method}: {details.get('message', 'component unavailable')}"
                for method, details in failures.items()
            ),
            metadata={
                "componentMethods": [result.method for result in successful],
                "componentFailures": failures,
                "supportCounts": support_counts,
                "timestampProvenance": (
                    "Each hybrid span is an unchanged span observed by a successful model; "
                    "energy, pitch, beat distance, confidence and consensus only select it."
                ),
                "timestampAveraging": False,
                "beatSnapping": False,
                "orderConstraint": "reject_only",
                "alignedText": "".join(token.text for token in output),
            },
        )
