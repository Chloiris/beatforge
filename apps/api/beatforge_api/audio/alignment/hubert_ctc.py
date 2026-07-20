"""BeatForge v0.6.1 Japanese HuBERT CTC mora alignment engine.

The adapter keeps the existing offline HuBERT/Viterbi process as the only
source of lexical time, then joins its observed spans to a Japanese
phoneme→mora→character plan and refines boundaries with measured vocal
features.  Lyrics and the tempo grid never create a timestamp in this module.
"""

from __future__ import annotations

import bisect
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .base import (
    AdapterDiagnostics,
    AdapterOutput,
    AlignmentAdapterError,
    AlignmentContext,
    alignment_token_id,
)
from .ctc_adapter import CTCAlignmentAdapter
from .lyric_processor import (
    LyricProcessingError,
    ProcessedLyrics,
    map_phoneme_sequences_dp,
)
from .mora_decoder import decode_moras
from .schema import (
    AlignmentAcousticEvidence,
    AlignmentHierarchy,
    AlignmentHierarchyUnit,
    AlignmentToken,
)

ENGINE_VERSION = "0.6.1"
_FEATURE_HOP_SECONDS = 0.010
_FEATURE_FRAME_SECONDS = 0.046
_BOUNDARY_WINDOW_SECONDS = 0.050
_NORMAL_BOUNDARY_THRESHOLD = 0.32
_RAP_BOUNDARY_THRESHOLD = 0.20
_RAP_PHONE_DENSITY_PER_SECOND = 7.0


@dataclass(frozen=True, slots=True)
class _AcousticFeatures:
    samples: np.ndarray
    energy: np.ndarray
    spectral_change: np.ndarray
    pitch_change: np.ndarray
    hop_samples: int
    frame_samples: int
    source_sample_rate: int


def _robust_unit_interval(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(array)
    if not np.any(finite):
        return np.zeros_like(array, dtype=np.float32)
    safe = np.where(finite, array, 0.0)
    observed = safe[finite]
    low = float(np.percentile(observed, 10.0))
    high = float(np.percentile(observed, 95.0))
    if high <= low + 1e-12:
        return np.zeros_like(array, dtype=np.float32)
    return np.asarray(np.clip((safe - low) / (high - low), 0.0, 1.0), dtype=np.float32)


def _extract_acoustic_features(
    context: AlignmentContext,
) -> _AcousticFeatures:
    try:
        import librosa
    except ImportError as error:  # pragma: no cover - dependency is pinned by the app
        raise AlignmentAdapterError(
            "HUBERT_ACOUSTIC_DEPENDENCY_MISSING",
            "librosa is required for HuBERT boundary refinement.",
        ) from error
    try:
        audio, source_rate = sf.read(
            context.vocals_path,
            dtype="float32",
            always_2d=True,
        )
    except (OSError, RuntimeError) as error:
        raise AlignmentAdapterError(
            "HUBERT_ACOUSTIC_AUDIO_FAILED",
            "The vocals stem could not be read for acoustic refinement.",
        ) from error
    if audio.size == 0 or source_rate <= 0 or not np.all(np.isfinite(audio)):
        raise AlignmentAdapterError(
            "HUBERT_ACOUSTIC_AUDIO_INVALID",
            "The vocals stem is empty or contains invalid samples.",
        )
    mono = np.asarray(np.mean(audio, axis=1), dtype=np.float32)
    if float(np.max(np.abs(mono))) <= 1e-8:
        raise AlignmentAdapterError(
            "HUBERT_ACOUSTIC_AUDIO_SILENT",
            "The vocals stem is silent and cannot refine HuBERT boundaries.",
        )
    hop_length = max(1, round(source_rate * _FEATURE_HOP_SECONDS))
    frame_length = max(256, round(source_rate * _FEATURE_FRAME_SECONDS))
    frame_length = 1 << int(math.ceil(math.log2(frame_length)))
    try:
        rms = librosa.feature.rms(
            y=mono,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )[0]
        magnitude = np.abs(
            librosa.stft(
                mono,
                n_fft=frame_length,
                hop_length=hop_length,
                win_length=frame_length,
                center=True,
            )
        )
        positive_change = np.maximum(np.diff(magnitude, axis=1, prepend=magnitude[:, :1]), 0.0)
        spectral_flux = np.sqrt(np.mean(np.square(positive_change), axis=0))
        spectral_flux /= np.mean(magnitude, axis=0) + 1e-8
        del magnitude, positive_change
        f0 = librosa.yin(
            mono,
            fmin=65.0,
            fmax=min(1_200.0, source_rate / 2.0 - 1.0),
            sr=source_rate,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        )
    except Exception as error:
        raise AlignmentAdapterError(
            "HUBERT_ACOUSTIC_FEATURE_FAILED",
            "RMS, spectral, or pitch features could not be measured from the vocals stem.",
            details={"exceptionType": type(error).__name__, "message": str(error)[:1_000]},
        ) from error

    frame_count = min(len(rms), len(spectral_flux), len(f0))
    if frame_count < 2:
        raise AlignmentAdapterError(
            "HUBERT_ACOUSTIC_FEATURE_EMPTY",
            "The vocals stem produced too few acoustic feature frames.",
        )
    rms = np.asarray(rms[:frame_count], dtype=np.float64)
    rms_delta = np.abs(np.diff(rms, prepend=rms[:1]))
    energy = np.maximum(_robust_unit_interval(rms), _robust_unit_interval(rms_delta))
    spectral = _robust_unit_interval(np.asarray(spectral_flux[:frame_count]))
    f0 = np.asarray(f0[:frame_count], dtype=np.float64)
    voiced = rms > max(float(np.percentile(rms, 20.0)), 1e-8)
    safe_f0 = np.where(voiced & np.isfinite(f0) & (f0 > 0.0), f0, np.nan)
    log_pitch = np.log2(safe_f0)
    pitch_delta = np.abs(np.diff(log_pitch, prepend=log_pitch[:1]))
    pitch_delta[~np.isfinite(pitch_delta)] = 0.0
    pitch = _robust_unit_interval(pitch_delta)

    source_samples = np.arange(frame_count, dtype=np.float64) * hop_length
    original_samples = np.rint(source_samples * context.sample_rate / source_rate).astype(
        np.int64
    )
    original_samples = np.clip(original_samples, 0, context.sample_count - 1)
    return _AcousticFeatures(
        samples=original_samples,
        energy=energy[:frame_count],
        spectral_change=spectral[:frame_count],
        pitch_change=pitch[:frame_count],
        hop_samples=max(1, round(hop_length * context.sample_rate / source_rate)),
        frame_samples=max(1, round(frame_length * context.sample_rate / source_rate)),
        source_sample_rate=int(source_rate),
    )


def _feature_evidence(
    features: _AcousticFeatures,
    sample: int,
) -> AlignmentAcousticEvidence:
    position = int(np.searchsorted(features.samples, sample, side="left"))
    indices = [min(max(position, 0), len(features.samples) - 1)]
    if position > 0:
        indices.append(position - 1)
    index = min(indices, key=lambda value: abs(int(features.samples[value]) - sample))
    return AlignmentAcousticEvidence(
        energy=float(features.energy[index]),
        spectral_change=float(features.spectral_change[index]),
        pitch_change=float(features.pitch_change[index]),
    )


def _phone_density(starts: list[int], sample: int, sample_rate: int) -> float:
    radius = sample_rate * 0.5
    left = bisect.bisect_left(starts, sample - radius)
    right = bisect.bisect_right(starts, sample + radius)
    return float(right - left)


def _boundary_score(
    confidence: float,
    evidence: AlignmentAcousticEvidence,
    *,
    rap: bool,
) -> float:
    if rap:
        return (
            0.58 * confidence
            + 0.14 * evidence.energy
            + 0.18 * evidence.spectral_change
            + 0.10 * evidence.pitch_change
        )
    return (
        0.34 * confidence
        + 0.24 * evidence.energy
        + 0.27 * evidence.spectral_change
        + 0.15 * evidence.pitch_change
    )


def _refine_boundary(
    features: _AcousticFeatures,
    raw_sample: int,
    lower: int,
    upper: int,
    confidence: float,
    *,
    rap: bool,
) -> tuple[int, AlignmentAcousticEvidence, bool]:
    if upper < lower:
        evidence = _feature_evidence(features, raw_sample)
        return raw_sample, evidence, False
    left = int(np.searchsorted(features.samples, lower, side="left"))
    right = int(np.searchsorted(features.samples, upper, side="right"))
    threshold = _RAP_BOUNDARY_THRESHOLD if rap else _NORMAL_BOUNDARY_THRESHOLD
    best_sample = raw_sample
    best_evidence = _feature_evidence(features, raw_sample)
    best_score = _boundary_score(confidence, best_evidence, rap=rap)
    for index in range(left, right):
        sample = int(features.samples[index])
        evidence = AlignmentAcousticEvidence(
            energy=float(features.energy[index]),
            spectral_change=float(features.spectral_change[index]),
            pitch_change=float(features.pitch_change[index]),
        )
        score = _boundary_score(confidence, evidence, rap=rap)
        if score > best_score + 1e-12 or (
            abs(score - best_score) <= 1e-12
            and abs(sample - raw_sample) < abs(best_sample - raw_sample)
        ):
            best_sample = sample
            best_evidence = evidence
            best_score = score
    acoustic_peak = max(
        best_evidence.energy,
        best_evidence.spectral_change,
        best_evidence.pitch_change,
    )
    if acoustic_peak < threshold:
        raw_evidence = _feature_evidence(features, raw_sample)
        return raw_sample, raw_evidence, False
    return best_sample, best_evidence, best_sample != raw_sample


def _merge_evidence(units: list[AlignmentHierarchyUnit]) -> AlignmentAcousticEvidence:
    evidence = [unit.evidence for unit in units if unit.evidence is not None]
    if not evidence:
        return AlignmentAcousticEvidence(energy=0.0, spectral_change=0.0, pitch_change=0.0)
    return AlignmentAcousticEvidence(
        energy=max(item.energy for item in evidence),
        spectral_change=max(item.spectral_change for item in evidence),
        pitch_change=max(item.pitch_change for item in evidence),
    )


def _aggregate_span(
    units: list[AlignmentHierarchyUnit],
) -> tuple[int, int, int, int, float, AlignmentAcousticEvidence]:
    if not units:
        raise AlignmentAdapterError(
            "HUBERT_HIERARCHY_UNALIGNED",
            "A lyric hierarchy unit has no observed HuBERT phone span.",
        )
    return (
        min(item.aligned_start_sample for item in units),
        max(item.aligned_end_sample for item in units),
        min(item.refined_start_sample for item in units),
        max(item.refined_end_sample for item in units),
        float(np.mean([item.confidence for item in units])),
        _merge_evidence(units),
    )


class HubertCTCAlignmentAdapter(CTCAlignmentAdapter):
    """HuBERT CTC plus Japanese hierarchy mapping and vocal boundary refinement."""

    def diagnostics(self, context: AlignmentContext | None = None) -> AdapterDiagnostics:
        base = super().diagnostics(context)
        project_root = context.project_root if context is not None else None
        if project_root is None:
            from ...config import get_settings

            project_root = get_settings().project_root
        processor = project_root / "apps" / "api" / "beatforge_api" / "audio" / "alignment"
        processor /= "lyric_processor.py"
        available = base.available and processor.is_file()
        reason = base.reason
        if base.available and not processor.is_file():
            reason = "Japanese lyric hierarchy processor is missing."
        return AdapterDiagnostics(
            available=available,
            reason=reason,
            model=base.model,
            automatic_downloads_enabled=False,
            details={
                **base.details,
                "engineVersion": ENGINE_VERSION,
                "lyricProcessor": str(processor),
                "lyricProcessorAvailable": processor.is_file(),
                "hierarchyMapping": "dynamic programming",
                "acousticRefinement": ["vocal RMS", "spectral change", "pitch change"],
                "timestampPolicy": "observed HuBERT frames and measured acoustic frames only",
            },
        )

    def _prepare_alignment_inputs(
        self,
        context: AlignmentContext,
        directory: Path,
        python: Path,
        lyrics_path: Path,
    ) -> tuple[str, ...]:
        processor = (
            context.project_root
            / "apps"
            / "api"
            / "beatforge_api"
            / "audio"
            / "alignment"
            / "lyric_processor.py"
        )
        plan_path = directory / "japanese-lyric-plan.json"
        try:
            completed = subprocess.run(
                [
                    str(python),
                    str(processor),
                    "--lyrics-file",
                    str(lyrics_path),
                    "--output",
                    str(plan_path),
                ],
                cwd=context.project_root,
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                timeout=float(os.environ.get("BEATFORGE_G2P_TIMEOUT", "120")),
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise AlignmentAdapterError(
                "HUBERT_G2P_TIMEOUT",
                "Japanese lyric hierarchy processing timed out.",
                details={"timeoutSec": error.timeout},
            ) from error
        try:
            payload = json.loads(plan_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AlignmentAdapterError(
                "HUBERT_G2P_PROCESS_FAILED",
                "Japanese lyric hierarchy processing returned no valid plan.",
                details={
                    "exitCode": completed.returncode,
                    "logTail": (completed.stderr or completed.stdout or "")[-4_000:],
                },
            ) from error
        if (
            completed.returncode != 0
            or not isinstance(payload, dict)
            or payload.get("status") != "ok"
        ):
            raise AlignmentAdapterError(
                str(payload.get("errorCode") or "HUBERT_G2P_PROCESS_FAILED"),
                str(payload.get("errorMessage") or "Japanese lyric hierarchy processing failed."),
                details={
                    "exitCode": completed.returncode,
                    "exceptionType": payload.get("exceptionType"),
                    "logTail": (completed.stderr or completed.stdout or "")[-4_000:],
                },
            )
        try:
            ProcessedLyrics.from_dict(payload)
        except LyricProcessingError as error:
            raise AlignmentAdapterError(
                "HUBERT_G2P_PLAN_INVALID",
                "Japanese lyric hierarchy processing returned an invalid plan.",
            ) from error
        return ("--g2p-plan", str(plan_path))

    def _postprocess_output(
        self,
        context: AlignmentContext,
        payload: dict[str, Any],
        output: AdapterOutput,
    ) -> AdapterOutput:
        started = time.monotonic()
        raw_metadata = payload.get("metadata")
        lyric_plan = raw_metadata.get("lyricPlan") if isinstance(raw_metadata, dict) else None
        if not isinstance(lyric_plan, dict):
            raise AlignmentAdapterError(
                "HUBERT_G2P_PLAN_MISSING",
                "HuBERT CTC completed without its Japanese lyric hierarchy plan.",
            )
        try:
            processed = ProcessedLyrics.from_dict(lyric_plan)
        except LyricProcessingError as error:
            raise AlignmentAdapterError(
                "HUBERT_G2P_PLAN_INVALID",
                "The HuBERT lyric hierarchy plan could not be deserialized.",
            ) from error
        observed_phones = [str(token.phoneme or "") for token in output.tokens]
        phone_mapping = map_phoneme_sequences_dp(processed.phone_sequence, observed_phones)
        if phone_mapping.inserted_observed_indices or any(
            match.observed_index is None for match in phone_mapping.matches
        ):
            raise AlignmentAdapterError(
                "HUBERT_PHONE_DP_INCOMPLETE",
                "Dynamic programming could not map every lyric phone to an observed CTC span.",
                details={
                    "cost": phone_mapping.cost,
                    "insertedObservedIndices": list(phone_mapping.inserted_observed_indices),
                },
            )
        features = _extract_acoustic_features(context)
        raw_starts = sorted(token.start_sample for token in output.tokens)
        window = max(1, round(context.sample_rate * _BOUNDARY_WINDOW_SECONDS))
        phone_units: list[AlignmentHierarchyUnit] = []
        refined_tokens: list[AlignmentToken] = []
        changed_boundaries = 0
        rap_phones = 0
        for plan_phone, match in zip(processed.phonemes, phone_mapping.matches, strict=True):
            observed_index = match.observed_index
            if observed_index is None:  # guarded above; keeps the type checker honest
                raise AlignmentAdapterError(
                    "HUBERT_PHONE_DP_INCOMPLETE",
                    "A lyric phone has no observed CTC span.",
                )
            token = output.tokens[observed_index]
            rap = (
                _phone_density(raw_starts, token.start_sample, context.sample_rate)
                >= _RAP_PHONE_DENSITY_PER_SECOND
            )
            rap_phones += int(rap)
            start_upper = min(token.end_sample - 1, token.start_sample + window)
            refined_start, start_evidence, start_changed = _refine_boundary(
                features,
                token.start_sample,
                token.start_sample,
                start_upper,
                token.confidence,
                rap=rap,
            )
            end_lower = max(refined_start + 1, token.end_sample - window)
            refined_end, end_evidence, end_changed = _refine_boundary(
                features,
                token.end_sample,
                end_lower,
                token.end_sample,
                token.confidence,
                rap=rap,
            )
            if refined_end <= refined_start:
                refined_start = token.start_sample
                refined_end = token.end_sample
                start_evidence = _feature_evidence(features, refined_start)
                end_evidence = _feature_evidence(features, refined_end)
                start_changed = False
                end_changed = False
            changed_boundaries += int(start_changed) + int(end_changed)
            evidence = AlignmentAcousticEvidence(
                energy=max(start_evidence.energy, end_evidence.energy),
                spectral_change=max(
                    start_evidence.spectral_change,
                    end_evidence.spectral_change,
                ),
                pitch_change=max(start_evidence.pitch_change, end_evidence.pitch_change),
            )
            phone_unit = AlignmentHierarchyUnit(
                id=plan_phone.id,
                index=plan_phone.index,
                level="phoneme",
                text=plan_phone.text,
                kana=plan_phone.kana,
                mora=plan_phone.kana,
                phoneme=plan_phone.phoneme,
                character_indices=list(plan_phone.character_indices),
                mora_indices=[plan_phone.mora_index],
                phoneme_indices=[plan_phone.index],
                aligned_start_sample=token.start_sample,
                aligned_end_sample=token.end_sample,
                refined_start_sample=refined_start,
                refined_end_sample=refined_end,
                aligned_sample=token.start_sample,
                refined_sample=refined_start,
                confidence=token.confidence,
                observed_token_index=observed_index,
                match_operation=match.operation,
                evidence=evidence,
            )
            phone_units.append(phone_unit)
            refined_tokens.append(
                AlignmentToken(
                    id=alignment_token_id(
                        context.track_id,
                        self.method,
                        plan_phone.index,
                        refined_start,
                        refined_end,
                    ),
                    text=plan_phone.text,
                    phoneme=plan_phone.phoneme,
                    start_sample=refined_start,
                    end_sample=refined_end,
                    confidence=token.confidence,
                    method=self.method,
                )
            )

        decoded_moras = decode_moras(phone_units, processed)
        if decoded_moras.missing_mora_indices:
            raise AlignmentAdapterError(
                "HUBERT_MORA_DP_INCOMPLETE",
                "Dynamic programming could not decode every expected mora from "
                "observed HuBERT phones.",
                details={
                    "coverage": decoded_moras.coverage,
                    "missingMoraIndices": decoded_moras.missing_mora_indices,
                    "totalDpCost": decoded_moras.total_dp_cost,
                },
            )
        decoded_by_plan_index = {
            event.plan_mora_index: event for event in decoded_moras.events
        }
        mora_units: list[AlignmentHierarchyUnit] = []
        for mora in processed.moras:
            decoded = decoded_by_plan_index[mora.index]
            members = [
                phone_units[index] for index in decoded.observed_phoneme_indices
            ]
            _raw_start, _raw_end, _refined_start, _refined_end, _confidence_value, evidence = (
                _aggregate_span(members)
            )
            mora_units.append(
                AlignmentHierarchyUnit(
                    id=mora.id,
                    index=mora.index,
                    level="mora",
                    text=mora.text,
                    kana=mora.kana,
                    mora=mora.kana,
                    kind=mora.kind,
                    character_indices=list(mora.character_indices),
                    mora_indices=[mora.index],
                    phoneme_indices=list(decoded.observed_phoneme_indices),
                    aligned_start_sample=decoded.aligned_start_sample,
                    aligned_end_sample=decoded.aligned_end_sample,
                    refined_start_sample=decoded.refined_start_sample,
                    refined_end_sample=decoded.refined_end_sample,
                    aligned_sample=decoded.aligned_sample,
                    refined_sample=decoded.refined_sample,
                    confidence=decoded.confidence,
                    evidence=evidence,
                )
            )

        character_units: list[AlignmentHierarchyUnit] = []
        for character in processed.characters:
            members = [mora_units[index] for index in character.mora_indices]
            aligned_start, aligned_end, refined_start, refined_end, confidence, evidence = (
                _aggregate_span(members)
            )
            character_units.append(
                AlignmentHierarchyUnit(
                    id=character.id,
                    index=character.index,
                    level="character",
                    text=character.text,
                    kana=character.kana,
                    character_indices=[character.index],
                    mora_indices=list(character.mora_indices),
                    phoneme_indices=list(character.phoneme_indices),
                    aligned_start_sample=aligned_start,
                    aligned_end_sample=aligned_end,
                    refined_start_sample=refined_start,
                    refined_end_sample=refined_end,
                    aligned_sample=aligned_start,
                    refined_sample=refined_start,
                    confidence=confidence,
                    evidence=evidence,
                )
            )

        hierarchy = AlignmentHierarchy(
            phonemes=phone_units,
            moras=mora_units,
            characters=character_units,
        )
        postprocess_elapsed = time.monotonic() - started
        base_total = output.metadata.get("totalElapsedSec")
        try:
            total_elapsed = float(base_total) + postprocess_elapsed
        except (TypeError, ValueError):
            total_elapsed = postprocess_elapsed
        warnings = tuple(dict.fromkeys((*output.warnings, *processed.warnings)))
        return AdapterOutput(
            tokens=tuple(refined_tokens),
            hierarchy=hierarchy,
            warnings=warnings,
            metadata={
                **output.metadata,
                "engineVersion": ENGINE_VERSION,
                "totalElapsedSec": round(total_elapsed, 3),
                "postprocessElapsedSec": round(postprocess_elapsed, 3),
                "lyricPlanHash": processed.ctc_plan()["planHash"],
                "g2pEngine": processed.g2p_engine,
                "rubyPolicy": processed.ruby_policy,
                "rubyAnnotationCount": len(processed.annotations),
                "hierarchyCounts": {
                    "characters": len(character_units),
                    "moras": len(mora_units),
                    "phonemes": len(phone_units),
                },
                "dynamicProgramming": {
                    "expectedObservedPhoneCost": phone_mapping.cost,
                    "expectedObservedPhoneOperationCounts": {
                        operation: sum(
                            match.operation == operation for match in phone_mapping.matches
                        )
                        for operation in ("match", "substitute", "delete")
                    },
                    "moraPhoneMapping": "global lexical partition DP",
                    "characterMoraMapping": "lexical surface/reading DP",
                    "moraDecoder": decoded_moras.mapping_algorithm,
                },
                "moraDecoder": decoded_moras.model_dump(mode="json", by_alias=True),
                "acousticRefinement": {
                    "features": ["vocalRms", "spectralChange", "pitchChange"],
                    "featureHopSamples": features.hop_samples,
                    "featureFrameSamples": features.frame_samples,
                    "sourceSampleRate": features.source_sample_rate,
                    "searchWindowMs": _BOUNDARY_WINDOW_SECONDS * 1_000.0,
                    "changedBoundaryCount": changed_boundaries,
                    "rapPhoneCount": rap_phones,
                    "rapDensityThresholdPerSec": _RAP_PHONE_DENSITY_PER_SECOND,
                    "normalOnsetThreshold": _NORMAL_BOUNDARY_THRESHOLD,
                    "rapOnsetThreshold": _RAP_BOUNDARY_THRESHOLD,
                    "boundaryCandidates": "measured vocal feature frames inside raw CTC spans",
                    "tempoUsed": False,
                },
                "alignedSampleProvenance": "raw HuBERT CTC Viterbi label-state frame",
                "refinedSampleProvenance": (
                    "measured vocal RMS/spectral/pitch frame, or unchanged raw CTC boundary "
                    "when acoustic evidence is below threshold"
                ),
                "timestampProvenance": (
                    "HuBERT CTC frames plus measured vocal feature frames; no even allocation, "
                    "text-length timing, BPM event creation, or manual correction"
                ),
            },
        )
