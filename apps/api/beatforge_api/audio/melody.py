"""Local monophonic melody note-onset extraction for the Demucs ``other`` stem."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

from .models import OnsetCandidate


@dataclass(frozen=True, slots=True)
class MelodyExtractionResult:
    candidates: list[OnsetCandidate]
    voiced_frame_count: int
    pitch_onset_count: int
    energy_reattack_count: int
    method: str = "librosa_pyin_local"


def _refine_attack(audio: np.ndarray, sample: int, sample_rate: int) -> int:
    start = max(0, sample - round(sample_rate * 0.018))
    end = min(audio.size, sample + round(sample_rate * 0.030))
    if end <= start + 2:
        return min(max(sample, 0), max(0, audio.size - 1))
    envelope = uniform_filter1d(
        np.abs(audio[start:end]).astype(np.float64),
        size=max(3, round(sample_rate * 0.003)),
        mode="nearest",
    )
    attack = np.maximum(np.gradient(envelope), 0.0)
    return start + int(np.argmax(attack))


def extract_melody_candidates(
    audio: np.ndarray,
    sample_rate: int,
    *,
    hop_length: int = 256,
) -> MelodyExtractionResult:
    """Find voiced pitch changes and re-attacks without any cloud or text input."""

    values = np.asarray(audio, dtype=np.float32)
    if values.ndim != 1:
        values = np.mean(values, axis=-1, dtype=np.float32)
    if values.size < 2_048 or float(np.max(np.abs(values))) < 1e-6:
        return MelodyExtractionResult([], 0, 0, 0)

    normalized = values / max(float(np.quantile(np.abs(values), 0.995)), 1e-6)
    frame_length = 2_048
    f0, voiced, voiced_probability = librosa.pyin(
        normalized.astype(np.float64),
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C7")),
        sr=sample_rate,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
        fill_na=np.nan,
    )
    if f0.size == 0:
        return MelodyExtractionResult([], 0, 0, 0)
    rms = librosa.feature.rms(
        y=normalized,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]
    rms_rise = np.maximum(np.gradient(rms.astype(np.float64)), 0.0)
    rise_reference = max(float(np.quantile(rms_rise, 0.94)), 1e-9)
    midi = librosa.hz_to_midi(f0)
    pitch_events: list[tuple[int, float, str]] = []
    previous_voiced = False
    previous_pitch = np.nan
    for frame in range(f0.size):
        is_voiced = bool(voiced[frame]) and np.isfinite(f0[frame])
        probability = float(np.clip(voiced_probability[frame], 0.0, 1.0))
        if not is_voiced or probability < 0.42:
            previous_voiced = False
            continue
        pitch_jump = (
            abs(float(midi[frame] - previous_pitch))
            if previous_voiced and np.isfinite(previous_pitch)
            else float("inf")
        )
        rise_score = float(np.clip(rms_rise[frame] / rise_reference, 0.0, 1.0))
        event_kind: str | None = None
        if not previous_voiced or pitch_jump >= 0.75:
            event_kind = "pitch_onset"
        elif rise_score >= 0.72:
            event_kind = "energy_reattack"
        if event_kind is not None:
            pitch_events.append((frame, probability, event_kind))
        previous_voiced = True
        previous_pitch = float(midi[frame])

    # Suppress pitch-tracker flutter while retaining fast but playable note changes.
    frame_scores = np.zeros(f0.size, dtype=np.float64)
    kinds: dict[int, str] = {}
    probabilities: dict[int, float] = {}
    for frame, probability, kind in pitch_events:
        score = probability + (0.18 if kind == "pitch_onset" else 0.0)
        frame_scores[frame] = max(frame_scores[frame], score)
        kinds[frame] = kind
        probabilities[frame] = probability
    peaks, _ = find_peaks(
        frame_scores,
        height=0.42,
        distance=max(1, round(sample_rate * 0.055 / hop_length)),
    )
    candidates: list[OnsetCandidate] = []
    pitch_onset_count = 0
    energy_reattack_count = 0
    for frame in peaks:
        coarse_sample = min(values.size - 1, max(0, int(frame) * hop_length))
        refined_sample = _refine_attack(values, coarse_sample, sample_rate)
        probability = probabilities.get(int(frame), float(voiced_probability[frame]))
        kind = kinds.get(int(frame), "pitch_onset")
        if kind == "pitch_onset":
            pitch_onset_count += 1
        else:
            energy_reattack_count += 1
        confidence = float(np.clip(0.30 + 0.62 * probability, 0.0, 0.94))
        candidates.append(
            OnsetCandidate(
                detected_sample=coarse_sample,
                refined_sample=refined_sample,
                sample=refined_sample,
                band="mid_hit",
                confidence=confidence,
                salience=float(np.clip(0.25 + 0.55 * probability, 0.0, 0.9)),
                source="stems",
                detector_votes=["melody_pitch_onset", kind],
                peak_value=probability,
                prominence=probability,
                loudness=float(rms[frame]),
                primary_stem="other",
                semantic_evidence={
                    "lyricAlignment": 0.0,
                    "phonemeConfidence": 0.0,
                    "pitchConfidence": probability,
                },
            )
        )
    return MelodyExtractionResult(
        candidates=candidates,
        voiced_frame_count=int(np.count_nonzero(voiced)),
        pitch_onset_count=pitch_onset_count,
        energy_reattack_count=energy_reattack_count,
    )
