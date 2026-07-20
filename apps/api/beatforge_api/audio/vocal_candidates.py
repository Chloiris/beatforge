"""Acoustic singing-event candidates independent of lyric timestamps."""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks


@dataclass(frozen=True, slots=True)
class VocalAcousticCandidate:
    sample: int
    confidence: float
    onset_score: float
    envelope_score: float
    pitch_score: float
    transition_score: float
    activity_score: float


@dataclass(frozen=True, slots=True)
class VocalAcousticResult:
    candidates: list[VocalAcousticCandidate]
    method: str = "local_vocal_onset_pitch_transition"


def _robust_unit(values: np.ndarray, quantile: float = 0.98) -> np.ndarray:
    reference = max(float(np.quantile(values, quantile)), 1e-9)
    return np.clip(values / reference, 0.0, 1.5)


def _refine_attack(audio: np.ndarray, sample: int, sample_rate: int) -> int:
    start = max(0, sample - round(sample_rate * 0.016))
    end = min(audio.size, sample + round(sample_rate * 0.032))
    if end <= start + 2:
        return min(max(sample, 0), max(0, audio.size - 1))
    envelope = uniform_filter1d(
        np.abs(audio[start:end]).astype(np.float64),
        size=max(3, round(sample_rate * 0.003)),
        mode="nearest",
    )
    return start + int(np.argmax(np.maximum(np.gradient(envelope), 0.0)))


def extract_vocal_acoustic_candidates(
    audio: np.ndarray,
    sample_rate: int,
    *,
    hop_length: int = 256,
) -> VocalAcousticResult:
    """Detect vocal onsets, envelope changes, pitch attacks, and spectral transitions."""

    values = np.asarray(audio, dtype=np.float32)
    if values.ndim != 1:
        values = np.mean(values, axis=-1, dtype=np.float32)
    if values.size < 2_048 or float(np.max(np.abs(values))) < 1e-6:
        return VocalAcousticResult([])

    scale = max(float(np.quantile(np.abs(values), 0.995)), 1e-6)
    normalized = np.asarray(values / scale, dtype=np.float64)
    n_fft = 1_024
    magnitude = np.abs(
        librosa.stft(
            normalized,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            center=True,
        )
    )
    rms = librosa.feature.rms(S=magnitude, frame_length=n_fft, hop_length=hop_length)[0]
    original_rms = rms * scale
    activity_reference = max(float(np.quantile(original_rms, 0.90)), 1e-9)
    noise_floor = max(float(np.quantile(original_rms, 0.20)), 1e-9)
    activity_floor = max(10.0 ** (-52.0 / 20.0), activity_reference * 0.025, noise_floor * 5.0)
    activity = np.clip(
        (original_rms - activity_floor)
        / max(activity_reference - activity_floor, 1e-9),
        0.0,
        1.0,
    )

    onset = librosa.onset.onset_strength(
        S=librosa.amplitude_to_db(np.maximum(magnitude, 1e-8), ref=np.max),
        sr=sample_rate,
        hop_length=hop_length,
        aggregate=np.median,
    )
    envelope_rise = np.maximum(np.gradient(rms.astype(np.float64)), 0.0)
    normalized_magnitude = magnitude / np.maximum(np.sum(magnitude, axis=0), 1e-9)
    spectral_flux = np.sqrt(
        np.sum(
            np.square(
                np.maximum(
                    np.diff(
                        normalized_magnitude,
                        axis=1,
                        prepend=normalized_magnitude[:, :1],
                    ),
                    0.0,
                )
            ),
            axis=0,
        )
    )
    f0 = librosa.yin(
        normalized,
        fmin=70.0,
        fmax=min(1_200.0, sample_rate * 0.45),
        sr=sample_rate,
        frame_length=2_048,
        hop_length=hop_length,
        center=True,
    )
    midi = librosa.hz_to_midi(f0)
    pitch_change = np.abs(np.diff(midi, prepend=midi[0]))
    pitch_change = np.where(activity > 0.05, pitch_change, 0.0)

    onset_score = _robust_unit(onset)
    envelope_score = _robust_unit(envelope_rise)
    transition_score = _robust_unit(spectral_flux)
    pitch_score = np.clip(pitch_change / 2.0, 0.0, 1.5)
    size = min(
        onset_score.size,
        envelope_score.size,
        transition_score.size,
        pitch_score.size,
        activity.size,
    )
    novelty = (
        0.34 * onset_score[:size]
        + 0.25 * envelope_score[:size]
        + 0.23 * transition_score[:size]
        + 0.18 * pitch_score[:size]
    ) * np.clip(activity[:size] + 0.18, 0.0, 1.0)
    peaks, properties = find_peaks(
        novelty,
        height=0.16,
        prominence=0.045,
        distance=max(1, round(sample_rate * 0.045 / hop_length)),
    )
    prominences = properties.get("prominences", np.zeros(peaks.size))
    candidates: list[VocalAcousticCandidate] = []
    for frame, prominence in zip(peaks, prominences, strict=False):
        if original_rms[frame] < activity_floor:
            continue
        coarse = min(values.size - 1, int(frame) * hop_length)
        refined = _refine_attack(values, coarse, sample_rate)
        event_confidence = float(
            np.clip(
                0.16
                + 0.24 * onset_score[frame]
                + 0.18 * envelope_score[frame]
                + 0.18 * pitch_score[frame]
                + 0.16 * transition_score[frame]
                + 0.08 * activity[frame]
                + 0.05 * prominence,
                0.0,
                0.94,
            )
        )
        candidates.append(
            VocalAcousticCandidate(
                sample=refined,
                confidence=event_confidence,
                onset_score=float(np.clip(onset_score[frame], 0.0, 1.0)),
                envelope_score=float(np.clip(envelope_score[frame], 0.0, 1.0)),
                pitch_score=float(np.clip(pitch_score[frame], 0.0, 1.0)),
                transition_score=float(np.clip(transition_score[frame], 0.0, 1.0)),
                activity_score=float(np.clip(activity[frame], 0.0, 1.0)),
            )
        )
    return VocalAcousticResult(candidates)
