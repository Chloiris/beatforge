"""Audio decoding, preprocessing, and exact sample-rate mapping."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from .config import AnalysisConfig
from .models import AudioData


class AudioDecodeError(ValueError):
    """Raised when libsndfile cannot decode a source file.

    The API layer may catch this exception and retry through ffmpeg without the
    analysis core needing to own temporary-file policy.
    """


def _mono_from_channels(samples: np.ndarray) -> np.ndarray:
    if samples.ndim == 1:
        return samples.astype(np.float32, copy=False)
    # An arithmetic mean is deterministic and keeps true inter-channel attacks.
    return np.mean(samples, axis=1, dtype=np.float32)


def _preprocess(mono: np.ndarray) -> tuple[np.ndarray, float, float]:
    if mono.size == 0:
        raise AudioDecodeError("Audio file contains no samples")
    dc_offset = float(np.median(mono))
    centered = mono.astype(np.float32, copy=True) - np.float32(dc_offset)
    robust_peak = float(np.quantile(np.abs(centered), 0.995))
    if not np.isfinite(robust_peak) or robust_peak < 1e-8:
        gain = 1.0
    else:
        # Leave headroom while preventing isolated impulses from setting the gain.
        gain = min(32.0, 0.92 / robust_peak)
    normalized = np.clip(centered * np.float32(gain), -1.0, 1.0)
    return normalized.astype(np.float32, copy=False), dc_offset, float(gain)


def detect_leading_silence(
    mono: np.ndarray, sample_rate: int, threshold_db: float = -52.0
) -> int:
    """Return the first sustained non-silent sample without modifying the audio."""

    if mono.size == 0:
        return 0
    window = max(8, int(round(sample_rate * 0.008)))
    hop = max(1, window // 4)
    if mono.size < window:
        peak = float(np.max(np.abs(mono)))
        return int(np.argmax(np.abs(mono) >= peak * 10.0 ** (threshold_db / 20.0)))
    starts = np.arange(0, mono.size - window + 1, hop, dtype=np.int64)
    # Cumulative energy avoids allocating a frame matrix for long uploads.
    squared = np.square(mono, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    energy = cumulative[starts + window] - cumulative[starts]
    rms = np.sqrt(np.maximum(energy / window, 0.0))
    reference = max(float(np.quantile(rms, 0.98)), 1e-7)
    threshold = max(reference * 10.0 ** (threshold_db / 20.0), 2e-6)
    active = rms >= threshold
    # Require two adjacent windows to reject isolated codec/dither spikes.
    sustained = active & np.r_[active[1:], False]
    indices = np.flatnonzero(sustained)
    if not indices.size:
        return 0 if np.max(np.abs(mono)) > 1e-7 else int(mono.size)
    coarse = int(starts[int(indices[0])])
    search_end = min(mono.size, coarse + window + hop)
    envelope = np.abs(mono[coarse:search_end])
    sample_threshold = max(threshold * 0.35, reference * 1e-4)
    crossings = np.flatnonzero(envelope >= sample_threshold)
    return coarse + int(crossings[0]) if crossings.size else coarse


def _resample_exact(mono: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return mono.astype(np.float32, copy=True)
    divisor = math.gcd(source_rate, target_rate)
    result = resample_poly(
        mono,
        up=target_rate // divisor,
        down=source_rate // divisor,
        window=("kaiser", 8.6),
    ).astype(np.float32, copy=False)
    expected_length = int(round(mono.size * target_rate / source_rate))
    if result.size > expected_length:
        result = result[:expected_length]
    elif result.size < expected_length:
        result = np.pad(result, (0, expected_length - result.size))
    return np.ascontiguousarray(result, dtype=np.float32)


def _prepare_analysis_channels(
    raw: np.ndarray,
    source_rate: int,
    target_rate: int,
    dc_offset: float,
    gain: float,
) -> np.ndarray:
    """Create a normalized, time-aligned channel-last copy for separation."""

    channels = raw[:, np.newaxis] if raw.ndim == 1 else raw
    normalized = np.clip(
        (channels.astype(np.float32) - np.float32(dc_offset)) * np.float32(gain),
        -1.0,
        1.0,
    )
    if source_rate == target_rate:
        return np.ascontiguousarray(normalized, dtype=np.float32)
    resampled = [
        _resample_exact(normalized[:, index], source_rate, target_rate)
        for index in range(normalized.shape[1])
    ]
    return np.ascontiguousarray(np.stack(resampled, axis=1), dtype=np.float32)


def audio_from_array(
    samples: np.ndarray,
    sample_rate: int,
    config: AnalysisConfig,
    *,
    path: Path | None = None,
) -> AudioData:
    """Build an :class:`AudioData` from samples, useful for decoding and tests."""

    if sample_rate <= 0:
        raise AudioDecodeError("Sample rate must be positive")
    raw = np.asarray(samples, dtype=np.float32)
    if raw.ndim not in (1, 2):
        raise AudioDecodeError("Audio samples must be mono or channel-last stereo")
    channels = 1 if raw.ndim == 1 else int(raw.shape[1])
    sample_count = int(raw.shape[0])
    mono_original = _mono_from_channels(raw)
    normalized, dc_offset, gain = _preprocess(mono_original)
    analysis_channels = _prepare_analysis_channels(
        raw,
        sample_rate,
        config.analysis_sample_rate,
        dc_offset,
        gain,
    )
    analysis_mono = np.mean(analysis_channels, axis=1, dtype=np.float32)
    leading_analysis = detect_leading_silence(
        analysis_mono,
        config.analysis_sample_rate,
        threshold_db=config.leading_silence_db,
    )
    leading_original = int(
        round(leading_analysis * sample_rate / config.analysis_sample_rate)
    )
    return AudioData(
        path=path,
        original=np.ascontiguousarray(raw),
        mono=np.ascontiguousarray(mono_original),
        analysis_channels=analysis_channels,
        analysis_mono=analysis_mono,
        original_sample_rate=int(sample_rate),
        analysis_sample_rate=config.analysis_sample_rate,
        channels=channels,
        sample_count=sample_count,
        duration_sec=sample_count / float(sample_rate),
        leading_silence_samples=min(leading_original, sample_count),
        normalization_gain=gain,
        dc_offset=dc_offset,
    )


def load_audio(path: str | Path, config: AnalysisConfig) -> AudioData:
    """Decode lossless/libsndfile-supported audio and retain exact source metadata."""

    source = Path(path)
    try:
        info = sf.info(source)
        samples, sample_rate = sf.read(
            source,
            dtype="float32",
            always_2d=True,
            fill_value=0.0,
        )
    except (RuntimeError, OSError) as exc:
        raise AudioDecodeError(f"Unable to decode audio with libsndfile: {exc}") from exc
    if info.frames != samples.shape[0] or info.samplerate != sample_rate:
        raise AudioDecodeError("Decoded audio metadata does not match sample data")
    return audio_from_array(samples, int(sample_rate), config, path=source)
