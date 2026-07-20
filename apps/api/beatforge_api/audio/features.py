"""HPSS and multi-scale, multi-band onset feature extraction."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d
from scipy.signal import istft, stft

from .config import AnalysisConfig
from .models import FeatureBundle, FloatArray

EPSILON = np.float32(1e-8)


def _pad_for_fft(audio: np.ndarray, fft_size: int) -> np.ndarray:
    if audio.size >= fft_size:
        return audio
    return np.pad(audio, (0, fft_size - audio.size))


def _complex_stft(
    audio: np.ndarray, sample_rate: int, fft_size: int, hop_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    padded = _pad_for_fft(audio, fft_size)
    frequencies, _, spectrum = stft(
        padded,
        fs=sample_rate,
        window="hann",
        nperseg=fft_size,
        noverlap=fft_size - hop_length,
        nfft=fft_size,
        boundary="zeros",
        padded=True,
        return_onesided=True,
    )
    frame_samples = np.arange(spectrum.shape[1], dtype=np.int64) * hop_length
    return frequencies.astype(np.float32), frame_samples, spectrum


def separate_harmonic_percussive(
    audio: np.ndarray, sample_rate: int, config: AnalysisConfig
) -> tuple[FloatArray, FloatArray]:
    """Perform deterministic median-mask HPSS without requiring librosa."""

    fft_size = 1024
    hop_length = 256
    _, _, spectrum = _complex_stft(audio, sample_rate, fft_size, hop_length)
    magnitude = np.abs(spectrum).astype(np.float32)
    harmonic_median = median_filter(
        magnitude,
        size=(1, max(3, config.hpss_kernel_time | 1)),
        mode="nearest",
    )
    percussive_median = median_filter(
        magnitude,
        size=(max(3, config.hpss_kernel_frequency | 1), 1),
        mode="nearest",
    )
    harmonic_power = np.square(
        harmonic_median / max(config.hpss_margin_harmonic, 1e-3)
    )
    percussive_power = np.square(
        percussive_median / max(config.hpss_margin_percussive, 1e-3)
    )
    denominator = harmonic_power + percussive_power + EPSILON
    harmonic_mask = harmonic_power / denominator
    percussive_mask = percussive_power / denominator
    _, harmonic = istft(
        spectrum * harmonic_mask,
        fs=sample_rate,
        window="hann",
        nperseg=fft_size,
        noverlap=fft_size - hop_length,
        nfft=fft_size,
        input_onesided=True,
        boundary=True,
    )
    _, percussive = istft(
        spectrum * percussive_mask,
        fs=sample_rate,
        window="hann",
        nperseg=fft_size,
        noverlap=fft_size - hop_length,
        nfft=fft_size,
        input_onesided=True,
        boundary=True,
    )

    def fit_length(values: np.ndarray) -> FloatArray:
        if values.size < audio.size:
            values = np.pad(values, (0, audio.size - values.size))
        return np.asarray(values[: audio.size], dtype=np.float32)

    return fit_length(percussive), fit_length(harmonic)


def robust_adaptive_normalize(
    curve: np.ndarray, window_frames: int, smooth_sigma: float = 0.65
) -> FloatArray:
    """Normalize a novelty curve against its rolling median and rolling MAD."""

    values = np.nan_to_num(np.asarray(curve, dtype=np.float32), copy=True)
    if not values.size or float(np.max(values)) <= 1e-12:
        return np.zeros(values.shape, dtype=np.float32)
    size = max(5, int(window_frames) | 1)
    local_median = median_filter(values, size=size, mode="reflect")
    absolute_deviation = np.abs(values - local_median)
    local_mad = median_filter(absolute_deviation, size=size, mode="reflect")
    # In sparse novelty functions the true MAD is often zero. A small local mean
    # deviation is a robust floor and preserves quiet-section recall.
    deviation_floor = uniform_filter1d(
        absolute_deviation, size=max(3, size // 4), mode="reflect"
    )
    scale = np.maximum(1.4826 * local_mad, 0.35 * deviation_floor)
    positive = np.maximum(values - local_median, 0.0)
    normalized = positive / np.maximum(scale, np.max(values) * 1e-5 + 1e-8)
    normalized = np.clip(normalized, 0.0, 16.0)
    if normalized.size > 3 and smooth_sigma > 0:
        normalized = gaussian_filter1d(normalized, smooth_sigma, mode="nearest")
    return normalized.astype(np.float32, copy=False)


def _positive_flux(magnitude: np.ndarray) -> FloatArray:
    # Log compression and per-frame L2 normalization prevent loud sustained bins
    # from overwhelming newly appearing spectral energy.
    compressed = np.log1p(80.0 * magnitude).astype(np.float32)
    norms = np.linalg.norm(compressed, axis=0, keepdims=True)
    normalized = compressed / np.maximum(norms, EPSILON)
    difference = np.diff(normalized, axis=1, prepend=normalized[:, :1])
    return np.sum(np.maximum(difference, 0.0), axis=0, dtype=np.float32)


def _resize_curve(curve: np.ndarray, length: int) -> FloatArray:
    if curve.size == length:
        return np.asarray(curve, dtype=np.float32)
    if curve.size <= 1:
        return np.full(length, float(curve[0]) if curve.size else 0.0, dtype=np.float32)
    source = np.linspace(0.0, 1.0, curve.size)
    target = np.linspace(0.0, 1.0, length)
    return np.interp(target, source, curve).astype(np.float32)


def _band_mask(
    frequencies: np.ndarray, low_hz: float, high_hz: float | None
) -> np.ndarray:
    if high_hz is None:
        return frequencies >= low_hz
    return (frequencies >= low_hz) & (frequencies < high_hz)


def _frame_energy_curve(
    audio: np.ndarray, frame_samples: np.ndarray, window_size: int
) -> FloatArray:
    power = uniform_filter1d(
        np.square(audio, dtype=np.float32), size=max(3, window_size), mode="nearest"
    )
    indices = np.minimum(frame_samples, audio.size - 1)
    return np.sqrt(np.maximum(power[indices], 0.0)).astype(np.float32)


def _fuse_curves(
    normalized: dict[str, FloatArray], config: AnalysisConfig
) -> FloatArray:
    def evidence(name: str) -> FloatArray:
        curve = normalized[name]
        return np.asarray(curve / (curve + 1.75), dtype=np.float32)

    # Correlated transforms of the same mix are one evidence family, not
    # independent votes. Grouping prevents broadband mastered material from
    # receiving eleven copies of essentially the same spectral change.
    spectral = 0.55 * evidence("mix_flux") + 0.45 * evidence("fine_flux")
    percussive = evidence("percussive_flux")
    energy = np.maximum(evidence("energy_derivative"), evidence("rms_derivative"))
    low = np.maximum(evidence("sub_low"), 0.82 * evidence("low_mid"))
    mid = evidence("mid")
    high = np.maximum.reduce(
        (evidence("high"), 0.82 * evidence("air"), 0.78 * evidence("hfc_change"))
    )
    fused = (
        0.27 * spectral
        + 0.24 * percussive
        + 0.17 * energy
        + 0.13 * low
        + 0.08 * mid
        + 0.11 * high
    )
    primary = np.maximum(spectral, percussive)
    fused = fused * (0.76 + 0.24 * primary)
    return gaussian_filter1d(fused, 0.72, mode="nearest").astype(np.float32)


def _feature_window_frames(config: AnalysisConfig) -> int:
    return max(
        5,
        int(
            round(
                config.robust_window_sec
                * config.analysis_sample_rate
                / config.coarse_hop_length
            )
        ),
    )


def extract_features(
    audio: np.ndarray,
    sample_rate: int,
    config: AnalysisConfig,
    *,
    extra_stems: dict[str, np.ndarray] | None = None,
) -> FeatureBundle:
    """Compute independent onset evidence at multiple FFT and time scales."""

    percussive, harmonic = separate_harmonic_percussive(audio, sample_rate, config)
    curves: dict[str, FloatArray] = {}
    scale_curves: list[FloatArray] = []
    base_magnitude: np.ndarray | None = None
    base_frequencies: np.ndarray | None = None
    frame_samples: np.ndarray | None = None

    for fft_size in config.fft_sizes:
        frequencies, samples, spectrum = _complex_stft(
            audio, sample_rate, fft_size, config.coarse_hop_length
        )
        magnitude = np.abs(spectrum).astype(np.float32)
        flux = _positive_flux(magnitude)
        if frame_samples is None:
            frame_samples = samples
            base_magnitude = magnitude
            base_frequencies = frequencies
        else:
            flux = _resize_curve(flux, frame_samples.size)
        scale_curves.append(flux)
        curves[f"mix_flux_scale_{fft_size}"] = flux

    assert frame_samples is not None
    assert base_magnitude is not None
    assert base_frequencies is not None
    curves["mix_flux"] = np.mean(np.stack(scale_curves), axis=0).astype(np.float32)

    _, _, percussive_spectrum = _complex_stft(
        percussive,
        sample_rate,
        config.fft_sizes[0],
        config.coarse_hop_length,
    )
    curves["percussive_flux"] = _resize_curve(
        _positive_flux(np.abs(percussive_spectrum).astype(np.float32)),
        frame_samples.size,
    )

    # Log compression and per-bin averaging make the raw band profiles usable
    # for classification. Summing linear magnitudes made wide high-frequency
    # bands dominate and robust-z values cannot be compared between bands.
    classification_magnitude = np.log1p(80.0 * base_magnitude).astype(np.float32)
    positive_difference = np.maximum(
        np.diff(
            classification_magnitude,
            axis=1,
            prepend=classification_magnitude[:, :1],
        ),
        0.0,
    )
    band_energy: dict[str, FloatArray] = {}
    for band_name, low_hz, high_hz in config.band_edges_hz:
        mask = _band_mask(base_frequencies, low_hz, high_hz)
        if np.any(mask):
            band_curve = np.mean(positive_difference[mask], axis=0, dtype=np.float32)
        else:
            band_curve = np.zeros(frame_samples.size, dtype=np.float32)
        curves[band_name] = band_curve
        band_energy[band_name] = band_curve

    rms = _frame_energy_curve(audio, frame_samples, config.fft_sizes[0] // 4)
    energy = np.square(rms, dtype=np.float32)
    curves["energy_derivative"] = np.maximum(
        np.diff(energy, prepend=energy[:1]), 0.0
    ).astype(np.float32)
    curves["rms_derivative"] = np.maximum(
        np.diff(rms, prepend=rms[:1]), 0.0
    ).astype(np.float32)
    frequency_weight = (
        base_frequencies / max(float(base_frequencies[-1]), 1.0)
    ).astype(np.float32)
    hfc = np.mean(
        classification_magnitude * frequency_weight[:, np.newaxis],
        axis=0,
        dtype=np.float32,
    )
    curves["hfc_change"] = np.maximum(
        np.diff(hfc, prepend=hfc[:1]), 0.0
    ).astype(np.float32)

    fine_frequencies, fine_samples, fine_spectrum = _complex_stft(
        audio, sample_rate, config.fine_fft_size, config.fine_hop_length
    )
    del fine_frequencies
    fine_flux = _positive_flux(np.abs(fine_spectrum).astype(np.float32))
    curves["fine_flux"] = np.interp(
        frame_samples,
        fine_samples,
        fine_flux,
        left=0.0,
        right=0.0,
    ).astype(np.float32)

    # Optional stems only contribute evidence. Their frame positions are never
    # used as final samples; final refinement always operates on the original mix.
    if extra_stems:
        for stem_name in ("drums", "bass", "other"):
            stem = extra_stems.get(stem_name)
            if stem is None or not np.asarray(stem).size:
                continue
            _, stem_samples, stem_spectrum = _complex_stft(
                np.asarray(stem, dtype=np.float32),
                sample_rate,
                config.fft_sizes[0],
                config.coarse_hop_length,
            )
            stem_flux = _positive_flux(np.abs(stem_spectrum).astype(np.float32))
            curves[f"stem_{stem_name}"] = np.interp(
                frame_samples, stem_samples, stem_flux, left=0.0, right=0.0
            ).astype(np.float32)

    window_frames = _feature_window_frames(config)
    normalized: dict[str, FloatArray] = {}
    # Scale-specific curves are retained for vote consistency but not directly
    # fused; their mean is the mix_flux feature.
    fusion_names: Iterable[str] = (
        "mix_flux",
        "percussive_flux",
        "sub_low",
        "low_mid",
        "mid",
        "high",
        "air",
        "energy_derivative",
        "rms_derivative",
        "hfc_change",
        "fine_flux",
    )
    for name, curve in curves.items():
        normalized[name] = robust_adaptive_normalize(curve, window_frames)

    if extra_stems:
        # Give each present stem a modest vote by blending it into the detector it
        # is most relevant to, preserving the same calibrated fusion weights.
        if "stem_drums" in normalized:
            normalized["percussive_flux"] = np.maximum(
                normalized["percussive_flux"], 0.72 * normalized["stem_drums"]
            )
        if "stem_bass" in normalized:
            normalized["sub_low"] = np.maximum(
                normalized["sub_low"], 0.62 * normalized["stem_bass"]
            )
        if "stem_other" in normalized:
            normalized["mid"] = np.maximum(
                normalized["mid"], 0.48 * normalized["stem_other"]
            )

    fused = _fuse_curves({name: normalized[name] for name in fusion_names}, config)
    scale_evidence = np.stack(
        [normalized[f"mix_flux_scale_{fft_size}"] for fft_size in config.fft_sizes]
    )
    scale_votes = np.sum(scale_evidence >= 1.0, axis=0, dtype=np.int16)
    return FeatureBundle(
        hop_length=config.coarse_hop_length,
        sample_rate=sample_rate,
        frame_samples=frame_samples,
        curves=curves,
        normalized_curves=normalized,
        fused=fused,
        band_energy=band_energy,
        scale_votes=scale_votes,
        percussive=percussive,
        harmonic=harmonic,
    )
