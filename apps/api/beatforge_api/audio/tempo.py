"""Tempo and beat-phase estimation independent from onset detection."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate, find_peaks

from .config import AnalysisConfig
from .models import FeatureBundle, OnsetCandidate, TempoEstimate


def _autocorrelation_candidates(
    envelope: np.ndarray, sample_rate: int, hop_length: int, config: AnalysisConfig
) -> tuple[list[float], np.ndarray, np.ndarray]:
    values = np.asarray(envelope, dtype=np.float64)
    values = np.maximum(values - np.median(values), 0.0)
    if values.size < 4 or float(np.max(values)) <= 1e-9:
        return [], np.zeros(1), np.zeros(1)
    values /= max(float(np.max(values)), 1e-9)
    autocorrelation = correlate(values, values, mode="full", method="fft")
    autocorrelation = autocorrelation[values.size - 1 :]
    overlap = np.arange(values.size, 0, -1, dtype=np.float64)
    autocorrelation /= np.maximum(overlap, 1.0)
    minimum_lag = max(
        1,
        int(math.floor(sample_rate * 60.0 / (config.tempo_max_bpm * hop_length))),
    )
    maximum_lag = min(
        autocorrelation.size - 1,
        int(math.ceil(sample_rate * 60.0 / (config.tempo_min_bpm * hop_length))),
    )
    if maximum_lag <= minimum_lag:
        return [], autocorrelation, np.zeros_like(autocorrelation)
    tempo_range = autocorrelation[minimum_lag : maximum_lag + 1]
    low = float(np.min(tempo_range))
    high = float(np.max(tempo_range))
    normalized = np.zeros_like(autocorrelation)
    normalized[minimum_lag : maximum_lag + 1] = (tempo_range - low) / max(
        high - low, 1e-9
    )
    peaks, properties = find_peaks(
        tempo_range,
        distance=max(1, minimum_lag // 8),
        prominence=max((high - low) * 0.015, 1e-10),
    )
    if peaks.size:
        order = np.argsort(properties["prominences"] + tempo_range[peaks])[-16:]
        selected_lags = peaks[order] + minimum_lag
    else:
        selected_lags = np.argsort(tempo_range)[-12:] + minimum_lag
    bpms: list[float] = []
    for integer_lag in selected_lags:
        lag = float(integer_lag)
        if 1 <= integer_lag < autocorrelation.size - 1:
            left, center, right = autocorrelation[integer_lag - 1 : integer_lag + 2]
            denominator = left - 2.0 * center + right
            if abs(denominator) > 1e-12:
                lag += float(np.clip(0.5 * (left - right) / denominator, -0.5, 0.5))
        bpm = sample_rate * 60.0 / (hop_length * lag)
        for multiplier in (0.5, 1.0, 2.0):
            hypothesis = bpm * multiplier
            if config.tempo_min_bpm <= hypothesis <= config.tempo_max_bpm:
                bpms.append(hypothesis)
    return bpms, autocorrelation, normalized


def _interval_candidates(
    candidates: list[OnsetCandidate], sample_rate: int, config: AnalysisConfig
) -> list[float]:
    if len(candidates) < 2:
        return []
    salient = sorted(candidates, key=lambda item: item.salience, reverse=True)
    salient = sorted(salient[: min(96, len(salient))], key=lambda item: item.sample)
    samples = np.array([item.sample for item in salient], dtype=np.float64)
    intervals: list[float] = []
    for skip in (1, 2, 3, 4):
        if samples.size <= skip:
            continue
        differences = samples[skip:] - samples[:-skip]
        for difference in differences:
            if difference <= 0:
                continue
            base = sample_rate * 60.0 / difference
            for multiplier in (0.5, 1.0, 2.0, 3.0, 4.0):
                bpm = base * multiplier
                if config.tempo_min_bpm <= bpm <= config.tempo_max_bpm:
                    intervals.append(float(bpm))
    if not intervals:
        return []
    # A small histogram suppresses one-off inter-onset intervals before fine search.
    bins = np.arange(config.tempo_min_bpm, config.tempo_max_bpm + 0.5, 0.5)
    histogram, edges = np.histogram(intervals, bins=bins)
    top = np.argsort(histogram)[-12:]
    return [float((edges[index] + edges[index + 1]) * 0.5) for index in top if histogram[index]]


def _event_weights(candidates: list[OnsetCandidate]) -> tuple[np.ndarray, np.ndarray]:
    samples = np.array([item.sample for item in candidates], dtype=np.float64)
    weights = np.array(
        [
            max(0.04, item.salience) ** 1.7
            * (1.24 if item.band in ("low_hit", "full_band_accent") else 1.0)
            * (0.72 + 0.28 * item.confidence)
            for item in candidates
        ],
        dtype=np.float64,
    )
    if samples.size > 160:
        threshold = float(np.quantile(weights, 0.30))
        keep = weights >= threshold
        samples, weights = samples[keep], weights[keep]
    return samples, weights


def _phase_and_alignment(
    bpm: float,
    samples: np.ndarray,
    weights: np.ndarray,
    sample_rate: int,
    tolerance_samples: float,
) -> tuple[float, float]:
    period = sample_rate * 60.0 / bpm
    residuals = np.mod(samples, period)
    bin_count = 192
    histogram, _ = np.histogram(
        residuals,
        bins=bin_count,
        range=(0.0, period),
        weights=weights,
    )
    sigma_bins = max(1.0, tolerance_samples / period * bin_count)
    smoothed = gaussian_filter1d(histogram.astype(np.float64), sigma_bins, mode="wrap")
    peak_bin = int(np.argmax(smoothed))
    phase = (peak_bin + 0.5) * period / bin_count
    signed = np.mod(residuals - phase + period / 2.0, period) - period / 2.0
    nearby = np.abs(signed) <= max(tolerance_samples * 1.75, period / bin_count)
    if np.any(nearby):
        phase = float(
            np.mod(phase + np.average(signed[nearby], weights=weights[nearby]), period)
        )
    distance = np.abs(np.mod(samples - phase + period / 2.0, period) - period / 2.0)
    similarities = np.exp(-0.5 * np.square(distance / max(tolerance_samples, 1.0)))
    alignment = float(np.average(similarities, weights=weights))
    return phase, alignment


def estimate_tempo(
    features: FeatureBundle,
    candidates: list[OnsetCandidate],
    config: AnalysisConfig,
    *,
    default_offset_sample: int = 0,
) -> TempoEstimate:
    """Estimate BPM and offset from the fused envelope and onset-grid agreement."""

    if len(candidates) < 2:
        return TempoEstimate(
            bpm=120.0,
            confidence=0.0,
            beat_offset_sample=max(0, int(default_offset_sample)),
            score=0.0,
        )
    autocorrelation_bpms, autocorrelation, normalized_autocorrelation = (
        _autocorrelation_candidates(
            features.fused, features.sample_rate, features.hop_length, config
        )
    )
    coarse_bpms = autocorrelation_bpms + _interval_candidates(
        candidates, features.sample_rate, config
    )
    if not coarse_bpms:
        coarse_bpms = [120.0]
    # Deduplicate coarse centers before a fine, deterministic local search.
    unique_centers = sorted({round(bpm * 2.0) / 2.0 for bpm in coarse_bpms})
    bpm_values: set[float] = set()
    radius = 1.6
    step = config.tempo_search_step_bpm
    for center in unique_centers:
        for bpm in np.arange(center - radius, center + radius + step / 2.0, step):
            if config.tempo_min_bpm <= bpm <= config.tempo_max_bpm:
                bpm_values.add(round(float(bpm), 4))

    samples, weights = _event_weights(candidates)
    tolerance = features.sample_rate * config.tempo_alignment_tolerance_ms / 1000.0
    scored: list[dict[str, float]] = []
    for bpm in sorted(bpm_values):
        phase, alignment = _phase_and_alignment(
            bpm, samples, weights, features.sample_rate, tolerance
        )
        lag = features.sample_rate * 60.0 / (bpm * features.hop_length)
        if normalized_autocorrelation.size > 1:
            lag_positions = np.arange(normalized_autocorrelation.size)
            autocorrelation_score = float(
                np.interp(lag, lag_positions, normalized_autocorrelation)
            )
            double_lag_score = float(
                np.interp(2.0 * lag, lag_positions, normalized_autocorrelation)
            )
        else:
            autocorrelation_score = double_lag_score = 0.0
        # The lag itself dominates, while 2x-lag evidence helps prefer a musical
        # beat over a metrical subdivision when both align well.
        rhythm_score = 0.78 * autocorrelation_score + 0.22 * double_lag_score
        # A very weak center prior only resolves otherwise exact metrical ties.
        center_prior = math.exp(-0.5 * ((bpm - 128.0) / 78.0) ** 2)
        score = 0.52 * rhythm_score + 0.44 * alignment + 0.04 * center_prior
        scored.append(
            {
                "bpm": bpm,
                "phase": phase,
                "alignment": alignment,
                "autocorrelation": autocorrelation_score,
                "score": score,
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    best = scored[0]

    # Sparse material can make a subdivision (commonly 2x tempo) dominate the
    # autocorrelation even though fewer than one detected attack occurs per beat.
    # Prefer the half-tempo metrical level only when it retains substantial rhythm
    # evidence. Dense double-kick material has multiple events per beat and is not
    # affected by this ambiguity rule.
    duration_sec = max(
        1e-6,
        float(features.frame_samples[-1]) / features.sample_rate
        if features.frame_samples.size
        else 0.0,
    )
    events_per_beat = len(candidates) / duration_sec * 60.0 / best["bpm"]
    if events_per_beat < 0.72 and best["bpm"] / 2.0 >= config.tempo_min_bpm:
        half_tempo = min(
            scored,
            key=lambda item: abs(item["bpm"] - best["bpm"] / 2.0),
        )
        if (
            abs(half_tempo["bpm"] - best["bpm"] / 2.0) <= 1.0
            and half_tempo["score"] >= best["score"] * 0.55
        ):
            best = half_tempo

    # Refine BPM around the grid optimum using the event alignment itself. This
    # gives sub-autocorrelation-bin accuracy over a 30-second track.
    fine_bpms = np.arange(
        best["bpm"] - 0.18,
        best["bpm"] + 0.18 + step / 10.0,
        max(0.01, step / 5.0),
    )
    for bpm in fine_bpms:
        if not config.tempo_min_bpm <= bpm <= config.tempo_max_bpm:
            continue
        phase, alignment = _phase_and_alignment(
            float(bpm), samples, weights, features.sample_rate, tolerance
        )
        if alignment > best["alignment"] + 1e-6:
            best = dict(best, bpm=float(bpm), phase=phase, alignment=alignment)

    runner_up = next(
        (
            item
            for item in scored[1:]
            if abs(item["bpm"] - best["bpm"]) > max(1.0, best["bpm"] * 0.015)
        ),
        scored[min(1, len(scored) - 1)],
    )
    margin = max(0.0, best["score"] - runner_up["score"])
    confidence = float(
        np.clip(
            0.46 * best["alignment"]
            + 0.34 * best["autocorrelation"]
            + 0.20 * min(1.0, margin / 0.12),
            0.0,
            1.0,
        )
    )
    period = features.sample_rate * 60.0 / best["bpm"]
    phase = float(best["phase"] % period)
    # Circular phase values just below one complete period represent a beat at
    # sample zero. Canonicalizing them avoids reporting the equivalent next beat
    # as a large positive offset.
    if period - phase <= tolerance:
        phase = 0.0
    offset = int(round(phase))
    return TempoEstimate(
        bpm=round(float(best["bpm"]), 3),
        confidence=round(confidence, 6),
        beat_offset_sample=offset,
        score=float(best["score"]),
        candidates=[dict(item) for item in scored[:8]],
    )
