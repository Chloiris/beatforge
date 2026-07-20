"""Regression tests for onset detection on dense, mastered musical material.

These signals deliberately contain a continuously moving harmonic/noise floor so
that smooth amplitude modulation cannot be mistaken for a stream of attacks.
Ground-truth samples are used by this test only after production analysis returns.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from beatforge_api.audio import analyze_samples

SAMPLE_RATE = 44_100
SEED = 0xBEA7F04


def _master(values: np.ndarray, drive: float = 2.35) -> np.ndarray:
    """Apply deterministic soft clipping similar to a heavily limited master."""

    clipped = np.tanh(drive * values) / np.tanh(drive)
    return np.asarray(0.92 * clipped, dtype=np.float32)


def _complex_am_bed(duration_sec: float, *, seed: int = SEED) -> np.ndarray:
    """Create a dense distorted harmonic bed with smooth, non-onset AM motion."""

    rng = np.random.default_rng(seed)
    sample_count = round(duration_sec * SAMPLE_RATE)
    time = np.arange(sample_count, dtype=np.float64) / SAMPLE_RATE
    frequencies = (73.0, 146.0, 219.0, 438.0, 876.0, 1752.0, 3504.0, 7008.0)
    amplitudes = (0.25, 0.18, 0.14, 0.115, 0.09, 0.068, 0.050, 0.034)
    phases = rng.uniform(-np.pi, np.pi, len(frequencies))

    bed = np.zeros(sample_count, dtype=np.float64)
    for index, (frequency, amplitude, phase) in enumerate(
        zip(frequencies, amplitudes, phases, strict=True)
    ):
        # Different smooth modulators prevent the whole spectrum from behaving
        # like a synthetic click while still exercising local adaptive thresholds.
        modulation = (
            0.73
            + 0.13 * np.sin(2.0 * np.pi * (2.1 + index * 0.23) * time + phase)
            + 0.08 * np.sin(2.0 * np.pi * (5.2 + index * 0.17) * time - phase)
        )
        bed += amplitude * modulation * np.sin(
            2.0 * np.pi * frequency * time + phase
        )

    # Band-limited random texture makes this closer to a compressed production
    # than a laboratory chord. Its 18 ms smoothing cannot create a real attack.
    noise = rng.standard_normal(sample_count)
    kernel = np.hanning(801)
    kernel /= np.sum(kernel)
    texture = np.convolve(noise, kernel, mode="same")
    texture /= max(float(np.quantile(np.abs(texture), 0.995)), 1e-9)
    bed += 0.045 * texture
    return _master(bed)


def _burst(
    frequency_hz: float,
    duration_sec: float,
    *,
    amplitude: float,
    phase: float = 0.0,
) -> np.ndarray:
    sample_count = round(duration_sec * SAMPLE_RATE)
    time = np.arange(sample_count, dtype=np.float64) / SAMPLE_RATE
    attack = 1.0 - np.exp(-time / 0.00028)
    decay = np.exp(-time / max(duration_sec * 0.26, 0.008))
    envelope = attack * decay
    return np.asarray(
        amplitude * envelope * np.sin(2.0 * np.pi * frequency_hz * time + phase),
        dtype=np.float64,
    )


def _inject(audio: np.ndarray, start_sec: float, signal: np.ndarray) -> int:
    sample = round(start_sec * SAMPLE_RATE)
    end = min(audio.size, sample + signal.size)
    audio[sample:end] += signal[: end - sample]
    return sample


def _predicted_samples(audio: np.ndarray) -> tuple[list[int], list[dict[str, object]]]:
    result = analyze_samples(audio, SAMPLE_RATE, mode="balanced", sensitivity=0.5)
    points = sorted(result.hit_points, key=lambda point: int(point["sample"]))
    return [int(point["sample"]) for point in points], points


def _one_to_one_matches(
    truth: Iterable[int], predictions: Iterable[int], tolerance_ms: float = 20.0
) -> list[tuple[int, int]]:
    tolerance = round(SAMPLE_RATE * tolerance_ms / 1000.0)
    pairs = sorted(
        (
            (abs(expected - predicted), expected, predicted)
            for expected in truth
            for predicted in predictions
            if abs(expected - predicted) <= tolerance
        ),
        key=lambda pair: pair[0],
    )
    used_truth: set[int] = set()
    used_predictions: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, expected, predicted in pairs:
        if expected in used_truth or predicted in used_predictions:
            continue
        used_truth.add(expected)
        used_predictions.add(predicted)
        matches.append((expected, predicted))
    return matches


def test_balanced_rejects_smooth_am_in_a_compressed_harmonic_bed() -> None:
    duration_sec = 3.8
    audio = _complex_am_bed(duration_sec)

    predictions, _ = _predicted_samples(audio)
    interior = [
        sample
        for sample in predictions
        if 0.30 * SAMPLE_RATE <= sample <= (duration_sec - 0.30) * SAMPLE_RATE
    ]

    # Smooth gain motion is not an attack. Allow a small residual budget because
    # this is a heuristic detector, but reject the unusable picket-fence failure.
    assert len(interior) <= 4
    assert len(interior) / (duration_sec - 0.60) <= 1.25


def test_balanced_recovers_multiband_bursts_without_full_band_collapse() -> None:
    audio = np.asarray(0.58 * _complex_am_bed(3.7, seed=SEED + 1), dtype=np.float64)
    truth: list[int] = []
    expected_bands = ["low_hit", "mid_hit", "high_hit", "full_band_accent"]

    # Starting away from a zero crossing models the hard leading edge of a kick
    # recorded inside an already-limited mix, rather than an isolated sine fade.
    truth.append(
        _inject(audio, 0.62, _burst(105.0, 0.120, amplitude=1.15, phase=1.0))
    )
    truth.append(_inject(audio, 1.38, _burst(1_450.0, 0.095, amplitude=0.92)))
    truth.append(_inject(audio, 2.14, _burst(6_200.0, 0.070, amplitude=0.78)))
    broad = (
        _burst(105.0, 0.125, amplitude=0.74)
        + _burst(1_650.0, 0.125, amplitude=0.66, phase=0.31)
        + _burst(6_400.0, 0.125, amplitude=0.58, phase=-0.42)
    )
    truth.append(_inject(audio, 2.90, broad))
    mastered = _master(audio, drive=2.30)

    predictions, points = _predicted_samples(mastered)
    matches = _one_to_one_matches(truth, predictions)
    true_positives = len(matches)
    f1 = 2.0 * true_positives / (len(truth) + len(predictions))

    assert true_positives == len(truth)
    assert f1 >= 0.80

    point_by_sample = {int(point["sample"]): point for point in points}
    matched_bands = [
        str(point_by_sample[predicted]["band"])
        for expected in truth
        for matched_expected, predicted in matches
        if matched_expected == expected
    ]
    assert len(matched_bands) == len(expected_bands)
    assert matched_bands[:3] == expected_bands[:3]
    assert matched_bands.count("full_band_accent") <= 1


def test_tail_suppression_keeps_an_86ms_double_but_drops_a_35ms_echo() -> None:
    audio = np.asarray(0.32 * _complex_am_bed(2.8, seed=SEED + 2), dtype=np.float64)

    tail_start = 0.56
    tail_primary = _burst(1_180.0, 0.150, amplitude=0.98)
    tail_echo = _burst(1_180.0, 0.095, amplitude=0.37, phase=0.25)
    _inject(audio, tail_start, tail_primary)
    _inject(audio, tail_start + 0.035, tail_echo)

    double_start = 1.62
    double_truth = [
        _inject(audio, double_start, _burst(118.0, 0.075, amplitude=1.06)),
        _inject(audio, double_start + 0.086, _burst(118.0, 0.075, amplitude=1.03)),
    ]
    mastered = _master(audio, drive=2.55)

    predictions, _ = _predicted_samples(mastered)
    tail_window = [
        sample
        for sample in predictions
        if round((tail_start - 0.020) * SAMPLE_RATE)
        <= sample
        <= round((tail_start + 0.070) * SAMPLE_RATE)
    ]
    double_matches = _one_to_one_matches(double_truth, predictions, tolerance_ms=18.0)

    assert len(tail_window) == 1
    assert len(double_matches) == 2
    assert abs(double_matches[0][1] - double_matches[1][1]) >= round(
        0.060 * SAMPLE_RATE
    )
