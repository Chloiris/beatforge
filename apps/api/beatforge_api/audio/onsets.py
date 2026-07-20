"""Candidate peak detection, sample-level refinement, merging, and classification."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks

from .config import AnalysisConfig
from .models import FeatureBundle, OnsetCandidate

DETECTOR_FAMILIES: dict[str, tuple[str, ...]] = {
    "spectral_flux": ("mix_flux", "fine_flux"),
    "percussive_flux": ("percussive_flux",),
    "energy_attack": ("energy_derivative", "rms_derivative"),
    "low_band": ("sub_low", "low_mid"),
    "mid_band": ("mid",),
    "high_band": ("high", "air", "hfc_change"),
}


def detector_family_count(votes: set[str] | list[str]) -> int:
    """Count statistically independent evidence groups, not correlated bands."""

    vote_set = set(votes)
    groups: set[str] = set()
    if "spectral_flux" in vote_set:
        groups.add("spectral")
    if "percussive_flux" in vote_set:
        groups.add("percussive")
    if "energy_attack" in vote_set:
        groups.add("waveform")
    if vote_set & {"low_band", "mid_band", "high_band"}:
        groups.add("band_flux")
    if any(name.startswith("stem_") for name in vote_set):
        groups.add("stems")
    return len(groups)


def _bounded_sigmoid(value: float, center: float = 1.0, slope: float = 1.0) -> float:
    exponent = float(np.clip(-slope * (value - center), -20.0, 20.0))
    return float(1.0 / (1.0 + np.exp(exponent)))


def _event_band_evidence(features: FeatureBundle, frame_index: int) -> dict[str, float]:
    start = max(0, frame_index - 1)
    end = min(features.fused.size, frame_index + 3)

    band_names = ("sub_low", "low_mid", "mid", "high", "air")
    local_band_flux = np.stack(
        [features.band_energy[name][start:end] for name in band_names], axis=0
    )
    # Use one common attack frame. Taking an independent maximum from each band
    # joined unrelated changes spread across ~12 ms into a fictitious broadband
    # accent on mastered material.
    common_offset = (
        int(np.argmax(np.sum(local_band_flux, axis=0)))
        if local_band_flux.shape[1]
        else 0
    )
    attack_frame = min(max(start + common_offset, 0), features.fused.size - 1)

    def normalized_evidence(name: str) -> float:
        curve = features.normalized_curves[name]
        value = float(curve[attack_frame]) if curve.size else 0.0
        return value / (value + 1.75)

    def raw_evidence(name: str) -> float:
        curve = features.band_energy[name]
        return float(curve[attack_frame]) if curve.size else 0.0

    sub_low_raw = raw_evidence("sub_low")
    low_mid_raw = raw_evidence("low_mid")
    mid_raw = raw_evidence("mid")
    high_band_raw = raw_evidence("high")
    air_raw = raw_evidence("air")
    low_raw = 0.58 * sub_low_raw + 0.42 * low_mid_raw
    high_raw = 0.68 * high_band_raw + 0.32 * air_raw
    raw_values = np.array([low_raw, mid_raw, high_raw], dtype=np.float64)
    raw_shares = raw_values / max(float(np.sum(raw_values)), 1e-10)

    activations = np.array(
        [
            max(normalized_evidence("sub_low"), normalized_evidence("low_mid")),
            normalized_evidence("mid"),
            max(normalized_evidence("high"), normalized_evidence("air")),
        ],
        dtype=np.float64,
    )
    activation_shares = activations / max(float(np.sum(activations)), 1e-10)
    scores = 0.72 * raw_shares + 0.28 * activation_shares
    scores /= max(float(np.sum(scores)), 1e-10)
    entropy = -float(
        np.sum(raw_shares * np.log(np.maximum(raw_shares, 1e-10))) / np.log(3.0)
    )
    return {
        "low": float(scores[0]),
        "mid": float(scores[1]),
        "high": float(scores[2]),
        "low_activation": float(activations[0]),
        "mid_activation": float(activations[1]),
        "high_activation": float(activations[2]),
        "synchronization": float(np.min(activations)),
        "broadband_strength": float(np.mean(activations)),
        "spectral_entropy": entropy,
        # Non-normalized log-flux density is retained for classification. Unlike
        # per-band robust z-scores, these values have the same physical scale and
        # can be compared without turning leakage in every band into an accent.
        "sub_low_raw": sub_low_raw,
        "low_mid_raw": low_mid_raw,
        "mid_raw": mid_raw,
        "high_raw": high_band_raw,
        "air_raw": air_raw,
    }


def classify_band(evidence: dict[str, float]) -> str:
    low = max(0.0, evidence.get("low", 0.0))
    mid = max(0.0, evidence.get("mid", 0.0))
    high = max(0.0, evidence.get("high", 0.0))
    if low >= max(mid, high):
        return "low_hit"
    if high >= max(mid, low):
        return "high_hit"
    return "mid_hit"


def classify_candidates(candidates: list[OnsetCandidate]) -> None:
    """Classify attacks from comparable raw band flux and track-relative strength.

    A high resonant transient often creates large robust z-scores in every quiet
    band, while a full-band kick/snare accent can be low-frequency dominated.  A
    per-event entropy threshold therefore inverts both cases.  This classifier
    instead combines equal-bin log-flux density with robust percentiles computed
    only from the detected attacks in the current track.
    """

    if not candidates:
        return

    profiles: list[tuple[float, float, float, float, float]] = []
    for candidate in candidates:
        evidence = candidate.band_evidence
        sub_low = max(0.0, evidence.get("sub_low_raw", evidence.get("low", 0.0)))
        low_mid = max(0.0, evidence.get("low_mid_raw", 0.0))
        mid = max(0.0, evidence.get("mid_raw", evidence.get("mid", 0.0)))
        high = max(0.0, evidence.get("high_raw", evidence.get("high", 0.0)))
        air = max(0.0, evidence.get("air_raw", 0.0))
        profiles.append((sub_low, low_mid, mid, high, air))

    values = np.asarray(profiles, dtype=np.float64)
    low_strengths = values[:, 0] + 0.65 * values[:, 1]
    upper_strengths = values[:, 2] + 0.65 * values[:, 3] + 0.35 * values[:, 4]
    low_floor = float(np.quantile(low_strengths, 0.24))
    low_high = float(np.quantile(low_strengths, 0.76))
    upper_floor = float(np.quantile(upper_strengths, 0.24))
    mid_floor = float(np.quantile(values[:, 2], 0.30))
    high_strengths = 0.65 * values[:, 3] + 0.35 * values[:, 4]
    high_floor = float(np.quantile(high_strengths, 0.30))
    balanced_low_threshold = float(
        np.quantile(low_strengths, 0.82 if len(candidates) >= 12 else 0.24)
    )

    for candidate, profile, low_strength, upper_strength in zip(
        candidates, profiles, low_strengths, upper_strengths, strict=True
    ):
        sub_low, low_mid, mid, high, air = profile
        total = max(low_strength + upper_strength, 1e-10)
        upper_ratio = upper_strength / max(low_strength, 1e-10)
        low_mid_ratio = low_mid / max(sub_low, 1e-10)
        high_strength = 0.65 * high + 0.35 * air
        high_focus = high_strength / total
        low_share = low_strength / total

        broadband = (
            mid >= mid_floor
            and high_strength >= high_floor
            and mid >= high_strength * 0.38
            and (
                low_share >= 0.55
                or (low_share >= 0.34 and low_strength >= balanced_low_threshold)
            )
        )
        if broadband:
            band = "full_band_accent"
        # A low-mid spectral focus or a dominant central-band attack is a useful
        # conservative mid label even when mastering leaks energy into highs.
        elif (
            mid >= mid_floor
            and (low_mid_ratio >= 0.38 or upper_ratio >= 0.35)
            and mid >= high_strength * 1.12
        ):
            band = "mid_hit"
        # A bass-only event has both a low cross-band ratio and bottom-quartile
        # upper-band attack. Requiring both avoids turning broadband kick/snare
        # combinations into low hits merely because their kick is dominant.
        elif upper_ratio <= 0.20 or (
            upper_strength <= upper_floor * 1.08 and low_mid_ratio < 0.34
        ):
            band = "low_hit"
        # Glass/hat attacks are upper-band dominated and have no unusually strong
        # simultaneous low attack. Strong low outliers remain eligible as accents.
        elif (
            (
                upper_ratio >= 0.82
                or (high_strength >= mid * 2.0 and high_focus >= 0.28)
            )
            and low_strength < low_high
            and (high_focus >= 0.20 or mid >= mid_floor)
        ):
            band = "high_hit"
        # A low-mid concentrated attack is the deliberately broad, conservative
        # interpretation of a mid hit (snare/chug), not an instrument promise.
        elif low_mid_ratio >= 0.36 and low_strength < low_high and mid >= mid_floor:
            band = "mid_hit"
        elif low_strength >= low_floor and upper_strength >= upper_floor:
            band = "full_band_accent"
        elif high_focus >= 0.25:
            band = "high_hit"
        elif low_mid_ratio >= 0.40:
            band = "mid_hit"
        else:
            band = "low_hit"
        candidate.band = band


def _sample_novelty(
    audio: np.ndarray,
    percussive: np.ndarray,
    sample_rate: int,
    smoothing_ms: float,
) -> np.ndarray:
    smooth_size = max(3, int(round(sample_rate * smoothing_ms / 1000.0)))
    if smooth_size % 2 == 0:
        smooth_size += 1
    energy_envelope = np.sqrt(
        np.maximum(
            uniform_filter1d(
                np.square(audio, dtype=np.float32), size=smooth_size, mode="nearest"
            ),
            0.0,
        )
    )
    percussive_envelope = np.sqrt(
        np.maximum(
            uniform_filter1d(
                np.square(percussive, dtype=np.float32),
                size=smooth_size,
                mode="nearest",
            ),
            0.0,
        )
    )
    envelope_rise = np.maximum(
        np.diff(energy_envelope, prepend=energy_envelope[:1]), 0.0
    )
    percussive_rise = np.maximum(
        np.diff(percussive_envelope, prepend=percussive_envelope[:1]), 0.0
    )
    waveform_change = uniform_filter1d(
        np.abs(np.diff(audio, prepend=audio[:1])),
        size=max(3, smooth_size // 2),
        mode="nearest",
    )

    def normalize(values: np.ndarray) -> np.ndarray:
        scale = float(np.quantile(values, 0.995))
        return values / max(scale, 1e-8)

    novelty = (
        0.46 * normalize(envelope_rise)
        + 0.34 * normalize(percussive_rise)
        + 0.20 * normalize(waveform_change)
    )
    return gaussian_filter1d(novelty, max(0.55, smooth_size / 5.0), mode="nearest")


def refine_candidate_sample(
    detected_sample: int,
    novelty: np.ndarray,
    audio: np.ndarray,
    sample_rate: int,
    config: AnalysisConfig,
) -> int:
    """Find the earliest stable local attack around a coarse STFT position."""

    pre = int(round(sample_rate * config.refinement_pre_ms / 1000.0))
    post = int(round(sample_rate * config.refinement_post_ms / 1000.0))
    start = max(0, detected_sample - pre)
    end = min(audio.size, detected_sample + post + 1)
    if end - start < 3:
        return min(max(detected_sample, 0), max(0, audio.size - 1))
    local = novelty[start:end]
    local_peak_index = int(np.argmax(local))
    peak_value = float(local[local_peak_index])
    if peak_value <= 1e-10:
        return min(max(detected_sample, 0), audio.size - 1)

    # Backtrack only through the current rise. Limiting the scan prevents a quiet
    # inter-hit region from moving the marker a fixed amount before the transient.
    max_backtrack = int(
        round(sample_rate * config.refinement_max_backtrack_ms / 1000.0)
    )
    lower_bound = max(0, local_peak_index - max_backtrack)
    threshold = peak_value * config.onset_backtrack_ratio
    attack_index = local_peak_index
    for index in range(local_peak_index - 1, lower_bound - 1, -1):
        if local[index] <= threshold:
            attack_index = index + 1
            break
        attack_index = index

    # Prefer the first real waveform departure immediately around the rise; this
    # reduces phase-dependent errors for low-frequency synthetic kicks.
    departure_start = max(0, attack_index - int(sample_rate * 0.0015))
    departure_end = min(local.size, local_peak_index + 1)
    absolute = np.abs(audio[start + departure_start : start + departure_end])
    if absolute.size:
        baseline_start = max(0, start + departure_start - int(sample_rate * 0.008))
        baseline = np.abs(audio[baseline_start : start + departure_start])
        noise = float(np.median(baseline)) if baseline.size else 0.0
        peak = float(np.max(absolute))
        crossings = np.flatnonzero(absolute >= max(noise * 5.0, peak * 0.015, 1e-6))
        if crossings.size:
            attack_index = departure_start + int(crossings[0])
    return int(start + attack_index)


def merge_candidates(
    candidates: list[OnsetCandidate], merge_window_samples: int
) -> list[OnsetCandidate]:
    """Merge detector duplicates while preserving every evidence vote."""

    if not candidates:
        return []
    ordered = sorted(candidates, key=lambda candidate: candidate.refined_sample)
    groups: list[list[OnsetCandidate]] = [[ordered[0]]]
    for candidate in ordered[1:]:
        # Compare with the first anchor rather than the previous item to prevent
        # single-link chaining from collapsing a real fast double hit.
        anchor = groups[-1][0].refined_sample
        if candidate.refined_sample - anchor <= merge_window_samples:
            groups[-1].append(candidate)
        else:
            groups.append([candidate])

    merged: list[OnsetCandidate] = []
    for group in groups:
        weights = np.array(
            [max(item.confidence, item.peak_value, 0.05) for item in group],
            dtype=np.float64,
        )
        samples = np.array([item.refined_sample for item in group], dtype=np.float64)
        weighted_sample = int(round(float(np.average(samples, weights=weights))))
        # Anchor at the earliest stable attack when weighted detectors differ by
        # more than two samples; this is preferable to a late spectral centroid.
        earliest = min(item.refined_sample for item in group)
        refined = (
            earliest
            if weighted_sample - earliest <= merge_window_samples
            else weighted_sample
        )
        evidence: dict[str, float] = defaultdict(float)
        for item in group:
            for name, value in item.band_evidence.items():
                evidence[name] = max(evidence[name], float(value))
        representative = max(group, key=lambda item: (item.confidence, item.peak_value))
        votes = sorted({vote for item in group for vote in item.detector_votes})
        band_evidence = dict(evidence) or dict(representative.band_evidence)
        merged.append(
            OnsetCandidate(
                detected_sample=min(item.detected_sample for item in group),
                refined_sample=refined,
                sample=refined,
                band=classify_band(band_evidence),
                confidence=float(max(item.confidence for item in group)),
                salience=float(max(item.salience for item in group)),
                source="fused" if len(votes) > 1 else representative.source,
                detector_votes=votes,
                band_evidence=band_evidence,
                peak_value=float(max(item.peak_value for item in group)),
                prominence=float(max(item.prominence for item in group)),
                loudness=float(max(item.loudness for item in group)),
            )
        )
    return merged


def _collect_peak_frames(
    features: FeatureBundle, config: AnalysisConfig
) -> list[tuple[int, float, float, set[str]]]:
    distance = max(
        1,
        int(
            round(
                config.minimum_interval_ms
                * features.sample_rate
                / (1000.0 * features.hop_length)
            )
        ),
    )
    fused_peaks, properties = find_peaks(
        features.fused,
        height=config.peak_height,
        prominence=config.peak_prominence,
        distance=distance,
    )
    detections: list[tuple[int, float, float, str]] = []
    for peak, height, prominence in zip(
        fused_peaks,
        properties.get("peak_heights", np.zeros(fused_peaks.size)),
        properties.get("prominences", np.zeros(fused_peaks.size)), strict=False,
    ):
        detections.append((int(peak), float(height), float(prominence), "fused"))

    # Individual features may recover a quiet narrow-band attack, but must agree
    # with a local maximum in the fused novelty curve. Without this support check,
    # oscillations inside a snare/kick decay each become a fresh detector peak.
    support_peaks, support_properties = find_peaks(
        features.fused,
        height=config.peak_height * config.individual_support_height_ratio,
        prominence=(
            config.peak_prominence * config.individual_support_prominence_ratio
        ),
        distance=distance,
    )
    support_map = {
        int(peak): (
            float(height),
            float(prominence),
        )
        for peak, height, prominence in zip(
            support_peaks,
            support_properties.get("peak_heights", np.zeros(support_peaks.size)),
            support_properties.get("prominences", np.zeros(support_peaks.size)), strict=False,
        )
    }

    for family, detector_names in DETECTOR_FAMILIES.items():
        detector_curves = [features.normalized_curves[name] for name in detector_names]
        if family == "spectral_flux" and len(detector_curves) > 1:
            curve = np.mean(np.stack(detector_curves), axis=0)
        else:
            curve = np.max(np.stack(detector_curves), axis=0)
        peaks, props = find_peaks(
            curve,
            height=config.individual_vote_height,
            prominence=config.family_vote_prominence,
            distance=distance,
        )
        for peak, robust_height in zip(
            peaks,
            props.get("peak_heights", np.zeros(peaks.size)),
            strict=False,
        ):
            nearby_support = [
                support
                for support in range(max(0, int(peak) - 2), int(peak) + 3)
                if support in support_map
            ]
            if not nearby_support:
                continue
            support = max(nearby_support, key=lambda item: support_map[item][0])
            fused_height, fused_prominence = support_map[support]
            # A family peak can rescue a narrow-band attack, but its prominence
            # never enters the fused-prominence score because those values have
            # different scales.
            if (
                fused_height >= config.peak_height * 0.75
                or robust_height >= config.individual_vote_height * 1.65
            ):
                detections.append(
                    (
                        support,
                        fused_height,
                        fused_prominence,
                        family,
                    )
                )

    if not detections:
        return []
    detections.sort(key=lambda item: item[0])
    cluster_frames = max(
        1,
        int(
            round(
                config.merge_window_ms
                * features.sample_rate
                / (1000.0 * features.hop_length)
            )
        ),
    )
    groups: list[list[tuple[int, float, float, str]]] = [[detections[0]]]
    for detection in detections[1:]:
        if detection[0] - groups[-1][0][0] <= cluster_frames:
            groups[-1].append(detection)
        else:
            groups.append([detection])

    output: list[tuple[int, float, float, set[str]]] = []
    for group in groups:
        best = max(group, key=lambda item: (item[1], item[2]))
        frame = best[0]
        prominence = max(item[2] for item in group)
        votes = {item[3] for item in group if item[3] != "fused"}
        if detector_family_count(votes) < config.minimum_family_votes:
            strong_override = (
                best[1] >= config.peak_height * config.strong_peak_override_ratio
                and prominence
                >= config.peak_prominence * config.strong_peak_override_ratio
            )
            if not strong_override:
                continue
        output.append((frame, float(features.fused[frame]), prominence, votes))
    return output


def detect_onsets(
    audio: np.ndarray, features: FeatureBundle, config: AnalysisConfig
) -> list[OnsetCandidate]:
    """Detect, locally refine, merge, classify, and score onset candidates."""

    peak_frames = _collect_peak_frames(features, config)
    if not peak_frames:
        return []
    novelty = _sample_novelty(
        audio,
        features.percussive,
        features.sample_rate,
        config.refinement_smoothing_ms,
    )
    candidates: list[OnsetCandidate] = []
    energy_window = max(
        1,
        int(round(features.sample_rate * config.energy_comparison_window_ms / 1000.0)),
    )
    energy_gap = max(
        0,
        int(round(features.sample_rate * config.energy_comparison_gap_ms / 1000.0)),
    )
    for frame_index, peak_value, prominence, raw_votes in peak_frames:
        detected = int(features.frame_samples[frame_index])
        refined = refine_candidate_sample(
            detected, novelty, audio, features.sample_rate, config
        )
        maximum_shift = int(
            round(
                features.sample_rate * config.maximum_refinement_shift_ms / 1000.0
            )
        )
        if abs(refined - detected) > maximum_shift:
            refined = detected
        before = audio[
            max(0, refined - energy_gap - energy_window) : max(0, refined - energy_gap)
        ]
        after = audio[
            min(audio.size, refined + energy_gap) : min(
                audio.size, refined + energy_gap + energy_window
            )
        ]
        before_rms = float(np.sqrt(np.mean(np.square(before)))) if before.size else 0.0
        after_rms = float(np.sqrt(np.mean(np.square(after)))) if after.size else 0.0
        attack_energy_ratio = (after_rms + 1e-7) / (before_rms + 1e-7)
        # Negative/release transients (for example a resonator ending abruptly)
        # are not attack events. Recall mode uses a softer configured threshold.
        if attack_energy_ratio < config.minimum_attack_energy_ratio:
            continue
        band_evidence = _event_band_evidence(features, frame_index)
        scale_consistency = float(features.scale_votes[frame_index]) / max(
            1, len(config.fft_sizes)
        )
        independent_votes = detector_family_count(raw_votes)
        maximum_independent_votes = 4 + int(
            any(name.startswith("stem_") for name in raw_votes)
        )
        vote_consistency = min(1.0, independent_votes / maximum_independent_votes)
        local_start = max(0, frame_index - 4)
        local_end = min(features.fused.size, frame_index + 5)
        local_floor = float(np.median(features.fused[local_start:local_end]))
        contrast = max(0.0, peak_value - local_floor)
        attack_score = _bounded_sigmoid(
            attack_energy_ratio,
            max(1.0, config.minimum_attack_energy_ratio * 1.12),
            3.2,
        )
        confidence = (
            0.30 * _bounded_sigmoid(peak_value, config.peak_height, 10.0)
            + 0.24 * _bounded_sigmoid(prominence, config.peak_prominence, 14.0)
            + 0.22 * vote_consistency
            + 0.14 * scale_consistency
            + 0.10 * attack_score
            + config.confidence_bias
        )
        confidence = float(np.clip(confidence, 0.0, 1.0))
        synchronization = float(band_evidence.get("synchronization", 0.0))
        band_activation = max(
            float(band_evidence.get("low_activation", 0.0)),
            float(band_evidence.get("mid_activation", 0.0)),
            float(band_evidence.get("high_activation", 0.0)),
        )
        loudness = min(1.0, contrast / max(config.peak_height, 1e-5) * 0.8)
        salience = float(
            np.clip(
                0.36 * confidence
                + 0.28 * loudness
                # Raw band-flux density is retained for classification but is
                # deliberately excluded here: it is not normalized to 0–1 and
                # otherwise saturates salience on dense mastered material.
                + 0.24 * band_activation
                + 0.12 * synchronization,
                0.0,
                1.0,
            )
        )
        votes = sorted(raw_votes)
        if any(name.startswith("stem_") for name in raw_votes):
            source = "stems"
        elif len(votes) > 1:
            source = "fused"
        elif raw_votes == {"percussive_flux"}:
            source = "percussive"
        else:
            source = "mix"
        candidates.append(
            OnsetCandidate(
                detected_sample=detected,
                refined_sample=refined,
                sample=refined,
                band=classify_band(band_evidence),
                confidence=confidence,
                salience=salience,
                source=source,
                detector_votes=votes,
                band_evidence=band_evidence,
                peak_value=peak_value,
                prominence=prominence,
                loudness=after_rms,
            )
        )

    merge_samples = int(round(features.sample_rate * config.merge_window_ms / 1000.0))
    merged = merge_candidates(candidates, merge_samples)
    # A final spacing guard removes lower-confidence detector echoes without
    # collapsing real events outside the preset's explicit minimum interval.
    minimum = int(round(features.sample_rate * config.minimum_interval_ms / 1000.0))
    output: list[OnsetCandidate] = []
    tail_window = int(round(features.sample_rate * config.tail_suppression_ms / 1000.0))

    def evidence_similarity(left: OnsetCandidate, right: OnsetCandidate) -> float:
        names = ("low", "mid", "high")
        first = np.array([left.band_evidence.get(name, 0.0) for name in names])
        second = np.array([right.band_evidence.get(name, 0.0) for name in names])
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        return float(np.dot(first, second) / denominator) if denominator > 1e-9 else 0.0

    def valley_reset(left: OnsetCandidate, right: OnsetCandidate) -> bool:
        left_frame = min(
            features.fused.size - 1,
            max(0, int(round(left.sample / features.hop_length))),
        )
        right_frame = min(
            features.fused.size - 1,
            max(0, int(round(right.sample / features.hop_length))),
        )
        if right_frame - left_frame <= 1:
            return False
        valley = float(np.min(features.fused[left_frame + 1 : right_frame]))
        reference = min(left.peak_value, right.peak_value)
        return valley <= reference * config.tail_valley_reset_ratio

    for candidate in merged:
        if output and candidate.sample - output[-1].sample < minimum:
            if candidate.confidence > output[-1].confidence:
                output[-1] = candidate
            else:
                output[-1].detector_votes = sorted(
                    set(output[-1].detector_votes) | set(candidate.detector_votes)
                )
            continue
        if output and candidate.sample - output[-1].sample <= tail_window:
            previous = output[-1]
            weaker_tail = candidate.peak_value <= (
                previous.peak_value * config.tail_strength_ratio
            )
            similar_band = (
                evidence_similarity(previous, candidate) >= config.tail_band_similarity
            )
            distinct_attack = valley_reset(previous, candidate)
            if weaker_tail and similar_band and not distinct_attack:
                previous.detector_votes = sorted(
                    set(previous.detector_votes) | set(candidate.detector_votes)
                )
                continue
            weaker_previous = previous.peak_value <= (
                candidate.peak_value * config.tail_strength_ratio
            )
            if weaker_previous and similar_band and not distinct_attack:
                candidate.detector_votes = sorted(
                    set(previous.detector_votes) | set(candidate.detector_votes)
                )
                output[-1] = candidate
                continue
        output.append(candidate)
    if output:
        duration_sec = max(audio.size / features.sample_rate, 1e-6)
        event_rate = len(output) / duration_sec
        reference_amplitude = max(float(np.quantile(np.abs(audio), 0.995)), 1e-5)
        # Dense output must make the loudness test stricter, not easier. The old
        # inverse relation formed a positive feedback loop on compressed masters.
        threshold_ratio = float(np.clip(0.36 + 0.055 * event_rate, 0.40, 0.68))
        loudness_threshold = reference_amplitude * threshold_ratio
        transition_width = max(loudness_threshold * 0.16, 1e-4)
        for candidate in output:
            loudness_score = _bounded_sigmoid(
                candidate.loudness,
                center=loudness_threshold,
                slope=1.0 / transition_width,
            )
            candidate.confidence = float(
                np.clip(
                    candidate.confidence * (0.54 + 0.46 * loudness_score),
                    0.0,
                    1.0,
                )
            )
            candidate.salience = float(
                np.clip(0.34 * candidate.salience + 0.66 * loudness_score, 0.0, 1.0)
            )
        output = [
            candidate
            for candidate in output
            if candidate.confidence >= config.minimum_candidate_confidence
            or candidate.salience >= min(0.95, config.minimum_candidate_confidence + 0.12)
        ]
        classify_candidates(output)
    return output
