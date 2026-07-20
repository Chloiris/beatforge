"""Configuration for BeatForge's CPU onset analysis pipeline.

Every detector threshold and fusion weight lives here so that analysis results are
reproducible and no sensitivity-affecting constants are hidden in the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

AnalysisMode = Literal["recall", "balanced", "clean", "accurate"]


@dataclass(frozen=True)
class FeatureWeights:
    mix_flux: float = 1.00
    percussive_flux: float = 1.15
    low_band: float = 0.92
    low_mid_band: float = 0.72
    mid_band: float = 0.75
    high_band: float = 0.82
    air_band: float = 0.58
    energy_derivative: float = 0.68
    rms_derivative: float = 0.62
    hfc_change: float = 0.66
    fine_flux: float = 0.82


@dataclass(frozen=True)
class AnalysisConfig:
    mode: AnalysisMode
    analysis_sample_rate: int = 44_100
    coarse_hop_length: int = 128
    fine_hop_length: int = 64
    fft_sizes: tuple[int, ...] = (1024, 2048, 4096)
    fine_fft_size: int = 512
    band_edges_hz: tuple[tuple[str, float, float | None], ...] = (
        ("sub_low", 20.0, 180.0),
        ("low_mid", 180.0, 800.0),
        ("mid", 800.0, 3000.0),
        ("high", 3000.0, 8000.0),
        ("air", 8000.0, None),
    )
    robust_window_sec: float = 1.50
    local_max_window_sec: float = 0.080
    minimum_interval_ms: float = 16.0
    merge_window_ms: float = 9.0
    peak_height: float = 0.44
    peak_prominence: float = 0.150
    individual_vote_height: float = 1.45
    family_vote_prominence: float = 0.34
    minimum_family_votes: int = 2
    strong_peak_override_ratio: float = 1.55
    individual_support_height_ratio: float = 1.0
    individual_support_prominence_ratio: float = 1.0
    refinement_pre_ms: float = 24.0
    refinement_post_ms: float = 18.0
    refinement_smoothing_ms: float = 0.35
    refinement_max_backtrack_ms: float = 20.0
    maximum_refinement_shift_ms: float = 8.0
    onset_backtrack_ratio: float = 0.18
    energy_comparison_window_ms: float = 12.0
    energy_comparison_gap_ms: float = 1.0
    minimum_attack_energy_ratio: float = 1.08
    tail_suppression_ms: float = 52.0
    tail_strength_ratio: float = 0.62
    tail_band_similarity: float = 0.82
    tail_valley_reset_ratio: float = 0.42
    minimum_candidate_confidence: float = 0.34
    chart_anchor_family_votes: int = 4
    chart_anchor_confidence: float = 0.58
    chart_anchor_salience: float = 0.62
    rhythmic_rescue_family_votes: int = 3
    rhythmic_rescue_confidence: float = 0.34
    rhythmic_rescue_tolerance_ms: float = 3.0
    local_rescue_window_sec: float = 0.72
    local_rescue_confidence: float = 0.48
    leading_silence_db: float = -52.0
    hpss_kernel_time: int = 31
    hpss_kernel_frequency: int = 31
    hpss_margin_harmonic: float = 1.0
    hpss_margin_percussive: float = 1.5
    tempo_min_bpm: float = 55.0
    tempo_max_bpm: float = 220.0
    tempo_search_step_bpm: float = 0.10
    tempo_alignment_tolerance_ms: float = 32.0
    snap_subdivisions_per_beat: int = 4
    confidence_bias: float = 0.0
    salience_density_window_sec: float = 0.30
    waveform_base_window: int = 256
    waveform_max_levels: int = 8
    feature_weights: FeatureWeights = field(default_factory=FeatureWeights)


PRESETS: dict[AnalysisMode, AnalysisConfig] = {
    "recall": AnalysisConfig(
        mode="recall",
        minimum_interval_ms=12.0,
        merge_window_ms=8.0,
        peak_height=0.34,
        peak_prominence=0.085,
        individual_vote_height=1.18,
        family_vote_prominence=0.24,
        minimum_family_votes=1,
        strong_peak_override_ratio=1.35,
        individual_support_height_ratio=0.72,
        individual_support_prominence_ratio=0.65,
        tail_strength_ratio=0.72,
        minimum_attack_energy_ratio=0.92,
        minimum_candidate_confidence=0.20,
        chart_anchor_family_votes=3,
        chart_anchor_confidence=0.42,
        chart_anchor_salience=0.46,
        rhythmic_rescue_family_votes=2,
        rhythmic_rescue_confidence=0.24,
        rhythmic_rescue_tolerance_ms=7.0,
        local_rescue_confidence=0.32,
        confidence_bias=-0.05,
    ),
    "balanced": AnalysisConfig(
        mode="balanced",
        peak_height=0.49,
        peak_prominence=0.165,
        individual_vote_height=1.35,
        individual_support_height_ratio=0.72,
        individual_support_prominence_ratio=0.65,
        tail_strength_ratio=0.82,
        minimum_attack_energy_ratio=0.90,
    ),
    "clean": AnalysisConfig(
        mode="clean",
        minimum_interval_ms=24.0,
        merge_window_ms=11.0,
        peak_height=0.60,
        peak_prominence=0.20,
        individual_vote_height=1.95,
        family_vote_prominence=0.42,
        minimum_family_votes=3,
        strong_peak_override_ratio=1.75,
        tail_suppression_ms=68.0,
        tail_strength_ratio=0.88,
        minimum_attack_energy_ratio=1.20,
        minimum_candidate_confidence=0.48,
        chart_anchor_family_votes=4,
        chart_anchor_confidence=0.62,
        chart_anchor_salience=0.66,
        rhythmic_rescue_family_votes=4,
        rhythmic_rescue_confidence=0.46,
        rhythmic_rescue_tolerance_ms=3.0,
        local_rescue_confidence=0.52,
        confidence_bias=0.08,
    ),
    # Accurate mode uses these settings only when a separator is available. The
    # pipeline intentionally falls back to the balanced preset otherwise.
    "accurate": AnalysisConfig(
        mode="accurate",
        minimum_interval_ms=14.0,
        merge_window_ms=8.0,
        peak_height=0.43,
        peak_prominence=0.13,
        individual_vote_height=1.42,
        family_vote_prominence=0.28,
        minimum_family_votes=2,
        strong_peak_override_ratio=1.42,
        individual_support_height_ratio=0.72,
        individual_support_prominence_ratio=0.65,
        tail_strength_ratio=0.76,
        minimum_attack_energy_ratio=1.02,
        minimum_candidate_confidence=0.28,
        chart_anchor_family_votes=4,
        chart_anchor_confidence=0.48,
        chart_anchor_salience=0.52,
        rhythmic_rescue_family_votes=2,
        rhythmic_rescue_confidence=0.30,
        rhythmic_rescue_tolerance_ms=5.5,
        local_rescue_confidence=0.38,
    ),
}


def get_config(mode: AnalysisMode = "balanced", sensitivity: float = 0.5) -> AnalysisConfig:
    """Return an immutable preset adjusted by a normalized UI sensitivity.

    A higher sensitivity lowers both fused and individual detector thresholds.
    It never changes the time mapping or minimum event spacing.
    """

    if mode not in PRESETS:
        raise ValueError(f"Unknown analysis mode: {mode}")
    if not 0.0 <= sensitivity <= 1.0:
        raise ValueError("Sensitivity must be between 0 and 1")
    base = PRESETS[mode]
    delta = sensitivity - 0.5
    return replace(
        base,
        peak_height=max(0.08, base.peak_height * (1.0 - 0.45 * delta)),
        peak_prominence=max(0.015, base.peak_prominence * (1.0 - 0.35 * delta)),
        individual_vote_height=max(
            0.75, base.individual_vote_height * (1.0 - 0.50 * delta)
        ),
    )
