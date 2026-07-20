from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from beatforge_api.audio import (
    NoopSeparator,
    OnsetCandidate,
    analyze_audio,
    analyze_samples,
    audio_from_array,
    build_waveform_lods,
    get_config,
    load_audio,
    merge_candidates,
    waveform_lods_from_samples,
)
from beatforge_api.audio.onsets import detector_family_count

ROOT = Path(__file__).resolve().parents[3]
DEMO_DIR = ROOT / "storage" / "demo"


def test_original_sample_mapping_is_exact_at_different_sample_rates() -> None:
    samples = np.zeros((48_000, 2), dtype=np.float32)
    samples[12_000] = 0.8
    audio = audio_from_array(samples, 48_000, get_config("balanced"))

    assert audio.original_sample_rate == 48_000
    assert audio.analysis_sample_rate == 44_100
    assert audio.sample_count == 48_000
    assert audio.analysis_mono.size == 44_100
    assert audio.analysis_to_original_sample(11_025) == 12_000
    assert audio.original_to_analysis_sample(12_000) == 11_025
    assert audio.analysis_to_original_sample(audio.original_to_analysis_sample(47_000)) == 47_000


def test_soundfile_decode_retains_source_metadata(tmp_path: Path) -> None:
    source = tmp_path / "stereo-48k.wav"
    samples = np.zeros((4_800, 2), dtype=np.float32)
    samples[480:, 0] = 0.1
    samples[480:, 1] = -0.05
    sf.write(source, samples, 48_000, subtype="PCM_16")

    result = load_audio(source, get_config("balanced"))

    assert result.channels == 2
    assert result.sample_count == 4_800
    assert result.duration_sec == pytest.approx(0.1)
    assert 0 <= result.leading_silence_samples <= 600


def test_waveform_lods_contain_true_min_max_and_partial_window() -> None:
    samples = np.array([-0.8, 0.3, -0.2, 0.9, 0.1], dtype=np.float32)

    levels = waveform_lods_from_samples(samples, base_window=2, max_levels=3)

    assert levels[0]["window_size"] == 2
    assert levels[0]["mins"] == pytest.approx([-0.8, -0.2, 0.1])
    assert levels[0]["maxs"] == pytest.approx([0.3, 0.9, 0.1])


def test_build_waveform_lods_decodes_real_audio(tmp_path: Path) -> None:
    source = tmp_path / "waveform.wav"
    samples = np.sin(np.linspace(0, np.pi * 8, 4096, dtype=np.float32))
    sf.write(source, samples, 44_100, subtype="FLOAT")

    levels = build_waveform_lods(source, base_window=256)

    assert levels
    assert levels[0]["level"] == 0
    assert len(levels[0]["mins"]) == 16
    assert min(levels[0]["mins"]) < -0.99
    assert max(levels[0]["maxs"]) > 0.99


def test_merge_candidates_preserves_votes_without_chain_collapsing() -> None:
    candidates = [
        OnsetCandidate(
            detected_sample=100,
            refined_sample=100,
            sample=100,
            confidence=0.8,
            salience=0.7,
            detector_votes=["mix_flux"],
            band_evidence={"low": 0.9, "mid": 0.1, "high": 0.1},
        ),
        OnsetCandidate(
            detected_sample=104,
            refined_sample=104,
            sample=104,
            confidence=0.9,
            salience=0.8,
            detector_votes=["percussive_flux", "low_band"],
            band_evidence={"low": 0.8, "mid": 0.2, "high": 0.1},
        ),
        OnsetCandidate(
            detected_sample=111,
            refined_sample=111,
            sample=111,
            confidence=0.7,
            detector_votes=["fine_flux"],
        ),
    ]

    merged = merge_candidates(candidates, merge_window_samples=8)

    assert len(merged) == 2
    assert merged[0].sample == 100
    assert merged[0].band == "low_hit"
    assert merged[0].detector_votes == ["low_band", "mix_flux", "percussive_flux"]
    assert merged[1].sample == 111


def test_correlated_band_votes_count_as_one_independent_family() -> None:
    votes = {
        "spectral_flux",
        "percussive_flux",
        "energy_attack",
        "low_band",
        "mid_band",
        "high_band",
    }

    assert detector_family_count(votes) == 4
    assert detector_family_count({"low_band", "mid_band", "high_band"}) == 1


def test_accurate_mode_falls_back_without_separator() -> None:
    sample_rate = 44_100
    samples = np.zeros(sample_rate, dtype=np.float32)
    samples[4_410 : 4_450] = np.hanning(80)[:40]

    result = analyze_samples(
        samples,
        sample_rate,
        mode="accurate",
        separator=NoopSeparator(),
    )

    assert result.metadata["mode"] == "accurate"
    assert result.metadata["effective_mode"] == "balanced"
    assert result.warnings
    assert "回退" in result.warnings[0]


@pytest.mark.parametrize(
    ("slug", "expected_bpm"),
    [
        ("neon-pulse", 128.0),
        ("iron-rift", 174.0),
        ("glass-tide", 96.0),
    ],
)
def test_real_demo_detection_and_bpm(slug: str, expected_bpm: float) -> None:
    audio_path = DEMO_DIR / f"{slug}.wav"
    truth_path = DEMO_DIR / f"{slug}.ground-truth.json"
    assert audio_path.exists(), "run scripts/generate_demo_audio.py before backend tests"
    truth = json.loads(truth_path.read_text(encoding="utf-8"))

    stages: list[tuple[str, float]] = []
    result = analyze_audio(
        audio_path,
        mode="balanced",
        progress_callback=lambda stage, progress: stages.append((stage, progress)),
    )
    predictions = sorted(int(point["sample"]) for point in result.hit_points)
    expected = sorted(int(point["sample"]) for point in truth["onsets"])
    tolerance = round(result.original_sample_rate * 0.020)
    matched = sum(
        1
        for sample in expected
        if any(abs(sample - prediction) <= tolerance for prediction in predictions)
    )

    assert result.bpm == pytest.approx(expected_bpm, rel=0.01)
    assert result.beat_offset_sample == pytest.approx(
        int(truth["beatOffsetSample"]), abs=round(result.original_sample_rate * 0.020)
    )
    assert matched / len(expected) >= 0.90
    assert all(0 <= prediction < result.sample_count for prediction in predictions)
    assert all(
        point["time_sec"] == pytest.approx(point["sample"] / result.original_sample_rate)
        for point in result.hit_points
    )
    assert all(
        later - earlier > round(result.original_sample_rate * 0.008)
        for earlier, later in zip(predictions, predictions[1:], strict=False)
    )
    assert stages[0][0] == "decoding_audio"
    assert stages[-1] == ("analysis_complete", 1.0)
    assert result.stage_timings_ms["feature_extraction"] > 0
