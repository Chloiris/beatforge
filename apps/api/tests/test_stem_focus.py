from __future__ import annotations

import numpy as np

from beatforge_api.audio.focus import build_focus_analysis, select_focus_candidates
from beatforge_api.audio.models import OnsetCandidate


def _section_fixture(sample_rate: int = 1_000) -> dict[str, np.ndarray]:
    length = 24 * sample_rate
    time = np.arange(length, dtype=np.float64) / sample_rate
    stems = {
        name: np.zeros(length, dtype=np.float32)
        for name in ("vocals", "drums", "bass", "other")
    }

    # 0–8s: melodic/piano proxy in Demucs' `other` stem.
    melodic = time < 8.0
    stems["other"][melodic] = (
        0.28
        * np.sin(2.0 * np.pi * 220.0 * time[melodic])
        * (0.55 + 0.45 * (np.sin(2.0 * np.pi * 2.0 * time[melodic]) > 0))
    )

    # 8–16s: foreground voice with a quieter drum accompaniment.
    vocal = (time >= 8.0) & (time < 16.0)
    stems["vocals"][vocal] = (
        0.38
        * np.sin(2.0 * np.pi * 180.0 * time[vocal])
        * (0.55 + 0.45 * np.maximum(0.0, np.sin(2.0 * np.pi * 3.3 * time[vocal])))
    )
    for onset_sec in np.arange(8.0, 16.0, 0.5):
        start = round(onset_sec * sample_rate)
        stems["drums"][start : start + 80] += 0.18 * np.exp(-np.arange(80) / 20.0)

    # 16–24s: drum solo with no vocal or melodic foreground.
    for onset_sec in np.arange(16.0, 24.0, 0.25):
        start = round(onset_sec * sample_rate)
        stems["drums"][start : start + 100] += 0.55 * np.exp(-np.arange(100) / 22.0)
    return stems


def _focus_at(segments: list[dict], sample: int) -> str:
    return next(
        str(segment["focus_source"])
        for segment in segments
        if int(segment["start_sample"]) <= sample < int(segment["end_sample"])
    )


def _candidate(sample: int, confidence: float = 0.72) -> OnsetCandidate:
    return OnsetCandidate(
        detected_sample=sample,
        refined_sample=sample,
        sample=sample,
        confidence=confidence,
        salience=0.7,
        detector_votes=["spectral_flux", "energy_attack"],
        peak_value=0.8,
        prominence=0.5,
    )


def test_focus_map_switches_melody_to_vocals_to_drum_solo() -> None:
    sample_rate = 1_000
    focus = build_focus_analysis(
        _section_fixture(sample_rate),
        sample_rate,
        duration_samples=24 * sample_rate,
    )

    assert _focus_at(focus.segments, 4 * sample_rate) == "other"
    assert _focus_at(focus.segments, 12 * sample_rate) == "vocals"
    assert _focus_at(focus.segments, 20 * sample_rate) == "drums"
    boundaries = [int(segment["start_sample"]) for segment in focus.segments[1:]]
    assert min(abs(boundary - 8 * sample_rate) for boundary in boundaries) <= 750
    assert min(abs(boundary - 16 * sample_rate) for boundary in boundaries) <= 750
    assert all("alternatives" in segment for segment in focus.segments)


def test_soft_routing_keeps_vocal_and_drum_candidates_in_the_same_section() -> None:
    sample_rate = 1_000
    focus = build_focus_analysis(
        _section_fixture(sample_rate),
        sample_rate,
        duration_samples=24 * sample_rate,
    )
    candidates = {
        "mix": [],
        "other": [_candidate(4_000), _candidate(12_000), _candidate(20_000)],
        "vocals": [_candidate(4_010), _candidate(12_010), _candidate(20_010)],
        "drums": [_candidate(4_020), _candidate(12_020), _candidate(20_020)],
        "bass": [],
    }

    selected, metadata = select_focus_candidates(candidates, focus, sample_rate)

    assert [(item.sample // 1_000, item.primary_stem) for item in selected] == [
        (4, "other"),
        (12, "vocals"),
        (12, "drums"),
        (20, "drums"),
    ]
    assert metadata["strategy"] == "soft_source_routing"
    assert metadata["rhythmRescued"] == 0
    assert all(item.source == "stems" for item in selected)
    assert all(item.stem_evidence for item in selected)


def test_absolute_activity_gate_rejects_outro_residue() -> None:
    sample_rate = 1_000
    length = 4 * sample_rate
    stems = {
        name: np.zeros(length, dtype=np.float32)
        for name in ("vocals", "drums", "bass", "other")
    }
    rng = np.random.default_rng(20260718)
    stems["other"] = rng.normal(0.0, 1e-5, length).astype(np.float32)
    focus = build_focus_analysis(stems, sample_rate, duration_samples=length)
    # Simulate a smoothed segment label extending into the residue. Candidate
    # acceptance must still consult absolute per-frame activity.
    focus.segments = [{
        "start_sample": 0,
        "end_sample": length,
        "focus_source": "other",
        "confidence": 0.8,
    }]

    selected, _metadata = select_focus_candidates(
        {"other": [_candidate(2_000)]},
        focus,
        sample_rate,
    )

    assert selected == []
