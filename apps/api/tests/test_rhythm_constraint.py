from __future__ import annotations

import pytest

from beatforge_api.audio.rhythm import constrain_hits_to_rhythm_grid
from beatforge_api.timing import nearest_grid_sample


def _hit(sample: int, *, confidence: float = 0.7, salience: float = 0.5) -> dict:
    return {
        "id": f"hit-{sample}",
        "sample": sample,
        "detected_sample": sample - 3,
        "refined_sample": sample,
        "snapped_sample": sample,
        "snap_error_ms": 0.0,
        "confidence": confidence,
        "salience": salience,
        "detector_votes": ["vocals:mix_flux"],
        "stem_evidence": {"vocals": 0.8},
    }


def test_rhythm_constraint_places_chart_sample_on_sixteenth_without_losing_refinement() -> None:
    sample_rate = 44_100
    bpm = 129.0
    offset = 3_576
    refined = 2_608_768
    expected = nearest_grid_sample(
        refined,
        sample_rate=sample_rate,
        bpm=bpm,
        beat_offset_sample=offset,
        subdivisions_per_beat=4,
    )

    hits, stats = constrain_hits_to_rhythm_grid(
        [_hit(refined)],
        sample_rate=sample_rate,
        sample_count=11_110_260,
        bpm=bpm,
        beat_offset_sample=offset,
        tempo_confidence=0.7,
        tempo_source="estimated",
    )

    assert stats["applied"] is True
    assert len(hits) == 1
    assert hits[0]["sample"] == refined
    assert hits[0]["acoustic_sample"] == refined
    assert hits[0]["chart_sample"] == expected
    assert hits[0]["snapped_sample"] == expected
    assert hits[0]["refined_sample"] == refined
    assert hits[0]["snap_error_ms"] == pytest.approx(
        (refined - expected) * 1000 / sample_rate
    )
    assert "rhythm_1_16" in hits[0]["detector_votes"]


def test_rhythm_constraint_rejects_far_off_grid_and_merges_one_cell() -> None:
    sample_rate = 44_100
    bpm = 120.0
    offset = 0
    grid = 22_050
    close_low_quality = _hit(grid + 200, confidence=0.4, salience=0.3)
    close_high_quality = _hit(grid - 100, confidence=0.9, salience=0.8)
    far = _hit(grid + 3_500)

    hits, stats = constrain_hits_to_rhythm_grid(
        [close_low_quality, close_high_quality, far],
        sample_rate=sample_rate,
        sample_count=100_000,
        bpm=bpm,
        beat_offset_sample=offset,
        tempo_confidence=0.8,
        tempo_source="estimated",
    )

    assert [hit["id"] for hit in hits] == [close_high_quality["id"], far["id"]]
    assert hits[0]["sample"] == grid - 100
    assert hits[0]["chart_sample"] == grid
    assert stats["rejected_off_grid"] == 0
    assert stats["retained_off_grid"] == 1
    assert stats["merged_same_grid"] == 1


def test_low_confidence_estimated_tempo_does_not_move_acoustic_samples() -> None:
    original = _hit(12_345)
    hits, stats = constrain_hits_to_rhythm_grid(
        [original],
        sample_rate=44_100,
        sample_count=100_000,
        bpm=120.0,
        beat_offset_sample=0,
        tempo_confidence=0.2,
        tempo_source="estimated",
    )
    assert stats["applied"] is False
    assert hits[0]["sample"] == original["sample"]


def test_manual_tempo_applies_even_when_previous_confidence_is_low() -> None:
    hits, stats = constrain_hits_to_rhythm_grid(
        [_hit(22_140)],
        sample_rate=44_100,
        sample_count=100_000,
        bpm=120.0,
        beat_offset_sample=0,
        tempo_confidence=0.1,
        tempo_source="manual",
    )
    assert stats["applied"] is True
    assert hits[0]["sample"] == 22_140
    assert hits[0]["chart_sample"] == 22_050


def test_same_grid_candidates_do_not_fake_detector_consensus_when_attacks_are_distinct() -> None:
    early = _hit(22_050 - 700, confidence=0.9)
    early["detector_votes"] = ["vocals:spectral_flux"]
    late = _hit(22_050 + 700, confidence=0.4)
    late["detector_votes"] = ["vocals:energy_attack"]

    hits, stats = constrain_hits_to_rhythm_grid(
        [early, late],
        sample_rate=44_100,
        sample_count=100_000,
        bpm=120.0,
        beat_offset_sample=0,
        tempo_confidence=0.8,
        tempo_source="estimated",
    )

    assert len(hits) == 1
    assert stats["merged_same_grid"] == 1
    assert "vocals:spectral_flux" in hits[0]["detector_votes"]
    assert "vocals:energy_attack" not in hits[0]["detector_votes"]
