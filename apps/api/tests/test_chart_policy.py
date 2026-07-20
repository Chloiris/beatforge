from __future__ import annotations

from beatforge_api.audio.chart_policy import apply_chart_policy
from beatforge_api.audio.models import OnsetCandidate


def _candidate(
    sample: int,
    stem: str,
    evidence: dict[str, float],
    *,
    confidence: float = 0.8,
) -> OnsetCandidate:
    return OnsetCandidate(
        detected_sample=sample,
        refined_sample=sample,
        sample=sample,
        confidence=confidence,
        salience=confidence,
        detector_votes=["spectral_flux", "energy_attack"],
        primary_stem=stem,  # type: ignore[arg-type]
        stem_evidence=evidence,
    )


def test_policy_keeps_multi_source_evidence_but_selects_one_chart_hit_per_cell() -> None:
    vocal = _candidate(22_040, "vocals", {"vocals": 0.8, "drums": 0.2})
    drum = _candidate(
        22_060,
        "drums",
        {"vocals": 0.6, "drums": 0.4},
        confidence=0.95,
    )

    result = apply_chart_policy(
        [vocal, drum],
        sample_rate=44_100,
        bpm=120.0,
        beat_offset_sample=0,
    )

    assert len(result.candidates) == 2
    assert len(result.accepted) == 1
    assert {item.candidate.primary_stem for item in result.candidates} == {
        "vocals",
        "drums",
    }
    assert sorted(item.status for item in result.candidates) == ["accepted", "rejected"]


def test_policy_retains_strong_off_grid_acoustic_event() -> None:
    off_grid = _candidate(24_800, "vocals", {"vocals": 1.0}, confidence=1.0)

    result = apply_chart_policy(
        [off_grid],
        sample_rate=44_100,
        bpm=120.0,
        beat_offset_sample=0,
    )

    assert result.accepted == [off_grid]
    assert result.candidates[0].grid_confidence < 0.5
    assert result.candidates[0].acoustic_sample == 24_800
    assert result.candidates[0].chart_sample != 24_800


def test_policy_never_creates_an_event_from_an_empty_grid() -> None:
    result = apply_chart_policy(
        [],
        sample_rate=44_100,
        bpm=120.0,
        beat_offset_sample=0,
    )

    assert result.accepted == []
    assert result.candidates == []


def test_difficulty_changes_selection_density_without_creating_candidates() -> None:
    borderline = _candidate(22_050, "vocals", {"vocals": 0.4}, confidence=0.4)

    easy = apply_chart_policy(
        [borderline],
        sample_rate=44_100,
        bpm=120.0,
        beat_offset_sample=0,
        difficulty_level=0.0,
    )
    hard = apply_chart_policy(
        [borderline],
        sample_rate=44_100,
        bpm=120.0,
        beat_offset_sample=0,
        difficulty_level=1.0,
    )

    assert easy.accepted == []
    assert hard.accepted == [borderline]
    assert len(easy.candidates) == len(hard.candidates) == 1
