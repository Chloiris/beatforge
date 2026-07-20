"""Rhythm-constrained chart placement for source-aware candidates.

The source detectors answer whether an acoustic attack exists and retain its exact
``detected_sample``/``refined_sample``.  This module answers the separate charting
question: whether that attack is close enough to the configured rhythmic lattice,
and which integer sample should be suggested for the playable chart point.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from ..timing import nearest_grid_sample, samples_per_beat


@dataclass(frozen=True, slots=True)
class RhythmConstraintConfig:
    subdivisions_per_beat: int = 4
    minimum_tempo_confidence: float = 0.65
    maximum_error_fraction: float = 0.25
    maximum_error_ms: float = 30.0
    confidence_weight: float = 0.30
    salience_weight: float = 0.12
    alignment_weight: float = 0.45
    detector_support_weight: float = 0.08
    focus_dominance_weight: float = 0.05
    evidence_merge_window_ms: float = 9.0


def _candidate_quality(
    hit: dict[str, Any], alignment: float, config: RhythmConstraintConfig
) -> tuple[float, float, float, float, int]:
    votes = {
        str(vote).split(":", 1)[-1]
        for vote in hit.get("detector_votes", [])
        if not str(vote).startswith("focus_")
    }
    detector_support = min(1.0, len(votes) / 4.0)
    primary_stem = str(hit.get("primary_stem", "mix"))
    focus_dominance = float(dict(hit.get("stem_evidence", {})).get(primary_stem, 0.0))
    confidence = float(hit.get("confidence", 0.0))
    score = (
        config.confidence_weight * float(hit.get("confidence", 0.0))
        + config.salience_weight * float(hit.get("salience", 0.0))
        + config.alignment_weight * alignment
        + config.detector_support_weight * detector_support
        + config.focus_dominance_weight * focus_dominance
    )
    refined = int(hit.get("refined_sample", hit.get("sample", 0)))
    return score, alignment, detector_support, confidence, -refined


def constrain_hits_to_rhythm_grid(
    hits: list[dict[str, Any]],
    *,
    sample_rate: int,
    sample_count: int,
    bpm: float,
    beat_offset_sample: int,
    tempo_confidence: float,
    tempo_source: str,
    config: RhythmConstraintConfig | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach a 1/16 chart suggestion without vetoing acoustic candidates.

    No empty grid position is synthesized: every output still requires a real
    source-stem candidate. Acoustic timing remains in ``sample`` and
    ``acoustic_sample`` while ``chart_sample`` carries the grid suggestion.
    """

    active = config or RhythmConstraintConfig()
    if sample_rate <= 0 or sample_count <= 0 or bpm <= 0:
        raise ValueError("sample_rate, sample_count, and bpm must be positive")
    if active.subdivisions_per_beat <= 0:
        raise ValueError("subdivisions_per_beat must be positive")

    manual_tempo = tempo_source == "manual"
    enabled = manual_tempo or tempo_confidence >= active.minimum_tempo_confidence
    step = samples_per_beat(sample_rate, bpm) / active.subdivisions_per_beat
    tolerance_by_fraction = step * Fraction(str(active.maximum_error_fraction))
    tolerance_by_ms = Fraction(
        round(sample_rate * active.maximum_error_ms), 1000
    )
    tolerance_samples = max(1, int(min(tolerance_by_fraction, tolerance_by_ms)))
    base_stats: dict[str, Any] = {
        "applied": enabled,
        "subdivision": (
            "1/16" if active.subdivisions_per_beat == 4 else active.subdivisions_per_beat
        ),
        "subdivisions_per_beat": active.subdivisions_per_beat,
        "bpm": bpm,
        "beat_offset_sample": beat_offset_sample,
        "tempo_source": tempo_source,
        "tempo_confidence": tempo_confidence,
        "maximum_error_ms": tolerance_samples * 1000.0 / sample_rate,
        "input_count": len(hits),
        "output_count": len(hits),
        "rejected_off_grid": 0,
        "retained_off_grid": 0,
        "low_grid_confidence": 0,
        "merged_same_grid": 0,
    }
    if not enabled:
        base_stats["reason"] = "tempo_confidence_too_low"
        return [dict(hit) for hit in hits], base_stats

    by_grid_sample: dict[int, tuple[tuple[float, float, float, float, int], dict[str, Any]]] = {}
    retained_off_grid = 0
    merged = 0
    for raw_hit in hits:
        hit = dict(raw_hit)
        refined = int(hit.get("refined_sample", hit.get("sample", 0)))
        grid = nearest_grid_sample(
            refined,
            sample_rate=sample_rate,
            bpm=bpm,
            beat_offset_sample=beat_offset_sample,
            subdivisions_per_beat=active.subdivisions_per_beat,
        )
        error_samples = refined - grid
        if grid < 0 or grid >= sample_count:
            # The acoustic event remains valid; clamp only its chart suggestion.
            grid = min(max(grid, 0), sample_count - 1)
            error_samples = refined - grid
        if abs(error_samples) > tolerance_samples:
            retained_off_grid += 1
        if grid < 0 or grid >= sample_count:
            continue
        alignment = float(
            max(0.0, 1.0 - abs(error_samples) / max(tolerance_samples, 1))
        )
        hit["sample"] = refined
        hit["acoustic_sample"] = refined
        hit["chart_sample"] = grid
        hit["snapped_sample"] = grid
        hit["snap_error_ms"] = error_samples * 1000.0 / sample_rate
        votes = [str(vote) for vote in hit.get("detector_votes", [])]
        rhythm_vote = f"rhythm_1_{active.subdivisions_per_beat * 4}"
        if rhythm_vote not in votes:
            votes.append(rhythm_vote)
        hit["detector_votes"] = votes
        quality = _candidate_quality(hit, alignment, active)
        existing = by_grid_sample.get(grid)
        if existing is None:
            by_grid_sample[grid] = (quality, hit)
            continue
        merged += 1
        _, existing_hit = existing
        if quality > existing[0]:
            winner, evidence_source = hit, existing_hit
            winner_quality = quality
        else:
            winner, evidence_source = existing_hit, hit
            winner_quality = existing[0]
        winner_refined = int(winner.get("refined_sample", winner.get("sample", 0)))
        evidence_refined = int(
            evidence_source.get("refined_sample", evidence_source.get("sample", 0))
        )
        merge_window = round(sample_rate * active.evidence_merge_window_ms / 1000.0)
        if abs(winner_refined - evidence_refined) <= merge_window:
            winner["detector_votes"] = sorted(
                set(winner.get("detector_votes", []))
                | set(evidence_source.get("detector_votes", []))
            )
            winner_evidence = dict(winner.get("stem_evidence", {}))
            for name, value in dict(evidence_source.get("stem_evidence", {})).items():
                winner_evidence[name] = max(
                    float(winner_evidence.get(name, 0.0)), float(value)
                )
            winner["stem_evidence"] = winner_evidence
        by_grid_sample[grid] = (winner_quality, winner)

    constrained = [
        item[1]
        for item in sorted(by_grid_sample.values(), key=lambda item: item[1]["sample"])
    ]
    base_stats.update(
        {
            "output_count": len(constrained),
            "rejected_off_grid": 0,
            "retained_off_grid": retained_off_grid,
            "low_grid_confidence": retained_off_grid,
            "merged_same_grid": merged,
        }
    )
    return constrained, base_stats
