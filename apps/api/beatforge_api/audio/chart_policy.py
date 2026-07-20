"""Rhythm-aware selection over source-conditioned acoustic candidates."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..timing import nearest_grid_sample
from .models import OnsetCandidate

CandidateStatus = Literal["accepted", "rejected", "uncertain"]


@dataclass(slots=True)
class ScoredCandidate:
    candidate: OnsetCandidate
    acoustic_sample: int
    chart_sample: int
    snap_error_ms: float
    source_evidence: dict[str, float]
    semantic_evidence: dict[str, float]
    confidence: float
    status: CandidateStatus
    grid_type: str
    grid_confidence: float


@dataclass(slots=True)
class ChartPolicyResult:
    accepted: list[OnsetCandidate]
    candidates: list[ScoredCandidate]


def _lane(primary_stem: str) -> str:
    return {
        "vocals": "vocals",
        "other": "melody",
        "drums": "drums",
    }.get(primary_stem, "mix")


def _source_evidence(candidate: OnsetCandidate) -> dict[str, float]:
    raw = candidate.stem_evidence
    lane = _lane(candidate.primary_stem)
    return {
        "vocals": float(np.clip(raw.get("vocals", 0.0), 0.0, 1.0)),
        "melody": float(
            np.clip(raw.get("other", raw.get("melody", 0.0)), 0.0, 1.0)
        ),
        "drums": float(np.clip(raw.get("drums", 0.0), 0.0, 1.0)),
        "mix": float(np.clip(raw.get("mix", 1.0 if lane == "mix" else 0.0), 0.0, 1.0)),
    }


def _semantic_score(evidence: dict[str, float]) -> float:
    return float(
        np.clip(
            max(
                evidence.get("lyricAlignment", 0.0),
                evidence.get("phonemeConfidence", 0.0),
                evidence.get("pitchConfidence", 0.0),
            ),
            0.0,
            1.0,
        )
    )


def apply_chart_policy(
    candidates: list[OnsetCandidate],
    *,
    sample_rate: int,
    bpm: float,
    beat_offset_sample: int,
    subdivisions_per_beat: int = 4,
    acceptance_threshold: float = 0.42,
    uncertainty_threshold: float = 0.28,
    enforce_density: bool = True,
    difficulty_level: float = 0.5,
) -> ChartPolicyResult:
    """Score existing sound events; never synthesize events from the tempo grid."""

    difficulty = float(np.clip(difficulty_level, 0.0, 1.0))
    effective_acceptance_threshold = float(
        np.clip(acceptance_threshold + (0.5 - difficulty) * 0.12, 0.0, 1.0)
    )
    scored: list[ScoredCandidate] = []
    for index, candidate in enumerate(sorted(candidates, key=lambda item: item.sample)):
        acoustic_sample = int(candidate.refined_sample)
        chart_sample = nearest_grid_sample(
            acoustic_sample,
            sample_rate=sample_rate,
            bpm=bpm,
            beat_offset_sample=beat_offset_sample,
            subdivisions_per_beat=subdivisions_per_beat,
        )
        snap_error_ms = (acoustic_sample - chart_sample) * 1000.0 / sample_rate
        grid_confidence = float(np.exp(-0.5 * (abs(snap_error_ms) / 30.0) ** 2))
        source_evidence = _source_evidence(candidate)
        semantic_evidence = {
            "lyricAlignment": float(candidate.semantic_evidence.get("lyricAlignment", 0.0)),
            "phonemeConfidence": float(
                candidate.semantic_evidence.get("phonemeConfidence", 0.0)
            ),
            "pitchConfidence": float(candidate.semantic_evidence.get("pitchConfidence", 0.0)),
            "beatConfidence": grid_confidence,
            **{
                str(name): float(value)
                for name, value in candidate.semantic_evidence.items()
                if name
                not in {
                    "lyricAlignment",
                    "phonemeConfidence",
                    "pitchConfidence",
                    "beatConfidence",
                }
            },
        }
        lane = _lane(candidate.primary_stem)
        source_score = source_evidence[lane]
        acoustic_confidence = float(
            np.clip(0.68 * candidate.confidence + 0.32 * candidate.salience, 0.0, 1.0)
        )
        score = float(
            np.clip(
                0.35 * source_score
                + 0.25 * acoustic_confidence
                + 0.20 * grid_confidence
                + 0.20 * _semantic_score(semantic_evidence),
                0.0,
                1.0,
            )
        )
        status: CandidateStatus = (
            "accepted"
            if score >= effective_acceptance_threshold
            else "uncertain"
            if score >= uncertainty_threshold
            else "rejected"
        )
        stable_key = (
            f"source-candidate:{candidate.primary_stem}:{acoustic_sample}:"
            f"{index}:{','.join(candidate.detector_votes)}"
        )
        candidate.candidate_id = str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key))
        scored.append(
            ScoredCandidate(
                candidate=candidate,
                acoustic_sample=acoustic_sample,
                chart_sample=chart_sample,
                snap_error_ms=snap_error_ms,
                source_evidence=source_evidence,
                semantic_evidence=semantic_evidence,
                confidence=score,
                status=status,
                grid_type=(
                    "straight_1_16"
                    if subdivisions_per_beat == 4
                    else f"straight_1_{subdivisions_per_beat * 4}"
                ),
                grid_confidence=grid_confidence,
            )
        )

    if not enforce_density:
        accepted_items = [item for item in scored if item.status == "accepted"]
        for item in accepted_items:
            item.candidate.sample = item.acoustic_sample
            item.candidate.refined_sample = item.acoustic_sample
            item.candidate.detector_votes = sorted(
                set(item.candidate.detector_votes) | {"chart_policy_selected"}
            )
        return ChartPolicyResult(
            accepted=[item.candidate for item in accepted_items],
            candidates=scored,
        )

    # The editable candidate layer retains every source. The final chart chooses
    # one strongest event per grid cell and keeps the rest as explicit alternatives.
    accepted_by_grid: dict[int, ScoredCandidate] = {}
    for item in scored:
        if item.status != "accepted":
            continue
        existing = accepted_by_grid.get(item.chart_sample)
        if existing is None or item.confidence > existing.confidence:
            if existing is not None:
                existing.status = "rejected"
            accepted_by_grid[item.chart_sample] = item
        else:
            item.status = "rejected"

    accepted_items = sorted(
        accepted_by_grid.values(), key=lambda item: item.acoustic_sample
    )
    minimum_spacing = max(1, round(sample_rate * (0.045 - 0.030 * difficulty)))
    final_items: list[ScoredCandidate] = []
    for item in accepted_items:
        if (
            not final_items
            or item.acoustic_sample - final_items[-1].acoustic_sample >= minimum_spacing
        ):
            final_items.append(item)
            continue
        if item.confidence > final_items[-1].confidence:
            final_items[-1].status = "rejected"
            final_items[-1] = item
        else:
            item.status = "rejected"

    accepted = [item.candidate for item in final_items]
    for item in final_items:
        item.candidate.sample = item.acoustic_sample
        item.candidate.refined_sample = item.acoustic_sample
        item.candidate.detector_votes = sorted(
            set(item.candidate.detector_votes) | {"chart_policy_selected"}
        )
    return ChartPolicyResult(accepted=accepted, candidates=scored)
