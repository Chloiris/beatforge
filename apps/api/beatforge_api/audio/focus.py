"""Section-level source focus for semantic, stem-aware chart candidates.

This module does not identify instruments from the mix.  It consumes time-aligned
Demucs stems and answers a narrower, inspectable question: which separated source
is foregrounded in each section?  Candidate times remain integer samples on the
44.1 kHz analysis timeline and are mapped to original samples only at serialization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .models import OnsetCandidate, StemKind

FOCUS_STEMS: tuple[StemKind, ...] = ("vocals", "drums", "bass", "other")


@dataclass(slots=True)
class FocusAnalysis:
    segments: list[dict[str, Any]]
    hop_samples: int
    activity: dict[str, np.ndarray]
    relative_energy: dict[str, np.ndarray]

    def evidence_at(self, sample: int) -> dict[str, float]:
        if not self.activity:
            return {"mix": 1.0}
        length = len(next(iter(self.activity.values())))
        index = min(max(int(round(sample / max(1, self.hop_samples))), 0), length - 1)
        evidence = {
            name: float(np.clip(self.activity[name][index], 0.0, 1.0))
            for name in FOCUS_STEMS
            if name in self.activity
        }
        total = max(sum(evidence.values()), 1e-8)
        return {name: round(value / total, 6) for name, value in evidence.items()}

    def activity_at(self, source: str, sample: int) -> float:
        values = self.activity.get(source)
        if values is None or values.size == 0:
            return 0.0
        index = min(
            max(int(round(sample / max(1, self.hop_samples))), 0),
            values.size - 1,
        )
        return float(np.clip(values[index], 0.0, 1.0))


def _window_rms(values: np.ndarray, starts: np.ndarray, window: int) -> np.ndarray:
    squared = np.square(np.asarray(values, dtype=np.float64))
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    ends = np.minimum(starts + window, values.size)
    lengths = np.maximum(1, ends - starts)
    return np.sqrt(np.maximum((cumulative[ends] - cumulative[starts]) / lengths, 0.0))


def _activity_from_rms(rms: np.ndarray) -> np.ndarray:
    db = 20.0 * np.log10(np.maximum(rms, 1e-8))
    floor = float(np.quantile(db, 0.12))
    foreground = float(np.quantile(db, 0.84))
    span = max(8.0, foreground - floor)
    relative_activity = np.clip(
        (db - floor - 3.0) / max(5.0, span - 3.0),
        0.0,
        1.0,
    )
    # Relative quantiles alone turn digital silence or separation residue into
    # foreground activity near a quiet outro. Keep a conservative absolute gate.
    absolute_activity = np.clip((db + 58.0) / 10.0, 0.0, 1.0)
    activity = relative_activity * absolute_activity
    return gaussian_filter1d(activity, 1.15, mode="nearest")


def _bridge_short_gaps(active: np.ndarray, maximum_gap_frames: int) -> np.ndarray:
    output = np.asarray(active, dtype=bool).copy()
    false_indices = np.flatnonzero(~output)
    if not false_indices.size:
        return output
    start = 0
    while start < false_indices.size:
        end = start
        while end + 1 < false_indices.size and false_indices[end + 1] == false_indices[end] + 1:
            end += 1
        left = int(false_indices[start])
        right = int(false_indices[end])
        if (
            right - left + 1 <= maximum_gap_frames
            and left > 0
            and right + 1 < output.size
            and output[left - 1]
            and output[right + 1]
        ):
            output[left : right + 1] = True
        start = end + 1
    return output


def _replace_short_runs(labels: list[StemKind], minimum_frames: int) -> list[StemKind]:
    output = list(labels)
    if len(output) < 2:
        return output
    changed = True
    while changed:
        changed = False
        start = 0
        while start < len(output):
            end = start + 1
            while end < len(output) and output[end] == output[start]:
                end += 1
            if end - start < minimum_frames:
                left = output[start - 1] if start > 0 else None
                right = output[end] if end < len(output) else None
                replacement = left if left == right and left is not None else left or right
                if replacement is not None and replacement != output[start]:
                    output[start:end] = [replacement] * (end - start)
                    changed = True
            start = end
    return output


def build_focus_analysis(
    stems: dict[str, np.ndarray],
    sample_rate: int,
    *,
    duration_samples: int | None = None,
    hop_sec: float = 0.25,
    window_sec: float = 0.75,
) -> FocusAnalysis:
    """Choose vocals, melodic ``other``, drums, bass, or mix for every section.

    Vocals deliberately receive priority while they are genuinely foregrounded.
    With vocals absent, ``other`` is treated as the melodic-lead proxy (piano,
    guitar, synth).  Drums win only when they dominate and melodic energy recedes,
    which is the observable signature of a drum solo/break.
    """

    available = {
        name: np.asarray(stems[name], dtype=np.float32)
        for name in FOCUS_STEMS
        if name in stems and np.asarray(stems[name]).size
    }
    if not available:
        length = max(1, int(duration_samples or 1))
        return FocusAnalysis(
            segments=[
                {
                    "start_sample": 0,
                    "end_sample": length,
                    "focus_source": "mix",
                    "confidence": 0.0,
                    "reason": "mixed",
                    "evidence": {"mix": 1.0},
                }
            ],
            hop_samples=max(1, round(sample_rate * hop_sec)),
            activity={},
            relative_energy={},
        )

    length = int(duration_samples or max(values.size for values in available.values()))
    hop = max(1, int(round(sample_rate * hop_sec)))
    window = max(hop, int(round(sample_rate * window_sec)))
    starts = np.arange(0, max(1, length), hop, dtype=np.int64)
    rms: dict[str, np.ndarray] = {}
    activity: dict[str, np.ndarray] = {}
    for name in FOCUS_STEMS:
        values = available.get(name)
        if values is None:
            values = np.zeros(length, dtype=np.float32)
        elif values.size < length:
            values = np.pad(values, (0, length - values.size))
        else:
            values = values[:length]
        rms[name] = _window_rms(values, starts, window)
        activity[name] = _activity_from_rms(rms[name])

    energy_stack = np.stack([np.square(rms[name]) for name in FOCUS_STEMS], axis=0)
    energy_total = np.maximum(np.sum(energy_stack, axis=0), 1e-12)
    relative = {
        name: energy_stack[index] / energy_total
        for index, name in enumerate(FOCUS_STEMS)
    }

    vocal_active = (activity["vocals"] >= 0.24) & (relative["vocals"] >= 0.075)
    vocal_active = _bridge_short_gaps(vocal_active, max(1, round(1.25 / hop_sec)))

    labels: list[StemKind] = []
    reasons: list[str] = []
    confidences: list[float] = []
    for index in range(starts.size):
        if vocal_active[index]:
            label: StemKind = "vocals"
            reason = "vocal_presence"
            confidence = 0.48 + 0.34 * activity["vocals"][index] + 0.18 * relative["vocals"][index]
        else:
            drum_dominance = relative["drums"][index]
            melodic_presence = max(activity["other"][index], relative["other"][index])
            drum_solo = (
                drum_dominance >= 0.42
                and activity["drums"][index] >= 0.30
                and relative["other"][index] <= 0.30
            )
            if drum_solo:
                label = "drums"
                reason = "drum_solo"
                confidence = 0.45 + 0.42 * drum_dominance
            elif melodic_presence >= 0.16:
                label = "other"
                reason = "melodic_lead"
                confidence = (
                    0.40
                    + 0.36 * activity["other"][index]
                    + 0.18 * relative["other"][index]
                )
            elif activity["drums"][index] >= 0.18:
                label = "drums"
                reason = "drum_solo"
                confidence = 0.36 + 0.38 * activity["drums"][index]
            elif activity["bass"][index] >= 0.25:
                label = "bass"
                reason = "melodic_lead"
                confidence = 0.34 + 0.34 * activity["bass"][index]
            else:
                label = "mix"
                reason = "mixed"
                confidence = 0.25
        labels.append(label)
        reasons.append(reason)
        confidences.append(float(np.clip(confidence, 0.0, 1.0)))

    labels = _replace_short_runs(labels, max(2, round(0.75 / hop_sec)))
    # Recompute the human-readable reason after label smoothing.
    reason_by_label = {
        "vocals": "vocal_presence",
        "drums": "drum_solo",
        "bass": "melodic_lead",
        "other": "melodic_lead",
        "mix": "mixed",
    }

    segments: list[dict[str, Any]] = []
    start_frame = 0
    while start_frame < len(labels):
        end_frame = start_frame + 1
        while end_frame < len(labels) and labels[end_frame] == labels[start_frame]:
            end_frame += 1
        start_sample = int(starts[start_frame])
        end_sample = min(length, int(starts[end_frame]) if end_frame < starts.size else length)
        evidence = {
            name: round(float(np.mean(activity[name][start_frame:end_frame])), 6)
            for name in FOCUS_STEMS
        }
        routing_scores = {
            name: float(
                np.mean(
                    0.55 * activity[name][start_frame:end_frame]
                    + 0.45 * relative[name][start_frame:end_frame]
                )
            )
            for name in FOCUS_STEMS
        }
        alternatives = [
            {"source": name, "score": round(score, 6)}
            for name, score in sorted(
                routing_scores.items(), key=lambda item: item[1], reverse=True
            )
            if name != labels[start_frame]
        ][:2]
        segments.append(
            {
                "start_sample": start_sample,
                "end_sample": max(start_sample + 1, end_sample),
                "focus_source": labels[start_frame],
                "confidence": round(float(np.mean(confidences[start_frame:end_frame])), 6),
                "reason": reason_by_label[labels[start_frame]],
                "evidence": evidence,
                "alternatives": alternatives,
            }
        )
        start_frame = end_frame

    return FocusAnalysis(
        segments=segments,
        hop_samples=hop,
        activity=activity,
        relative_energy=relative,
    )


def _quality(candidate: OnsetCandidate) -> float:
    return float(candidate.confidence + 0.34 * candidate.salience + 0.08 * candidate.prominence)


def _suppress_nearby(
    candidates: list[OnsetCandidate], minimum_samples: int
) -> list[OnsetCandidate]:
    if not candidates:
        return []
    output: list[OnsetCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.sample):
        if not output or candidate.sample - output[-1].sample >= minimum_samples:
            output.append(candidate)
        elif _quality(candidate) > _quality(output[-1]):
            output[-1] = candidate
    return output


def select_focus_candidates(
    candidates_by_stem: dict[str, list[OnsetCandidate]],
    focus: FocusAnalysis,
    sample_rate: int,
) -> tuple[list[OnsetCandidate], dict[str, Any]]:
    """Retain candidates from every lane and treat focus only as soft evidence."""

    minimum_interval_ms = {
        "vocals": 72.0,
        "drums": 34.0,
        "bass": 58.0,
        "other": 56.0,
        "mix": 48.0,
    }
    confidence_floor = {"vocals": 0.24, "drums": 0.22, "bass": 0.30, "other": 0.26, "mix": 0.40}
    selected: list[OnsetCandidate] = []
    counts: dict[str, int] = {name: 0 for name in (*FOCUS_STEMS, "mix")}
    for source, source_candidates in candidates_by_stem.items():
        if source not in {*FOCUS_STEMS, "mix"}:
            continue
        source_candidates = candidates_by_stem.get(source, [])
        within = [
            candidate
            for candidate in source_candidates
            if (source == "mix" or focus.activity_at(source, candidate.sample) >= 0.035)
            and (
                candidate.confidence >= confidence_floor.get(source, 0.4)
                or candidate.salience >= min(0.92, confidence_floor.get(source, 0.4) + 0.12)
            )
        ]
        within = _suppress_nearby(
            within,
            max(1, round(sample_rate * minimum_interval_ms.get(source, 50.0) / 1000.0)),
        )
        for candidate in within:
            candidate.primary_stem = source if source in FOCUS_STEMS else "mix"  # type: ignore[assignment]
            candidate.stem_evidence = focus.evidence_at(candidate.sample)
            candidate.source = "stems" if source != "mix" else candidate.source
            candidate.detector_votes = [
                f"{source}:{vote}" for vote in candidate.detector_votes
            ] + [f"soft_route_{source}"]
        selected.extend(within)
        counts[source] = counts.get(source, 0) + len(within)

    return selected, {
        "strategy": "soft_source_routing",
        "detectedByStem": {
            name: len(items) for name, items in candidates_by_stem.items()
        },
        "retainedByStem": counts,
        "selectedByStem": counts,
        "selected": len(selected),
        "rhythmRescued": 0,
    }
