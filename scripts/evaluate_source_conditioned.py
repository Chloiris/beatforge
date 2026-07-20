#!/usr/bin/env python3
"""Measure source-conditioned coverage on a stored vocal stem without mutating a project."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

from beatforge_api.audio.vocal_candidates import extract_vocal_acoustic_candidates

EVENT_WINDOWS_MS = (10, 20, 30, 50)


def merge_nearby_samples(samples: Iterable[int], threshold_samples: int) -> list[int]:
    merged: list[int] = []
    for sample in sorted({int(value) for value in samples}):
        if not merged or sample - merged[-1] > threshold_samples:
            merged.append(sample)
    return merged


def active_frame_centers(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold_dbfs: float = -40.0,
    window_ms: float = 40.0,
    hop_ms: float = 20.0,
) -> tuple[np.ndarray, int]:
    """Return centers of absolute-threshold active frames using channel-safe RMS."""

    values = np.asarray(audio, dtype=np.float64)
    power = np.square(values)
    if power.ndim > 1:
        power = np.mean(power, axis=1)
    window = max(1, round(sample_rate * window_ms / 1_000))
    hop = max(1, round(sample_rate * hop_ms / 1_000))
    if power.size < window:
        return np.asarray([], dtype=np.int64), hop
    starts = np.arange(0, power.size - window + 1, hop, dtype=np.int64)
    cumulative = np.concatenate(([0.0], np.cumsum(power)))
    rms = np.sqrt((cumulative[starts + window] - cumulative[starts]) / window)
    dbfs = 20.0 * np.log10(np.maximum(rms, 1e-12))
    return starts[dbfs >= threshold_dbfs] + window // 2, hop


def nearest_distances(query_samples: np.ndarray, event_samples: Iterable[int]) -> np.ndarray:
    events = np.asarray(sorted({int(value) for value in event_samples}), dtype=np.int64)
    if query_samples.size == 0:
        return np.asarray([], dtype=np.float64)
    if events.size == 0:
        return np.full(query_samples.size, np.inf)
    positions = np.searchsorted(events, query_samples)
    left = events[np.clip(positions - 1, 0, events.size - 1)]
    right = events[np.clip(positions, 0, events.size - 1)]
    return np.minimum(np.abs(query_samples - left), np.abs(query_samples - right)).astype(float)


def contiguous_runs(samples: np.ndarray, hop_samples: int) -> list[tuple[int, int, int]]:
    if samples.size == 0:
        return []
    boundaries = np.flatnonzero(np.diff(samples) > hop_samples * 1.5) + 1
    groups = np.split(samples, boundaries)
    return [
        (int(group[0] - hop_samples // 2), int(group[-1] + hop_samples // 2), len(group))
        for group in groups
        if group.size
    ]


def coverage_metrics(
    active_samples: np.ndarray,
    event_samples: Iterable[int],
    sample_rate: int,
    hop_samples: int,
) -> dict[str, Any]:
    events = sorted({int(value) for value in event_samples})
    distances = nearest_distances(active_samples, events)
    within_250 = distances <= sample_rate * 0.250
    within_500 = distances <= sample_rate * 0.500
    uncovered_samples = active_samples[~within_250]
    uncovered_runs = sorted(
        contiguous_runs(uncovered_samples, hop_samples),
        key=lambda item: item[2],
        reverse=True,
    )
    active_runs = [
        run
        for run in contiguous_runs(active_samples, hop_samples)
        if run[2] * hop_samples >= sample_rate * 0.2
    ]
    covered_sections = sum(
        any(start - sample_rate * 0.25 <= event <= end + sample_rate * 0.25 for event in events)
        for start, end, _ in active_runs
    )
    finite_distances = distances[np.isfinite(distances)]
    return {
        "eventCount": len(events),
        "activeFrameCount": int(active_samples.size),
        "activeDurationSec": round(active_samples.size * hop_samples / sample_rate, 3),
        "within250msPct": round(float(np.mean(within_250)) * 100, 3)
        if active_samples.size
        else None,
        "within500msPct": round(float(np.mean(within_500)) * 100, 3)
        if active_samples.size
        else None,
        "nearestEventP95Sec": round(
            float(np.quantile(finite_distances, 0.95)) / sample_rate,
            3,
        )
        if finite_distances.size
        else None,
        "uncoveredActiveDurationAt250msSec": round(
            uncovered_samples.size * hop_samples / sample_rate,
            3,
        ),
        "activeSectionCount": len(active_runs),
        "coveredActiveSectionCount": covered_sections,
        "activeSectionCoveragePct": round(covered_sections / len(active_runs) * 100, 3)
        if active_runs
        else None,
        "longestUncoveredActiveRuns": [
            {
                "startSec": round(max(0, start) / sample_rate, 3),
                "endSec": round(end / sample_rate, 3),
                "durationSec": round(count * hop_samples / sample_rate, 3),
            }
            for start, end, count in uncovered_runs[:8]
        ],
    }


def coverage_chunk_summary(alignment: dict[str, Any], stored_events: Iterable[int]) -> dict[str, Any]:
    chunks = alignment.get("coverage_chunks") or alignment.get("coverageChunks") or []
    if chunks:
        statuses = Counter(str(chunk.get("status", "unknown")) for chunk in chunks)
        uncovered_duration = sum(
            max(0, int(chunk.get("endSample", 0)) - int(chunk.get("startSample", 0)))
            for chunk in chunks
            if chunk.get("status") != "success"
        )
        return {
            "schemaAvailable": True,
            "chunkCount": len(chunks),
            "statusCounts": dict(sorted(statuses.items())),
            "uncoveredDurationSamples": uncovered_duration,
        }

    diagnostics = alignment.get("chunk_diagnostics") or []
    events = tuple(int(value) for value in stored_events)
    regions_with_events = sum(
        any(
            int(chunk.get("startSample", 0)) <= event < int(chunk.get("endSample", 0))
            for event in events
        )
        for chunk in diagnostics
    )
    return {
        "schemaAvailable": False,
        "reason": "Stored alignment predates interval coverageChunks; legacy diagnostics are not reclassified.",
        "legacySemanticRegionCount": len(diagnostics),
        "legacyRegionsContainingStoredEvents": regions_with_events,
    }


def lyric_phrase_coverage(
    alignment: dict[str, Any],
    lyrics_text: str,
    stored_events: Iterable[int],
) -> dict[str, Any]:
    """Measure saved lyric-line assignment coverage without claiming boundary accuracy."""

    phrases = [line.strip() for line in lyrics_text.splitlines() if line.strip()]
    diagnostics = alignment.get("chunk_diagnostics") or []
    coverage_chunks = {
        int(chunk.get("index", index)): chunk
        for index, chunk in enumerate(
            alignment.get("coverage_chunks") or alignment.get("coverageChunks") or []
        )
    }
    events = tuple(int(value) for value in stored_events)
    assigned: set[int] = set()
    covered: set[int] = set()
    for fallback_index, chunk in enumerate(diagnostics):
        start_line = chunk.get("lineStart")
        end_line = chunk.get("lineEnd")
        if start_line is None or end_line is None:
            continue
        indexes = set(
            range(
                max(0, int(start_line)),
                min(len(phrases), max(int(start_line), int(end_line))),
            )
        )
        assigned.update(indexes)
        chunk_index = int(chunk.get("index", fallback_index))
        explicit_coverage = coverage_chunks.get(chunk_index)
        successful = (
            explicit_coverage.get("status") == "success"
            if explicit_coverage is not None
            else chunk.get("status") == "ok" and chunk.get("alignmentStatus") == "ok"
        )
        has_event = any(
            int(chunk.get("startSample", 0))
            <= event
            < int(chunk.get("endSample", 0))
            for event in events
        )
        if successful and has_event:
            covered.update(indexes)
    return {
        "status": "measured_proxy" if phrases else "not_available",
        "savedLyricPhraseCount": len(phrases),
        "assignedPhraseCount": len(assigned),
        "coveredPhraseCount": len(covered),
        "uncoveredPhraseCount": max(0, len(phrases) - len(covered)),
        "coveragePct": round(len(covered) / len(phrases) * 100, 3)
        if phrases
        else None,
        "humanBoundaryGroundTruthAvailable": False,
        "warning": (
            "Coverage means a saved lyric line was assigned to a successful semantic region "
            "that contains a stored vocal event; it is not phrase-boundary precision."
        ),
    }


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")}


def _unavailable_event_metrics(reason: str) -> dict[str, Any]:
    return {
        str(window): {
            "windowMs": window,
            "precision": None,
            "recall": None,
            "f1": None,
            "status": "not_measured",
            "reason": reason,
        }
        for window in EVENT_WINDOWS_MS
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=root / "storage" / "beatforge.db")
    parser.add_argument("--track-id")
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "reports" / "source-conditioned-evaluation.json",
    )
    args = parser.parse_args()

    connection = sqlite3.connect(args.database)
    connection.row_factory = sqlite3.Row
    if args.track_id:
        track = connection.execute(
            "SELECT t.*, p.title AS project_title FROM tracks t "
            "JOIN projects p ON p.id = t.project_id WHERE t.id = ?",
            (args.track_id,),
        ).fetchone()
    else:
        track = connection.execute(
            "SELECT t.*, p.title AS project_title FROM tracks t "
            "JOIN projects p ON p.id = t.project_id "
            "WHERE t.vocal_alignment_json != '{}' ORDER BY t.updated_at DESC LIMIT 1"
        ).fetchone()
    if track is None:
        raise SystemExit("No stored track with vocal alignment was found")

    track_id = str(track["id"])
    vocal_path = root / "storage" / "stems" / track_id / "vocals.flac"
    if not vocal_path.exists():
        raise SystemExit(f"Missing local vocal stem: {vocal_path}")
    audio, stem_sample_rate = sf.read(vocal_path, always_2d=False)
    sample_rate = int(track["original_sample_rate"])
    if stem_sample_rate != sample_rate:
        raise SystemExit(
            f"Vocal stem sample rate {stem_sample_rate} does not match track {sample_rate}"
        )

    hit_columns = _table_columns(connection, "hit_points")
    acoustic_expression = (
        "COALESCE(acoustic_sample, refined_sample, sample)"
        if "acoustic_sample" in hit_columns
        else "COALESCE(refined_sample, sample)"
    )
    hit_rows = connection.execute(
        f"SELECT {acoustic_expression} AS acoustic_sample, manually_edited, primary_stem "
        "FROM hit_points WHERE track_id = ?",
        (track_id,),
    ).fetchall()
    stored_vocal_events = [
        int(row["acoustic_sample"])
        for row in hit_rows
        if row["primary_stem"] == "vocals" and not row["manually_edited"]
    ]
    manual_edit_count = sum(bool(row["manually_edited"]) for row in hit_rows)
    alignment = json.loads(str(track["vocal_alignment_json"] or "{}"))

    active_samples, hop_samples = active_frame_centers(audio, sample_rate)
    acoustic_result = extract_vocal_acoustic_candidates(audio, sample_rate)
    acoustic_events = [candidate.sample for candidate in acoustic_result.candidates]
    dry_run_union = merge_nearby_samples(
        [*stored_vocal_events, *acoustic_events],
        round(sample_rate * 0.030),
    )
    before = coverage_metrics(active_samples, stored_vocal_events, sample_rate, hop_samples)
    after = coverage_metrics(active_samples, dry_run_union, sample_rate, hop_samples)
    report = {
        "schemaVersion": "2.0",
        "createdAt": datetime.now(UTC).isoformat(),
        "track": {
            "id": track_id,
            "projectId": track["project_id"],
            "title": track["project_title"],
            "originalFileName": track["original_file_name"],
            "sampleRate": sample_rate,
            "sampleCount": track["sample_count"],
            "durationSec": track["duration_sec"],
        },
        "method": {
            "vocalStem": str(vocal_path.relative_to(root)),
            "activityThresholdDbfs": -40,
            "frameWindowMs": 40,
            "frameHopMs": 20,
            "dryRunDetector": acoustic_result.method,
            "writesProjectData": False,
            "warning": (
                "Active-frame proximity is a coverage proxy, not human-labeled vocal onset "
                "precision or recall. The after scenario is a non-persisted local detector dry run."
            ),
        },
        "coverageEvaluation": {
            "lyricPhraseCoverage": lyric_phrase_coverage(
                alignment,
                str(track["lyrics_text"] or ""),
                stored_vocal_events,
            ),
            "storedCoverageChunks": coverage_chunk_summary(alignment, stored_vocal_events),
            "vocalActiveSectionCoverage": {
                "beforeStoredAutomaticVocalHits": before,
                "afterDryRunAcousticFallbackUnion": after,
                "deltaWithin250msPercentagePoints": round(
                    float(after["within250msPct"] or 0) - float(before["within250msPct"] or 0),
                    3,
                ),
                "deltaUncoveredActiveDurationSec": round(
                    float(after["uncoveredActiveDurationAt250msSec"])
                    - float(before["uncoveredActiveDurationAt250msSec"]),
                    3,
                ),
            },
        },
        "eventEvaluation": {
            "groundTruthAvailable": False,
            "windows": _unavailable_event_metrics(
                "No human vocal-onset labels exist for the current user song."
            ),
        },
        "chartEvaluation": {
            "gridCellAccuracy": None,
            "status": "not_measured",
            "reason": "No human-authored target chart exists for this user song.",
        },
        "productEvaluation": {
            "storedManualEditCount": manual_edit_count,
            "editingTimeMinutes": None,
            "status": "partially_observed",
            "reason": "The database stores edit flags but has no timed editing-session telemetry.",
        },
        "dryRunCandidateStats": {
            "detectorCandidateCount": len(acoustic_events),
            "storedAutomaticVocalHitCount": len(stored_vocal_events),
            "unionAfter30msDedupCount": len(dry_run_union),
            "minimumConfidence": round(
                min((candidate.confidence for candidate in acoustic_result.candidates), default=0.0),
                6,
            ),
            "medianConfidence": round(
                float(
                    np.median(
                        [candidate.confidence for candidate in acoustic_result.candidates]
                    )
                ),
                6,
            )
            if acoustic_result.candidates
            else None,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"{track['project_title']}: active-frame coverage within 250ms "
        f"{before['within250msPct']}% -> {after['within250msPct']}% "
        f"(dry-run candidates={len(acoustic_events)})"
    )


if __name__ == "__main__":
    main()
