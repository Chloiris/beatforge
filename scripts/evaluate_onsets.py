#!/usr/bin/env python3
"""Evaluate BeatForge predictions against synthesizer truth with one-to-one matches."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

WINDOWS_MS = (10, 20, 30, 50)
STRONG_TRUTH_THRESHOLD = 0.80
GRID_SUBDIVISIONS_PER_BEAT = 4


@dataclass(frozen=True)
class MatchResult:
    matches: tuple[tuple[int, int, int], ...]
    unmatched_truth: tuple[int, ...]
    unmatched_predictions: tuple[int, ...]


def one_to_one_match(truth_samples: Iterable[int], prediction_samples: Iterable[int], tolerance_samples: int) -> MatchResult:
    truth = tuple(int(value) for value in truth_samples)
    predictions = tuple(int(value) for value in prediction_samples)
    candidates = sorted(
        (
            (abs(prediction - expected), truth_index, prediction_index)
            for truth_index, expected in enumerate(truth)
            for prediction_index, prediction in enumerate(predictions)
            if abs(prediction - expected) <= tolerance_samples
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    used_truth: set[int] = set()
    used_predictions: set[int] = set()
    matches: list[tuple[int, int, int]] = []
    for distance, truth_index, prediction_index in candidates:
        if truth_index in used_truth or prediction_index in used_predictions:
            continue
        used_truth.add(truth_index)
        used_predictions.add(prediction_index)
        matches.append((truth_index, prediction_index, distance))
    return MatchResult(
        matches=tuple(sorted(matches)),
        unmatched_truth=tuple(index for index in range(len(truth)) if index not in used_truth),
        unmatched_predictions=tuple(index for index in range(len(predictions)) if index not in used_predictions),
    )


def duplicate_predictions(samples: Iterable[int], threshold_samples: int) -> int:
    ordered = sorted(int(value) for value in samples)
    if not ordered:
        return 0
    duplicates = 0
    cluster_anchor = ordered[0]
    for sample in ordered[1:]:
        if sample - cluster_anchor <= threshold_samples:
            duplicates += 1
        else:
            cluster_anchor = sample
    return duplicates


def _f_scores(true_positive: int, false_positive: int, false_negative: int) -> dict[str, float]:
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 1.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 1.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def grid_cell_accuracy(
    truth_samples: Iterable[int],
    chart_samples: Iterable[int],
    *,
    sample_rate: int,
    bpm: float,
    beat_offset_sample: int,
    subdivisions_per_beat: int = GRID_SUBDIVISIONS_PER_BEAT,
) -> dict[str, Any]:
    """Compare occupied chart cells on the known evaluation grid.

    Multiple acoustic events in one cell count as one playable chart decision. The
    Jaccard score is reported as accuracy so both missing and extra cells are visible.
    """

    cell_width = sample_rate * 60.0 / max(bpm, 1e-9) / subdivisions_per_beat

    def cells(samples: Iterable[int]) -> set[int]:
        return {
            round((int(sample) - beat_offset_sample) / cell_width)
            for sample in samples
        }

    truth_cells = cells(truth_samples)
    predicted_cells = cells(chart_samples)
    correct = truth_cells & predicted_cells
    false_positive = predicted_cells - truth_cells
    false_negative = truth_cells - predicted_cells
    union = truth_cells | predicted_cells
    return {
        "definition": "Jaccard occupancy accuracy on the expected straight 1/16 grid",
        "subdivisionsPerBeat": subdivisions_per_beat,
        "groundTruthCellCount": len(truth_cells),
        "predictedCellCount": len(predicted_cells),
        "correctCellCount": len(correct),
        "extraCellCount": len(false_positive),
        "missingCellCount": len(false_negative),
        "accuracy": round(len(correct) / len(union), 6) if union else 1.0,
        **_f_scores(len(correct), len(false_positive), len(false_negative)),
    }


def metric_block(truth: list[int], predictions: list[int], sample_rate: int, duration_sec: float, window_ms: int) -> dict[str, Any]:
    tolerance = round(sample_rate * window_ms / 1_000)
    result = one_to_one_match(truth, predictions, tolerance)
    true_positive = len(result.matches)
    false_positive = len(result.unmatched_predictions)
    false_negative = len(result.unmatched_truth)
    scores = _f_scores(true_positive, false_positive, false_negative)
    errors_ms = [distance * 1_000 / sample_rate for _, _, distance in result.matches]
    sorted_errors = sorted(errors_ms)
    p95_index = max(0, math.ceil(len(sorted_errors) * 0.95) - 1) if sorted_errors else 0
    minutes = max(duration_sec / 60.0, 1 / 60.0)
    return {
        "windowMs": window_ms,
        "truePositives": true_positive,
        "falsePositives": false_positive,
        "falseNegatives": false_negative,
        **scores,
        "medianAbsoluteTimingErrorMs": round(statistics.median(errors_ms), 6) if errors_ms else None,
        "p95AbsoluteTimingErrorMs": round(sorted_errors[p95_index], 6) if sorted_errors else None,
        "falsePositivesPerMinute": round(false_positive / minutes, 6),
        "falseNegativesPerMinute": round(false_negative / minutes, 6),
    }


def strong_metric_block(
    all_truth: list[int],
    strong_truth: list[int],
    predictions: list[int],
    sample_rate: int,
    duration_sec: float,
    window_ms: int,
) -> dict[str, Any]:
    """Measure strong-event coverage without calling valid weak hits false positives.

    All predictions are first matched against the complete scored event set. A matched
    weak event is still a real onset, so it is neutral for the strong subset. An
    unmatched prediction remains a false positive, while unmatched strong truth is a
    false negative.
    """

    tolerance = round(sample_rate * window_ms / 1_000)
    result = one_to_one_match(all_truth, predictions, tolerance)
    strong_samples = set(strong_truth)
    strong_matches = [
        match for match in result.matches if all_truth[match[0]] in strong_samples
    ]
    true_positive = len(strong_matches)
    false_positive = len(result.unmatched_predictions)
    false_negative = len(strong_truth) - true_positive
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 1.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 1.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    errors_ms = [distance * 1_000 / sample_rate for _, _, distance in strong_matches]
    sorted_errors = sorted(errors_ms)
    p95_index = max(0, math.ceil(len(sorted_errors) * 0.95) - 1) if sorted_errors else 0
    minutes = max(duration_sec / 60.0, 1 / 60.0)
    return {
        "windowMs": window_ms,
        "truePositives": true_positive,
        "falsePositives": false_positive,
        "falseNegatives": false_negative,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "medianAbsoluteTimingErrorMs": round(statistics.median(errors_ms), 6)
        if errors_ms
        else None,
        "p95AbsoluteTimingErrorMs": round(sorted_errors[p95_index], 6)
        if sorted_errors
        else None,
        "falsePositivesPerMinute": round(false_positive / minutes, 6),
        "falseNegativesPerMinute": round(false_negative / minutes, 6),
    }


def get_nested(payload: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        value: Any = payload
        for part in path:
            if not isinstance(value, dict) or part not in value:
                break
            value = value[part]
        else:
            return value
    return None


def prediction_data(payload: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    points = get_nested(payload, ("hitPoints",), ("track", "hitPoints"), ("analysis", "hitPoints")) or []
    tempo = get_nested(payload, ("tempoMap",), ("track", "tempoMap"), ("analysis", "tempoMap")) or []
    return list(points), list(tempo)


def evaluate_track(truth: dict[str, Any], prediction: dict[str, Any]) -> dict[str, Any]:
    points, tempo_map = prediction_data(prediction)
    sample_rate = int(truth["sampleRate"])
    duration_sec = float(truth["durationSec"])
    truth_samples = [int(item["sample"]) for item in truth["onsets"]]
    prediction_samples = [
        int(item.get("acousticSample", item.get("refinedSample", item["sample"])))
        for item in points
    ]
    chart_samples = [
        int(item.get("chartSample", item.get("snappedSample", item["sample"])))
        for item in points
    ]
    strong_truth = [
        int(item["sample"])
        for item in truth["onsets"]
        if float(item.get("strength", 1)) >= STRONG_TRUTH_THRESHOLD
    ]
    tempo = tempo_map[0] if tempo_map else {}
    estimated_bpm = float(tempo.get("bpm", 0) or 0)
    estimated_offset = int(tempo.get("beatOffsetSample", 0) or 0)
    return {
        "slug": truth["slug"],
        "title": truth["title"],
        "sampleRate": sample_rate,
        "durationSec": duration_sec,
        "groundTruthCount": len(truth_samples),
        "predictionCount": len(prediction_samples),
        "strongGroundTruthCount": len(strong_truth),
        "strongEvaluationCandidateCount": len(prediction_samples),
        "duplicatePredictionsWithin8Ms": duplicate_predictions(prediction_samples, round(sample_rate * 0.008)),
        "bpm": {
            "expected": truth["bpm"],
            "estimated": estimated_bpm,
            "absoluteError": round(abs(float(truth["bpm"]) - estimated_bpm), 6),
            "relativeError": round(abs(float(truth["bpm"]) - estimated_bpm) / float(truth["bpm"]), 6),
        },
        "offset": {
            "expectedSample": int(truth["beatOffsetSample"]),
            "estimatedSample": estimated_offset,
            "absoluteErrorMs": round(abs(estimated_offset - int(truth["beatOffsetSample"])) * 1_000 / sample_rate, 6),
        },
        "allEvents": {str(window): metric_block(truth_samples, prediction_samples, sample_rate, duration_sec, window) for window in WINDOWS_MS},
        "strongEvents": {
            str(window): strong_metric_block(
                truth_samples,
                strong_truth,
                prediction_samples,
                sample_rate,
                duration_sec,
                window,
            )
            for window in WINDOWS_MS
        },
        "chartEvaluation": {
            "gridCellAccuracy": grid_cell_accuracy(
                truth_samples,
                chart_samples,
                sample_rate=sample_rate,
                bpm=float(truth["bpm"]),
                beat_offset_sample=int(truth["beatOffsetSample"]),
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--demo-dir", type=Path, default=root / "storage" / "demo")
    parser.add_argument("--predictions-dir", type=Path, default=root / "storage" / "analyses")
    parser.add_argument("--output", type=Path, default=root / "reports" / "demo-evaluation.json")
    args = parser.parse_args()
    tracks = []
    missing = []
    for truth_path in sorted(args.demo_dir.glob("*.ground-truth.json")):
        slug = truth_path.name.removesuffix(".ground-truth.json")
        candidates = (
            args.predictions_dir / f"{slug}.analysis.json",
            args.predictions_dir / f"{slug}.json",
        )
        prediction_path = next((path for path in candidates if path.exists()), None)
        if prediction_path is None:
            missing.append(slug)
            continue
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        tracks.append(evaluate_track(truth, prediction))
    if missing:
        raise SystemExit(f"Missing real analysis outputs for: {', '.join(missing)}")
    if not tracks:
        raise SystemExit("No demo ground-truth and prediction pairs found")
    report = {
        "schemaVersion": "2.0",
        "evaluation": "one-to-one nearest matching; ground truth is not supplied to the detector",
        "windowsMs": list(WINDOWS_MS),
        "strongEventDefinition": {
            "groundTruthStrengthAtLeast": STRONG_TRUTH_THRESHOLD,
            "predictionSet": "all candidates; predictions matched to weaker scored onsets are neutral, unmatched predictions are false positives",
        },
        "dimensions": {
            "coverageEvaluation": {
                "status": "not_applicable",
                "reason": "Synthetic demo fixtures have no isolated vocal or lyric phrase labels.",
            },
            "eventEvaluation": {
                "status": "measured",
                "metrics": "precision/recall/F1 at ±10/±20/±30/±50 ms",
            },
            "chartEvaluation": {
                "status": "measured",
                "metrics": "straight 1/16 grid-cell occupancy accuracy",
            },
            "productEvaluation": {
                "status": "not_measured",
                "reason": "The offline fixture run has no timed human editing session.",
            },
        },
        "tracks": tracks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for track in tracks:
        print(
            f"{track['title']}: strong F1@10={track['strongEvents']['10']['f1']:.3f}, "
            f"F1@20={track['strongEvents']['20']['f1']:.3f}, BPM error={track['bpm']['relativeError']:.2%}"
        )


if __name__ == "__main__":
    main()
