import numpy as np

from scripts.evaluate_source_conditioned import (
    active_frame_centers,
    coverage_metrics,
    lyric_phrase_coverage,
    merge_nearby_samples,
)


def test_active_frame_coverage_reports_uncovered_duration() -> None:
    sample_rate = 1_000
    audio = np.concatenate([np.zeros(500), np.ones(1_000) * 0.1, np.zeros(500)])
    active, hop = active_frame_centers(audio, sample_rate)
    covered = coverage_metrics(active, [1_000], sample_rate, hop)
    uncovered = coverage_metrics(active, [], sample_rate, hop)

    assert active.size > 0
    assert covered["within250msPct"] > 0
    assert uncovered["within250msPct"] == 0
    assert uncovered["uncoveredActiveDurationAt250msSec"] > 0


def test_merge_nearby_samples_deduplicates_local_fallback_union() -> None:
    assert merge_nearby_samples([100, 115, 300, 300], 20) == [100, 300]


def test_lyric_phrase_coverage_counts_unassigned_lines_without_claiming_accuracy() -> None:
    metrics = lyric_phrase_coverage(
        {
            "chunk_diagnostics": [
                {
                    "index": 0,
                    "startSample": 0,
                    "endSample": 1_000,
                    "status": "ok",
                    "alignmentStatus": "ok",
                    "lineStart": 0,
                    "lineEnd": 2,
                }
            ]
        },
        "first\nsecond\nthird",
        [500],
    )

    assert metrics["savedLyricPhraseCount"] == 3
    assert metrics["coveredPhraseCount"] == 2
    assert metrics["uncoveredPhraseCount"] == 1
    assert metrics["humanBoundaryGroundTruthAvailable"] is False
