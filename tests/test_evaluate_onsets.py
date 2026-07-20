from scripts.evaluate_onsets import (
    duplicate_predictions,
    evaluate_track,
    grid_cell_accuracy,
    one_to_one_match,
)


def test_one_prediction_cannot_match_two_truth_points() -> None:
    result = one_to_one_match([1_000, 1_010], [1_005], tolerance_samples=10)
    assert len(result.matches) == 1
    assert len(result.unmatched_truth) == 1
    assert not result.unmatched_predictions


def test_nearest_unique_matches_are_selected() -> None:
    result = one_to_one_match([100, 200, 300], [98, 206, 600], tolerance_samples=10)
    assert [(truth_index, prediction_index) for truth_index, prediction_index, _ in result.matches] == [(0, 0), (1, 1)]
    assert result.unmatched_truth == (2,)
    assert result.unmatched_predictions == (2,)


def test_duplicate_prediction_clusters_count_extras() -> None:
    assert duplicate_predictions([100, 103, 107, 200], threshold_samples=8) == 2


def test_grid_cell_accuracy_penalizes_missing_and_extra_cells() -> None:
    metrics = grid_cell_accuracy(
        [0, 250, 500],
        [0, 500, 750],
        sample_rate=1_000,
        bpm=60,
        beat_offset_sample=0,
    )

    assert metrics["correctCellCount"] == 2
    assert metrics["missingCellCount"] == 1
    assert metrics["extraCellCount"] == 1
    assert metrics["accuracy"] == 0.5


def test_event_metrics_use_acoustic_time_and_grid_metrics_use_chart_time() -> None:
    truth = {
        "slug": "dual-time",
        "title": "Dual time",
        "sampleRate": 1_000,
        "durationSec": 1,
        "bpm": 60,
        "beatOffsetSample": 0,
        "onsets": [{"sample": 100, "strength": 1.0}],
    }
    prediction = {
        "hitPoints": [
            {
                "sample": 100,
                "acousticSample": 100,
                "chartSample": 0,
            }
        ],
        "tempoMap": [{"bpm": 60, "beatOffsetSample": 0}],
    }

    result = evaluate_track(truth, prediction)

    assert result["allEvents"]["10"]["f1"] == 1.0
    assert result["chartEvaluation"]["gridCellAccuracy"]["accuracy"] == 1.0
