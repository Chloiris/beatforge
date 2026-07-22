from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytest.importorskip("torch", reason="chart sequence-model tests require beatforge-api[chart-ml]")

from beatforge_api.chart_engine.learning import (  # noqa: E402
    CHECKPOINT_SCHEMA_VERSION,
    FEATURE_NAMES,
    LocalChartModel,
    TrainingConfig,
    candidate_records,
    load_completed_dataset_samples,
    sequence_example,
    train_chart_transformer,
)
from beatforge_api.chart_engine.model import ChartTransformerConfig  # noqa: E402


def _real_dataset_root() -> Path:
    configured = os.environ.get("BEATFORGE_CHART_DATASET_DIR", "").strip()
    root = (
        Path(configured).expanduser().resolve()
        if configured
        else Path(__file__).resolve().parents[3] / "storage" / "chart-engine" / "dataset"
    )
    if not root.is_dir():
        pytest.skip("the completed real SPEED chart dataset is not available")
    has_real_train = False
    for metadata_path in root.glob("*/metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("realData") is True and metadata.get("split") == "train":
            has_real_train = True
            break
    if not has_real_train:
        pytest.skip("the completed real SPEED dataset has no training split")
    return root


def test_real_dataset_candidate_features_and_targets_are_aligned() -> None:
    sample = load_completed_dataset_samples(
        _real_dataset_root(), split="train", verify_audio_hashes=False
    )[0]
    example = sequence_example(sample)

    assert sample.metadata["realData"] is True
    assert sample.chart.mode == "pump-single"
    assert len(example.records) == len(example.lane_targets) == len(example.hold_targets)
    assert example.matched_event_count > 0
    assert len(example.records[0].features) == len(FEATURE_NAMES)
    assert any(any(lanes) for lanes in example.lane_targets)


def test_real_dataset_manifest_closes_all_five_lane_training_triples() -> None:
    root = _real_dataset_root()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((root / "build_report.json").read_text(encoding="utf-8"))
    samples = load_completed_dataset_samples(root, verify_audio_hashes=False)

    assert manifest["realDataOnly"] is True
    assert manifest["mode"] == "pump-single"
    assert manifest["sampleCount"] == len(samples) == 55
    assert manifest["uniqueAudioCount"] == len({sample.audio_sha256 for sample in samples}) == 52
    assert sum(manifest["splits"].values()) == 55
    assert report["sourceChartCount"] == report["completed"] == 55
    assert report["failed"] == report["skipped"] == 0
    assert all(sample.metadata["realData"] is True for sample in samples)


def test_train_checkpoint_and_local_inference_use_real_candidates(tmp_path: Path) -> None:
    dataset_root = _real_dataset_root()
    checkpoint = tmp_path / "chart-transformer.pt"
    result = train_chart_transformer(
        dataset_root,
        checkpoint,
        training=TrainingConfig(
            epochs=1,
            batch_size=1,
            sequence_length=64,
            validation_split=None,
            verify_audio_hashes=False,
            max_batches_per_epoch=1,
            device="cpu",
            seed=17,
        ),
        model_config=ChartTransformerConfig(
            input_dim=len(FEATURE_NAMES),
            d_model=16,
            nhead=2,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            max_sequence_length=64,
        ),
    )

    assert result.checkpoint_path == checkpoint.resolve()
    assert result.metadata["schemaVersion"] == CHECKPOINT_SCHEMA_VERSION
    assert result.metadata["realDataOnly"] is True
    assert result.metadata["trainSampleCount"] > 0
    assert result.metadata["matchedEventCount"] > 0

    sample = load_completed_dataset_samples(dataset_root, split="train", verify_audio_hashes=False)[
        0
    ]
    expected = candidate_records(sample.beatforge)
    runtime = LocalChartModel.load(checkpoint, device="cpu")
    inference = runtime.predict(sample.beatforge, difficulty=sample.training_difficulty)

    assert len(inference.predictions) == len(expected)
    assert inference.predictions[0].candidate_id == expected[0].candidate_id
    assert len(inference.predictions[0].lane_probabilities) == 5
    assert all(
        0.0 <= probability <= 1.0
        for prediction in inference.predictions
        for probability in prediction.lane_probabilities
    )
    assert all(0.0 <= prediction.hold_probability <= 1.0 for prediction in inference.predictions)
