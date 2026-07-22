from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from beatforge_api import chart_routes
from beatforge_api.chart_engine import export_sm, parse_sm, validate_chart
from beatforge_api.chart_engine.footwork import analyze_no_spin_footwork
from beatforge_api.chart_engine.generator import (
    _source_points,
    _tempo_timeline,
    generate_chart,
)
from beatforge_api.chart_engine.learning import (
    FEATURE_NAMES,
    RealDatasetSample,
    TrainingConfig,
    TrainingResult,
    load_completed_dataset_samples,
    train_chart_transformer,
)
from beatforge_api.chart_engine.model import ChartTransformerConfig
from beatforge_api.chart_engine.models import ChartDocument, ChartEvent, ChartNote
from beatforge_api.chart_engine.optimizer import optimize_events
from beatforge_api.chart_engine.statistics import corpus_statistics
from beatforge_api.config import Settings, get_settings
from beatforge_api.database import SessionLocal
from beatforge_api.models import (
    CandidateEventModel,
    HitPointModel,
    ProjectModel,
    TempoSegmentModel,
    TrackModel,
)
from beatforge_api.serialization import dumps


def _real_speed_dataset() -> Path:
    root = get_settings().project_root / "storage" / "chart-engine" / "dataset"
    if not root.is_dir():
        pytest.skip("the completed real SPEED chart dataset is not available")
    return root


@pytest.fixture(scope="module")
def real_speed_sample() -> RealDatasetSample:
    try:
        samples = load_completed_dataset_samples(
            _real_speed_dataset(), split="train", verify_audio_hashes=False
        )
    except ValueError as exc:
        pytest.skip(f"the completed real SPEED training split is unavailable: {exc}")
    return min(
        samples,
        key=lambda sample: len(sample.beatforge["analysis"]["candidate_events"]),
    )


@pytest.fixture(scope="module")
def tiny_real_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> TrainingResult:
    pytest.importorskip(
        "torch", reason="local-checkpoint route integration requires beatforge-api[chart-ml]"
    )
    storage_dir = tmp_path_factory.mktemp("chart-route-model") / "storage"
    checkpoint = storage_dir / "chart-engine" / "models" / "chart-transformer.pt"
    result = train_chart_transformer(
        _real_speed_dataset(),
        checkpoint,
        training=TrainingConfig(
            epochs=1,
            batch_size=1,
            sequence_length=64,
            validation_split=None,
            verify_audio_hashes=False,
            max_batches_per_epoch=1,
            device="cpu",
            seed=29,
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
    return result


def _use_route_storage(monkeypatch: pytest.MonkeyPatch, storage_dir: Path) -> Settings:
    settings = replace(get_settings(), storage_dir=storage_dir.resolve())
    settings.ensure_directories()
    monkeypatch.setattr(chart_routes, "get_settings", lambda: settings)
    chart_routes._load_local_chart_model.cache_clear()
    return settings


def _seed_real_speed_track(sample: RealDatasetSample) -> str:
    analysis = sample.beatforge["analysis"]
    track_id = "10000000-0000-0000-0000-000000000001"
    project = ProjectModel(
        id="20000000-0000-0000-0000-000000000002",
        title=sample.chart.title,
        artist=sample.chart.artist,
        genre="SPEED",
        status="completed",
    )
    track = TrackModel(
        id=track_id,
        project=project,
        original_file_name=sample.chart.music,
        stored_file_name=f"{sample.sample_id}.mp3",
        file_path=str(sample.sample_dir / "audio.mp3"),
        format="mp3",
        original_sample_rate=int(analysis["original_sample_rate"]),
        channels=int(analysis.get("channels", 2)),
        sample_count=int(analysis["sample_count"]),
        duration_sec=float(analysis["duration_sec"]),
        leading_silence_samples=int(analysis.get("leading_silence_samples", 0)),
        analysis_json="{}",
    )
    track.tempo_segments.append(
        TempoSegmentModel(
            start_sample=0,
            bpm=float(analysis["bpm"]),
            beat_offset_sample=int(analysis.get("beat_offset_sample", 0)),
            confidence=float(analysis.get("bpm_confidence", 0.0)),
        )
    )
    for item in analysis.get("hit_points", []):
        track.hit_points.append(_hit_point_model(item))
    with SessionLocal() as session:
        session.add(project)
        session.flush()
        for item in analysis["candidate_events"]:
            track.candidate_events.append(_candidate_model(item))
        session.commit()
    return track_id


def _candidate_model(item: dict[str, Any]) -> CandidateEventModel:
    acoustic_sample = int(item["acoustic_sample"])
    return CandidateEventModel(
        id=str(item["id"]),
        sample=acoustic_sample,
        acoustic_sample=acoustic_sample,
        chart_sample=int(item.get("chart_sample", acoustic_sample)),
        hit_point_id=item.get("hit_point_id"),
        snap_error_ms=float(item.get("snap_error_ms", 0.0)),
        lane=str(item.get("lane", "mix")),
        source_evidence_json=dumps(item.get("source_evidence", {})),
        semantic_evidence_json=dumps(item.get("semantic_evidence", {})),
        confidence=float(item.get("confidence", 0.0)),
        status=str(item.get("status", "uncertain")),
        grid_type=str(item.get("grid_type", "straight_1_16")),
        grid_confidence=float(item.get("grid_confidence", 0.0)),
        source=str(item.get("source", item.get("lane", "mix"))),
        generator=str(item.get("generator", "analysis")),
        character=item.get("character"),
        mora=item.get("mora"),
        phoneme=item.get("phoneme"),
        event_level=str(item.get("event_level", "analysis")),
        event_policy=item.get("event_policy"),
        alignment_unit_id=item.get("alignment_unit_id"),
        alignment_unit_index=item.get("alignment_unit_index"),
        alignment_run_id=item.get("alignment_run_id"),
        character_indices_json=json.dumps(item.get("character_indices", [])),
        phonemes_json=json.dumps(item.get("phonemes", [])),
        aligned_sample=item.get("aligned_sample"),
        refined_sample=item.get("refined_sample"),
        evidence_json=dumps(item.get("evidence", {})),
    )


def _hit_point_model(item: dict[str, Any]) -> HitPointModel:
    sample = int(item.get("sample", item["acoustic_sample"]))
    acoustic_sample = int(item.get("acoustic_sample", sample))
    chart_sample = int(item.get("chart_sample", acoustic_sample))
    return HitPointModel(
        id=str(item["id"]),
        sample=sample,
        acoustic_sample=acoustic_sample,
        chart_sample=chart_sample,
        detected_sample=int(item.get("detected_sample", acoustic_sample)),
        refined_sample=int(item.get("refined_sample", acoustic_sample)),
        snapped_sample=int(item.get("snapped_sample", chart_sample)),
        snap_error_ms=float(item.get("snap_error_ms", 0.0)),
        band=str(item.get("band", "full_band_accent")),
        confidence=float(item.get("confidence", 0.0)),
        salience=float(item.get("salience", item.get("confidence", 0.0))),
        source=str(item.get("source", "fused")),
        detector_votes_json=dumps(item.get("detector_votes", [])),
        primary_stem=str(item.get("primary_stem", "mix")),
        stem_evidence_json=dumps(item.get("stem_evidence", {})),
        manually_edited=bool(item.get("manually_edited", False)),
        locked=bool(item.get("locked", False)),
    )


def _generate(client: TestClient, track_id: str, *, use_model: bool) -> dict[str, Any]:
    response = client.post(
        f"/api/tracks/{track_id}/chart/generate",
        json={
            "difficulty": 7,
            "enableSpin": False,
            "useLocalModel": use_model,
            "seed": 20_260_721,
        },
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_model_suppresses_low_probability_real_candidate_events(
    real_speed_sample: RealDatasetSample,
) -> None:
    analysis = real_speed_sample.beatforge["analysis"]
    # Uncertain proposals remain model-selectable; this deliberately does not
    # exercise the accepted BeatForge evidence policy covered by the next tests.
    candidates = [
        {**candidate, "status": "uncertain"} for candidate in analysis["candidate_events"]
    ]
    selected = candidates[len(candidates) // 2]
    predictions = {
        str(item["id"]): {
            "laneProbabilities": [0.01, 0.01, 0.01, 0.01, 0.01],
            "holdProbability": 0.01,
        }
        for item in candidates
    }
    predictions[str(selected["id"])] = {
        "laneProbabilities": [0.99, 0.01, 0.01, 0.01, 0.01],
        "holdProbability": 0.01,
    }

    chart = generate_chart(
        track_id=real_speed_sample.sample_id,
        title=real_speed_sample.chart.title,
        artist=real_speed_sample.chart.artist,
        music=real_speed_sample.chart.music,
        duration_sec=float(analysis["duration_sec"]),
        sample_rate=int(analysis["original_sample_rate"]),
        tempo_segments=[
            {
                "start_sample": 0,
                "bpm": analysis["bpm"],
                "beat_offset_sample": analysis.get("beat_offset_sample", 0),
            }
        ],
        candidates=candidates,
        # Candidate events are model-selectable. Confirmed hit points have a
        # separate hard-anchor contract covered below.
        hit_points=[],
        difficulty=real_speed_sample.training_difficulty,
        seed=20_260_721,
        model_predictions=predictions,
        model_provenance={"checkpointSha256": "a" * 64},
    )

    assert len(chart.events) == 1
    assert chart.events[0].source_event_id == selected["id"]
    assert chart.generator == "local_chart_transformer"


def test_opening_accepted_hubert_mora_sequence_has_no_silent_hole() -> None:
    """Three accepted 1/16 markers must remain three event rows."""

    sample_rate = 48_000
    candidates = [
        {
            "id": "opening-marker-1",
            "sample": 78_000,
            "acoustic_sample": 78_000,
            "chart_sample": 78_000,
            "hit_point_id": "opening-hit-1",
            "confidence": 0.9,
            "grid_confidence": 0.95,
            "status": "accepted",
            "source": "mix",
            "generator": "analysis",
            "event_level": "analysis",
        },
        {
            "id": "opening-marker-2",
            "sample": 84_000,
            "acoustic_sample": 84_000,
            "chart_sample": 84_000,
            "hit_point_id": None,
            "confidence": 0.85,
            "grid_confidence": 0.9,
            "status": "accepted",
            "source": "vocals",
            "generator": "hubert_ctc",
            "character": "A",
            "mora": "a",
            "phoneme": "a",
            "event_level": "mora",
        },
        {
            "id": "opening-marker-3",
            "sample": 90_000,
            "acoustic_sample": 90_000,
            "chart_sample": 90_000,
            "hit_point_id": None,
            "confidence": 0.8,
            "grid_confidence": 0.85,
            "status": "accepted",
            "source": "vocals",
            "generator": "hubert_ctc",
            "character": "B",
            "mora": "b",
            "phoneme": "b",
            "event_level": "mora",
        },
    ]
    predictions = {
        candidates[0]["id"]: {
            "laneProbabilities": [
                0.1,
                0.8,
                0.2,
                0.4,
                0.1,
            ],
            "holdProbability": 0.01,
        },
        candidates[1]["id"]: {
            "laneProbabilities": [
                0.15,
                0.75,
                0.1,
                0.3,
                0.05,
            ],
            "holdProbability": 0.01,
        },
        candidates[2]["id"]: {
            "laneProbabilities": [
                0.2,
                0.6,
                0.15,
                0.35,
                0.1,
            ],
            "holdProbability": 0.01,
        },
    }

    chart = generate_chart(
        track_id="opening-sequence-regression",
        title="Opening marker contract",
        artist="BeatForge test fixture",
        music="opening-sequence.mp3",
        duration_sec=8.0,
        sample_rate=sample_rate,
        tempo_segments=[
            {
                "start_sample": 0,
                "bpm": 120.0,
                "beat_offset_sample": 0,
            }
        ],
        candidates=candidates,
        hit_points=[],
        difficulty=9,
        seed=42,
        model_predictions=predictions,
        model_provenance={"checkpointSha256": "a" * 64},
    )

    opening_events = [event for event in chart.events if event.beat in {3.25, 3.5, 3.75}]
    assert {event.beat for event in opening_events} == {3.25, 3.5, 3.75}
    assert all(len(event.notes) == 1 for event in opening_events)
    assert {candidate["id"] for candidate in candidates} <= {
        source_id for event in opening_events for source_id in event.source_event_ids
    }


@pytest.mark.parametrize(
    ("seed", "lane_probabilities"),
    [
        (15, None),
        (0, [0.99, 0.01, 0.01, 0.01, 0.01]),
    ],
)
def test_lv8_scattered_marker_does_not_gain_an_unjustified_jump(
    seed: int,
    lane_probabilities: list[float] | None,
) -> None:
    sample_rate = 48_000
    beat = 4.0
    sample = int(beat * 0.5 * sample_rate)
    candidate_id = f"scattered-marker-{seed}"
    predictions = {}
    if lane_probabilities is not None:
        predictions[candidate_id] = {
            "laneProbabilities": lane_probabilities,
            "holdProbability": 0.0,
        }
    chart = generate_chart(
        track_id=candidate_id,
        title="Scattered marker contract",
        artist="BeatForge",
        music="single.mp3",
        duration_sec=8.0,
        sample_rate=sample_rate,
        tempo_segments=[{"start_sample": 0, "bpm": 120.0, "beat_offset_sample": 0}],
        candidates=[
            {
                "id": candidate_id,
                "sample": sample,
                "acoustic_sample": sample,
                "chart_sample": sample,
                "confidence": 0.9,
                "grid_confidence": 0.9,
                "status": "accepted",
                "source": "mix",
                "generator": "analysis",
                "event_level": "analysis",
            }
        ],
        hit_points=[],
        difficulty=8,
        seed=seed,
        model_predictions=predictions,
    )

    assert len(chart.events) == 1
    assert len(chart.events[0].notes) == 1


def test_sparse_model_marker_can_still_generate_a_jump() -> None:
    sample_rate = 48_000
    beat = 4.0
    sample = int(beat * 0.5 * sample_rate)
    candidate_id = "sparse-model-jump"
    chart = generate_chart(
        track_id="sparse-model-jump",
        title="Sparse jump contract",
        artist="BeatForge",
        music="jump.mp3",
        duration_sec=8.0,
        sample_rate=sample_rate,
        tempo_segments=[{"start_sample": 0, "bpm": 120.0, "beat_offset_sample": 0}],
        candidates=[
            {
                "id": candidate_id,
                "sample": sample,
                "acoustic_sample": sample,
                "chart_sample": sample,
                "confidence": 0.9,
                "grid_confidence": 0.9,
                "status": "uncertain",
                "source": "mix",
                "generator": "analysis",
                "event_level": "analysis",
            }
        ],
        hit_points=[],
        difficulty=8,
        seed=20_260_721,
        model_predictions={
            candidate_id: {
                "laneProbabilities": [0.99, 0.90, 0.01, 0.01, 0.01],
                "holdProbability": 0.0,
            }
        },
        model_provenance={"checkpointSha256": "a" * 64},
    )

    assert len(chart.events) == 1
    assert len(chart.events[0].notes) == 2
    assert {note.lane for note in chart.events[0].notes} == {0, 1}


def test_lv8_full_band_downbeat_can_generate_an_accent_jump() -> None:
    sample_rate = 48_000
    beat = 4.0
    sample = int(beat * 0.5 * sample_rate)
    chart = generate_chart(
        track_id="full-band-accent-jump",
        title="Full-band accent contract",
        artist="BeatForge",
        music="accent.mp3",
        duration_sec=8.0,
        sample_rate=sample_rate,
        tempo_segments=[{"start_sample": 0, "bpm": 120.0, "beat_offset_sample": 0}],
        candidates=[],
        hit_points=[
            {
                "id": "full-band-accent-hit",
                "sample": sample,
                "chart_sample": sample,
                "confidence": 0.9,
                "grid_confidence": 0.9,
                "salience": 0.9,
                "band": "full_band_accent",
                "primary_stem": "drums",
            }
        ],
        difficulty=8,
        seed=20_260_721,
        model_predictions={},
    )

    assert len(chart.events) == 1
    assert len(chart.events[0].notes) == 2


def test_lv10_quantizes_accepted_marker_to_sixteenth_grid() -> None:
    sample_rate = 48_000
    source_beat = 4.0 + 1.0 / 6.0
    source_sample = int(round(source_beat * 0.5 * sample_rate))
    chart = generate_chart(
        track_id="lv10-sixteenth-grid",
        title="Lv10 grid contract",
        artist="BeatForge",
        music="grid.mp3",
        duration_sec=8.0,
        sample_rate=sample_rate,
        tempo_segments=[{"start_sample": 0, "bpm": 120.0, "beat_offset_sample": 0}],
        candidates=[
            {
                "id": "accepted-triplet-shaped-marker",
                "sample": source_sample,
                "acoustic_sample": source_sample,
                "chart_sample": source_sample,
                "confidence": 0.9,
                "grid_confidence": 0.1,
                "grid_type": "straight_1_16",
                "status": "accepted",
                "source": "vocals",
                "generator": "hubert_ctc",
                "event_level": "mora",
            }
        ],
        hit_points=[],
        difficulty=10,
        seed=20_260_721,
    )

    assert len(chart.events) == 1
    assert chart.events[0].subdivision == 16
    assert math.isclose(chart.events[0].beat, 4.25, rel_tol=0.0, abs_tol=1e-9)
    assert chart.validation is not None and chart.validation.valid


def test_lv11_preserves_mixed_sixteenth_and_twenty_fourth_markers(
    tmp_path: Path,
) -> None:
    sample_rate = 48_000

    def accepted_marker(marker_id: str, beat: float, *, grid_confidence: float) -> dict[str, Any]:
        sample = int(round(beat * 0.5 * sample_rate))
        return {
            "id": marker_id,
            "sample": sample,
            "acoustic_sample": sample,
            "chart_sample": sample,
            "confidence": 0.9,
            "grid_confidence": grid_confidence,
            "grid_type": "straight_1_16",
            "status": "accepted",
            "source": "vocals",
            "generator": "hubert_ctc",
            "event_level": "mora",
        }

    sixteenth_beat = 4.25
    twenty_fourth_beat = 4.0 + 1.0 / 6.0
    chart = generate_chart(
        track_id="lv11-mixed-grid",
        title="Lv11 grid contract",
        artist="BeatForge",
        music="grid.mp3",
        duration_sec=8.0,
        sample_rate=sample_rate,
        tempo_segments=[{"start_sample": 0, "bpm": 120.0, "beat_offset_sample": 0}],
        candidates=[
            accepted_marker("sixteenth-marker", sixteenth_beat, grid_confidence=0.95),
            accepted_marker("twenty-fourth-marker", twenty_fourth_beat, grid_confidence=0.10),
        ],
        hit_points=[],
        difficulty=11,
        seed=20_260_721,
    )

    assert any(
        event.subdivision == 16
        and math.isclose(event.beat, sixteenth_beat, rel_tol=0.0, abs_tol=1e-9)
        for event in chart.events
    )
    assert any(
        event.subdivision == 24
        and math.isclose(event.beat, twenty_fourth_beat, rel_tol=0.0, abs_tol=1e-9)
        for event in chart.events
    )
    assert chart.validation is not None and chart.validation.valid

    lv10_validation = validate_chart(chart.model_copy(update={"meter": 10}))
    assert lv10_validation.valid is False
    assert "SUBDIVISION_TOO_FINE_FOR_LEVEL" in {
        issue.code for issue in lv10_validation.issues
    }

    exported_path = tmp_path / "mixed-sixteenth-twenty-fourth.sm"
    exported_path.write_text(export_sm(chart), encoding="utf-8")
    reparsed = parse_sm(exported_path)
    assert {round(event.beat, 9) for event in reparsed.events} == {
        round(sixteenth_beat, 9),
        round(twenty_fourth_beat, 9),
    }


def test_model_filters_candidates_but_preserves_real_hit_point_anchors(
    real_speed_sample: RealDatasetSample,
) -> None:
    """Confirmed BeatForge hit points are inputs, not model-selectable candidates."""

    analysis = real_speed_sample.beatforge["analysis"]
    sample_rate = int(analysis["original_sample_rate"])
    difficulty = real_speed_sample.training_difficulty
    step = 0.25  # This real fixture is Lv.8, whose generator grid is 1/16 notes.
    timeline, _changes = _tempo_timeline(
        [
            {
                "start_sample": 0,
                "bpm": analysis["bpm"],
                "beat_offset_sample": analysis.get("beat_offset_sample", 0),
            }
        ],
        sample_rate,
    )

    def slot(item: dict[str, Any]) -> int:
        chart_sample = int(item.get("chart_sample", item["sample"]))
        return int(round(timeline.time_to_beat(chart_sample / sample_rate) / step))

    # Use the real SPEED structures, but keep the case minimal: three confirmed
    # anchors in distinct quantization slots plus two candidate-only events.
    anchors: list[dict[str, Any]] = []
    anchor_slots: set[int] = set()
    for hit_point in analysis["hit_points"]:
        hit_slot = slot(hit_point)
        if hit_slot in anchor_slots:
            continue
        anchors.append(hit_point)
        anchor_slots.add(hit_slot)
        if len(anchors) == 3:
            break
    assert len(anchors) == 3

    candidates_by_id = {
        str(candidate["id"]): candidate for candidate in analysis["candidate_events"]
    }
    anchor_candidate_ids = {
        str(anchor["candidate_event_id"]) for anchor in anchors if anchor.get("candidate_event_id")
    }
    anchor_candidates = [candidates_by_id[candidate_id] for candidate_id in anchor_candidate_ids]
    assert len(anchor_candidates) == 3

    candidate_only: list[dict[str, Any]] = []
    occupied_slots = set(anchor_slots)
    for candidate in analysis["candidate_events"]:
        candidate_id = str(candidate["id"])
        candidate_slot = slot(candidate)
        if candidate_id in anchor_candidate_ids or candidate_slot in occupied_slots:
            continue
        # This case exercises model-selectable proposals. Accepted candidates
        # are rhythm anchors and have their own preservation regression above.
        candidate_only.append({**candidate, "status": "uncertain"})
        occupied_slots.add(candidate_slot)
        if len(candidate_only) == 2:
            break
    assert len(candidate_only) == 2
    selected_candidate, rejected_candidate = candidate_only

    predictions = {
        str(item["id"]): {
            "laneProbabilities": [0.01, 0.01, 0.01, 0.01, 0.01],
            "holdProbability": 0.01,
        }
        for item in anchor_candidates + candidate_only
    }
    predictions[str(selected_candidate["id"])] = {
        "laneProbabilities": [0.99, 0.01, 0.01, 0.01, 0.01],
        "holdProbability": 0.01,
    }

    chart = generate_chart(
        track_id=real_speed_sample.sample_id,
        title=real_speed_sample.chart.title,
        artist=real_speed_sample.chart.artist,
        music=real_speed_sample.chart.music,
        duration_sec=float(analysis["duration_sec"]),
        sample_rate=sample_rate,
        tempo_segments=[
            {
                "start_sample": 0,
                "bpm": analysis["bpm"],
                "beat_offset_sample": analysis.get("beat_offset_sample", 0),
            }
        ],
        candidates=anchor_candidates + candidate_only,
        hit_points=anchors,
        difficulty=difficulty,
        seed=20_260_721,
        model_predictions=predictions,
        model_provenance={"checkpointSha256": "a" * 64},
    )

    # Quantization may move an anchor by at most half a grid step. Each of the
    # three distinct anchor slots must still be represented by a chart event.
    tolerance_sec = (60.0 / float(analysis["bpm"])) * step / 2.0 + 1.0 / sample_rate
    event_times = [event.time_sec for event in chart.events]
    for anchor in anchors:
        anchor_time = int(anchor["chart_sample"]) / sample_rate
        assert any(abs(event_time - anchor_time) <= tolerance_sec for event_time in event_times), (
            f"missing confirmed hit-point anchor {anchor['id']} at {anchor_time:.6f}s"
        )

    source_event_ids = {event.source_event_id for event in chart.events}
    assert str(selected_candidate["id"]) in source_event_ids
    assert str(rejected_candidate["id"]) not in source_event_ids


def test_uncertain_aligned_vocal_mora_is_a_required_rhythm_marker() -> None:
    sample_rate = 1_000
    tempo_segments = [{"start_sample": 0, "bpm": 120.0, "beat_offset_sample": 0}]
    timeline, _changes = _tempo_timeline(tempo_segments, sample_rate)
    vocal_marker = {
        "id": "aligned-vocal-mora",
        "sample": 8_128,
        "acoustic_sample": 8_128,
        "chart_sample": 8_104,
        "lane": "vocals",
        "source": "vocals",
        "generator": "hubert_ctc",
        "event_level": "mora",
        "alignment_unit_id": "mora-event:mora-42",
        "alignment_unit_index": 42,
        "alignment_run_id": "alignment-run",
        "mora": "ツ",
        "status": "uncertain",
        "confidence": 0.405,
        "grid_confidence": 0.731,
        "salience": 0.405,
    }
    uncertain_mix = {
        **vocal_marker,
        "id": "uncertain-mix",
        "source": "mix",
        "lane": "mix",
        "generator": "analysis",
        "event_level": "analysis",
        "chart_sample": 8_604,
    }
    rejected_vocal = {
        **vocal_marker,
        "id": "rejected-vocal-mora",
        "status": "rejected",
        "chart_sample": 9_104,
    }
    candidates = [vocal_marker, uncertain_mix, rejected_vocal]
    predictions = {
        str(candidate["id"]): {
            "laneProbabilities": [0.06, 0.45, 0.03, 0.47, 0.28],
            "holdProbability": 0.01,
        }
        for candidate in candidates
    }

    points = _source_points(
        timeline=timeline,
        sample_rate=sample_rate,
        duration_sec=12.0,
        candidates=candidates,
        hit_points=[],
        difficulty=8,
        model_predictions=predictions,
    )

    assert [point.source_id for point in points] == ["aligned-vocal-mora"]
    assert points[0].anchor_priority == 1

    chart = generate_chart(
        track_id="vocal-marker-regression",
        title="Vocal marker regression",
        artist="BeatForge",
        music="fixture.mp3",
        duration_sec=12.0,
        sample_rate=sample_rate,
        tempo_segments=tempo_segments,
        candidates=candidates,
        hit_points=[],
        difficulty=8,
        seed=20_260_721,
        model_predictions=predictions,
    )

    assert len(chart.events) == 1
    assert chart.events[0].source_event_ids == ["aligned-vocal-mora"]
    assert chart.events[0].time_sec == pytest.approx(8.125)
    assert chart.optimization is not None
    assert chart.optimization["vocal_mora_markers_input"] == 1
    assert chart.optimization["vocal_mora_markers_output"] == 1


def test_same_quantized_slot_is_the_only_hit_point_merge_boundary(
    real_speed_sample: RealDatasetSample,
) -> None:
    analysis = real_speed_sample.beatforge["analysis"]
    sample_rate = int(analysis["original_sample_rate"])
    difficulty = real_speed_sample.training_difficulty
    step = 0.25
    timeline, _changes = _tempo_timeline(
        [
            {
                "start_sample": 0,
                "bpm": analysis["bpm"],
                "beat_offset_sample": analysis.get("beat_offset_sample", 0),
            }
        ],
        sample_rate,
    )
    anchor = dict(analysis["hit_points"][2])
    anchor_beat = (
        round(timeline.time_to_beat(int(anchor["chart_sample"]) / sample_rate) / step) * step
    )
    same_slot = {
        **anchor,
        "id": "same-slot-confirmed-hit",
        "sample": int(anchor["chart_sample"]) + 1,
        "chart_sample": int(anchor["chart_sample"]) + 1,
        "candidate_event_id": None,
    }
    next_slot_sample = int(round(timeline.beat_to_time(anchor_beat + step) * sample_rate))
    next_slot = {
        **anchor,
        "id": "next-slot-confirmed-hit",
        "sample": next_slot_sample,
        "chart_sample": next_slot_sample,
        "candidate_event_id": None,
    }

    points = _source_points(
        timeline=timeline,
        sample_rate=sample_rate,
        duration_sec=float(analysis["duration_sec"]),
        candidates=[],
        hit_points=[anchor, same_slot, next_slot],
        difficulty=difficulty,
        model_predictions=None,
    )

    assert [point.beat for point in points] == [anchor_beat, anchor_beat + step]
    assert set(points[0].source_hit_point_ids) == {
        str(anchor["id"]),
        "same-slot-confirmed-hit",
    }
    assert points[1].source_hit_point_ids == ("next-slot-confirmed-hit",)


def test_point_budget_never_downsamples_real_hit_point_slots(
    real_speed_sample: RealDatasetSample,
) -> None:
    analysis = real_speed_sample.beatforge["analysis"]
    sample_rate = int(analysis["original_sample_rate"])
    difficulty = 1
    timeline, _changes = _tempo_timeline(
        [
            {
                "start_sample": 0,
                "bpm": analysis["bpm"],
                "beat_offset_sample": analysis.get("beat_offset_sample", 0),
            }
        ],
        sample_rate,
    )
    # This tail of the real SPEED sample has more confirmed quantized slots than
    # the Lv.1 point budget. Budgeting may reject candidates, never these anchors.
    tail_start = float(analysis["duration_sec"]) * 0.70
    anchors = [
        hit_point
        for hit_point in analysis["hit_points"]
        if int(hit_point["chart_sample"]) / sample_rate >= tail_start
    ]
    source_points = _source_points(
        timeline=timeline,
        sample_rate=sample_rate,
        duration_sec=float(analysis["duration_sec"]),
        candidates=[],
        hit_points=anchors,
        difficulty=difficulty,
        model_predictions=None,
    )
    assert len(source_points) >= 16

    chart = generate_chart(
        track_id=real_speed_sample.sample_id,
        title=real_speed_sample.chart.title,
        artist=real_speed_sample.chart.artist,
        music=real_speed_sample.chart.music,
        duration_sec=float(analysis["duration_sec"]),
        sample_rate=sample_rate,
        tempo_segments=[
            {
                "start_sample": 0,
                "bpm": analysis["bpm"],
                "beat_offset_sample": analysis.get("beat_offset_sample", 0),
            }
        ],
        candidates=[],
        hit_points=anchors,
        difficulty=difficulty,
        seed=20_260_721,
        # An empty prediction map selects no candidate-only points and disables
        # inferred grid fill, isolating the real confirmed-anchor budget.
        model_predictions={},
    )

    expected_anchor_beats = {point.beat for point in source_points}
    assert expected_anchor_beats <= {event.beat for event in chart.events}


def test_density_optimizer_downgrades_optional_jump_before_hit_point_anchor(
    real_speed_sample: RealDatasetSample,
) -> None:
    analysis = real_speed_sample.beatforge["analysis"]
    sample_rate = int(analysis["original_sample_rate"])
    difficulty = real_speed_sample.training_difficulty
    step = 0.25
    timeline, _changes = _tempo_timeline(
        [
            {
                "start_sample": 0,
                "bpm": analysis["bpm"],
                "beat_offset_sample": analysis.get("beat_offset_sample", 0),
            }
        ],
        sample_rate,
    )
    first_beat = 8.0
    candidate_ids: list[str] = []
    events: list[ChartEvent] = []
    for index, candidate in enumerate(analysis["candidate_events"][:7]):
        beat = first_beat + index * step
        candidate_id = str(candidate["id"])
        candidate_ids.append(candidate_id)
        events.append(
            ChartEvent(
                time_sec=timeline.beat_to_time(beat),
                beat=beat,
                measure=int(beat // 4),
                subdivision=16,
                row_index=int(round((beat % 4) * 4)),
                notes=[
                    ChartNote(lane=0, source="candidate", confidence=1.0 - index * 0.01),
                    ChartNote(lane=4, source="candidate", confidence=0.9 - index * 0.01),
                ],
                source_event_id=candidate_id,
                source_event_ids=[candidate_id],
                anchor_priority=0,
            )
        )

    anchor_beat = first_beat + 7 * step
    anchor_id = "density-protected-hit"
    events.append(
        ChartEvent(
            time_sec=timeline.beat_to_time(anchor_beat),
            beat=anchor_beat,
            measure=int(anchor_beat // 4),
            subdivision=16,
            row_index=int(round((anchor_beat % 4) * 4)),
            notes=[ChartNote(lane=2, source="hit_point", confidence=1.0)],
            source_event_id=anchor_id,
            source_hit_point_ids=[anchor_id],
            anchor_priority=2,
        )
    )

    optimized, report = optimize_events(events, difficulty)

    assert any(event.source_hit_point_ids == [anchor_id] for event in optimized)
    retained_optional_ids = {
        source_id for event in optimized for source_id in event.source_event_ids
    }
    assert retained_optional_ids == set(candidate_ids)
    assert sum(len(event.notes) == 1 for event in optimized[:-1]) == 1
    assert report.simultaneous_notes_removed == 1
    assert report.density_events_removed == 0

    accepted_events = [
        event.model_copy(update={"anchor_priority": 1}) for event in events[:-1]
    ]
    accepted_optimized, accepted_report = optimize_events(
        [*accepted_events, events[-1]], difficulty
    )
    assert len(accepted_optimized) == len(events)
    assert accepted_report.density_events_removed == 0


def test_lv8_optimizer_breaks_disjoint_jump_chain_without_losing_rows() -> None:
    def jump_event(beat: float, lanes: tuple[int, int], source_id: str) -> ChartEvent:
        return ChartEvent(
            time_sec=beat * 0.5,
            beat=beat,
            measure=int(beat // 4),
            subdivision=8,
            row_index=int(round((beat % 4) * 2)),
            notes=[
                ChartNote(lane=lanes[0], source="candidate", confidence=0.9),
                ChartNote(lane=lanes[1], source="candidate", confidence=0.8),
            ],
            source_event_id=source_id,
            source_event_ids=[source_id],
            anchor_priority=1,
        )

    raw_events = [
        jump_event(4.0, (0, 1), "first-jump"),
        jump_event(4.5, (2, 3), "reposition-jump"),
        jump_event(8.0, (3, 4), "isolated-jump"),
    ]
    raw_chart = ChartDocument(
        id="jump-chain-validator",
        title="Jump chain validator",
        mode="pump-single",
        lane_count=5,
        meter=8,
        bpm=120.0,
        duration_sec=5.0,
        measure_count=3,
        tempo_map=[{"beat": 0.0, "bpm": 120.0, "time_sec": 0.0}],
        events=raw_events,
    )

    raw_validation = validate_chart(raw_chart)
    assert "DISJOINT_JUMP_TRANSITION" in {
        issue.code for issue in raw_validation.issues
    }

    optimized, report = optimize_events(raw_events, 8, bpm=120.0)

    assert [len(event.notes) for event in optimized] == [2, 1, 2]
    assert [event.source_event_id for event in optimized] == [
        "first-jump",
        "reposition-jump",
        "isolated-jump",
    ]
    assert report.simultaneous_notes_removed == 1
    optimized_chart = raw_chart.model_copy(update={"events": optimized})
    assert "DISJOINT_JUMP_TRANSITION" not in {
        issue.code for issue in validate_chart(optimized_chart).issues
    }


def test_lv8_optimizer_repairs_forced_spin_without_losing_vocal_rows() -> None:
    times = [
        39.3834140167695,
        39.49969308653694,
        39.61597215630438,
        39.84853029583927,
        39.964809365606705,
        40.19736750514159,
    ]
    beats = [84.5, 84.75, 85.0, 85.5, 85.75, 86.25]
    # User-reported 5(R), 3(L), 4(R), 1(L), 2(R), 4(L). The final
    # left-foot right-up step places L=right-up/R=left-up and faces backward.
    lanes = [4, 2, 3, 0, 1, 3]
    stale_feet = ["right", "left", "right", "left", "left", "right"]
    source_ids = [f"aligned-vocal-{index}" for index in range(len(lanes))]
    events = [
        ChartEvent(
            time_sec=time_sec,
            beat=beat,
            measure=int(beat // 4),
            subdivision=16,
            row_index=int(round((beat % 4) * 4)),
            notes=[
                ChartNote(
                    lane=lane,
                    source="vocals",
                    confidence=0.9,
                    foot=foot,
                )
            ],
            source_event_id=source_id,
            source_event_ids=[source_id],
            anchor_priority=1,
        )
        for time_sec, beat, lane, foot, source_id in zip(
            times, beats, lanes, stale_feet, source_ids, strict=True
        )
    ]
    raw_chart = ChartDocument(
        id="forced-spin-regression",
        title="Forced spin regression",
        source_group="BEATFORGE_GENERATED",
        mode="pump-single",
        lane_count=5,
        meter=8,
        bpm=129.0,
        duration_sec=42.0,
        measure_count=22,
        tempo_map=[{"beat": 0.0, "bpm": 129.0, "time_sec": 0.0}],
        events=events,
        generator="local_chart_transformer",
        spin_enabled=False,
    )

    raw_analysis = analyze_no_spin_footwork(events)
    assert raw_analysis.full_step_reachable is False
    assert raw_analysis.violations[0].beat == 86.25
    raw_issue = next(
        issue
        for issue in validate_chart(raw_chart).issues
        if issue.code == "NO_FULL_ALTERNATING_PATH"
    )
    assert raw_issue.time_sec == pytest.approx(times[-1])

    optimized, report = optimize_events(events, 8, bpm=129.0)

    assert len(optimized) == len(events)
    assert [event.time_sec for event in optimized] == times
    assert [event.source_event_id for event in optimized] == source_ids
    assert [event.source_event_ids for event in optimized] == [[value] for value in source_ids]
    assert report.footwork_lanes_reassigned == 1
    assert [event.notes[0].foot for event in optimized] == [
        "right",
        "left",
        "right",
        "left",
        "right",
        "left",
    ]
    assert [event.notes[0].lane for event in optimized] == [4, 2, 3, 0, 1, 2]
    assert analyze_no_spin_footwork(optimized).full_step_reachable is True
    optimized_chart = raw_chart.model_copy(update={"events": optimized})
    assert "NO_FULL_ALTERNATING_PATH" not in {
        issue.code for issue in validate_chart(optimized_chart).issues
    }


def test_no_spin_solver_keeps_legal_135_degree_crossovers() -> None:
    # DL, UL, center, UR requires a deep crossover for either starting foot,
    # but never faces backward. A 90-degree-only rule would reject real SPEED
    # technique; the no-spin limit intentionally includes +/-135 degrees.
    events = [
        ChartEvent(
            time_sec=index * 0.2,
            beat=index * 0.25,
            measure=0,
            subdivision=16,
            row_index=index,
            notes=[ChartNote(lane=lane)],
        )
        for index, lane in enumerate((0, 1, 2, 3))
    ]

    analysis = analyze_no_spin_footwork(events)

    assert analysis.full_step_reachable is True
    assert analysis.max_abs_heading == pytest.approx(135.0)


def test_corpus_transitions_do_not_bridge_across_a_jump() -> None:
    events = [
        ChartEvent(
            time_sec=0.0,
            beat=0.0,
            measure=0,
            notes=[ChartNote(lane=4)],
        ),
        ChartEvent(
            time_sec=0.25,
            beat=0.5,
            measure=0,
            notes=[ChartNote(lane=1), ChartNote(lane=2)],
        ),
        ChartEvent(
            time_sec=0.5,
            beat=1.0,
            measure=0,
            notes=[ChartNote(lane=3)],
        ),
    ]
    chart = ChartDocument(
        id="jump-transition-boundary",
        title="Jump transition boundary",
        mode="pump-single",
        lane_count=5,
        meter=14,
        bpm=120.0,
        duration_sec=1.0,
        measure_count=1,
        tempo_map=[{"beat": 0.0, "bpm": 120.0, "time_sec": 0.0}],
        events=events,
    )

    transitions = corpus_statistics([chart]).lane_transition_probabilities

    assert transitions == [[0.2] * 5 for _ in range(5)]


def test_hit_point_replacement_keeps_candidate_prediction_provenance(
    real_speed_sample: RealDatasetSample,
) -> None:
    analysis = real_speed_sample.beatforge["analysis"]
    predictions = {
        str(item["id"]): {
            "laneProbabilities": [0.9, 0.1, 0.1, 0.1, 0.1],
            "holdProbability": 0.1,
        }
        for item in analysis["candidate_events"]
    }
    timeline, _changes = _tempo_timeline(
        [
            {
                "start_sample": 0,
                "bpm": analysis["bpm"],
                "beat_offset_sample": analysis.get("beat_offset_sample", 0),
            }
        ],
        int(analysis["original_sample_rate"]),
    )

    points = _source_points(
        timeline=timeline,
        sample_rate=int(analysis["original_sample_rate"]),
        duration_sec=float(analysis["duration_sec"]),
        candidates=analysis["candidate_events"],
        hit_points=analysis["hit_points"],
        difficulty=real_speed_sample.training_difficulty,
        model_predictions=predictions,
    )

    probability_points = [point for point in points if point.lane_probabilities is not None]
    assert any(point.source == "hit_point" for point in probability_points)
    assert all(point.source_id in predictions for point in probability_points)


def test_route_uses_local_checkpoint_and_records_real_training_provenance(
    client: TestClient,
    real_speed_sample: RealDatasetSample,
    tiny_real_checkpoint: TrainingResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_route_storage(monkeypatch, tiny_real_checkpoint.checkpoint_path.parents[2])
    track_id = _seed_real_speed_track(real_speed_sample)

    payload = _generate(client, track_id, use_model=True)

    model_status = payload["referenceCorpus"]["model"]
    chart = payload["chart"]
    provenance = chart["modelProvenance"]
    assert model_status == {"requested": True, "available": True, "used": True}
    assert chart["generator"] == "local_chart_transformer"
    assert provenance["schemaVersion"] == "beatforge.chart-transformer.checkpoint.v1"
    assert provenance["architecture"] == "beatforge.chart-transformer.encoder.v1"
    assert provenance["datasetFingerprint"] == tiny_real_checkpoint.metadata["datasetFingerprint"]
    assert provenance["sampleCount"] == tiny_real_checkpoint.metadata["sampleCount"]
    assert provenance["realDataOnly"] is True
    assert (
        provenance["checkpointSha256"]
        == hashlib.sha256(tiny_real_checkpoint.checkpoint_path.read_bytes()).hexdigest()
    )
    real_candidate_ids = {
        item["id"] for item in real_speed_sample.beatforge["analysis"]["candidate_events"]
    }
    assert chart["events"]
    assert all(event["sourceEventId"] in real_candidate_ids for event in chart["events"])

    rule_payload = _generate(client, track_id, use_model=False)
    assert rule_payload["chart"]["generator"] == "real_corpus_profile_rules"
    assert rule_payload["generationId"] != payload["generationId"]
    saved_model_chart = chart_routes._load_generated(track_id, payload["generationId"])
    assert saved_model_chart.generator == "local_chart_transformer"
    assert saved_model_chart.model_provenance == provenance

    torch = pytest.importorskip("torch")
    checkpoint_payload = torch.load(
        tiny_real_checkpoint.checkpoint_path, map_location="cpu", weights_only=True
    )
    checkpoint_payload["model_state_dict"]["lane_head.bias"][0] += 0.125
    torch.save(checkpoint_payload, tiny_real_checkpoint.checkpoint_path)
    chart_routes._load_local_chart_model.cache_clear()
    changed_payload = _generate(client, track_id, use_model=True)
    changed_provenance = changed_payload["chart"]["modelProvenance"]
    assert changed_provenance["checkpointSha256"] != provenance["checkpointSha256"]
    assert changed_payload["generationId"] != payload["generationId"]
    assert (
        chart_routes._load_generated(track_id, payload["generationId"]).model_provenance
        == provenance
    )


def test_route_falls_back_to_rules_when_checkpoint_is_absent(
    client: TestClient,
    real_speed_sample: RealDatasetSample,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _use_route_storage(monkeypatch, tmp_path / "fallback-storage")
    assert not (settings.chart_models_dir / "chart-transformer.pt").exists()
    track_id = _seed_real_speed_track(real_speed_sample)

    first = _generate(client, track_id, use_model=True)
    second = _generate(client, track_id, use_model=True)

    assert first == second
    assert first["referenceCorpus"]["model"] == {
        "requested": True,
        "available": False,
        "used": False,
    }
    assert first["chart"]["generator"] == "real_corpus_profile_rules"
    assert first["chart"]["modelProvenance"] is None


def test_route_rejects_rule_generation_without_the_real_speed_corpus(
    client: TestClient,
    real_speed_sample: RealDatasetSample,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _use_route_storage(monkeypatch, tmp_path / "empty-corpus-storage")
    empty_corpus = tmp_path / "empty-speed-corpus"
    empty_corpus.mkdir()
    monkeypatch.setenv("BEATFORGE_SPEED_CHARTS_DIR", str(empty_corpus))
    chart_routes._library_for_root.cache_clear()
    chart_routes._statistics_for_root.cache_clear()
    track_id = _seed_real_speed_track(real_speed_sample)

    response = client.post(
        f"/api/tracks/{track_id}/chart/generate",
        json={"difficulty": 7, "useLocalModel": False, "seed": 20_260_721},
    )

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "REFERENCE_CORPUS_NOT_FOUND"
