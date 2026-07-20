from __future__ import annotations

import json

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from beatforge_api.config import get_settings
from beatforge_api.database import SessionLocal
from beatforge_api.models import (
    HitPointModel,
    TempoSegmentModel,
    TrackModel,
    VocalAlignmentJobModel,
)
from beatforge_api.vocal_jobs import (
    VocalJobError,
    _build_anchors,
    _build_chunk_coverage,
    _detect_vocal_fallback_anchors,
    _focus_supports_vocals,
    _replace_vocal_hits,
    _replace_vocal_hits_with_stats,
    _run_job,
    clean_lyrics,
)

from .test_api import upload


def _prepare_vocal_track(track_id: str) -> None:
    settings = get_settings()
    sample_rate = 8_000
    samples = np.arange(sample_rate, dtype=np.float32)
    envelope = np.zeros(sample_rate, dtype=np.float32)
    for onset in (800, 1_800):
        stop = min(sample_rate, onset + 600)
        envelope[onset:stop] += np.linspace(0.0, 1.0, stop - onset, dtype=np.float32)
    vocals = 0.18 * np.sin(2 * np.pi * 220 * samples / sample_rate) * envelope
    stem_path = settings.stems_dir / track_id / "vocals.flac"
    stem_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(stem_path, vocals, sample_rate, format="FLAC")
    with SessionLocal() as session:
        track = session.get(TrackModel, track_id)
        assert track is not None
        track.tempo_segments.append(
            TempoSegmentModel(
                start_sample=0,
                bpm=120.0,
                beat_offset_sample=0,
                confidence=1.0,
                manually_edited=True,
            )
        )
        session.commit()


def test_vocal_lyrics_save_and_job_contract(client: TestClient, monkeypatch) -> None:
    track_id = upload(client)["track"]["id"]
    empty = client.get(f"/api/tracks/{track_id}/vocal-lyrics")
    assert empty.status_code == 200
    assert empty.json()["status"] == "empty"

    saved = client.put(
        f"/api/tracks/{track_id}/vocal-lyrics",
        json={"text": "ひかり\nひらく", "inputFormat": "japanese"},
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["status"] == "saved"
    assert saved.json()["text"] == "ひかり\nひらく"

    monkeypatch.setattr("beatforge_api.routes.submit_vocal_job", lambda _job_id: None)
    started = client.post(f"/api/tracks/{track_id}/vocal-lyrics/align")
    assert started.status_code == 202, started.text
    job = client.get(f"/api/vocal-lyrics-jobs/{started.json()['jobId']}")
    assert job.status_code == 200, job.text
    assert job.json()["kind"] == "alignment"
    assert job.json()["stage"] == "queued"
    assert job.json()["result"] is None


def test_romaji_alignment_is_rejected_before_model_start(client: TestClient) -> None:
    track_id = upload(client)["track"]["id"]
    saved = client.put(
        f"/api/tracks/{track_id}/vocal-lyrics",
        json={"text": "mirai wo hiraku", "inputFormat": "romaji"},
    )
    assert saved.status_code == 200

    started = client.post(f"/api/tracks/{track_id}/vocal-lyrics/align")

    assert started.status_code == 422
    assert started.json()["error"]["code"] == "ROMAJI_REQUIRES_KANA"


def test_vocal_alignment_job_persists_grid_anchors_and_hits(
    client: TestClient, monkeypatch
) -> None:
    track_id = upload(client)["track"]["id"]
    _prepare_vocal_track(track_id)
    with SessionLocal() as session:
        track = session.get(TrackModel, track_id)
        assert track is not None
        track.lyrics_text = "ひか"
        track.lyrics_format = "japanese"
        job = VocalAlignmentJobModel(
            track=track,
            operation="alignment",
            replace_vocal_hits=True,
        )
        session.add(job)
        session.commit()
        job_id = job.id

    operations: list[str] = []

    def fake_qwen(operation: str, **_kwargs: object) -> dict[str, object]:
        operations.append(operation)
        return {
            "status": "ok",
            "model": "local-test-aligner",
            "device": "mps",
            "warnings": [],
            "timestamps": [
                {
                    "text": "か",
                    "kana": "か",
                    "romaji": "ka",
                    "start_sample": 760,
                    "end_sample": 1_240,
                    "chunk_index": 0,
                    "chunk_match_confidence": 0.9,
                },
                {
                    "text": "わ",
                    "kana": "わ",
                    "romaji": "wa",
                    "start_sample": 1_760,
                    "end_sample": 2_240,
                    "chunk_index": 0,
                    "chunk_match_confidence": 0.9,
                },
            ],
            "chunks": [
                {
                    "index": 0,
                    "startSample": 0,
                    "endSample": 8_000,
                    "status": "ok",
                    "alignmentStatus": "ok",
                    "matchConfidence": 0.9,
                }
            ],
        }

    monkeypatch.setattr("beatforge_api.vocal_jobs._run_qwen", fake_qwen)
    _run_job(job_id)

    assert operations == ["align_song"]

    response = client.get(f"/api/vocal-lyrics-jobs/{job_id}")
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "completed", payload["error"]["message"]
    assert payload["result"]["status"] == "completed"
    anchors = payload["result"]["anchors"]
    assert [anchor["gridSample"] for anchor in anchors] == [1_000, 2_000], [
        (anchor["alignedSample"], anchor["refinedSample"], anchor["gridSample"])
        for anchor in anchors
    ]
    assert [anchor["alignedSample"] for anchor in anchors] == [760, 1_760]
    assert all(anchor["chartCandidate"] for anchor in anchors)
    assert all(anchor["active"] for anchor in anchors)
    coverage_chunk = payload["result"]["coverageChunks"][0]
    assert coverage_chunk["index"] == 0
    assert coverage_chunk["startSample"] == 0
    assert coverage_chunk["endSample"] == 8_000
    assert coverage_chunk["status"] == "success"
    assert coverage_chunk["anchorCount"] == 2
    assert coverage_chunk["rawTimestampCount"] == 2
    assert 0.5 < coverage_chunk["confidence"] < 0.9

    track = client.get(f"/api/tracks/{track_id}").json()
    vocal_hits = [hit for hit in track["hitPoints"] if hit["primaryStem"] == "vocals"]
    assert [hit["sample"] for hit in vocal_hits] == [
        hit["refinedSample"] for hit in vocal_hits
    ]
    assert [hit["snappedSample"] for hit in vocal_hits] == [1_000, 2_000]
    assert all(hit["sample"] != hit["snappedSample"] for hit in vocal_hits)
    assert all(hit["acousticSample"] == hit["sample"] for hit in vocal_hits)
    assert [hit["chartSample"] for hit in vocal_hits] == [1_000, 2_000]
    assert [hit["detectedSample"] for hit in vocal_hits] == [760, 1_760]
    assert all("qwen_forced_alignment" in hit["detectorVotes"] for hit in vocal_hits)
    assert all("absolute_vocal_activity" in hit["detectorVotes"] for hit in vocal_hits)
    candidates = track["candidateEvents"]
    assert len(candidates) == 2
    assert all(candidate["lane"] == "vocals" for candidate in candidates)
    assert all(candidate["status"] == "accepted" for candidate in candidates)
    assert all(candidate["acousticSample"] != candidate["chartSample"] for candidate in candidates)
    assert all(
        candidate["semanticEvidence"]["phonemeConfidence"] == 0.0
        for candidate in candidates
    )

    with SessionLocal() as session:
        stored = session.get(TrackModel, track_id)
        assert stored is not None
        metadata = json.loads(stored.vocal_alignment_json)
        assert metadata["created_hit_count"] == 2


def test_asr_draft_is_saved_for_user_review(client: TestClient, monkeypatch) -> None:
    track_id = upload(client)["track"]["id"]
    _prepare_vocal_track(track_id)
    with SessionLocal() as session:
        track = session.get(TrackModel, track_id)
        assert track is not None
        job = VocalAlignmentJobModel(track=track, operation="asr_draft")
        session.add(job)
        session.commit()
        job_id = job.id

    monkeypatch.setattr(
        "beatforge_api.vocal_jobs._run_qwen",
        lambda *_args, **_kwargs: {
            "status": "ok",
            "text": "雨を待つ",
            "model": "local-test-asr",
            "device": "mps",
            "warnings": [],
        },
    )
    _run_job(job_id)
    result = client.get(f"/api/vocal-lyrics-jobs/{job_id}").json()["result"]
    assert result["status"] == "draft"
    assert result["text"] == "雨を待つ"
    assert result["anchors"] == []


def test_alignment_failure_uses_acoustic_fallback_without_hiding_failure(
    client: TestClient, monkeypatch
) -> None:
    track_id = upload(client)["track"]["id"]
    _prepare_vocal_track(track_id)
    with SessionLocal() as session:
        track = session.get(TrackModel, track_id)
        assert track is not None
        track.lyrics_text = "ひか"
        track.lyrics_format = "japanese"
        job = VocalAlignmentJobModel(
            track=track,
            operation="alignment",
            replace_vocal_hits=True,
        )
        session.add(job)
        session.commit()
        job_id = job.id

    def fail_alignment(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise VocalJobError("ALIGNMENT_COLLAPSED", "test collapse")

    monkeypatch.setattr("beatforge_api.vocal_jobs._run_qwen", fail_alignment)
    _run_job(job_id)

    payload = client.get(f"/api/vocal-lyrics-jobs/{job_id}").json()
    assert payload["status"] == "completed", payload
    assert payload["result"]["coverageChunks"][0]["status"] == "alignment_failed"
    assert any("未隐藏失败" in warning for warning in payload["warnings"])
    track = client.get(f"/api/tracks/{track_id}").json()
    fallback_hits = [
        hit
        for hit in track["hitPoints"]
        if "vocal_acoustic_onset" in hit["detectorVotes"]
    ]
    assert fallback_hits
    assert all(hit["sample"] == hit["refinedSample"] for hit in fallback_hits)
    assert all(hit["sample"] != hit["snappedSample"] for hit in fallback_hits)


def test_vocal_fallback_uses_real_beat_aligned_energy_when_onset_detector_is_empty() -> None:
    sample_rate = 8_000
    samples = np.arange(sample_rate * 2, dtype=np.float32)
    envelope = np.zeros(samples.size, dtype=np.float32)
    for onset in (2_000, 6_000, 10_000):
        envelope[onset : onset + 800] = np.linspace(0.0, 1.0, 800, dtype=np.float32)
        envelope[onset + 800 : onset + 1_600] = np.linspace(
            1.0, 0.0, 800, dtype=np.float32
        )
    audio = 0.2 * np.sin(2 * np.pi * 220 * samples / sample_rate) * envelope

    anchors = _detect_vocal_fallback_anchors(
        audio,
        sample_rate=sample_rate,
        target_sample_rate=sample_rate,
        bpm=120.0,
        beat_offset_sample=0,
        coverage_chunks=[
            {
                "index": 0,
                "startSample": 0,
                "endSample": audio.size,
                "status": "alignment_failed",
            }
        ],
        acoustic_candidates=[],
    )

    assert anchors
    assert all(anchor["fallback_level"] == "beat_aligned_vocal_energy_peak" for anchor in anchors)
    assert all(anchor["activity_score"] > 0 for anchor in anchors)
    assert all(anchor["attack_score"] > 0 or anchor["rise_score"] > 0 for anchor in anchors)


def test_lrc_cleanup_keeps_only_lyric_phrases() -> None:
    assert clean_lyrics("[ar:Test]\n[00:01.20]ひかり\n[00:02:03.4]ひらく", "lrc") == (
        "ひかり\nひらく"
    )


def test_collapsed_timestamps_are_rejected_instead_of_expanded_to_morae() -> None:
    sample_rate = 8_000
    samples = np.arange(sample_rate, dtype=np.float32)
    audio = 0.1 * np.sin(2 * np.pi * 220 * samples / sample_rate)
    result = _build_anchors(
        [
            {
                "text": "飽き",
                "kana": "アキ",
                "romaji": "aki",
                "start_sample": 1_000,
                "end_sample": 1_001,
            },
            {
                "text": "た",
                "kana": "タ",
                "romaji": "ta",
                "start_sample": 1_000,
                "end_sample": 1_001,
            },
        ],
        audio=audio,
        sample_rate=sample_rate,
        bpm=120.0,
        beat_offset_sample=0,
    )

    assert result.anchors == []
    assert result.statistics["rejectedShortTimestampCount"] == 2


def test_word_timestamp_produces_one_chart_anchor_not_one_per_mora() -> None:
    sample_rate = 8_000
    samples = np.arange(sample_rate, dtype=np.float32)
    envelope = np.zeros(sample_rate, dtype=np.float32)
    envelope[800:2_400] = np.linspace(0.0, 1.0, 1_600, dtype=np.float32)
    audio = 0.18 * np.sin(2 * np.pi * 220 * samples / sample_rate) * envelope

    result = _build_anchors(
        [{
            "text": "飽き",
            "kana": "アキ",
            "romaji": "aki",
            "start_sample": 760,
            "end_sample": 2_200,
        }],
        audio=audio,
        sample_rate=sample_rate,
        bpm=120.0,
        beat_offset_sample=0,
    )

    assert len(result.anchors) == 1
    assert result.anchors[0]["original_text"] == "飽き"
    assert result.anchors[0]["word_start"] is True


def test_silent_vocal_timestamp_cannot_become_a_chart_anchor() -> None:
    sample_rate = 8_000
    audio = np.full(sample_rate, 1e-5, dtype=np.float32)

    result = _build_anchors(
        [{
            "text": "声",
            "kana": "コエ",
            "romaji": "koe",
            "start_sample": 800,
            "end_sample": 2_400,
        }],
        audio=audio,
        sample_rate=sample_rate,
        bpm=120.0,
        beat_offset_sample=0,
    )

    assert result.anchors == []
    assert result.statistics["rejectedSilentCount"] == 1


def test_three_millisecond_leakage_click_cannot_become_a_vocal_anchor() -> None:
    sample_rate = 8_000
    audio = np.zeros(sample_rate, dtype=np.float32)
    audio[1_000:1_024] = np.hanning(24).astype(np.float32) * 0.3

    result = _build_anchors(
        [{
            "text": "声",
            "start_sample": 960,
            "end_sample": 1_400,
            "chunk_match_confidence": 0.9,
        }],
        audio=audio,
        sample_rate=sample_rate,
        bpm=120.0,
        beat_offset_sample=0,
    )

    assert result.anchors == []
    assert result.statistics["rejectedSilentCount"] == 1


def test_focus_map_gaps_fail_closed_after_valid_segments() -> None:
    track = TrackModel(
        original_file_name="voice.wav",
        stored_file_name="voice.wav",
        format="wav",
        original_sample_rate=8_000,
        channels=1,
        sample_count=8_000,
        duration_sec=1.0,
        analysis_json=json.dumps(
            {
                "focusMap": [
                    {"startSample": 0, "endSample": 2_000, "focusSource": "vocals"},
                    {"startSample": 4_000, "endSample": 6_000, "focusSource": "drums"},
                ]
            }
        ),
    )

    assert _focus_supports_vocals(track, 1_000) is True
    assert _focus_supports_vocals(track, 3_000) is False
    assert _focus_supports_vocals(track, 5_000) is False


def test_vocal_hit_does_not_duplicate_an_existing_sixteenth_cell() -> None:
    track = TrackModel(
        original_file_name="voice.wav",
        stored_file_name="voice.wav",
        format="wav",
        original_sample_rate=8_000,
        channels=1,
        sample_count=8_000,
        duration_sec=1.0,
        analysis_json="{}",
    )
    track.hit_points.append(
        HitPointModel(
            sample=850,
            detected_sample=850,
            refined_sample=850,
            snapped_sample=1_000,
            snap_error_ms=-18.75,
            band="mid_hit",
            confidence=0.8,
            salience=0.8,
            source="stems",
            detector_votes_json="[]",
            primary_stem="other",
            stem_evidence_json="{}",
        )
    )

    created = _replace_vocal_hits(
        track,
        [{
            "id": "anchor",
            "aligned_sample": 1_080,
            "refined_sample": 1_100,
            "grid_sample": 1_000,
            "confidence": 0.8,
            "activity_score": 0.8,
            "attack_score": 0.8,
            "active": True,
            "chart_candidate": True,
        }],
    )

    assert created == 0
    assert len(track.hit_points) == 1


def test_chunk_coverage_distinguishes_success_collapse_and_failure() -> None:
    coverage = _build_chunk_coverage(
        [
            {
                "index": 0,
                "startSample": 0,
                "endSample": 1_000,
                "status": "ok",
                "alignmentStatus": "ok",
                "matchConfidence": 0.9,
            },
            {
                "index": 1,
                "startSample": 1_000,
                "endSample": 2_000,
                "status": "ok",
                "alignmentStatus": "ok",
                "matchConfidence": 0.8,
            },
            {
                "index": 2,
                "startSample": 2_000,
                "endSample": 3_000,
                "status": "ok",
                "alignmentStatus": "failed",
                "matchConfidence": 0.7,
            },
        ],
        [
            {"chunk_index": 0},
            {"chunk_index": 0},
            {"chunk_index": 1},
            {"chunk_index": 2},
        ],
        [
            {"chunk_index": 0, "active": True, "chart_candidate": True, "confidence": 0.8},
            {"chunk_index": 0, "active": True, "chart_candidate": True, "confidence": 0.7},
        ],
        source_sample_rate=8_000,
        target_sample_rate=48_000,
    )

    assert [chunk["status"] for chunk in coverage] == [
        "success",
        "alignment_collapse",
        "alignment_failed",
    ]
    assert coverage[0]["startSample"] == 0
    assert coverage[0]["endSample"] == 6_000
    assert coverage[0]["anchorCount"] == 2


def test_vocal_replacement_preserves_fallback_outside_successful_chunks() -> None:
    track = TrackModel(
        id="track",
        original_file_name="voice.wav",
        stored_file_name="voice.wav",
        format="wav",
        original_sample_rate=8_000,
        channels=1,
        sample_count=8_000,
        duration_sec=1.0,
        analysis_json="{}",
    )

    def vocal_hit(hit_id: str, sample: int, *, manually_edited: bool = False) -> HitPointModel:
        return HitPointModel(
            id=hit_id,
            sample=sample,
            detected_sample=sample,
            refined_sample=sample,
            snapped_sample=sample,
            snap_error_ms=0.0,
            band="mid_hit",
            confidence=0.7,
            salience=0.7,
            source="stems",
            detector_votes_json="[]",
            primary_stem="vocals",
            stem_evidence_json="{}",
            manually_edited=manually_edited,
            locked=False,
        )

    track.hit_points.extend(
        [
            vocal_hit("covered-fallback", 800),
            vocal_hit("failed-chunk-fallback", 3_000),
            vocal_hit("empty-chunk-fallback", 5_000),
            vocal_hit("manual-covered", 1_500, manually_edited=True),
        ]
    )
    coverage = [
        {
            "index": 0,
            "startSample": 0,
            "endSample": 2_000,
            "status": "success",
            "confidence": 0.8,
            "anchorCount": 2,
        },
        {
            "index": 1,
            "startSample": 2_000,
            "endSample": 4_000,
            "status": "alignment_failed",
            "confidence": 0.0,
            "anchorCount": 0,
        },
        {
            "index": 2,
            "startSample": 4_000,
            "endSample": 6_000,
            "status": "insufficient_anchors",
            "confidence": 0.6,
            "anchorCount": 1,
        },
    ]
    anchors = [
        {
            "id": "new-0-a",
            "chunk_index": 0,
            "aligned_sample": 880,
            "refined_sample": 900,
            "grid_sample": 1_000,
            "confidence": 0.8,
            "activity_score": 0.8,
            "attack_score": 0.8,
            "active": True,
            "chart_candidate": True,
        },
        {
            "id": "new-0-b",
            "chunk_index": 0,
            "aligned_sample": 1_680,
            "refined_sample": 1_700,
            "grid_sample": 2_000,
            "confidence": 0.8,
            "activity_score": 0.8,
            "attack_score": 0.8,
            "active": True,
            "chart_candidate": True,
        },
        {
            "id": "failed-anchor",
            "chunk_index": 1,
            "aligned_sample": 2_880,
            "refined_sample": 2_900,
            "grid_sample": 3_000,
            "confidence": 0.8,
            "activity_score": 0.8,
            "attack_score": 0.8,
            "active": True,
            "chart_candidate": True,
        },
    ]

    replacement = _replace_vocal_hits_with_stats(track, anchors, coverage)

    ids = {hit.id for hit in track.hit_points}
    assert "covered-fallback" not in ids
    assert {"failed-chunk-fallback", "empty-chunk-fallback", "manual-covered"} <= ids
    assert replacement.created_hit_count == 2
    assert replacement.removed_hit_count == 1
    assert replacement.replaced_chunk_count == 1
    assert replacement.preserved_fallback_hit_count == 2


def test_vocal_anchor_samples_map_back_to_original_sample_rate() -> None:
    source_rate = 8_000
    original_rate = 48_000
    samples = np.arange(source_rate, dtype=np.float32)
    envelope = np.zeros(source_rate, dtype=np.float32)
    envelope[800:2_400] = np.linspace(0.0, 1.0, 1_600, dtype=np.float32)
    audio = 0.18 * np.sin(2 * np.pi * 220 * samples / source_rate) * envelope

    result = _build_anchors(
        [{
            "text": "声",
            "kana": "コエ",
            "romaji": "koe",
            "start_sample": 760,
            "end_sample": 2_200,
        }],
        audio=audio,
        sample_rate=source_rate,
        original_sample_rate=original_rate,
        bpm=120.0,
        beat_offset_sample=0,
    )

    assert len(result.anchors) == 1
    assert result.anchors[0]["aligned_sample"] == 4_560
    assert result.anchors[0]["refined_sample"] > result.anchors[0]["aligned_sample"]
    assert result.statistics["originalSampleRate"] == original_rate
