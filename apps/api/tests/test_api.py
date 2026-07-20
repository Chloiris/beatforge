from __future__ import annotations

import io
import json
import math
import struct
import time
import wave
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from beatforge_api.config import get_settings
from beatforge_api.database import SessionLocal
from beatforge_api.jobs import _run_analysis
from beatforge_api.models import (
    AnalysisJobModel,
    CandidateEventModel,
    HitPointModel,
    TempoSegmentModel,
    TrackModel,
)
from beatforge_api.serialization import dumps
from beatforge_api.storage_paths import storage_relative_path
from beatforge_api.waveform_store import write_waveform_lods


def wav_bytes(sample_rate: int = 8_000, duration: float = 1.0) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        samples = []
        for index in range(round(sample_rate * duration)):
            value = 0.2 * math.sin(2 * math.pi * 220 * index / sample_rate)
            if index in {800, 2_400, 4_000, 5_600}:
                value = 0.95
            samples.append(struct.pack("<h", int(max(-1, min(1, value)) * 32767)))
        output.writeframes(b"".join(samples))
    return buffer.getvalue()


def upload(client: TestClient) -> dict:
    response = client.post(
        "/api/tracks/upload",
        data={"title": "API 测试曲", "artist": "Test Lab", "genre": "Synthetic"},
        files={"file": ("../unsafe.wav", wav_bytes(), "audio/wav")},
    )
    assert response.status_code == 201, response.text
    return response.json()


def attach_test_stem_assets(track_id: str) -> Path:
    """Persist deterministic waveform/stem fixtures without running source separation."""
    settings = get_settings()
    waveform_path = settings.waveform_dir / f"{track_id}.json.gz"
    write_waveform_lods(
        waveform_path,
        {
            "trackId": track_id,
            "sampleRate": 8_000,
            "sampleCount": 8_000,
            "levels": [
                {
                    "level": 0,
                    "window_size": 4_000,
                    "mins": [-0.25, -0.5],
                    "maxs": [0.5, 0.75],
                }
            ],
            "stems": {
                "vocals": {
                    "sampleRate": 8_000,
                    "sampleCount": 8_000,
                    "levels": [
                        {
                            "level": 0,
                            "window_size": 4_000,
                            "mins": [-0.1, -0.2],
                            "maxs": [0.2, 0.3],
                        }
                    ],
                }
            },
        },
    )
    stem_path = settings.stems_dir / track_id / "vocals.flac"
    stem_path.parent.mkdir(parents=True, exist_ok=True)
    phase = np.arange(800, dtype=np.float32)
    samples = (0.15 * np.sin(2 * np.pi * 220 * phase / 8_000)).astype(np.float32)
    sf.write(stem_path, samples, 8_000, format="FLAC", subtype="PCM_16")
    with SessionLocal() as session:
        track = session.get(TrackModel, track_id)
        assert track is not None
        track.waveform_path = storage_relative_path(waveform_path, settings)
        session.commit()
    return stem_path


def test_health_and_empty_projects(client: TestClient) -> None:
    assert client.get("/api/health").json()["status"] == "ok"
    assert client.get("/api/projects").json() == {"items": [], "total": 0}


def test_upload_validation_and_safe_name(client: TestClient) -> None:
    rejected = client.post(
        "/api/tracks/upload",
        files={"file": ("malware.exe", b"MZ", "application/octet-stream")},
    )
    assert rejected.status_code == 415
    assert rejected.json()["error"]["code"] == "UNSUPPORTED_AUDIO_FORMAT"

    payload = upload(client)
    assert payload["track"]["originalFileName"] == "unsafe.wav"
    assert payload["track"]["originalSampleRate"] == 8_000
    assert payload["track"]["sampleCount"] == 8_000
    projects = client.get("/api/projects").json()
    assert projects["total"] == 1
    assert projects["items"][0]["trackId"] == payload["track"]["id"]


def test_audio_range_request(client: TestClient) -> None:
    uploaded = upload(client)
    track_id = uploaded["track"]["id"]
    with SessionLocal() as session:
        track = session.get(TrackModel, track_id)
        assert track is not None
        assert not Path(track.file_path).is_absolute()
    response = client.get(f"/api/tracks/{track_id}/audio", headers={"Range": "bytes=0-99"})
    assert response.status_code == 206
    assert len(response.content) == 100
    assert response.headers["content-range"].startswith("bytes 0-99/")
    assert response.headers["accept-ranges"] == "bytes"


def test_audio_range_rebases_legacy_container_path(client: TestClient) -> None:
    track_id = upload(client)["track"]["id"]
    with SessionLocal() as session:
        track = session.get(TrackModel, track_id)
        assert track is not None
        track.file_path = f"/app/storage/audio/{track.stored_file_name}"
        session.commit()

    response = client.get(f"/api/tracks/{track_id}/audio", headers={"Range": "bytes=0-99"})
    assert response.status_code == 206
    assert len(response.content) == 100


def test_waveform_source_selects_independent_mix_and_stem_peaks(
    client: TestClient,
) -> None:
    track_id = upload(client)["track"]["id"]
    attach_test_stem_assets(track_id)

    mix = client.get(f"/api/tracks/{track_id}/waveform?source=mix&level=0")
    assert mix.status_code == 200, mix.text
    assert mix.json() == {
        "trackId": track_id,
        "sampleRate": 8_000,
        "sampleCount": 8_000,
        "source": "mix",
        "level": 0,
        "windowSize": 4_000,
        "mins": [-0.25, -0.5],
        "maxs": [0.5, 0.75],
    }

    vocals = client.get(f"/api/tracks/{track_id}/waveform?source=vocals&level=0")
    assert vocals.status_code == 200, vocals.text
    assert vocals.json()["source"] == "vocals"
    assert vocals.json()["mins"] == [-0.1, -0.2]
    assert vocals.json()["maxs"] == [0.2, 0.3]


def test_waveform_source_errors_are_structured(client: TestClient) -> None:
    track_id = upload(client)["track"]["id"]

    not_analyzed = client.get(f"/api/tracks/{track_id}/waveform?source=vocals")
    assert not_analyzed.status_code == 409
    assert not_analyzed.json()["error"]["code"] == "WAVEFORM_NOT_READY"

    attach_test_stem_assets(track_id)
    missing_stem = client.get(f"/api/tracks/{track_id}/waveform?source=drums")
    assert missing_stem.status_code == 404
    assert missing_stem.json()["error"] == {
        "code": "STEM_WAVEFORM_NOT_READY",
        "message": "Waveform peaks for the requested source are unavailable",
        "details": None,
    }

    invalid_source = client.get(f"/api/tracks/{track_id}/waveform?source=piano")
    assert invalid_source.status_code == 422
    assert invalid_source.json()["error"]["code"] == "VALIDATION_ERROR"


def test_stem_audio_get_head_and_range(client: TestClient) -> None:
    track_id = upload(client)["track"]["id"]
    stem_path = attach_test_stem_assets(track_id)
    expected = stem_path.read_bytes()

    full = client.get(f"/api/tracks/{track_id}/stems/vocals/audio")
    assert full.status_code == 200, full.text
    assert full.content == expected
    assert full.headers["content-type"].startswith("audio/flac")
    assert full.headers["accept-ranges"] == "bytes"
    assert int(full.headers["content-length"]) == len(expected)

    head = client.head(f"/api/tracks/{track_id}/stems/vocals/audio")
    assert head.status_code == 200
    assert head.content == b""
    assert head.headers["content-type"].startswith("audio/flac")
    assert head.headers["accept-ranges"] == "bytes"
    assert int(head.headers["content-length"]) == len(expected)

    partial = client.get(
        f"/api/tracks/{track_id}/stems/vocals/audio",
        headers={"Range": "bytes=8-39"},
    )
    assert partial.status_code == 206, partial.text
    assert partial.content == expected[8:40]
    assert partial.headers["content-range"] == f"bytes 8-39/{len(expected)}"
    assert partial.headers["content-length"] == "32"


def test_stem_audio_errors_are_structured(client: TestClient) -> None:
    track_id = upload(client)["track"]["id"]
    stem_path = attach_test_stem_assets(track_id)

    invalid_source = client.get(f"/api/tracks/{track_id}/stems/piano/audio")
    assert invalid_source.status_code == 404
    assert invalid_source.json()["error"]["code"] == "INVALID_STEM"

    missing_stem = client.get(f"/api/tracks/{track_id}/stems/drums/audio")
    assert missing_stem.status_code == 404
    assert missing_stem.json()["error"] == {
        "code": "STEM_NOT_READY",
        "message": "This track has no persisted audio for the requested source",
        "details": None,
    }

    invalid_range = client.get(
        f"/api/tracks/{track_id}/stems/vocals/audio",
        headers={"Range": "bytes=999999-1000000"},
    )
    assert invalid_range.status_code == 416
    assert invalid_range.json()["error"]["code"] == "INVALID_RANGE"
    assert invalid_range.json()["error"]["details"] == {"size": stem_path.stat().st_size}
    assert invalid_range.headers["content-range"] == f"bytes */{stem_path.stat().st_size}"


def test_hit_point_crud_tempo_and_export(client: TestClient) -> None:
    uploaded = upload(client)
    track_id = uploaded["track"]["id"]
    created = client.post(
        f"/api/tracks/{track_id}/hit-points",
        json={"sample": 800, "band": "low_hit", "source": "manual"},
    )
    assert created.status_code == 201, created.text
    hit = created.json()
    assert hit["sample"] == 800
    assert hit["acousticSample"] == 800
    assert hit["chartSample"] == 800
    assert hit["timeSec"] == 0.1
    assert hit["manuallyEdited"] is True

    moved = client.patch(
        f"/api/tracks/{track_id}/hit-points/{hit['id']}", json={"sample": 801}
    )
    assert moved.status_code == 200
    assert moved.json()["sample"] == 801
    assert moved.json()["acousticSample"] == 801

    with SessionLocal() as session:
        track = session.get(TrackModel, track_id)
        assert track is not None
        track.candidate_events.append(
            CandidateEventModel(
                id="candidate-tempo-test",
                hit_point_id=hit["id"],
                sample=801,
                acoustic_sample=801,
                chart_sample=801,
                snap_error_ms=0.0,
                lane="mix",
                source_evidence_json=dumps({"mix": 1.0}),
                semantic_evidence_json=dumps({"beatConfidence": 1.0}),
                confidence=0.8,
                status="accepted",
                grid_type="straight_1_16",
                grid_confidence=1.0,
            )
        )
        session.commit()

    tempo = client.patch(
        f"/api/tracks/{track_id}/tempo-map",
        json={
            "tempoMap": [
                {
                    "startSample": 0,
                    "bpm": 128.5,
                    "beatOffsetSample": 123,
                    "timeSignatureNumerator": 4,
                    "timeSignatureDenominator": 4,
                }
            ]
        },
    )
    assert tempo.status_code == 200, tempo.text
    assert tempo.json()[0]["beatOffsetSample"] == 123
    candidate = client.get(f"/api/tracks/{track_id}/candidate-events").json()[0]
    assert candidate["acousticSample"] == 801
    assert candidate["chartSample"] != 801
    assert candidate["semanticEvidence"]["beatConfidence"] < 1.0
    summary = client.get("/api/projects").json()["items"][0]
    assert summary["bpm"] == 128.5
    assert summary["durationSec"] == 1.0
    assert summary["hitPointCount"] == 1

    json_export = client.get(f"/api/tracks/{track_id}/export?format=json")
    assert json_export.status_code == 200
    exported = json_export.json()
    assert exported["schemaVersion"] == "1.0"
    assert exported["audio"]["sampleRate"] == 8_000
    assert exported["hitPoints"][0]["sample"] == 801
    assert exported["hitPoints"][0]["acousticSample"] == 801
    assert "chartSample" in exported["hitPoints"][0]
    assert exported["candidateEvents"] == [candidate]

    csv_export = client.get(f"/api/tracks/{track_id}/export?format=csv")
    assert csv_export.status_code == 200
    assert "sample,acoustic_sample,chart_sample,time_sec" in csv_export.text

    deleted = client.delete(f"/api/tracks/{track_id}/hit-points/{hit['id']}")
    assert deleted.status_code == 204
    assert client.get(f"/api/tracks/{track_id}/hit-points").json() == []


def test_bulk_save_preserves_ids_and_rejects_out_of_range(client: TestClient) -> None:
    track_id = upload(client)["track"]["id"]
    first = client.post(
        f"/api/tracks/{track_id}/hit-points", json={"sample": 100, "band": "mid_hit"}
    ).json()
    response = client.put(
        f"/api/tracks/{track_id}/hit-points",
        json={
            "hitPoints": [
                {**first, "sample": 101},
                {"sample": 200, "band": "high_hit", "source": "manual"},
            ]
        },
    )
    assert response.status_code == 200, response.text
    assert len(response.json()) == 2
    assert response.json()[0]["id"] == first["id"]

    invalid = client.post(
        f"/api/tracks/{track_id}/hit-points", json={"sample": 8_000}
    )
    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "SAMPLE_OUT_OF_RANGE"


def test_analysis_job_reaches_terminal_persisted_state(client: TestClient) -> None:
    track_id = upload(client)["track"]["id"]
    response = client.post(
        f"/api/tracks/{track_id}/analyze", json={"mode": "balanced", "sensitivity": 0.5}
    )
    assert response.status_code == 202, response.text
    job_id = response.json()["jobId"]
    deadline = time.monotonic() + 60
    job = None
    while time.monotonic() < deadline:
        job_response = client.get(f"/api/analysis-jobs/{job_id}")
        assert job_response.status_code == 200
        job = job_response.json()
        if job["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)
    assert job is not None
    assert job["status"] == "completed", job
    assert job["progress"] == 1.0
    track = client.get(f"/api/tracks/{track_id}").json()
    assert track["analysis"]["version"]
    assert track["analysis"]["mode"] == "balanced"
    assert "elapsedMs" in track["analysis"]
    assert "bpmConfidence" in track["analysis"]
    assert track["tempoMap"]
    assert len(track["candidateEvents"]) == len(track["hitPoints"])
    assert all(
        candidate["acousticSample"] == candidate["sample"]
        for candidate in track["candidateEvents"]
    )
    candidate_response = client.get(f"/api/tracks/{track_id}/candidate-events")
    assert candidate_response.status_code == 200
    assert candidate_response.json() == track["candidateEvents"]
    summary = client.get("/api/projects").json()["items"][0]
    assert summary["analysisMode"] == "balanced"
    assert summary["hitPointCount"] == len(track["hitPoints"])
    waveform = client.get(f"/api/tracks/{track_id}/waveform?maxPoints=1000")
    assert waveform.status_code == 200, waveform.text
    assert len(waveform.json()["mins"]) == len(waveform.json()["maxs"])


def test_reanalysis_preserves_user_edits_and_avoids_near_duplicates(
    client: TestClient, monkeypatch
) -> None:
    track_id = upload(client)["track"]["id"]
    preserved_ids = {
        "edited": "00000000-0000-0000-0000-000000000101",
        "locked": "00000000-0000-0000-0000-000000000102",
        "manual": "00000000-0000-0000-0000-000000000103",
    }
    manual_tempo_id = "00000000-0000-0000-0000-000000000201"
    job_id = "00000000-0000-0000-0000-000000000301"

    def existing_hit(
        *,
        hit_id: str,
        sample: int,
        source: str = "fused",
        manually_edited: bool = False,
        locked: bool = False,
    ) -> HitPointModel:
        return HitPointModel(
            id=hit_id,
            sample=sample,
            detected_sample=sample - 2,
            refined_sample=sample - 1,
            snapped_sample=sample + 3,
            snap_error_ms=-0.375,
            band="mid_hit",
            confidence=0.7,
            salience=0.6,
            source=source,
            detector_votes_json=dumps(["existing"]),
            manually_edited=manually_edited,
            locked=locked,
        )

    with SessionLocal() as session:
        track = session.get(TrackModel, track_id)
        assert track is not None
        track.hit_points.extend(
            [
                existing_hit(
                    hit_id=preserved_ids["edited"],
                    sample=1_000,
                    manually_edited=True,
                ),
                existing_hit(
                    hit_id=preserved_ids["locked"], sample=2_000, locked=True
                ),
                existing_hit(
                    hit_id=preserved_ids["manual"], sample=3_000, source="manual"
                ),
                existing_hit(
                    hit_id="00000000-0000-0000-0000-000000000104",
                    sample=4_000,
                ),
            ]
        )
        track.tempo_segments.extend(
            [
                TempoSegmentModel(
                    id=manual_tempo_id,
                    start_sample=0,
                    bpm=137.5,
                    time_signature_numerator=7,
                    time_signature_denominator=8,
                    beat_offset_sample=321,
                    confidence=0.8,
                    manually_edited=True,
                ),
                TempoSegmentModel(
                    start_sample=4_000,
                    bpm=90.0,
                    beat_offset_sample=0,
                    confidence=0.2,
                    manually_edited=False,
                ),
            ]
        )
        session.add(
            AnalysisJobModel(
                id=job_id,
                track_id=track_id,
                mode="balanced",
                sensitivity=0.5,
            )
        )
        session.commit()

    def fake_analyze_audio(*_args, **_kwargs) -> dict:
        def hit(hit_id: str, sample: int) -> dict:
            return {
                "id": hit_id,
                "sample": sample,
                "detected_sample": sample,
                "refined_sample": sample,
                "band": "low_hit",
                "confidence": 0.9,
                "salience": 0.8,
                "source": "fused",
                "detector_votes": ["new"],
            }

        return {
            "original_sample_rate": 8_000,
            "sample_count": 8_000,
            "channels": 1,
            "duration_sec": 1.0,
            "leading_silence_samples": 0,
            "bpm": 150.0,
            "bpm_confidence": 0.95,
            "beat_offset_sample": 40,
            "warnings": [],
            "metadata": {"version": "test"},
            "stage_timings_ms": {},
            "hit_points": [
                hit("00000000-0000-0000-0000-000000000401", 1_068),
                hit("00000000-0000-0000-0000-000000000402", 1_073),
                hit("00000000-0000-0000-0000-000000000403", 2_005),
                hit("00000000-0000-0000-0000-000000000404", 3_070),
                hit("00000000-0000-0000-0000-000000000405", 4_500),
            ],
        }

    import beatforge_api.audio as audio_module

    monkeypatch.setattr(audio_module, "analyze_audio", fake_analyze_audio)
    monkeypatch.setattr(
        audio_module,
        "build_waveform_lods",
        lambda _path: [
            {"level": 0, "window_size": 8_000, "mins": [0.0], "maxs": [0.0]}
        ],
    )

    _run_analysis(job_id)

    with SessionLocal() as session:
        track = session.get(TrackModel, track_id)
        job = session.get(AnalysisJobModel, job_id)
        assert track is not None
        assert job is not None
        assert job.status == "completed"
        assert track.project.status == "edited"

        hits_by_id = {item.id: item for item in track.hit_points}
        assert set(preserved_ids.values()) <= hits_by_id.keys()
        assert hits_by_id[preserved_ids["edited"]].manually_edited is True
        assert hits_by_id[preserved_ids["locked"]].locked is True
        assert hits_by_id[preserved_ids["manual"]].source == "manual"
        assert sorted(item.sample for item in track.hit_points) == [
            1_000,
            1_073,
            2_000,
            3_000,
            4_500,
        ]

        assert len(track.tempo_segments) == 1
        tempo = track.tempo_segments[0]
        assert tempo.id == manual_tempo_id
        assert tempo.bpm == 137.5
        assert tempo.beat_offset_sample == 321
        assert tempo.time_signature_numerator == 7
        assert tempo.time_signature_denominator == 8
        assert tempo.manually_edited is True
        assert json.loads(track.analysis_json)["hitPointCount"] == 5
