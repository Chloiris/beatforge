from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from beatforge_api.config import get_settings
from beatforge_api.database import SessionLocal
from beatforge_api.models import TrackModel
from beatforge_api.storage_paths import resolve_storage_path

from .test_api import wav_bytes

STEMS = ("mix", "vocals", "drums", "bass", "other")
SEPARATED_STEMS = STEMS[1:]


def _upload_named(
    client: TestClient,
    filename: str = "package-source.wav",
    *,
    audio_bytes: bytes | None = None,
) -> str:
    response = client.post(
        "/api/tracks/upload",
        data={"title": "Package export test", "artist": "Test Lab"},
        files={
            "file": (
                filename,
                audio_bytes if audio_bytes is not None else wav_bytes(),
                "audio/wav",
            )
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["track"]["id"]


def _stereo_wav_bytes() -> bytes:
    phase = np.arange(8_000, dtype=np.float32)
    samples = np.column_stack(
        (
            0.1 * np.sin(2 * np.pi * 220.0 * phase / 8_000),
            0.1 * np.sin(2 * np.pi * 330.0 * phase / 8_000),
        )
    ).astype(np.float32)
    output = io.BytesIO()
    sf.write(output, samples, 8_000, format="WAV", subtype="PCM_16")
    return output.getvalue()


def _open_package(response) -> zipfile.ZipFile:
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("application/zip")
    return zipfile.ZipFile(io.BytesIO(response.content))


def _json_member(archive: zipfile.ZipFile, path: str) -> dict:
    with archive.open(path) as handle:
        return json.load(handle)


def _attach_all_stems(track_id: str) -> dict[str, Path]:
    settings = get_settings()
    stem_dir = settings.stems_dir / track_id
    stem_dir.mkdir(parents=True, exist_ok=True)
    phase = np.arange(8_000, dtype=np.float32)
    paths: dict[str, Path] = {}
    for index, stem in enumerate(SEPARATED_STEMS, start=1):
        path = stem_dir / f"{stem}.flac"
        frequency = 110.0 * index
        samples = (0.04 * index * np.sin(2 * np.pi * frequency * phase / 8_000)).astype(
            np.float32
        )
        sf.write(path, samples, 8_000, format="FLAC", subtype="PCM_16")
        paths[stem] = path
    return paths


def test_package_export_partitions_markers_and_preserves_both_timebases(
    client: TestClient,
) -> None:
    track_id = _upload_named(client)
    expected_ids: dict[str, str] = {}
    for index, stem in enumerate(STEMS):
        acoustic_sample = 880 + index * 1_000
        chart_sample = acoustic_sample - 80
        response = client.post(
            f"/api/tracks/{track_id}/hit-points",
            json={
                "sample": acoustic_sample,
                "acousticSample": acoustic_sample,
                "chartSample": chart_sample,
                "primaryStem": stem,
                "stemEvidence": {stem: 0.9},
                "source": "manual",
            },
        )
        assert response.status_code == 201, response.text
        expected_ids[stem] = response.json()["id"]

    response = client.get(f"/api/tracks/{track_id}/export?format=package&audio=none")
    with _open_package(response) as archive:
        assert set(archive.namelist()) == {
            "manifest.json",
            "analysis/candidates.json",
            *(f"markers/{stem}.json" for stem in STEMS),
        }

        manifest = _json_member(archive, "manifest.json")
        assert manifest["schemaVersion"] == "2.0"
        assert manifest["packageType"] == "beatforge.chart-package"
        assert manifest["timebase"] == "samples"
        assert manifest["sampleRate"] == 8_000
        assert manifest["sampleCount"] == 8_000
        assert manifest["channels"] == 1
        assert manifest["leadingSilenceSamples"] == 0
        assert manifest["timeOrigin"] == {
            "sample": 0,
            "timeSec": 0.0,
            "definition": "startOfDecodedReferenceAudio",
        }
        assert manifest["audio"]["mode"] == "none"
        assert manifest["audio"]["reference"] is None
        assert manifest["audio"]["stems"] == []
        assert manifest["audio"]["source"]["originalFileName"] == "package-source.wav"
        assert manifest["audio"]["source"]["format"] == "wav"
        assert len(manifest["audio"]["source"]["sha256"]) == 64
        assert {
            item["primaryStem"]: (item["path"], item["count"])
            for item in manifest["markerTracks"]
        } == {stem: (f"markers/{stem}.json", 1) for stem in STEMS}

        for index, stem in enumerate(STEMS):
            marker_file = _json_member(archive, f"markers/{stem}.json")
            assert marker_file["schemaVersion"] == "2.0"
            assert marker_file["primaryStem"] == stem
            assert marker_file["markerCount"] == 1
            marker = marker_file["markers"][0]
            acoustic_sample = 880 + index * 1_000
            chart_sample = acoustic_sample - 80
            assert marker["id"] == expected_ids[stem]
            assert marker["primaryStem"] == stem
            assert marker["origin"] == "manual"
            assert marker["source"] == "manual"
            assert marker["acousticSample"] == acoustic_sample
            assert marker["acousticTimeSec"] == pytest.approx(acoustic_sample / 8_000)
            assert marker["chartSample"] == chart_sample
            assert marker["chartTimeSec"] == pytest.approx(chart_sample / 8_000)
            assert marker["acousticMinusChartMs"] == pytest.approx(10.0)
            assert marker["chartTimingStatus"] == "suggested"
            assert marker["chartTimingSource"] == "gridReference"
            assert marker["manual"] is True
            assert marker["manuallyEdited"] is True
            assert marker["evidence"]["stemEvidence"] == {stem: 0.9}


def test_package_audio_modes_emit_verified_flac_reference_and_available_stems(
    client: TestClient,
) -> None:
    track_id = _upload_named(client)
    stem_paths = _attach_all_stems(track_id)

    default_response = client.get(f"/api/tracks/{track_id}/export?format=package")
    with _open_package(default_response) as archive:
        names = set(archive.namelist())
        assert "audio/reference.flac" in names
        assert not any(name.startswith("stems/") for name in names)
        reference_bytes = archive.read("audio/reference.flac")
        reference_info = sf.info(io.BytesIO(reference_bytes))
        assert reference_info.format == "FLAC"
        assert reference_info.samplerate == 8_000
        assert reference_info.frames == 8_000
        assert reference_info.channels == 1

        manifest = _json_member(archive, "manifest.json")
        assert manifest["audio"]["mode"] == "reference"
        reference = manifest["audio"]["reference"]
        assert reference["path"] == "audio/reference.flac"
        assert reference["format"] == "flac"
        assert reference["sampleRate"] == 8_000
        assert reference["sampleCount"] == 8_000
        assert reference["channels"] == 1
        assert reference["sha256"] == hashlib.sha256(reference_bytes).hexdigest()
        assert manifest["audio"]["stems"] == []

    full_response = client.get(f"/api/tracks/{track_id}/export?format=package&audio=full")
    with _open_package(full_response) as archive:
        names = set(archive.namelist())
        assert "audio/reference.flac" in names
        assert {f"stems/{stem}.flac" for stem in SEPARATED_STEMS}.issubset(names)
        manifest = _json_member(archive, "manifest.json")
        assert manifest["audio"]["mode"] == "full"
        descriptors = {
            Path(item["path"]).stem: item for item in manifest["audio"]["stems"]
        }
        assert set(descriptors) == set(SEPARATED_STEMS)
        for stem, source_path in stem_paths.items():
            member = f"stems/{stem}.flac"
            member_bytes = archive.read(member)
            assert member_bytes == source_path.read_bytes()
            assert descriptors[stem]["path"] == member
            assert descriptors[stem]["sha256"] == hashlib.sha256(member_bytes).hexdigest()
            assert descriptors[stem]["sampleRate"] == 8_000
            assert descriptors[stem]["sampleCount"] == 8_000


def test_full_package_rejects_a_stem_on_a_different_sample_timeline(
    client: TestClient,
) -> None:
    track_id = _upload_named(client)
    settings = get_settings()
    stem_path = settings.stems_dir / track_id / "vocals.flac"
    stem_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(
        stem_path,
        np.zeros(4_000, dtype=np.float32),
        16_000,
        format="FLAC",
        subtype="PCM_16",
    )

    response = client.get(f"/api/tracks/{track_id}/export?format=package&audio=full")
    assert response.status_code == 422, response.text
    error = response.json()["error"]
    assert error["code"] == "PACKAGE_AUDIO_TIMELINE_MISMATCH"
    assert error["details"] == {
        "path": "stems/vocals.flac",
        "expected": {"sampleRate": 8_000, "sampleCount": 8_000},
        "actual": {"sampleRate": 16_000, "sampleCount": 4_000},
        "channelPolicy": "independentStem",
        "reportedChannels": 1,
    }
    assert list(settings.storage_dir.glob("beatforge-package-*")) == []


def test_full_package_allows_mono_stems_for_a_stereo_reference(
    client: TestClient,
) -> None:
    track_id = _upload_named(client, audio_bytes=_stereo_wav_bytes())
    _attach_all_stems(track_id)

    response = client.get(f"/api/tracks/{track_id}/export?format=package&audio=full")
    with _open_package(response) as archive:
        manifest = _json_member(archive, "manifest.json")
        assert manifest["channels"] == 2
        assert manifest["audio"]["reference"]["channels"] == 2
        assert {
            descriptor["source"]: descriptor["channels"]
            for descriptor in manifest["audio"]["stems"]
        } == {stem: 1 for stem in SEPARATED_STEMS}
        assert sf.info(io.BytesIO(archive.read("audio/reference.flac"))).channels == 2


def test_data_only_package_survives_a_missing_original_audio(
    client: TestClient,
) -> None:
    track_id = _upload_named(client)
    settings = get_settings()
    with SessionLocal() as session:
        track = session.get(TrackModel, track_id)
        assert track is not None
        resolve_storage_path(track.file_path, settings).unlink()

    response = client.get(f"/api/tracks/{track_id}/export?format=package&audio=none")
    with _open_package(response) as archive:
        source = _json_member(archive, "manifest.json")["audio"]["source"]
        assert source["available"] is False
        assert source["sha256"] is None

    reference_response = client.get(f"/api/tracks/{track_id}/export?format=package")
    assert reference_response.status_code == 422
    assert (
        reference_response.json()["error"]["code"]
        == "REFERENCE_AUDIO_EXPORT_UNAVAILABLE"
    )


def test_exports_redact_machine_local_paths_from_analysis_metadata(
    client: TestClient,
) -> None:
    track_id = _upload_named(client)
    with SessionLocal() as session:
        track = session.get(TrackModel, track_id)
        assert track is not None
        track.analysis_json = json.dumps(
            {
                "vocalLyrics": {
                    "model": "/opt/beatforge-fixture/models/Qwen3-ForcedAligner-0.6B",
                    "python": r"Z:\beatforge-fixture\.venv\Scripts\python.exe",
                    "warning": "runtime failed under /home/fixture/project/cache",
                }
            }
        )
        session.commit()

    json_response = client.get(f"/api/tracks/{track_id}/export?format=json")
    assert json_response.status_code == 200
    json_bytes = json_response.content
    assert b"/home/fixture" not in json_bytes
    assert b"beatforge-fixture" not in json_bytes
    metadata = json_response.json()["analysisMetadata"]["vocalLyrics"]
    assert metadata["model"] == "<local-path>/Qwen3-ForcedAligner-0.6B"
    assert metadata["python"] == "<local-path>/python.exe"
    assert metadata["warning"] == "runtime failed under <local-path>"

    package_response = client.get(
        f"/api/tracks/{track_id}/export?format=package&audio=none"
    )
    with _open_package(package_response) as archive:
        manifest_bytes = archive.read("manifest.json")
        assert b"/home/fixture" not in manifest_bytes
        assert b"beatforge-fixture" not in manifest_bytes
        package_metadata = json.loads(manifest_bytes)["analysis"]["metadata"]["vocalLyrics"]
        assert package_metadata == metadata


@pytest.mark.parametrize(
    ("query", "suffix"),
    (("format=json", ".json"), ("format=csv", ".csv"), ("format=package&audio=none", ".zip")),
)
def test_export_content_disposition_supports_unicode_original_names(
    client: TestClient, query: str, suffix: str
) -> None:
    track_id = _upload_named(client, "合成音声・テスト.wav")
    response = client.get(f"/api/tracks/{track_id}/export?{query}")
    assert response.status_code == 200, response.text
    disposition = response.headers["content-disposition"]
    assert disposition.startswith("attachment;")
    assert "filename=" in disposition
    assert "filename*=UTF-8''" in disposition
    assert quote("合成音声・テスト") in disposition
    assert quote(suffix) in disposition or disposition.endswith(suffix)
