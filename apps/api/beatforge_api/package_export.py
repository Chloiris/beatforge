from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import soundfile as sf

from .config import Settings
from .errors import BeatForgeError
from .export_safety import sanitize_export_metadata
from .media import prepare_analysis_source, probe_audio
from .models import CandidateEventModel, HitPointModel, TrackModel
from .serialization import (
    camelize_keys,
    candidate_event_dict,
    json_safe,
    load_json,
    project_dict,
    tempo_dict,
)

PACKAGE_SCHEMA_VERSION = "2.0"
PACKAGE_TYPE = "beatforge.chart-package"
MARKER_STEMS = ("mix", "vocals", "drums", "bass", "other")
SEPARATED_STEMS = ("vocals", "drums", "bass", "other")


@dataclass(frozen=True)
class PackageExport:
    archive_path: Path
    temporary_root: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_stem(value: str | None) -> str:
    return value if value in MARKER_STEMS else "mix"


def _marker_payload(hit: HitPointModel, sample_rate: int) -> dict[str, Any]:
    acoustic_sample = (
        hit.acoustic_sample if hit.acoustic_sample is not None else hit.refined_sample
    )
    chart_sample = hit.chart_sample if hit.chart_sample is not None else hit.snapped_sample
    stem_evidence = load_json(hit.stem_evidence_json, {})
    detector_votes = load_json(hit.detector_votes_json, [])
    origin = hit.source or "fused"
    return {
        "id": hit.id,
        "primaryStem": _normalized_stem(hit.primary_stem),
        # ``source`` is retained for existing consumers. ``origin`` makes it
        # explicit that this value describes provenance, not marker-track ownership.
        "origin": origin,
        "source": origin,
        "band": hit.band,
        "acousticSample": acoustic_sample,
        "acousticTimeSec": acoustic_sample / sample_rate,
        "chartSample": chart_sample,
        "chartTimeSec": chart_sample / sample_rate,
        "acousticMinusChartMs": (acoustic_sample - chart_sample) * 1000.0 / sample_rate,
        # The current database does not persist whether chart timing was accepted
        # or independently edited, so exporting a stronger status would be false.
        "chartTimingStatus": "suggested",
        "chartTimingSource": "gridReference",
        "confidence": hit.confidence,
        "salience": hit.salience,
        "manual": origin == "manual",
        "manuallyEdited": hit.manually_edited,
        "locked": hit.locked,
        "evidence": {
            "detectorVotes": detector_votes if isinstance(detector_votes, list) else [],
            "stemEvidence": stem_evidence if isinstance(stem_evidence, dict) else {},
        },
    }


def _candidate_payload(
    candidate: CandidateEventModel, sample_rate: int
) -> dict[str, Any]:
    payload = camelize_keys(candidate_event_dict(candidate, sample_rate))
    acoustic_sample = candidate.acoustic_sample
    chart_sample = candidate.chart_sample
    payload.update(
        {
            "acousticSample": acoustic_sample,
            "acousticTimeSec": acoustic_sample / sample_rate,
            "chartSample": chart_sample,
            "chartTimeSec": chart_sample / sample_rate,
            "acousticMinusChartMs": (
                (acoustic_sample - chart_sample) * 1000.0 / sample_rate
            ),
            "chartTimingStatus": "suggested",
            "chartTimingSource": "gridReference",
        }
    )
    return payload


def _reference_error(
    message: str, *, details: dict[str, Any] | None = None
) -> BeatForgeError:
    return BeatForgeError(
        "REFERENCE_AUDIO_EXPORT_UNAVAILABLE",
        message,
        status_code=422,
        details=details,
    )


def _validate_timeline(
    *,
    sample_rate: int,
    sample_count: int,
    channels: int,
    track: TrackModel,
    stage: str,
) -> None:
    expected = {
        "sampleRate": track.original_sample_rate,
        "sampleCount": track.sample_count,
        "channels": track.channels,
    }
    actual = {
        "sampleRate": sample_rate,
        "sampleCount": sample_count,
        "channels": channels,
    }
    if actual != expected:
        raise _reference_error(
            "The decoded reference audio does not match the saved sample timeline",
            details={"stage": stage, "expected": expected, "actual": actual},
        )


def _create_reference_flac(
    track: TrackModel,
    source_path: Path,
    destination: Path,
    decoded_dir: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        decoded_path, _ = prepare_analysis_source(source_path, decoded_dir)
        with sf.SoundFile(str(decoded_path), mode="r") as source:
            _validate_timeline(
                sample_rate=int(source.samplerate),
                sample_count=int(source.frames),
                channels=int(source.channels),
                track=track,
                stage="decoded-source",
            )
            with sf.SoundFile(
                str(destination),
                mode="w",
                samplerate=track.original_sample_rate,
                channels=track.channels,
                format="FLAC",
                subtype="PCM_24",
            ) as output:
                while True:
                    block = source.read(65_536, dtype="float32", always_2d=True)
                    if len(block) == 0:
                        break
                    output.write(block)

        info = sf.info(str(destination))
        _validate_timeline(
            sample_rate=int(info.samplerate),
            sample_count=int(info.frames),
            channels=int(info.channels),
            track=track,
            stage="reference-flac",
        )
    except BeatForgeError as exc:
        destination.unlink(missing_ok=True)
        if exc.code == "REFERENCE_AUDIO_EXPORT_UNAVAILABLE":
            raise
        raise _reference_error(
            "The source audio could not be decoded for package export",
            details={"cause": exc.code, "causeDetails": exc.details},
        ) from exc
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        destination.unlink(missing_ok=True)
        raise _reference_error(
            "A canonical FLAC reference could not be created",
            details={"reason": str(exc)[-500:]},
        ) from exc


def _audio_descriptor(
    path: Path,
    archive_path: str,
    *,
    expected_track: TrackModel | None = None,
    require_track_channels: bool = False,
) -> dict[str, Any]:
    try:
        probe = probe_audio(path)
    except BeatForgeError as exc:
        raise BeatForgeError(
            "PACKAGE_AUDIO_INVALID",
            "An audio asset selected for package export is not readable",
            status_code=422,
            details={"path": archive_path, "cause": exc.code},
        ) from exc
    if expected_track is not None:
        expected = {
            "sampleRate": expected_track.original_sample_rate,
            "sampleCount": expected_track.sample_count,
        }
        actual = {
            "sampleRate": probe.sample_rate,
            "sampleCount": probe.sample_count,
        }
        if require_track_channels:
            expected["channels"] = expected_track.channels
            actual["channels"] = probe.channels
        if actual != expected:
            raise BeatForgeError(
                "PACKAGE_AUDIO_TIMELINE_MISMATCH",
                "An audio asset does not match the package sample timeline",
                status_code=422,
                details={
                    "path": archive_path,
                    "expected": expected,
                    "actual": actual,
                    "channelPolicy": (
                        "matchReference" if require_track_channels else "independentStem"
                    ),
                    "reportedChannels": probe.channels,
                },
            )
    return {
        "path": archive_path,
        "format": probe.format,
        "sampleRate": probe.sample_rate,
        "sampleCount": probe.sample_count,
        "channels": probe.channels,
        "durationSec": probe.duration_sec,
        "sha256": _sha256(path),
    }


def _available_stems(track: TrackModel, settings: Settings) -> list[tuple[str, Path]]:
    stem_root = settings.stems_dir.resolve()
    result: list[tuple[str, Path]] = []
    for stem in SEPARATED_STEMS:
        path = (stem_root / track.id / f"{stem}.flac").resolve()
        if path.is_relative_to(stem_root) and path.is_file():
            result.append((stem, path))
    return result


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(
        json_safe(payload), ensure_ascii=False, indent=2, separators=(",", ": ")
    ).encode("utf-8")


def build_package_export(
    track: TrackModel,
    source_path: Path | None,
    settings: Settings,
    *,
    audio_mode: str = "reference",
) -> PackageExport:
    if audio_mode not in {"none", "reference", "full"}:
        raise BeatForgeError(
            "INVALID_EXPORT_AUDIO_MODE",
            "Package audio must be none, reference or full",
            status_code=422,
            details={"audio": audio_mode},
        )

    source_available = source_path is not None and source_path.is_file()
    if audio_mode in {"reference", "full"} and not source_available:
        raise _reference_error(
            "The original audio is required to create a canonical package reference",
            details={"originalFileName": track.original_file_name},
        )
    try:
        source_sha256 = _sha256(source_path) if source_available and source_path else None
    except OSError as exc:
        if audio_mode in {"reference", "full"}:
            raise _reference_error(
                "The original audio could not be read for package export",
                details={"reason": str(exc)[-500:]},
            ) from exc
        source_available = False
        source_sha256 = None

    settings.ensure_directories()
    temporary_root = Path(
        tempfile.mkdtemp(prefix="beatforge-package-", dir=settings.storage_dir)
    )
    archive_path = temporary_root / "export.beatforge.zip"
    try:
        markers_by_stem: dict[str, list[dict[str, Any]]] = {
            stem: [] for stem in MARKER_STEMS
        }
        for hit in track.hit_points:
            marker = _marker_payload(hit, track.original_sample_rate)
            markers_by_stem[marker["primaryStem"]].append(marker)
        for markers in markers_by_stem.values():
            markers.sort(
                key=lambda marker: (
                    marker["chartSample"],
                    marker["acousticSample"],
                    marker["id"],
                )
            )

        candidates = [
            _candidate_payload(candidate, track.original_sample_rate)
            for candidate in track.candidate_events
        ]
        candidates.sort(key=lambda candidate: (candidate["acousticSample"], candidate["id"]))

        reference_path: Path | None = None
        reference_descriptor: dict[str, Any] | None = None
        if audio_mode in {"reference", "full"}:
            assert source_path is not None
            reference_path = temporary_root / "reference.flac"
            _create_reference_flac(
                track,
                source_path,
                reference_path,
                temporary_root / "decoded",
            )
            reference_descriptor = {
                **_audio_descriptor(
                    reference_path,
                    "audio/reference.flac",
                    expected_track=track,
                    require_track_channels=True,
                ),
                "role": "canonicalReference",
            }

        stem_assets: list[tuple[str, Path, dict[str, Any]]] = []
        if audio_mode == "full":
            for stem, path in _available_stems(track, settings):
                archive_name = f"stems/{stem}.flac"
                stem_assets.append(
                    (
                        stem,
                        path,
                        {
                            **_audio_descriptor(
                                path,
                                archive_name,
                                expected_track=track,
                            ),
                            "source": stem,
                            "role": "separatedStem",
                        },
                    )
                )

        marker_descriptors = [
            {
                "primaryStem": stem,
                "path": f"markers/{stem}.json",
                "count": len(markers_by_stem[stem]),
            }
            for stem in MARKER_STEMS
        ]
        duration_sec = track.sample_count / track.original_sample_rate
        manifest = {
            "schemaVersion": PACKAGE_SCHEMA_VERSION,
            "packageType": PACKAGE_TYPE,
            "generatedAt": datetime.now(UTC),
            "project": camelize_keys(project_dict(track.project)),
            "originalFileName": track.original_file_name,
            "timebase": "samples",
            "samplesAreAuthoritative": True,
            "sampleRate": track.original_sample_rate,
            "sampleCount": track.sample_count,
            "channels": track.channels,
            "durationSec": duration_sec,
            "leadingSilenceSamples": track.leading_silence_samples,
            "timeOrigin": {
                "sample": 0,
                "timeSec": 0.0,
                "definition": "startOfDecodedReferenceAudio",
            },
            "tempoMap": [camelize_keys(tempo_dict(item)) for item in track.tempo_segments],
            "markerTracks": marker_descriptors,
            "markerCount": sum(item["count"] for item in marker_descriptors),
            "analysis": {
                "candidatesPath": "analysis/candidates.json",
                "candidateCount": len(candidates),
                "metadata": camelize_keys(
                    sanitize_export_metadata(load_json(track.analysis_json, {}))
                ),
            },
            "audio": {
                "mode": audio_mode,
                "source": {
                    "role": "originalUploadIdentity",
                    "available": source_available,
                    "originalFileName": track.original_file_name,
                    "format": track.format,
                    "sampleRate": track.original_sample_rate,
                    "sampleCount": track.sample_count,
                    "channels": track.channels,
                    "sha256": source_sha256,
                },
                "reference": reference_descriptor,
                "stems": [descriptor for _, _, descriptor in stem_assets],
            },
        }

        with zipfile.ZipFile(
            archive_path,
            mode="w",
            compression=zipfile.ZIP_DEFLATED,
            allowZip64=True,
        ) as archive:
            archive.writestr("manifest.json", _json_bytes(manifest))
            for stem in MARKER_STEMS:
                archive.writestr(
                    f"markers/{stem}.json",
                    _json_bytes(
                        {
                            "schemaVersion": PACKAGE_SCHEMA_VERSION,
                            "primaryStem": stem,
                            "markerCount": len(markers_by_stem[stem]),
                            "markers": markers_by_stem[stem],
                        }
                    ),
                )
            archive.writestr(
                "analysis/candidates.json",
                _json_bytes(
                    {
                        "schemaVersion": PACKAGE_SCHEMA_VERSION,
                        "candidateCount": len(candidates),
                        "candidates": candidates,
                    }
                ),
            )
            if reference_path is not None:
                archive.write(
                    reference_path,
                    "audio/reference.flac",
                    compress_type=zipfile.ZIP_STORED,
                )
            for stem, path, _ in stem_assets:
                archive.write(
                    path,
                    f"stems/{stem}.flac",
                    compress_type=zipfile.ZIP_STORED,
                )

        return PackageExport(archive_path=archive_path, temporary_root=temporary_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
