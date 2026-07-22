from __future__ import annotations

import hashlib
import json
import mimetypes
import re
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, Header, Query
from fastapi.responses import Response, StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from .chart_engine.generator import generate_chart
from .chart_engine.library import ReferenceLibrary
from .chart_engine.models import (
    ChartDocument,
    ChartGenerationResponse,
    CorpusStatistics,
    GenerateChartRequest,
)
from .chart_engine.sm import export_sm
from .config import get_settings
from .database import get_db
from .errors import BeatForgeError, not_found
from .models import TrackModel
from .serialization import candidate_event_dict

chart_router = APIRouter(prefix="/api", tags=["chart-engine"])
_GENERATION_ID = re.compile(r"^[a-f0-9]{20}$")


@lru_cache(maxsize=4)
def _library_for_root(root: str) -> ReferenceLibrary:
    return ReferenceLibrary(root)


def _library() -> ReferenceLibrary:
    return _library_for_root(str(get_settings().speed_charts_dir))


@lru_cache(maxsize=4)
def _statistics_for_root(root: str) -> CorpusStatistics:
    return _library_for_root(root).statistics()


def _corpus_statistics() -> CorpusStatistics:
    return _statistics_for_root(str(get_settings().speed_charts_dir))


@lru_cache(maxsize=2)
def _load_local_chart_model(path: str, modified_ns: int):
    del modified_ns
    from .chart_engine.learning import LocalChartModel

    return LocalChartModel.load(path, device="auto")


def _checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_predictions(
    track: TrackModel, difficulty: int, enabled: bool
) -> tuple[dict[str, dict[str, Any]] | None, dict[str, Any] | None, bool]:
    checkpoint = get_settings().chart_models_dir / "chart-transformer.pt"
    available = checkpoint.is_file()
    if not enabled or not available or not track.candidate_events:
        return None, None, available
    analysis = {
        "original_sample_rate": track.original_sample_rate,
        "sample_count": track.sample_count,
        "duration_sec": track.duration_sec,
        "bpm": track.tempo_segments[0].bpm,
        "bpm_confidence": track.tempo_segments[0].confidence,
        "beat_offset_sample": track.tempo_segments[0].beat_offset_sample,
        "candidate_events": [
            candidate_event_dict(candidate, track.original_sample_rate)
            for candidate in track.candidate_events
        ],
    }
    try:
        checkpoint_stat = checkpoint.stat()
        checkpoint_sha256 = _checkpoint_sha256(checkpoint)
        runtime = _load_local_chart_model(str(checkpoint.resolve()), checkpoint_stat.st_mtime_ns)
        inference = runtime.predict({"analysis": analysis}, difficulty=difficulty)
    except Exception as exc:
        raise BeatForgeError(
            "LOCAL_CHART_MODEL_FAILED",
            f"The local chart model could not run: {exc}",
            status_code=409,
        ) from exc
    predictions = {
        item.candidate_id: {
            "laneProbabilities": list(item.lane_probabilities),
            "holdProbability": item.hold_probability,
        }
        for item in inference.predictions
    }
    metadata = inference.checkpoint_metadata
    provenance = {
        "schemaVersion": metadata.get("schemaVersion"),
        "architecture": metadata.get("architecture"),
        "createdAt": metadata.get("createdAt"),
        "datasetFingerprint": metadata.get("datasetFingerprint"),
        "sampleCount": metadata.get("sampleCount"),
        "bestLoss": metadata.get("bestLoss"),
        "realDataOnly": metadata.get("realDataOnly") is True,
        "checkpointSha256": checkpoint_sha256,
    }
    return predictions, provenance, available


def _get_track(session: Session, track_id: str) -> TrackModel:
    statement = (
        select(TrackModel)
        .where(TrackModel.id == track_id)
        .options(
            selectinload(TrackModel.project),
            selectinload(TrackModel.hit_points),
            selectinload(TrackModel.candidate_events),
            selectinload(TrackModel.tempo_segments),
        )
    )
    track = session.scalar(statement)
    if not track:
        raise not_found("track", track_id)
    return track


def _parse_range(value: str, size: int) -> tuple[int, int]:
    if not value.startswith("bytes=") or "," in value:
        raise ValueError
    start_text, end_text = value[6:].split("-", 1)
    if not start_text:
        length = int(end_text)
        if length <= 0:
            raise ValueError
        return max(0, size - length), size - 1
    start = int(start_text)
    end = int(end_text) if end_text else size - 1
    if start < 0 or start >= size or end < start:
        raise ValueError
    return start, min(end, size - 1)


def _audio_response(path: Path, range_header: str | None) -> StreamingResponse:
    size = path.stat().st_size
    start, end, status_code = 0, size - 1, 200
    if range_header:
        try:
            start, end = _parse_range(range_header, size)
        except (TypeError, ValueError):
            raise BeatForgeError(
                "INVALID_RANGE",
                "The requested byte range is invalid",
                status_code=416,
                details={"size": size},
            ) from None
        status_code = 206

    def chunks():
        remaining = end - start + 1
        with path.open("rb") as handle:
            handle.seek(start)
            while remaining > 0:
                data = handle.read(min(1024 * 1024, remaining))
                if not data:
                    return
                remaining -= len(data)
                yield data

    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1),
        "Content-Disposition": f"inline; filename*=UTF-8''{quote(path.name)}",
    }
    if status_code == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    media_type = mimetypes.guess_type(path.name)[0] or "audio/mpeg"
    return StreamingResponse(
        chunks(), status_code=status_code, media_type=media_type, headers=headers
    )


def _generated_path(track_id: str, generation_id: str) -> Path:
    if not _GENERATION_ID.fullmatch(generation_id):
        raise BeatForgeError(
            "INVALID_GENERATION_ID", "The generation id is invalid", status_code=404
        )
    root = get_settings().generated_charts_dir.resolve()
    path = (root / track_id / f"{generation_id}.json").resolve()
    if not path.is_relative_to(root):
        raise BeatForgeError("INVALID_CHART_PATH", "The chart path is invalid", status_code=404)
    return path


def _save_generated(track_id: str, chart: ChartDocument) -> None:
    directory = get_settings().generated_charts_dir / track_id
    directory.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        chart.model_dump(by_alias=True, mode="json"), ensure_ascii=False, separators=(",", ":")
    )
    path = directory / f"{chart.id}.json"
    temporary = directory / f".{chart.id}.tmp"
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)
    latest = directory / "latest.json"
    latest_temporary = directory / ".latest.tmp"
    latest_temporary.write_text(payload, encoding="utf-8")
    latest_temporary.replace(latest)


def _load_generated(track_id: str, generation_id: str | None = None) -> ChartDocument:
    if generation_id:
        path = _generated_path(track_id, generation_id)
    else:
        root = get_settings().generated_charts_dir.resolve()
        path = (root / track_id / "latest.json").resolve()
        if not path.is_relative_to(root):
            raise BeatForgeError("INVALID_CHART_PATH", "The chart path is invalid", status_code=404)
    if not path.is_file():
        raise BeatForgeError(
            "CHART_NOT_GENERATED",
            "No generated chart is available for this track.",
            status_code=404,
        )
    try:
        return ChartDocument.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        raise BeatForgeError(
            "GENERATED_CHART_INVALID",
            "The saved chart artifact is invalid.",
            status_code=500,
        ) from exc


@chart_router.get("/chart-engine/reference-charts")
def list_reference_charts(
    mode: str | None = Query(default="pump-single", pattern=r"^(pump-single|pump-double)$"),
    group: str | None = Query(default=None, pattern=r"^SPEED_(CLUB|DEVIL|REMIX)$"),
    search: str = Query(default="", max_length=300),
) -> dict[str, Any]:
    library = _library()
    items = library.summaries(mode=mode, group=group, search=search)
    return {
        "items": [item.model_dump(by_alias=True, mode="json") for item in items],
        "total": len(items),
        "corpusTotal": len(library),
        "source": "local_reference_corpus",
    }


@chart_router.get("/chart-engine/reference-charts/{chart_id}", response_model=ChartDocument)
def get_reference_chart(chart_id: str) -> ChartDocument:
    try:
        return _library().chart(chart_id)
    except KeyError:
        raise BeatForgeError(
            "REFERENCE_CHART_NOT_FOUND", "Reference chart not found", status_code=404
        ) from None


@chart_router.get("/chart-engine/reference-charts/{chart_id}/audio")
def get_reference_audio(
    chart_id: str,
    range_header: Annotated[str | None, Header(alias="Range")] = None,
) -> StreamingResponse:
    try:
        asset = _library().asset(chart_id)
    except KeyError:
        raise BeatForgeError(
            "REFERENCE_CHART_NOT_FOUND", "Reference chart not found", status_code=404
        ) from None
    return _audio_response(asset.audio_path, range_header)


@chart_router.head("/chart-engine/reference-charts/{chart_id}/audio")
def head_reference_audio(chart_id: str) -> Response:
    try:
        asset = _library().asset(chart_id)
    except KeyError:
        raise BeatForgeError(
            "REFERENCE_CHART_NOT_FOUND", "Reference chart not found", status_code=404
        ) from None
    media_type = mimetypes.guess_type(asset.audio_path.name)[0] or "audio/mpeg"
    return Response(
        media_type=media_type,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(asset.audio_path.stat().st_size),
            "Content-Disposition": f"inline; filename*=UTF-8''{quote(asset.audio_path.name)}",
        },
    )


@chart_router.get("/chart-engine/statistics", response_model=CorpusStatistics)
def get_corpus_statistics() -> CorpusStatistics:
    if len(_library()) == 0:
        raise BeatForgeError(
            "REFERENCE_CORPUS_NOT_FOUND",
            "The local reference chart corpus is not available.",
            status_code=404,
        )
    return _corpus_statistics()


@chart_router.post("/tracks/{track_id}/chart/generate", response_model=ChartGenerationResponse)
def generate_track_chart(
    track_id: str,
    payload: GenerateChartRequest,
    session: Annotated[Session, Depends(get_db)],
) -> ChartGenerationResponse:
    track = _get_track(session, track_id)
    if not track.tempo_segments or not (track.candidate_events or track.hit_points):
        raise BeatForgeError(
            "CHART_ANALYSIS_REQUIRED",
            "Run BeatForge analysis before generating a chart.",
            status_code=409,
        )
    if len(_library()) == 0:
        raise BeatForgeError(
            "REFERENCE_CORPUS_NOT_FOUND",
            "The local reference chart corpus is not available.",
            status_code=404,
        )
    corpus = _corpus_statistics()
    model_predictions, model_provenance, model_available = _model_predictions(
        track, payload.difficulty, payload.use_local_model
    )
    try:
        chart = generate_chart(
            track_id=track.id,
            title=track.project.title,
            artist=track.project.artist,
            music=track.original_file_name,
            duration_sec=track.duration_sec,
            sample_rate=track.original_sample_rate,
            tempo_segments=track.tempo_segments,
            candidates=track.candidate_events,
            hit_points=track.hit_points,
            difficulty=payload.difficulty,
            enable_spin=payload.enable_spin,
            seed=payload.seed,
            transition_probabilities=corpus.lane_transition_probabilities,
            model_predictions=model_predictions,
            model_provenance=model_provenance,
        )
    except ValueError as exc:
        raise BeatForgeError("CHART_GENERATION_FAILED", str(exc), status_code=422) from exc
    _save_generated(track.id, chart)
    return ChartGenerationResponse(
        generation_id=chart.id,
        chart=chart,
        reference_corpus={
            "source": "SPEED_DEVIL + SPEED_REMIX + pump-single SPEED_CLUB",
            "chartCount": corpus.single_chart_count,
            "songCount": corpus.single_song_count,
            "difficultyRange": [corpus.difficulty_min, min(corpus.difficulty_max, 15)],
            "model": {
                "requested": payload.use_local_model,
                "available": model_available,
                "used": model_predictions is not None,
            },
        },
    )


@chart_router.get("/tracks/{track_id}/chart/latest", response_model=ChartDocument)
def get_latest_track_chart(
    track_id: str, session: Annotated[Session, Depends(get_db)]
) -> ChartDocument:
    _get_track(session, track_id)
    return _load_generated(track_id)


@chart_router.get("/tracks/{track_id}/chart/export")
def export_track_chart(
    track_id: str,
    session: Annotated[Session, Depends(get_db)],
    generation_id: str | None = Query(default=None, alias="generationId"),
) -> Response:
    track = _get_track(session, track_id)
    chart = _load_generated(track_id, generation_id)
    filename = f"{Path(track.original_file_name).stem}_Lv{chart.meter}.sm"
    return Response(
        export_sm(chart).encode("utf-8-sig"),
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": (
                f"attachment; filename=beatforge-Lv{chart.meter}.sm; "
                f"filename*=UTF-8''{quote(filename)}"
            )
        },
    )
