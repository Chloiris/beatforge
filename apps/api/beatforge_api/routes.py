from __future__ import annotations

import asyncio
import csv
import io
import json
import math
import mimetypes
import re
import shutil
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, Form, Header, Query, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, selectinload
from starlette.background import BackgroundTask

from .audio.alignment import AlignmentRunner
from .audio.alignment.base import AlignmentContext, TempoReference
from .audio.alignment.hubert_engine import HubertAlignmentReport, HubertCandidateBundle
from .audio.alignment.schema import (
    AlignmentMethod,
    AlignmentMethodId,
    AlignmentReport,
    AlignmentResult,
    AlignmentRunRequest,
)
from .config import get_settings
from .database import SessionLocal, get_db
from .errors import BeatForgeError, not_found
from .export_safety import sanitize_export_metadata
from .jobs import submit_analysis
from .media import persist_upload, probe_audio
from .models import (
    AnalysisJobModel,
    HitPointModel,
    ProjectModel,
    TempoSegmentModel,
    TrackModel,
    VocalAlignmentJobModel,
    new_id,
)
from .package_export import build_package_export
from .schemas import (
    AnalysisJob,
    AnalyzeRequest,
    AnalyzeResponse,
    CandidateEvent,
    HitPoint,
    HitPointBulkUpdate,
    HitPointCreate,
    HitPointPatch,
    ProjectCreate,
    ProjectDetail,
    ProjectList,
    ProjectPatch,
    ProjectResponse,
    TempoMapUpdate,
    TempoSegment,
    TrackResponse,
    UploadResponse,
    VocalLyricsDraftRequest,
    VocalLyricsJob,
    VocalLyricsJobResponse,
    VocalLyricsResponse,
    VocalLyricsUpdate,
    WaveformResponse,
)
from .serialization import (
    camelize_keys,
    candidate_event_dict,
    dumps,
    hit_dict,
    job_dict,
    json_safe,
    load_json,
    project_dict,
    track_dict,
    vocal_job_dict,
    vocal_lyrics_dict,
)
from .storage_paths import resolve_storage_path, storage_relative_path
from .timing import nearest_grid_sample
from .vocal_jobs import submit_vocal_job, vocal_runtime_diagnostics
from .waveform_store import read_waveform_lods, select_waveform_level

router = APIRouter(prefix="/api")
_COVER_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{0,119}\.svg$")
_alignment_runner: AlignmentRunner | None = None


def _attachment_content_disposition(filename: str) -> str:
    """Return an RFC 6266 filename that is safe for Starlette's Latin-1 headers."""
    suffixes = "".join(Path(filename).suffixes)
    normalized = unicodedata.normalize("NFKD", filename)
    ascii_name = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_name = re.sub(r"[^A-Za-z0-9._ -]+", "_", ascii_name)
    ascii_name = ascii_name.replace('"', "_").replace("\\", "_").strip(" ._-")
    if not ascii_name or ascii_name in {".", ".."} or not Path(ascii_name).stem.strip(" ._-"):
        ascii_name = f"beatforge-export{suffixes or '.dat'}"
    if len(ascii_name) > 180:
        stem = Path(ascii_name).stem[: max(1, 180 - len(suffixes))].rstrip(" .")
        ascii_name = f"{stem}{suffixes}"
    encoded = quote(filename, safe="")
    return f'attachment; filename="{ascii_name}"; filename*=UTF-8\'\'{encoded}'


def _get_alignment_runner() -> AlignmentRunner:
    global _alignment_runner
    if _alignment_runner is None:
        settings = get_settings()
        _alignment_runner = AlignmentRunner(settings.storage_dir, settings.project_root)
    return _alignment_runner


def _project_statement(project_id: str):
    return (
        select(ProjectModel)
        .where(ProjectModel.id == project_id)
        .options(
            selectinload(ProjectModel.track).selectinload(TrackModel.hit_points),
            selectinload(ProjectModel.track).selectinload(TrackModel.candidate_events),
            selectinload(ProjectModel.track).selectinload(TrackModel.tempo_segments),
        )
    )


def _track_statement(track_id: str):
    return (
        select(TrackModel)
        .where(TrackModel.id == track_id)
        .options(
            selectinload(TrackModel.project),
            selectinload(TrackModel.hit_points),
            selectinload(TrackModel.candidate_events),
            selectinload(TrackModel.tempo_segments),
        )
    )


def _get_project(session: Session, project_id: str) -> ProjectModel:
    project = session.scalar(_project_statement(project_id))
    if not project:
        raise not_found("project", project_id)
    return project


def _get_track(session: Session, track_id: str) -> TrackModel:
    track = session.scalar(_track_statement(track_id))
    if not track:
        raise not_found("track", track_id)
    return track


def _alignment_context(track: TrackModel) -> AlignmentContext:
    if not track.lyrics_text.strip():
        raise BeatForgeError(
            "LYRICS_REQUIRED",
            "Save Japanese lyrics before running an alignment experiment.",
            status_code=422,
        )
    if track.lyrics_format == "romaji":
        raise BeatForgeError(
            "ROMAJI_REQUIRES_KANA",
            "Convert ambiguous romaji to Japanese text or kana before alignment.",
            status_code=422,
        )
    settings = get_settings()
    stem_root = settings.stems_dir.resolve()
    vocals_path = (stem_root / track.id / "vocals.flac").resolve()
    if not vocals_path.is_relative_to(stem_root) or not vocals_path.is_file():
        raise BeatForgeError(
            "VOCALS_STEM_NOT_READY",
            "Generate a vocals stem before running the Alignment Lab.",
            status_code=409,
        )
    qwen_payload = load_json(track.vocal_alignment_json, {})
    return AlignmentContext(
        track_id=track.id,
        lyrics=track.lyrics_text,
        lyrics_format=track.lyrics_format,
        vocals_path=vocals_path,
        sample_rate=track.original_sample_rate,
        sample_count=track.sample_count,
        tempo_map=tuple(
            TempoReference(
                start_sample=segment.start_sample,
                bpm=segment.bpm,
                beat_offset_sample=segment.beat_offset_sample,
            )
            for segment in track.tempo_segments
        ),
        models_dir=settings.models_dir,
        storage_dir=settings.storage_dir,
        project_root=settings.project_root,
        song=track.project.title,
        artist=track.project.artist,
        qwen_payload=qwen_payload if isinstance(qwen_payload, dict) else {},
    )


def _validate_sample(track: TrackModel, sample: int, field: str = "sample") -> None:
    if sample < 0 or sample >= track.sample_count:
        raise BeatForgeError(
            "SAMPLE_OUT_OF_RANGE",
            f"{field} must be within the original audio sample range",
            status_code=422,
            details={"field": field, "sample": sample, "sampleCount": track.sample_count},
        )


def _recommended_snap(track: TrackModel, sample: int) -> int:
    if not track.tempo_segments:
        return sample
    segment = max(
        (item for item in track.tempo_segments if item.start_sample <= sample),
        key=lambda item: item.start_sample,
        default=track.tempo_segments[0],
    )
    snapped = nearest_grid_sample(
        sample,
        sample_rate=track.original_sample_rate,
        bpm=segment.bpm,
        beat_offset_sample=segment.beat_offset_sample,
        subdivisions_per_beat=4,
    )
    return min(max(0, snapped), max(0, track.sample_count - 1))


def _fill_hit_model(model: HitPointModel, payload: HitPointCreate | HitPointPatch) -> None:
    fields = payload.model_fields_set
    for field in (
        "sample",
        "detected_sample",
        "refined_sample",
        "snapped_sample",
        "snap_error_ms",
        "band",
        "confidence",
        "salience",
        "source",
        "primary_stem",
        "manually_edited",
        "locked",
    ):
        if field in fields:
            value = getattr(payload, field)
            if value is not None:
                setattr(model, field, value)
    if "acoustic_sample" in fields and payload.acoustic_sample is not None:
        model.acoustic_sample = payload.acoustic_sample
        model.sample = payload.acoustic_sample
        model.refined_sample = payload.acoustic_sample
    elif "sample" in fields and payload.sample is not None:
        model.acoustic_sample = payload.sample
        model.sample = payload.sample
        model.refined_sample = payload.sample
    if "chart_sample" in fields and payload.chart_sample is not None:
        model.chart_sample = payload.chart_sample
        model.snapped_sample = payload.chart_sample
    elif "snapped_sample" in fields and payload.snapped_sample is not None:
        model.chart_sample = payload.snapped_sample
    if "detector_votes" in fields and payload.detector_votes is not None:
        model.detector_votes_json = dumps(payload.detector_votes)
    if "stem_evidence" in fields and payload.stem_evidence is not None:
        model.stem_evidence_json = dumps(payload.stem_evidence)


def _new_hit(track: TrackModel, payload: HitPointCreate) -> HitPointModel:
    acoustic = (
        payload.acoustic_sample
        if payload.acoustic_sample is not None
        else payload.refined_sample if payload.refined_sample is not None else payload.sample
    )
    detected = payload.detected_sample if payload.detected_sample is not None else acoustic
    chart = (
        payload.chart_sample
        if payload.chart_sample is not None
        else payload.snapped_sample
        if payload.snapped_sample is not None
        else _recommended_snap(track, acoustic)
    )
    snap_error_ms = (
        payload.snap_error_ms
        if "snap_error_ms" in payload.model_fields_set
        else (acoustic - chart) * 1000.0 / track.original_sample_rate
    )
    for name, value in (
        ("sample", acoustic),
        ("acousticSample", acoustic),
        ("chartSample", chart),
        ("detectedSample", detected),
        ("refinedSample", acoustic),
        ("snappedSample", chart),
    ):
        _validate_sample(track, value, name)
    return HitPointModel(
        id=payload.id or new_id(),
        sample=acoustic,
        acoustic_sample=acoustic,
        chart_sample=chart,
        detected_sample=detected,
        refined_sample=acoustic,
        snapped_sample=chart,
        snap_error_ms=snap_error_ms,
        band=payload.band,
        confidence=payload.confidence,
        salience=payload.salience,
        source=payload.source,
        detector_votes_json=dumps(payload.detector_votes),
        primary_stem=payload.primary_stem,
        stem_evidence_json=dumps(payload.stem_evidence),
        manually_edited=payload.manually_edited,
        locked=payload.locked,
    )


def _mark_edited(track: TrackModel) -> None:
    track.updated_at = datetime.now(UTC)
    track.project.status = "edited"
    track.project.updated_at = track.updated_at


@router.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "beatforge-api", "version": "0.7.1"}


@router.get("/alignment/methods", response_model=list[AlignmentMethod])
def get_alignment_methods() -> list[AlignmentMethod]:
    return _get_alignment_runner().methods()


@router.get("/assets/covers/{filename}")
def get_cover(filename: str) -> Response:
    if not _COVER_NAME.fullmatch(filename) or Path(filename).name != filename:
        raise BeatForgeError("INVALID_COVER_PATH", "The cover filename is invalid", status_code=404)
    cover_dir = (get_settings().storage_dir / "covers").resolve()
    path = (cover_dir / filename).resolve()
    if not path.is_relative_to(cover_dir) or not path.is_file():
        raise BeatForgeError("COVER_NOT_FOUND", "Cover image not found", status_code=404)
    return Response(
        path.read_bytes(),
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@router.get("/projects", response_model=ProjectList)
def list_projects(
    session: Annotated[Session, Depends(get_db)],
    search: str | None = Query(default=None, max_length=300),
    status: str | None = Query(default=None),
) -> dict[str, Any]:
    statement = select(ProjectModel).options(
        selectinload(ProjectModel.track).selectinload(TrackModel.hit_points),
        selectinload(ProjectModel.track).selectinload(TrackModel.tempo_segments),
    )
    count_statement = select(func.count(ProjectModel.id))
    if search:
        pattern = f"%{search.strip()}%"
        condition = or_(ProjectModel.title.ilike(pattern), ProjectModel.artist.ilike(pattern))
        statement = statement.where(condition)
        count_statement = count_statement.where(condition)
    if status:
        allowed = {"unprocessed", "processing", "completed", "edited", "failed"}
        if status not in allowed:
            raise BeatForgeError(
                "INVALID_PROJECT_STATUS", "Unknown project status filter", status_code=422
            )
        statement = statement.where(ProjectModel.status == status)
        count_statement = count_statement.where(ProjectModel.status == status)
    projects = session.scalars(statement.order_by(ProjectModel.updated_at.desc())).all()
    return {
        "items": [project_dict(project) for project in projects],
        "total": int(session.scalar(count_statement) or 0),
    }


@router.post("/projects", response_model=ProjectResponse, status_code=201)
def create_project(
    payload: ProjectCreate, session: Annotated[Session, Depends(get_db)]
) -> dict[str, Any]:
    project = ProjectModel(**payload.model_dump())
    session.add(project)
    session.commit()
    session.refresh(project)
    return project_dict(project)


@router.get("/projects/{project_id}", response_model=ProjectDetail)
def get_project(project_id: str, session: Annotated[Session, Depends(get_db)]) -> dict[str, Any]:
    project = _get_project(session, project_id)
    return {
        **project_dict(project),
        "track": track_dict(project.track) if project.track else None,
    }


@router.patch("/projects/{project_id}", response_model=ProjectResponse)
def patch_project(
    project_id: str,
    payload: ProjectPatch,
    session: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    project = _get_project(session, project_id)
    for field in payload.model_fields_set:
        value = getattr(payload, field)
        if value is not None:
            setattr(project, field, value)
    project.updated_at = datetime.now(UTC)
    session.commit()
    return project_dict(project)


@router.delete("/projects/{project_id}", status_code=204)
def delete_project(project_id: str, session: Annotated[Session, Depends(get_db)]) -> Response:
    project = _get_project(session, project_id)
    paths: list[tuple[Path, Path]] = []
    settings = get_settings()
    if project.track:
        try:
            paths.append(
                (resolve_storage_path(project.track.file_path, settings), settings.audio_dir)
            )
        except ValueError:
            pass
        if project.track.waveform_path:
            try:
                paths.append(
                    (
                        resolve_storage_path(project.track.waveform_path, settings),
                        settings.waveform_dir,
                    )
                )
            except ValueError:
                pass
        paths.append(
            (settings.analyses_dir / f"{project.track.id}.json", settings.analyses_dir)
        )
    session.delete(project)
    session.commit()
    for path, allowed_root in paths:
        try:
            resolved = path.resolve()
            if resolved.is_relative_to(allowed_root.resolve()):
                resolved.unlink(missing_ok=True)
        except (OSError, RuntimeError):
            pass
    if project.track:
        stem_directory = (settings.stems_dir / project.track.id).resolve()
        try:
            if stem_directory.is_relative_to(settings.stems_dir.resolve()):
                shutil.rmtree(stem_directory, ignore_errors=True)
        except (OSError, RuntimeError):
            pass
    return Response(status_code=204)


@router.post("/tracks/upload", response_model=UploadResponse, status_code=201)
async def upload_track(
    file: Annotated[UploadFile, File()],
    session: Annotated[Session, Depends(get_db)],
    title: Annotated[str | None, Form(max_length=300)] = None,
    artist: Annotated[str, Form(max_length=300)] = "未知艺术家",
    genre: Annotated[str, Form(max_length=120)] = "Unknown",
) -> dict[str, Any]:
    settings = get_settings()
    path, original_name, audio_format, _size = await persist_upload(file, settings)
    try:
        probe = await asyncio.to_thread(probe_audio, path)
        display_title = (title or Path(original_name).stem).strip() or Path(original_name).stem
        project = ProjectModel(title=display_title, artist=artist, genre=genre)
        track = TrackModel(
            project=project,
            original_file_name=original_name,
            stored_file_name=path.name,
            file_path=storage_relative_path(path, settings),
            format=audio_format,
            original_sample_rate=probe.sample_rate,
            channels=probe.channels,
            sample_count=probe.sample_count,
            duration_sec=probe.duration_sec,
            leading_silence_samples=0,
        )
        session.add(project)
        session.add(track)
        session.commit()
        project = _get_project(session, project.id)
        assert project.track is not None
        return {"project": project_dict(project), "track": track_dict(project.track)}
    except Exception:
        path.unlink(missing_ok=True)
        raise


@router.post("/tracks/{track_id}/analyze", response_model=AnalyzeResponse, status_code=202)
def analyze_track(
    track_id: str,
    payload: AnalyzeRequest,
    session: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    track = _get_track(session, track_id)
    running = session.scalar(
        select(AnalysisJobModel).where(
            AnalysisJobModel.track_id == track_id,
            AnalysisJobModel.status.in_(["queued", "processing"]),
        )
    )
    if running:
        raise BeatForgeError(
            "ANALYSIS_ALREADY_RUNNING",
            "An analysis job is already running for this track",
            status_code=409,
            details={"jobId": running.id},
        )
    job = AnalysisJobModel(track=track, mode=payload.mode, sensitivity=payload.sensitivity)
    track.project.status = "processing"
    track.project.updated_at = datetime.now(UTC)
    session.add(job)
    session.commit()
    submit_analysis(job.id)
    return {"job_id": job.id, "status": job.status}


@router.get("/analysis-jobs/{job_id}", response_model=AnalysisJob)
def get_analysis_job(
    job_id: str, session: Annotated[Session, Depends(get_db)]
) -> dict[str, Any]:
    job = session.get(AnalysisJobModel, job_id)
    if not job:
        raise not_found("analysis job", job_id)
    return job_dict(job)


@router.get("/analysis-jobs/{job_id}/events")
async def analysis_job_events(job_id: str) -> StreamingResponse:
    with SessionLocal() as initial_session:
        if not initial_session.get(AnalysisJobModel, job_id):
            raise not_found("analysis job", job_id)

    async def events():
        last_payload = ""
        while True:
            with SessionLocal() as session:
                job = session.get(AnalysisJobModel, job_id)
                if not job:
                    yield "event: error\ndata: {\"error\":\"job deleted\"}\n\n"
                    return
                data = AnalysisJob.model_validate(job_dict(job)).model_dump(
                    by_alias=True, mode="json"
                )
                payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
                status = job.status
            if payload != last_payload:
                yield f"event: progress\ndata: {payload}\n\n"
                last_payload = payload
            if status in {"completed", "failed"}:
                return
            await asyncio.sleep(0.25)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/tracks/{track_id}", response_model=TrackResponse)
def get_track(track_id: str, session: Annotated[Session, Depends(get_db)]) -> dict[str, Any]:
    return track_dict(_get_track(session, track_id))


@router.post(
    "/tracks/{track_id}/alignment/run",
    response_model=AlignmentResult,
    status_code=202,
)
def run_alignment(
    track_id: str,
    payload: AlignmentRunRequest,
    session: Annotated[Session, Depends(get_db)],
) -> AlignmentResult:
    track = _get_track(session, track_id)
    return _get_alignment_runner().submit(_alignment_context(track), payload.method)


@router.get(
    "/tracks/{track_id}/alignment/ctc/candidate-events",
    response_model=HubertCandidateBundle,
)
def get_hubert_candidate_events(
    track_id: str,
    session: Annotated[Session, Depends(get_db)],
) -> HubertCandidateBundle:
    _get_track(session, track_id)
    candidates = _get_alignment_runner().get_hubert_candidates(track_id)
    if candidates is None:
        raise BeatForgeError(
            "HUBERT_CANDIDATES_NOT_FOUND",
            "No completed HuBERT hierarchy candidate artifact exists for this track.",
            status_code=404,
            details={"trackId": track_id, "method": "ctc"},
        )
    return candidates


@router.get(
    "/tracks/{track_id}/alignment/ctc/hubert-report",
    response_model=HubertAlignmentReport,
)
def get_hubert_alignment_report(
    track_id: str,
    session: Annotated[Session, Depends(get_db)],
) -> HubertAlignmentReport:
    _get_track(session, track_id)
    report = _get_alignment_runner().get_hubert_report(track_id)
    if report is None:
        raise BeatForgeError(
            "HUBERT_REPORT_NOT_FOUND",
            "No completed v0.6.1 HuBERT report exists for this track.",
            status_code=404,
            details={"trackId": track_id, "method": "ctc"},
        )
    return report


@router.get(
    "/tracks/{track_id}/alignment/{method}/report",
    response_model=AlignmentReport,
)
def get_alignment_report(
    track_id: str,
    method: AlignmentMethodId,
    session: Annotated[Session, Depends(get_db)],
) -> AlignmentReport:
    _get_track(session, track_id)
    report = _get_alignment_runner().get_report(track_id, method)
    if report is None:
        raise BeatForgeError(
            "ALIGNMENT_REPORT_NOT_FOUND",
            "No completed proxy evaluation exists for this track and method.",
            status_code=404,
            details={"trackId": track_id, "method": method},
        )
    return report


@router.get(
    "/tracks/{track_id}/alignment/{method}",
    response_model=AlignmentResult,
)
def get_alignment_result(
    track_id: str,
    method: AlignmentMethodId,
    session: Annotated[Session, Depends(get_db)],
) -> AlignmentResult:
    _get_track(session, track_id)
    result = _get_alignment_runner().get_result(track_id, method)
    if result is None:
        raise BeatForgeError(
            "ALIGNMENT_RESULT_NOT_FOUND",
            "This alignment method has not been run for the track.",
            status_code=404,
            details={"trackId": track_id, "method": method},
        )
    return result


@router.get("/vocal-runtime")
def get_vocal_runtime() -> dict[str, Any]:
    return vocal_runtime_diagnostics()


@router.get("/tracks/{track_id}/vocal-lyrics", response_model=VocalLyricsResponse)
def get_vocal_lyrics(
    track_id: str, session: Annotated[Session, Depends(get_db)]
) -> dict[str, Any]:
    return vocal_lyrics_dict(_get_track(session, track_id))


@router.put("/tracks/{track_id}/vocal-lyrics", response_model=VocalLyricsResponse)
def save_vocal_lyrics(
    track_id: str,
    payload: VocalLyricsUpdate,
    session: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    track = _get_track(session, track_id)
    track.lyrics_text = payload.text
    track.lyrics_format = payload.input_format
    now = datetime.now(UTC)
    track.vocal_alignment_json = dumps(
        {
            "status": "saved" if payload.text.strip() else "empty",
            "stage": "idle",
            "progress": 0.0,
            "anchors": [],
            "error": None,
            "updated_at": now.isoformat(),
        }
    )
    track.updated_at = now
    track.project.status = "edited"
    track.project.updated_at = now
    session.commit()
    return vocal_lyrics_dict(track)


def _start_vocal_job(
    track: TrackModel,
    *,
    operation: str,
    session: Session,
) -> dict[str, Any]:
    running = session.scalar(
        select(VocalAlignmentJobModel).where(
            VocalAlignmentJobModel.track_id == track.id,
            VocalAlignmentJobModel.status.in_(["queued", "processing"]),
        )
    )
    if running is not None:
        raise BeatForgeError(
            "VOCAL_JOB_ALREADY_RUNNING",
            "An ASR or lyric-alignment job is already running for this track",
            status_code=409,
            details={"jobId": running.id},
        )
    if operation == "alignment" and not track.lyrics_text.strip():
        raise BeatForgeError(
            "LYRICS_REQUIRED",
            "Save Japanese lyrics or generate an ASR draft before alignment",
            status_code=422,
        )
    if operation == "alignment" and track.lyrics_format == "romaji":
        raise BeatForgeError(
            "ROMAJI_REQUIRES_KANA",
            "Convert ambiguous romaji to Japanese text or kana before alignment",
            status_code=422,
        )
    job = VocalAlignmentJobModel(
        track=track,
        operation=operation,
        replace_vocal_hits=True,
    )
    session.add(job)
    session.commit()
    submit_vocal_job(job.id)
    return {"job_id": job.id, "status": job.status}


@router.post(
    "/tracks/{track_id}/vocal-lyrics/align",
    response_model=VocalLyricsJobResponse,
    status_code=202,
)
def align_vocal_lyrics(
    track_id: str, session: Annotated[Session, Depends(get_db)]
) -> dict[str, Any]:
    return _start_vocal_job(_get_track(session, track_id), operation="alignment", session=session)


@router.post(
    "/tracks/{track_id}/vocal-lyrics/asr-draft",
    response_model=VocalLyricsJobResponse,
    status_code=202,
)
def draft_vocal_lyrics(
    track_id: str,
    _payload: VocalLyricsDraftRequest,
    session: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    return _start_vocal_job(_get_track(session, track_id), operation="asr_draft", session=session)


@router.get("/vocal-lyrics-jobs/{job_id}", response_model=VocalLyricsJob)
def get_vocal_lyrics_job(
    job_id: str, session: Annotated[Session, Depends(get_db)]
) -> dict[str, Any]:
    job = session.scalar(
        select(VocalAlignmentJobModel)
        .where(VocalAlignmentJobModel.id == job_id)
        .options(selectinload(VocalAlignmentJobModel.track))
    )
    if job is None:
        raise not_found("vocal lyrics job", job_id)
    return vocal_job_dict(job)


def _resolve_audio_path(track: TrackModel) -> Path:
    settings = get_settings()
    try:
        path = resolve_storage_path(track.file_path, settings)
    except ValueError:
        raise BeatForgeError(
            "UNSAFE_AUDIO_PATH",
            "Stored audio path is outside the configured storage directory",
            status_code=500,
        ) from None
    if not path.is_file():
        raise BeatForgeError(
            "AUDIO_FILE_MISSING", "The original audio file is missing", status_code=404
        )
    return path


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


@router.get("/tracks/{track_id}/audio")
def get_track_audio(
    track_id: str,
    session: Annotated[Session, Depends(get_db)],
    range_header: Annotated[str | None, Header(alias="Range")] = None,
) -> StreamingResponse:
    track = _get_track(session, track_id)
    path = _resolve_audio_path(track)
    size = path.stat().st_size
    start, end, status_code = 0, size - 1, 200
    if range_header:
        try:
            start, end = _parse_range(range_header, size)
        except (ValueError, TypeError):
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

    mime = mimetypes.guess_type(track.original_file_name)[0] or "application/octet-stream"
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1),
        "Content-Disposition": f"inline; filename*=UTF-8''{quote(track.original_file_name)}",
    }
    if status_code == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return StreamingResponse(chunks(), status_code=status_code, media_type=mime, headers=headers)


@router.head("/tracks/{track_id}/audio")
def head_track_audio(
    track_id: str, session: Annotated[Session, Depends(get_db)]
) -> Response:
    track = _get_track(session, track_id)
    path = _resolve_audio_path(track)
    mime = mimetypes.guess_type(track.original_file_name)[0] or "application/octet-stream"
    return Response(
        status_code=200,
        media_type=mime,
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(path.stat().st_size),
            "Content-Disposition": f"inline; filename*=UTF-8''{quote(track.original_file_name)}",
        },
    )


@router.get("/tracks/{track_id}/stems/{source}/audio")
def get_stem_audio(
    track_id: str,
    source: str,
    session: Annotated[Session, Depends(get_db)],
    range_header: Annotated[str | None, Header(alias="Range")] = None,
) -> StreamingResponse:
    track = _get_track(session, track_id)
    if source not in {"vocals", "drums", "bass", "other"}:
        raise BeatForgeError("INVALID_STEM", "Unknown separated source", status_code=404)
    settings = get_settings()
    stem_root = settings.stems_dir.resolve()
    path = (stem_root / track.id / f"{source}.flac").resolve()
    if not path.is_relative_to(stem_root) or not path.is_file():
        raise BeatForgeError(
            "STEM_NOT_READY",
            "This track has no persisted audio for the requested source",
            status_code=404,
        )
    size = path.stat().st_size
    start, end, status_code = 0, size - 1, 200
    if range_header:
        try:
            start, end = _parse_range(range_header, size)
        except (ValueError, TypeError):
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
        "Content-Disposition": f"inline; filename={source}.flac",
    }
    if status_code == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return StreamingResponse(
        chunks(), status_code=status_code, media_type="audio/flac", headers=headers
    )


@router.head("/tracks/{track_id}/stems/{source}/audio")
def head_stem_audio(
    track_id: str,
    source: str,
    session: Annotated[Session, Depends(get_db)],
) -> Response:
    track = _get_track(session, track_id)
    if source not in {"vocals", "drums", "bass", "other"}:
        raise BeatForgeError("INVALID_STEM", "Unknown separated source", status_code=404)
    settings = get_settings()
    stem_root = settings.stems_dir.resolve()
    path = (stem_root / track.id / f"{source}.flac").resolve()
    if not path.is_relative_to(stem_root) or not path.is_file():
        raise BeatForgeError(
            "STEM_NOT_READY",
            "This track has no persisted audio for the requested source",
            status_code=404,
        )
    return Response(
        status_code=200,
        media_type="audio/flac",
        headers={
            "Accept-Ranges": "bytes",
            "Content-Length": str(path.stat().st_size),
            "Content-Disposition": f"inline; filename={source}.flac",
        },
    )


@router.get("/tracks/{track_id}/waveform", response_model=WaveformResponse)
def get_waveform(
    track_id: str,
    session: Annotated[Session, Depends(get_db)],
    level: str = Query(default="auto", pattern=r"^(auto|\d+)$"),
    max_points: int = Query(default=4000, alias="maxPoints", ge=100, le=100_000),
    source: str = Query(default="mix", pattern=r"^(mix|vocals|drums|bass|other)$"),
) -> dict[str, Any]:
    track = _get_track(session, track_id)
    if not track.waveform_path:
        raise BeatForgeError(
            "WAVEFORM_NOT_READY", "Waveform peaks have not been generated yet", status_code=409
        )
    settings = get_settings()
    storage = settings.waveform_dir.resolve()
    try:
        path = resolve_storage_path(track.waveform_path, settings)
    except ValueError:
        path = storage.parent / "__invalid_waveform_path__"
    if not path.is_relative_to(storage) or not path.is_file():
        raise BeatForgeError(
            "WAVEFORM_FILE_MISSING", "Waveform peak data is missing", status_code=404
        )
    payload = read_waveform_lods(path)
    selected_payload = payload
    if source != "mix":
        selected_payload = payload.get("stems", {}).get(source)
        if not isinstance(selected_payload, dict):
            raise BeatForgeError(
                "STEM_WAVEFORM_NOT_READY",
                "Waveform peaks for the requested source are unavailable",
                status_code=404,
            )
    try:
        selected_level, selected = select_waveform_level(
            selected_payload, level, max_points
        )
    except (ValueError, KeyError):
        raise BeatForgeError(
            "WAVEFORM_LEVEL_NOT_FOUND",
            "The requested waveform level is unavailable",
            status_code=404,
        ) from None
    return {
        "track_id": track.id,
        "sample_rate": track.original_sample_rate,
        "sample_count": track.sample_count,
        "source": source,
        "level": selected_level,
        "window_size": int(selected.get("window_size", selected.get("windowSize", 1))),
        "mins": selected.get("mins", []),
        "maxs": selected.get("maxs", []),
    }


@router.get("/tracks/{track_id}/hit-points", response_model=list[HitPoint])
def list_hit_points(
    track_id: str, session: Annotated[Session, Depends(get_db)]
) -> list[dict[str, Any]]:
    track = _get_track(session, track_id)
    return [hit_dict(hit, track.original_sample_rate) for hit in track.hit_points]


@router.get("/tracks/{track_id}/candidate-events", response_model=list[CandidateEvent])
def list_candidate_events(
    track_id: str, session: Annotated[Session, Depends(get_db)]
) -> list[dict[str, Any]]:
    track = _get_track(session, track_id)
    return [
        candidate_event_dict(candidate, track.original_sample_rate)
        for candidate in track.candidate_events
    ]


@router.put("/tracks/{track_id}/hit-points", response_model=list[HitPoint])
def replace_hit_points(
    track_id: str,
    payload: HitPointBulkUpdate,
    session: Annotated[Session, Depends(get_db)],
) -> list[dict[str, Any]]:
    track = _get_track(session, track_id)
    requested_ids = [item.id for item in payload.hit_points if item.id]
    if len(requested_ids) != len(set(requested_ids)):
        raise BeatForgeError(
            "DUPLICATE_HIT_POINT_ID", "Hit point IDs must be unique", status_code=422
        )
    existing = {item.id: item for item in track.hit_points}
    requested_set = set(requested_ids)
    for item in list(track.hit_points):
        if item.id not in requested_set:
            track.hit_points.remove(item)
    for item in payload.hit_points:
        for name, value in (
            ("sample", item.sample),
            (
                "detectedSample",
                item.detected_sample if item.detected_sample is not None else item.sample,
            ),
            (
                "refinedSample",
                item.refined_sample if item.refined_sample is not None else item.sample,
            ),
            (
                "snappedSample",
                item.snapped_sample if item.snapped_sample is not None else item.sample,
            ),
        ):
            _validate_sample(track, value, name)
        if item.id and item.id in existing:
            _fill_hit_model(existing[item.id], item)
        else:
            if item.id and session.get(HitPointModel, item.id):
                raise BeatForgeError(
                    "HIT_POINT_ID_CONFLICT",
                    "A hit point with this ID belongs to another track",
                    status_code=409,
                )
            track.hit_points.append(_new_hit(track, item))
    _mark_edited(track)
    session.commit()
    session.refresh(track)
    track = _get_track(session, track_id)
    return [hit_dict(hit, track.original_sample_rate) for hit in track.hit_points]


@router.post("/tracks/{track_id}/hit-points", response_model=HitPoint, status_code=201)
def create_hit_point(
    track_id: str,
    payload: HitPointCreate,
    session: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    track = _get_track(session, track_id)
    if payload.id and session.get(HitPointModel, payload.id):
        raise BeatForgeError(
            "HIT_POINT_ID_CONFLICT", "Hit point ID already exists", status_code=409
        )
    hit = _new_hit(track, payload)
    hit.manually_edited = True
    track.hit_points.append(hit)
    _mark_edited(track)
    session.commit()
    return hit_dict(hit, track.original_sample_rate)


@router.patch("/tracks/{track_id}/hit-points/{hit_point_id}", response_model=HitPoint)
def patch_hit_point(
    track_id: str,
    hit_point_id: str,
    payload: HitPointPatch,
    session: Annotated[Session, Depends(get_db)],
) -> dict[str, Any]:
    track = _get_track(session, track_id)
    hit = next((item for item in track.hit_points if item.id == hit_point_id), None)
    if not hit:
        raise not_found("hit point", hit_point_id)
    moving_locked_hit = (
        hit.locked
        and payload.sample is not None
        and payload.sample != hit.sample
        and payload.locked is not False
    )
    if moving_locked_hit:
        raise BeatForgeError(
            "HIT_POINT_LOCKED", "Unlock the hit point before moving it", status_code=409
        )
    for field in (
        "sample",
        "acoustic_sample",
        "chart_sample",
        "detected_sample",
        "refined_sample",
        "snapped_sample",
    ):
        value = getattr(payload, field)
        if value is not None:
            _validate_sample(track, value, field)
    _fill_hit_model(hit, payload)
    acoustic_changed = payload.sample is not None or payload.acoustic_sample is not None
    chart_explicit = bool(
        {"chart_sample", "snapped_sample"} & payload.model_fields_set
    )
    if acoustic_changed and not chart_explicit:
        hit.chart_sample = _recommended_snap(track, hit.sample)
        hit.snapped_sample = hit.chart_sample
    if (
        acoustic_changed or chart_explicit
    ) and "snap_error_ms" not in payload.model_fields_set:
        chart_sample = hit.chart_sample if hit.chart_sample is not None else hit.snapped_sample
        hit.snap_error_ms = (
            (hit.sample - chart_sample) * 1000.0 / track.original_sample_rate
        )
    hit.manually_edited = True
    _mark_edited(track)
    session.commit()
    return hit_dict(hit, track.original_sample_rate)


@router.delete("/tracks/{track_id}/hit-points/{hit_point_id}", status_code=204)
def delete_hit_point(
    track_id: str,
    hit_point_id: str,
    session: Annotated[Session, Depends(get_db)],
) -> Response:
    track = _get_track(session, track_id)
    hit = next((item for item in track.hit_points if item.id == hit_point_id), None)
    if not hit:
        raise not_found("hit point", hit_point_id)
    session.delete(hit)
    _mark_edited(track)
    session.commit()
    return Response(status_code=204)


@router.patch("/tracks/{track_id}/tempo-map", response_model=list[TempoSegment])
def patch_tempo_map(
    track_id: str,
    payload: TempoMapUpdate,
    session: Annotated[Session, Depends(get_db)],
) -> list[dict[str, Any]]:
    track = _get_track(session, track_id)
    ids = [item.id for item in payload.tempo_map if item.id]
    if len(ids) != len(set(ids)):
        raise BeatForgeError(
            "DUPLICATE_TEMPO_SEGMENT_ID",
            "Tempo segment IDs must be unique",
            status_code=422,
        )
    starts = [item.start_sample for item in payload.tempo_map]
    if len(starts) != len(set(starts)):
        raise BeatForgeError(
            "DUPLICATE_TEMPO_START",
            "Tempo segment start samples must be unique",
            status_code=422,
        )
    for item in payload.tempo_map:
        _validate_sample(track, item.start_sample, "startSample")
    track.tempo_segments.clear()
    session.flush()
    for item in sorted(payload.tempo_map, key=lambda segment: segment.start_sample):
        track.tempo_segments.append(
            TempoSegmentModel(
                id=item.id or new_id(),
                start_sample=item.start_sample,
                bpm=item.bpm,
                time_signature_numerator=item.time_signature_numerator,
                time_signature_denominator=item.time_signature_denominator,
                beat_offset_sample=item.beat_offset_sample,
                confidence=item.confidence,
                manually_edited=True,
            )
        )
    for hit in track.hit_points:
        acoustic_sample = (
            hit.acoustic_sample if hit.acoustic_sample is not None else hit.refined_sample
        )
        hit.chart_sample = _recommended_snap(track, acoustic_sample)
        hit.snapped_sample = hit.chart_sample
        hit.snap_error_ms = (
            (acoustic_sample - hit.chart_sample) * 1000.0 / track.original_sample_rate
        )
    for candidate in track.candidate_events:
        candidate.chart_sample = _recommended_snap(track, candidate.acoustic_sample)
        candidate.snap_error_ms = (
            (candidate.acoustic_sample - candidate.chart_sample)
            * 1000.0
            / track.original_sample_rate
        )
        candidate.grid_confidence = math.exp(
            -0.5 * (abs(candidate.snap_error_ms) / 30.0) ** 2
        )
        semantic_evidence = load_json(candidate.semantic_evidence_json, {})
        if not isinstance(semantic_evidence, dict):
            semantic_evidence = {}
        semantic_evidence["beatConfidence"] = candidate.grid_confidence
        candidate.semantic_evidence_json = dumps(semantic_evidence)
        if candidate.generator == "hubert_ctc" and candidate.source == "vocals":
            hubert_evidence = load_json(candidate.evidence_json, {})
            if not isinstance(hubert_evidence, dict):
                hubert_evidence = {}
            hubert_evidence["rhythm"] = candidate.grid_confidence
            candidate.evidence_json = dumps(hubert_evidence)
    _mark_edited(track)
    session.commit()
    track = _get_track(session, track_id)
    from .serialization import tempo_dict

    return [tempo_dict(segment) for segment in track.tempo_segments]


@router.get("/tracks/{track_id}/export")
def export_track(
    track_id: str,
    session: Annotated[Session, Depends(get_db)],
    export_format: str = Query(alias="format", pattern=r"^(json|csv|package)$"),
    audio_mode: str = Query(
        default="reference", alias="audio", pattern=r"^(none|reference|full)$"
    ),
) -> Response:
    track = _get_track(session, track_id)
    source_stem = Path(track.original_file_name).stem
    filename = f"{source_stem}-beatforge"
    if export_format == "package":
        try:
            source_path = _resolve_audio_path(track)
        except BeatForgeError as exc:
            if exc.code != "AUDIO_FILE_MISSING":
                raise
            source_path = None
        package = build_package_export(
            track,
            source_path,
            get_settings(),
            audio_mode=audio_mode,
        )
        package_filename = f"{source_stem}.beatforge.zip"
        return FileResponse(
            package.archive_path,
            media_type="application/zip",
            headers={
                "Content-Disposition": _attachment_content_disposition(package_filename)
            },
            background=BackgroundTask(
                shutil.rmtree, package.temporary_root, ignore_errors=True
            ),
        )

    hits = [hit_dict(hit, track.original_sample_rate) for hit in track.hit_points]
    if export_format == "csv":
        output = io.StringIO(newline="")
        fieldnames = [
            "id",
            "sample",
            "acoustic_sample",
            "chart_sample",
            "time_sec",
            "band",
            "confidence",
            "salience",
            "source",
            "primary_stem",
            "snapped_sample",
            "snap_error_ms",
            "manually_edited",
            "locked",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for hit in hits:
            writer.writerow({field: hit[field] for field in fieldnames})
        return Response(
            output.getvalue().encode("utf-8-sig"),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": _attachment_content_disposition(f"{filename}.csv")
            },
        )

    payload = {
        "schemaVersion": "1.0",
        "project": project_dict(track.project),
        "audio": {
            "originalFileName": track.original_file_name,
            "sampleRate": track.original_sample_rate,
            "sampleCount": track.sample_count,
            "durationSec": track.duration_sec,
        },
        "tempoMap": [
            TempoSegment.model_validate(segment).model_dump(by_alias=True, mode="json")
            for segment in track.tempo_segments
        ],
        "hitPoints": [
            HitPoint.model_validate(hit).model_dump(by_alias=True, mode="json") for hit in hits
        ],
        "candidateEvents": [
            CandidateEvent.model_validate(
                candidate_event_dict(candidate, track.original_sample_rate)
            ).model_dump(by_alias=True, mode="json")
            for candidate in track.candidate_events
        ],
        "analysisMetadata": camelize_keys(
            sanitize_export_metadata(json.loads(track.analysis_json or "{}"))
        ),
    }
    return Response(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        media_type="application/json",
        headers={
            "Content-Disposition": _attachment_content_disposition(f"{filename}.json")
        },
    )
