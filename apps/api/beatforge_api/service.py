from __future__ import annotations

import argparse
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from .config import get_settings
from .database import SessionLocal, init_db
from .jobs import submit_analysis, wait_for_job
from .media import probe_audio
from .models import AnalysisJobModel, ProjectModel, TrackModel
from .serialization import job_dict
from .storage_paths import resolve_storage_path, storage_relative_path

SEED_NAMESPACE = uuid.UUID("efb4f834-c069-49a2-9f99-d070c8b63cab")


def ensure_project_for_audio(
    *,
    slug: str,
    title: str,
    artist: str,
    genre: str,
    cover_url: str,
    audio_path: str | Path,
) -> tuple[str, str]:
    """Idempotently create or refresh a local project without inspecting any ground truth."""
    init_db()
    settings = get_settings()
    path = Path(audio_path).expanduser().resolve()
    stored_path = storage_relative_path(path, settings)
    probe = probe_audio(path)
    project_id = str(uuid.uuid5(SEED_NAMESPACE, f"project:{slug}"))
    track_id = str(uuid.uuid5(SEED_NAMESPACE, f"track:{slug}"))
    with SessionLocal() as session:
        project = session.scalar(
            select(ProjectModel)
            .where(ProjectModel.id == project_id)
            .options(selectinload(ProjectModel.track))
        )
        if project is None:
            project = ProjectModel(
                id=project_id,
                title=title,
                artist=artist,
                genre=genre,
                cover_url=cover_url,
                status="unprocessed",
            )
            session.add(project)
        else:
            project.title = title
            project.artist = artist
            project.genre = genre
            project.cover_url = cover_url
            project.updated_at = datetime.now(UTC)
        track = project.track
        if track is None:
            track = TrackModel(
                id=track_id,
                project=project,
                original_file_name=path.name,
                stored_file_name=path.name,
                file_path=stored_path,
                format=probe.format,
                original_sample_rate=probe.sample_rate,
                channels=probe.channels,
                sample_count=probe.sample_count,
                duration_sec=probe.duration_sec,
                leading_silence_samples=0,
            )
            session.add(track)
        else:
            track.original_file_name = path.name
            track.stored_file_name = path.name
            track.file_path = stored_path
            if track.waveform_path:
                try:
                    track.waveform_path = storage_relative_path(
                        resolve_storage_path(track.waveform_path, settings), settings
                    )
                except ValueError:
                    # Keep the legacy value so the API can report a missing cache
                    # instead of discarding the user's persisted reference.
                    pass
            track.format = probe.format
            track.original_sample_rate = probe.sample_rate
            track.channels = probe.channels
            track.sample_count = probe.sample_count
            track.duration_sec = probe.duration_sec
            track.updated_at = datetime.now(UTC)
        session.commit()
    return project_id, track_id


def ensure_analysis(
    track_id: str,
    *,
    mode: str = "balanced",
    sensitivity: float = 0.5,
    force: bool = False,
    timeout: float = 900.0,
) -> dict[str, Any]:
    """Run the same persisted analysis job used by the API and wait for its real result."""
    init_db()
    with SessionLocal() as session:
        track = session.get(TrackModel, track_id)
        if track is None:
            raise ValueError(f"track not found: {track_id}")
        if not force and track.analysis_json != "{}" and track.hit_points:
            completed = session.scalar(
                select(AnalysisJobModel)
                .where(
                    AnalysisJobModel.track_id == track_id,
                    AnalysisJobModel.status == "completed",
                )
                .order_by(AnalysisJobModel.updated_at.desc())
            )
            if completed:
                return {**job_dict(completed), "skipped": True}
        running = session.scalar(
            select(AnalysisJobModel).where(
                AnalysisJobModel.track_id == track_id,
                AnalysisJobModel.status.in_(["queued", "processing"]),
            )
        )
        if running:
            running.status = "failed"
            running.stage = "failed"
            running.error_json = json.dumps(
                {
                    "code": "STALE_ANALYSIS_JOB",
                    "message": "A seed run replaced an analysis job without a live worker.",
                }
            )
        job = AnalysisJobModel(
            track=track,
            mode=mode,
            sensitivity=min(1.0, max(0.0, sensitivity)),
        )
        track.project.status = "processing"
        session.add(job)
        session.commit()
        job_id = job.id
    submit_analysis(job_id)
    wait_for_job(job_id, timeout=timeout)
    with SessionLocal() as session:
        job = session.get(AnalysisJobModel, job_id)
        assert job is not None
        result = job_dict(job)
        result["skipped"] = False
        if job.status != "completed":
            raise RuntimeError(f"analysis failed: {result.get('error')}")
        return result


def _main() -> None:
    parser = argparse.ArgumentParser(description="BeatForge local persistence service")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init-db")
    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("track_id")
    analyze.add_argument(
        "--mode", default="balanced", choices=("recall", "balanced", "clean", "accurate")
    )
    analyze.add_argument("--sensitivity", type=float, default=0.5)
    analyze.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    if arguments.command == "init-db":
        init_db()
        print(json.dumps({"status": "ok"}))
    elif arguments.command == "analyze":
        print(
            json.dumps(
                ensure_analysis(
                    arguments.track_id,
                    mode=arguments.mode,
                    sensitivity=arguments.sensitivity,
                    force=arguments.force,
                ),
                default=str,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    _main()
