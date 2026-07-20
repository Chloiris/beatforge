#!/usr/bin/env python3
"""Idempotently seed demo projects and run the production analyzer.

This script reads only the public demo manifest. Ground-truth onset files are never
imported and are reserved for the separate evaluation command.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _bootstrap_api(root: Path) -> None:
    api_path = str(root / "apps" / "api")
    if api_path not in sys.path:
        sys.path.insert(0, api_path)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=root / "storage/demo/manifest.json")
    parser.add_argument("--mode", choices=("recall", "balanced", "clean", "accurate"), default="balanced")
    parser.add_argument("--sensitivity", type=float, default=0.5)
    parser.add_argument("--force-analysis", action="store_true")
    args = parser.parse_args()
    if not args.manifest.exists():
        raise SystemExit("Demo manifest is missing; run scripts/generate_demo_audio.py first")

    _bootstrap_api(root)
    from sqlalchemy import select
    from sqlalchemy.orm import selectinload

    from beatforge_api.database import SessionLocal
    from beatforge_api.models import ProjectModel, TrackModel
    from beatforge_api.schemas import ProjectResponse, TrackResponse
    from beatforge_api.serialization import project_dict, track_dict
    from beatforge_api.service import ensure_analysis, ensure_project_for_audio

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    seeded: list[dict[str, object]] = []
    for item in manifest["tracks"]:
        slug = str(item["slug"])
        audio_path = root / str(item["audioFile"])
        project_id, track_id = ensure_project_for_audio(
            slug=slug,
            title=str(item["title"]),
            artist=str(item["artist"]),
            genre=str(item["genre"]),
            cover_url=f"/api/assets/covers/{slug}.svg",
            audio_path=audio_path,
        )
        job = ensure_analysis(
            track_id,
            mode=args.mode,
            sensitivity=args.sensitivity,
            force=args.force_analysis,
        )
        with SessionLocal() as session:
            project = session.scalar(
                select(ProjectModel)
                .where(ProjectModel.id == project_id)
                .options(
                    selectinload(ProjectModel.track).selectinload(TrackModel.hit_points),
                    selectinload(ProjectModel.track).selectinload(TrackModel.tempo_segments),
                )
            )
            if project is None or project.track is None:
                raise RuntimeError(f"Seeded project disappeared: {slug}")
            project_payload = ProjectResponse.model_validate(project_dict(project)).model_dump(
                by_alias=True, mode="json"
            )
            track_payload = TrackResponse.model_validate(track_dict(project.track)).model_dump(
                by_alias=True, mode="json"
            )
        snapshot = {
            "schemaVersion": "1.0",
            "slug": slug,
            "project": project_payload,
            "track": track_payload,
            "tempoMap": track_payload["tempoMap"],
            "hitPoints": track_payload["hitPoints"],
            "analysisMetadata": track_payload["analysis"],
        }
        analysis_path = root / "storage/analyses" / f"{slug}.analysis.json"
        analysis_path.parent.mkdir(parents=True, exist_ok=True)
        analysis_path.write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        row = {
            "slug": slug,
            "projectId": project_id,
            "trackId": track_id,
            "hitPointCount": len(track_payload["hitPoints"]),
            "bpm": track_payload["tempoMap"][0]["bpm"],
        }
        seeded.append(row)
        reused = bool(job.get("skipped"))
        print(
            f"{item['title']}: {row['hitPointCount']} hits, {row['bpm']:.3f} BPM"
            f" ({'reused' if reused else 'analyzed'})"
        )
    summary_path = root / "reports/demo-seed.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps({"schemaVersion": "1.0", "projects": seeded}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
