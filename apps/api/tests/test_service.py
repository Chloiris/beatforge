from __future__ import annotations

from fastapi.testclient import TestClient
from sqlalchemy import func, select

from beatforge_api.config import get_settings
from beatforge_api.database import SessionLocal
from beatforge_api.models import ProjectModel, TrackModel
from beatforge_api.service import ensure_project_for_audio

from .test_api import wav_bytes


def test_seed_service_is_idempotent(client: TestClient) -> None:
    demo_dir = get_settings().storage_dir / "demo"
    demo_dir.mkdir(parents=True, exist_ok=True)
    audio_path = demo_dir / "idempotent.wav"
    audio_path.write_bytes(wav_bytes())
    first = ensure_project_for_audio(
        slug="idempotent",
        title="幂等测试",
        artist="BeatForge Lab",
        genre="Synthetic",
        cover_url="/api/assets/covers/idempotent.svg",
        audio_path=audio_path,
    )
    second = ensure_project_for_audio(
        slug="idempotent",
        title="幂等测试（更新）",
        artist="BeatForge Lab",
        genre="Synthetic",
        cover_url="/api/assets/covers/idempotent.svg",
        audio_path=audio_path,
    )
    assert first == second
    with SessionLocal() as session:
        assert session.scalar(select(func.count(ProjectModel.id))) == 1
        assert session.scalar(select(func.count(TrackModel.id))) == 1
        assert session.get(ProjectModel, first[0]).title == "幂等测试（更新）"
