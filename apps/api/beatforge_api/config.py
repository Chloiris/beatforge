from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    project_root: Path
    storage_dir: Path
    database_url: str
    max_upload_bytes: int
    allowed_origins: tuple[str, ...]

    @property
    def audio_dir(self) -> Path:
        return self.storage_dir / "audio"

    @property
    def waveform_dir(self) -> Path:
        return self.storage_dir / "waveform"

    @property
    def analyses_dir(self) -> Path:
        return self.storage_dir / "analyses"

    @property
    def stems_dir(self) -> Path:
        return self.storage_dir / "stems"

    @property
    def models_dir(self) -> Path:
        return self.storage_dir / "models"

    @property
    def vocal_alignment_dir(self) -> Path:
        return self.storage_dir / "vocal-alignment"

    @property
    def alignment_dir(self) -> Path:
        return self.storage_dir / "alignment"

    def ensure_directories(self) -> None:
        for directory in (
            self.storage_dir,
            self.audio_dir,
            self.waveform_dir,
            self.analyses_dir,
            self.stems_dir,
            self.models_dir,
            self.vocal_alignment_dir,
            self.alignment_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    inferred_project_root = Path(__file__).resolve().parents[3]
    configured_project_root = os.environ.get("BEATFORGE_PROJECT_ROOT", "").strip()
    if configured_project_root:
        project_value = Path(configured_project_root).expanduser()
        project_root = (
            project_value.resolve()
            if project_value.is_absolute()
            else (Path.cwd() / project_value).resolve()
        )
    else:
        project_root = inferred_project_root
    load_dotenv(project_root / ".env", override=False)
    storage_value = Path(
        os.environ.get("BEATFORGE_STORAGE_DIR", str(project_root / "storage"))
    ).expanduser()
    storage_dir = (
        storage_value.resolve()
        if storage_value.is_absolute()
        else (project_root / storage_value).resolve()
    )
    database_url = os.environ.get(
        "BEATFORGE_DATABASE_URL", f"sqlite:///{storage_dir / 'beatforge.db'}"
    )
    sqlite_prefix = "sqlite:///"
    if database_url.startswith(sqlite_prefix):
        database_path = database_url.removeprefix(sqlite_prefix)
        if database_path != ":memory:" and not Path(database_path).is_absolute():
            database_url = f"{sqlite_prefix}{(project_root / database_path).resolve()}"
    origins = tuple(
        value.strip()
        for value in os.environ.get(
            "BEATFORGE_ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ).split(",")
        if value.strip()
    )
    return Settings(
        project_root=project_root,
        storage_dir=storage_dir,
        database_url=database_url,
        max_upload_bytes=int(os.environ.get("BEATFORGE_MAX_UPLOAD_BYTES", 250 * 1024 * 1024)),
        allowed_origins=origins,
    )
