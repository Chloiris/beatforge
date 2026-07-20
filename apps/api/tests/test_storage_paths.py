from __future__ import annotations

from pathlib import Path

import pytest

from beatforge_api.config import Settings
from beatforge_api.storage_paths import resolve_storage_path


def _settings(storage_dir: Path) -> Settings:
    return Settings(
        project_root=storage_dir.parent,
        storage_dir=storage_dir,
        database_url="sqlite:///:memory:",
        max_upload_bytes=1,
        allowed_origins=(),
    )


@pytest.mark.parametrize(
    "legacy_path",
    (
        "/app/storage/audio/portable.wav",
        r"C:\app\storage\audio\portable.wav",
    ),
)
def test_resolve_storage_path_rebases_foreign_absolute_path_syntax(
    tmp_path: Path, legacy_path: str
) -> None:
    storage_dir = tmp_path / "storage"

    assert resolve_storage_path(legacy_path, _settings(storage_dir)) == (
        storage_dir / "audio" / "portable.wav"
    ).resolve()


@pytest.mark.parametrize(
    "legacy_path",
    (
        "/app/storage/../private.wav",
        r"C:\app\storage\..\private.wav",
    ),
)
def test_resolve_storage_path_rejects_foreign_path_traversal(
    tmp_path: Path, legacy_path: str
) -> None:
    with pytest.raises(ValueError, match="outside"):
        resolve_storage_path(legacy_path, _settings(tmp_path / "storage"))
