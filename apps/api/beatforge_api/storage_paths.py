from __future__ import annotations

from pathlib import Path

from .config import Settings


def storage_relative_path(path: str | Path, settings: Settings) -> str:
    """Serialize a path relative to storage so local and container runs can share SQLite."""
    root = settings.storage_dir.resolve()
    value = Path(path).expanduser()
    resolved = value.resolve() if value.is_absolute() else (root / value).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("path is outside the configured storage directory")
    return resolved.relative_to(root).as_posix()


def resolve_storage_path(value: str | Path, settings: Settings) -> Path:
    """Resolve current and legacy storage paths without allowing directory traversal.

    Older builds persisted absolute paths. When SQLite was shared with Docker, those
    values looked like ``/app/storage/...`` on the host. The storage suffix is safe
    to rebase because the final candidate must still resolve below the current root.
    """

    root = settings.storage_dir.resolve()
    raw = Path(value).expanduser()
    candidates: list[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
        storage_indexes = [
            index for index, part in enumerate(raw.parts) if part.casefold() == "storage"
        ]
        if storage_indexes:
            suffix = raw.parts[storage_indexes[-1] + 1 :]
            if suffix:
                candidates.append(root.joinpath(*suffix))
    else:
        candidates.append(root / raw)

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_relative_to(root):
            return resolved
    raise ValueError("stored path is outside the configured storage directory")
