from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
_WINDOWS_HOME_IN_TEXT = re.compile(
    r"[A-Za-z]:[\\/](?:Users|Documents and Settings)[\\/][^\s\"'<>]+",
    re.IGNORECASE,
)
_POSIX_HOME_IN_TEXT = re.compile(
    r"/(?:Users|home|var/folders|private/var/folders)/[^\s\"'<>]+"
)


def _looks_like_local_path(value: str) -> bool:
    candidate = value.strip()
    return candidate.startswith(("/", "~/", "./", "../", "file://")) or bool(
        _WINDOWS_ABSOLUTE.match(candidate)
    )


def public_model_identifier(value: str | None) -> str | None:
    """Keep remote model IDs intact while removing a local directory prefix."""

    if value is None or not _looks_like_local_path(value):
        return value
    normalized = value.replace("\\", "/").rstrip("/")
    return normalized.rsplit("/", 1)[-1] or "local-model"


def _redacted_path(value: str) -> str:
    name = public_model_identifier(value)
    return f"<local-path>/{name}" if name else "<local-path>"


def _redact_embedded_home_paths(value: str) -> str:
    redacted = _WINDOWS_HOME_IN_TEXT.sub("<local-path>", value)
    return _POSIX_HOME_IN_TEXT.sub("<local-path>", redacted)


def sanitize_export_metadata(value: Any) -> Any:
    """Remove machine-local paths from user-shareable JSON metadata."""

    if isinstance(value, Mapping):
        return {str(key): sanitize_export_metadata(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
        return [sanitize_export_metadata(item) for item in value]
    if isinstance(value, str):
        if _looks_like_local_path(value):
            return _redacted_path(value)
        return _redact_embedded_home_paths(value)
    return value
