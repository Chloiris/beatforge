from __future__ import annotations

import os
import unicodedata
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import AlignmentHierarchy, AlignmentMethodId, AlignmentStatus, AlignmentToken

TOKEN_NAMESPACE = uuid.UUID("cfe4f693-f1bf-41db-8514-eecf22615b7a")


@dataclass(frozen=True, slots=True)
class TempoReference:
    start_sample: int
    bpm: float
    beat_offset_sample: int


@dataclass(frozen=True, slots=True)
class AlignmentContext:
    track_id: str
    lyrics: str
    lyrics_format: str
    vocals_path: Path
    sample_rate: int
    sample_count: int
    tempo_map: tuple[TempoReference, ...]
    models_dir: Path
    storage_dir: Path
    project_root: Path
    song: str = ""
    artist: str = ""
    qwen_payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AdapterDiagnostics:
    available: bool
    reason: str | None = None
    model: str | None = None
    automatic_downloads_enabled: bool = False
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AdapterOutput:
    tokens: tuple[AlignmentToken, ...]
    hierarchy: AlignmentHierarchy | None = None
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class AlignmentAdapterError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: AlignmentStatus = "failed",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.details = details


class AlignmentAdapter(ABC):
    method: AlignmentMethodId
    name: str

    @abstractmethod
    def diagnostics(self, context: AlignmentContext | None = None) -> AdapterDiagnostics:
        raise NotImplementedError

    @abstractmethod
    def run(self, context: AlignmentContext) -> AdapterOutput:
        raise NotImplementedError


def alignment_token_id(
    track_id: str,
    method: AlignmentMethodId,
    index: int,
    start_sample: int,
    end_sample: int,
) -> str:
    value = f"{track_id}:{method}:{index}:{start_sample}:{end_sample}"
    return str(uuid.uuid5(TOKEN_NAMESPACE, value))


def clean_lyrics(text: str, input_format: str = "japanese") -> str:
    """Remove formatting metadata without inventing or redistributing lyric timing."""

    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for raw_line in normalized.splitlines():
        line = raw_line.strip()
        if input_format == "lrc":
            while line.startswith("[") and "]" in line:
                closing = line.find("]")
                tag = line[1:closing]
                if not any(character.isdigit() for character in tag):
                    line = line[closing + 1 :].strip()
                    continue
                if ":" in tag:
                    line = line[closing + 1 :].strip()
                    continue
                break
        if line:
            lines.append(line)
    return "\n".join(lines)


def lyric_units(text: str) -> list[str]:
    """Return deterministic coverage units; this function never assigns timestamps."""

    units: list[str] = []
    latin_buffer: list[str] = []

    def flush_latin() -> None:
        if latin_buffer:
            units.append("".join(latin_buffer).casefold())
            latin_buffer.clear()

    for character in unicodedata.normalize("NFKC", text):
        category = unicodedata.category(character)
        if character.isascii() and character.isalnum():
            latin_buffer.append(character)
            continue
        flush_latin()
        if category[0] in {"L", "N"}:
            units.append(character.casefold())
    flush_latin()
    return units


def executable_from_environment(name: str, default: Path | str) -> Path:
    value = Path(os.environ.get(name, str(default))).expanduser()
    return value if value.is_absolute() else Path.cwd() / value
