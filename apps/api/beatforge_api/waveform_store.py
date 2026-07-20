from __future__ import annotations

import gzip
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .serialization import json_safe


def write_waveform_lods(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6) as compressed:
                compressed.write(
                    json.dumps(json_safe(payload), separators=(",", ":")).encode("utf-8")
                )
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return path


def read_waveform_lods(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def select_waveform_level(
    payload: dict[str, Any], level: str, max_points: int
) -> tuple[int, dict[str, Any]]:
    levels: list[dict[str, Any]] = payload.get("levels", [])
    if not levels:
        raise ValueError("waveform has no levels")
    if level != "auto":
        requested = int(level)
        for item in levels:
            if int(item.get("level", -1)) == requested:
                return requested, item
        raise KeyError(requested)
    ordered = sorted(
        levels,
        key=lambda item: int(item.get("window_size", item.get("windowSize", 0))),
    )
    for item in ordered:
        if len(item.get("mins", [])) <= max_points:
            return int(item.get("level", 0)), item
    item = ordered[-1]
    return int(item.get("level", len(ordered) - 1)), item
