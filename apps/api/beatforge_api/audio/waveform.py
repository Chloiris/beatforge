"""Multi-resolution min/max waveform peak generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .io import AudioDecodeError


def _aggregate_min_max(samples: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    if samples.size == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)
    bin_count = int(np.ceil(samples.size / window_size))
    padded_size = bin_count * window_size
    if padded_size != samples.size:
        samples = np.pad(samples, (0, padded_size - samples.size), mode="edge")
    frames = samples.reshape(bin_count, window_size)
    return (
        np.min(frames, axis=1).astype(np.float32),
        np.max(frames, axis=1).astype(np.float32),
    )


def waveform_lods_from_samples(
    samples: np.ndarray,
    *,
    base_window: int = 256,
    max_levels: int = 8,
) -> list[dict[str, Any]]:
    """Build JSON-ready LOD levels in original-sample coordinates."""

    values = np.asarray(samples, dtype=np.float32)
    if values.ndim == 2:
        values = np.mean(values, axis=1, dtype=np.float32)
    if values.ndim != 1:
        raise ValueError("Waveform samples must be mono or channel-last")
    if base_window <= 0 or max_levels <= 0:
        raise ValueError("Waveform LOD parameters must be positive")
    levels: list[dict[str, Any]] = []
    window_size = int(base_window)
    for level in range(max_levels):
        minima, maxima = _aggregate_min_max(values, window_size)
        levels.append(
            {
                "level": level,
                "window_size": window_size,
                "mins": minima.round(7).tolist(),
                "maxs": maxima.round(7).tolist(),
            }
        )
        if minima.size <= 32:
            break
        window_size *= 4
    return levels


def build_waveform_lods(
    path: str | Path,
    *,
    base_window: int = 256,
    max_levels: int = 8,
) -> list[dict[str, Any]]:
    """Decode a source with libsndfile and generate min/max peak LODs."""

    try:
        samples, _ = sf.read(path, dtype="float32", always_2d=True)
    except (RuntimeError, OSError) as exc:
        raise AudioDecodeError(f"Unable to decode waveform source: {exc}") from exc
    return waveform_lods_from_samples(
        samples,
        base_window=base_window,
        max_levels=max_levels,
    )
