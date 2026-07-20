"""Optional stem-separation boundary.

The base installation never imports Demucs. Applications can inject a separator
implementing this interface; accurate mode transparently falls back when it is not
available or cannot produce stems.
"""

from __future__ import annotations

import importlib.util
import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np

from .models import FloatArray


@dataclass(slots=True)
class SeparationResult:
    stems: dict[str, FloatArray]
    warning: str | None = None
    model_name: str | None = None
    device: str | None = None


class StemSeparator(ABC):
    @property
    @abstractmethod
    def available(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def separate(self, audio: FloatArray, sample_rate: int) -> SeparationResult:
        raise NotImplementedError


class NoopSeparator(StemSeparator):
    @property
    def available(self) -> bool:
        return False

    def separate(self, audio: FloatArray, sample_rate: int) -> SeparationResult:
        return SeparationResult(
            stems={},
            warning="精确模式依赖 Demucs；当前环境不可用，已回退到平衡模式。",
        )


class DemucsSeparator(StemSeparator):
    """CPU Demucs adapter that only uses weights already present on disk.

    ``demucs.pretrained.get_model`` normally downloads missing weights. BeatForge
    disables that code path explicitly, so selecting accurate mode can never start
    an implicit model download or require an API key.
    """

    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        self.model_name = model_name or os.environ.get("BEATFORGE_DEMUCS_MODEL", "htdemucs")
        configured_device = device or os.environ.get("BEATFORGE_DEMUCS_DEVICE", "auto")
        self.device = self._resolve_device(configured_device)
        self._model: Any | None = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except (ImportError, AttributeError):
            pass
        return "cpu"

    @staticmethod
    def _checkpoint_directories() -> tuple[Path, ...]:
        directories: list[Path] = []
        try:
            import torch

            directories.append(Path(torch.hub.get_dir()) / "checkpoints")
        except ImportError:
            pass
        return tuple(directories)

    @property
    def available(self) -> bool:
        if importlib.util.find_spec("demucs") is None:
            return False
        if importlib.util.find_spec("torch") is None:
            return False
        # Demucs signatures are hashed filenames, so the adapter deliberately does
        # not guess one exact filename. Loading is still protected by _no_download.
        return any(
            directory.is_dir() and any(directory.glob("*.th"))
            for directory in self._checkpoint_directories()
        )

    @staticmethod
    @contextmanager
    def _no_download() -> Any:
        def reject_download(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError(
                "Demucs model weights are not cached; automatic download is disabled"
            )

        with patch("torch.hub.download_url_to_file", side_effect=reject_download):
            yield

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.available:
            raise RuntimeError("Demucs dependency or local model weights are unavailable")
        from demucs.pretrained import get_model

        with self._no_download():
            model = get_model(self.model_name)
        model.to(self.device)
        model.eval()
        self._model = model
        return model

    def separate(self, audio: FloatArray, sample_rate: int) -> SeparationResult:
        import torch
        from demucs.apply import apply_model

        model = self._get_model()
        values = np.asarray(audio, dtype=np.float32)
        if values.ndim == 1:
            channel_first = np.repeat(values[np.newaxis, :], 2, axis=0)
        elif values.ndim == 2 and values.shape[1] in {1, 2}:
            channel_first = values.T
            if channel_first.shape[0] == 1:
                channel_first = np.repeat(channel_first, 2, axis=0)
        else:
            raise ValueError("DemucsSeparator expects mono or channel-last stereo audio")
        model_sample_rate = int(getattr(model, "samplerate", 44_100))
        if sample_rate != model_sample_rate:
            raise ValueError(
                f"Demucs model expects {model_sample_rate} Hz, received {sample_rate} Hz"
            )
        waveform = torch.from_numpy(np.ascontiguousarray(channel_first)).to(self.device)
        waveform = waveform.unsqueeze(0)
        mean = waveform.mean(dim=-1, keepdim=True)
        standard_deviation = waveform.std(dim=-1, keepdim=True).clamp_min(1e-8)
        normalized = (waveform - mean) / standard_deviation
        with torch.inference_mode():
            separated = apply_model(
                model,
                normalized,
                device=self.device,
                shifts=0,
                split=True,
                overlap=0.25,
                progress=False,
            )[0]
        separated = separated * standard_deviation[0] + mean[0]
        source_names = list(getattr(model, "sources", []))
        stems: dict[str, FloatArray] = {}
        for index, source_name in enumerate(source_names):
            if source_name not in {"drums", "bass", "other", "vocals"}:
                continue
            mono = separated[index].mean(dim=0).detach().cpu().numpy()
            stems[source_name] = np.asarray(mono[: values.shape[0]], dtype=np.float32)
        return SeparationResult(
            stems=stems,
            model_name=self.model_name,
            device=self.device,
        )
