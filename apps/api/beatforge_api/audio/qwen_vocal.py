"""Optional, local-only Qwen3 vocal transcription and forced alignment.

This module deliberately has no import-time dependency on Qwen, Transformers, or
Hugging Face. Model references are resolved to existing local cache snapshots before
the official ``qwen-asr`` loader is called, so analysis can never trigger an implicit
model download. The adapter first tries Apple MPS and retries once on CPU when an MPS
load or inference operation is unsupported.

Qwen3-ForcedAligner returns Japanese words/characters, not phonemes or morae. Callers
that need romaji/mora beat points should perform Japanese reading conversion after
alignment and keep the returned integer sample positions as the timing source of truth.
"""

from __future__ import annotations

import gc
import importlib.metadata
import importlib.util
import os
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray

VocalStatus = Literal["ok", "unavailable", "failed"]
FloatArray = NDArray[np.float32]

DEFAULT_ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"
DEFAULT_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"


@dataclass(frozen=True, slots=True)
class QwenVocalConfig:
    """Configuration for the optional official Qwen runtime."""

    asr_model: str = DEFAULT_ASR_MODEL
    aligner_model: str = DEFAULT_ALIGNER_MODEL
    device: Literal["auto", "mps", "cpu"] = "auto"
    max_inference_batch_size: int = 1
    max_new_tokens: int = 512
    max_alignment_seconds: float = 300.0

    @classmethod
    def from_environment(cls) -> QwenVocalConfig:
        device = os.environ.get("BEATFORGE_QWEN_DEVICE", "auto").lower()
        if device not in {"auto", "mps", "cpu"}:
            device = "auto"
        return cls(
            asr_model=os.environ.get("BEATFORGE_QWEN_ASR_MODEL", DEFAULT_ASR_MODEL),
            aligner_model=os.environ.get(
                "BEATFORGE_QWEN_ALIGNER_MODEL", DEFAULT_ALIGNER_MODEL
            ),
            device=device,  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class VocalRuntimeDiagnostics:
    """Non-loading runtime and cache availability information."""

    available: bool
    asr_available: bool
    aligner_available: bool
    dependency_available: bool
    dependency_version: str | None
    preferred_device: str
    asr_model: str
    aligner_model: str
    asr_model_path: str | None
    aligner_model_path: str | None
    automatic_downloads_enabled: bool
    issues: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BackendTranscription:
    """Small backend-neutral transcription value used by injected runners."""

    text: str
    language: str


@dataclass(frozen=True, slots=True)
class BackendAlignmentUnit:
    """Backend-neutral alignment span in seconds."""

    text: str
    start_sec: float
    end_sec: float


@dataclass(frozen=True, slots=True)
class VocalTimestamp:
    """Aligned text span with integer original-audio sample positions."""

    text: str
    start_sample: int
    end_sample: int
    start_sec: float
    end_sec: float


@dataclass(frozen=True, slots=True)
class VocalTranscriptionResult:
    """Structured result that never represents missing dependencies as success."""

    status: VocalStatus
    text: str = ""
    language: str = ""
    timestamps: tuple[VocalTimestamp, ...] = ()
    aligned: bool = False
    model: str | None = None
    aligner_model: str | None = None
    device: str | None = None
    warnings: tuple[str, ...] = ()
    error_code: str | None = None
    error_message: str | None = None

    @property
    def available(self) -> bool:
        return self.status == "ok"


@dataclass(frozen=True, slots=True)
class VocalAlignmentResult:
    """Structured known-text forced-alignment result."""

    status: VocalStatus
    text: str = ""
    timestamps: tuple[VocalTimestamp, ...] = ()
    model: str | None = None
    device: str | None = None
    warnings: tuple[str, ...] = ()
    error_code: str | None = None
    error_message: str | None = None

    @property
    def available(self) -> bool:
        return self.status == "ok"


class QwenVocalRunner(Protocol):
    """Dependency-injection boundary around the official package."""

    def load_asr(
        self,
        model_path: Path,
        *,
        device: str,
        max_inference_batch_size: int,
        max_new_tokens: int,
    ) -> Any: ...

    def load_aligner(self, model_path: Path, *, device: str) -> Any: ...

    def transcribe(
        self,
        model: Any,
        audio: FloatArray,
        sample_rate: int,
        *,
        language: str,
        context: str,
    ) -> BackendTranscription: ...

    def align(
        self,
        model: Any,
        audio: FloatArray,
        sample_rate: int,
        *,
        text: str,
        language: str,
    ) -> Sequence[BackendAlignmentUnit]: ...


class OfficialQwenRunner:
    """Thin lazy wrapper around the public ``qwen-asr==0.0.6`` API."""

    @staticmethod
    def _dtype_for_device(device: str) -> Any:
        import torch

        # float32 is the safest common dtype for current macOS MPS and CPU runtimes.
        return torch.float32

    def load_asr(
        self,
        model_path: Path,
        *,
        device: str,
        max_inference_batch_size: int,
        max_new_tokens: int,
    ) -> Any:
        from qwen_asr import Qwen3ASRModel

        return Qwen3ASRModel.from_pretrained(
            str(model_path),
            dtype=self._dtype_for_device(device),
            device_map=device,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
        )

    def load_aligner(self, model_path: Path, *, device: str) -> Any:
        from qwen_asr import Qwen3ForcedAligner

        return Qwen3ForcedAligner.from_pretrained(
            str(model_path),
            dtype=self._dtype_for_device(device),
            device_map=device,
        )

    def transcribe(
        self,
        model: Any,
        audio: FloatArray,
        sample_rate: int,
        *,
        language: str,
        context: str,
    ) -> BackendTranscription:
        results = model.transcribe(
            audio=(audio, sample_rate),
            context=context,
            language=language,
            return_time_stamps=False,
        )
        if not results:
            return BackendTranscription(text="", language=language)
        result = results[0]
        return BackendTranscription(
            text=str(getattr(result, "text", "")),
            language=str(getattr(result, "language", language)),
        )

    def align(
        self,
        model: Any,
        audio: FloatArray,
        sample_rate: int,
        *,
        text: str,
        language: str,
    ) -> Sequence[BackendAlignmentUnit]:
        results = model.align(
            audio=(audio, sample_rate),
            text=text,
            language=language,
        )
        if not results:
            return ()
        first_result = results[0]
        items = getattr(first_result, "items", first_result)
        return tuple(
            BackendAlignmentUnit(
                text=str(getattr(item, "text", "")),
                start_sec=float(getattr(item, "start_time", 0.0)),
                end_sec=float(getattr(item, "end_time", 0.0)),
            )
            for item in items
        )


def _huggingface_cache_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    for variable in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE"):
        value = os.environ.get(variable)
        if value:
            roots.append(Path(value).expanduser())
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home).expanduser() / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return tuple(dict.fromkeys(roots))


def resolve_cached_model(reference: str) -> Path | None:
    """Resolve a local directory or Hugging Face cache snapshot without networking."""

    explicit_path = Path(reference).expanduser()
    if explicit_path.is_dir() and (explicit_path / "config.json").is_file():
        return explicit_path.resolve()
    if "/" not in reference or reference.startswith((".", "~", "/")):
        return None

    cache_name = "models--" + reference.replace("/", "--")
    for root in _huggingface_cache_roots():
        model_root = root / cache_name
        snapshots_root = model_root / "snapshots"
        candidates: list[Path] = []
        main_ref = model_root / "refs" / "main"
        if main_ref.is_file():
            try:
                revision = main_ref.read_text(encoding="utf-8").strip()
            except OSError:
                revision = ""
            if revision:
                candidates.append(snapshots_root / revision)
        if snapshots_root.is_dir():
            candidates.extend(
                sorted(
                    (path for path in snapshots_root.iterdir() if path.is_dir()),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
            )
        for candidate in candidates:
            if (candidate / "config.json").is_file():
                return candidate.resolve()
    return None


def _dependency_available() -> bool:
    return importlib.util.find_spec("qwen_asr") is not None


def _dependency_version() -> str | None:
    try:
        return importlib.metadata.version("qwen-asr")
    except importlib.metadata.PackageNotFoundError:
        return None


def _default_device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
    except (ImportError, AttributeError):
        pass
    return "cpu"


class QwenVocalAnalyzer:
    """Lazy, local-only vocal ASR and Japanese known-lyrics alignment adapter."""

    def __init__(
        self,
        config: QwenVocalConfig | None = None,
        *,
        runner: QwenVocalRunner | None = None,
        model_resolver: Callable[[str], Path | None] = resolve_cached_model,
        dependency_probe: Callable[[], bool] | None = None,
        version_probe: Callable[[], str | None] = _dependency_version,
        device_probe: Callable[[], str] = _default_device,
    ) -> None:
        self.config = config or QwenVocalConfig.from_environment()
        self._runner = runner or OfficialQwenRunner()
        self._model_resolver = model_resolver
        self._dependency_probe = dependency_probe or (
            (lambda: True) if runner is not None else _dependency_available
        )
        self._version_probe = version_probe
        self._device_probe = device_probe
        self._asr_models: dict[str, Any] = {}
        self._aligner_models: dict[str, Any] = {}
        self._asr_mps_disabled = False
        self._aligner_mps_disabled = False
        self._lock = threading.RLock()

    def release_cached_models(
        self,
        *,
        asr: bool = True,
        aligner: bool = True,
    ) -> None:
        """Release loaded local models between heavyweight pipeline stages."""

        with self._lock:
            if asr:
                self._asr_models.clear()
            if aligner:
                self._aligner_models.clear()
        gc.collect()
        try:
            import torch

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except (ImportError, AttributeError, RuntimeError):
            pass

    def _preferred_device(self) -> str:
        if self.config.device == "auto":
            detected = self._device_probe()
            return "mps" if detected == "mps" else "cpu"
        return self.config.device

    def diagnostics(self) -> VocalRuntimeDiagnostics:
        dependency = self._dependency_probe()
        asr_path = self._model_resolver(self.config.asr_model)
        aligner_path = self._model_resolver(self.config.aligner_model)
        issues: list[str] = []
        warnings: list[str] = []
        if not dependency:
            issues.append("dependency_missing")
            warnings.append(
                "Qwen 本地依赖未安装；运行 pip install -r apps/api/requirements-vocal.txt。"
            )
        if asr_path is None:
            issues.append("asr_model_not_cached")
            warnings.append(
                f"ASR 模型 {self.config.asr_model} 未在本地缓存；自动下载已禁用。"
            )
        if aligner_path is None:
            issues.append("aligner_model_not_cached")
            warnings.append(
                f"对齐模型 {self.config.aligner_model} 未在本地缓存；自动下载已禁用。"
            )
        asr_available = dependency and asr_path is not None
        aligner_available = dependency and aligner_path is not None
        return VocalRuntimeDiagnostics(
            available=asr_available and aligner_available,
            asr_available=asr_available,
            aligner_available=aligner_available,
            dependency_available=dependency,
            dependency_version=self._version_probe() if dependency else None,
            preferred_device=self._preferred_device(),
            asr_model=self.config.asr_model,
            aligner_model=self.config.aligner_model,
            asr_model_path=str(asr_path) if asr_path else None,
            aligner_model_path=str(aligner_path) if aligner_path else None,
            automatic_downloads_enabled=False,
            issues=tuple(issues),
            warnings=tuple(warnings),
        )

    @property
    def available(self) -> bool:
        return self.diagnostics().available

    @staticmethod
    def _prepare_audio(audio: NDArray[Any], sample_rate: int) -> FloatArray:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        values = np.asarray(audio, dtype=np.float32)
        if values.ndim == 2:
            if values.shape[0] in {1, 2} and values.shape[1] > values.shape[0]:
                values = values.mean(axis=0)
            elif values.shape[1] in {1, 2}:
                values = values.mean(axis=1)
            else:
                raise ValueError("audio must be mono or one/two-channel PCM")
        if values.ndim != 1 or values.size == 0:
            raise ValueError("audio must contain mono PCM samples")
        if not np.isfinite(values).all():
            raise ValueError("audio contains non-finite samples")
        return np.ascontiguousarray(values, dtype=np.float32)

    def _load_asr(self, device: str) -> Any:
        model = self._asr_models.get(device)
        if model is not None:
            return model
        path = self._model_resolver(self.config.asr_model)
        if path is None:
            raise RuntimeError("ASR model is not available in the local cache")
        model = self._runner.load_asr(
            path,
            device=device,
            max_inference_batch_size=self.config.max_inference_batch_size,
            max_new_tokens=self.config.max_new_tokens,
        )
        self._asr_models[device] = model
        return model

    def _load_aligner(self, device: str) -> Any:
        model = self._aligner_models.get(device)
        if model is not None:
            return model
        path = self._model_resolver(self.config.aligner_model)
        if path is None:
            raise RuntimeError("forced aligner model is not available in the local cache")
        model = self._runner.load_aligner(path, device=device)
        self._aligner_models[device] = model
        return model

    @staticmethod
    def _timestamps_from_backend(
        units: Sequence[BackendAlignmentUnit], sample_rate: int, sample_count: int
    ) -> tuple[VocalTimestamp, ...]:
        duration = sample_count / sample_rate
        timestamps: list[VocalTimestamp] = []
        for unit in units:
            start_sec = min(max(float(unit.start_sec), 0.0), duration)
            end_sec = min(max(float(unit.end_sec), start_sec), duration)
            start_sample = min(max(int(round(start_sec * sample_rate)), 0), sample_count)
            end_sample = min(max(int(round(end_sec * sample_rate)), start_sample), sample_count)
            timestamps.append(
                VocalTimestamp(
                    text=unit.text,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    start_sec=start_sample / sample_rate,
                    end_sec=end_sample / sample_rate,
                )
            )
        return tuple(timestamps)

    def _align_backend(
        self,
        audio: FloatArray,
        sample_rate: int,
        *,
        text: str,
        language: str,
    ) -> tuple[Sequence[BackendAlignmentUnit], str, tuple[str, ...]]:
        preferred = "cpu" if self._aligner_mps_disabled else self._preferred_device()
        warnings: list[str] = []
        try:
            model = self._load_aligner(preferred)
            units = self._runner.align(
                model,
                audio,
                sample_rate,
                text=text,
                language=language,
            )
            return units, preferred, tuple(warnings)
        except Exception as mps_error:
            if preferred != "mps":
                raise
            self._aligner_models.pop("mps", None)
            self._aligner_mps_disabled = True
            warnings.append(
                f"Qwen 对齐器在 MPS 上不可用，已回退 CPU：{type(mps_error).__name__}。"
            )
            model = self._load_aligner("cpu")
            units = self._runner.align(
                model,
                audio,
                sample_rate,
                text=text,
                language=language,
            )
            return units, "cpu", tuple(warnings)

    def align_known_japanese(
        self,
        audio: NDArray[Any],
        sample_rate: int,
        text: str,
    ) -> VocalAlignmentResult:
        """Align known Japanese lyrics and return sample-accurate token spans."""

        diagnostics = self.diagnostics()
        if not diagnostics.aligner_available:
            error_code = (
                "dependency_missing"
                if not diagnostics.dependency_available
                else "aligner_model_not_cached"
            )
            return VocalAlignmentResult(
                status="unavailable",
                text=text,
                model=self.config.aligner_model,
                warnings=diagnostics.warnings,
                error_code=error_code,
                error_message="Qwen3 日语强制对齐器当前不可用。",
            )
        if not text.strip():
            return VocalAlignmentResult(
                status="failed",
                text=text,
                model=self.config.aligner_model,
                error_code="empty_transcript",
                error_message="Known Japanese text must not be empty.",
            )
        try:
            values = self._prepare_audio(audio, sample_rate)
        except ValueError as error:
            return VocalAlignmentResult(
                status="failed",
                text=text,
                model=self.config.aligner_model,
                error_code="invalid_audio",
                error_message=str(error),
            )
        if values.size / sample_rate > self.config.max_alignment_seconds:
            return VocalAlignmentResult(
                status="failed",
                text=text,
                model=self.config.aligner_model,
                error_code="alignment_audio_too_long",
                error_message=(
                    f"Forced alignment supports at most {self.config.max_alignment_seconds:g} "
                    "seconds per chunk."
                ),
            )
        try:
            with self._lock:
                units, device, warnings = self._align_backend(
                    values,
                    sample_rate,
                    text=text,
                    language="Japanese",
                )
            timestamps = self._timestamps_from_backend(units, sample_rate, values.size)
            return VocalAlignmentResult(
                status="ok",
                text=text,
                timestamps=timestamps,
                model=self.config.aligner_model,
                device=device,
                warnings=warnings,
            )
        except Exception as error:
            return VocalAlignmentResult(
                status="failed",
                text=text,
                model=self.config.aligner_model,
                error_code="alignment_failed",
                error_message=f"{type(error).__name__}: {error}",
            )

    def transcribe_vocals(
        self,
        audio: NDArray[Any],
        sample_rate: int,
        *,
        language: str = "Japanese",
        context: str = "",
        align: bool = True,
    ) -> VocalTranscriptionResult:
        """Transcribe an already-separated vocal stem, optionally aligning its text."""

        diagnostics = self.diagnostics()
        if not diagnostics.asr_available:
            error_code = (
                "dependency_missing"
                if not diagnostics.dependency_available
                else "asr_model_not_cached"
            )
            return VocalTranscriptionResult(
                status="unavailable",
                model=self.config.asr_model,
                aligner_model=self.config.aligner_model,
                warnings=diagnostics.warnings,
                error_code=error_code,
                error_message="Qwen3 本地人声识别器当前不可用。",
            )
        try:
            values = self._prepare_audio(audio, sample_rate)
        except ValueError as error:
            return VocalTranscriptionResult(
                status="failed",
                model=self.config.asr_model,
                error_code="invalid_audio",
                error_message=str(error),
            )

        preferred = "cpu" if self._asr_mps_disabled else self._preferred_device()
        warnings: list[str] = []
        try:
            with self._lock:
                try:
                    model = self._load_asr(preferred)
                    transcription = self._runner.transcribe(
                        model,
                        values,
                        sample_rate,
                        language=language,
                        context=context,
                    )
                    device = preferred
                except Exception as mps_error:
                    if preferred != "mps":
                        raise
                    self._asr_models.pop("mps", None)
                    self._asr_mps_disabled = True
                    warnings.append(
                        "Qwen ASR 在 MPS 上不可用，已回退 CPU："
                        f"{type(mps_error).__name__}。"
                    )
                    model = self._load_asr("cpu")
                    transcription = self._runner.transcribe(
                        model,
                        values,
                        sample_rate,
                        language=language,
                        context=context,
                    )
                    device = "cpu"
        except Exception as error:
            return VocalTranscriptionResult(
                status="failed",
                model=self.config.asr_model,
                aligner_model=self.config.aligner_model,
                warnings=tuple(warnings),
                error_code="transcription_failed",
                error_message=f"{type(error).__name__}: {error}",
            )

        timestamps: tuple[VocalTimestamp, ...] = ()
        aligned = False
        aligner_device: str | None = None
        if (
            align
            and transcription.text.strip()
            and values.size / sample_rate > self.config.max_alignment_seconds
        ):
            warnings.append(
                "转写成功，但音频超过单次对齐时长限制；需分段后再生成发音时间点。"
            )
        elif align and transcription.text.strip():
            if diagnostics.aligner_available:
                try:
                    with self._lock:
                        units, aligner_device, align_warnings = self._align_backend(
                            values,
                            sample_rate,
                            text=transcription.text,
                            language=language,
                        )
                    timestamps = self._timestamps_from_backend(
                        units, sample_rate, values.size
                    )
                    warnings.extend(align_warnings)
                    aligned = True
                except Exception as error:
                    warnings.append(
                        "转写成功，但时间对齐失败："
                        f"{type(error).__name__}: {error}"
                    )
            else:
                warnings.append("转写成功，但本地对齐模型不可用，未生成发音时间点。")

        if aligner_device and aligner_device != device:
            warnings.append(f"ASR 使用 {device}，对齐器使用 {aligner_device}。")
        return VocalTranscriptionResult(
            status="ok",
            text=transcription.text,
            language=transcription.language,
            timestamps=timestamps,
            aligned=aligned,
            model=self.config.asr_model,
            aligner_model=self.config.aligner_model if aligned else None,
            device=device,
            warnings=tuple(warnings),
        )
