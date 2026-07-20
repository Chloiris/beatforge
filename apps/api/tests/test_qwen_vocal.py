from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from beatforge_api.audio.qwen_vocal import (
    BackendAlignmentUnit,
    BackendTranscription,
    OfficialQwenRunner,
    QwenVocalAnalyzer,
    QwenVocalConfig,
    resolve_cached_model,
)


class FakeQwenRunner:
    def __init__(
        self,
        *,
        fail_mps_asr_load: bool = False,
        fail_mps_asr_inference: bool = False,
        fail_mps_aligner_load: bool = False,
    ) -> None:
        self.fail_mps_asr_load = fail_mps_asr_load
        self.fail_mps_asr_inference = fail_mps_asr_inference
        self.fail_mps_aligner_load = fail_mps_aligner_load
        self.asr_loads: list[str] = []
        self.aligner_loads: list[str] = []
        self.transcriptions: list[str] = []
        self.alignments: list[str] = []

    def load_asr(
        self,
        model_path: Path,
        *,
        device: str,
        max_inference_batch_size: int,
        max_new_tokens: int,
    ) -> dict[str, Any]:
        assert model_path.name == "asr"
        assert max_inference_batch_size == 1
        assert max_new_tokens == 512
        self.asr_loads.append(device)
        if device == "mps" and self.fail_mps_asr_load:
            raise RuntimeError("unsupported MPS operation")
        return {"kind": "asr", "device": device}

    def load_aligner(self, model_path: Path, *, device: str) -> dict[str, str]:
        assert model_path.name == "aligner"
        self.aligner_loads.append(device)
        if device == "mps" and self.fail_mps_aligner_load:
            raise RuntimeError("unsupported MPS aligner operation")
        return {"kind": "aligner", "device": device}

    def transcribe(
        self,
        model: dict[str, str],
        audio: np.ndarray,
        sample_rate: int,
        *,
        language: str,
        context: str,
    ) -> BackendTranscription:
        assert audio.dtype == np.float32
        assert audio.flags.c_contiguous
        assert sample_rate == 44_100
        assert language == "Japanese"
        assert context == "song lyrics"
        self.transcriptions.append(model["device"])
        if model["device"] == "mps" and self.fail_mps_asr_inference:
            raise RuntimeError("MPS inference failed")
        return BackendTranscription(text="君と歌う", language="Japanese")

    def align(
        self,
        model: dict[str, str],
        audio: np.ndarray,
        sample_rate: int,
        *,
        text: str,
        language: str,
    ) -> tuple[BackendAlignmentUnit, ...]:
        assert model["kind"] == "aligner"
        assert language == "Japanese"
        assert text in {"君と歌う", "夜空へ"}
        self.alignments.append(model["device"])
        return (
            BackendAlignmentUnit(text="君", start_sec=0.1, end_sec=0.2),
            BackendAlignmentUnit(text="歌う", start_sec=0.333, end_sec=0.51),
        )


def _model_directories(tmp_path: Path) -> dict[str, Path]:
    directories = {
        "Qwen/Qwen3-ASR-0.6B": tmp_path / "asr",
        "Qwen/Qwen3-ForcedAligner-0.6B": tmp_path / "aligner",
    }
    for directory in directories.values():
        directory.mkdir()
        (directory / "config.json").write_text("{}", encoding="utf-8")
    return directories


def _analyzer(
    tmp_path: Path,
    runner: FakeQwenRunner,
    *,
    device: str = "cpu",
    include_aligner: bool = True,
) -> QwenVocalAnalyzer:
    directories = _model_directories(tmp_path)
    if not include_aligner:
        directories.pop("Qwen/Qwen3-ForcedAligner-0.6B")
    return QwenVocalAnalyzer(
        QwenVocalConfig(device=device),  # type: ignore[arg-type]
        runner=runner,
        model_resolver=directories.get,
        version_probe=lambda: "test-runtime",
        device_probe=lambda: device,
    )


def test_missing_dependency_returns_structured_unavailable_without_loading(tmp_path: Path) -> None:
    runner = FakeQwenRunner()
    directories = _model_directories(tmp_path)
    analyzer = QwenVocalAnalyzer(
        runner=runner,
        model_resolver=directories.get,
        dependency_probe=lambda: False,
    )

    diagnostics = analyzer.diagnostics()
    result = analyzer.transcribe_vocals(np.zeros(100, dtype=np.float32), 44_100)

    assert diagnostics.available is False
    assert diagnostics.dependency_available is False
    assert diagnostics.automatic_downloads_enabled is False
    assert result.status == "unavailable"
    assert result.error_code == "dependency_missing"
    assert runner.asr_loads == []
    assert runner.aligner_loads == []


def test_models_load_lazily_are_cached_and_timestamps_use_integer_samples(
    tmp_path: Path,
) -> None:
    runner = FakeQwenRunner()
    analyzer = _analyzer(tmp_path, runner)
    audio = np.zeros(44_100, dtype=np.float32)

    assert analyzer.diagnostics().available is True
    assert runner.asr_loads == []
    first = analyzer.transcribe_vocals(audio, 44_100, context="song lyrics")
    second = analyzer.transcribe_vocals(audio, 44_100, context="song lyrics")

    assert first.status == "ok"
    assert first.aligned is True
    assert first.text == "君と歌う"
    assert [item.start_sample for item in first.timestamps] == [4_410, 14_685]
    assert first.timestamps[1].start_sec == 14_685 / 44_100
    assert runner.asr_loads == ["cpu"]
    assert runner.aligner_loads == ["cpu"]
    assert len(runner.transcriptions) == 2
    assert second.timestamps == first.timestamps


def test_known_japanese_text_alignment(tmp_path: Path) -> None:
    runner = FakeQwenRunner()
    analyzer = _analyzer(tmp_path, runner)

    result = analyzer.align_known_japanese(
        np.zeros((44_100, 2), dtype=np.float32),
        44_100,
        "夜空へ",
    )

    assert result.status == "ok"
    assert result.device == "cpu"
    assert result.timestamps[0].text == "君"
    assert result.timestamps[0].start_sample == 4_410
    assert result.timestamps[0].end_sample == 8_820


def test_mps_load_failure_falls_back_to_cpu_once(tmp_path: Path) -> None:
    runner = FakeQwenRunner(fail_mps_asr_load=True)
    analyzer = _analyzer(tmp_path, runner, device="mps")
    audio = np.zeros(44_100, dtype=np.float32)

    first = analyzer.transcribe_vocals(
        audio,
        44_100,
        context="song lyrics",
        align=False,
    )
    second = analyzer.transcribe_vocals(
        audio,
        44_100,
        context="song lyrics",
        align=False,
    )

    assert first.status == "ok"
    assert first.device == "cpu"
    assert any("回退 CPU" in warning for warning in first.warnings)
    assert runner.asr_loads == ["mps", "cpu"]
    assert second.device == "cpu"


def test_mps_inference_failure_also_falls_back_to_cpu(tmp_path: Path) -> None:
    runner = FakeQwenRunner(fail_mps_asr_inference=True)
    analyzer = _analyzer(tmp_path, runner, device="mps")

    result = analyzer.transcribe_vocals(
        np.zeros(44_100, dtype=np.float32),
        44_100,
        context="song lyrics",
        align=False,
    )

    assert result.status == "ok"
    assert result.device == "cpu"
    assert runner.transcriptions == ["mps", "cpu"]
    assert runner.asr_loads == ["mps", "cpu"]


def test_transcription_can_succeed_without_aligner_cache(tmp_path: Path) -> None:
    runner = FakeQwenRunner()
    analyzer = _analyzer(tmp_path, runner, include_aligner=False)

    result = analyzer.transcribe_vocals(
        np.zeros(44_100, dtype=np.float32),
        44_100,
        context="song lyrics",
    )

    assert result.status == "ok"
    assert result.aligned is False
    assert result.timestamps == ()
    assert any("对齐模型不可用" in warning for warning in result.warnings)
    assert runner.aligner_loads == []


def test_long_known_text_alignment_fails_before_loading(tmp_path: Path) -> None:
    runner = FakeQwenRunner()
    analyzer = QwenVocalAnalyzer(
        QwenVocalConfig(max_alignment_seconds=0.5),
        runner=runner,
        model_resolver=_model_directories(tmp_path).get,
    )

    result = analyzer.align_known_japanese(
        np.zeros(44_100, dtype=np.float32),
        44_100,
        "夜空へ",
    )

    assert result.status == "failed"
    assert result.error_code == "alignment_audio_too_long"
    assert runner.aligner_loads == []


def test_huggingface_cache_resolution_never_downloads(tmp_path: Path, monkeypatch: Any) -> None:
    cache_root = tmp_path / "hub"
    revision = "abc123"
    model_root = cache_root / "models--Qwen--Qwen3-ASR-0.6B"
    snapshot = model_root / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")
    (model_root / "refs").mkdir()
    (model_root / "refs" / "main").write_text(revision, encoding="utf-8")
    monkeypatch.setenv("HF_HUB_CACHE", str(cache_root))

    assert resolve_cached_model("Qwen/Qwen3-ASR-0.6B") == snapshot.resolve()


def test_official_runner_accepts_forced_align_result_items_container() -> None:
    class Item:
        text = "歌"
        start_time = 0.25
        end_time = 0.5

    class Result:
        items = [Item()]

    class Model:
        def align(self, **kwargs: Any) -> list[Result]:
            assert kwargs["language"] == "Japanese"
            return [Result()]

    units = OfficialQwenRunner().align(
        Model(),
        np.zeros(16, dtype=np.float32),
        16_000,
        text="歌",
        language="Japanese",
    )

    assert units == (BackendAlignmentUnit(text="歌", start_sec=0.25, end_sec=0.5),)
