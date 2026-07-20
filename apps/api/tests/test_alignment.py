from __future__ import annotations

import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from beatforge_api.audio.alignment.base import (
    AdapterDiagnostics,
    AdapterOutput,
    AlignmentAdapter,
    AlignmentAdapterError,
    AlignmentContext,
    TempoReference,
    alignment_token_id,
)
from beatforge_api.audio.alignment.hybrid import HybridAlignmentAdapter
from beatforge_api.audio.alignment.qwen_adapter import QwenAlignmentAdapter
from beatforge_api.audio.alignment.runner import AlignmentRunner
from beatforge_api.audio.alignment.schema import (
    AlignmentReport,
    AlignmentResult,
    AlignmentToken,
)
from beatforge_api.timing import map_sample_index


class FakeAdapter(AlignmentAdapter):
    def __init__(
        self,
        method: str,
        *,
        start: int = 100,
        fail: bool = False,
        available: bool = True,
    ) -> None:
        self.method = method  # type: ignore[assignment]
        self.name = f"Fake {method}"
        self.start = start
        self.fail = fail
        self.available = available

    def diagnostics(self, context: AlignmentContext | None = None) -> AdapterDiagnostics:
        return AdapterDiagnostics(
            available=self.available,
            reason=None if self.available else "dependency missing",
        )

    def run(self, context: AlignmentContext) -> AdapterOutput:
        if self.fail:
            raise AlignmentAdapterError(
                f"{self.method.upper()}_FAILED",
                f"{self.method} failed independently",
            )
        end = self.start + 80
        return AdapterOutput(
            tokens=(
                AlignmentToken(
                    id=alignment_token_id(
                        context.track_id,
                        self.method,  # type: ignore[arg-type]
                        0,
                        self.start,
                        end,
                    ),
                    text="歌",
                    phoneme="u" if self.method == "ctc" else None,
                    start_sample=self.start,
                    end_sample=end,
                    confidence=0.8,
                    method=self.method,  # type: ignore[arg-type]
                ),
            ),
            metadata={"alignedText": "歌"},
        )


class FakeEvaluator:
    def evaluate(self, context: AlignmentContext, result: AlignmentResult) -> AlignmentReport:
        return AlignmentReport(
            run_id=result.run_id,
            track_id=context.track_id,
            method=result.method,
            score=0.8,
            coverage=1.0,
            acoustic=0.5,
            rhythm=0.75,
            stability=1.0,
            lyric_token_count=1,
            aligned_token_count=len(result.tokens),
            details={"groundTruth": "proxy_only"},
            created_at=datetime.now(UTC),
        )


def _context(tmp_path: Path) -> AlignmentContext:
    storage = tmp_path / "storage"
    models = storage / "models"
    models.mkdir(parents=True)
    vocals = storage / "stems" / "track-test" / "vocals.flac"
    vocals.parent.mkdir(parents=True)
    phase = np.arange(8_000, dtype=np.float32)
    sf.write(vocals, 0.1 * np.sin(2 * np.pi * 220 * phase / 8_000), 8_000)
    return AlignmentContext(
        track_id="track-test",
        lyrics="歌",
        lyrics_format="japanese",
        vocals_path=vocals,
        sample_rate=8_000,
        sample_count=8_000,
        tempo_map=(TempoReference(start_sample=0, bpm=120.0, beat_offset_sample=0),),
        models_dir=models,
        storage_dir=storage,
        project_root=tmp_path,
        song="test song",
        artist="test artist",
    )


def _wait_terminal(
    runner: AlignmentRunner,
    track_id: str,
    method: str,
) -> AlignmentResult:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        result = runner.get_result(track_id, method)  # type: ignore[arg-type]
        if result is not None and result.status in {"completed", "failed", "unavailable"}:
            return result
        time.sleep(0.01)
    raise AssertionError("alignment runner did not reach a terminal state")


def test_all_adapters_share_strict_span_schema() -> None:
    token = AlignmentToken(
        id="token-1",
        text="歌",
        phoneme="u",
        startSample=4_410,
        endSample=8_820,
        confidence=0.8,
        method="ctc",
    )
    assert token.start_sample == 4_410
    assert token.model_dump(by_alias=True)["endSample"] == 8_820
    with pytest.raises(ValueError):
        AlignmentToken(
            id="invalid",
            text="歌",
            startSample=100,
            endSample=100,
            confidence=0.8,
            method="qwen",
        )


def test_hybrid_keeps_successes_when_individual_models_fail(tmp_path: Path) -> None:
    adapters = {
        "qwen": FakeAdapter("qwen", start=100),
        "mfa": FakeAdapter("mfa", fail=True),
        "ctc": FakeAdapter("ctc", start=110),
        "singing": FakeAdapter("singing", available=False),
        "hybrid": HybridAlignmentAdapter(),
    }
    context = _context(tmp_path)
    runner = AlignmentRunner(
        context.storage_dir,
        context.project_root,
        adapters=adapters,  # type: ignore[arg-type]
        evaluator=FakeEvaluator(),  # type: ignore[arg-type]
    )
    try:
        queued = runner.submit(context, "hybrid")
        assert queued.status == "queued"
        hybrid = _wait_terminal(runner, context.track_id, "hybrid")
        assert hybrid.status == "completed"
        assert hybrid.tokens
        assert (hybrid.tokens[0].start_sample, hybrid.tokens[0].end_sample) in {
            (100, 180),
            (110, 190),
        }
        assert hybrid.metadata["timestampAveraging"] is False
        assert runner.get_result(context.track_id, "qwen").status == "completed"  # type: ignore[union-attr]
        assert runner.get_result(context.track_id, "ctc").status == "completed"  # type: ignore[union-attr]
        assert runner.get_result(context.track_id, "mfa").status == "failed"  # type: ignore[union-attr]
        assert runner.get_result(context.track_id, "singing").status == "unavailable"  # type: ignore[union-attr]
        comparison_path = context.project_root / "reports" / "alignment-comparison.json"
        deadline = time.monotonic() + 2.0
        while not comparison_path.is_file() and time.monotonic() < deadline:
            time.sleep(0.01)
        report = json.loads(comparison_path.read_text(encoding="utf-8"))
        assert report["song"] == "test song"
        assert {item["id"] for item in report["methods"]} == {
            "qwen",
            "mfa",
            "ctc",
            "singing",
            "hybrid",
        }
    finally:
        runner._executor.shutdown(wait=True)  # noqa: SLF001


@pytest.mark.parametrize("failure_stage", ["evaluate", "write"])
def test_hybrid_keeps_fused_tokens_when_proxy_reporting_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    class SelectiveEvaluator(FakeEvaluator):
        def evaluate(
            self,
            context: AlignmentContext,
            result: AlignmentResult,
        ) -> AlignmentReport:
            if failure_stage == "evaluate" and result.method == "hybrid":
                raise RuntimeError("proxy metric failed")
            return super().evaluate(context, result)

    adapters = {
        "qwen": FakeAdapter("qwen", start=100),
        "mfa": FakeAdapter("mfa", fail=True),
        "ctc": FakeAdapter("ctc", start=110),
        "singing": FakeAdapter("singing", available=False),
        "hybrid": HybridAlignmentAdapter(),
    }
    context = _context(tmp_path)
    runner = AlignmentRunner(
        context.storage_dir,
        context.project_root,
        adapters=adapters,  # type: ignore[arg-type]
        evaluator=SelectiveEvaluator(),  # type: ignore[arg-type]
    )
    if failure_stage == "write":
        original_write_report = runner._write_report  # noqa: SLF001

        def fail_hybrid_report(report: AlignmentReport) -> None:
            if report.method == "hybrid":
                raise OSError("report storage failed")
            original_write_report(report)

        monkeypatch.setattr(runner, "_write_report", fail_hybrid_report)

    try:
        runner.submit(context, "hybrid")
        hybrid = _wait_terminal(runner, context.track_id, "hybrid")
        assert hybrid.status == "completed"
        assert hybrid.tokens
        assert hybrid.error is None
        assert hybrid.hierarchy is None
        assert any("Proxy evaluation failed" in warning for warning in hybrid.warnings)
        expected = (
            {"type": "RuntimeError", "message": "proxy metric failed"}
            if failure_stage == "evaluate"
            else {"type": "OSError", "message": "report storage failed"}
        )
        assert hybrid.metadata["evaluationError"] == expected
        assert runner.get_report(context.track_id, "hybrid") is None
    finally:
        runner._executor.shutdown(wait=True)  # noqa: SLF001


def test_qwen_adapter_maps_model_spans_to_original_sample_domain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    # The model source domain is deliberately different from the original track.
    sf.write(context.vocals_path, np.zeros(16_000, dtype=np.float32), 16_000)
    python = tmp_path / "python"
    script = tmp_path / "scripts" / "qwen_vocal_cli.py"
    asr = tmp_path / "asr"
    aligner = tmp_path / "aligner"
    python.write_text("", encoding="utf-8")
    script.parent.mkdir()
    script.write_text("", encoding="utf-8")
    for model in (asr, aligner):
        model.mkdir()
        (model / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("BEATFORGE_QWEN_PYTHON", str(python))
    monkeypatch.setenv("BEATFORGE_QWEN_ASR_MODEL", str(asr))
    monkeypatch.setenv("BEATFORGE_QWEN_ALIGNER_MODEL", str(aligner))

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        output_path = Path(command[command.index("--output") + 1])
        output_path.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "timestamps": [
                        {
                            "text": "歌",
                            "start_sample": 1_600,
                            "end_sample": 3_200,
                            "chunk_match_confidence": 0.75,
                        }
                    ],
                    "model": "test-qwen",
                    "device": "mps",
                    "alignment_strategy": "test",
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    output = QwenAlignmentAdapter().run(context)

    assert len(output.tokens) == 1
    assert output.tokens[0].start_sample == map_sample_index(1_600, 16_000, 8_000)
    assert output.tokens[0].end_sample == map_sample_index(3_200, 16_000, 8_000)
    assert output.metadata["timestampProvenance"].startswith("Qwen3-ForcedAligner")


def test_alignment_api_runs_and_reads_one_method_without_fabricating_others(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from beatforge_api import routes
    from beatforge_api.config import get_settings

    from .test_api import attach_test_stem_assets, upload

    track_id = upload(client)["track"]["id"]
    attach_test_stem_assets(track_id)
    saved = client.put(
        f"/api/tracks/{track_id}/vocal-lyrics",
        json={"text": "歌", "inputFormat": "japanese"},
    )
    assert saved.status_code == 200, saved.text

    adapters = {
        "qwen": FakeAdapter("qwen", start=100),
        "mfa": FakeAdapter("mfa", fail=True),
        "ctc": FakeAdapter("ctc", start=110),
        "singing": FakeAdapter("singing", available=False),
        "hybrid": HybridAlignmentAdapter(),
    }
    settings = get_settings()
    runner = AlignmentRunner(
        settings.storage_dir,
        tmp_path,
        adapters=adapters,  # type: ignore[arg-type]
        evaluator=FakeEvaluator(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(routes, "_alignment_runner", runner)
    try:
        methods_response = client.get("/api/alignment/methods")
        assert methods_response.status_code == 200
        assert [item["id"] for item in methods_response.json()] == [
            "qwen",
            "mfa",
            "ctc",
            "singing",
            "hybrid",
        ]
        missing = client.get(f"/api/tracks/{track_id}/alignment/ctc")
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "ALIGNMENT_RESULT_NOT_FOUND"

        started = client.post(
            f"/api/tracks/{track_id}/alignment/run",
            json={"method": "qwen"},
        )
        assert started.status_code == 202, started.text
        assert started.json()["status"] in {"queued", "processing", "completed"}
        result = _wait_terminal(runner, track_id, "qwen")
        response = client.get(f"/api/tracks/{track_id}/alignment/qwen")
        assert response.status_code == 200
        assert response.json()["runId"] == result.run_id
        token = response.json()["tokens"][0]
        assert set(token) == {
            "id",
            "text",
            "phoneme",
            "startSample",
            "endSample",
            "confidence",
            "method",
        }
        assert token["startSample"] == 100
        assert token["endSample"] == 180
        report = client.get(f"/api/tracks/{track_id}/alignment/qwen/report")
        assert report.status_code == 200
        assert report.json()["runId"] == result.run_id
    finally:
        runner._executor.shutdown(wait=True)  # noqa: SLF001
