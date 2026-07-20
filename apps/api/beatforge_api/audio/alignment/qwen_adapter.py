from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import soundfile as sf

from ...config import get_settings
from ...platform_paths import venv_executable
from ...timing import map_sample_index
from .base import (
    AdapterDiagnostics,
    AdapterOutput,
    AlignmentAdapter,
    AlignmentAdapterError,
    AlignmentContext,
    alignment_token_id,
    clean_lyrics,
    executable_from_environment,
)
from .schema import AlignmentToken


class QwenAlignmentAdapter(AlignmentAdapter):
    method = "qwen"
    name = "Qwen Baseline"

    @staticmethod
    def _paths(context: AlignmentContext | None = None) -> tuple[Path, Path, Path, Path]:
        settings = get_settings() if context is None else None
        project_root = context.project_root if context else settings.project_root
        models_dir = context.models_dir if context else settings.models_dir
        python = executable_from_environment(
            "BEATFORGE_QWEN_PYTHON",
            venv_executable(project_root, ".venv-qwen"),
        )
        script = project_root / "scripts" / "qwen_vocal_cli.py"
        asr_model = executable_from_environment(
            "BEATFORGE_QWEN_ASR_MODEL",
            models_dir / "Qwen3-ASR-1.7B",
        )
        aligner_model = executable_from_environment(
            "BEATFORGE_QWEN_ALIGNER_MODEL",
            models_dir / "Qwen3-ForcedAligner-0.6B",
        )
        return python, script, asr_model, aligner_model

    def diagnostics(self, context: AlignmentContext | None = None) -> AdapterDiagnostics:
        python, script, asr_model, aligner_model = self._paths(context)
        checks = {
            "pythonAvailable": python.is_file(),
            "scriptAvailable": script.is_file(),
            "asrModelAvailable": (asr_model / "config.json").is_file(),
            "alignerModelAvailable": (aligner_model / "config.json").is_file(),
            "python": str(python),
            "asrModel": str(asr_model),
            "alignerModel": str(aligner_model),
        }
        missing = [
            name
            for name, value in checks.items()
            if name.endswith("Available") and not value
        ]
        return AdapterDiagnostics(
            available=not missing,
            reason=(
                None
                if not missing
                else (
                    "Qwen 本地运行时或模型缺失；运行 "
                    "python scripts/beatforge.py prepare-vocal-models 后重试。"
                )
            ),
            model="Qwen3-ForcedAligner-0.6B",
            automatic_downloads_enabled=False,
            details={**checks, "missing": missing},
        )

    def run(self, context: AlignmentContext) -> AdapterOutput:
        diagnostics = self.diagnostics(context)
        if not diagnostics.available:
            raise AlignmentAdapterError(
                "QWEN_RUNTIME_UNAVAILABLE",
                diagnostics.reason or "Qwen alignment runtime is unavailable.",
                status="unavailable",
                details=diagnostics.details,
            )
        lyrics = clean_lyrics(context.lyrics, context.lyrics_format)
        if not lyrics:
            raise AlignmentAdapterError("LYRICS_REQUIRED", "Qwen alignment requires saved lyrics.")
        python, script, asr_model, aligner_model = self._paths(context)
        alignment_root = context.storage_dir / "alignment"
        alignment_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="qwen-lab-", dir=alignment_root) as temporary:
            directory = Path(temporary)
            lyrics_path = directory / "lyrics.txt"
            output_path = directory / "result.json"
            lyrics_path.write_text(lyrics, encoding="utf-8")
            command = [
                str(python),
                str(script),
                "align_song",
                "--audio",
                str(context.vocals_path),
                "--text-file",
                str(lyrics_path),
                "--output",
                str(output_path),
                "--asr-model",
                str(asr_model),
                "--aligner-model",
                str(aligner_model),
                "--device",
                os.environ.get("BEATFORGE_QWEN_DEVICE", "auto"),
            ]
            environment = os.environ.copy()
            environment.update(
                {
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "PYTORCH_ENABLE_MPS_FALLBACK": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                }
            )
            try:
                completed = subprocess.run(
                    command,
                    cwd=context.project_root,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=float(os.environ.get("BEATFORGE_ALIGNMENT_QWEN_TIMEOUT", "7200")),
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                raise AlignmentAdapterError(
                    "QWEN_ALIGNMENT_TIMEOUT",
                    "Qwen baseline alignment timed out.",
                    details={"timeoutSec": error.timeout},
                ) from error
            if completed.returncode != 0 or not output_path.is_file():
                raise AlignmentAdapterError(
                    "QWEN_ALIGNMENT_PROCESS_FAILED",
                    "Qwen baseline alignment process failed.",
                    details={
                        "exitCode": completed.returncode,
                        "logTail": (completed.stderr or completed.stdout or "")[-4_000:],
                    },
                )
            try:
                payload: dict[str, Any] = json.loads(output_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise AlignmentAdapterError(
                    "QWEN_ALIGNMENT_OUTPUT_INVALID",
                    "Qwen returned an unreadable alignment result.",
                ) from error

        if payload.get("status") != "ok":
            raise AlignmentAdapterError(
                str(payload.get("error_code") or "QWEN_ALIGNMENT_FAILED").upper(),
                str(payload.get("error_message") or "Qwen returned no alignment."),
                details={"warnings": payload.get("warnings", [])},
            )
        timestamps = payload.get("timestamps")
        if not isinstance(timestamps, list) or not timestamps:
            raise AlignmentAdapterError(
                "QWEN_ALIGNMENT_EMPTY",
                "Qwen completed without any model timestamps.",
            )
        stem_rate = int(sf.info(context.vocals_path).samplerate)
        tokens: list[AlignmentToken] = []
        for item in timestamps:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            try:
                raw_start = int(item["start_sample"])
                raw_end = int(item["end_sample"])
            except (KeyError, TypeError, ValueError):
                continue
            start = min(
                max(0, map_sample_index(raw_start, stem_rate, context.sample_rate)),
                context.sample_count - 1,
            )
            end = min(
                max(0, map_sample_index(raw_end, stem_rate, context.sample_rate)),
                context.sample_count,
            )
            if end <= start:
                continue
            match_confidence = item.get("chunk_match_confidence", 1.0)
            try:
                confidence = min(1.0, max(0.0, float(match_confidence)))
            except (TypeError, ValueError):
                confidence = 1.0
            index = len(tokens)
            tokens.append(
                AlignmentToken(
                    id=alignment_token_id(context.track_id, self.method, index, start, end),
                    text=text,
                    start_sample=start,
                    end_sample=end,
                    confidence=confidence,
                    method=self.method,
                )
            )
        if not tokens:
            raise AlignmentAdapterError(
                "QWEN_ALIGNMENT_EMPTY",
                "Qwen output contained no valid sample spans.",
            )
        return AdapterOutput(
            tokens=tuple(tokens),
            warnings=tuple(str(item) for item in payload.get("warnings", []) if item),
            metadata={
                "model": payload.get("model"),
                "device": payload.get("device"),
                "alignmentStrategy": payload.get("alignment_strategy"),
                "source": "fresh_qwen_model_run",
                "sourceSampleRate": stem_rate,
                "alignedText": "".join(token.text for token in tokens),
                "timestampProvenance": "Qwen3-ForcedAligner start/end spans",
            },
        )
