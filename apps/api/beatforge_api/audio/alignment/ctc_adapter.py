from __future__ import annotations

import json
import math
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

MODEL_ID = "prj-beatrice/japanese-hubert-base-phoneme-ctc-v4"
MODEL_REVISION = "f5fe07043bcb0b77a86faf72ac6d8fc1ae558f99"
MODEL_LICENSE = "Apache-2.0"
MODEL_DIRECTORY_NAME = "japanese-hubert-base-phoneme-ctc-v4"
EXPECTED_WEIGHT_BYTES = 377_659_928
REQUIRED_MODULES = (
    "numpy",
    "pyopenjtalk",
    "scipy",
    "soundfile",
    "torch",
    "transformers",
)


def _runtime_modules(python: Path) -> tuple[list[str], str | None]:
    """Probe imports without importing the ML stack into the API process."""

    if not python.is_file():
        return list(REQUIRED_MODULES), "Python executable does not exist."
    source = (
        "import importlib.util,json;"
        f"mods={list(REQUIRED_MODULES)!r};"
        "print(json.dumps([m for m in mods if importlib.util.find_spec(m) is None]))"
    )
    try:
        completed = subprocess.run(
            [str(python), "-c", source],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return list(REQUIRED_MODULES), f"Runtime probe failed: {type(error).__name__}."
    if completed.returncode != 0:
        return list(REQUIRED_MODULES), (completed.stderr or completed.stdout or "")[-1_000:]
    try:
        missing = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return list(REQUIRED_MODULES), "Runtime probe returned invalid JSON."
    if not isinstance(missing, list) or not all(isinstance(item, str) for item in missing):
        return list(REQUIRED_MODULES), "Runtime probe returned an invalid module list."
    return missing, None


def _model_files(model_dir: Path) -> dict[str, bool]:
    safetensors = model_dir / "model.safetensors"
    return {
        "configAvailable": (model_dir / "config.json").is_file(),
        "preprocessorAvailable": (model_dir / "preprocessor_config.json").is_file(),
        "vocabularyAvailable": (model_dir / "vocab.json").is_file(),
        "weightsAvailable": (
            safetensors.is_file() and safetensors.stat().st_size == EXPECTED_WEIGHT_BYTES
        ),
    }


class CTCAlignmentAdapter(AlignmentAdapter):
    """Japanese phone CTC emissions followed by an observed global Viterbi path."""

    method = "ctc"
    name = "Japanese HuBERT Phoneme CTC"

    @staticmethod
    def _paths(context: AlignmentContext | None = None) -> tuple[Path, Path, Path]:
        settings = get_settings() if context is None else None
        project_root = context.project_root if context else settings.project_root
        models_dir = context.models_dir if context else settings.models_dir
        python = executable_from_environment(
            "BEATFORGE_CTC_PYTHON",
            venv_executable(project_root, ".venv-qwen"),
        )
        script = executable_from_environment(
            "BEATFORGE_CTC_SCRIPT",
            project_root / "scripts" / "ctc_phoneme_align.py",
        )
        model_dir = executable_from_environment(
            "BEATFORGE_CTC_MODEL",
            models_dir / MODEL_DIRECTORY_NAME,
        )
        return python, script, model_dir

    def diagnostics(self, context: AlignmentContext | None = None) -> AdapterDiagnostics:
        python, script, model_dir = self._paths(context)
        missing_modules, runtime_error = _runtime_modules(python)
        files = _model_files(model_dir)
        manifest_path = model_dir / "beatforge-model.json"
        manifest: dict[str, Any] = {}
        manifest_error: str | None = None
        if manifest_path.is_file():
            try:
                loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    manifest = loaded
                else:
                    manifest_error = "Model manifest is not a JSON object."
            except (OSError, json.JSONDecodeError) as error:
                manifest_error = f"Model manifest is unreadable: {type(error).__name__}."

        revision_verified = (
            manifest.get("modelId") == MODEL_ID
            and manifest.get("revision") == MODEL_REVISION
        )
        checks: dict[str, Any] = {
            "pythonAvailable": python.is_file(),
            "scriptAvailable": script.is_file(),
            **files,
            "manifestAvailable": manifest_path.is_file() and manifest_error is None,
            "pinnedRevisionAvailable": revision_verified,
            "runtimeModulesAvailable": not missing_modules and runtime_error is None,
            "python": str(python),
            "script": str(script),
            "modelPath": str(model_dir),
            "modelId": MODEL_ID,
            "revision": MODEL_REVISION,
            "revisionVerified": revision_verified,
            "license": MODEL_LICENSE,
            "missingModules": missing_modules,
            "runtimeProbeError": runtime_error,
            "manifestError": manifest_error,
            "devicePreference": os.environ.get("BEATFORGE_CTC_DEVICE", "auto"),
            "timestampAlgorithm": "HuBERT emissions + global CTC Viterbi",
        }
        missing = [
            key
            for key, value in checks.items()
            if key.endswith("Available") and value is False
        ]
        available = not missing
        reason: str | None = None
        if missing_modules or runtime_error:
            reason = (
                "CTC 本地运行环境缺少依赖："
                + ", ".join(missing_modules or ["runtime probe failed"])
                + "。"
            )
        elif not script.is_file():
            reason = "CTC alignment helper script is missing."
        elif not all(files.values()) or not revision_verified:
            reason = (
                "日语 HuBERT CTC checkpoint 未完成或未通过 pinned revision 检查；运行 "
                "python scripts/beatforge.py prepare-alignment-models 后重试。"
            )
        elif not python.is_file():
            reason = "CTC Python runtime is missing."
        return AdapterDiagnostics(
            available=available,
            reason=reason,
            model=MODEL_ID,
            automatic_downloads_enabled=False,
            details={**checks, "missing": missing},
        )

    def _prepare_alignment_inputs(
        self,
        context: AlignmentContext,
        directory: Path,
        python: Path,
        lyrics_path: Path,
    ) -> tuple[str, ...]:
        """Return optional helper arguments prepared inside the temporary run directory."""

        del context, directory, python, lyrics_path
        return ()

    def _postprocess_output(
        self,
        context: AlignmentContext,
        payload: dict[str, Any],
        output: AdapterOutput,
    ) -> AdapterOutput:
        """Allow the v0.6.1 engine to add hierarchy and acoustic refinement."""

        del context, payload
        return output

    def run(self, context: AlignmentContext) -> AdapterOutput:
        diagnostics = self.diagnostics(context)
        if not diagnostics.available:
            raise AlignmentAdapterError(
                "CTC_RUNTIME_UNAVAILABLE",
                diagnostics.reason or "Japanese phoneme CTC runtime is unavailable.",
                status="unavailable",
                details=diagnostics.details,
            )
        lyrics = clean_lyrics(context.lyrics, context.lyrics_format)
        if not lyrics:
            raise AlignmentAdapterError("LYRICS_REQUIRED", "CTC alignment requires saved lyrics.")

        python, script, model_dir = self._paths(context)
        alignment_root = context.storage_dir / "alignment"
        alignment_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="ctc-lab-", dir=alignment_root) as temporary:
            directory = Path(temporary)
            lyrics_path = directory / "lyrics.txt"
            output_path = directory / "result.json"
            lyrics_path.write_text(lyrics, encoding="utf-8")
            extra_arguments = self._prepare_alignment_inputs(
                context,
                directory,
                python,
                lyrics_path,
            )
            command = [
                str(python),
                str(script),
                "--audio",
                str(context.vocals_path),
                "--lyrics-file",
                str(lyrics_path),
                "--model",
                str(model_dir),
                "--output",
                str(output_path),
                "--device",
                os.environ.get("BEATFORGE_CTC_DEVICE", "auto"),
                "--chunk-seconds",
                os.environ.get("BEATFORGE_CTC_CHUNK_SECONDS", "20"),
                "--overlap-seconds",
                os.environ.get("BEATFORGE_CTC_OVERLAP_SECONDS", "2"),
                *extra_arguments,
            ]
            environment = os.environ.copy()
            environment.update(
                {
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "HF_HUB_DISABLE_TELEMETRY": "1",
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
                    timeout=float(os.environ.get("BEATFORGE_ALIGNMENT_CTC_TIMEOUT", "7200")),
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                raise AlignmentAdapterError(
                    "CTC_ALIGNMENT_TIMEOUT",
                    "Japanese phoneme CTC alignment timed out.",
                    details={"timeoutSec": error.timeout},
                ) from error

            payload: dict[str, Any] | None = None
            if output_path.is_file():
                try:
                    decoded = json.loads(output_path.read_text(encoding="utf-8"))
                    if isinstance(decoded, dict):
                        payload = decoded
                except (OSError, json.JSONDecodeError):
                    payload = None
            if payload and payload.get("status") != "ok":
                raise AlignmentAdapterError(
                    str(payload.get("error_code") or "CTC_ALIGNMENT_FAILED").upper(),
                    str(payload.get("error_message") or "CTC alignment failed."),
                    details={
                        **(
                            payload.get("details")
                            if isinstance(payload.get("details"), dict)
                            else {}
                        ),
                        "exitCode": completed.returncode,
                        "logTail": (completed.stderr or completed.stdout or "")[-4_000:],
                    },
                )
            if completed.returncode != 0 or payload is None:
                raise AlignmentAdapterError(
                    "CTC_ALIGNMENT_PROCESS_FAILED",
                    "Japanese phoneme CTC alignment process failed.",
                    details={
                        "exitCode": completed.returncode,
                        "logTail": (completed.stderr or completed.stdout or "")[-4_000:],
                    },
                )

        phones = payload.get("phones")
        if not isinstance(phones, list) or not phones:
            raise AlignmentAdapterError(
                "CTC_ALIGNMENT_EMPTY",
                "CTC Viterbi completed without any observed phone spans.",
            )
        try:
            stem_info = sf.info(context.vocals_path)
            stem_rate = int(payload.get("source_sample_rate") or stem_info.samplerate)
            stem_count = int(stem_info.frames)
        except (RuntimeError, TypeError, ValueError) as error:
            raise AlignmentAdapterError(
                "CTC_SOURCE_METADATA_INVALID",
                "Could not validate the CTC source sample domain.",
            ) from error
        if stem_rate <= 0 or stem_count <= 0:
            raise AlignmentAdapterError(
                "CTC_SOURCE_METADATA_INVALID",
                "CTC source sample rate or sample count is invalid.",
            )

        tokens: list[AlignmentToken] = []
        previous_source_end = 0
        for index, item in enumerate(phones):
            if not isinstance(item, dict):
                raise AlignmentAdapterError(
                    "CTC_ALIGNMENT_OUTPUT_INVALID",
                    "CTC output contains a non-object phone span.",
                    details={"phoneIndex": index},
                )
            text = str(item.get("surface") or "").strip()
            phoneme = str(item.get("phoneme") or "").strip()
            try:
                raw_start = int(item["start_sample"])
                raw_end = int(item["end_sample"])
                confidence = float(item["confidence"])
            except (KeyError, TypeError, ValueError) as error:
                raise AlignmentAdapterError(
                    "CTC_ALIGNMENT_OUTPUT_INVALID",
                    "CTC output contains an invalid phone span.",
                    details={"phoneIndex": index},
                ) from error
            if (
                not text
                or not phoneme
                or raw_start < previous_source_end
                or raw_end <= raw_start
                or raw_end > stem_count
                or not math.isfinite(confidence)
                or confidence < 0.0
                or confidence > 1.0
            ):
                raise AlignmentAdapterError(
                    "CTC_ALIGNMENT_OUTPUT_INVALID",
                    "CTC phone spans failed monotonicity, bounds, or confidence validation.",
                    details={
                        "phoneIndex": index,
                        "surface": text,
                        "phoneme": phoneme,
                        "startSample": raw_start,
                        "endSample": raw_end,
                        "previousEndSample": previous_source_end,
                        "sourceSampleCount": stem_count,
                    },
                )
            previous_source_end = raw_end
            start = map_sample_index(raw_start, stem_rate, context.sample_rate)
            end = map_sample_index(raw_end, stem_rate, context.sample_rate)
            if start >= context.sample_count or end > context.sample_count or end <= start:
                raise AlignmentAdapterError(
                    "CTC_ALIGNMENT_SAMPLE_MAPPING_INVALID",
                    "A real CTC phone span could not be represented in the original sample domain.",
                    details={"phoneIndex": index, "startSample": start, "endSample": end},
                )
            tokens.append(
                AlignmentToken(
                    id=alignment_token_id(context.track_id, self.method, index, start, end),
                    text=text,
                    phoneme=phoneme,
                    start_sample=start,
                    end_sample=end,
                    confidence=confidence,
                    method=self.method,
                )
            )

        if len(tokens) != len(phones):
            raise AlignmentAdapterError(
                "CTC_ALIGNMENT_OUTPUT_INVALID",
                "CTC output was not converted atomically.",
            )
        warnings = payload.get("warnings")
        metadata = payload.get("metadata")
        compact_surfaces: list[str] = []
        for token in tokens:
            if not compact_surfaces or compact_surfaces[-1] != token.text:
                compact_surfaces.append(token.text)
        script_surface_sequence = (
            metadata.get("surfaceSequence") if isinstance(metadata, dict) else None
        )
        aligned_surfaces = (
            [item for item in script_surface_sequence if isinstance(item, str) and item]
            if isinstance(script_surface_sequence, list)
            else compact_surfaces
        )
        output = AdapterOutput(
            tokens=tuple(tokens),
            warnings=tuple(str(item) for item in warnings if item)
            if isinstance(warnings, list)
            else (),
            metadata={
                **(metadata if isinstance(metadata, dict) else {}),
                "model": MODEL_ID,
                "revision": MODEL_REVISION,
                "license": MODEL_LICENSE,
                "source": "fresh_local_hubert_ctc_viterbi_run",
                "sourceSampleRate": stem_rate,
                "targetPhoneCount": len(tokens),
                "alignedText": "".join(aligned_surfaces),
                "timestampProvenance": (
                    "Japanese HuBERT frame emissions and a global CTC Viterbi path; "
                    "no lyric timestamps or even-duration allocation"
                ),
            },
        )
        return self._postprocess_output(context, payload, output)
