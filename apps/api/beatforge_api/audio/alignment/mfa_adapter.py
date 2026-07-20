from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import soundfile as sf

from ...config import get_settings
from ...platform_paths import venv_executable
from .base import (
    AdapterDiagnostics,
    AdapterOutput,
    AlignmentAdapter,
    AlignmentAdapterError,
    AlignmentContext,
    alignment_token_id,
    clean_lyrics,
)
from .schema import AlignmentToken

_MODEL_NAME = "japanese_mfa"
# MFA 3's Japanese 3.x model performs its own text tokenization.  The 2.0.1a
# model is the documented legacy model for supplying an explicit pronunciation
# dictionary, which is required here so OpenJTalk remains the actual G2P source.
_MODEL_VERSION = "2.0.1a"
_SILENCE_PHONES = frozenset(
    {"", "sil", "sp", "spn", "<eps>", "<sil>", "<unk>", "speech_noise"}
)

# pyopenjtalk emits the compact Open JTalk phone set.  The Japanese MFA 2.x
# acoustic model uses MFA's Japanese IPA phone set, so every emitted phone is
# converted explicitly.  Unknown phones fail the run; they are never dropped or
# assigned a made-up timestamp.
_OPENJTALK_TO_MFA = {
    "a": "a",
    "A": "a",
    "i": "i",
    "I": "i̥",
    "u": "ɯ",
    "U": "ɯ̥",
    "e": "e",
    "E": "e",
    "o": "o",
    "O": "o",
    "b": "b",
    "by": "bʲ",
    "ch": "tɕ",
    "d": "d",
    "dy": "dʲ",
    "f": "ɸ",
    "fy": "ɸʲ",
    "g": "ɡ",
    "gy": "ɟ",
    "h": "h",
    "hy": "ç",
    "j": "dʑ",
    "k": "k",
    "ky": "c",
    "kw": "k",
    "m": "m",
    "my": "mʲ",
    "n": "n",
    "ny": "ɲ",
    "p": "p",
    "py": "pʲ",
    "r": "ɾ",
    "ry": "ɾʲ",
    "s": "s",
    "sh": "ɕ",
    "t": "t",
    "ts": "ts",
    "ty": "tʲ",
    "v": "v",
    "vy": "vʲ",
    "w": "w",
    "y": "j",
    "z": "z",
    "zh": "ʑ",
    "N": "ɴ",
    "ng": "ŋ",
    "gw": "ɡ",
}
_LONG_CONSONANTS = frozenset(
    {
        "bː",
        "cː",
        "dː",
        "dʑː",
        "hː",
        "kː",
        "mː",
        "nː",
        "pː",
        "sː",
        "tː",
        "tsː",
        "tɕː",
        "zː",
        "çː",
        "ɕː",
        "ɡː",
        "ɸː",
        "ɾː",
        "ʑː",
    }
)

_G2P_SCRIPT = r"""
import json
import sys

import pyopenjtalk

payload = json.loads(sys.stdin.read())
output = []
for raw_line in str(payload.get("text", "")).splitlines():
    line = raw_line.strip()
    if not line:
        continue
    for feature in pyopenjtalk.run_frontend(line):
        surface = str(feature.get("string", "")).strip()
        if not surface:
            continue
        phones = str(pyopenjtalk.g2p(surface, kana=False)).split()
        output.append({"text": surface, "phones": phones})
print(json.dumps({"words": output}, ensure_ascii=False, separators=(",", ":")))
"""


@dataclass(frozen=True, slots=True)
class _G2PWord:
    label: str
    text: str
    phones: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _TextGridInterval:
    start_sec: float
    end_sec: float
    text: str


def _openjtalk_phones_to_mfa(phones: list[str]) -> tuple[str, ...]:
    """Convert real OpenJTalk output to the Japanese MFA 2.x phone inventory."""

    output: list[str] = []
    geminate = False
    for raw_phone in phones:
        phone = raw_phone.strip()
        if not phone or phone in {"pau", "sil"}:
            continue
        if phone == "cl":
            # Open JTalk represents sokuon as a closure preceding the following
            # consonant.  MFA represents medial gemination as that consonant's
            # long form.  A final closure remains the model's real glottal phone.
            geminate = True
            continue
        mapped = _OPENJTALK_TO_MFA.get(phone)
        if mapped is None:
            raise AlignmentAdapterError(
                "MFA_G2P_UNSUPPORTED_PHONE",
                f"OpenJTalk returned a phone unsupported by Japanese MFA: {phone}",
                details={"phone": phone, "openJtalkPhones": phones},
            )
        if geminate:
            long_phone = f"{mapped}ː"
            if long_phone in _LONG_CONSONANTS:
                mapped = long_phone
            else:
                output.append("ʔ")
            geminate = False
        output.append(mapped)
    if geminate:
        output.append("ʔ")
    return tuple(output)


_ITEM_HEADER = re.compile(r"^\s*item \[\d+\]:\s*$")
_INTERVAL_HEADER = re.compile(r"^\s*intervals \[\d+\]:\s*$")
_TIER_NAME = re.compile(r'^\s*name\s*=\s*"(.*)"\s*$')
_INTERVAL_TEXT = re.compile(r'^\s*text\s*=\s*"(.*)"\s*$')
_XMIN = re.compile(r"^\s*xmin\s*=\s*([^\s]+)\s*$")
_XMAX = re.compile(r"^\s*xmax\s*=\s*([^\s]+)\s*$")


def _praat_text(value: str) -> str:
    return value.replace('""', '"')


def _parse_long_textgrid(text: str) -> dict[str, list[_TextGridInterval]]:
    """Parse MFA's explicitly requested long TextGrid format without extra deps."""

    if 'Object class = "TextGrid"' not in text:
        raise AlignmentAdapterError(
            "MFA_TEXTGRID_INVALID",
            "MFA output is not a long TextGrid file.",
        )
    tiers: dict[str, list[_TextGridInterval]] = {}
    tier_name: str | None = None
    interval: dict[str, Any] | None = None

    def commit_interval() -> None:
        nonlocal interval
        if interval is None:
            return
        if tier_name is not None and {"xmin", "xmax", "text"} <= interval.keys():
            try:
                start = float(interval["xmin"])
                end = float(interval["xmax"])
            except (TypeError, ValueError) as error:
                raise AlignmentAdapterError(
                    "MFA_TEXTGRID_INVALID",
                    "MFA TextGrid contains a non-numeric interval boundary.",
                ) from error
            if not math.isfinite(start) or not math.isfinite(end) or end < start:
                raise AlignmentAdapterError(
                    "MFA_TEXTGRID_INVALID",
                    "MFA TextGrid contains an invalid interval boundary.",
                    details={"startSec": start, "endSec": end},
                )
            tiers.setdefault(tier_name, []).append(
                _TextGridInterval(start_sec=start, end_sec=end, text=str(interval["text"]))
            )
        interval = None

    for line in text.splitlines():
        if _ITEM_HEADER.match(line):
            commit_interval()
            tier_name = None
            continue
        if _INTERVAL_HEADER.match(line):
            commit_interval()
            interval = {}
            continue
        if interval is None:
            name_match = _TIER_NAME.match(line)
            if name_match:
                tier_name = _praat_text(name_match.group(1))
            continue
        xmin_match = _XMIN.match(line)
        if xmin_match:
            interval["xmin"] = xmin_match.group(1)
            continue
        xmax_match = _XMAX.match(line)
        if xmax_match:
            interval["xmax"] = xmax_match.group(1)
            continue
        text_match = _INTERVAL_TEXT.match(line)
        if text_match:
            interval["text"] = _praat_text(text_match.group(1))
    commit_interval()
    return tiers


def _tier(
    tiers: dict[str, list[_TextGridInterval]],
    expected: str,
) -> list[_TextGridInterval] | None:
    for name, intervals in tiers.items():
        normalized = re.sub(r"[^a-z]", "", name.casefold())
        if normalized == expected or normalized.endswith(expected):
            return intervals
    return None


def _find_acoustic_model(model_root: Path) -> Path | None:
    if not model_root.is_dir():
        return None
    candidates = sorted(
        (
            path
            for path in model_root.rglob("*")
            if path.is_file()
            and _MODEL_NAME in path.name.casefold()
            and "acoustic" in path.as_posix().casefold()
            and path.suffix.casefold() == ".zip"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


class MFAAlignmentAdapter(AlignmentAdapter):
    method = "mfa"
    name = "MFA Japanese"

    @staticmethod
    def _project_root(context: AlignmentContext | None) -> Path:
        return context.project_root if context else get_settings().project_root

    @classmethod
    def _mfa_executable(cls, context: AlignmentContext | None = None) -> Path | None:
        configured = os.environ.get("BEATFORGE_MFA_EXECUTABLE", "").strip()
        if configured:
            expanded = Path(configured).expanduser()
            if expanded.is_absolute() or expanded.parent != Path("."):
                return expanded
            resolved = shutil.which(configured)
            return Path(resolved) if resolved else expanded
        project_candidate = venv_executable(
            cls._project_root(context), ".venv-mfa", "mfa"
        )
        if project_candidate.is_file():
            return project_candidate
        resolved = shutil.which("mfa")
        return Path(resolved) if resolved else None

    @classmethod
    def _g2p_python(cls, context: AlignmentContext | None = None) -> Path:
        configured = os.environ.get("BEATFORGE_PYOPENJTALK_PYTHON", "").strip()
        if configured:
            value = Path(configured).expanduser()
            return value if value.is_absolute() else cls._project_root(context) / value
        qwen_python = venv_executable(cls._project_root(context), ".venv-qwen")
        if qwen_python.is_file():
            return qwen_python
        if importlib.util.find_spec("pyopenjtalk") is not None:
            return Path(sys.executable)
        return qwen_python

    @staticmethod
    def _model_root(context: AlignmentContext | None) -> Path:
        if context is not None:
            return context.models_dir / "mfa"
        return get_settings().models_dir / "mfa"

    @staticmethod
    def _probe(command: list[str], *, timeout: float = 15.0) -> tuple[bool, str]:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            return False, f"{type(error).__name__}: {error}"
        output = (completed.stdout or completed.stderr or "").strip()
        return completed.returncode == 0, output[-1_000:]

    def diagnostics(self, context: AlignmentContext | None = None) -> AdapterDiagnostics:
        mfa = self._mfa_executable(context)
        mfa_exists = bool(mfa and mfa.is_file() and os.access(mfa, os.X_OK))
        mfa_ok = False
        mfa_version = ""
        if mfa_exists and mfa is not None:
            mfa_ok, mfa_version = self._probe([str(mfa), "version"])
            if not mfa_ok:
                mfa_ok, mfa_version = self._probe([str(mfa), "--version"])

        g2p_python = self._g2p_python(context)
        g2p_exists = g2p_python.is_file() and os.access(g2p_python, os.X_OK)
        g2p_ok = False
        g2p_detail = ""
        if g2p_exists:
            g2p_ok, g2p_detail = self._probe(
                [
                    str(g2p_python),
                    "-c",
                    (
                        "import os; from pathlib import Path; import pyopenjtalk; "
                        "p=Path(os.fsdecode(pyopenjtalk.OPEN_JTALK_DICT_DIR)); "
                        "assert p.is_dir(); print(getattr(pyopenjtalk,'__version__','available'))"
                    ),
                ]
            )

        issues: list[str] = []
        if not mfa_exists or not mfa_ok:
            issues.append("mfa_cli_missing_or_broken")
        if not g2p_exists or not g2p_ok:
            issues.append("pyopenjtalk_missing_or_broken")
        model_root = self._model_root(context)
        cached_model = _find_acoustic_model(model_root)
        if "mfa_cli_missing_or_broken" in issues:
            reason = (
                "Montreal Forced Aligner is unavailable. Install it with conda-forge "
                "or set BEATFORGE_MFA_EXECUTABLE."
            )
        elif "pyopenjtalk_missing_or_broken" in issues:
            reason = (
                "pyopenjtalk/OpenJTalk dictionary is unavailable. Run "
                "python scripts/beatforge.py install-vocal and then "
                "python scripts/beatforge.py prepare-vocal-models."
            )
        else:
            reason = None
        return AdapterDiagnostics(
            available=not issues,
            reason=reason,
            model=f"{_MODEL_NAME}@{_MODEL_VERSION}",
            automatic_downloads_enabled=True,
            details={
                "mfaExecutable": str(mfa) if mfa else None,
                "mfaVersion": mfa_version or None,
                "g2pPython": str(g2p_python),
                "pyopenjtalk": g2p_detail or None,
                "modelRoot": str(model_root),
                "modelCached": cached_model is not None,
                "acousticModel": str(cached_model) if cached_model else None,
                "issues": issues,
            },
        )

    @staticmethod
    def _mfa_environment(model_root: Path) -> dict[str, str]:
        environment = os.environ.copy()
        environment["MFA_ROOT_DIR"] = str(model_root)
        environment.setdefault("OMP_NUM_THREADS", "1")
        environment.setdefault("OPENBLAS_NUM_THREADS", "1")
        environment.setdefault("MKL_NUM_THREADS", "1")
        return environment

    def _ensure_acoustic_model(
        self,
        context: AlignmentContext,
        mfa: Path,
    ) -> tuple[Path, bool]:
        configured = os.environ.get("BEATFORGE_MFA_ACOUSTIC_MODEL", "").strip()
        if configured:
            configured_path = Path(configured).expanduser()
            if not configured_path.is_absolute():
                configured_path = context.project_root / configured_path
            if not configured_path.is_file():
                raise AlignmentAdapterError(
                    "MFA_MODEL_UNAVAILABLE",
                    "The configured MFA acoustic model does not exist.",
                    status="unavailable",
                    details={"path": str(configured_path)},
                )
            return configured_path, False

        model_root = self._model_root(context)
        cached = _find_acoustic_model(model_root)
        if cached is not None:
            return cached, False
        model_root.mkdir(parents=True, exist_ok=True)
        command = [
            str(mfa),
            "model",
            "download",
            "--version",
            _MODEL_VERSION,
            "acoustic",
            _MODEL_NAME,
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=context.project_root,
                env=self._mfa_environment(model_root),
                capture_output=True,
                text=True,
                timeout=float(os.environ.get("BEATFORGE_ALIGNMENT_MFA_DOWNLOAD_TIMEOUT", "1800")),
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise AlignmentAdapterError(
                "MFA_MODEL_DOWNLOAD_TIMEOUT",
                "Japanese MFA acoustic-model download timed out.",
                status="unavailable",
                details={"timeoutSec": error.timeout, "modelRoot": str(model_root)},
            ) from error
        except OSError as error:
            raise AlignmentAdapterError(
                "MFA_MODEL_DOWNLOAD_FAILED",
                "Japanese MFA acoustic-model download could not start.",
                status="unavailable",
                details={"error": str(error), "modelRoot": str(model_root)},
            ) from error
        if completed.returncode != 0:
            raise AlignmentAdapterError(
                "MFA_MODEL_DOWNLOAD_FAILED",
                "Japanese MFA acoustic-model download failed.",
                status="unavailable",
                details={
                    "exitCode": completed.returncode,
                    "logTail": (completed.stderr or completed.stdout or "")[-4_000:],
                    "modelRoot": str(model_root),
                },
            )
        cached = _find_acoustic_model(model_root)
        if cached is None:
            raise AlignmentAdapterError(
                "MFA_MODEL_DOWNLOAD_INVALID",
                "MFA reported a successful download but no acoustic model was stored.",
                status="unavailable",
                details={"modelRoot": str(model_root)},
            )
        return cached, True

    def _generate_pronunciations(
        self,
        context: AlignmentContext,
        lyrics: str,
    ) -> list[_G2PWord]:
        python = self._g2p_python(context)
        try:
            completed = subprocess.run(
                [str(python), "-c", _G2P_SCRIPT],
                input=json.dumps({"text": lyrics}, ensure_ascii=False),
                capture_output=True,
                text=True,
                timeout=float(os.environ.get("BEATFORGE_ALIGNMENT_G2P_TIMEOUT", "120")),
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise AlignmentAdapterError(
                "MFA_G2P_TIMEOUT",
                "OpenJTalk G2P timed out.",
                details={"timeoutSec": error.timeout},
            ) from error
        except OSError as error:
            raise AlignmentAdapterError(
                "MFA_G2P_FAILED",
                "OpenJTalk G2P process could not start.",
                status="unavailable",
                details={"error": str(error), "python": str(python)},
            ) from error
        if completed.returncode != 0:
            raise AlignmentAdapterError(
                "MFA_G2P_FAILED",
                "OpenJTalk failed to generate a Japanese pronunciation.",
                details={
                    "exitCode": completed.returncode,
                    "logTail": (completed.stderr or completed.stdout or "")[-4_000:],
                },
            )
        try:
            payload = json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            raise AlignmentAdapterError(
                "MFA_G2P_OUTPUT_INVALID",
                "OpenJTalk returned unreadable G2P output.",
            ) from error
        raw_words = payload.get("words")
        if not isinstance(raw_words, list):
            raise AlignmentAdapterError(
                "MFA_G2P_OUTPUT_INVALID",
                "OpenJTalk G2P output does not contain words.",
            )
        words: list[_G2PWord] = []
        for item in raw_words:
            if not isinstance(item, dict):
                continue
            surface = str(item.get("text") or "").strip()
            raw_phones = item.get("phones")
            if not surface or not isinstance(raw_phones, list):
                continue
            phones = _openjtalk_phones_to_mfa([str(phone) for phone in raw_phones])
            if not phones:
                continue
            words.append(
                _G2PWord(
                    label=f"bfw{len(words):05d}",
                    text=surface,
                    phones=phones,
                )
            )
        if not words:
            raise AlignmentAdapterError(
                "MFA_G2P_EMPTY",
                "OpenJTalk did not produce any pronounceable lyric tokens.",
            )
        return words

    def _legacy_align_command(
        self,
        mfa: Path,
        environment: dict[str, str],
    ) -> str:
        configured = os.environ.get("BEATFORGE_MFA_ALIGN_COMMAND", "").strip()
        candidates = [configured] if configured else ["align_one", "align_one_legacy"]
        diagnostics: dict[str, str] = {}
        for candidate in candidates:
            if not candidate:
                continue
            try:
                completed = subprocess.run(
                    [str(mfa), candidate, "--help"],
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=30.0,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                diagnostics[candidate] = f"{type(error).__name__}: {error}"
                continue
            help_text = completed.stdout or completed.stderr or ""
            diagnostics[candidate] = help_text[-1_000:]
            if (
                completed.returncode == 0
                and "DICTIONARY_PATH" in help_text
                and "ACOUSTIC_MODEL_PATH" in help_text
            ):
                return candidate
        raise AlignmentAdapterError(
            "MFA_LEGACY_ALIGNMENT_UNAVAILABLE",
            "This MFA installation has no legacy align_one command for a custom dictionary.",
            status="unavailable",
            details={"commands": diagnostics},
        )

    def _tokens_from_textgrid(
        self,
        context: AlignmentContext,
        textgrid_path: Path,
        words: list[_G2PWord],
    ) -> tuple[list[AlignmentToken], dict[str, Any]]:
        try:
            tiers = _parse_long_textgrid(textgrid_path.read_text(encoding="utf-8"))
        except OSError as error:
            raise AlignmentAdapterError(
                "MFA_TEXTGRID_UNREADABLE",
                "MFA TextGrid could not be read.",
            ) from error
        word_intervals = _tier(tiers, "words")
        phone_intervals = _tier(tiers, "phones")
        if word_intervals is None:
            raise AlignmentAdapterError(
                "MFA_TEXTGRID_WORD_TIER_MISSING",
                "MFA TextGrid has no word interval tier.",
            )
        if phone_intervals is None:
            raise AlignmentAdapterError(
                "MFA_TEXTGRID_PHONE_TIER_MISSING",
                "MFA TextGrid has no phone interval tier.",
            )
        surfaces = {word.label.casefold(): word.text for word in words}
        known_words = [
            interval for interval in word_intervals if interval.text.casefold() in surfaces
        ]
        tokens: list[AlignmentToken] = []
        unmatched_phone_count = 0
        skipped_silence_count = 0
        for interval in sorted(phone_intervals, key=lambda item: (item.start_sec, item.end_sec)):
            phoneme = interval.text.strip()
            if phoneme.casefold() in _SILENCE_PHONES:
                skipped_silence_count += 1
                continue
            owner = max(
                known_words,
                key=lambda word: max(
                    0.0,
                    min(interval.end_sec, word.end_sec)
                    - max(interval.start_sec, word.start_sec),
                ),
                default=None,
            )
            overlap = (
                max(
                    0.0,
                    min(interval.end_sec, owner.end_sec)
                    - max(interval.start_sec, owner.start_sec),
                )
                if owner is not None
                else 0.0
            )
            if owner is None or overlap <= 0.0:
                unmatched_phone_count += 1
                continue
            if context.sample_count <= 0:
                break
            start = min(
                max(0, int(round(interval.start_sec * context.sample_rate))),
                context.sample_count - 1,
            )
            end = min(
                max(0, int(round(interval.end_sec * context.sample_rate))),
                context.sample_count,
            )
            if end <= start:
                continue
            index = len(tokens)
            tokens.append(
                AlignmentToken(
                    id=alignment_token_id(context.track_id, self.method, index, start, end),
                    text=surfaces[owner.text.casefold()],
                    phoneme=phoneme,
                    start_sample=start,
                    end_sample=end,
                    # Long TextGrid intervals contain boundaries but no calibrated
                    # posterior.  Zero records unknown confidence honestly.
                    confidence=0.0,
                    method=self.method,
                )
            )
        if not tokens:
            raise AlignmentAdapterError(
                "MFA_ALIGNMENT_EMPTY",
                "MFA returned no usable phone intervals.",
                details={
                    "wordIntervalCount": len(word_intervals),
                    "phoneIntervalCount": len(phone_intervals),
                    "unmatchedPhoneCount": unmatched_phone_count,
                },
            )
        return tokens, {
            "wordIntervalCount": len(word_intervals),
            "phoneIntervalCount": len(phone_intervals),
            "unmatchedPhoneCount": unmatched_phone_count,
            "skippedSilencePhoneCount": skipped_silence_count,
        }

    def run(self, context: AlignmentContext) -> AdapterOutput:
        diagnostics = self.diagnostics(context)
        if not diagnostics.available:
            raise AlignmentAdapterError(
                "MFA_RUNTIME_UNAVAILABLE",
                diagnostics.reason or "MFA alignment runtime is unavailable.",
                status="unavailable",
                details=diagnostics.details,
            )
        lyrics = clean_lyrics(context.lyrics, context.lyrics_format)
        if not lyrics:
            raise AlignmentAdapterError("LYRICS_REQUIRED", "MFA alignment requires saved lyrics.")
        if context.lyrics_format == "romaji":
            raise AlignmentAdapterError(
                "ROMAJI_REQUIRES_KANA",
                "MFA Japanese alignment requires Japanese text or confirmed kana.",
            )
        mfa = self._mfa_executable(context)
        if mfa is None:
            raise AlignmentAdapterError(
                "MFA_RUNTIME_UNAVAILABLE",
                "MFA executable disappeared after diagnostics.",
                status="unavailable",
            )
        acoustic_model, downloaded = self._ensure_acoustic_model(context, mfa)
        words = self._generate_pronunciations(context, lyrics)
        alignment_root = context.storage_dir / "alignment"
        alignment_root.mkdir(parents=True, exist_ok=True)
        model_root = self._model_root(context)
        environment = self._mfa_environment(model_root)
        command_name = self._legacy_align_command(mfa, environment)
        with tempfile.TemporaryDirectory(prefix="mfa-lab-", dir=alignment_root) as temporary:
            directory = Path(temporary)
            transcript_path = directory / "lyrics.lab"
            dictionary_path = directory / "openjtalk.dict"
            output_path = directory / "alignment.TextGrid"
            work_path = directory / "work"
            transcript_path.write_text(" ".join(word.label for word in words), encoding="utf-8")
            dictionary_path.write_text(
                "".join(
                    f"{word.label}\t{' '.join(word.phones)}\n" for word in words
                ),
                encoding="utf-8",
            )
            command = [
                str(mfa),
                command_name,
                "--output_format",
                "long_textgrid",
                "--single_speaker",
                "--no_use_mp",
                "--temporary_directory",
                str(work_path),
                str(context.vocals_path),
                str(transcript_path),
                str(dictionary_path),
                str(acoustic_model),
                str(output_path),
            ]
            try:
                completed = subprocess.run(
                    command,
                    cwd=context.project_root,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=float(os.environ.get("BEATFORGE_ALIGNMENT_MFA_TIMEOUT", "7200")),
                    check=False,
                )
            except subprocess.TimeoutExpired as error:
                raise AlignmentAdapterError(
                    "MFA_ALIGNMENT_TIMEOUT",
                    "MFA Japanese alignment timed out.",
                    details={"timeoutSec": error.timeout},
                ) from error
            except OSError as error:
                raise AlignmentAdapterError(
                    "MFA_ALIGNMENT_PROCESS_FAILED",
                    "MFA Japanese alignment process could not start.",
                    details={"error": str(error)},
                ) from error
            if completed.returncode != 0 or not output_path.is_file():
                raise AlignmentAdapterError(
                    "MFA_ALIGNMENT_PROCESS_FAILED",
                    "MFA Japanese alignment failed without usable TextGrid output.",
                    details={
                        "exitCode": completed.returncode,
                        "logTail": (completed.stderr or completed.stdout or "")[-4_000:],
                    },
                )
            tokens, textgrid_metadata = self._tokens_from_textgrid(
                context,
                output_path,
                words,
            )

        source_rate = int(sf.info(context.vocals_path).samplerate)
        warnings = [
            "MFA TextGrid has no calibrated token posterior; token confidence is reported as 0.0."
        ]
        if textgrid_metadata["unmatchedPhoneCount"]:
            warnings.append(
                f"{textgrid_metadata['unmatchedPhoneCount']} MFA phone intervals were outside "
                "known word intervals and were omitted without inventing timestamps."
            )
        return AdapterOutput(
            tokens=tuple(tokens),
            warnings=tuple(warnings),
            metadata={
                "model": f"{_MODEL_NAME}@{_MODEL_VERSION}",
                "modelPath": str(acoustic_model),
                "modelDownloaded": downloaded,
                "g2p": "pyopenjtalk/OpenJTalk",
                "phoneSet": "Japanese MFA IPA",
                "sourceSampleRate": source_rate,
                "alignedText": "".join(word.text for word in words),
                "pronunciationWordCount": len(words),
                "timestampProvenance": "MFA TextGrid phone intervals",
                "confidenceProvenance": "unavailable_from_mfa_textgrid",
                **textgrid_metadata,
            },
        )


__all__ = ["MFAAlignmentAdapter"]
