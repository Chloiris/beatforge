"""Persistent local jobs for Japanese singing transcription and lyric alignment."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import threading
import time
import unicodedata
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from .audio.vocal_candidates import (
    VocalAcousticCandidate,
    extract_vocal_acoustic_candidates,
)
from .audio.vocal_timing import (
    VocalGridConfig,
    VocalTimingAnchor,
    align_vocal_anchors_to_grid,
)
from .config import get_settings
from .database import SessionLocal
from .models import CandidateEventModel, HitPointModel, TrackModel, VocalAlignmentJobModel
from .platform_paths import venv_executable
from .serialization import dumps
from .timing import nearest_grid_sample

_executor = ThreadPoolExecutor(max_workers=1)
_futures: dict[str, Future[None]] = {}
_future_lock = threading.Lock()
_LRC_TIMESTAMP = re.compile(r"\[(?:\d{1,3}:)?\d{1,2}:\d{2}(?:[.:]\d{1,3})?\]")
_LRC_METADATA = re.compile(r"^\[[a-zA-Z]+:[^]]*]\s*$")
_MIN_RELIABLE_TIMESTAMP_MS = 30.0
_MAX_VOCAL_SNAP_DISTANCE_STEPS = 0.30
_MIN_ANCHOR_CONFIDENCE = 0.34
_CROSS_STEM_MERGE_MS = 10.0
_MIN_ACTIVITY_OR_ATTACK_ACTIVITY = 0.08
_MIN_ACTIVITY_OR_ATTACK_ATTACK = 0.16
_MIN_CHUNK_REPLACEMENT_ANCHORS = 2
_MIN_CHUNK_REPLACEMENT_CONFIDENCE = 0.28
_FALLBACK_CHUNK_SECONDS = 20.0


@dataclass(frozen=True, slots=True)
class VocalActivityProfile:
    """Absolute vocal activity evidence used to reject separated-stem leakage."""

    envelope: np.ndarray
    attack: np.ndarray
    rms: np.ndarray
    presence_rms: np.ndarray
    activity_floor: float
    activity_reference: float
    attack_reference: float


@dataclass(frozen=True, slots=True)
class VocalAnchorBuildResult:
    anchors: list[dict[str, Any]]
    statistics: dict[str, int | float]


@dataclass(frozen=True, slots=True)
class VocalHitReplacementResult:
    created_hit_count: int
    removed_hit_count: int
    replaced_chunk_count: int
    preserved_fallback_hit_count: int


class VocalJobError(RuntimeError):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


def clean_lyrics(text: str, input_format: str) -> str:
    """Normalize user text while retaining phrase line breaks."""

    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    lines: list[str] = []
    for raw_line in normalized.splitlines():
        line = raw_line.strip()
        if input_format == "lrc":
            if _LRC_METADATA.fullmatch(line):
                continue
            line = _LRC_TIMESTAMP.sub("", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _update_job(job_id: str, **values: Any) -> None:
    with SessionLocal() as session:
        job = session.get(VocalAlignmentJobModel, job_id)
        if job is None:
            return
        for name, value in values.items():
            setattr(job, name, value)
        job.updated_at = datetime.now(UTC)
        session.commit()


def _runtime_paths() -> tuple[Path, Path, Path, Path]:
    settings = get_settings()
    def resolve(value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else settings.project_root / path

    python = resolve(
        os.environ.get(
            "BEATFORGE_QWEN_PYTHON",
            str(venv_executable(settings.project_root, ".venv-qwen")),
        )
    )
    script = settings.project_root / "scripts" / "qwen_vocal_cli.py"
    asr_model = resolve(
        os.environ.get(
            "BEATFORGE_QWEN_ASR_MODEL",
            str(settings.models_dir / "Qwen3-ASR-1.7B"),
        )
    )
    aligner_model = resolve(
        os.environ.get(
            "BEATFORGE_QWEN_ALIGNER_MODEL",
            str(settings.models_dir / "Qwen3-ForcedAligner-0.6B"),
        )
    )
    return python, script, asr_model, aligner_model


def vocal_runtime_diagnostics() -> dict[str, Any]:
    python, script, asr_model, aligner_model = _runtime_paths()
    return {
        "pythonAvailable": python.is_file(),
        "scriptAvailable": script.is_file(),
        "asrModelAvailable": (asr_model / "config.json").is_file(),
        "alignerModelAvailable": (aligner_model / "config.json").is_file(),
        "python": str(python),
        "asrModel": str(asr_model),
        "alignerModel": str(aligner_model),
        "automaticDownloadsEnabled": False,
    }


def _run_qwen(
    operation: str,
    *,
    audio_path: Path,
    text: str | None = None,
    timeout: float = 3_600.0,
) -> dict[str, Any]:
    python, script, asr_model, aligner_model = _runtime_paths()
    diagnostics = vocal_runtime_diagnostics()
    required = ["pythonAvailable", "scriptAvailable"]
    if operation == "transcribe":
        required.append("asrModelAvailable")
    elif operation == "align_song":
        required.extend(("asrModelAvailable", "alignerModelAvailable"))
    else:
        required.append("alignerModelAvailable")
    missing = [name for name in required if not diagnostics[name]]
    if missing:
        raise VocalJobError(
            "VOCAL_MODEL_NOT_READY",
            (
                "本地日语人声模型尚未准备好，请先运行 "
                "python scripts/beatforge.py prepare-vocal-models。"
            ),
            {"missing": missing, **diagnostics},
        )

    settings = get_settings()
    with tempfile.TemporaryDirectory(
        prefix="qwen-vocal-", dir=settings.vocal_alignment_dir
    ) as temporary_directory:
        directory = Path(temporary_directory)
        output_path = directory / "result.json"
        command = [
            str(python),
            str(script),
            operation,
            "--audio",
            str(audio_path),
            "--output",
            str(output_path),
            "--asr-model",
            str(asr_model),
            "--aligner-model",
            str(aligner_model),
            "--device",
            os.environ.get("BEATFORGE_QWEN_DEVICE", "auto"),
        ]
        if text is not None:
            text_path = directory / "lyrics.txt"
            text_path.write_text(text, encoding="utf-8")
            command.extend(("--text-file", str(text_path)))
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
                cwd=settings.project_root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            raise VocalJobError(
                "VOCAL_MODEL_TIMEOUT",
                "本地人声模型运行超时。",
                {"timeoutSec": timeout},
            ) from error
        if completed.returncode != 0 or not output_path.is_file():
            detail = (completed.stderr or completed.stdout or "")[-4_000:]
            raise VocalJobError(
                "VOCAL_MODEL_PROCESS_FAILED",
                "本地人声模型进程失败。",
                {"exitCode": completed.returncode, "logTail": detail},
            )
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    if payload.get("status") != "ok":
        raise VocalJobError(
            str(payload.get("error_code") or "VOCAL_MODEL_FAILED").upper(),
            str(payload.get("error_message") or "本地人声模型未返回结果。"),
            {"warnings": payload.get("warnings", [])},
        )
    return payload


def _load_vocals(track_id: str) -> tuple[Path, np.ndarray, int]:
    settings = get_settings()
    stem_root = settings.stems_dir.resolve()
    path = (stem_root / track_id / "vocals.flac").resolve()
    if not path.is_relative_to(stem_root) or not path.is_file():
        raise VocalJobError(
            "VOCALS_STEM_NOT_READY",
            "尚未生成人声分轨，请先使用精确模式重新分析歌曲。",
        )
    values, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if values.ndim == 2:
        values = np.mean(values, axis=1, dtype=np.float32)
    return path, np.ascontiguousarray(values, dtype=np.float32), int(sample_rate)


def _vocal_activity_profile(audio: np.ndarray, sample_rate: int) -> VocalActivityProfile:
    """Build absolute and relative evidence without normalizing silence into an onset."""

    nyquist = sample_rate / 2.0
    high = min(1_600.0, nyquist * 0.92)
    low = min(90.0, high * 0.4)
    filtered = sosfiltfilt(
        butter(3, (low / nyquist, high / nyquist), btype="bandpass", output="sos"),
        audio.astype(np.float64, copy=False),
    )
    window = max(3, int(round(sample_rate * 0.003)))
    envelope = uniform_filter1d(np.abs(filtered), size=window, mode="nearest")
    attack = np.maximum(np.gradient(envelope), 0.0)
    rms_window = max(3, int(round(sample_rate * 0.024)))
    rms = np.sqrt(
        np.maximum(
            uniform_filter1d(
                np.square(audio, dtype=np.float64),
                size=rms_window,
                mode="nearest",
            ),
            0.0,
        )
    )
    presence_window = max(2, int(round(sample_rate * 0.004)))
    presence_rms = np.sqrt(
        np.maximum(
            uniform_filter1d(
                np.square(audio, dtype=np.float64),
                size=presence_window,
                mode="nearest",
            ),
            0.0,
        )
    )
    activity_reference = max(float(np.quantile(rms, 0.90)), 1e-9)
    noise_floor = max(float(np.quantile(rms, 0.20)), 1e-9)
    activity_floor = max(
        10.0 ** (-52.0 / 20.0),
        activity_reference * 0.025,
        noise_floor * 6.0,
    )
    attack_reference = max(float(np.quantile(attack, 0.99)), 1e-9)
    return VocalActivityProfile(
        envelope=envelope.astype(np.float32),
        attack=attack.astype(np.float32),
        rms=rms.astype(np.float32),
        presence_rms=presence_rms.astype(np.float32),
        activity_floor=activity_floor,
        activity_reference=activity_reference,
        attack_reference=attack_reference,
    )


def _first_sustained_activity(
    profile: VocalActivityProfile,
    start_sample: int,
    end_sample: int,
    sample_rate: int,
) -> int | None:
    """Return the first stable vocal region inside an aligner word span."""

    start = min(max(start_sample, 0), max(0, profile.rms.size - 1))
    stop = min(max(end_sample, start + 1), profile.rms.size)
    # A short RMS window prevents a click from being expanded into an apparent
    # 18 ms voiced region by the 24 ms confidence envelope.
    active = profile.presence_rms[start:stop] >= profile.activity_floor
    if not np.any(active):
        return None
    hold = min(active.size, max(3, int(round(sample_rate * 0.018))))
    required = max(1, int(np.ceil(hold * 0.72)))
    prefix = np.concatenate(([0], np.cumsum(active, dtype=np.int64)))
    support = prefix[hold:] - prefix[:-hold]
    stable = np.flatnonzero(support >= required)
    if stable.size:
        return start + int(stable[0])
    return None


def _refine_sample(
    aligned_sample: int,
    end_sample: int,
    *,
    profile: VocalActivityProfile,
    sample_rate: int,
) -> tuple[int, float, float, float]:
    start = max(0, aligned_sample - round(sample_rate * 0.008))
    stop = min(
        profile.attack.size,
        max(
            aligned_sample + 1,
            min(
                end_sample + round(sample_rate * 0.015),
                aligned_sample + round(sample_rate * 0.090),
            ),
        ),
    )
    if stop <= start + 2:
        sample = min(max(aligned_sample, 0), max(0, profile.attack.size - 1))
        return sample, 0.0, 0.0, float(profile.rms[sample])
    local_attack = profile.attack[start:stop]
    local_envelope = profile.envelope[start:stop]
    local_rms = profile.rms[start:stop]
    delay = np.arange(start, stop, dtype=np.float64) - aligned_sample
    delay_penalty = np.maximum(delay - sample_rate * 0.020, 0.0) / (sample_rate * 0.030)
    score = (
        0.66 * np.clip(local_attack / profile.attack_reference, 0.0, 2.0)
        + 0.20 * np.clip(local_envelope / profile.activity_reference, 0.0, 1.5)
        + 0.14 * np.clip(local_rms / profile.activity_reference, 0.0, 1.5)
        - 0.80 * delay_penalty
    )
    relative = int(np.argmax(score))
    refined = start + relative
    attack_score = float(
        np.clip(local_attack[relative] / profile.attack_reference, 0.0, 1.0)
    )
    activity_score = float(
        np.clip(
            (local_rms[relative] - profile.activity_floor)
            / max(profile.activity_reference - profile.activity_floor, 1e-9),
            0.0,
            1.0,
        )
    )
    return refined, attack_score, activity_score, float(local_rms[relative])


def _map_sample(sample: int, source_rate: int, target_rate: int) -> int:
    if source_rate == target_rate:
        return int(sample)
    return int(round(int(sample) * target_rate / source_rate))


def _build_anchors(
    timestamps: list[dict[str, Any]],
    *,
    audio: np.ndarray,
    sample_rate: int,
    original_sample_rate: int | None = None,
    bpm: float,
    beat_offset_sample: int,
    acoustic_candidates: list[VocalAcousticCandidate] | None = None,
) -> VocalAnchorBuildResult:
    target_sample_rate = original_sample_rate or sample_rate
    profile = _vocal_activity_profile(audio, sample_rate)
    acoustic: list[VocalTimingAnchor] = []
    evidence: list[dict[str, float | bool | int]] = []
    rejected_short = 0
    rejected_silent = 0
    rejected_low_confidence = 0
    rejected_chunk_match = 0
    rejected_weak_attack = 0
    used_acoustic_samples: set[int] = set()
    for item in timestamps:
        start = min(max(int(item.get("start_sample", 0)), 0), max(0, audio.size - 1))
        end = min(max(int(item.get("end_sample", start + 1)), start + 1), audio.size)
        raw_text = str(item.get("text", "")).strip()
        if not raw_text:
            continue
        chunk_match_confidence = float(item.get("chunk_match_confidence", 1.0))
        if not np.isfinite(chunk_match_confidence) or chunk_match_confidence < 0.28:
            rejected_chunk_match += 1
            continue
        span = max(1, end - start)
        duration_ms = span * 1000.0 / sample_rate
        if duration_ms < _MIN_RELIABLE_TIMESTAMP_MS:
            rejected_short += 1
            continue
        active_start = _first_sustained_activity(profile, start, end, sample_rate)
        if active_start is None:
            rejected_silent += 1
            continue
        acoustic_window_start = max(0, start - round(sample_rate * 0.080))
        acoustic_window_end = min(
            audio.size,
            min(end + round(sample_rate * 0.015), start + round(sample_rate * 0.220)),
        )
        nearby_acoustic = [
            candidate
            for candidate in acoustic_candidates or []
            if acoustic_window_start <= candidate.sample < acoustic_window_end
            and candidate.sample not in used_acoustic_samples
        ]
        matched_acoustic = (
            min(
                nearby_acoustic,
                key=lambda candidate: (
                    abs(candidate.sample - active_start),
                    -candidate.confidence,
                ),
            )
            if nearby_acoustic
            else None
        )
        if matched_acoustic is not None:
            used_acoustic_samples.add(matched_acoustic.sample)
            refined = matched_acoustic.sample
            attack_score = max(
                matched_acoustic.onset_score,
                matched_acoustic.envelope_score,
            )
            activity_score = matched_acoustic.activity_score
            rms_value = float(profile.rms[min(refined, profile.rms.size - 1)])
            pitch_score = matched_acoustic.pitch_score
            transition_score = matched_acoustic.transition_score
            acoustic_confidence = matched_acoustic.confidence
        else:
            refined, attack_score, activity_score, rms_value = _refine_sample(
                active_start,
                end,
                profile=profile,
                sample_rate=sample_rate,
            )
            pitch_score = 0.0
            transition_score = 0.0
            acoustic_confidence = float(
                np.clip(0.58 * attack_score + 0.42 * activity_score, 0.0, 1.0)
            )
        if rms_value < profile.activity_floor:
            rejected_silent += 1
            continue
        if (
            activity_score < _MIN_ACTIVITY_OR_ATTACK_ACTIVITY
            and attack_score < _MIN_ACTIVITY_OR_ATTACK_ATTACK
        ):
            rejected_weak_attack += 1
            continue
        alignment_shift_ms = abs(refined - start) * 1000.0 / sample_rate
        duration_quality = float(
            np.exp(-abs(np.log(max(duration_ms, 1.0) / 340.0)) * 0.24)
        )
        shift_quality = float(
            np.exp(-max(0.0, alignment_shift_ms - 100.0) / 520.0)
        )
        confidence = float(
            np.clip(
                (
                    0.15
                    + 0.25 * duration_quality
                    + 0.38 * activity_score
                    + 0.14 * attack_score
                    + 0.08 * acoustic_confidence
                )
                * shift_quality
                * (0.62 + 0.38 * chunk_match_confidence),
                0.0,
                0.94,
            )
        )
        if confidence < _MIN_ANCHOR_CONFIDENCE:
            rejected_low_confidence += 1
            continue
        aligned_target = _map_sample(start, sample_rate, target_sample_rate)
        refined_target = _map_sample(refined, sample_rate, target_sample_rate)
        if acoustic and refined_target <= acoustic[-1].refined_sample:
            refined_target = acoustic[-1].refined_sample + 1
        reading = str(item.get("kana") or raw_text).strip() or raw_text
        item_romaji = str(item.get("romaji", "")).strip()
        acoustic.append(
            VocalTimingAnchor(
                original_text=raw_text,
                kana=reading,
                romaji=item_romaji or None,
                aligned_sample=aligned_target,
                refined_sample=refined_target,
                confidence=confidence,
                kind="phrase",
            )
        )
        evidence.append(
            {
                "chunk_index": int(item.get("chunk_index", -1)),
                "word_start": True,
                "active": True,
                "activity_score": round(activity_score, 6),
                "attack_score": round(attack_score, 6),
                "pitch_score": round(pitch_score, 6),
                "transition_score": round(transition_score, 6),
                "acoustic_confidence": round(acoustic_confidence, 6),
                "alignment_shift_ms": round(alignment_shift_ms, 3),
                "chunk_match_confidence": round(chunk_match_confidence, 6),
            }
        )
    quantized = align_vocal_anchors_to_grid(
        acoustic,
        sample_rate=target_sample_rate,
        bpm=bpm,
        beat_offset_sample=beat_offset_sample,
        config=VocalGridConfig(
            max_snap_distance_steps=_MAX_VOCAL_SNAP_DISTANCE_STEPS,
        ),
    )
    output: list[dict[str, Any]] = []
    for index, (anchor, anchor_evidence) in enumerate(zip(quantized, evidence, strict=True)):
        stable_key = f"vocal-anchor:{index}:{anchor.aligned_sample}:{anchor.original_text}"
        output.append(
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key)),
                "index": index,
                "original_text": anchor.original_text,
                "kana": anchor.kana,
                "romaji": anchor.romaji or "",
                "aligned_sample": anchor.aligned_sample,
                "refined_sample": anchor.refined_sample,
                "grid_sample": anchor.grid_sample,
                "confidence": round(anchor.confidence, 6),
                "word_start": bool(anchor_evidence["word_start"]),
                "active": bool(anchor_evidence["active"]),
                "chart_candidate": anchor.grid_sample is not None,
                "activity_score": float(anchor_evidence["activity_score"]),
                "attack_score": float(anchor_evidence["attack_score"]),
                "pitch_score": float(anchor_evidence["pitch_score"]),
                "transition_score": float(anchor_evidence["transition_score"]),
                "acoustic_confidence": float(anchor_evidence["acoustic_confidence"]),
                "alignment_shift_ms": float(anchor_evidence["alignment_shift_ms"]),
                "chunk_match_confidence": float(
                    anchor_evidence["chunk_match_confidence"]
                ),
                "chunk_index": int(anchor_evidence["chunk_index"]),
                "semantic_unit": "phrase",
            }
        )
    grid_anchor_count = sum(anchor["grid_sample"] is not None for anchor in output)
    return VocalAnchorBuildResult(
        anchors=output,
        statistics={
            "timestampCount": len(timestamps),
            "rejectedShortTimestampCount": rejected_short,
            "rejectedSilentCount": rejected_silent,
            "rejectedLowConfidenceCount": rejected_low_confidence,
            "rejectedChunkMatchCount": rejected_chunk_match,
            "rejectedWeakAttackCount": rejected_weak_attack,
            "reliableAnchorCount": len(output),
            "gridAnchorCount": grid_anchor_count,
            "activityFloorDbfs": round(20.0 * np.log10(profile.activity_floor), 3),
            "sourceSampleRate": sample_rate,
            "originalSampleRate": target_sample_rate,
        },
    )


def _build_chunk_coverage(
    diagnostics: list[dict[str, Any]],
    timestamps: list[dict[str, Any]],
    anchors: list[dict[str, Any]],
    *,
    source_sample_rate: int,
    target_sample_rate: int,
) -> list[dict[str, Any]]:
    """Normalize model diagnostics into conservative replacement intervals."""

    raw_counts: dict[int, int] = {}
    for timestamp in timestamps:
        chunk_index = int(timestamp.get("chunk_index", -1))
        if chunk_index >= 0:
            raw_counts[chunk_index] = raw_counts.get(chunk_index, 0) + 1

    usable_anchors: dict[int, list[dict[str, Any]]] = {}
    for anchor in anchors:
        chunk_index = int(anchor.get("chunk_index", -1))
        if (
            chunk_index >= 0
            and anchor.get("active", False)
            and anchor.get("chart_candidate", False)
        ):
            usable_anchors.setdefault(chunk_index, []).append(anchor)

    coverage: list[dict[str, Any]] = []
    for fallback_index, diagnostic in enumerate(diagnostics):
        chunk_index = int(diagnostic.get("index", fallback_index))
        source_start = max(0, int(diagnostic.get("startSample", 0)))
        source_end = max(source_start + 1, int(diagnostic.get("endSample", source_start + 1)))
        start_sample = _map_sample(source_start, source_sample_rate, target_sample_rate)
        end_sample = _map_sample(source_end, source_sample_rate, target_sample_rate)
        chunk_anchors = usable_anchors.get(chunk_index, [])
        anchor_count = len(chunk_anchors)
        raw_timestamp_count = raw_counts.get(chunk_index, 0)
        match_confidence = float(diagnostic.get("matchConfidence", 0.0))
        if not np.isfinite(match_confidence):
            match_confidence = 0.0
        anchor_confidence = (
            float(np.mean([float(anchor.get("confidence", 0.0)) for anchor in chunk_anchors]))
            if chunk_anchors
            else 0.0
        )
        confidence = min(max(match_confidence, 0.0), max(anchor_confidence, 0.0))
        asr_status = str(diagnostic.get("status", "unknown"))
        alignment_status = diagnostic.get("alignmentStatus")
        if asr_status == "silent":
            status = "silent"
        elif asr_status != "ok":
            status = "asr_failed"
        elif alignment_status is None:
            status = "unassigned"
        elif alignment_status != "ok":
            status = "alignment_failed"
        elif match_confidence < _MIN_CHUNK_REPLACEMENT_CONFIDENCE:
            status = "low_confidence"
        elif raw_timestamp_count > 0 and anchor_count == 0:
            status = "alignment_collapse"
        elif anchor_count < _MIN_CHUNK_REPLACEMENT_ANCHORS:
            status = "insufficient_anchors"
        elif confidence < _MIN_CHUNK_REPLACEMENT_CONFIDENCE:
            status = "low_confidence"
        else:
            status = "success"
        coverage.append(
            {
                "index": chunk_index,
                "startSample": start_sample,
                "endSample": max(start_sample + 1, end_sample),
                "status": status,
                "confidence": round(float(np.clip(confidence, 0.0, 1.0)), 6),
                "anchorCount": anchor_count,
                "rawTimestampCount": raw_timestamp_count,
            }
        )
    return coverage


def _failed_chunk_coverage(
    sample_count: int,
    sample_rate: int,
) -> list[dict[str, Any]]:
    chunk_size = max(1, round(sample_rate * _FALLBACK_CHUNK_SECONDS))
    return [
        {
            "index": index,
            "startSample": start,
            "endSample": min(sample_count, start + chunk_size),
            "status": "alignment_failed",
            "confidence": 0.0,
            "anchorCount": 0,
            "rawTimestampCount": 0,
        }
        for index, start in enumerate(range(0, sample_count, chunk_size))
    ]


def _detect_vocal_fallback_anchors(
    audio: np.ndarray,
    *,
    sample_rate: int,
    target_sample_rate: int,
    bpm: float,
    beat_offset_sample: int,
    coverage_chunks: list[dict[str, Any]],
    acoustic_candidates: list[VocalAcousticCandidate] | None = None,
) -> list[dict[str, Any]]:
    """Detect real vocal-stem attacks only inside uncovered alignment chunks."""

    uncovered = [
        chunk for chunk in coverage_chunks if str(chunk.get("status")) != "success"
    ]
    if not uncovered or audio.size < 3:
        return []
    detected = acoustic_candidates
    if detected is None:
        detected = extract_vocal_acoustic_candidates(audio, sample_rate).candidates
    output: list[dict[str, Any]] = []
    for candidate in detected:
        source_sample = candidate.sample
        acoustic_sample = _map_sample(source_sample, sample_rate, target_sample_rate)
        chunk = next(
            (
                item
                for item in uncovered
                if int(item.get("startSample", 0))
                <= acoustic_sample
                < int(item.get("endSample", 0))
            ),
            None,
        )
        if chunk is None:
            continue
        activity = candidate.activity_score
        attack = max(candidate.onset_score, candidate.envelope_score)
        rise = candidate.envelope_score
        chart_sample = nearest_grid_sample(
            acoustic_sample,
            sample_rate=target_sample_rate,
            bpm=bpm,
            beat_offset_sample=beat_offset_sample,
            subdivisions_per_beat=4,
        )
        confidence = candidate.confidence
        stable_key = f"vocal-fallback:{acoustic_sample}:{int(chunk.get('index', -1))}"
        output.append(
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key)),
                "chunk_index": int(chunk.get("index", -1)),
                "acoustic_sample": acoustic_sample,
                "chart_sample": chart_sample,
                "snap_error_ms": (acoustic_sample - chart_sample)
                * 1000.0
                / target_sample_rate,
                "confidence": confidence,
                "activity_score": activity,
                "attack_score": attack,
                "rise_score": rise,
                "pitch_score": candidate.pitch_score,
                "transition_score": candidate.transition_score,
                "fallback_level": "vocal_acoustic_onset",
            }
        )
    covered_chunk_indexes = {int(anchor["chunk_index"]) for anchor in output}
    energy_only_chunks = [
        chunk
        for chunk in uncovered
        if int(chunk.get("index", -1)) not in covered_chunk_indexes
    ]
    output.extend(
        _beat_aligned_vocal_energy_anchors(
            audio,
            sample_rate=sample_rate,
            target_sample_rate=target_sample_rate,
            bpm=bpm,
            beat_offset_sample=beat_offset_sample,
            coverage_chunks=energy_only_chunks,
        )
    )
    return output


def _beat_aligned_vocal_energy_anchors(
    audio: np.ndarray,
    *,
    sample_rate: int,
    target_sample_rate: int,
    bpm: float,
    beat_offset_sample: int,
    coverage_chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Last-resort grid-aware fallback over real vocal-energy attacks.

    The grid only ranks locally detected energy rises. It never fills a cell that
    lacks absolute vocal activity and a measured envelope/RMS change.
    """

    if not coverage_chunks or audio.size < max(2_048, sample_rate // 2):
        return []
    profile = _vocal_activity_profile(audio, sample_rate)
    rms_rise = np.maximum(np.gradient(profile.rms.astype(np.float64)), 0.0)
    rise_reference = max(float(np.quantile(rms_rise, 0.99)), 1e-9)
    novelty = uniform_filter1d(
        0.62 * np.clip(profile.attack / profile.attack_reference, 0.0, 1.5)
        + 0.38 * np.clip(rms_rise / rise_reference, 0.0, 1.5),
        size=max(3, round(sample_rate * 0.004)),
        mode="nearest",
    )
    peaks, properties = find_peaks(
        novelty,
        height=0.10,
        prominence=0.025,
        distance=max(1, round(sample_rate * 0.080)),
    )
    prominences = properties.get("prominences", np.zeros(peaks.size))
    step_samples = target_sample_rate * 60.0 / max(bpm, 1e-9) / 4.0
    anchors: list[dict[str, Any]] = []
    for source_sample, prominence in zip(peaks, prominences, strict=False):
        if profile.rms[source_sample] < profile.activity_floor:
            continue
        acoustic_sample = _map_sample(
            int(source_sample),
            sample_rate,
            target_sample_rate,
        )
        chunk = next(
            (
                item
                for item in coverage_chunks
                if int(item.get("startSample", 0))
                <= acoustic_sample
                < int(item.get("endSample", 0))
            ),
            None,
        )
        if chunk is None:
            continue
        chart_sample = nearest_grid_sample(
            acoustic_sample,
            sample_rate=target_sample_rate,
            bpm=bpm,
            beat_offset_sample=beat_offset_sample,
            subdivisions_per_beat=4,
        )
        grid_distance_steps = abs(acoustic_sample - chart_sample) / max(step_samples, 1.0)
        grid_confidence = float(np.exp(-0.5 * (grid_distance_steps / 0.35) ** 2))
        if grid_confidence < 0.18:
            continue
        activity_score = float(
            np.clip(
                (profile.rms[source_sample] - profile.activity_floor)
                / max(profile.activity_reference - profile.activity_floor, 1e-9),
                0.0,
                1.0,
            )
        )
        attack_score = float(
            np.clip(profile.attack[source_sample] / profile.attack_reference, 0.0, 1.0)
        )
        rise_score = float(np.clip(rms_rise[source_sample] / rise_reference, 0.0, 1.0))
        if activity_score < 0.08 or max(attack_score, rise_score) < 0.10:
            continue
        confidence = float(
            np.clip(
                0.18
                + 0.25 * activity_score
                + 0.22 * attack_score
                + 0.17 * rise_score
                + 0.13 * grid_confidence
                + 0.05 * float(np.clip(prominence, 0.0, 1.0)),
                0.0,
                0.72,
            )
        )
        stable_key = (
            f"vocal-energy-grid:{acoustic_sample}:{int(chunk.get('index', -1))}"
        )
        anchors.append(
            {
                "id": str(uuid.uuid5(uuid.NAMESPACE_URL, stable_key)),
                "chunk_index": int(chunk.get("index", -1)),
                "acoustic_sample": acoustic_sample,
                "chart_sample": chart_sample,
                "snap_error_ms": (acoustic_sample - chart_sample)
                * 1000.0
                / target_sample_rate,
                "confidence": confidence,
                "activity_score": activity_score,
                "attack_score": attack_score,
                "rise_score": rise_score,
                "pitch_score": 0.0,
                "transition_score": 0.0,
                "grid_confidence": grid_confidence,
                "fallback_level": "beat_aligned_vocal_energy_peak",
            }
        )
    return anchors


def _add_vocal_fallback_hits(
    track: TrackModel,
    anchors: list[dict[str, Any]],
) -> int:
    tolerance = max(
        1,
        round(track.original_sample_rate * _CROSS_STEM_MERGE_MS / 1000.0),
    )
    occupied = [(hit.sample, hit.snapped_sample) for hit in track.hit_points]
    created = 0
    for anchor in anchors:
        acoustic_sample = min(
            max(int(anchor["acoustic_sample"]), 0),
            max(0, track.sample_count - 1),
        )
        chart_sample = min(
            max(int(anchor["chart_sample"]), 0),
            max(0, track.sample_count - 1),
        )
        if any(
            abs(acoustic_sample - sample) <= tolerance or chart_sample == snapped
            for sample, snapped in occupied
        ):
            continue
        confidence = float(anchor["confidence"])
        activity_score = float(anchor["activity_score"])
        attack_score = float(anchor["attack_score"])
        if anchor.get("fallback_level") == "beat_aligned_vocal_energy_peak":
            detector_votes = [
                "beat_aligned_vocal_energy_peak",
                "vocal_energy_attack",
                "rhythm_grid_posterior",
            ]
        else:
            detector_votes = [
                "vocal_acoustic_onset",
                "vocal_envelope_change",
                "vocal_energy_attack",
                "rhythm_grid_posterior",
            ]
        if float(anchor.get("pitch_score", 0.0)) >= 0.20:
            detector_votes.append("vocal_pitch_attack")
        if float(anchor.get("transition_score", 0.0)) >= 0.20:
            detector_votes.append("vocal_phoneme_transition")
        track.hit_points.append(
            HitPointModel(
                id=str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"vocal-fallback-hit:{track.id}:{anchor['id']}",
                    )
                ),
                sample=acoustic_sample,
                acoustic_sample=acoustic_sample,
                chart_sample=chart_sample,
                detected_sample=acoustic_sample,
                refined_sample=acoustic_sample,
                snapped_sample=chart_sample,
                snap_error_ms=float(anchor["snap_error_ms"]),
                band="mid_hit",
                confidence=confidence,
                salience=float(
                    np.clip(
                        0.20 + 0.42 * confidence + 0.23 * activity_score + 0.15 * attack_score,
                        0.0,
                        1.0,
                    )
                ),
                source="stems",
                detector_votes_json=dumps(detector_votes),
                primary_stem="vocals",
                stem_evidence_json=dumps({"vocals": activity_score}),
                manually_edited=False,
                locked=False,
            )
        )
        occupied.append((acoustic_sample, chart_sample))
        created += 1
    return created


def _sync_vocal_candidate_events(
    track: TrackModel,
    anchors: list[dict[str, Any]],
    fallback_anchors: list[dict[str, Any]],
    coverage_chunks: list[dict[str, Any]],
) -> None:
    accepted_hit_ids = {hit.id for hit in track.hit_points}
    successful_intervals = [
        (int(chunk["startSample"]), int(chunk["endSample"]))
        for chunk in coverage_chunks
        if chunk.get("status") == "success"
    ]
    desired: dict[str, CandidateEventModel] = {}

    for anchor in anchors:
        acoustic_sample = int(anchor["refined_sample"])
        grid_sample = anchor.get("grid_sample")
        chart_sample = int(grid_sample) if grid_sample is not None else acoustic_sample
        snap_error_ms = (
            (acoustic_sample - chart_sample) * 1000.0 / track.original_sample_rate
        )
        grid_confidence = (
            float(np.exp(-0.5 * (abs(snap_error_ms) / 30.0) ** 2))
            if grid_sample is not None
            else 0.0
        )
        hit_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"vocal-hit:{track.id}:{anchor['id']}")
        )
        accepted = hit_id in accepted_hit_ids
        candidate_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"candidate:vocal:{track.id}:{anchor['id']}")
        )
        activity_score = float(anchor.get("activity_score", 0.0))
        desired[candidate_id] = CandidateEventModel(
            id=candidate_id,
            hit_point_id=hit_id if accepted else None,
            sample=acoustic_sample,
            acoustic_sample=acoustic_sample,
            chart_sample=chart_sample,
            snap_error_ms=snap_error_ms,
            lane="vocals",
            source_evidence_json=dumps(
                {"vocals": activity_score, "melody": 0.0, "drums": 0.0, "mix": 0.0}
            ),
            semantic_evidence_json=dumps(
                {
                    "lyricAlignment": float(anchor.get("chunk_match_confidence", 0.0)),
                    "phonemeConfidence": 0.0,
                    "pitchConfidence": float(anchor.get("pitch_score", 0.0)),
                    "beatConfidence": grid_confidence,
                    "attackConfidence": float(anchor.get("attack_score", 0.0)),
                    "acousticConfidence": float(
                        anchor.get("acoustic_confidence", 0.0)
                    ),
                    "phonemeTransitionConfidence": float(
                        anchor.get("transition_score", 0.0)
                    ),
                    "alignmentShiftMs": float(anchor.get("alignment_shift_ms", 0.0)),
                }
            ),
            confidence=float(anchor.get("confidence", 0.0)),
            status="accepted" if accepted else "uncertain",
            grid_type="straight_1_16" if grid_sample is not None else "unsnapped",
            grid_confidence=grid_confidence,
        )

    for anchor in fallback_anchors:
        acoustic_sample = int(anchor["acoustic_sample"])
        chart_sample = int(anchor["chart_sample"])
        hit_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"vocal-fallback-hit:{track.id}:{anchor['id']}",
            )
        )
        accepted = hit_id in accepted_hit_ids
        candidate_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"candidate:vocal-fallback:{track.id}:{anchor['id']}",
            )
        )
        snap_error_ms = float(anchor["snap_error_ms"])
        grid_confidence = float(np.exp(-0.5 * (abs(snap_error_ms) / 30.0) ** 2))
        desired[candidate_id] = CandidateEventModel(
            id=candidate_id,
            hit_point_id=hit_id if accepted else None,
            sample=acoustic_sample,
            acoustic_sample=acoustic_sample,
            chart_sample=chart_sample,
            snap_error_ms=snap_error_ms,
            lane="vocals",
            source_evidence_json=dumps(
                {
                    "vocals": float(anchor.get("activity_score", 0.0)),
                    "melody": 0.0,
                    "drums": 0.0,
                    "mix": 0.0,
                }
            ),
            semantic_evidence_json=dumps(
                {
                    "lyricAlignment": 0.0,
                    "phonemeConfidence": 0.0,
                    "pitchConfidence": float(anchor.get("pitch_score", 0.0)),
                    "beatConfidence": grid_confidence,
                    "attackConfidence": float(anchor.get("attack_score", 0.0)),
                    "energyRiseConfidence": float(anchor.get("rise_score", 0.0)),
                    "beatAlignedEnergyConfidence": float(
                        anchor.get("grid_confidence", 0.0)
                        if anchor.get("fallback_level")
                        == "beat_aligned_vocal_energy_peak"
                        else 0.0
                    ),
                    "phonemeTransitionConfidence": float(
                        anchor.get("transition_score", 0.0)
                    ),
                }
            ),
            confidence=float(anchor.get("confidence", 0.0)),
            status="accepted" if accepted else "uncertain",
            grid_type="straight_1_16",
            grid_confidence=grid_confidence,
        )

    existing_by_id = {candidate.id: candidate for candidate in track.candidate_events}
    for candidate in list(track.candidate_events):
        if (
            candidate.lane == "vocals"
            and candidate.id not in desired
            and any(start <= candidate.acoustic_sample < end for start, end in successful_intervals)
        ):
            track.candidate_events.remove(candidate)
    for candidate_id, replacement in desired.items():
        existing = existing_by_id.get(candidate_id)
        if existing is None:
            track.candidate_events.append(replacement)
            continue
        for field in (
            "hit_point_id",
            "sample",
            "acoustic_sample",
            "chart_sample",
            "snap_error_ms",
            "lane",
            "source_evidence_json",
            "semantic_evidence_json",
            "confidence",
            "status",
            "grid_type",
            "grid_confidence",
        ):
            setattr(existing, field, getattr(replacement, field))


def _focus_supports_vocals(track: TrackModel, sample: int) -> bool:
    return _vocal_routing_score(track, sample) >= 0.5


def _vocal_routing_score(track: TrackModel, sample: int) -> float:
    analysis = json.loads(track.analysis_json or "{}")
    segments = analysis.get("focusMap", analysis.get("focus_map", []))
    if not isinstance(segments, list) or not segments:
        return 0.5
    has_valid_segment = False
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        start = int(segment.get("startSample", segment.get("start_sample", 0)))
        end = int(segment.get("endSample", segment.get("end_sample", 0)))
        if end <= start:
            continue
        has_valid_segment = True
        if start <= sample < end:
            source = segment.get("focusSource", segment.get("focus_source"))
            confidence = float(
                segment.get("confidence", 1.0 if source == "vocals" else 0.0)
            )
            evidence = segment.get("evidence", {})
            vocal_evidence = (
                float(evidence.get("vocals", 0.0)) if isinstance(evidence, dict) else 0.0
            )
            alternatives = segment.get("alternatives", [])
            alternative_score = max(
                (
                    float(item.get("score", 0.0))
                    for item in alternatives
                    if isinstance(item, dict) and item.get("source") == "vocals"
                ),
                default=0.0,
            )
            return float(
                np.clip(
                    max(
                        vocal_evidence,
                        alternative_score,
                        confidence if source == "vocals" else 0.0,
                    ),
                    0.0,
                    1.0,
                )
            )
    return 0.0 if has_valid_segment else 0.5


def _replace_vocal_hits_with_stats(
    track: TrackModel,
    anchors: list[dict[str, Any]],
    coverage_chunks: list[dict[str, Any]] | None = None,
) -> VocalHitReplacementResult:
    """Replace only automatic vocal hits inside proven alignment coverage."""

    automatic_vocal_hits = [
        hit
        for hit in track.hit_points
        if hit.primary_stem == "vocals" and not hit.manually_edited and not hit.locked
    ]
    successful_chunks = {
        int(chunk.get("index", index)): chunk
        for index, chunk in enumerate(coverage_chunks or [])
        if chunk.get("status") == "success"
        and int(chunk.get("anchorCount", 0)) >= _MIN_CHUNK_REPLACEMENT_ANCHORS
    }
    successful_intervals = {
        index: (
            max(0, int(chunk.get("startSample", 0))),
            min(track.sample_count, int(chunk.get("endSample", track.sample_count))),
        )
        for index, chunk in successful_chunks.items()
    }
    initially_removable = [
        hit
        for hit in automatic_vocal_hits
        if any(start <= hit.sample < end for start, end in successful_intervals.values())
    ]
    initially_removable_ids = {hit.id for hit in initially_removable}
    protected_positions = sorted(
        (hit.sample, hit.snapped_sample)
        for hit in track.hit_points
        if hit.id not in initially_removable_ids
    )
    tolerance = max(
        1,
        round(track.original_sample_rate * _CROSS_STEM_MERGE_MS / 1000.0),
    )
    pending: list[HitPointModel] = []
    pending_chunk_indexes: set[int] = set()
    for anchor in anchors:
        chunk_index = int(anchor.get("chunk_index", -1))
        if coverage_chunks is not None:
            interval = successful_intervals.get(chunk_index)
            if interval is None:
                continue
        else:
            interval = None
        grid_sample = anchor.get("grid_sample")
        if (
            grid_sample is None
            or not anchor.get("active", False)
            or not anchor.get("chart_candidate", False)
        ):
            continue
        snapped = min(max(int(grid_sample), 0), max(0, track.sample_count - 1))
        refined = min(max(int(anchor["refined_sample"]), 0), max(0, track.sample_count - 1))
        if interval is not None and not (interval[0] <= refined < interval[1]):
            continue
        routing_score = _vocal_routing_score(track, refined)
        if any(
            abs(refined - protected_sample) <= tolerance
            or snapped == protected_grid
            for protected_sample, protected_grid in protected_positions
        ):
            continue
        aligned = min(max(int(anchor["aligned_sample"]), 0), max(0, track.sample_count - 1))
        confidence = float(anchor["confidence"])
        activity_score = float(anchor.get("activity_score", 0.0))
        attack_score = float(anchor.get("attack_score", 0.0))
        salience = float(
            np.clip(
                0.14
                + 0.39 * confidence
                + 0.23 * activity_score
                + 0.14 * attack_score
                + 0.10 * routing_score,
                0.0,
                1.0,
            )
        )
        detector_votes = [
            "qwen_forced_alignment",
            "qwen_phrase_boundary",
            "vocal_acoustic_fusion",
            "absolute_vocal_activity",
            "voiced_attack",
            "rhythm_1_16_candidate",
        ]
        if float(anchor.get("pitch_score", 0.0)) >= 0.20:
            detector_votes.append("vocal_pitch_attack")
        if float(anchor.get("transition_score", 0.0)) >= 0.20:
            detector_votes.append("vocal_phoneme_transition")
        detector_votes.append(f"soft_vocal_route:{routing_score:.3f}")
        pending.append(
            HitPointModel(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"vocal-hit:{track.id}:{anchor['id']}")),
                sample=refined,
                acoustic_sample=refined,
                chart_sample=snapped,
                detected_sample=aligned,
                refined_sample=refined,
                snapped_sample=snapped,
                snap_error_ms=(refined - snapped) * 1000.0 / track.original_sample_rate,
                band="mid_hit",
                confidence=confidence,
                salience=salience,
                source="stems",
                detector_votes_json=dumps(detector_votes),
                primary_stem="vocals",
                stem_evidence_json=dumps({"vocals": activity_score}),
                manually_edited=False,
                locked=False,
            )
        )
        protected_positions.append((refined, snapped))
        if interval is not None:
            pending_chunk_indexes.add(chunk_index)
    if not pending:
        return VocalHitReplacementResult(
            created_hit_count=0,
            removed_hit_count=0,
            replaced_chunk_count=0,
            preserved_fallback_hit_count=len(automatic_vocal_hits),
        )
    removable = [
        hit
        for hit in initially_removable
        if any(
            index in pending_chunk_indexes and start <= hit.sample < end
            for index, (start, end) in successful_intervals.items()
        )
    ]
    for hit in removable:
        track.hit_points.remove(hit)
    track.hit_points.extend(pending)
    return VocalHitReplacementResult(
        created_hit_count=len(pending),
        removed_hit_count=len(removable),
        replaced_chunk_count=len(pending_chunk_indexes),
        preserved_fallback_hit_count=len(automatic_vocal_hits) - len(removable),
    )


def _replace_vocal_hits(
    track: TrackModel,
    anchors: list[dict[str, Any]],
    coverage_chunks: list[dict[str, Any]] | None = None,
) -> int:
    return _replace_vocal_hits_with_stats(track, anchors, coverage_chunks).created_hit_count


def _run_job(job_id: str) -> None:
    stage_started = time.perf_counter()
    stage_timings: dict[str, float] = {}
    current_stage = "queued"

    def progress(stage: str, amount: float) -> None:
        nonlocal current_stage, stage_started
        now = time.perf_counter()
        if current_stage != stage:
            stage_timings[current_stage] = round((now - stage_started) * 1000.0, 3)
            current_stage = stage
            stage_started = now
        _update_job(
            job_id,
            status="processing",
            stage=stage,
            progress=amount,
            stage_timings_json=dumps(stage_timings),
        )

    warnings: list[str] = []
    try:
        with SessionLocal() as session:
            job = session.get(VocalAlignmentJobModel, job_id)
            if job is None:
                return
            track = session.scalar(
                select(TrackModel)
                .where(TrackModel.id == job.track_id)
                .options(
                    selectinload(TrackModel.tempo_segments),
                    selectinload(TrackModel.hit_points),
                    selectinload(TrackModel.project),
                )
            )
            if track is None:
                raise VocalJobError("TRACK_NOT_FOUND", "歌曲工程已被删除。")
            operation = job.operation
            replace_hits = job.replace_vocal_hits
            lyrics_text = track.lyrics_text
            lyrics_format = track.lyrics_format
            track_id = track.id

        progress("separating_vocals", 0.08)
        vocals_path, vocals, sample_rate = _load_vocals(track_id)
        progress("detecting_vocal_activity", 0.16)
        if float(np.sqrt(np.mean(np.square(vocals, dtype=np.float64)))) < 1e-5:
            raise VocalJobError("VOCALS_STEM_SILENT", "人声分轨几乎为空，无法进行歌词对齐。")

        if operation == "asr_draft":
            progress("transcribing", 0.28)
            result = _run_qwen("transcribe", audio_path=vocals_path)
            transcript = clean_lyrics(str(result.get("text", "")), "japanese")
            if not transcript:
                raise VocalJobError("EMPTY_ASR_TRANSCRIPT", "本地 ASR 没有识别出可用日语歌词。")
            warnings.extend(str(item) for item in result.get("warnings", []))
            progress("normalizing_pronunciation", 0.88)
            now = datetime.now(UTC)
            with SessionLocal() as session:
                track = session.get(TrackModel, track_id)
                if track is None:
                    raise VocalJobError("TRACK_NOT_FOUND", "歌曲工程已被删除。")
                track.lyrics_text = transcript
                track.lyrics_format = "japanese"
                track.vocal_alignment_json = dumps(
                    {
                        "status": "draft",
                        "stage": "completed",
                        "progress": 1.0,
                        "anchors": [],
                        "model": result.get("model"),
                        "device": result.get("device"),
                        "warnings": warnings,
                        "updated_at": now.isoformat(),
                    }
                )
                track.updated_at = now
                session.commit()
        else:
            lyrics = clean_lyrics(lyrics_text, lyrics_format)
            if not lyrics:
                raise VocalJobError(
                    "LYRICS_REQUIRED",
                    "请先保存日文歌词，或先生成本地 ASR 草稿。",
                )
            if lyrics_format == "romaji":
                raise VocalJobError(
                    "ROMAJI_REQUIRES_KANA",
                    "罗马音需要先转换并确认假名读音；当前请使用日文原文或假名对齐。",
                )
            progress("normalizing_pronunciation", 0.22)
            progress("aligning_lyrics", 0.30)
            alignment_failure: dict[str, Any] | None = None
            try:
                result = _run_qwen("align_song", audio_path=vocals_path, text=lyrics)
            except VocalJobError as error:
                alignment_failure = {
                    "code": error.code,
                    "message": error.message,
                    "details": error.details,
                }
                result = {
                    "status": "failed",
                    "timestamps": [],
                    "chunks": [],
                    "warnings": [],
                    "alignment_strategy": "acoustic_fallback_after_alignment_failure",
                }
                warnings.append(
                    f"歌词对齐失败（{error.code}）；未隐藏失败，已改用本地人声声学 fallback。"
                )
            timestamps = list(result.get("timestamps", []))
            warnings.extend(str(item) for item in result.get("warnings", []))
            warnings.append("锚点置信度是时长与局部浊音攻击的启发式分数，并非 Qwen 校准概率。")
            progress("refining_samples", 0.78)
            try:
                vocal_acoustic_result = extract_vocal_acoustic_candidates(
                    vocals,
                    sample_rate,
                )
                vocal_acoustic_candidates = vocal_acoustic_result.candidates
            except Exception as error:
                vocal_acoustic_result = None
                vocal_acoustic_candidates = []
                warnings.append(f"本地 vocal acoustic detector 失败：{error}")
            with SessionLocal() as session:
                track = session.scalar(
                    select(TrackModel)
                    .where(TrackModel.id == track_id)
                    .options(
                        selectinload(TrackModel.tempo_segments),
                        selectinload(TrackModel.hit_points),
                        selectinload(TrackModel.candidate_events),
                        selectinload(TrackModel.project),
                    )
                )
                if track is None:
                    raise VocalJobError("TRACK_NOT_FOUND", "歌曲工程已被删除。")
                if not track.tempo_segments:
                    raise VocalJobError(
                        "TEMPO_REQUIRED",
                        "歌曲还没有可用 BPM 与 offset，请先完成基础分析。",
                    )
                tempo = min(track.tempo_segments, key=lambda item: item.start_sample)
                anchor_build = _build_anchors(
                    timestamps,
                    audio=vocals,
                    sample_rate=sample_rate,
                    original_sample_rate=track.original_sample_rate,
                    bpm=float(tempo.bpm),
                    beat_offset_sample=int(tempo.beat_offset_sample),
                    acoustic_candidates=vocal_acoustic_candidates,
                )
                anchors = anchor_build.anchors
                coverage_chunks = _build_chunk_coverage(
                    list(result.get("chunks", [])),
                    timestamps,
                    anchors,
                    source_sample_rate=sample_rate,
                    target_sample_rate=track.original_sample_rate,
                )
                if not coverage_chunks:
                    coverage_chunks = _failed_chunk_coverage(
                        track.sample_count,
                        track.original_sample_rate,
                    )
                reliable_grid_count = sum(
                    bool(anchor.get("chart_candidate")) for anchor in anchors
                )
                rejected_short = int(
                    anchor_build.statistics.get("rejectedShortTimestampCount", 0)
                )
                rejected_silent = int(
                    anchor_build.statistics.get("rejectedSilentCount", 0)
                )
                warnings.append(
                    "歌词先由演唱 ASR 分段定位，再以短片段强制对齐；"
                    "Qwen 只提供短语语义区域，最终时间来自区域内声学起音；"
                    f"已拒绝 {rejected_short} 个塌缩时间戳和 {rejected_silent} 个静音候选。"
                )
                if reliable_grid_count == 0:
                    warnings.append(
                        "本次没有可靠歌词网格锚点；旧 vocal detector 点保持不变，"
                        "并对未覆盖段运行声学 fallback。"
                    )
                progress("saving_results", 0.94)
                replacement = (
                    _replace_vocal_hits_with_stats(track, anchors, coverage_chunks)
                    if replace_hits
                    else VocalHitReplacementResult(0, 0, 0, 0)
                )
                fallback_anchors = _detect_vocal_fallback_anchors(
                    vocals,
                    sample_rate=sample_rate,
                    target_sample_rate=track.original_sample_rate,
                    bpm=float(tempo.bpm),
                    beat_offset_sample=int(tempo.beat_offset_sample),
                    coverage_chunks=coverage_chunks,
                    acoustic_candidates=vocal_acoustic_candidates,
                )
                fallback_created_hits = (
                    _add_vocal_fallback_hits(track, fallback_anchors) if replace_hits else 0
                )
                session.flush()
                _sync_vocal_candidate_events(
                    track,
                    anchors,
                    fallback_anchors,
                    coverage_chunks,
                )
                created_hits = replacement.created_hit_count + fallback_created_hits
                automatic_vocal_hit_count = sum(
                    hit.primary_stem == "vocals"
                    and not hit.manually_edited
                    and not hit.locked
                    for hit in track.hit_points
                )
                if replace_hits and created_hits == 0 and automatic_vocal_hit_count == 0:
                    raise VocalJobError(
                        "NO_VOCAL_EVIDENCE",
                        "歌词对齐与本地声学 fallback 均未发现可靠人声事件；未生成空 BPM 假点。",
                        anchor_build.statistics,
                    )
                now = datetime.now(UTC)
                track.vocal_alignment_json = dumps(
                    {
                        "status": "completed",
                        "stage": "completed",
                        "progress": 1.0,
                        "anchors": anchors,
                        "model": result.get("model"),
                        "device": result.get("device"),
                        "alignment_strategy": result.get(
                            "alignment_strategy",
                            "singing_asr_guided_chunks",
                        ),
                        "chunk_diagnostics": result.get("chunks", []),
                        "coverage_chunks": coverage_chunks,
                        "fallback_anchors": fallback_anchors,
                        "alignment_failure": alignment_failure,
                        "vocal_acoustic_analysis": {
                            "method": (
                                vocal_acoustic_result.method
                                if vocal_acoustic_result is not None
                                else "failed"
                            ),
                            "candidateCount": len(vocal_acoustic_candidates),
                        },
                        "bpm": tempo.bpm,
                        "beat_offset_sample": tempo.beat_offset_sample,
                        "subdivisions_per_beat": 4,
                        "created_hit_count": created_hits,
                        "alignment_created_hit_count": replacement.created_hit_count,
                        "fallback_created_hit_count": fallback_created_hits,
                        "removed_hit_count": replacement.removed_hit_count,
                        "replaced_chunk_count": replacement.replaced_chunk_count,
                        "preserved_fallback_hit_count": replacement.preserved_fallback_hit_count,
                        "alignment_quality": anchor_build.statistics,
                        "warnings": warnings,
                        "updated_at": now.isoformat(),
                    }
                )
                analysis = json.loads(track.analysis_json or "{}")
                analysis["vocalLyrics"] = {
                    "status": "completed",
                    "anchorCount": len(anchors),
                    "gridAnchorCount": sum(anchor["grid_sample"] is not None for anchor in anchors),
                    "createdHitCount": created_hits,
                    "alignmentCreatedHitCount": replacement.created_hit_count,
                    "fallbackCreatedHitCount": fallback_created_hits,
                    "removedHitCount": replacement.removed_hit_count,
                    "replacedChunkCount": replacement.replaced_chunk_count,
                    "preservedFallbackHitCount": replacement.preserved_fallback_hit_count,
                    "coverageChunks": coverage_chunks,
                    "alignmentFailure": alignment_failure,
                    "vocalAcousticAnalysis": {
                        "method": (
                            vocal_acoustic_result.method
                            if vocal_acoustic_result is not None
                            else "failed"
                        ),
                        "candidateCount": len(vocal_acoustic_candidates),
                    },
                    "alignmentQuality": anchor_build.statistics,
                    "model": result.get("model"),
                    "device": result.get("device"),
                    "strategy": result.get(
                        "alignment_strategy",
                        "singing_asr_guided_chunks",
                    ),
                }
                analysis["hitPointCount"] = len(track.hit_points)
                track.analysis_json = dumps(analysis)
                track.project.status = "edited"
                track.project.updated_at = now
                track.updated_at = now
                session.commit()

        stage_timings[current_stage] = round((time.perf_counter() - stage_started) * 1000.0, 3)
        _update_job(
            job_id,
            status="completed",
            stage="completed",
            progress=1.0,
            stage_timings_json=dumps(stage_timings),
            warnings_json=dumps(warnings),
            error_json=None,
        )
    except Exception as error:
        payload = {
            "code": getattr(error, "code", "VOCAL_ALIGNMENT_FAILED"),
            "message": getattr(error, "message", str(error)) or "人声歌词对齐失败。",
            "details": getattr(error, "details", None),
        }
        _update_job(
            job_id,
            status="failed",
            stage=current_stage if current_stage != "queued" else "queued",
            error_json=dumps(payload),
            stage_timings_json=dumps(stage_timings),
        )
    finally:
        with _future_lock:
            _futures.pop(job_id, None)


def submit_vocal_job(job_id: str) -> None:
    with _future_lock:
        existing = _futures.get(job_id)
        if existing is not None and not existing.done():
            return
        _futures[job_id] = _executor.submit(_run_job, job_id)


def wait_for_vocal_job(job_id: str, timeout: float = 3_600.0) -> None:
    with _future_lock:
        future = _futures.get(job_id)
    if future is not None:
        future.result(timeout=timeout)
