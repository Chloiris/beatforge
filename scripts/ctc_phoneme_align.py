#!/usr/bin/env python3
"""Offline Japanese phone alignment using HuBERT emissions and CTC Viterbi.

This process deliberately has no timestamp fallback.  Every returned span is
derived from frames visited by the best valid CTC path through model emissions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

MODEL_ID = "prj-beatrice/japanese-hubert-base-phoneme-ctc-v4"
MODEL_REVISION = "f5fe07043bcb0b77a86faf72ac6d8fc1ae558f99"
MODEL_SAMPLE_RATE = 16_000
IGNORED_G2P_PHONES = frozenset({"pau", "sil"})


class AlignmentScriptError(RuntimeError):
    def __init__(self, code: str, message: str, **details: Any) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details


@dataclass(frozen=True, slots=True)
class PhoneTarget:
    surface: str
    phoneme: str
    frontend_index: int
    plan_phone_index: int | None = None
    mora_index: int | None = None
    character_indices: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class PhonePath:
    target_index: int
    first_frame: int
    last_frame: int
    confidence: float


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _recover_surface(
    original: str,
    node_surface: str,
    cursor: int,
) -> tuple[str, int, bool]:
    """Recover the saved lyric spelling after OpenJTalk's full-width rewrite."""

    candidate = unicodedata.normalize("NFKC", node_surface)
    if not candidate:
        return "", cursor, True
    if candidate.isspace():
        end = cursor
        while end < len(original) and original[end].isspace():
            end += 1
        return "", end, True

    # Search only forward and compare normalized substrings.  This preserves
    # ASCII case/digits from the actual saved lyrics while following the exact
    # OpenJTalk token order.
    search_limit = min(len(original), cursor + max(64, len(candidate) * 4))
    for start in range(cursor, search_limit):
        if original[start].isspace() and not candidate.isspace():
            continue
        end_limit = min(len(original), start + max(16, len(candidate) * 3))
        for end in range(start + 1, end_limit + 1):
            normalized = unicodedata.normalize("NFKC", original[start:end])
            if normalized == candidate:
                return original[start:end], end, True
            if len(normalized) > len(candidate) + 2:
                break

    # MeCab/OpenJTalk rewrites Arabic numerals as Kanji.  A single source digit
    # still is the honest display surface for every phone belonging to it.
    if candidate in "〇一二三四五六七八九十百千万億兆":
        for index in range(cursor, min(len(original), cursor + 16)):
            if original[index].isdigit():
                return original[index], index + 1, True
    return candidate, cursor, False


def build_phone_targets(text: str) -> tuple[list[PhoneTarget], dict[str, Any]]:
    try:
        import pyopenjtalk
    except ImportError as error:
        raise AlignmentScriptError(
            "CTC_G2P_DEPENDENCY_MISSING",
            "pyopenjtalk is required to generate Japanese phone targets.",
        ) from error

    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    try:
        nodes = pyopenjtalk.run_frontend(normalized)
    except Exception as error:  # pyopenjtalk raises native extension exceptions
        raise AlignmentScriptError(
            "CTC_G2P_FAILED",
            "OpenJTalk could not analyze the saved lyrics.",
            exceptionType=type(error).__name__,
        ) from error

    targets: list[PhoneTarget] = []
    cursor = 0
    recovery_fallbacks: list[dict[str, Any]] = []
    skipped_nodes: list[str] = []
    for node_index, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        node_surface = str(node.get("string") or "")
        surface, cursor, recovered = _recover_surface(normalized, node_surface, cursor)
        try:
            mora_size = int(node.get("mora_size") or 0)
        except (TypeError, ValueError):
            mora_size = 0
        reading = str(node.get("read") or "").strip()
        if mora_size <= 0 or not surface.strip() or not reading:
            if node_surface.strip() and mora_size > 0:
                skipped_nodes.append(node_surface)
            continue
        if not recovered:
            recovery_fallbacks.append(
                {"frontendIndex": node_index, "frontendSurface": node_surface, "surface": surface}
            )
        try:
            phones = [
                phone
                for phone in pyopenjtalk.g2p(reading, kana=False).split()
                if phone not in IGNORED_G2P_PHONES
            ]
        except Exception as error:
            raise AlignmentScriptError(
                "CTC_G2P_FAILED",
                "OpenJTalk failed while converting a lyric surface to phones.",
                frontendIndex=node_index,
                surface=surface,
                exceptionType=type(error).__name__,
            ) from error
        if not phones:
            skipped_nodes.append(surface)
            continue
        for phone in phones:
            targets.append(
                PhoneTarget(
                    surface=surface,
                    phoneme=phone,
                    frontend_index=node_index,
                )
            )

    if not targets:
        raise AlignmentScriptError(
            "CTC_G2P_EMPTY",
            "OpenJTalk produced no lexical phone targets from the saved lyrics.",
        )
    surface_sequence: list[str] = []
    previous_frontend_index: int | None = None
    for target in targets:
        if target.frontend_index != previous_frontend_index:
            surface_sequence.append(target.surface)
            previous_frontend_index = target.frontend_index
    return targets, {
        "frontendNodeCount": len(nodes),
        "phoneTargetCount": len(targets),
        "surfaceSequence": surface_sequence,
        "surfaceRecoveryFallbacks": recovery_fallbacks,
        "skippedNodes": skipped_nodes,
        "pausePolicy": "pau/sil omitted from the target; CTC blank absorbs non-lexical gaps",
        "g2p": "pyopenjtalk frontend reading -> OpenJTalk phones",
    }


def _contains_forbidden_timing(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).casefold()
            if "sample" in normalized or "timestamp" in normalized:
                return True
            if _contains_forbidden_timing(item):
                return True
    elif isinstance(value, list):
        return any(_contains_forbidden_timing(item) for item in value)
    return False


def load_phone_targets_from_plan(
    path: Path,
    lyrics: str,
) -> tuple[list[PhoneTarget], dict[str, Any]]:
    """Load a DP lyric plan that contains relationships, never timestamps."""

    try:
        decoded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AlignmentScriptError(
            "CTC_G2P_PLAN_INVALID",
            "The Japanese lyric hierarchy plan is missing or invalid.",
            planPath=str(path),
        ) from error
    if not isinstance(decoded, dict) or decoded.get("status") != "ok":
        raise AlignmentScriptError(
            "CTC_G2P_PLAN_INVALID",
            "The Japanese lyric hierarchy process did not return a usable plan.",
            planPath=str(path),
        )
    plan_payload = {key: value for key, value in decoded.items() if key != "status"}
    if _contains_forbidden_timing(plan_payload):
        raise AlignmentScriptError(
            "CTC_G2P_PLAN_HAS_TIMESTAMPS",
            "The lyric plan must not contain samples or timestamps.",
        )
    normalized_lyrics = unicodedata.normalize("NFC", lyrics).replace("\r\n", "\n").replace(
        "\r", "\n"
    )
    if plan_payload.get("sourceText") != normalized_lyrics:
        raise AlignmentScriptError(
            "CTC_G2P_PLAN_MISMATCH",
            "The lyric hierarchy plan was built from different saved lyrics.",
        )
    ctc_plan = plan_payload.get("ctcPlan")
    phonemes = plan_payload.get("phonemes")
    moras = plan_payload.get("moras")
    characters = plan_payload.get("characters")
    if not all(isinstance(value, list) for value in (phonemes, moras, characters)):
        raise AlignmentScriptError(
            "CTC_G2P_PLAN_INVALID",
            "The lyric plan hierarchy is incomplete.",
        )
    if not isinstance(ctc_plan, dict) or not isinstance(ctc_plan.get("targets"), list):
        raise AlignmentScriptError(
            "CTC_G2P_PLAN_INVALID",
            "The lyric plan has no CTC target sequence.",
        )
    targets_payload = ctc_plan["targets"]
    if (
        ctc_plan.get("version") != 1
        or ctc_plan.get("spokenText") != plan_payload.get("spokenText")
        or not phonemes
        or not moras
        or not characters
    ):
        raise AlignmentScriptError(
            "CTC_G2P_PLAN_INVALID",
            "The lyric plan version or spoken-text projection is inconsistent.",
        )
    hash_payload = {
        "version": ctc_plan.get("version"),
        "spokenText": ctc_plan.get("spokenText"),
        "targets": targets_payload,
    }
    encoded = json.dumps(
        hash_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    calculated_hash = hashlib.sha256(encoded).hexdigest()
    if ctc_plan.get("planHash") != calculated_hash:
        raise AlignmentScriptError(
            "CTC_G2P_PLAN_HASH_MISMATCH",
            "The lyric plan hash does not match its CTC targets.",
        )

    targets: list[PhoneTarget] = []
    for index, item in enumerate(targets_payload):
        if not isinstance(item, dict):
            raise AlignmentScriptError(
                "CTC_G2P_PLAN_INVALID",
                "A CTC lyric target is not an object.",
                targetIndex=index,
            )
        try:
            target_index = int(item["targetIndex"])
            frontend_index = int(item["frontendIndex"])
            mora_index = int(item["moraIndex"])
            character_indices = tuple(int(value) for value in item["characterIndices"])
        except (KeyError, TypeError, ValueError) as error:
            raise AlignmentScriptError(
                "CTC_G2P_PLAN_INVALID",
                "A CTC lyric target has invalid hierarchy indices.",
                targetIndex=index,
            ) from error
        surface = str(item.get("surface") or "").strip()
        phoneme = str(item.get("phoneme") or "").strip()
        hierarchy_phone = phonemes[index] if index < len(phonemes) else None
        hierarchy_matches = (
            isinstance(hierarchy_phone, dict)
            and hierarchy_phone.get("index") == target_index
            and hierarchy_phone.get("text") == surface
            and hierarchy_phone.get("phoneme") == phoneme
            and hierarchy_phone.get("frontendIndex") == frontend_index
            and hierarchy_phone.get("moraIndex") == mora_index
            and hierarchy_phone.get("characterIndices") == list(character_indices)
        )
        if (
            target_index != index
            or frontend_index < 0
            or mora_index < 0
            or mora_index >= len(moras)
            or not character_indices
            or any(value < 0 for value in character_indices)
            or any(value >= len(characters) for value in character_indices)
            or list(character_indices) != sorted(set(character_indices))
            or not surface
            or not phoneme
            or not hierarchy_matches
        ):
            raise AlignmentScriptError(
                "CTC_G2P_PLAN_INVALID",
                "A CTC lyric target failed ordering or content validation.",
                targetIndex=index,
            )
        targets.append(
            PhoneTarget(
                surface=surface,
                phoneme=phoneme,
                frontend_index=frontend_index,
                plan_phone_index=target_index,
                mora_index=mora_index,
                character_indices=character_indices,
            )
        )
    if not targets or len(targets) != len(phonemes):
        raise AlignmentScriptError(
            "CTC_G2P_PLAN_INVALID",
            "The lyric plan phone hierarchy and CTC targets differ in size.",
        )
    surface_sequence: list[str] = []
    previous_frontend_index: int | None = None
    for target in targets:
        if target.frontend_index != previous_frontend_index:
            surface_sequence.append(target.surface)
            previous_frontend_index = target.frontend_index
    return targets, {
        "frontendNodeCount": len({target.frontend_index for target in targets}),
        "phoneTargetCount": len(targets),
        "surfaceSequence": surface_sequence,
        "surfaceRecoveryFallbacks": [],
        "skippedNodes": [],
        "pausePolicy": "pau/sil omitted; CTC blank absorbs non-lexical gaps",
        "g2p": str(plan_payload.get("g2pEngine") or "unknown"),
        "lyricPlanHash": calculated_hash,
        "lyricPlan": plan_payload,
        "hierarchyMapping": "character/mora/phoneme relationships from dynamic programming",
    }


def _load_audio(path: Path) -> tuple[np.ndarray, int, int]:
    try:
        import soundfile as sf
    except ImportError as error:
        raise AlignmentScriptError(
            "CTC_AUDIO_DEPENDENCY_MISSING",
            "soundfile is required to read the local vocals stem.",
        ) from error
    try:
        audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    except (OSError, RuntimeError) as error:
        raise AlignmentScriptError(
            "CTC_AUDIO_READ_FAILED",
            "The local vocals stem could not be decoded.",
            exceptionType=type(error).__name__,
        ) from error
    if audio.size == 0 or audio.shape[0] == 0 or source_rate <= 0:
        raise AlignmentScriptError("CTC_AUDIO_EMPTY", "The local vocals stem is empty.")
    if not np.all(np.isfinite(audio)):
        raise AlignmentScriptError(
            "CTC_AUDIO_NONFINITE",
            "The local vocals stem contains non-finite samples.",
        )
    mono = np.mean(audio, axis=1, dtype=np.float32)
    source_count = int(mono.size)
    peak = float(np.max(np.abs(mono)))
    rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float64))))
    if peak <= 1e-8 or rms <= 1e-10:
        raise AlignmentScriptError("CTC_AUDIO_SILENT", "The local vocals stem is silent.")
    if source_rate == MODEL_SAMPLE_RATE:
        return np.ascontiguousarray(mono, dtype=np.float32), int(source_rate), source_count

    try:
        from scipy.signal import resample_poly
    except ImportError as error:
        raise AlignmentScriptError(
            "CTC_AUDIO_DEPENDENCY_MISSING",
            "scipy is required to resample the vocals stem to 16 kHz.",
        ) from error
    divisor = math.gcd(int(source_rate), MODEL_SAMPLE_RATE)
    resampled = resample_poly(
        mono,
        MODEL_SAMPLE_RATE // divisor,
        int(source_rate) // divisor,
    )
    resampled = np.ascontiguousarray(resampled, dtype=np.float32)
    if resampled.size == 0 or not np.all(np.isfinite(resampled)):
        raise AlignmentScriptError(
            "CTC_AUDIO_RESAMPLE_FAILED",
            "Resampling the vocals stem produced invalid audio.",
        )
    return resampled, int(source_rate), source_count


def _load_vocabulary(model_dir: Path) -> tuple[dict[str, int], int]:
    path = model_dir / "vocab.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AlignmentScriptError(
            "CTC_VOCABULARY_INVALID",
            "The local checkpoint vocabulary is missing or invalid.",
            path=str(path),
        ) from error
    if not isinstance(payload, dict):
        raise AlignmentScriptError(
            "CTC_VOCABULARY_INVALID",
            "The local checkpoint vocabulary must be a JSON object.",
        )
    vocabulary: dict[str, int] = {}
    for phone, raw_id in payload.items():
        try:
            token_id = int(raw_id)
        except (TypeError, ValueError) as error:
            raise AlignmentScriptError(
                "CTC_VOCABULARY_INVALID",
                "The local checkpoint vocabulary contains a non-integer id.",
                token=phone,
            ) from error
        vocabulary[str(phone)] = token_id
    blank_id = vocabulary.get("PAD")
    if blank_id is None:
        raise AlignmentScriptError(
            "CTC_VOCABULARY_INVALID",
            "The local checkpoint vocabulary has no PAD/CTC blank token.",
        )
    return vocabulary, blank_id


def _select_device(requested: str) -> tuple[str, dict[str, Any]]:
    try:
        import torch
    except ImportError as error:
        raise AlignmentScriptError(
            "CTC_RUNTIME_DEPENDENCY_MISSING",
            "torch is required for local HuBERT inference.",
        ) from error
    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    if requested == "auto":
        selected = "mps" if mps_available else "cpu"
    elif requested in {"cpu", "mps"}:
        selected = requested
    else:
        raise AlignmentScriptError(
            "CTC_DEVICE_INVALID",
            "CTC device must be auto, cpu, or mps.",
            requestedDevice=requested,
        )
    if selected == "mps" and not mps_available:
        raise AlignmentScriptError(
            "CTC_MPS_UNAVAILABLE",
            "Apple Metal (MPS) was requested but is unavailable.",
        )
    return selected, {
        "requestedDevice": requested,
        "mpsAvailable": mps_available,
        "torchVersion": torch.__version__,
    }


def emit_log_probs(
    audio: np.ndarray,
    model_dir: Path,
    *,
    device: str,
    chunk_seconds: float,
    overlap_seconds: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        import torch
        from transformers import AutoModelForCTC
    except ImportError as error:
        raise AlignmentScriptError(
            "CTC_RUNTIME_DEPENDENCY_MISSING",
            "torch and transformers are required for local HuBERT inference.",
            missingModule=getattr(error, "name", None),
        ) from error

    chunk_samples = int(round(chunk_seconds * MODEL_SAMPLE_RATE))
    overlap_samples = int(round(overlap_seconds * MODEL_SAMPLE_RATE))
    if chunk_samples < MODEL_SAMPLE_RATE:
        raise AlignmentScriptError(
            "CTC_CHUNK_CONFIGURATION_INVALID",
            "CTC chunks must be at least one second long.",
        )
    if overlap_samples < 0 or overlap_samples * 2 >= chunk_samples:
        raise AlignmentScriptError(
            "CTC_CHUNK_CONFIGURATION_INVALID",
            "CTC overlap must be non-negative and less than half the chunk length.",
        )
    step = chunk_samples - overlap_samples
    starts = list(range(0, int(audio.size), step))
    if starts and starts[-1] + overlap_samples >= audio.size and len(starts) > 1:
        starts.pop()
    if not starts:
        starts = [0]

    try:
        model = AutoModelForCTC.from_pretrained(
            model_dir,
            local_files_only=True,
            torch_dtype=torch.float32,
        )
        model.eval()
        model.to(device)
    except Exception as error:
        raise AlignmentScriptError(
            "CTC_MODEL_LOAD_FAILED",
            "The pinned local HuBERT CTC checkpoint could not be loaded.",
            modelPath=str(model_dir),
            exceptionType=type(error).__name__,
            exceptionMessage=str(error)[:1_000],
        ) from error

    emissions: list[np.ndarray] = []
    centers: list[np.ndarray] = []
    chunk_records: list[dict[str, Any]] = []
    started = time.monotonic()
    with torch.inference_mode():
        for chunk_index, start in enumerate(starts):
            end = min(int(audio.size), start + chunk_samples)
            chunk = audio[start:end]
            if chunk.size < 400:
                continue
            try:
                values = torch.from_numpy(np.ascontiguousarray(chunk)).unsqueeze(0).to(device)
                logits = model(input_values=values).logits[0]
                log_probs = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()
            except Exception as error:
                raise AlignmentScriptError(
                    "CTC_EMISSION_FAILED",
                    "HuBERT failed while emitting phone probabilities for an audio chunk.",
                    chunkIndex=chunk_index,
                    chunkStartSample=start,
                    chunkEndSample=end,
                    device=device,
                    exceptionType=type(error).__name__,
                    exceptionMessage=str(error)[:1_000],
                ) from error
            if log_probs.ndim != 2 or log_probs.shape[0] == 0 or not np.all(np.isfinite(log_probs)):
                raise AlignmentScriptError(
                    "CTC_EMISSION_INVALID",
                    "HuBERT returned empty or non-finite phone probabilities.",
                    chunkIndex=chunk_index,
                    shape=list(log_probs.shape),
                )

            local_centers = start + (np.arange(log_probs.shape[0], dtype=np.float64) + 0.5) * (
                chunk.size / log_probs.shape[0]
            )
            keep_start = 0.0 if chunk_index == 0 else start + overlap_samples / 2.0
            keep_end = (
                float(audio.size)
                if chunk_index == len(starts) - 1
                else end - overlap_samples / 2.0
            )
            keep = (local_centers >= keep_start) & (local_centers < keep_end)
            if not np.any(keep):
                raise AlignmentScriptError(
                    "CTC_EMISSION_STITCH_FAILED",
                    "Overlap trimming removed every frame from a HuBERT chunk.",
                    chunkIndex=chunk_index,
                )
            emissions.append(np.ascontiguousarray(log_probs[keep], dtype=np.float32))
            centers.append(np.ascontiguousarray(local_centers[keep], dtype=np.float64))
            chunk_records.append(
                {
                    "index": chunk_index,
                    "startModelSample": start,
                    "endModelSample": end,
                    "emissionFrames": int(log_probs.shape[0]),
                    "keptFrames": int(np.count_nonzero(keep)),
                }
            )
            del values, logits, log_probs

    if not emissions:
        raise AlignmentScriptError(
            "CTC_EMISSION_EMPTY",
            "HuBERT produced no usable emission frames.",
        )
    joined = np.concatenate(emissions, axis=0)
    joined_centers = np.concatenate(centers, axis=0)
    if np.any(np.diff(joined_centers) <= 0):
        raise AlignmentScriptError(
            "CTC_EMISSION_STITCH_FAILED",
            "Stitched HuBERT emission frames are not strictly monotonic.",
        )
    return joined, joined_centers, {
        "chunkSeconds": chunk_seconds,
        "overlapSeconds": overlap_seconds,
        "chunkCount": len(chunk_records),
        "chunks": chunk_records,
        "emissionFrameCount": int(joined.shape[0]),
        "vocabularySize": int(joined.shape[1]),
        "emissionElapsedSec": round(time.monotonic() - started, 3),
    }


def ctc_viterbi(
    log_probs: np.ndarray,
    target_ids: Sequence[int],
    *,
    blank_id: int,
) -> tuple[list[PhonePath], float]:
    """Return target-label frame spans from the best legal CTC state path."""

    emissions = np.ascontiguousarray(log_probs, dtype=np.float32)
    if emissions.ndim != 2 or emissions.shape[0] == 0:
        raise AlignmentScriptError("CTC_EMISSION_INVALID", "CTC emissions must be a non-empty matrix.")
    target = np.asarray(target_ids, dtype=np.int64)
    if target.ndim != 1 or target.size == 0:
        raise AlignmentScriptError("CTC_TARGET_EMPTY", "CTC target phone sequence is empty.")
    if blank_id < 0 or blank_id >= emissions.shape[1]:
        raise AlignmentScriptError("CTC_BLANK_INVALID", "CTC blank id is outside the vocabulary.")
    if np.any(target < 0) or np.any(target >= emissions.shape[1]):
        raise AlignmentScriptError("CTC_TARGET_INVALID", "A CTC target id is outside the vocabulary.")
    if np.any(target == blank_id):
        raise AlignmentScriptError("CTC_TARGET_INVALID", "The lexical target contains the CTC blank id.")

    repeated = int(np.count_nonzero(target[1:] == target[:-1]))
    minimum_frames = int(target.size + repeated)
    frame_count = int(emissions.shape[0])
    if frame_count < minimum_frames:
        raise AlignmentScriptError(
            "CTC_PATH_IMPOSSIBLE",
            "There are too few HuBERT frames for a legal CTC path.",
            emissionFrames=frame_count,
            targetPhones=int(target.size),
            adjacentRepeats=repeated,
            minimumFrames=minimum_frames,
        )

    state_count = int(target.size * 2 + 1)
    state_labels = np.full(state_count, blank_id, dtype=np.int64)
    state_labels[1::2] = target
    skip_allowed = np.zeros(state_count, dtype=bool)
    if state_count > 3:
        skip_allowed[3::2] = state_labels[3::2] != state_labels[1:-2:2]

    negative = np.float32(-1.0e30)
    previous = np.full(state_count, negative, dtype=np.float32)
    previous[0] = emissions[0, blank_id]
    if state_count > 1:
        previous[1] = emissions[0, target[0]]
    backpointers = np.zeros((frame_count, state_count), dtype=np.uint8)
    step_scores = np.empty(state_count, dtype=np.float32)
    skip_scores = np.empty(state_count, dtype=np.float32)
    current = np.empty(state_count, dtype=np.float32)

    for frame in range(1, frame_count):
        step_scores[0] = negative
        step_scores[1:] = previous[:-1]
        skip_scores[:2] = negative
        skip_scores[2:] = previous[:-2]
        skip_scores[~skip_allowed] = negative

        np.copyto(current, previous)
        choice = backpointers[frame]
        choice.fill(0)
        mask = step_scores > current
        current[mask] = step_scores[mask]
        choice[mask] = 1
        mask = skip_scores > current
        current[mask] = skip_scores[mask]
        choice[mask] = 2
        current += emissions[frame, state_labels]
        previous, current = current, previous

    final_candidates = [(state_count - 1, previous[state_count - 1])]
    if state_count > 1:
        final_candidates.append((state_count - 2, previous[state_count - 2]))
    final_state, final_score = max(final_candidates, key=lambda item: float(item[1]))
    if float(final_score) <= float(negative) / 2 or not math.isfinite(float(final_score)):
        raise AlignmentScriptError(
            "CTC_PATH_IMPOSSIBLE",
            "No finite global CTC Viterbi path exists for the lyric phones.",
            emissionFrames=frame_count,
            targetPhones=int(target.size),
        )

    states = np.empty(frame_count, dtype=np.int32)
    state = int(final_state)
    states[-1] = state
    for frame in range(frame_count - 1, 0, -1):
        transition = int(backpointers[frame, state])
        state -= transition
        if state < 0:
            raise AlignmentScriptError(
                "CTC_BACKTRACE_INVALID",
                "The CTC Viterbi backtrace left the state graph.",
                frame=frame,
            )
        states[frame - 1] = state
    if states[0] not in {0, 1}:
        raise AlignmentScriptError(
            "CTC_BACKTRACE_INVALID",
            "The CTC Viterbi backtrace did not reach a valid initial state.",
        )

    paths: list[PhonePath] = []
    for target_index, token_id in enumerate(target):
        token_state = target_index * 2 + 1
        frames = np.flatnonzero(states == token_state)
        if frames.size == 0:
            raise AlignmentScriptError(
                "CTC_BACKTRACE_INVALID",
                "A target phone has no observed frame in the CTC Viterbi path.",
                targetIndex=target_index,
            )
        selected = emissions[frames, int(token_id)]
        confidence = float(np.exp(np.mean(selected, dtype=np.float64)))
        confidence = min(1.0, max(0.0, confidence))
        paths.append(
            PhonePath(
                target_index=target_index,
                first_frame=int(frames[0]),
                last_frame=int(frames[-1]),
                confidence=confidence,
            )
        )
    return paths, float(final_score)


def _frame_edge(centers: np.ndarray, frame: int, *, left: bool, audio_count: int) -> float:
    if left:
        if frame == 0:
            return 0.0
        return float((centers[frame - 1] + centers[frame]) / 2.0)
    if frame + 1 >= centers.size:
        return float(audio_count)
    return float((centers[frame] + centers[frame + 1]) / 2.0)


def _source_sample(model_sample: float, source_rate: int, source_count: int) -> int:
    mapped = int(math.floor(model_sample * source_rate / MODEL_SAMPLE_RATE + 0.5))
    return min(source_count, max(0, mapped))


def align(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    model_dir = args.model.resolve()
    required = [
        model_dir / "config.json",
        model_dir / "preprocessor_config.json",
        model_dir / "vocab.json",
    ]
    if not all(path.is_file() for path in required) or not any(
        path.is_file()
        for path in (model_dir / "model.safetensors", model_dir / "pytorch_model.bin")
    ):
        raise AlignmentScriptError(
            "CTC_MODEL_MISSING",
            "The local HuBERT CTC checkpoint is incomplete.",
            modelPath=str(model_dir),
        )
    try:
        lyrics = args.lyrics_file.read_text(encoding="utf-8")
    except OSError as error:
        raise AlignmentScriptError(
            "CTC_LYRICS_READ_FAILED",
            "The saved lyric text could not be read.",
        ) from error
    if not lyrics.strip():
        raise AlignmentScriptError("CTC_LYRICS_EMPTY", "The saved lyric text is empty.")

    if args.g2p_plan is not None:
        targets, g2p_metadata = load_phone_targets_from_plan(args.g2p_plan, lyrics)
    else:
        targets, g2p_metadata = build_phone_targets(lyrics)
    vocabulary, blank_id = _load_vocabulary(model_dir)
    missing_phones = sorted({target.phoneme for target in targets if target.phoneme not in vocabulary})
    if missing_phones:
        raise AlignmentScriptError(
            "CTC_PHONESET_MISMATCH",
            "OpenJTalk produced phones that the pinned HuBERT checkpoint cannot score.",
            missingPhones=missing_phones,
            modelId=MODEL_ID,
        )
    target_ids = [vocabulary[target.phoneme] for target in targets]

    selected_device, device_metadata = _select_device(args.device)
    audio, source_rate, source_count = _load_audio(args.audio)
    log_probs, centers, emission_metadata = emit_log_probs(
        audio,
        model_dir,
        device=selected_device,
        chunk_seconds=args.chunk_seconds,
        overlap_seconds=args.overlap_seconds,
    )
    if log_probs.shape[1] <= max(vocabulary.values()):
        raise AlignmentScriptError(
            "CTC_VOCABULARY_MISMATCH",
            "The checkpoint logits are smaller than its declared vocabulary.",
            logitVocabularySize=int(log_probs.shape[1]),
            declaredMaximumId=max(vocabulary.values()),
        )
    viterbi_started = time.monotonic()
    paths, path_score = ctc_viterbi(log_probs, target_ids, blank_id=blank_id)

    phone_payload: list[dict[str, Any]] = []
    previous_end = 0
    for path, target in zip(paths, targets, strict=True):
        left_model_sample = _frame_edge(
            centers,
            path.first_frame,
            left=True,
            audio_count=int(audio.size),
        )
        right_model_sample = _frame_edge(
            centers,
            path.last_frame,
            left=False,
            audio_count=int(audio.size),
        )
        start_sample = _source_sample(left_model_sample, source_rate, source_count)
        end_sample = _source_sample(right_model_sample, source_rate, source_count)
        if start_sample < previous_end or end_sample <= start_sample or end_sample > source_count:
            raise AlignmentScriptError(
                "CTC_SAMPLE_SPAN_INVALID",
                "A Viterbi phone frame span could not be mapped monotonically to source samples.",
                targetIndex=path.target_index,
                startSample=start_sample,
                endSample=end_sample,
                previousEndSample=previous_end,
                sourceSampleCount=source_count,
            )
        phone_payload.append(
            {
                "target_index": path.target_index,
                "surface": target.surface,
                "phoneme": target.phoneme,
                "frontend_index": target.frontend_index,
                "plan_phone_index": target.plan_phone_index,
                "mora_index": target.mora_index,
                "character_indices": list(target.character_indices),
                "start_sample": start_sample,
                "end_sample": end_sample,
                "confidence": path.confidence,
                "first_emission_frame": path.first_frame,
                "last_emission_frame": path.last_frame,
            }
        )
        previous_end = end_sample

    confidences = np.asarray([item["confidence"] for item in phone_payload], dtype=np.float64)
    warnings: list[str] = []
    mean_confidence = float(np.mean(confidences))
    if mean_confidence < 0.20:
        warnings.append(
            "HuBERT phone confidence is low for this singing performance; timestamps remain the "
            "raw best CTC path and were not repaired or redistributed."
        )
    if g2p_metadata.get("surfaceRecoveryFallbacks"):
        warnings.append(
            "Some OpenJTalk token spellings could not be mapped exactly back to the saved lyric surface."
        )
    return {
        "status": "ok",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "device": selected_device,
        "source_sample_rate": source_rate,
        "source_sample_count": source_count,
        "model_sample_rate": MODEL_SAMPLE_RATE,
        "phones": phone_payload,
        "warnings": warnings,
        "metadata": {
            **device_metadata,
            **g2p_metadata,
            **emission_metadata,
            "device": selected_device,
            "ctcBlankId": blank_id,
            "adjacentPhoneRepeats": int(
                np.count_nonzero(np.asarray(target_ids[1:]) == np.asarray(target_ids[:-1]))
            ),
            "viterbiPathLogScore": path_score,
            "viterbiElapsedSec": round(time.monotonic() - viterbi_started, 3),
            "meanPhoneConfidence": mean_confidence,
            "minimumPhoneConfidence": float(np.min(confidences)),
            "maximumPhoneConfidence": float(np.max(confidences)),
            "totalElapsedSec": round(time.monotonic() - started, 3),
            "timestampUnits": "source vocals-stem samples",
            "timestampPolicy": "observed CTC label-state frames only",
            "modelDownloadsDuringRun": False,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--lyrics-file", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--chunk-seconds", type=float, default=20.0)
    parser.add_argument("--overlap-seconds", type=float, default=2.0)
    parser.add_argument("--g2p-plan", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    try:
        payload = align(args)
        _write_json(args.output, payload)
        return 0
    except AlignmentScriptError as error:
        _write_json(
            args.output,
            {
                "status": "error",
                "error_code": error.code,
                "error_message": error.message,
                "details": error.details,
            },
        )
        print(f"{error.code}: {error.message}", file=sys.stderr)
        return 2
    except Exception as error:  # keep an explicit record for unexpected local failures
        _write_json(
            args.output,
            {
                "status": "error",
                "error_code": "CTC_UNEXPECTED_FAILURE",
                "error_message": "Unexpected local CTC alignment failure.",
                "details": {
                    "exceptionType": type(error).__name__,
                    "exceptionMessage": str(error)[:1_000],
                },
            },
        )
        traceback.print_exc(file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
