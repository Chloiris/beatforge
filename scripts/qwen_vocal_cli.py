"""Run the optional Qwen vocal models in their isolated Python environment.

The API invokes this script as a subprocess so Qwen's tightly pinned Transformers
stack cannot replace BeatForge's base audio dependencies. Inputs and outputs stay on
the local filesystem; no cloud API is used.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))

from beatforge_api.audio.qwen_vocal import QwenVocalAnalyzer, QwenVocalConfig  # noqa: E402
from beatforge_api.audio.lyric_chunking import assign_lyric_lines  # noqa: E402
from beatforge_api.export_safety import public_model_identifier  # noqa: E402


_ROMAJI = {
    "ア": "a", "イ": "i", "ウ": "u", "エ": "e", "オ": "o",
    "カ": "ka", "キ": "ki", "ク": "ku", "ケ": "ke", "コ": "ko",
    "ガ": "ga", "ギ": "gi", "グ": "gu", "ゲ": "ge", "ゴ": "go",
    "サ": "sa", "シ": "shi", "ス": "su", "セ": "se", "ソ": "so",
    "ザ": "za", "ジ": "ji", "ズ": "zu", "ゼ": "ze", "ゾ": "zo",
    "タ": "ta", "チ": "chi", "ツ": "tsu", "テ": "te", "ト": "to",
    "ダ": "da", "ヂ": "ji", "ヅ": "zu", "デ": "de", "ド": "do",
    "ナ": "na", "ニ": "ni", "ヌ": "nu", "ネ": "ne", "ノ": "no",
    "ハ": "ha", "ヒ": "hi", "フ": "fu", "ヘ": "he", "ホ": "ho",
    "バ": "ba", "ビ": "bi", "ブ": "bu", "ベ": "be", "ボ": "bo",
    "パ": "pa", "ピ": "pi", "プ": "pu", "ペ": "pe", "ポ": "po",
    "マ": "ma", "ミ": "mi", "ム": "mu", "メ": "me", "モ": "mo",
    "ヤ": "ya", "ユ": "yu", "ヨ": "yo", "ラ": "ra", "リ": "ri",
    "ル": "ru", "レ": "re", "ロ": "ro", "ワ": "wa", "ヲ": "o",
    "ン": "n", "ヴ": "vu",
    "キャ": "kya", "キュ": "kyu", "キョ": "kyo",
    "ギャ": "gya", "ギュ": "gyu", "ギョ": "gyo",
    "シャ": "sha", "シュ": "shu", "ショ": "sho",
    "ジャ": "ja", "ジュ": "ju", "ジョ": "jo",
    "チャ": "cha", "チュ": "chu", "チョ": "cho",
    "ニャ": "nya", "ニュ": "nyu", "ニョ": "nyo",
    "ヒャ": "hya", "ヒュ": "hyu", "ヒョ": "hyo",
    "ビャ": "bya", "ビュ": "byu", "ビョ": "byo",
    "ピャ": "pya", "ピュ": "pyu", "ピョ": "pyo",
    "ミャ": "mya", "ミュ": "myu", "ミョ": "myo",
    "リャ": "rya", "リュ": "ryu", "リョ": "ryo",
    "ティ": "ti", "ディ": "di", "ファ": "fa", "フィ": "fi",
    "フェ": "fe", "フォ": "fo", "ウィ": "wi", "ウェ": "we", "ウォ": "wo",
}

_ALIGNMENT_CHUNK_SECONDS = 20.0
_MAX_LINES_PER_CHUNK = 10
_ALIGNMENT_PADDING_SECONDS = 1.25
_MIN_CHUNK_ACTIVE_RATIO = 0.02


def _romanize_katakana(text: str) -> str:
    output: list[str] = []
    geminate = False
    index = 0
    while index < len(text):
        character = text[index]
        if character == "ッ":
            geminate = True
            index += 1
            continue
        if character == "ー":
            if output and output[-1]:
                vowels = [letter for letter in output[-1] if letter in "aeiou"]
                if vowels:
                    output.append(vowels[-1])
            index += 1
            continue
        pair = text[index : index + 2]
        syllable = _ROMAJI.get(pair)
        if syllable is not None:
            index += 2
        else:
            syllable = _ROMAJI.get(character, character)
            index += 1
        if geminate and syllable and syllable[0] not in "aeioun":
            syllable = syllable[0] + syllable
        geminate = False
        output.append(syllable)
    return "".join(output)


def _add_japanese_readings(payload: dict[str, Any]) -> dict[str, Any]:
    """Attach display-only kana/romaji without changing aligner time spans."""

    try:
        import pyopenjtalk
    except ImportError:
        return payload
    dictionary_path = Path(os.fsdecode(pyopenjtalk.OPEN_JTALK_DICT_DIR))
    if not dictionary_path.is_dir():
        warnings = payload.setdefault("warnings", [])
        if isinstance(warnings, list):
            warnings.append(
                "Open JTalk dictionary is not installed; run "
                "python scripts/beatforge.py prepare-vocal-models."
            )
        return payload
    timestamps = payload.get("timestamps")
    if not isinstance(timestamps, (list, tuple)):
        return payload
    for item in timestamps:
        if not isinstance(item, dict):
            continue
        source = str(item.get("text", ""))
        try:
            reading = str(pyopenjtalk.g2p(source, kana=True))
        except Exception:
            reading = source
        item["kana"] = reading
        item["romaji"] = _romanize_katakana(reading)
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    values, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if values.ndim == 2:
        values = np.mean(values, axis=1, dtype=np.float32)
    return np.ascontiguousarray(values, dtype=np.float32), int(sample_rate)


def _song_chunks(sample_count: int, sample_rate: int) -> list[tuple[int, int]]:
    chunk_size = max(1, round(sample_rate * _ALIGNMENT_CHUNK_SECONDS))
    return [
        (start, min(sample_count, start + chunk_size))
        for start in range(0, sample_count, chunk_size)
    ]


def _chunk_activity_ratios(
    audio: np.ndarray,
    sample_rate: int,
    chunks: list[tuple[int, int]],
) -> tuple[list[float], float]:
    """Measure short-time absolute activity so silent chunks never reach ASR."""

    window = max(2, round(sample_rate * 0.040))
    hop = max(1, round(sample_rate * 0.020))
    frame_sets: list[np.ndarray] = []
    for start, end in chunks:
        values = np.asarray(audio[start:end], dtype=np.float32)
        if values.size == 0:
            frame_sets.append(np.zeros(1, dtype=np.float64))
            continue
        squared = np.square(values, dtype=np.float64)
        cumulative = np.concatenate(([0.0], np.cumsum(squared)))
        frame_starts = np.arange(0, values.size, hop, dtype=np.int64)
        frame_ends = np.minimum(frame_starts + window, values.size)
        energy = (cumulative[frame_ends] - cumulative[frame_starts]) / np.maximum(
            frame_ends - frame_starts,
            1,
        )
        frame_sets.append(np.sqrt(np.maximum(energy, 0.0)))
    all_frames = np.concatenate(frame_sets)
    reference = max(float(np.quantile(all_frames, 0.90)), 1e-9)
    noise = max(float(np.quantile(all_frames, 0.20)), 1e-9)
    floor = max(
        10.0 ** (-55.0 / 20.0),
        reference * 0.02,
        min(noise * 4.0, reference * 0.25),
    )
    ratios = [float(np.mean(frames >= floor)) for frames in frame_sets]
    return ratios, floor


def _align_song(
    analyzer: QwenVocalAnalyzer,
    audio: np.ndarray,
    sample_rate: int,
    text: str,
) -> dict[str, Any]:
    """Use singing ASR to localize lyric lines before short forced-alignment calls."""

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    chunks = _song_chunks(audio.size, sample_rate)
    warnings: list[str] = []
    transcripts: list[str] = []
    diagnostics: list[dict[str, Any]] = []
    activity_ratios, activity_floor = _chunk_activity_ratios(audio, sample_rate, chunks)
    for chunk_index, (start, end) in enumerate(chunks):
        activity_ratio = activity_ratios[chunk_index]
        if activity_ratio < _MIN_CHUNK_ACTIVE_RATIO:
            transcripts.append("")
            diagnostics.append(
                {
                    "index": chunk_index,
                    "startSample": start,
                    "endSample": end,
                    "transcript": "",
                    "status": "silent",
                    "activeRatio": round(activity_ratio, 6),
                    "activityFloorDbfs": round(20.0 * np.log10(activity_floor), 3),
                }
            )
            continue
        result = analyzer.transcribe_vocals(
            audio[start:end],
            sample_rate,
            language="Japanese",
            align=False,
        )
        transcript = result.text.strip() if result.status == "ok" else ""
        transcripts.append(transcript)
        warnings.extend(result.warnings)
        if result.status != "ok":
            warnings.append(
                f"第 {chunk_index + 1} 个分段 ASR 失败，已跳过："
                f"{result.error_code or 'unknown'}。"
            )
        diagnostics.append(
            {
                "index": chunk_index,
                "startSample": start,
                "endSample": end,
                "transcript": transcript,
                "status": result.status,
                "activeRatio": round(activity_ratio, 6),
                "activityFloorDbfs": round(20.0 * np.log10(activity_floor), 3),
            }
        )
    if not any(transcripts):
        return {
            "status": "failed",
            "text": text,
            "timestamps": [],
            "model": public_model_identifier(analyzer.config.aligner_model),
            "warnings": warnings,
            "error_code": "empty_chunk_transcripts",
            "error_message": "Singing ASR did not localize any lyric chunks.",
        }

    assignments, unassigned_lines = assign_lyric_lines(
        transcripts,
        lines,
        max_lines_per_chunk=_MAX_LINES_PER_CHUNK,
    )
    if not assignments:
        return {
            "status": "failed",
            "text": text,
            "timestamps": [],
            "model": public_model_identifier(analyzer.config.aligner_model),
            "warnings": warnings,
            "error_code": "lyrics_not_localized",
            "error_message": "Singing ASR could not match the supplied lyric lines.",
        }

    # Loading the 0.6B aligner while the 1.7B ASR remains resident is needlessly
    # expensive on Apple Silicon. The analyzer will lazily load the next model.
    analyzer.release_cached_models(asr=True, aligner=False)
    gc.collect()

    padding = round(sample_rate * _ALIGNMENT_PADDING_SECONDS)
    timestamps: list[dict[str, Any]] = []
    devices: list[str] = []
    for assignment in assignments:
        nominal_start, nominal_end = chunks[assignment.chunk_index]
        start = max(0, nominal_start - padding)
        end = min(audio.size, nominal_end + padding)
        aligned = analyzer.align_known_japanese(
            audio[start:end],
            sample_rate,
            assignment.text,
        )
        warnings.extend(aligned.warnings)
        diagnostics[assignment.chunk_index].update(
            {
                "lineStart": assignment.start_line,
                "lineEnd": assignment.end_line,
                "matchSimilarity": round(assignment.similarity, 6),
                "matchConfidence": round(assignment.confidence, 6),
                "alignmentStatus": aligned.status,
            }
        )
        if aligned.status != "ok":
            warnings.append(
                f"第 {assignment.chunk_index + 1} 个歌词分段对齐失败，已跳过："
                f"{aligned.error_code or 'unknown'}。"
            )
            continue
        if aligned.device:
            devices.append(aligned.device)
        for timestamp in aligned.timestamps:
            item = asdict(timestamp)
            start_sample = min(max(start + int(item["start_sample"]), 0), audio.size)
            end_sample = min(
                max(start + int(item["end_sample"]), start_sample),
                audio.size,
            )
            item.update(
                {
                    "start_sample": start_sample,
                    "end_sample": end_sample,
                    "start_sec": start_sample / sample_rate,
                    "end_sec": end_sample / sample_rate,
                    "chunk_index": assignment.chunk_index,
                    "chunk_match_confidence": round(assignment.confidence, 6),
                }
            )
            timestamps.append(item)

    timestamps.sort(key=lambda item: (int(item["start_sample"]), int(item["end_sample"])))
    if unassigned_lines:
        warnings.append(
            f"{len(unassigned_lines)} 行歌词未通过分段 ASR 匹配，已安全跳过而不是强塞入静音区。"
        )
    if not timestamps:
        return {
            "status": "failed",
            "text": text,
            "timestamps": [],
            "model": public_model_identifier(analyzer.config.aligner_model),
            "warnings": warnings,
            "chunks": diagnostics,
            "error_code": "all_chunk_alignments_failed",
            "error_message": "Every localized lyric chunk failed forced alignment.",
        }
    payload = {
        "status": "ok",
        "text": text,
        "timestamps": timestamps,
        "model": public_model_identifier(analyzer.config.aligner_model),
        "device": devices[0] if devices and len(set(devices)) == 1 else "mixed",
        "warnings": warnings,
        "chunks": diagnostics,
        "alignment_strategy": "singing_asr_guided_chunks",
        "unassigned_line_count": len(unassigned_lines),
    }
    return _add_japanese_readings(payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "operation",
        choices=("diagnostics", "transcribe", "align", "align_song"),
    )
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asr-model", required=True)
    parser.add_argument("--aligner-model", required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args()

    analyzer = QwenVocalAnalyzer(
        QwenVocalConfig(
            asr_model=args.asr_model,
            aligner_model=args.aligner_model,
            device=args.device,
        )
    )
    if args.operation == "diagnostics":
        payload = asdict(analyzer.diagnostics())
    else:
        if args.audio is None:
            parser.error("--audio is required")
        audio, sample_rate = _read_audio(args.audio)
        if args.operation == "transcribe":
            payload = asdict(
                analyzer.transcribe_vocals(
                    audio,
                    sample_rate,
                    language="Japanese",
                    align=False,
                )
            )
        elif args.operation == "align":
            if args.text_file is None:
                parser.error("--text-file is required for alignment")
            text = args.text_file.read_text(encoding="utf-8")
            payload = _add_japanese_readings(
                asdict(analyzer.align_known_japanese(audio, sample_rate, text))
            )
        else:
            if args.text_file is None:
                parser.error("--text-file is required for song alignment")
            text = args.text_file.read_text(encoding="utf-8")
            payload = _align_song(analyzer, audio, sample_rate, text)
    if args.operation != "diagnostics":
        for field in ("model", "aligner_model"):
            model = payload.get(field)
            if isinstance(model, str):
                payload[field] = public_model_identifier(model)
    _write_json(args.output, payload)


if __name__ == "__main__":
    main()
