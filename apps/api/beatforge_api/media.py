from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import time
import uuid
import wave
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
from fastapi import UploadFile

from .config import Settings
from .errors import BeatForgeError

ALLOWED_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg"}
ALLOWED_MIME_TYPES = {
    ".wav": {"audio/wav", "audio/wave", "audio/x-wav", "application/octet-stream"},
    ".flac": {"audio/flac", "audio/x-flac", "application/octet-stream"},
    ".mp3": {"audio/mpeg", "audio/mp3", "application/octet-stream"},
    ".m4a": {"audio/mp4", "audio/x-m4a", "video/mp4", "application/octet-stream"},
    ".aac": {"audio/aac", "audio/mpeg", "application/octet-stream"},
    ".ogg": {"audio/ogg", "application/ogg", "application/octet-stream"},
}


@dataclass(frozen=True)
class AudioProbe:
    sample_rate: int
    channels: int
    sample_count: int
    duration_sec: float
    format: str


def safe_original_name(name: str | None) -> str:
    normalized = (name or "audio").replace("\\", "/")
    cleaned = Path(normalized).name.replace("\x00", "").strip()
    return cleaned[:500] or "audio"


def validate_upload_metadata(filename: str | None, content_type: str | None) -> tuple[str, str]:
    safe_name = safe_original_name(filename)
    extension = Path(safe_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise BeatForgeError(
            "UNSUPPORTED_AUDIO_FORMAT",
            "Supported formats are WAV, FLAC, MP3, M4A, AAC and OGG",
            status_code=415,
            details={"extension": extension or None},
        )
    normalized_mime = (content_type or "application/octet-stream").split(";", 1)[0].lower()
    if normalized_mime not in ALLOWED_MIME_TYPES[extension]:
        raise BeatForgeError(
            "INVALID_AUDIO_MIME",
            "The file MIME type does not match its audio extension",
            status_code=415,
            details={"extension": extension, "mimeType": normalized_mime},
        )
    return safe_name, extension


async def persist_upload(upload: UploadFile, settings: Settings) -> tuple[Path, str, str, int]:
    original_name, extension = validate_upload_metadata(upload.filename, upload.content_type)
    settings.ensure_directories()
    destination = settings.audio_dir / f"{uuid.uuid4().hex}{extension}"
    total = 0
    try:
        with destination.open("xb") as output:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > settings.max_upload_bytes:
                    raise BeatForgeError(
                        "UPLOAD_TOO_LARGE",
                        (
                            "Audio files are limited to "
                            f"{settings.max_upload_bytes // (1024 * 1024)} MB"
                        ),
                        status_code=413,
                        details={"maxBytes": settings.max_upload_bytes},
                    )
                output.write(chunk)
        if total == 0:
            raise BeatForgeError("EMPTY_UPLOAD", "The uploaded audio file is empty")
        await asyncio.to_thread(probe_audio, destination)
        return destination, original_name, extension.lstrip("."), total
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


def probe_audio(path: Path) -> AudioProbe:
    try:
        info = sf.info(str(path))
        if info.samplerate > 0 and info.frames > 0 and info.channels > 0:
            return AudioProbe(
                sample_rate=int(info.samplerate),
                channels=int(info.channels),
                sample_count=int(info.frames),
                duration_sec=float(info.frames / info.samplerate),
                format=path.suffix.lower().lstrip("."),
            )
    except (RuntimeError, TypeError):
        pass

    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise BeatForgeError(
            "AUDIO_DECODE_UNAVAILABLE",
            "This format requires ffmpeg/ffprobe, but ffprobe was not found",
            status_code=422,
        )
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels,duration,duration_ts,time_base,codec_name",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=30, check=False)
    if completed.returncode != 0:
        raise BeatForgeError(
            "INVALID_AUDIO_FILE",
            "The uploaded file could not be decoded as audio",
            status_code=422,
            details={"decoder": completed.stderr.strip()[-500:]},
        )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if not streams:
        raise BeatForgeError(
            "NO_AUDIO_STREAM", "The uploaded file has no audio stream", status_code=422
        )
    stream = streams[0]
    sample_rate = int(stream.get("sample_rate") or 0)
    channels = int(stream.get("channels") or 0)
    duration = float(stream.get("duration") or payload.get("format", {}).get("duration") or 0)
    if sample_rate <= 0 or channels <= 0 or duration <= 0:
        raise BeatForgeError(
            "INVALID_AUDIO_METADATA",
            "Audio sample rate, channels or duration is invalid",
            status_code=422,
        )
    return AudioProbe(
        sample_rate=sample_rate,
        channels=channels,
        sample_count=max(1, round(duration * sample_rate)),
        duration_sec=duration,
        format=path.suffix.lower().lstrip("."),
    )


def prepare_analysis_source(
    path: Path, output_dir: Path, *, force_ffmpeg: bool = False
) -> tuple[Path, bool]:
    """Return a libsndfile-readable source, decoding through ffmpeg when necessary.

    ``sf.info`` can succeed for a damaged or unusual MP3 even when decoding the
    complete stream later fails. Callers that already observed a libsndfile
    decode error can therefore force the existing ffmpeg recovery path.
    """
    if not force_ffmpeg:
        try:
            sf.info(str(path))
            return path, False
        except RuntimeError:
            pass
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise BeatForgeError(
            "AUDIO_DECODE_UNAVAILABLE",
            "ffmpeg is required to decode this audio format",
            status_code=422,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    decoded = output_dir / f"{path.stem}-decoded.wav"
    command = [
        ffmpeg,
        "-v",
        "error",
        "-y",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-c:a",
        "pcm_f32le",
        str(decoded),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
    if completed.returncode != 0:
        decoded.unlink(missing_ok=True)
        raise BeatForgeError(
            "AUDIO_DECODE_FAILED",
            "ffmpeg could not decode the audio file",
            status_code=422,
            details={"decoder": completed.stderr.strip()[-500:]},
        )
    return decoded, True


def cleanup_stale_decoded(output_dir: Path, max_age_seconds: float = 24 * 60 * 60) -> int:
    """Remove only abandoned ffmpeg analysis copies older than the safety window."""
    if not output_dir.is_dir():
        return 0
    cutoff = time.time() - max_age_seconds
    removed = 0
    resolved_root = output_dir.resolve()
    for path in output_dir.glob("*-decoded.wav"):
        try:
            resolved = path.resolve()
            if resolved.is_relative_to(resolved_root) and resolved.stat().st_mtime < cutoff:
                resolved.unlink(missing_ok=True)
                removed += 1
        except OSError:
            continue
    return removed


def probe_wav_standard_library(path: Path) -> AudioProbe:
    """Tiny WAV probe used by tests and as a diagnostic fallback."""
    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        sample_count = handle.getnframes()
        return AudioProbe(
            sample_rate=sample_rate,
            channels=handle.getnchannels(),
            sample_count=sample_count,
            duration_sec=sample_count / sample_rate,
            format="wav",
        )
