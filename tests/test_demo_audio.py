from __future__ import annotations

import json
import wave
from pathlib import Path

from scripts.generate_demo_audio import SAMPLE_RATE, SPECS, generate


def test_demo_generation_is_deterministic_and_sample_based(tmp_path: Path) -> None:
    first = generate(SPECS[0], tmp_path, force=True)
    second = generate(SPECS[0], tmp_path, force=True)
    assert first["onsets"] == second["onsets"]
    assert all(isinstance(onset["sample"], int) for onset in first["onsets"])
    assert all(onset["timeSec"] == onset["sample"] / SAMPLE_RATE for onset in first["onsets"])


def test_generated_wav_metadata_matches_truth(tmp_path: Path) -> None:
    truth = generate(SPECS[2], tmp_path, force=True)
    audio_path = tmp_path / truth["audioFile"]
    with wave.open(str(audio_path), "rb") as audio:
        assert audio.getframerate() == truth["sampleRate"]
        assert audio.getnframes() == truth["sampleCount"]
        assert audio.getnchannels() == truth["channels"]
    stored = json.loads((tmp_path / "storage/demo/glass-tide.ground-truth.json").read_text())
    assert stored["bpm"] == 96.0
