from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scripts.qwen_vocal_cli import _align_song, _chunk_activity_ratios, _song_chunks

from beatforge_api.audio.qwen_vocal import (
    VocalAlignmentResult,
    VocalTimestamp,
    VocalTranscriptionResult,
)


@dataclass(frozen=True)
class _FakeConfig:
    aligner_model: str = "local-aligner"


class _FakeAnalyzer:
    config = _FakeConfig()

    def __init__(self) -> None:
        self.transcripts = iter(("最初の声", "次の声"))
        self.released = False

    def transcribe_vocals(
        self,
        _audio: np.ndarray,
        _sample_rate: int,
        **_kwargs: object,
    ) -> VocalTranscriptionResult:
        return VocalTranscriptionResult(status="ok", text=next(self.transcripts))

    def release_cached_models(self, **_kwargs: object) -> None:
        self.released = True

    def align_known_japanese(
        self,
        _audio: np.ndarray,
        sample_rate: int,
        text: str,
    ) -> VocalAlignmentResult:
        return VocalAlignmentResult(
            status="ok",
            text=text,
            model="local-aligner",
            device="cpu",
            timestamps=(
                VocalTimestamp(
                    text=text,
                    start_sample=50,
                    end_sample=100,
                    start_sec=50 / sample_rate,
                    end_sec=100 / sample_rate,
                ),
            ),
        )


def test_song_alignment_localizes_lines_before_short_alignment() -> None:
    analyzer = _FakeAnalyzer()
    sample_rate = 100
    samples = np.arange(sample_rate * 40, dtype=np.float32)
    audio = 0.2 * np.sin(2 * np.pi * 8 * samples / sample_rate)
    payload = _align_song(
        analyzer,  # type: ignore[arg-type]
        audio,
        sample_rate,
        "最初の声\n次の声",
    )

    assert payload["status"] == "ok"
    assert payload["alignment_strategy"] == "singing_asr_guided_chunks"
    assert analyzer.released is True
    assert [item["chunk_index"] for item in payload["timestamps"]] == [0, 1]
    assert payload["timestamps"][0]["start_sample"] == 50
    assert payload["timestamps"][1]["start_sample"] == 1_925
    assert all(item["chunk_match_confidence"] == 1.0 for item in payload["timestamps"])


def test_chunk_activity_rejects_silent_half_of_song() -> None:
    sample_rate = 100
    audio = np.zeros(sample_rate * 40, dtype=np.float32)
    samples = np.arange(sample_rate * 20, dtype=np.float32)
    audio[: sample_rate * 20] = 0.2 * np.sin(2 * np.pi * 8 * samples / sample_rate)
    chunks = _song_chunks(audio.size, sample_rate)

    ratios, floor = _chunk_activity_ratios(audio, sample_rate, chunks)

    assert floor > 0
    assert ratios[0] > 0.9
    assert ratios[1] == 0.0
