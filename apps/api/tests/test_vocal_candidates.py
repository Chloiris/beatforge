from __future__ import annotations

import numpy as np

from beatforge_api.audio.vocal_candidates import (
    VocalAcousticCandidate,
    extract_vocal_acoustic_candidates,
)
from beatforge_api.vocal_jobs import _build_anchors


def _vocal_phrase(sample_rate: int) -> tuple[np.ndarray, list[int]]:
    audio = np.zeros(round(sample_rate * 2.0), dtype=np.float32)
    starts = [round(sample_rate * value) for value in (0.35, 1.12)]
    for start, frequency in zip(starts, (180.0, 285.0), strict=True):
        duration = round(sample_rate * 0.42)
        time = np.arange(duration, dtype=np.float64) / sample_rate
        attack = 1.0 - np.exp(-time / 0.010)
        release = np.minimum(1.0, (duration / sample_rate - time) / 0.05)
        signal = 0.24 * attack * release * np.sin(2.0 * np.pi * frequency * time)
        audio[start : start + duration] += np.asarray(signal, dtype=np.float32)
    return audio, starts


def test_vocal_acoustic_detector_finds_real_phrase_attacks() -> None:
    sample_rate = 16_000
    audio, expected = _vocal_phrase(sample_rate)

    result = extract_vocal_acoustic_candidates(audio, sample_rate)

    predicted = [candidate.sample for candidate in result.candidates]
    tolerance = round(sample_rate * 0.10)
    assert all(
        any(abs(actual - sample) <= tolerance for actual in predicted)
        for sample in expected
    )
    assert any(
        candidate.pitch_score > 0.2 or candidate.transition_score > 0.2
        for candidate in result.candidates
    )


def test_qwen_phrase_boundary_fuses_to_acoustic_candidate_sample() -> None:
    sample_rate = 8_000
    samples = np.arange(sample_rate, dtype=np.float32)
    envelope = np.zeros(sample_rate, dtype=np.float32)
    envelope[700:2_400] = np.linspace(0.0, 1.0, 1_700, dtype=np.float32)
    audio = 0.18 * np.sin(2 * np.pi * 220 * samples / sample_rate) * envelope
    acoustic = VocalAcousticCandidate(
        sample=920,
        confidence=0.8,
        onset_score=0.75,
        envelope_score=0.7,
        pitch_score=0.6,
        transition_score=0.5,
        activity_score=0.65,
    )

    result = _build_anchors(
        [
            {
                "text": "phrase",
                "start_sample": 760,
                "end_sample": 2_200,
                "chunk_index": 0,
                "chunk_match_confidence": 0.9,
            }
        ],
        audio=audio,
        sample_rate=sample_rate,
        bpm=120.0,
        beat_offset_sample=0,
        acoustic_candidates=[acoustic],
    )

    assert len(result.anchors) == 1
    assert result.anchors[0]["aligned_sample"] == 760
    assert result.anchors[0]["refined_sample"] == 920
    assert result.anchors[0]["pitch_score"] == 0.6
    assert result.anchors[0]["transition_score"] == 0.5
