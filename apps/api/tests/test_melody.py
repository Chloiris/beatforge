from __future__ import annotations

import numpy as np

from beatforge_api.audio.melody import extract_melody_candidates


def _three_note_phrase(sample_rate: int) -> tuple[np.ndarray, list[int]]:
    audio = np.zeros(round(sample_rate * 1.6), dtype=np.float32)
    starts = [round(sample_rate * value) for value in (0.20, 0.65, 1.10)]
    frequencies = (220.0, 329.63, 440.0)
    duration = round(sample_rate * 0.30)
    fade = max(2, round(sample_rate * 0.012))
    envelope = np.ones(duration, dtype=np.float64)
    envelope[:fade] = np.linspace(0.0, 1.0, fade)
    envelope[-fade:] = np.linspace(1.0, 0.0, fade)
    for start, frequency in zip(starts, frequencies, strict=True):
        time = np.arange(duration, dtype=np.float64) / sample_rate
        audio[start : start + duration] = np.asarray(
            0.28 * envelope * np.sin(2.0 * np.pi * frequency * time),
            dtype=np.float32,
        )
    return audio, starts


def test_local_pitch_extractor_finds_note_onsets_in_other_stem() -> None:
    sample_rate = 44_100
    audio, expected = _three_note_phrase(sample_rate)

    result = extract_melody_candidates(audio, sample_rate)

    predicted = [candidate.refined_sample for candidate in result.candidates]
    tolerance = round(sample_rate * 0.080)
    assert all(
        any(abs(actual - sample) <= tolerance for actual in predicted)
        for sample in expected
    )
    assert result.method == "librosa_pyin_local"
    assert result.voiced_frame_count > 0
    assert all(
        candidate.semantic_evidence["pitchConfidence"] > 0.4
        for candidate in result.candidates
    )


def test_silent_other_stem_does_not_create_melody_notes() -> None:
    result = extract_melody_candidates(np.zeros(44_100, dtype=np.float32), 44_100)

    assert result.candidates == []
    assert result.voiced_frame_count == 0
