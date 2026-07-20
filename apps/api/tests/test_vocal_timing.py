from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from fractions import Fraction

import pytest

from beatforge_api.audio.vocal_timing import (
    VocalGridConfig,
    VocalTimingAnchor,
    align_vocal_anchors_to_grid,
    tokenize_kana_morae,
)


def _anchor(
    kana: str,
    refined_sample: int,
    *,
    aligned_sample: int | None = None,
    confidence: float = 0.9,
    kind: str = "mora",
) -> VocalTimingAnchor:
    return VocalTimingAnchor(
        original_text=kana,
        kana=kana,
        romaji=None,
        aligned_sample=refined_sample if aligned_sample is None else aligned_sample,
        refined_sample=refined_sample,
        confidence=confidence,
        kind=kind,  # type: ignore[arg-type]
    )


def test_tokenizes_yoon_sokuon_and_regular_morae() -> None:
    tokens = tokenize_kana_morae("きゃっぷ")

    assert [token.kana for token in tokens] == ["きゃ", "っ", "ぷ"]
    assert [token.kind for token in tokens] == ["mora", "sokuon", "mora"]


def test_tokenizes_extended_katakana_nasal_and_long_vowel() -> None:
    tokens = tokenize_kana_morae("ファンキー")

    assert [token.kana for token in tokens] == ["ファ", "ン", "キ", "ー"]
    assert [token.kind for token in tokens] == ["mora", "nasal", "mora", "sustain"]
    assert all(token.romaji is None for token in tokens)


def test_tokenizer_retains_punctuation_and_normalizes_dakuten() -> None:
    tokens = tokenize_kana_morae("か\u3099ん、 きょう！")

    assert [token.kana for token in tokens] == ["が", "ん", "、 ", "きょ", "う", "！"]
    assert [token.kind for token in tokens] == [
        "mora",
        "nasal",
        "silence",
        "mora",
        "mora",
        "silence",
    ]


def test_tokenizer_rejects_text_without_a_kana_reading() -> None:
    with pytest.raises(ValueError, match="provide a kana reading"):
        tokenize_kana_morae("今日は")


def test_maps_pronunciations_to_sixteenth_grid_without_overwriting_acoustics() -> None:
    anchors = [
        _anchor("か", 5_900, aligned_sample=5_750),
        _anchor("な", 12_100, aligned_sample=12_250),
        _anchor("た", 18_050, aligned_sample=17_900),
    ]

    result = align_vocal_anchors_to_grid(
        anchors,
        sample_rate=48_000,
        bpm=120.0,
        beat_offset_sample=0,
    )

    assert [anchor.grid_sample for anchor in result] == [6_000, 12_000, 18_000]
    assert [anchor.aligned_sample for anchor in result] == [5_750, 12_250, 17_900]
    assert [anchor.refined_sample for anchor in result] == [5_900, 12_100, 18_050]
    assert [anchor.grid_sample for anchor in anchors] == [None, None, None]


def test_grid_calculation_has_no_long_phrase_float_drift() -> None:
    sample_rate = 44_100
    bpm = 129.5
    offset = 777
    grid_index = 100_000
    step = Fraction(sample_rate * 60, 1) / (Fraction(Decimal(str(bpm))) * 4)
    exact = Fraction(offset, 1) + grid_index * step
    expected = (2 * exact.numerator + exact.denominator) // (2 * exact.denominator)
    anchor = _anchor("ら", expected + 1)

    result = align_vocal_anchors_to_grid(
        [anchor],
        sample_rate=sample_rate,
        bpm=bpm,
        beat_offset_sample=offset,
    )

    assert result[0].grid_sample == expected


def test_same_grid_conflict_never_creates_duplicate_attacks() -> None:
    anchors = [
        _anchor("か", 6_000, confidence=0.98),
        _anchor("な", 6_120, confidence=0.12),
        _anchor("た", 12_000, confidence=0.95),
    ]

    result = align_vocal_anchors_to_grid(
        anchors,
        sample_rate=48_000,
        bpm=120.0,
        beat_offset_sample=0,
    )
    assigned = [anchor.grid_sample for anchor in result if anchor.grid_sample is not None]

    assert len(assigned) == len(set(assigned))
    assert result[0].grid_sample == 6_000
    assert result[1].grid_sample is None
    assert result[2].grid_sample == 12_000


def test_conflict_drops_anchor_instead_of_forcing_a_distant_grid_cell() -> None:
    anchors = [
        _anchor("か", 6_000, confidence=0.98),
        _anchor("な", 8_200, confidence=0.98),
    ]

    result = align_vocal_anchors_to_grid(
        anchors,
        sample_rate=48_000,
        bpm=120.0,
        beat_offset_sample=0,
    )

    assert result[0].grid_sample == 6_000
    assert result[1].grid_sample is None


def test_non_vocalic_morae_do_not_create_pronunciation_attacks() -> None:
    anchors = [
        _anchor("か", 6_000),
        _anchor("ん", 8_000, kind="nasal"),
        _anchor("っ", 10_000, kind="sokuon"),
        _anchor("た", 12_000),
    ]

    result = align_vocal_anchors_to_grid(
        anchors,
        sample_rate=48_000,
        bpm=120.0,
        beat_offset_sample=0,
    )

    assert [anchor.grid_sample for anchor in result] == [6_000, None, None, 12_000]


def test_sustain_and_silence_do_not_consume_grid_cells() -> None:
    anchors = [
        _anchor("か", 6_000),
        _anchor("ー", 8_000, kind="sustain"),
        _anchor("、", 9_000, kind="silence"),
        _anchor("な", 12_000),
    ]

    result = align_vocal_anchors_to_grid(
        anchors,
        sample_rate=48_000,
        bpm=120.0,
        beat_offset_sample=0,
    )

    assert [anchor.grid_sample for anchor in result] == [6_000, None, None, 12_000]


def test_pickup_can_use_grid_indices_before_beat_offset() -> None:
    anchor = _anchor("あ", 36_000)

    result = align_vocal_anchors_to_grid(
        [anchor],
        sample_rate=48_000,
        bpm=120.0,
        beat_offset_sample=48_000,
        config=VocalGridConfig(allow_pickup=True),
    )

    assert result[0].grid_sample == 36_000


def test_rejects_non_monotonic_refined_anchors() -> None:
    anchors = [_anchor("か", 12_000), _anchor("な", 6_000)]

    with pytest.raises(ValueError, match="monotonic"):
        align_vocal_anchors_to_grid(
            anchors,
            sample_rate=48_000,
            bpm=120.0,
            beat_offset_sample=0,
        )


def test_realigning_replaces_only_previous_grid_suggestion() -> None:
    anchor = replace(_anchor("か", 6_050), grid_sample=123)

    result = align_vocal_anchors_to_grid(
        [anchor],
        sample_rate=48_000,
        bpm=120.0,
        beat_offset_sample=0,
    )

    assert result[0].grid_sample == 6_000
    assert result[0].aligned_sample == anchor.aligned_sample
    assert result[0].refined_sample == anchor.refined_sample
