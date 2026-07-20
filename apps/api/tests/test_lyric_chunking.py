from __future__ import annotations

from beatforge_api.audio.lyric_chunking import (
    assign_lyric_lines,
    normalize_lyric_text,
)


def test_normalize_lyric_text_ignores_spacing_width_and_punctuation() -> None:
    assert normalize_lyric_text("Ｍｉｒａｉ、 を 待つ！") == "miraiを待つ"


def test_chunk_matching_skips_instrumental_and_unmatched_lines() -> None:
    assignments, unassigned = assign_lyric_lines(
        ["星明かりを探して歩く", "", "静かな海に響く声"],
        [
            "No Destiny",
            "星明かりを探して歩く",
            "instrumental",
            "静かな海に響く声",
            "ending",
        ],
        max_lines_per_chunk=3,
    )

    assert [(item.chunk_index, item.start_line, item.end_line) for item in assignments] == [
        (0, 1, 2),
        (2, 3, 4),
    ]
    assert unassigned == [0, 2, 4]
    assert all(item.confidence == 1.0 for item in assignments)


def test_chunk_matching_is_monotonic_for_repeated_phrases() -> None:
    assignments, _unassigned = assign_lyric_lines(
        ["雨を待つ次の朝", "雨を待つ最後の夜"],
        ["雨を待つ", "次の朝", "雨を待つ", "最後の夜"],
        max_lines_per_chunk=2,
    )

    assert [(item.chunk_index, item.start_line, item.end_line) for item in assignments] == [
        (0, 0, 2),
        (1, 2, 4),
    ]


def test_unrelated_asr_text_is_not_forced_into_silence() -> None:
    assignments, unassigned = assign_lyric_lines(
        ["ピアノだけの間奏", "ドラムソロ"],
        ["遠い空に声を重ねる", "明日を迎える"],
    )

    assert assignments == []
    assert unassigned == [0, 1]
