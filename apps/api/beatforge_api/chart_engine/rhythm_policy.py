from __future__ import annotations

import math


def density_note_limit(
    difficulty: int,
    *,
    bpm: float | None = None,
    window_sec: float = 2.0,
) -> int:
    """Return a shared integer note cap for a half-open density window.

    Lv.8+ supports sustained 1/16 rows. The BPM-aware floor keeps that legal
    rhythm from being rejected simply because a song is faster, while jumps
    still count as two notes and 1/24 pressure can exceed the cap.
    """

    level = min(max(int(difficulty), 1), 15)
    learned_cap = math.floor((2.4 + level * 0.58) * window_sec)
    if level < 8 or bpm is None or not math.isfinite(bpm) or bpm <= 0:
        return max(1, learned_cap)
    sixteenth_cap = math.ceil(window_sec * bpm / 15.0 - 1e-9)
    return max(1, learned_cap, sixteenth_cap)


def density_limit_nps(difficulty: int, *, bpm: float | None = None) -> float:
    return density_note_limit(difficulty, bpm=bpm) / 2.0


def maximum_subdivision(difficulty: int) -> int:
    return 16 if int(difficulty) <= 10 else 24
