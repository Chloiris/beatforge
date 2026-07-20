from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass, replace
from decimal import Decimal
from fractions import Fraction
from typing import Literal

MoraKind = Literal["mora", "phrase", "sokuon", "nasal", "sustain", "silence"]

_VALID_KINDS: frozenset[str] = frozenset(
    {"mora", "phrase", "sokuon", "nasal", "sustain", "silence"}
)
_SOKUON = frozenset("っッ")
_MORAIC_NASALS = frozenset("んン")
_LONG_VOWEL_MARKS = frozenset("ー")
_KANA_ITERATION_MARKS = frozenset("ゝゞヽヾ")
_SMALL_MORA_MODIFIERS = frozenset(
    "ゃゅょぁぃぅぇぉゎャュョァィゥェォヮㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇺㇻㇼㇽㇾㇿ"
)
_SILENCE_SYMBOLS = frozenset("♪♫♬♩〜~…")


@dataclass(frozen=True, slots=True)
class MoraToken:
    """A kana mora or a non-pronounced boundary retained from the lyric text."""

    original_text: str
    kana: str
    romaji: str | None = None
    kind: MoraKind = "mora"

    def __post_init__(self) -> None:
        if not self.original_text:
            raise ValueError("original_text must not be empty")
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"unsupported mora kind: {self.kind}")
        if self.kind != "silence" and not self.kana:
            raise ValueError("kana must not be empty for a pronounced token")


@dataclass(frozen=True, slots=True)
class VocalTimingAnchor:
    """A monotonic acoustic pronunciation anchor and its optional rhythmic suggestion.

    ``aligned_sample`` is the forced-aligner location and ``refined_sample`` is the
    acoustic onset refinement. Quantization only writes ``grid_sample``; neither
    acoustic location is overwritten.
    """

    original_text: str
    kana: str
    aligned_sample: int
    refined_sample: int
    confidence: float
    romaji: str | None = None
    grid_sample: int | None = None
    kind: MoraKind = "mora"

    def __post_init__(self) -> None:
        if not self.original_text:
            raise ValueError("original_text must not be empty")
        if self.kind not in _VALID_KINDS:
            raise ValueError(f"unsupported mora kind: {self.kind}")
        if self.kind != "silence" and not self.kana:
            raise ValueError("kana must not be empty for a pronounced anchor")
        if self.aligned_sample < 0 or self.refined_sample < 0:
            raise ValueError("acoustic sample indices must be non-negative")
        if self.grid_sample is not None and self.grid_sample < 0:
            raise ValueError("grid_sample must be non-negative when present")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be a finite value in [0, 1]")


@dataclass(frozen=True, slots=True)
class VocalGridConfig:
    """Configuration for sentence-level acoustic-to-sixteenth-note alignment."""

    subdivisions_per_beat: int = 4
    candidate_radius: int = 2
    refined_weight: float = 0.72
    aligned_weight: float = 0.18
    transition_weight: float = 0.22
    unassigned_penalty: float = 1.05
    max_snap_distance_steps: float = 0.60
    allow_pickup: bool = True
    max_states: int = 256
    quantized_kinds: tuple[MoraKind, ...] = ("mora", "phrase")

    def __post_init__(self) -> None:
        if self.subdivisions_per_beat <= 0:
            raise ValueError("subdivisions_per_beat must be positive")
        if self.candidate_radius < 0:
            raise ValueError("candidate_radius must be non-negative")
        if self.max_states <= 0:
            raise ValueError("max_states must be positive")
        weights = (
            self.refined_weight,
            self.aligned_weight,
            self.transition_weight,
            self.unassigned_penalty,
            self.max_snap_distance_steps,
        )
        if any(not math.isfinite(weight) or weight < 0.0 for weight in weights):
            raise ValueError("alignment weights must be finite and non-negative")
        if self.refined_weight + self.aligned_weight <= 0.0:
            raise ValueError("at least one acoustic weight must be positive")
        if self.max_snap_distance_steps <= 0.0:
            raise ValueError("max_snap_distance_steps must be positive")
        if any(kind not in _VALID_KINDS for kind in self.quantized_kinds):
            raise ValueError("quantized_kinds contains an unsupported kind")


@dataclass(slots=True)
class _DpState:
    cost: float
    last_grid_index: int | None
    last_anchor_position: int | None
    previous: _DpState | None
    assignment: int | None


def _is_kana(character: str) -> bool:
    name = unicodedata.name(character, "")
    return "HIRAGANA LETTER" in name or "KATAKANA LETTER" in name


def _is_silence_character(character: str) -> bool:
    category = unicodedata.category(character)
    return (
        character.isspace()
        or character in _SILENCE_SYMBOLS
        or category.startswith("P")
        or category.startswith("Z")
    )


def tokenize_kana_morae(text: str) -> list[MoraToken]:
    """Split normalized kana into morae while retaining punctuation as silence.

    Small ``ya/yu/yo`` and small-vowel spellings are joined to their preceding
    kana. Sokuon, moraic nasal, and the long-vowel mark each remain explicit so a
    downstream aligner can model their duration without inventing an attack.
    Kanji and Latin text are rejected: callers should provide a corrected kana
    reading rather than guessing pronunciation from display text.
    """

    normalized = unicodedata.normalize("NFC", text)
    if not normalized:
        return []

    tokens: list[MoraToken] = []
    for character in normalized:
        if character in _LONG_VOWEL_MARKS:
            tokens.append(MoraToken(original_text=character, kana=character, kind="sustain"))
            continue
        if character in _SOKUON:
            tokens.append(MoraToken(original_text=character, kana=character, kind="sokuon"))
            continue
        if character in _MORAIC_NASALS:
            tokens.append(MoraToken(original_text=character, kana=character, kind="nasal"))
            continue
        if _is_silence_character(character):
            if tokens and tokens[-1].kind == "silence":
                previous = tokens[-1]
                tokens[-1] = replace(
                    previous,
                    original_text=previous.original_text + character,
                    kana=previous.kana + character,
                )
            else:
                tokens.append(MoraToken(original_text=character, kana=character, kind="silence"))
            continue
        if character in _SMALL_MORA_MODIFIERS:
            if tokens and tokens[-1].kind == "mora":
                previous = tokens[-1]
                tokens[-1] = replace(
                    previous,
                    original_text=previous.original_text + character,
                    kana=previous.kana + character,
                )
            else:
                tokens.append(MoraToken(original_text=character, kana=character))
            continue
        if _is_kana(character) or character in _KANA_ITERATION_MARKS:
            tokens.append(MoraToken(original_text=character, kana=character))
            continue
        raise ValueError(
            f"unsupported character {character!r}; provide a kana reading before tokenization"
        )
    return tokens


def _round_fraction(value: Fraction) -> int:
    if value >= 0:
        return (2 * value.numerator + value.denominator) // (2 * value.denominator)
    return -_round_fraction(-value)


def _grid_step(sample_rate: int, bpm: float, subdivisions_per_beat: int) -> Fraction:
    bpm_fraction = Fraction(Decimal(str(bpm)))
    return Fraction(sample_rate * 60, 1) / (bpm_fraction * subdivisions_per_beat)


def _exact_grid_position(
    grid_index: int,
    *,
    beat_offset_sample: int,
    step: Fraction,
) -> Fraction:
    return Fraction(beat_offset_sample, 1) + grid_index * step


def _grid_sample(
    grid_index: int,
    *,
    beat_offset_sample: int,
    step: Fraction,
) -> int:
    return _round_fraction(
        _exact_grid_position(
            grid_index,
            beat_offset_sample=beat_offset_sample,
            step=step,
        )
    )


def _candidate_indices(
    anchor: VocalTimingAnchor,
    *,
    beat_offset_sample: int,
    step: Fraction,
    config: VocalGridConfig,
) -> tuple[int, ...]:
    centers = {
        _round_fraction(Fraction(anchor.refined_sample - beat_offset_sample, 1) / step),
        _round_fraction(Fraction(anchor.aligned_sample - beat_offset_sample, 1) / step),
    }
    candidates: set[int] = set()
    for center in centers:
        for index in range(center - config.candidate_radius, center + config.candidate_radius + 1):
            if not config.allow_pickup and index < 0:
                continue
            sample = _grid_sample(index, beat_offset_sample=beat_offset_sample, step=step)
            if sample < 0:
                continue
            distance = float(abs(Fraction(anchor.refined_sample - sample, 1) / step))
            if distance > config.max_snap_distance_steps:
                continue
            candidates.add(index)
    return tuple(sorted(candidates))


def _emission_cost(
    anchor: VocalTimingAnchor,
    grid_index: int,
    *,
    beat_offset_sample: int,
    step: Fraction,
    config: VocalGridConfig,
) -> float:
    position = _exact_grid_position(
        grid_index,
        beat_offset_sample=beat_offset_sample,
        step=step,
    )
    refined_error = float(abs(Fraction(anchor.refined_sample, 1) - position) / step)
    aligned_error = float(abs(Fraction(anchor.aligned_sample, 1) - position) / step)
    confidence_scale = 0.35 + 0.65 * anchor.confidence
    return confidence_scale * (
        config.refined_weight * refined_error**2 + config.aligned_weight * aligned_error**2
    )


def _drop_cost(anchor: VocalTimingAnchor, config: VocalGridConfig) -> float:
    return config.unassigned_penalty * (0.20 + 0.80 * anchor.confidence)


def _update_state(
    states: dict[tuple[int | None, int | None], _DpState],
    state: _DpState,
) -> None:
    key = (state.last_grid_index, state.last_anchor_position)
    previous = states.get(key)
    if previous is None or state.cost < previous.cost - 1e-12:
        states[key] = state


def _prune_states(
    states: dict[tuple[int | None, int | None], _DpState],
    max_states: int,
) -> dict[tuple[int | None, int | None], _DpState]:
    if len(states) <= max_states:
        return states
    ranked = sorted(
        states.items(),
        key=lambda item: (
            item[1].cost,
            item[1].last_grid_index if item[1].last_grid_index is not None else -(10**18),
        ),
    )
    return dict(ranked[:max_states])


def _has_silence_between(
    silence_prefix: list[int],
    previous_original_index: int,
    current_original_index: int,
) -> bool:
    return silence_prefix[current_original_index] > silence_prefix[previous_original_index + 1]


def align_vocal_anchors_to_grid(
    anchors: list[VocalTimingAnchor],
    *,
    sample_rate: int,
    bpm: float,
    beat_offset_sample: int,
    config: VocalGridConfig | None = None,
) -> list[VocalTimingAnchor]:
    """Map a monotonic lyric phrase to a BPM grid with sentence-level DP.

    The default four subdivisions per beat are a 1/16-note grid in 4/4. Grid
    indices are computed directly from exact rational steps, including negative
    indices for pickup notes, so long phrases do not accumulate floating drift.
    Strictly increasing assigned indices prevent same-cell duplicate attacks. If
    two anchors cannot share a cell, the DP moves one to a plausible neighbor or
    leaves the weaker one unassigned. Silence and sustained-vowel anchors never
    consume a rhythmic cell by default.
    """

    resolved_config = config or VocalGridConfig()
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not math.isfinite(bpm) or bpm <= 0.0:
        raise ValueError("bpm must be a finite positive value")
    if not isinstance(beat_offset_sample, int):
        raise ValueError("beat_offset_sample must be an integer")
    if any(
        current.refined_sample < previous.refined_sample
        for previous, current in zip(anchors, anchors[1:], strict=False)
    ):
        raise ValueError("anchors must be monotonic by refined_sample")
    if not anchors:
        return []

    step = _grid_step(sample_rate, bpm, resolved_config.subdivisions_per_beat)
    quantized_kind_set = set(resolved_config.quantized_kinds)
    active: list[tuple[int, VocalTimingAnchor]] = [
        (index, anchor) for index, anchor in enumerate(anchors) if anchor.kind in quantized_kind_set
    ]
    if not active:
        return [replace(anchor, grid_sample=None) for anchor in anchors]

    silence_prefix = [0]
    for anchor in anchors:
        silence_prefix.append(silence_prefix[-1] + int(anchor.kind == "silence"))

    initial = _DpState(
        cost=0.0,
        last_grid_index=None,
        last_anchor_position=None,
        previous=None,
        assignment=None,
    )
    states: dict[tuple[int | None, int | None], _DpState] = {(None, None): initial}

    for active_position, (original_index, anchor) in enumerate(active):
        next_states: dict[tuple[int | None, int | None], _DpState] = {}
        candidates = _candidate_indices(
            anchor,
            beat_offset_sample=beat_offset_sample,
            step=step,
            config=resolved_config,
        )
        for state in states.values():
            _update_state(
                next_states,
                _DpState(
                    cost=state.cost + _drop_cost(anchor, resolved_config),
                    last_grid_index=state.last_grid_index,
                    last_anchor_position=state.last_anchor_position,
                    previous=state,
                    assignment=None,
                ),
            )
            for candidate in candidates:
                if state.last_grid_index is not None and candidate <= state.last_grid_index:
                    continue
                transition_cost = 0.0
                if state.last_grid_index is not None and state.last_anchor_position is not None:
                    previous_original_index, previous_anchor = active[state.last_anchor_position]
                    if not _has_silence_between(
                        silence_prefix,
                        previous_original_index,
                        original_index,
                    ):
                        expected_gap = float(
                            Fraction(
                                anchor.refined_sample - previous_anchor.refined_sample,
                                1,
                            )
                            / step
                        )
                        actual_gap = candidate - state.last_grid_index
                        transition_cost = resolved_config.transition_weight * (
                            abs(actual_gap - expected_gap) / max(1.0, expected_gap)
                        )
                _update_state(
                    next_states,
                    _DpState(
                        cost=(
                            state.cost
                            + _emission_cost(
                                anchor,
                                candidate,
                                beat_offset_sample=beat_offset_sample,
                                step=step,
                                config=resolved_config,
                            )
                            + transition_cost
                        ),
                        last_grid_index=candidate,
                        last_anchor_position=active_position,
                        previous=state,
                        assignment=candidate,
                    ),
                )
        states = _prune_states(next_states, resolved_config.max_states)

    best = min(states.values(), key=lambda state: state.cost)
    assignments: list[int | None] = []
    cursor = best
    while cursor.previous is not None:
        assignments.append(cursor.assignment)
        cursor = cursor.previous
    assignments.reverse()
    if len(assignments) != len(active):
        raise RuntimeError("vocal timing DP produced an invalid assignment path")

    grid_by_original_index = {
        original_index: (
            None
            if grid_index is None
            else _grid_sample(
                grid_index,
                beat_offset_sample=beat_offset_sample,
                step=step,
            )
        )
        for (original_index, _anchor), grid_index in zip(active, assignments, strict=True)
    }
    return [
        replace(anchor, grid_sample=grid_by_original_index.get(index))
        for index, anchor in enumerate(anchors)
    ]
