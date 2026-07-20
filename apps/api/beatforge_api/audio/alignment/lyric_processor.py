"""Japanese lyric normalization and hierarchy planning for HuBERT CTC.

The module is deliberately dependency-light.  ``pyopenjtalk`` is imported only
when :class:`OpenJTalkBackend` is constructed, so the API process can deserialize
a plan produced by the isolated CTC runtime without importing the native G2P
extension itself.

No function in this module assigns time.  Character/mora/phone relationships are
found with dynamic programming and are later joined to observed CTC spans.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

MoraKind = Literal["mora", "yoon", "sokuon", "nasal", "long_vowel"]
PhoneMapOperation = Literal["match", "substitute", "delete"]

_SMALL_KANA = frozenset(
    "ゃゅょぁぃぅぇぉゎャュョァィゥェォヮ"
    "ㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇺㇻㇼㇽㇾㇿ"
)
_YOON_KANA = frozenset("ゃゅょャュョ")
_SOKUON = frozenset("っッ")
_NASAL = frozenset("んン")
_LONG_VOWEL = frozenset("ー")
_IGNORED_PHONES = frozenset({"pau", "sil"})


class LyricProcessingError(RuntimeError):
    """A deterministic lyric/G2P plan could not be constructed."""


class JapaneseG2PBackend(Protocol):
    """Small interface used by the pure planning algorithms and their tests."""

    @property
    def engine_name(self) -> str: ...

    def frontend(self, text: str) -> list[dict[str, Any]]: ...

    def kana(self, text: str) -> str: ...

    def phones(self, text: str) -> tuple[str, ...]: ...


class OpenJTalkBackend:
    """Real Japanese G2P backed by the locally installed ``pyopenjtalk``."""

    def __init__(self) -> None:
        try:
            import pyopenjtalk
        except ImportError as error:
            raise LyricProcessingError(
                "pyopenjtalk is unavailable; run lyric processing in the CTC runtime"
            ) from error
        self._module = pyopenjtalk

    @property
    def engine_name(self) -> str:
        version = getattr(self._module, "__version__", "unknown")
        return f"pyopenjtalk-{version}"

    def frontend(self, text: str) -> list[dict[str, Any]]:
        try:
            result = self._module.run_frontend(text)
        except Exception as error:  # native extension has no stable exception hierarchy
            raise LyricProcessingError(
                f"OpenJTalk frontend failed: {type(error).__name__}"
            ) from error
        return [item for item in result if isinstance(item, dict)]

    def kana(self, text: str) -> str:
        try:
            return str(self._module.g2p(text, kana=True))
        except Exception as error:
            raise LyricProcessingError(
                f"OpenJTalk kana G2P failed: {type(error).__name__}"
            ) from error

    def phones(self, text: str) -> tuple[str, ...]:
        try:
            raw = str(self._module.g2p(text, kana=False))
        except Exception as error:
            raise LyricProcessingError(
                f"OpenJTalk phoneme G2P failed: {type(error).__name__}"
            ) from error
        return tuple(phone for phone in raw.split() if phone not in _IGNORED_PHONES)


@dataclass(frozen=True, slots=True)
class RubyAnnotation:
    base: str
    reading: str
    source_start: int
    source_end: int
    notation_end: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "base": self.base,
            "reading": self.reading,
            "sourceStart": self.source_start,
            "sourceEnd": self.source_end,
            "notationEnd": self.notation_end,
        }


@dataclass(frozen=True, slots=True)
class TextProjection:
    spoken_start: int
    spoken_end: int
    source_start: int
    source_end: int
    display_text: str
    ruby_index: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "spokenStart": self.spoken_start,
            "spokenEnd": self.spoken_end,
            "sourceStart": self.source_start,
            "sourceEnd": self.source_end,
            "displayText": self.display_text,
            "rubyIndex": self.ruby_index,
        }


@dataclass(frozen=True, slots=True)
class PronunciationText:
    source_text: str
    spoken_text: str
    annotations: tuple[RubyAnnotation, ...]
    projections: tuple[TextProjection, ...]
    ruby_policy: str = "base(kana) is sung as kana once; base remains display text"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sourceText": self.source_text,
            "spokenText": self.spoken_text,
            "annotations": [item.to_dict() for item in self.annotations],
            "projections": [item.to_dict() for item in self.projections],
            "rubyPolicy": self.ruby_policy,
        }


@dataclass(frozen=True, slots=True)
class MoraPiece:
    kana: str
    kind: MoraKind


@dataclass(frozen=True, slots=True)
class LyricCharacter:
    id: str
    index: int
    text: str
    kana: str
    source_start: int
    source_end: int
    occurrence: int
    mora_indices: tuple[int, ...]
    phoneme_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "index": self.index,
            "text": self.text,
            "kana": self.kana,
            "sourceStart": self.source_start,
            "sourceEnd": self.source_end,
            "occurrence": self.occurrence,
            "moraIndices": list(self.mora_indices),
            "phonemeIndices": list(self.phoneme_indices),
        }


@dataclass(frozen=True, slots=True)
class LyricMora:
    id: str
    index: int
    text: str
    kana: str
    kind: MoraKind
    character_indices: tuple[int, ...]
    phoneme_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "index": self.index,
            "text": self.text,
            "kana": self.kana,
            "kind": self.kind,
            "characterIndices": list(self.character_indices),
            "phonemeIndices": list(self.phoneme_indices),
        }


@dataclass(frozen=True, slots=True)
class LyricPhoneme:
    id: str
    index: int
    phoneme: str
    text: str
    kana: str
    frontend_index: int
    mora_index: int
    character_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "index": self.index,
            "phoneme": self.phoneme,
            "text": self.text,
            "kana": self.kana,
            "frontendIndex": self.frontend_index,
            "moraIndex": self.mora_index,
            "characterIndices": list(self.character_indices),
        }


@dataclass(frozen=True, slots=True)
class ProcessedLyrics:
    source_text: str
    spoken_text: str
    characters: tuple[LyricCharacter, ...]
    moras: tuple[LyricMora, ...]
    phonemes: tuple[LyricPhoneme, ...]
    annotations: tuple[RubyAnnotation, ...]
    projections: tuple[TextProjection, ...]
    g2p_engine: str
    ruby_policy: str
    warnings: tuple[str, ...] = ()

    @property
    def phone_sequence(self) -> tuple[str, ...]:
        return tuple(item.phoneme for item in self.phonemes)

    def ctc_plan(self) -> dict[str, Any]:
        targets = [
            {
                "targetIndex": item.index,
                "surface": item.text,
                "phoneme": item.phoneme,
                "frontendIndex": item.frontend_index,
                "moraIndex": item.mora_index,
                "characterIndices": list(item.character_indices),
            }
            for item in self.phonemes
        ]
        hash_payload = {
            "version": 1,
            "spokenText": self.spoken_text,
            "targets": targets,
        }
        encoded = json.dumps(
            hash_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return {
            **hash_payload,
            "planHash": hashlib.sha256(encoded).hexdigest(),
            "g2pEngine": self.g2p_engine,
            "rubyPolicy": self.ruby_policy,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "sourceText": self.source_text,
            "spokenText": self.spoken_text,
            "characters": [item.to_dict() for item in self.characters],
            "moras": [item.to_dict() for item in self.moras],
            "phonemes": [item.to_dict() for item in self.phonemes],
            "annotations": [item.to_dict() for item in self.annotations],
            "projections": [item.to_dict() for item in self.projections],
            "g2pEngine": self.g2p_engine,
            "rubyPolicy": self.ruby_policy,
            "warnings": list(self.warnings),
            "ctcPlan": self.ctc_plan(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProcessedLyrics:
        try:
            characters = tuple(
                LyricCharacter(
                    id=str(item["id"]),
                    index=int(item["index"]),
                    text=str(item["text"]),
                    kana=str(item["kana"]),
                    source_start=int(item["sourceStart"]),
                    source_end=int(item["sourceEnd"]),
                    occurrence=int(item["occurrence"]),
                    mora_indices=tuple(int(value) for value in item["moraIndices"]),
                    phoneme_indices=tuple(int(value) for value in item["phonemeIndices"]),
                )
                for item in payload["characters"]
            )
            moras = tuple(
                LyricMora(
                    id=str(item["id"]),
                    index=int(item["index"]),
                    text=str(item["text"]),
                    kana=str(item["kana"]),
                    kind=str(item["kind"]),  # type: ignore[arg-type]
                    character_indices=tuple(int(value) for value in item["characterIndices"]),
                    phoneme_indices=tuple(int(value) for value in item["phonemeIndices"]),
                )
                for item in payload["moras"]
            )
            phonemes = tuple(
                LyricPhoneme(
                    id=str(item["id"]),
                    index=int(item["index"]),
                    phoneme=str(item["phoneme"]),
                    text=str(item["text"]),
                    kana=str(item["kana"]),
                    frontend_index=int(item["frontendIndex"]),
                    mora_index=int(item["moraIndex"]),
                    character_indices=tuple(
                        int(value) for value in item["characterIndices"]
                    ),
                )
                for item in payload["phonemes"]
            )
            annotations = tuple(
                RubyAnnotation(
                    base=str(item["base"]),
                    reading=str(item["reading"]),
                    source_start=int(item["sourceStart"]),
                    source_end=int(item["sourceEnd"]),
                    notation_end=int(item["notationEnd"]),
                )
                for item in payload.get("annotations", [])
            )
            projections = tuple(
                TextProjection(
                    spoken_start=int(item["spokenStart"]),
                    spoken_end=int(item["spokenEnd"]),
                    source_start=int(item["sourceStart"]),
                    source_end=int(item["sourceEnd"]),
                    display_text=str(item["displayText"]),
                    ruby_index=(
                        int(item["rubyIndex"]) if item.get("rubyIndex") is not None else None
                    ),
                )
                for item in payload.get("projections", [])
            )
        except (KeyError, TypeError, ValueError) as error:
            raise LyricProcessingError("serialized lyric plan is invalid") from error
        return cls(
            source_text=str(payload["sourceText"]),
            spoken_text=str(payload["spokenText"]),
            characters=characters,
            moras=moras,
            phonemes=phonemes,
            annotations=annotations,
            projections=projections,
            g2p_engine=str(payload["g2pEngine"]),
            ruby_policy=str(payload["rubyPolicy"]),
            warnings=tuple(str(value) for value in payload.get("warnings", [])),
        )


@dataclass(frozen=True, slots=True)
class MoraPhoneMapping:
    assignments: tuple[tuple[int, ...], ...]
    cost: float


@dataclass(frozen=True, slots=True)
class CharacterMoraGroup:
    character_indices: tuple[int, ...]
    mora_indices: tuple[int, ...]
    cost: float


@dataclass(frozen=True, slots=True)
class CharacterMoraMapping:
    assignments: tuple[tuple[int, ...], ...]
    groups: tuple[CharacterMoraGroup, ...]
    cost: float


@dataclass(frozen=True, slots=True)
class PhonemeMatch:
    expected_index: int
    observed_index: int | None
    operation: PhoneMapOperation
    cost: float


@dataclass(frozen=True, slots=True)
class PhonemeSequenceMapping:
    matches: tuple[PhonemeMatch, ...]
    inserted_observed_indices: tuple[int, ...]
    cost: float


def _is_han(character: str) -> bool:
    name = unicodedata.name(character, "")
    return "CJK UNIFIED IDEOGRAPH" in name or character in "々〆ヵヶ"


def _is_kana(character: str) -> bool:
    name = unicodedata.name(character, "")
    return (
        "HIRAGANA LETTER" in name
        or "KATAKANA LETTER" in name
        or character in _LONG_VOWEL
    )


def _ruby_base_start(text: str, opening: int) -> int | None:
    """Find a conservative ``kanji + optional okurigana`` base before ``(``."""

    index = opening - 1
    saw_han = False
    while index >= 0:
        character = text[index]
        if _is_han(character):
            saw_han = True
            index -= 1
            continue
        if _is_kana(character) and not saw_han:
            index -= 1
            continue
        break
    return index + 1 if saw_han else None


def preprocess_ruby(text: str) -> PronunciationText:
    """Replace ``base(kana)`` notation with its one sung kana pronunciation.

    Both ASCII and full-width parentheses are accepted.  A base must contain a
    Han character, which avoids treating ordinary parenthetical kana as ruby.
    Source offsets always refer to the NFC-normalized saved lyrics.
    """

    source = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    annotations: list[RubyAnnotation] = []
    cursor = 0
    while cursor < len(source):
        opening = next(
            (index for index in range(cursor, len(source)) if source[index] in "(（"),
            None,
        )
        if opening is None:
            break
        closing_character = ")" if source[opening] == "(" else "）"
        closing = source.find(closing_character, opening + 1)
        if closing < 0:
            cursor = opening + 1
            continue
        reading = source[opening + 1 : closing]
        base_start = _ruby_base_start(source, opening)
        if base_start is None or not reading or not all(_is_kana(char) for char in reading):
            cursor = closing + 1
            continue
        annotations.append(
            RubyAnnotation(
                base=source[base_start:opening],
                reading=reading,
                source_start=base_start,
                source_end=opening,
                notation_end=closing + 1,
            )
        )
        cursor = closing + 1

    spoken_parts: list[str] = []
    projections: list[TextProjection] = []
    source_cursor = 0
    spoken_cursor = 0

    def append_normal(start: int, end: int) -> None:
        nonlocal spoken_cursor
        if end <= start:
            return
        value = source[start:end]
        spoken_parts.append(value)
        projections.append(
            TextProjection(
                spoken_start=spoken_cursor,
                spoken_end=spoken_cursor + len(value),
                source_start=start,
                source_end=end,
                display_text=value,
            )
        )
        spoken_cursor += len(value)

    for annotation_index, annotation in enumerate(annotations):
        append_normal(source_cursor, annotation.source_start)
        spoken_parts.append(annotation.reading)
        projections.append(
            TextProjection(
                spoken_start=spoken_cursor,
                spoken_end=spoken_cursor + len(annotation.reading),
                source_start=annotation.source_start,
                source_end=annotation.source_end,
                display_text=annotation.base,
                ruby_index=annotation_index,
            )
        )
        spoken_cursor += len(annotation.reading)
        source_cursor = annotation.notation_end
    append_normal(source_cursor, len(source))
    return PronunciationText(
        source_text=source,
        spoken_text="".join(spoken_parts),
        annotations=tuple(annotations),
        projections=tuple(projections),
    )


def tokenize_moras(kana: str) -> tuple[MoraPiece, ...]:
    """Parse kana into explicit morae, retaining long vowels and sokuon."""

    result: list[MoraPiece] = []
    for character in unicodedata.normalize("NFC", kana):
        if character in _LONG_VOWEL:
            result.append(MoraPiece(character, "long_vowel"))
        elif character in _SOKUON:
            result.append(MoraPiece(character, "sokuon"))
        elif character in _NASAL:
            result.append(MoraPiece(character, "nasal"))
        elif character in _SMALL_KANA and result and result[-1].kind in {"mora", "yoon"}:
            previous = result[-1]
            result[-1] = MoraPiece(
                previous.kana + character,
                "yoon" if character in _YOON_KANA else previous.kind,
            )
        elif _is_kana(character):
            result.append(MoraPiece(character, "mora"))
    return tuple(result)


def _hiragana(value: str) -> str:
    converted: list[str] = []
    for character in unicodedata.normalize("NFKC", value):
        code = ord(character)
        converted.append(chr(code - 0x60) if 0x30A1 <= code <= 0x30F6 else character)
    return "".join(converted)


def _phone_key(phone: str) -> str:
    return phone.casefold() if phone in {"I", "U"} else phone


def _edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    previous = list(range(len(right) + 1))
    for row, left_value in enumerate(left, start=1):
        current = [row]
        for column, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def _mora_phone_hints(
    moras: Sequence[MoraPiece],
    backend: JapaneseG2PBackend,
) -> tuple[tuple[str, ...], ...]:
    hints: list[tuple[str, ...]] = []
    previous_vowel = "u"
    for mora in moras:
        if mora.kind == "long_vowel":
            phones = (previous_vowel,)
        elif mora.kind == "sokuon":
            phones = ("cl",)
        elif mora.kind == "nasal":
            phones = ("N",)
        else:
            phones = backend.phones(mora.kana)
        for phone in reversed(phones):
            if _phone_key(phone) in {"a", "i", "u", "e", "o"}:
                previous_vowel = _phone_key(phone)
                break
        hints.append(tuple(_phone_key(phone) for phone in phones))
    return tuple(hints)


def map_phones_to_moras_dp(
    moras: Sequence[MoraPiece],
    phones: Sequence[str],
    backend: JapaneseG2PBackend,
) -> MoraPhoneMapping:
    """Partition a real G2P phone sequence over morae with dynamic programming."""

    if not moras:
        if phones:
            raise LyricProcessingError("phones cannot be mapped without morae")
        return MoraPhoneMapping((), 0.0)
    hints = _mora_phone_hints(moras, backend)
    normalized = tuple(_phone_key(phone) for phone in phones)
    states: dict[tuple[int, int], tuple[float, tuple[tuple[int, ...], ...]]] = {
        (0, 0): (0.0, ())
    }
    for mora_index in range(len(moras)):
        next_states: dict[tuple[int, int], tuple[float, tuple[tuple[int, ...], ...]]] = {}
        for (completed, phone_start), (cost, assignments) in states.items():
            if completed != mora_index:
                continue
            remaining = len(normalized) - phone_start
            counts = set(range(0, min(4, remaining) + 1))
            counts.add(remaining)
            for count in sorted(counts):
                if len(moras) - mora_index - 1 > remaining - count:
                    continue
                segment = normalized[phone_start : phone_start + count]
                hint = hints[mora_index]
                distance = _edit_distance(hint, segment)
                local = distance / max(1, len(hint), len(segment))
                if count == 0:
                    local += 2.0
                local += 0.02 * abs(count - len(hint))
                key = (mora_index + 1, phone_start + count)
                candidate = (
                    cost + local,
                    assignments + (tuple(range(phone_start, phone_start + count)),),
                )
                current = next_states.get(key)
                if current is None or candidate[0] < current[0] - 1e-12:
                    next_states[key] = candidate
        states = next_states
    terminal = states.get((len(moras), len(normalized)))
    if terminal is None:
        raise LyricProcessingError("dynamic-programming mora/phone mapping failed")
    return MoraPhoneMapping(assignments=terminal[1], cost=terminal[0])


def map_characters_to_moras_dp(
    characters: Sequence[str],
    moras: Sequence[MoraPiece],
    backend: JapaneseG2PBackend,
) -> CharacterMoraMapping:
    """Map display characters to reading morae using lexical G2P candidates and DP."""

    if not characters or not moras:
        return CharacterMoraMapping(tuple(() for _ in characters), (), 0.0)
    target = tuple(_hiragana(item.kana) for item in moras)
    states: dict[
        tuple[int, int], tuple[float, tuple[CharacterMoraGroup, ...]]
    ] = {(0, 0): (0.0, ())}
    char_count = len(characters)
    mora_count = len(moras)
    for _ in range(char_count + mora_count):
        changed = False
        for (char_start, mora_start), (cost, groups) in list(states.items()):
            if char_start >= char_count or mora_start >= mora_count:
                continue
            char_ends = set(range(char_start + 1, min(char_count, char_start + 6) + 1))
            char_ends.add(char_count)
            mora_ends = set(range(mora_start + 1, min(mora_count, mora_start + 9) + 1))
            mora_ends.add(mora_count)
            for char_end in sorted(char_ends):
                surface = "".join(characters[char_start:char_end])
                candidate_kana = backend.kana(surface)
                candidate = tuple(_hiragana(item.kana) for item in tokenize_moras(candidate_kana))
                for mora_end in sorted(mora_ends):
                    observed = target[mora_start:mora_end]
                    distance = _edit_distance(candidate, observed)
                    local = distance / max(1, len(candidate), len(observed))
                    local += 0.025 * (char_end - char_start - 1)
                    local += 0.015 * (mora_end - mora_start - 1)
                    key = (char_end, mora_end)
                    group = CharacterMoraGroup(
                        character_indices=tuple(range(char_start, char_end)),
                        mora_indices=tuple(range(mora_start, mora_end)),
                        cost=local,
                    )
                    value = (cost + local, groups + (group,))
                    current = states.get(key)
                    if current is None or value[0] < current[0] - 1e-12:
                        states[key] = value
                        changed = True
        if not changed:
            break
    terminal = states.get((char_count, mora_count))
    if terminal is None:
        raise LyricProcessingError("dynamic-programming character/mora mapping failed")
    assignments: list[set[int]] = [set() for _ in characters]
    for group in terminal[1]:
        refined = _refine_character_group_dp(
            characters,
            moras,
            group,
            backend,
        )
        for character_index, mora_indices in refined.items():
            assignments[character_index].update(mora_indices)
    return CharacterMoraMapping(
        assignments=tuple(tuple(sorted(values)) for values in assignments),
        groups=terminal[1],
        cost=terminal[0],
    )


def _refine_character_group_dp(
    characters: Sequence[str],
    moras: Sequence[MoraPiece],
    group: CharacterMoraGroup,
    backend: JapaneseG2PBackend,
) -> dict[int, tuple[int, ...]]:
    """Use a second monotonic DP to localize morae inside one lexical group.

    The outer DP discovers the correct multi-character lexical reading.  This
    inner DP prevents every character in that word from receiving the whole
    word interval. Small kana remain attached to their preceding base kana.
    """

    atomic_characters: list[tuple[int, ...]] = []
    for character_index in group.character_indices:
        character = characters[character_index]
        if character in _SMALL_KANA and atomic_characters:
            atomic_characters[-1] = (*atomic_characters[-1], character_index)
        else:
            atomic_characters.append((character_index,))
    target_moras = tuple(group.mora_indices)
    if not atomic_characters or len(target_moras) < len(atomic_characters):
        return {
            character_index: target_moras
            for character_index in group.character_indices
        }

    states: dict[tuple[int, int], tuple[float, tuple[tuple[int, ...], ...]]] = {
        (0, 0): (0.0, ())
    }
    for atom_index, atom in enumerate(atomic_characters):
        next_states: dict[
            tuple[int, int], tuple[float, tuple[tuple[int, ...], ...]]
        ] = {}
        for (completed, mora_offset), (cost, allocations) in states.items():
            if completed != atom_index:
                continue
            remaining_atoms = len(atomic_characters) - atom_index - 1
            maximum = len(target_moras) - mora_offset - remaining_atoms
            surface = "".join(characters[index] for index in atom)
            hint = tuple(
                _hiragana(item.kana)
                for item in tokenize_moras(backend.kana(surface))
            )
            for count in range(1, maximum + 1):
                observed_indices = target_moras[mora_offset : mora_offset + count]
                observed = tuple(_hiragana(moras[index].kana) for index in observed_indices)
                distance = _edit_distance(hint, observed)
                local = distance / max(1, len(hint), len(observed))
                local += 0.01 * abs(len(hint) - len(observed))
                key = (atom_index + 1, mora_offset + count)
                candidate = (cost + local, allocations + (observed_indices,))
                current = next_states.get(key)
                if current is None or candidate[0] < current[0] - 1e-12:
                    next_states[key] = candidate
        states = next_states
    terminal = states.get((len(atomic_characters), len(target_moras)))
    if terminal is None:
        return {
            character_index: target_moras
            for character_index in group.character_indices
        }
    result: dict[int, tuple[int, ...]] = {}
    for atom, mora_indices in zip(atomic_characters, terminal[1], strict=True):
        for character_index in atom:
            result[character_index] = mora_indices
    return result


def map_phoneme_sequences_dp(
    expected: Sequence[str],
    observed: Sequence[str],
) -> PhonemeSequenceMapping:
    """Globally map expected G2P phones to observed CTC phones with edit-distance DP."""

    left = tuple(_phone_key(value) for value in expected)
    right = tuple(_phone_key(value) for value in observed)
    rows = len(left) + 1
    columns = len(right) + 1
    costs = [[math.inf] * columns for _ in range(rows)]
    traces: list[list[tuple[int, int, str] | None]] = [
        [None] * columns for _ in range(rows)
    ]
    costs[0][0] = 0.0
    for i in range(1, rows):
        costs[i][0] = costs[i - 1][0] + 1.1
        traces[i][0] = (i - 1, 0, "delete")
    for j in range(1, columns):
        costs[0][j] = costs[0][j - 1] + 1.0
        traces[0][j] = (0, j - 1, "insert")
    for i in range(1, rows):
        for j in range(1, columns):
            substitution = 0.0 if left[i - 1] == right[j - 1] else 1.0
            candidates = (
                (costs[i - 1][j - 1] + substitution, i - 1, j - 1, "diagonal"),
                (costs[i][j - 1] + 1.0, i, j - 1, "insert"),
                (costs[i - 1][j] + 1.1, i - 1, j, "delete"),
            )
            best = min(candidates, key=lambda item: (item[0], item[3] != "diagonal"))
            costs[i][j] = best[0]
            traces[i][j] = (best[1], best[2], best[3])

    matches: list[PhonemeMatch] = []
    inserted: list[int] = []
    i, j = len(left), len(right)
    while i or j:
        trace = traces[i][j]
        if trace is None:
            raise LyricProcessingError("phoneme DP trace is incomplete")
        previous_i, previous_j, operation = trace
        if operation == "diagonal":
            item_cost = 0.0 if left[i - 1] == right[j - 1] else 1.0
            matches.append(
                PhonemeMatch(
                    expected_index=i - 1,
                    observed_index=j - 1,
                    operation="match" if item_cost == 0.0 else "substitute",
                    cost=item_cost,
                )
            )
        elif operation == "delete":
            matches.append(PhonemeMatch(i - 1, None, "delete", 1.1))
        else:
            inserted.append(j - 1)
        i, j = previous_i, previous_j
    matches.reverse()
    inserted.reverse()
    return PhonemeSequenceMapping(tuple(matches), tuple(inserted), costs[-1][-1])


@dataclass(slots=True)
class _CharacterBuilder:
    index: int
    text: str
    source_start: int
    occurrence: int
    mora_indices: set[int]
    phoneme_indices: set[int]


def _recover_spoken_span(
    spoken: str,
    surface: str,
    cursor: int,
) -> tuple[int, int, bool]:
    candidate = unicodedata.normalize("NFKC", surface)
    if not candidate:
        return cursor, cursor, True
    search_limit = min(len(spoken), cursor + max(64, len(candidate) * 4))
    for start in range(cursor, search_limit):
        for end in range(start + 1, min(len(spoken), start + max(16, len(candidate) * 3)) + 1):
            normalized = unicodedata.normalize("NFKC", spoken[start:end])
            if normalized == candidate:
                return start, end, True
            if len(normalized) > len(candidate) + 2:
                break
    if candidate in "〇一二三四五六七八九十百千万億兆":
        for index in range(cursor, min(len(spoken), cursor + 16)):
            if spoken[index].isdigit():
                return index, index + 1, True
    return cursor, min(len(spoken), cursor + max(1, len(surface))), False


def _display_refs(
    pronunciation: PronunciationText,
    spoken_start: int,
    spoken_end: int,
) -> list[tuple[int, str]]:
    references: list[tuple[int, str]] = []
    seen: set[int] = set()
    for projection in pronunciation.projections:
        overlap_start = max(spoken_start, projection.spoken_start)
        overlap_end = min(spoken_end, projection.spoken_end)
        if overlap_end <= overlap_start:
            continue
        if projection.ruby_index is not None:
            values = [
                (projection.source_start + offset, character)
                for offset, character in enumerate(projection.display_text)
            ]
        else:
            offset_start = overlap_start - projection.spoken_start
            offset_end = overlap_end - projection.spoken_start
            values = [
                (projection.source_start + offset, projection.display_text[offset])
                for offset in range(offset_start, offset_end)
            ]
        for source_index, character in values:
            if source_index not in seen and not character.isspace():
                references.append((source_index, character))
                seen.add(source_index)
    return references


def process_japanese_lyrics(
    text: str,
    *,
    backend: JapaneseG2PBackend | None = None,
) -> ProcessedLyrics:
    """Create an occurrence-stable character→mora→phoneme G2P plan."""

    g2p = backend or OpenJTalkBackend()
    pronunciation = preprocess_ruby(text)
    nodes = g2p.frontend(pronunciation.spoken_text)
    builders: dict[int, _CharacterBuilder] = {}
    occurrence_counts: dict[str, int] = {}
    moras_out: list[LyricMora] = []
    phones_out: list[LyricPhoneme] = []
    warnings: list[str] = []
    cursor = 0

    for frontend_index, node in enumerate(nodes):
        surface = str(node.get("string") or "")
        reading = str(node.get("read") or "").strip()
        if not surface or not reading:
            continue
        spoken_start, spoken_end, recovered = _recover_spoken_span(
            pronunciation.spoken_text,
            surface,
            cursor,
        )
        cursor = max(cursor, spoken_end)
        if not recovered:
            warnings.append(f"frontend surface recovery fallback at node {frontend_index}")
        references = _display_refs(pronunciation, spoken_start, spoken_end)
        if not references:
            continue
        mora_pieces = tokenize_moras(reading)
        node_phones = g2p.phones(reading)
        if not mora_pieces or not node_phones:
            continue
        local_character_indices: list[int] = []
        for source_index, character in references:
            builder = builders.get(source_index)
            if builder is None:
                occurrence = occurrence_counts.get(character, 0)
                occurrence_counts[character] = occurrence + 1
                builder = _CharacterBuilder(
                    index=len(builders),
                    text=character,
                    source_start=source_index,
                    occurrence=occurrence,
                    mora_indices=set(),
                    phoneme_indices=set(),
                )
                builders[source_index] = builder
            local_character_indices.append(builder.index)

        character_mapping = map_characters_to_moras_dp(
            [character for _, character in references],
            mora_pieces,
            g2p,
        )
        phone_mapping = map_phones_to_moras_dp(mora_pieces, node_phones, g2p)
        global_mora_indices = tuple(
            range(len(moras_out), len(moras_out) + len(mora_pieces))
        )
        global_phone_indices = tuple(
            range(len(phones_out), len(phones_out) + len(node_phones))
        )
        mora_to_characters: list[set[int]] = [set() for _ in mora_pieces]
        for local_character, assigned_moras in enumerate(character_mapping.assignments):
            global_character = local_character_indices[local_character]
            builder = next(item for item in builders.values() if item.index == global_character)
            for local_mora in assigned_moras:
                global_mora = global_mora_indices[local_mora]
                builder.mora_indices.add(global_mora)
                mora_to_characters[local_mora].add(global_character)

        phone_to_mora: dict[int, int] = {}
        for local_mora, assigned_phones in enumerate(phone_mapping.assignments):
            for local_phone in assigned_phones:
                phone_to_mora[local_phone] = local_mora

        for local_phone, phone in enumerate(node_phones):
            local_mora = phone_to_mora[local_phone]
            character_indices = tuple(sorted(mora_to_characters[local_mora]))
            global_phone = global_phone_indices[local_phone]
            display = "".join(
                next(item.text for item in builders.values() if item.index == index)
                for index in character_indices
            )
            for index in character_indices:
                next(item for item in builders.values() if item.index == index).phoneme_indices.add(
                    global_phone
                )
            phones_out.append(
                LyricPhoneme(
                    id=f"phoneme-{global_phone}",
                    index=global_phone,
                    phoneme=phone,
                    text=display,
                    kana=mora_pieces[local_mora].kana,
                    frontend_index=frontend_index,
                    mora_index=global_mora_indices[local_mora],
                    character_indices=character_indices,
                )
            )

        for local_mora, piece in enumerate(mora_pieces):
            global_mora = global_mora_indices[local_mora]
            phone_indices = tuple(
                global_phone_indices[index]
                for index in phone_mapping.assignments[local_mora]
            )
            character_indices = tuple(sorted(mora_to_characters[local_mora]))
            display = "".join(
                next(item.text for item in builders.values() if item.index == index)
                for index in character_indices
            )
            moras_out.append(
                LyricMora(
                    id=f"mora-{global_mora}",
                    index=global_mora,
                    text=display or piece.kana,
                    kana=piece.kana,
                    kind=piece.kind,
                    character_indices=character_indices,
                    phoneme_indices=phone_indices,
                )
            )

    if not phones_out:
        raise LyricProcessingError("OpenJTalk produced no lexical phoneme targets")
    ordered_builders = sorted(builders.values(), key=lambda item: item.index)
    characters_out: list[LyricCharacter] = []
    for builder in ordered_builders:
        mora_indices = tuple(sorted(builder.mora_indices))
        kana = "".join(moras_out[index].kana for index in mora_indices)
        characters_out.append(
            LyricCharacter(
                id=f"character-{builder.source_start}-{builder.occurrence}",
                index=builder.index,
                text=builder.text,
                kana=kana,
                source_start=builder.source_start,
                source_end=builder.source_start + 1,
                occurrence=builder.occurrence,
                mora_indices=mora_indices,
                phoneme_indices=tuple(sorted(builder.phoneme_indices)),
            )
        )
    return ProcessedLyrics(
        source_text=pronunciation.source_text,
        spoken_text=pronunciation.spoken_text,
        characters=tuple(characters_out),
        moras=tuple(moras_out),
        phonemes=tuple(phones_out),
        annotations=pronunciation.annotations,
        projections=pronunciation.projections,
        g2p_engine=g2p.engine_name,
        ruby_policy=pronunciation.ruby_policy,
        warnings=tuple(warnings),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Japanese lyric G2P hierarchy plan")
    parser.add_argument("--lyrics-file", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        text = arguments.lyrics_file.read_text(encoding="utf-8")
        result = process_japanese_lyrics(text)
        _write_json(arguments.output, {"status": "ok", **result.to_dict()})
    except Exception as error:
        _write_json(
            arguments.output,
            {
                "status": "error",
                "errorCode": "LYRIC_PROCESSING_FAILED",
                "errorMessage": str(error),
                "exceptionType": type(error).__name__,
            },
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
