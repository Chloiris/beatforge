from __future__ import annotations

import pytest

from beatforge_api.audio.alignment.lyric_processor import (
    LyricCharacter,
    LyricMora,
    LyricPhoneme,
    ProcessedLyrics,
)
from beatforge_api.audio.alignment.mora_decoder import MoraDecoder, MoraEvent, decode_moras
from beatforge_api.audio.alignment.schema import (
    AlignmentAcousticEvidence,
    AlignmentHierarchyUnit,
)


def _plan() -> ProcessedLyrics:
    source = "声キャーっ 声キャーっ"
    source_positions = (0, 1, 2, 3, 4, 6, 7, 8, 9, 10)
    phone_groups = (
        (("k", "o"), (0,), 0, "コ"),
        (("e",), (0,), 1, "エ"),
        (("ky", "a"), (1, 2), 2, "キャ"),
        (("a",), (3,), 3, "ー"),
        (("cl",), (4,), 4, "ッ"),
        (("k", "o"), (5,), 5, "コ"),
        (("e",), (5,), 6, "エ"),
        (("ky", "a"), (6, 7), 7, "キャ"),
        (("a",), (8,), 8, "ー"),
        (("cl",), (9,), 9, "ッ"),
    )
    mora_kinds = (
        "mora",
        "mora",
        "yoon",
        "long_vowel",
        "sokuon",
        "mora",
        "mora",
        "yoon",
        "long_vowel",
        "sokuon",
    )
    phonemes: list[LyricPhoneme] = []
    moras: list[LyricMora] = []
    mora_phone_indices: list[tuple[int, ...]] = []
    for mora_index, (symbols, character_indices, _, kana) in enumerate(phone_groups):
        indices: list[int] = []
        for symbol in symbols:
            index = len(phonemes)
            indices.append(index)
            phonemes.append(
                LyricPhoneme(
                    id=f"phoneme-{index}",
                    index=index,
                    phoneme=symbol,
                    text="".join(
                        source[source_positions[item]] for item in character_indices
                    ),
                    kana=kana,
                    frontend_index=0 if mora_index < 5 else 1,
                    mora_index=mora_index,
                    character_indices=character_indices,
                )
            )
        mora_phone_indices.append(tuple(indices))
        moras.append(
            LyricMora(
                id=f"mora-{mora_index}",
                index=mora_index,
                text="".join(
                    source[source_positions[item]] for item in character_indices
                ),
                kana=kana,
                kind=mora_kinds[mora_index],  # type: ignore[arg-type]
                character_indices=character_indices,
                phoneme_indices=tuple(indices),
            )
        )

    character_text = ("声", "キ", "ャ", "ー", "っ", "声", "キ", "ャ", "ー", "っ")
    character_kana = ("コエ", "キャ", "キャ", "ー", "ッ") * 2
    character_moras = ((0, 1), (2,), (2,), (3,), (4,), (5, 6), (7,), (7,), (8,), (9,))
    characters: list[LyricCharacter] = []
    occurrences: dict[str, int] = {}
    for index, text in enumerate(character_text):
        occurrence = occurrences.get(text, 0)
        occurrences[text] = occurrence + 1
        mora_indices = character_moras[index]
        phone_indices = tuple(
            phone
            for mora_index in mora_indices
            for phone in mora_phone_indices[mora_index]
        )
        characters.append(
            LyricCharacter(
                id=f"character-{source_positions[index]}-{occurrence}",
                index=index,
                text=text,
                kana=character_kana[index],
                source_start=source_positions[index],
                source_end=source_positions[index] + 1,
                occurrence=occurrence,
                mora_indices=mora_indices,
                phoneme_indices=phone_indices,
            )
        )
    return ProcessedLyrics(
        source_text=source,
        spoken_text=source,
        characters=tuple(characters),
        moras=tuple(moras),
        phonemes=tuple(phonemes),
        annotations=(),
        projections=(),
        g2p_engine="test-japanese-g2p",
        ruby_policy="test plan",
    )


def _observed(plan: ProcessedLyrics, *, insert_noise: bool = False) -> list[AlignmentHierarchyUnit]:
    symbols = list(plan.phone_sequence)
    if insert_noise:
        symbols.insert(4, "x")
    starts = [100 + index * 137 + (index % 3) * 23 for index in range(len(symbols))]
    units: list[AlignmentHierarchyUnit] = []
    for index, (symbol, start) in enumerate(zip(symbols, starts, strict=True)):
        duration = 31 + index % 5 * 11
        units.append(
            AlignmentHierarchyUnit(
                id=f"observed-{index}",
                index=index,
                level="phoneme",
                text="observed",
                kana="オ",
                mora="オ",
                phoneme=symbol,
                character_indices=[0],
                mora_indices=[0],
                phoneme_indices=[index],
                aligned_start_sample=start,
                aligned_end_sample=start + duration,
                refined_start_sample=start + 3 + index % 2,
                refined_end_sample=start + duration + 7,
                aligned_sample=start,
                refined_sample=start + 3 + index % 2,
                confidence=0.92 - index * 0.01,
                observed_token_index=index,
                match_operation="match",
                evidence=AlignmentAcousticEvidence(
                    energy=0.7,
                    spectral_change=0.6,
                    pitch_change=0.4,
                ),
            )
        )
    return units


def test_decoder_preserves_observed_spans_and_japanese_hierarchy() -> None:
    plan = _plan()
    observed = _observed(plan, insert_noise=True)

    result = MoraDecoder().decode(observed, plan)

    assert result.expected_mora_count == result.decoded_mora_count == 10
    assert result.coverage == 1.0
    assert result.missing_mora_indices == []
    assert result.inserted_observed_phoneme_indices == [4]
    assert result.mapping_algorithm == "global_phoneme_edit_distance_dp"
    assert result.even_duration_allocation is False
    assert result.text_length_timing is False
    assert {event.kind for event in result.events} >= {"yoon", "long_vowel", "sokuon"}

    first_yoon = result.events[2]
    assert first_yoon.kana == "キャ"
    assert first_yoon.character == "キャ"
    assert [parent.text for parent in first_yoon.parent_characters] == ["キ", "ャ"]
    assert first_yoon.phonemes == ["ky", "a"]
    assert first_yoon.boundary_provenance == "observed_hubert_phoneme_children"

    repeated = [event for event in result.events if event.kana == "コ"]
    assert len(repeated) == 2
    assert repeated[0].id != repeated[1].id
    assert repeated[0].parent_characters[0].occurrence == 0
    assert repeated[1].parent_characters[0].occurrence == 1
    assert repeated[0].parent_characters[0].source_start == 0
    assert repeated[1].parent_characters[0].source_start == 6

    by_index = {unit.index: unit for unit in observed}
    all_raw_boundaries = {
        boundary
        for unit in observed
        for boundary in (unit.aligned_start_sample, unit.aligned_end_sample)
    }
    all_refined_boundaries = {
        boundary
        for unit in observed
        for boundary in (unit.refined_start_sample, unit.refined_end_sample)
    }
    for event in result.events:
        assert event.text == event.mora
        children = [by_index[index] for index in event.observed_phoneme_indices]
        assert event.aligned_start_sample == min(unit.aligned_start_sample for unit in children)
        assert event.aligned_end_sample == max(unit.aligned_end_sample for unit in children)
        assert event.refined_start_sample == min(unit.refined_start_sample for unit in children)
        assert event.refined_end_sample == max(unit.refined_end_sample for unit in children)
        assert event.start_sample == event.refined_start_sample
        assert event.end_sample == event.refined_end_sample
        assert event.aligned_start_sample in all_raw_boundaries
        assert event.aligned_end_sample in all_raw_boundaries
        assert event.refined_start_sample in all_refined_boundaries
        assert event.refined_end_sample in all_refined_boundaries

    serialized = first_yoon.model_dump(by_alias=True)
    assert serialized["character"] == "キャ"
    assert serialized["startSample"] == first_yoon.refined_start_sample
    assert serialized["endSample"] == first_yoon.refined_end_sample
    assert serialized["parentCharacters"][0]["sourceStart"] == 1
    assert serialized["alignedSample"] == first_yoon.aligned_start_sample
    assert serialized["refinedSample"] == first_yoon.refined_start_sample

    legacy_payload = dict(serialized)
    legacy_payload.pop("character")
    legacy_payload.pop("startSample")
    legacy_payload.pop("endSample")
    restored = MoraEvent.model_validate(legacy_payload)
    assert (restored.character, restored.start_sample, restored.end_sample) == (
        first_yoon.character,
        first_yoon.start_sample,
        first_yoon.end_sample,
    )


def test_wholly_unobserved_mora_is_missing_without_fabricated_time() -> None:
    plan = _plan()
    observed = _observed(plan)
    removed = observed.pop()

    result = decode_moras(observed, plan)

    assert removed.phoneme == "cl"
    assert result.expected_mora_count == 10
    assert result.decoded_mora_count == 9
    assert result.coverage == pytest.approx(0.9)
    assert result.missing_mora_indices == [9]
    assert result.deleted_expected_phoneme_indices == [13]
    assert all(event.plan_mora_index != 9 for event in result.events)
    assert all(
        event.aligned_end_sample <= observed[-1].aligned_end_sample
        for event in result.events
    )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("character", "別"),
        ("startSample", 0),
        ("endSample", 999_999),
    ],
)
def test_mora_event_rejects_public_fields_that_contradict_provenance(
    field: str,
    invalid_value: str | int,
) -> None:
    event = decode_moras(_observed(_plan()), _plan()).events[0]
    payload = event.model_dump(by_alias=True)
    payload[field] = invalid_value

    with pytest.raises(ValueError):
        MoraEvent.model_validate(payload)


def test_substitution_is_retained_with_dp_operation_and_lower_confidence() -> None:
    plan = _plan()
    exact_observed = _observed(plan)
    substituted_observed = list(exact_observed)
    unit = substituted_observed[3]
    substituted_observed[3] = unit.model_copy(update={"phoneme": "sh"})

    exact = decode_moras(exact_observed, plan)
    substituted = decode_moras(substituted_observed, plan)

    assert substituted.total_dp_cost == pytest.approx(1.0)
    assert substituted.events[2].mapping_operations == ["substitute", "match"]
    assert substituted.events[2].confidence < exact.events[2].confidence
    assert substituted.events[2].aligned_start_sample == exact.events[2].aligned_start_sample
    assert substituted.events[2].refined_end_sample == exact.events[2].refined_end_sample


def test_decoder_rejects_non_phoneme_or_reverse_order_input() -> None:
    plan = _plan()
    observed = _observed(plan)
    non_phone = observed[0].model_copy(update={"level": "mora"})
    with pytest.raises(ValueError, match="only labelled phoneme"):
        decode_moras([non_phone, *observed[1:]], plan)

    reversed_start = observed[1].model_copy(
        update={
            "aligned_start_sample": 0,
            "aligned_sample": 0,
            "refined_start_sample": 1,
            "refined_sample": 1,
        }
    )
    with pytest.raises(ValueError, match="must be monotonic"):
        decode_moras([observed[0], reversed_start, *observed[2:]], plan)
