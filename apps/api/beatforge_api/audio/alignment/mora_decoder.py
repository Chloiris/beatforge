"""Dynamic-programming HuBERT phoneme to Japanese mora decoder.

The decoder is intentionally downstream of HuBERT.  It may aggregate raw and
refined boundaries from observed phoneme children, but it never derives time
from character count, lyric length, tempo, or an evenly divided parent span.
An expected mora with no observed phoneme child is reported as missing rather
than receiving a fabricated interval.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import Field, model_validator

from ...schemas import ApiModel
from .lyric_processor import ProcessedLyrics, map_phoneme_sequences_dp
from .schema import AlignmentHierarchy, AlignmentHierarchyUnit

MoraMatchOperation = Literal["match", "substitute", "delete"]


class MoraParentCharacter(ApiModel):
    """Occurrence-stable display character provenance for one decoded mora."""

    id: str
    index: int = Field(ge=0)
    text: str
    kana: str
    source_start: int = Field(ge=0)
    source_end: int = Field(ge=0)
    occurrence: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_source_span(self) -> MoraParentCharacter:
        if self.source_end <= self.source_start:
            raise ValueError("parent character source span must have positive length")
        return self


class MoraEvent(ApiModel):
    """One expected mora supported by one or more observed HuBERT phone spans."""

    id: str
    index: int = Field(ge=0)
    plan_mora_index: int = Field(ge=0)
    text: str
    character: str
    mora: str
    kana: str
    kind: str
    parent_characters: list[MoraParentCharacter]
    character_indices: list[int]
    phonemes: list[str]
    expected_phonemes: list[str]
    expected_phoneme_indices: list[int]
    observed_phoneme_indices: list[int]
    mapping_operations: list[MoraMatchOperation]
    aligned_start_sample: int = Field(ge=0)
    aligned_end_sample: int = Field(ge=0)
    refined_start_sample: int = Field(ge=0)
    refined_end_sample: int = Field(ge=0)
    aligned_sample: int = Field(ge=0)
    refined_sample: int = Field(ge=0)
    start_sample: int = Field(ge=0)
    end_sample: int = Field(ge=0)
    confidence: float = Field(ge=0, le=1)
    matched_phone_count: int = Field(ge=1)
    expected_phone_count: int = Field(ge=1)
    mapping_cost: float = Field(ge=0)
    boundary_provenance: Literal["observed_hubert_phoneme_children"] = (
        "observed_hubert_phoneme_children"
    )

    @model_validator(mode="before")
    @classmethod
    def populate_public_span_aliases(cls, value: Any) -> Any:
        """Backfill additive public fields from their authoritative provenance.

        Historical v0.6.1 artifacts predate these convenience fields. Loading
        them remains safe because every value is reconstructed from the saved
        parent-character and refined observed-phone boundaries, never from text
        length or tempo.
        """

        if not isinstance(value, Mapping):
            return value
        payload = dict(value)
        mora_label = str(payload.get("mora", payload.get("kana", "")))
        # MoraEvent.text is the playable mora label. The display/source text is
        # retained losslessly by character + parentCharacters.
        payload["text"] = mora_label
        parents = payload.get("parent_characters", payload.get("parentCharacters", []))

        def parent_text(parent: Any) -> str:
            if isinstance(parent, Mapping):
                return str(parent.get("text", ""))
            return str(getattr(parent, "text", ""))

        payload.setdefault("character", "".join(parent_text(parent) for parent in parents))
        if "start_sample" not in payload and "startSample" not in payload:
            payload["start_sample"] = payload.get(
                "refined_start_sample", payload.get("refinedStartSample")
            )
        if "end_sample" not in payload and "endSample" not in payload:
            payload["end_sample"] = payload.get(
                "refined_end_sample", payload.get("refinedEndSample")
            )
        return payload

    @model_validator(mode="after")
    def validate_observed_span(self) -> MoraEvent:
        if not self.parent_characters or not self.character_indices:
            raise ValueError("a mora event requires at least one parent character")
        if not self.phonemes or not self.observed_phoneme_indices:
            raise ValueError("a mora event requires at least one observed HuBERT phoneme")
        if self.aligned_end_sample <= self.aligned_start_sample:
            raise ValueError("raw HuBERT mora span must have positive duration")
        if self.refined_end_sample <= self.refined_start_sample:
            raise ValueError("refined mora span must have positive duration")
        if self.aligned_sample != self.aligned_start_sample:
            raise ValueError("aligned_sample must preserve the raw child start")
        if self.refined_sample != self.refined_start_sample:
            raise ValueError("refined_sample must preserve the refined child start")
        expected_character = "".join(parent.text for parent in self.parent_characters)
        if self.character != expected_character:
            raise ValueError("character must match the saved parent character occurrences")
        if self.text != self.mora:
            raise ValueError("text must expose the playable mora label")
        if self.start_sample != self.refined_start_sample:
            raise ValueError("start_sample must expose the refined observed-phone boundary")
        if self.end_sample != self.refined_end_sample:
            raise ValueError("end_sample must expose the refined observed-phone boundary")
        if self.matched_phone_count != len(self.observed_phoneme_indices):
            raise ValueError("matched_phone_count must equal observed phoneme count")
        if self.expected_phone_count != len(self.expected_phoneme_indices):
            raise ValueError("expected_phone_count must equal expected phoneme count")
        if len(self.mapping_operations) != self.expected_phone_count:
            raise ValueError("every expected phoneme requires a DP operation")
        for values in (
            self.character_indices,
            self.expected_phoneme_indices,
            self.observed_phoneme_indices,
        ):
            if values != sorted(set(values)):
                raise ValueError("mora provenance indices must be sorted and unique")
        return self


class MoraDecodeResult(ApiModel):
    events: list[MoraEvent]
    expected_mora_count: int = Field(ge=0)
    decoded_mora_count: int = Field(ge=0)
    coverage: float = Field(ge=0, le=1)
    missing_mora_indices: list[int]
    inserted_observed_phoneme_indices: list[int]
    deleted_expected_phoneme_indices: list[int]
    total_dp_cost: float = Field(ge=0)
    mapping_algorithm: Literal["global_phoneme_edit_distance_dp"] = (
        "global_phoneme_edit_distance_dp"
    )
    timestamp_provenance: Literal[
        "min/max of DP-matched observed HuBERT phoneme child spans"
    ] = "min/max of DP-matched observed HuBERT phoneme child spans"
    even_duration_allocation: Literal[False] = False
    text_length_timing: Literal[False] = False

    @model_validator(mode="after")
    def validate_counts(self) -> MoraDecodeResult:
        if self.decoded_mora_count != len(self.events):
            raise ValueError("decoded_mora_count must equal event count")
        if self.expected_mora_count != self.decoded_mora_count + len(
            self.missing_mora_indices
        ):
            raise ValueError("decoded and missing morae must partition the expected plan")
        expected_coverage = (
            self.decoded_mora_count / self.expected_mora_count
            if self.expected_mora_count
            else 0.0
        )
        if not math.isclose(self.coverage, expected_coverage, abs_tol=1e-12):
            raise ValueError("coverage must be derived from decoded expected morae")
        return self


def _observed_phones(
    hierarchy: AlignmentHierarchy | Sequence[AlignmentHierarchyUnit],
) -> list[AlignmentHierarchyUnit]:
    phones = list(hierarchy.phonemes if isinstance(hierarchy, AlignmentHierarchy) else hierarchy)
    if not phones:
        raise ValueError("Mora Decoder requires observed HuBERT phoneme hierarchy units")
    if any(unit.level != "phoneme" or not unit.phoneme for unit in phones):
        raise ValueError("Mora Decoder input must contain only labelled phoneme units")
    if any(
        current.aligned_start_sample < previous.aligned_start_sample
        or current.refined_start_sample < previous.refined_start_sample
        for previous, current in zip(phones, phones[1:], strict=False)
    ):
        raise ValueError("observed HuBERT phoneme units must be monotonic")
    return phones


def _parent_characters(
    plan: ProcessedLyrics,
    character_indices: Sequence[int],
) -> list[MoraParentCharacter]:
    parents: list[MoraParentCharacter] = []
    for index in character_indices:
        if index < 0 or index >= len(plan.characters):
            raise ValueError("expected mora references a character outside the lyric plan")
        character = plan.characters[index]
        parents.append(
            MoraParentCharacter(
                id=character.id,
                index=character.index,
                text=character.text,
                kana=character.kana,
                source_start=character.source_start,
                source_end=character.source_end,
                occurrence=character.occurrence,
            )
        )
    return parents


def _confidence(
    children: Sequence[AlignmentHierarchyUnit],
    operations: Sequence[MoraMatchOperation],
    expected_count: int,
) -> float:
    observed_mean = sum(unit.confidence for unit in children) / len(children)
    coverage = len(children) / expected_count
    substitutions = sum(operation == "substitute" for operation in operations)
    substitution_factor = max(0.0, 1.0 - 0.25 * substitutions / expected_count)
    return float(min(1.0, max(0.0, observed_mean * coverage * substitution_factor)))


def decode_moras(
    observed_hierarchy: AlignmentHierarchy | Sequence[AlignmentHierarchyUnit],
    expected_plan: ProcessedLyrics,
) -> MoraDecodeResult:
    """Decode expected morae from observed HuBERT phonemes using global DP.

    Only DP-matched child units contribute boundaries.  Partial morae retain a
    lower confidence; wholly missing morae appear in ``missing_mora_indices``
    and do not produce a timestamped event.
    """

    observed = _observed_phones(observed_hierarchy)
    mapping = map_phoneme_sequences_dp(
        expected_plan.phone_sequence,
        [str(unit.phoneme) for unit in observed],
    )
    match_by_expected = {item.expected_index: item for item in mapping.matches}
    events: list[MoraEvent] = []
    missing_moras: list[int] = []
    deleted_phones = sorted(
        item.expected_index for item in mapping.matches if item.observed_index is None
    )

    for mora in expected_plan.moras:
        expected_indices = list(mora.phoneme_indices)
        if not expected_indices:
            raise ValueError("expected lyric plan contains a mora without phonemes")
        mora_matches = [match_by_expected[index] for index in expected_indices]
        matched = [item for item in mora_matches if item.observed_index is not None]
        if not matched:
            missing_moras.append(mora.index)
            continue
        observed_positions = [
            item.observed_index for item in matched if item.observed_index is not None
        ]
        children = [observed[position] for position in observed_positions]
        operations = [item.operation for item in mora_matches]
        parent_characters = _parent_characters(expected_plan, mora.character_indices)
        aligned_start = min(unit.aligned_start_sample for unit in children)
        aligned_end = max(unit.aligned_end_sample for unit in children)
        refined_start = min(unit.refined_start_sample for unit in children)
        refined_end = max(unit.refined_end_sample for unit in children)
        events.append(
            MoraEvent(
                id=f"mora-event:{mora.id}",
                index=len(events),
                plan_mora_index=mora.index,
                text=mora.kana,
                mora=mora.kana,
                kana=mora.kana,
                kind=mora.kind,
                parent_characters=parent_characters,
                character_indices=list(mora.character_indices),
                phonemes=[str(unit.phoneme) for unit in children],
                expected_phonemes=[
                    expected_plan.phonemes[index].phoneme for index in expected_indices
                ],
                expected_phoneme_indices=expected_indices,
                observed_phoneme_indices=sorted(unit.index for unit in children),
                mapping_operations=operations,
                aligned_start_sample=aligned_start,
                aligned_end_sample=aligned_end,
                refined_start_sample=refined_start,
                refined_end_sample=refined_end,
                aligned_sample=aligned_start,
                refined_sample=refined_start,
                confidence=_confidence(children, operations, len(expected_indices)),
                matched_phone_count=len(children),
                expected_phone_count=len(expected_indices),
                mapping_cost=sum(item.cost for item in mora_matches),
            )
        )

    expected_count = len(expected_plan.moras)
    decoded_count = len(events)
    return MoraDecodeResult(
        events=events,
        expected_mora_count=expected_count,
        decoded_mora_count=decoded_count,
        coverage=decoded_count / expected_count if expected_count else 0.0,
        missing_mora_indices=missing_moras,
        inserted_observed_phoneme_indices=list(mapping.inserted_observed_indices),
        deleted_expected_phoneme_indices=deleted_phones,
        total_dp_cost=mapping.cost,
    )


class MoraDecoder:
    """Object-oriented façade for pipelines that keep decoder instances."""

    def decode(
        self,
        observed_hierarchy: AlignmentHierarchy | Sequence[AlignmentHierarchyUnit],
        expected_plan: ProcessedLyrics,
    ) -> MoraDecodeResult:
        return decode_moras(observed_hierarchy, expected_plan)


__all__ = [
    "MoraDecodeResult",
    "MoraDecoder",
    "MoraEvent",
    "MoraParentCharacter",
    "decode_moras",
]
