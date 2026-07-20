"""HuBERT hierarchy to vocal chart candidates and the v0.6.1 evaluation artifact.

The module is deliberately downstream of forced alignment.  It may rank or
quantize an already observed acoustic event, but it never creates an event from
lyrics, text length, or a tempo-grid cell.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import unicodedata
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import Field
from sqlalchemy import select
from sqlalchemy.orm import Session, selectinload

from ...database import SessionLocal
from ...models import CandidateEventModel, TrackModel
from ...schemas import ApiModel
from ...serialization import dumps
from ...timing import nearest_grid_sample
from .base import AlignmentContext, TempoReference
from .evaluator import COVERAGE_CONFIDENCE_THRESHOLD
from .mora_decoder import MoraDecodeResult, MoraEvent
from .schema import AlignmentReport, AlignmentResult

_GENERATOR = "hubert_ctc"
_EVENT_NAMESPACE = uuid.UUID("eab3e912-1647-43b0-a22c-4336ec84a601")
_RAP_DENSITY_PER_SECOND = 7.0
_NORMAL_ONSET_THRESHOLD = 0.14
_RAP_ONSET_THRESHOLD = 0.08
_LONG_VOWEL_MIN_PITCH_CHANGE = 0.35
_LONG_VOWEL_MIN_HUBERT_CONFIDENCE = COVERAGE_CONFIDENCE_THRESHOLD
_VOWEL_PHONEMES = frozenset({"a", "i", "u", "e", "o"})


class HubertCandidateEvidence(ApiModel):
    hubert: float = Field(ge=0, le=1)
    energy: float = Field(ge=0, le=1)
    spectral_change: float = Field(ge=0, le=1)
    pitch: float = Field(ge=0, le=1)
    rhythm: float = Field(ge=0, le=1)
    rap_density: float = Field(ge=0)
    rap_policy: float = Field(ge=0, le=1)
    onset_threshold: float = Field(ge=0, le=1)
    long_vowel_split: float = Field(ge=0, le=1)


class HubertCandidateEvent(ApiModel):
    id: str
    source: Literal["vocals"] = "vocals"
    generator: Literal["hubert_ctc"] = "hubert_ctc"
    character: str
    character_indices: list[int] = Field(default_factory=list)
    mora: str
    mora_index: int | None = Field(default=None, ge=0)
    alignment_unit_id: str | None = None
    phoneme: str
    phonemes: list[str] = Field(default_factory=list)
    aligned_sample: int = Field(ge=0)
    refined_sample: int = Field(ge=0)
    acoustic_sample: int = Field(ge=0)
    chart_sample: int = Field(ge=0)
    evidence: HubertCandidateEvidence
    confidence: float = Field(ge=0, le=1)
    status: Literal["accepted", "rejected", "uncertain"]
    policy: Literal[
        "mora",
        "character",
        "long_vowel_pitch_split",
        "legacy_phoneme",
    ] = "character"


class HubertCandidateBundle(ApiModel):
    run_id: str
    track_id: str
    sample_rate: int = Field(gt=0)
    sample_count: int = Field(gt=0)
    mora_events: list[MoraEvent] = Field(default_factory=list)
    events: list[HubertCandidateEvent]
    policy: dict[str, Any]
    created_at: datetime


class HubertMetrics(ApiModel):
    run_id: str
    character_coverage: float = Field(ge=0, le=1)
    mora_coverage: float = Field(ge=0, le=1)
    phoneme_coverage: float = Field(ge=0, le=1)
    forced_character_coverage: float = Field(ge=0, le=1)
    forced_mora_coverage: float = Field(ge=0, le=1)
    forced_phoneme_coverage: float = Field(ge=0, le=1)
    coverage_confidence_threshold: float = Field(ge=0, le=1)
    acoustic_consistency: float = Field(ge=0, le=1)
    rhythm_consistency: float = Field(ge=0, le=1)
    runtime_sec: float | None = Field(default=None, ge=0)
    runtime_source: Literal["ctc_metadata", "missing"]


class HubertAlignmentReport(ApiModel):
    schema_version: Literal["1.0"] = "1.0"
    track_id: str
    song: str
    artist: str
    sample_rate: int = Field(gt=0)
    sample_count: int = Field(gt=0)
    generated_at: datetime
    ground_truth: Literal["proxy_only"] = "proxy_only"
    hubert: HubertMetrics
    qwen_coverage: float | None = Field(default=None, ge=0, le=1)
    qwen_proxy_coverage: float | None = Field(default=None, ge=0, le=1)
    coverage_delta: float | None = Field(default=None, ge=-1, le=1)
    run_ids: dict[str, str | None]
    candidate_event_count: int = Field(ge=0)
    counts: dict[str, int]
    details: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class HubertArtifacts:
    candidates: HubertCandidateBundle
    report: HubertAlignmentReport


def _value(item: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(item, dict) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return default


def _items(item: Any, *names: str) -> list[Any]:
    value = _value(item, *names, default=[])
    if isinstance(value, list | tuple):
        return list(value)
    return []


def _unit_indices(item: Any, name: str) -> list[int]:
    camel = name.split("_")[0] + "".join(part.title() for part in name.split("_")[1:])
    return [
        int(value)
        for value in _items(item, name, camel)
        if isinstance(value, int) and value >= 0
    ]


def _bounded(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return float(np.clip(number if math.isfinite(number) else 0.0, 0.0, 1.0))


def _sample(item: Any, snake: str, camel: str, fallback: int | None = None) -> int | None:
    raw = _value(item, snake, camel, default=fallback)
    if raw is None:
        return None
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _unit_span_is_valid(item: Any, *, refined: bool = True) -> bool:
    prefix = "refined" if refined else "aligned"
    start = _sample(item, f"{prefix}_start_sample", f"{prefix}StartSample")
    end = _sample(item, f"{prefix}_end_sample", f"{prefix}EndSample")
    return start is not None and end is not None and end > start


def _phone_is_aligned(phone: Any) -> bool:
    operation = str(
        _value(phone, "match_operation", "matchOperation", default="") or ""
    ).casefold()
    observed = _value(phone, "observed_token_index", "observedTokenIndex")
    if operation in {"delete", "deletion", "unmatched", "missing"}:
        return False
    return observed is not None and _unit_span_is_valid(phone)


def _phone_evidence(phone: Any) -> tuple[float, float, float, float]:
    evidence = _value(phone, "evidence", default={})
    hubert = _bounded(_value(phone, "confidence", default=0.0))
    energy = _bounded(_value(evidence, "energy", "rms", default=0.0))
    spectral = _bounded(
        _value(evidence, "spectral_change", "spectralChange", default=0.0)
    )
    pitch = _bounded(_value(evidence, "pitch_change", "pitchChange", "pitch", default=0.0))
    return hubert, energy, spectral, pitch


def _aggregate_evidence(phones: list[Any]) -> tuple[float, float, float, float]:
    values = [_phone_evidence(phone) for phone in phones if _phone_is_aligned(phone)]
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    array = np.asarray(values, dtype=np.float64)
    # HuBERT confidence represents the full mapped pronunciation. Acoustic
    # changes are boundary cues, so their strongest observed boundary is the
    # useful evidence for a character event.
    return (
        float(np.mean(array[:, 0])),
        float(np.max(array[:, 1])),
        float(np.max(array[:, 2])),
        float(np.max(array[:, 3])),
    )


def _active_tempo(context: AlignmentContext, sample: int) -> TempoReference | None:
    if not context.tempo_map:
        return None
    return max(
        (item for item in context.tempo_map if item.start_sample <= sample),
        key=lambda item: item.start_sample,
        default=context.tempo_map[0],
    )


def _chart_position(context: AlignmentContext, sample: int) -> tuple[int, float, str]:
    tempo = _active_tempo(context, sample)
    if tempo is None:
        return sample, 0.0, "unsnapped"
    chart_sample = nearest_grid_sample(
        sample,
        sample_rate=context.sample_rate,
        bpm=tempo.bpm,
        beat_offset_sample=tempo.beat_offset_sample,
        subdivisions_per_beat=4,
    )
    chart_sample = min(max(chart_sample, 0), context.sample_count - 1)
    distance_ms = abs(sample - chart_sample) * 1000.0 / context.sample_rate
    rhythm = float(np.exp(-0.5 * (distance_ms / 30.0) ** 2))
    return chart_sample, rhythm, "straight_1_16"


def _density_at(samples: list[int], sample: int, sample_rate: int) -> float:
    if not samples:
        return 0.0
    radius = sample_rate * 0.5
    # The one-second window is based solely on observed acoustic locations.
    lower = sample - radius
    upper = sample + radius
    count = sum(lower <= value <= upper for value in samples)
    return float(count)


def _candidate_score(
    hubert: float,
    energy: float,
    spectral: float,
    pitch: float,
    rhythm: float,
    *,
    rap: bool,
) -> tuple[float, str, float]:
    if rap:
        score = 0.55 * hubert + 0.15 * energy + 0.15 * spectral + 0.05 * pitch + 0.10 * rhythm
        onset_threshold = _RAP_ONSET_THRESHOLD
    else:
        score = 0.38 * hubert + 0.22 * energy + 0.18 * spectral + 0.12 * pitch + 0.10 * rhythm
        onset_threshold = _NORMAL_ONSET_THRESHOLD
    onset = max(energy, spectral, pitch)
    if onset < onset_threshold and hubert < 0.75:
        status = "rejected"
    elif score >= 0.42:
        status = "accepted"
    elif score >= 0.28:
        status = "uncertain"
    else:
        status = "rejected"
    return float(np.clip(score, 0.0, 1.0)), status, onset_threshold


def _long_vowel_phone_indices(item: Any, moras: list[Any]) -> set[int]:
    """Return only phones owned by an explicit long-vowel mora.

    A character may own several moras (for example a Latin letter read in
    Japanese).  Looking only at the character text would allow an unrelated
    consonant later in that reading to masquerade as the sustained-vowel
    boundary, so the relation must come from the typed mora layer.
    """

    indices: set[int] = set()
    for index in _unit_indices(item, "mora_indices"):
        if index >= len(moras):
            continue
        mora = moras[index]
        kind = str(_value(mora, "kind", default="") or "").casefold()
        if kind in {"sustain", "long_vowel"}:
            indices.update(_unit_indices(mora, "phoneme_indices"))
    return indices


def _is_vowel_phone(phone: Any) -> bool:
    phoneme = str(_value(phone, "phoneme", default="") or "").casefold().strip()
    phoneme = phoneme.replace(":", "").replace("ː", "")
    return phoneme in _VOWEL_PHONEMES


def _qualified_unit_count(units: list[Any], phones: list[Any]) -> int:
    count = 0
    for unit in units:
        indices = _unit_indices(unit, "phoneme_indices")
        mapped = [phones[index] for index in indices if index < len(phones)]
        observed = [phone for phone in mapped if _phone_is_aligned(phone)]
        if not observed or len(observed) != len(mapped):
            continue
        mean_confidence = float(
            np.mean([_phone_evidence(phone)[0] for phone in observed])
        )
        if mean_confidence >= COVERAGE_CONFIDENCE_THRESHOLD:
            count += 1
    return count


def _canonical_character_key(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).casefold()


def _qwen_token_units(result: AlignmentResult) -> list[str]:
    units: list[str] = []
    for token in result.tokens:
        if token.confidence < COVERAGE_CONFIDENCE_THRESHOLD:
            continue
        for character in unicodedata.normalize("NFKC", token.text):
            if unicodedata.category(character)[0] in {"L", "N"}:
                units.append(character.casefold())
    return units


def _qualified_qwen_coverage(
    canonical_characters: list[Any],
    result: AlignmentResult,
) -> tuple[float, int, int]:
    expected = [
        _canonical_character_key(_value(unit, "text", default=""))
        for unit in canonical_characters
    ]
    observed = _qwen_token_units(result)
    if not expected:
        return 0.0, 0, len(observed)
    matcher = SequenceMatcher(a=expected, b=observed, autojunk=False)
    matched = sum(block.size for block in matcher.get_matching_blocks())
    return float(np.clip(matched / len(expected), 0.0, 1.0)), matched, len(observed)


def _event_id(track_id: str, unit_id: str, sample: int, policy: str) -> str:
    return str(uuid.uuid5(_EVENT_NAMESPACE, f"{track_id}:{unit_id}:{sample}:{policy}"))


def _event_for_unit(
    context: AlignmentContext,
    unit: Any,
    phones: list[Any],
    moras: list[Any],
    mora_samples: list[int],
    *,
    policy: Literal["character", "long_vowel_pitch_split"] = "character",
    override_phone: Any | None = None,
) -> HubertCandidateEvent | None:
    selected_phones = [override_phone] if override_phone is not None else phones
    aligned = _sample(unit, "aligned_sample", "alignedSample")
    refined = _sample(unit, "refined_sample", "refinedSample")
    if override_phone is not None:
        aligned = _sample(override_phone, "aligned_sample", "alignedSample", aligned)
        refined = _sample(override_phone, "refined_sample", "refinedSample", refined)
    if aligned is None or refined is None or refined >= context.sample_count:
        return None
    hubert, energy, spectral, pitch = _aggregate_evidence(selected_phones)
    if not selected_phones:
        hubert = _bounded(_value(unit, "confidence", default=0.0))
    chart_sample, rhythm, _grid_type = _chart_position(context, refined)
    density = _density_at(mora_samples, refined, context.sample_rate)
    rap = density >= _RAP_DENSITY_PER_SECOND
    confidence, status, onset_threshold = _candidate_score(
        hubert,
        energy,
        spectral,
        pitch,
        rhythm,
        rap=rap,
    )
    character = str(_value(unit, "text", default="") or "")
    if not character:
        return None
    mora_texts = [
        str(_value(moras[index], "kana", "mora", "text", default="") or "")
        for index in _unit_indices(unit, "mora_indices")
        if index < len(moras)
    ]
    phonemes = [
        str(_value(phone, "phoneme", default="") or "")
        for phone in selected_phones
    ]
    unit_id = str(_value(unit, "id", default=f"character-{_value(unit, 'index', default=0)}"))
    if override_phone is not None:
        unit_id += ":" + str(_value(override_phone, "id", default="split"))
    return HubertCandidateEvent(
        id=_event_id(context.track_id, unit_id, refined, policy),
        character=character,
        character_indices=_unit_indices(unit, "character_indices"),
        mora="".join(value for value in mora_texts if value),
        alignment_unit_id=str(_value(unit, "id", default=unit_id)),
        phoneme=" ".join(value for value in phonemes if value)
        or str(_value(unit, "phoneme", default="") or ""),
        phonemes=[value for value in phonemes if value],
        aligned_sample=aligned,
        refined_sample=refined,
        acoustic_sample=refined,
        chart_sample=chart_sample,
        evidence=HubertCandidateEvidence(
            hubert=hubert,
            energy=energy,
            spectral_change=spectral,
            pitch=pitch,
            rhythm=rhythm,
            rap_density=density,
            rap_policy=1.0 if rap else 0.0,
            onset_threshold=onset_threshold,
            long_vowel_split=1.0 if policy == "long_vowel_pitch_split" else 0.0,
        ),
        confidence=confidence,
        status=status,  # type: ignore[arg-type]
        policy=policy,
    )


def _event_for_mora(
    context: AlignmentContext,
    mora_event: MoraEvent,
    phones: list[Any],
    mora_samples: list[int],
    *,
    policy: Literal["mora", "long_vowel_pitch_split"] = "mora",
    override_phone: Any | None = None,
) -> HubertCandidateEvent | None:
    """Create a chart candidate from one decoded acoustic mora observation."""

    selected_phones = [override_phone] if override_phone is not None else phones
    if not selected_phones:
        return None
    aligned = mora_event.aligned_sample
    refined = mora_event.refined_sample
    if override_phone is not None:
        aligned = _sample(override_phone, "aligned_sample", "alignedSample", aligned)
        refined = _sample(override_phone, "refined_sample", "refinedSample", refined)
    if aligned is None or refined is None or refined >= context.sample_count:
        return None
    hubert, energy, spectral, pitch = _aggregate_evidence(selected_phones)
    chart_sample, rhythm, _grid_type = _chart_position(context, refined)
    density = _density_at(mora_samples, refined, context.sample_rate)
    rap = density >= _RAP_DENSITY_PER_SECOND
    confidence, status, onset_threshold = _candidate_score(
        hubert,
        energy,
        spectral,
        pitch,
        rhythm,
        rap=rap,
    )
    character = "".join(parent.text for parent in mora_event.parent_characters)
    if not character:
        return None
    unit_id = mora_event.id
    if override_phone is not None:
        unit_id += ":" + str(_value(override_phone, "id", default="split"))
    phonemes = [
        str(_value(phone, "phoneme", default="") or "")
        for phone in selected_phones
    ]
    return HubertCandidateEvent(
        id=_event_id(context.track_id, unit_id, refined, policy),
        character=character,
        character_indices=list(mora_event.character_indices),
        mora=mora_event.mora,
        mora_index=mora_event.plan_mora_index,
        alignment_unit_id=mora_event.id,
        phoneme=" ".join(value for value in phonemes if value),
        phonemes=[value for value in phonemes if value],
        aligned_sample=aligned,
        refined_sample=refined,
        acoustic_sample=refined,
        chart_sample=chart_sample,
        evidence=HubertCandidateEvidence(
            hubert=hubert,
            energy=energy,
            spectral_change=spectral,
            pitch=pitch,
            rhythm=rhythm,
            rap_density=density,
            rap_policy=1.0 if rap else 0.0,
            onset_threshold=onset_threshold,
            long_vowel_split=1.0 if policy == "long_vowel_pitch_split" else 0.0,
        ),
        confidence=confidence,
        status=status,  # type: ignore[arg-type]
        policy=policy,
    )


def _flat_token_events(
    context: AlignmentContext,
    result: AlignmentResult,
) -> list[HubertCandidateEvent]:
    events: list[HubertCandidateEvent] = []
    samples = [token.start_sample for token in result.tokens]
    for token in result.tokens:
        chart_sample, rhythm, _grid_type = _chart_position(context, token.start_sample)
        density = _density_at(samples, token.start_sample, context.sample_rate)
        rap = density >= _RAP_DENSITY_PER_SECOND
        confidence, status, onset_threshold = _candidate_score(
            token.confidence,
            0.0,
            0.0,
            0.0,
            rhythm,
            rap=rap,
        )
        events.append(
            HubertCandidateEvent(
                id=_event_id(
                    context.track_id,
                    token.id,
                    token.start_sample,
                    "legacy_phoneme",
                ),
                character=token.text,
                mora="",
                phoneme=token.phoneme or "",
                aligned_sample=token.start_sample,
                refined_sample=token.start_sample,
                acoustic_sample=token.start_sample,
                chart_sample=chart_sample,
                evidence=HubertCandidateEvidence(
                    hubert=token.confidence,
                    energy=0.0,
                    spectral_change=0.0,
                    pitch=0.0,
                    rhythm=rhythm,
                    rap_density=density,
                    rap_policy=1.0 if rap else 0.0,
                    onset_threshold=onset_threshold,
                    long_vowel_split=0.0,
                ),
                confidence=confidence,
                status=status,  # type: ignore[arg-type]
                policy="legacy_phoneme",
            )
        )
    return events


def _runtime(result: AlignmentResult) -> tuple[float | None, Literal["ctc_metadata", "missing"]]:
    for key in ("totalElapsedSec", "runtimeSec", "runtime_sec"):
        value = result.metadata.get(key)
        try:
            runtime = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(runtime) and runtime >= 0.0:
            return runtime, "ctc_metadata"
    return None, "missing"


def _decoded_mora_timeline(
    result: AlignmentResult,
    moras: list[Any],
    phones: list[Any],
) -> MoraDecodeResult | None:
    payload = result.metadata.get("moraDecoder")
    engine_version = str(result.metadata.get("engineVersion", "") or "")
    if payload is None:
        if engine_version.startswith("0.6.1"):
            raise ValueError("v0.6.1 HuBERT results require saved Mora Decoder output")
        return None
    try:
        decoded = MoraDecodeResult.model_validate(payload)
    except (TypeError, ValueError) as error:
        raise ValueError("HuBERT result contains invalid Mora Decoder output") from error
    if decoded.expected_mora_count != len(moras):
        raise ValueError("Mora Decoder expected count does not match the saved mora layer")
    if decoded.missing_mora_indices or decoded.decoded_mora_count != len(moras):
        raise ValueError("chart candidates require one observed MoraEvent per saved mora")
    event_by_index = {event.plan_mora_index: event for event in decoded.events}
    if sorted(event_by_index) != list(range(len(moras))):
        raise ValueError("Mora Decoder events must cover sequential saved mora indices")
    for index, mora in enumerate(moras):
        event = event_by_index[index]
        phone_indices = _unit_indices(mora, "phoneme_indices")
        if event.observed_phoneme_indices != phone_indices:
            raise ValueError("MoraEvent phone provenance differs from the saved hierarchy")
        if event.character_indices != _unit_indices(mora, "character_indices"):
            raise ValueError("MoraEvent character provenance differs from the saved hierarchy")
        if any(phone_index >= len(phones) for phone_index in phone_indices):
            raise ValueError("MoraEvent references a phone outside the saved hierarchy")
        if any(not _phone_is_aligned(phones[phone_index]) for phone_index in phone_indices):
            raise ValueError("MoraEvent must contain only observed HuBERT phone children")
        saved_bounds = (
            _sample(mora, "aligned_start_sample", "alignedStartSample"),
            _sample(mora, "aligned_end_sample", "alignedEndSample"),
            _sample(mora, "refined_start_sample", "refinedStartSample"),
            _sample(mora, "refined_end_sample", "refinedEndSample"),
        )
        event_bounds = (
            event.aligned_start_sample,
            event.aligned_end_sample,
            event.refined_start_sample,
            event.refined_end_sample,
        )
        if saved_bounds != event_bounds:
            raise ValueError("MoraEvent boundaries differ from observed hierarchy boundaries")
    return decoded


def _coverage(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def build_hubert_artifacts(
    context: AlignmentContext,
    result: AlignmentResult,
    *,
    qwen_result: AlignmentResult | None = None,
    qwen_report: AlignmentReport | None = None,
    created_at: datetime | None = None,
) -> HubertArtifacts:
    """Build candidates and proxy metrics from a completed real CTC result."""

    if result.method != "ctc" or result.status != "completed" or not result.tokens:
        raise ValueError("HuBERT artifacts require a completed non-empty CTC result")
    now = created_at or datetime.now(UTC)
    hierarchy = _value(result, "hierarchy")
    phones = _items(hierarchy, "phonemes") if hierarchy is not None else []
    moras = _items(hierarchy, "moras") if hierarchy is not None else []
    characters = _items(hierarchy, "characters") if hierarchy is not None else []
    hierarchy_mode = bool(phones or moras or characters)
    engine_version = str(result.metadata.get("engineVersion", "") or "")
    if engine_version.startswith("0.6") and not hierarchy_mode:
        raise ValueError("v0.6+ HuBERT results require a typed non-empty hierarchy")
    decoded_moras = _decoded_mora_timeline(result, moras, phones)
    mora_events = list(decoded_moras.events) if decoded_moras is not None else []
    mora_samples = (
        [event.refined_sample for event in mora_events]
        if mora_events
        else [
            sample
            for mora in moras
            if (sample := _sample(mora, "refined_sample", "refinedSample"))
            is not None
        ]
    )
    events: list[HubertCandidateEvent] = []
    if mora_events:
        for mora_event in mora_events:
            mapped_phone_pairs = [
                (index, phones[index])
                for index in mora_event.observed_phoneme_indices
                if index < len(phones) and _phone_is_aligned(phones[index])
            ]
            mapped_phones = [phone for _index, phone in mapped_phone_pairs]
            event = _event_for_mora(
                context,
                mora_event,
                mapped_phones,
                mora_samples,
            )
            if event is None:
                raise ValueError("an observed MoraEvent failed to create its base candidate")
            events.append(event)

            # Splits are optional and remain acoustic: an explicit long-vowel
            # mora needs another observed voiced-vowel boundary, HuBERT
            # confidence, pitch change and minimum elapsed duration.
            if mora_event.kind not in {"long_vowel", "sustain"} or len(mapped_phones) < 2:
                continue
            for _phone_index, phone in mapped_phone_pairs:
                if not _is_vowel_phone(phone):
                    continue
                hubert, _energy, _spectral, pitch = _phone_evidence(phone)
                split_sample = _sample(phone, "refined_sample", "refinedSample")
                if (
                    hubert < _LONG_VOWEL_MIN_HUBERT_CONFIDENCE
                    or pitch < _LONG_VOWEL_MIN_PITCH_CHANGE
                    or split_sample is None
                    or split_sample - event.refined_sample
                    < round(context.sample_rate * 0.040)
                ):
                    continue
                split = _event_for_mora(
                    context,
                    mora_event,
                    mapped_phones,
                    mora_samples,
                    policy="long_vowel_pitch_split",
                    override_phone=phone,
                )
                if split is not None:
                    events.append(split)
    else:
        # Read-only compatibility for historical v0.6.0 artifacts. New
        # v0.6.1 results are required above to use MoraEvent candidates.
        for character in characters:
            mapped_phone_pairs = [
                (index, phones[index])
                for index in _unit_indices(character, "phoneme_indices")
                if index < len(phones) and _phone_is_aligned(phones[index])
            ]
            mapped_phones = [phone for _index, phone in mapped_phone_pairs]
            if not mapped_phones:
                continue
            event = _event_for_unit(
                context,
                character,
                mapped_phones,
                moras,
                mora_samples,
            )
            if event is not None:
                events.append(event)
            long_vowel_phone_indices = _long_vowel_phone_indices(character, moras)
            if not long_vowel_phone_indices or len(mapped_phones) < 2:
                continue
            for phone_index, phone in mapped_phone_pairs:
                if (
                    phone_index not in long_vowel_phone_indices
                    or not _is_vowel_phone(phone)
                ):
                    continue
                hubert, _energy, _spectral, pitch = _phone_evidence(phone)
                split_sample = _sample(phone, "refined_sample", "refinedSample")
                base_sample = event.refined_sample if event is not None else None
                if (
                    hubert < _LONG_VOWEL_MIN_HUBERT_CONFIDENCE
                    or pitch < _LONG_VOWEL_MIN_PITCH_CHANGE
                    or split_sample is None
                    or base_sample is None
                    or split_sample - base_sample
                    < round(context.sample_rate * 0.040)
                ):
                    continue
                split = _event_for_unit(
                    context,
                    character,
                    mapped_phones,
                    moras,
                    mora_samples,
                    policy="long_vowel_pitch_split",
                    override_phone=phone,
                )
                if split is not None:
                    events.append(split)

    if not events and not hierarchy_mode:
        events = _flat_token_events(context, result)

    events.sort(key=lambda item: (item.acoustic_sample, item.id))
    base_mora_candidate_count = sum(event.policy == "mora" for event in events)
    long_vowel_split_count = sum(
        event.policy == "long_vowel_pitch_split" for event in events
    )
    if mora_events and base_mora_candidate_count != len(mora_events):
        raise ValueError("every decoded MoraEvent must produce exactly one base candidate")
    if mora_events and len(events) != len(mora_events) + long_vowel_split_count:
        raise ValueError("only observed long-vowel splits may add candidates beyond morae")
    forced_aligned_phone_count = sum(_phone_is_aligned(phone) for phone in phones)
    forced_aligned_mora_count = sum(
        _unit_span_is_valid(mora)
        and any(
            index < len(phones) and _phone_is_aligned(phones[index])
            for index in _unit_indices(mora, "phoneme_indices")
        )
        for mora in moras
    )
    forced_aligned_character_count = sum(
        _unit_span_is_valid(character)
        and any(
            index < len(phones) and _phone_is_aligned(phones[index])
            for index in _unit_indices(character, "phoneme_indices")
        )
        for character in characters
    )
    aligned_phone_count = sum(
        _phone_is_aligned(phone)
        and _phone_evidence(phone)[0] >= COVERAGE_CONFIDENCE_THRESHOLD
        for phone in phones
    )
    aligned_mora_count = _qualified_unit_count(moras, phones)
    aligned_character_count = _qualified_unit_count(characters, phones)
    phone_consistency = [
        0.45 * hubert + 0.25 * energy + 0.20 * spectral + 0.10 * pitch
        for phone in phones
        if _phone_is_aligned(phone)
        for hubert, energy, spectral, pitch in [_phone_evidence(phone)]
    ]
    acoustic_consistency = (
        float(np.clip(np.mean(phone_consistency), 0.0, 1.0))
        if phone_consistency
        else 0.0
    )
    rhythm_consistency = (
        float(np.mean([event.evidence.rhythm for event in events])) if events else 0.0
    )
    character_coverage = _coverage(aligned_character_count, len(characters))
    mora_coverage = _coverage(aligned_mora_count, len(moras))
    phoneme_coverage = _coverage(aligned_phone_count, len(phones))
    forced_character_coverage = _coverage(
        forced_aligned_character_count, len(characters)
    )
    forced_mora_coverage = _coverage(forced_aligned_mora_count, len(moras))
    forced_phoneme_coverage = _coverage(forced_aligned_phone_count, len(phones))
    runtime_sec, runtime_source = _runtime(result)
    qwen_proxy_coverage = qwen_report.coverage if qwen_report is not None else None
    qwen_matched_units: int | None = None
    qwen_observed_units: int | None = None
    if qwen_result is not None:
        qwen_coverage, qwen_matched_units, qwen_observed_units = (
            _qualified_qwen_coverage(characters, qwen_result)
        )
        coverage_delta = character_coverage - qwen_coverage
        qwen_coverage_source = (
            "confidence_qualified_qwen_tokens_mapped_to_hubert_canonical_characters"
        )
    else:
        qwen_coverage = qwen_proxy_coverage
        coverage_delta = None
        qwen_coverage_source = "legacy_proxy_fallback_not_comparable"

    candidates = HubertCandidateBundle(
        run_id=result.run_id,
        track_id=context.track_id,
        sample_rate=context.sample_rate,
        sample_count=context.sample_count,
        mora_events=mora_events,
        events=events,
        policy={
            "eventSource": (
                "DP-decoded MoraEvent; one base candidate per observed mora"
                if mora_events
                else "historical character/flat-token compatibility"
            ),
            "baseEventLayer": "mora" if mora_events else "character",
            "baseMoraEventCount": len(mora_events),
            "oneCandidatePerMora": bool(mora_events),
            "chartSamplePolicy": "nearest 1/16 recommendation for an existing event",
            "tempoCreatesEvents": False,
            "rapDensityThresholdPerSec": _RAP_DENSITY_PER_SECOND,
            "normalOnsetThreshold": _NORMAL_ONSET_THRESHOLD,
            "rapOnsetThreshold": _RAP_ONSET_THRESHOLD,
            "normalWeights": {
                "hubert": 0.38,
                "energy": 0.22,
                "spectralChange": 0.18,
                "pitch": 0.12,
                "rhythm": 0.10,
            },
            "rapWeights": {
                "hubert": 0.55,
                "energy": 0.15,
                "spectralChange": 0.15,
                "pitch": 0.05,
                "rhythm": 0.10,
            },
            "longVowelSplitPolicy": (
                "explicit long_vowel mora + voiced-vowel HuBERT boundary + "
                f"confidence >= {_LONG_VOWEL_MIN_HUBERT_CONFIDENCE:.2f} + "
                f"pitch change >= {_LONG_VOWEL_MIN_PITCH_CHANGE:.2f}"
            ),
            "hierarchyAvailable": hierarchy_mode,
            "legacyFlatTokenCompatibility": not hierarchy_mode,
        },
        created_at=now,
    )
    report = HubertAlignmentReport(
        track_id=context.track_id,
        song=context.song or context.track_id,
        artist=context.artist,
        sample_rate=context.sample_rate,
        sample_count=context.sample_count,
        generated_at=now,
        hubert=HubertMetrics(
            run_id=result.run_id,
            character_coverage=character_coverage,
            mora_coverage=mora_coverage,
            phoneme_coverage=phoneme_coverage,
            forced_character_coverage=forced_character_coverage,
            forced_mora_coverage=forced_mora_coverage,
            forced_phoneme_coverage=forced_phoneme_coverage,
            coverage_confidence_threshold=COVERAGE_CONFIDENCE_THRESHOLD,
            acoustic_consistency=acoustic_consistency,
            rhythm_consistency=rhythm_consistency,
            runtime_sec=runtime_sec,
            runtime_source=runtime_source,
        ),
        qwen_coverage=qwen_coverage,
        qwen_proxy_coverage=qwen_proxy_coverage,
        coverage_delta=coverage_delta,
        run_ids={
            "hubert": result.run_id,
            "qwen": qwen_result.run_id if qwen_result is not None else None,
        },
        candidate_event_count=len(events),
        counts={
            "characters": len(characters),
            "alignedCharacters": aligned_character_count,
            "forcedAlignedCharacters": forced_aligned_character_count,
            "moras": len(moras),
            "alignedMoras": aligned_mora_count,
            "forcedAlignedMoras": forced_aligned_mora_count,
            "phonemes": len(phones),
            "alignedPhonemes": aligned_phone_count,
            "forcedAlignedPhonemes": forced_aligned_phone_count,
            "moraEvents": len(mora_events),
            "baseMoraCandidates": base_mora_candidate_count,
            "longVowelSplits": long_vowel_split_count,
            "rapCandidates": sum(event.evidence.rap_policy == 1.0 for event in events),
        },
        details={
            "coverageBasis": (
                "typed lyric hierarchy units with mean mapped observed-phone "
                f"confidence >= {COVERAGE_CONFIDENCE_THRESHOLD:.2f}"
            ),
            "forcedTargetCoverage": {
                "characters": forced_character_coverage,
                "moras": forced_mora_coverage,
                "phonemes": forced_phoneme_coverage,
            },
            "acousticBasis": "HuBERT confidence + vocal RMS + spectral + pitch evidence",
            "rhythmBasis": "distance of existing acoustic candidates to the tempo grid",
            "qwenCoverageMetric": qwen_coverage_source,
            "qwenMatchedLyricUnits": qwen_matched_units,
            "qwenObservedLyricUnits": qwen_observed_units,
            "canonicalLyricUnitCount": len(characters),
            "canonicalLyricUnitSource": "hubert_typed_character_hierarchy",
            "qwenProxyCoverageMetric": "Alignment Lab full-text SequenceMatcher proxy",
            "coverageDeltaMetric": (
                "confidence-qualified HuBERT minus Qwen coverage over the same "
                "typed canonical character targets"
                if qwen_result is not None
                else "not comparable without the Qwen result"
            ),
            "coverageConfidenceThreshold": COVERAGE_CONFIDENCE_THRESHOLD,
            "chartCandidateSource": (
                "MoraEvent" if mora_events else "historical compatibility path"
            ),
            "oneBaseCandidatePerObservedMora": bool(mora_events),
            "moraDecoderMappingAlgorithm": (
                decoded_moras.mapping_algorithm if decoded_moras is not None else None
            ),
            "moraDecoderTimestampProvenance": (
                decoded_moras.timestamp_provenance if decoded_moras is not None else None
            ),
            "timestampsFabricated": False,
            "legacyFlatTokenCompatibility": not hierarchy_mode,
            "manualCorrections": False,
            "evenDurationAllocation": False,
        },
    )
    return HubertArtifacts(candidates=candidates, report=report)


def _candidate_model(event: HubertCandidateEvent, run_id: str) -> CandidateEventModel:
    evidence = event.evidence.model_dump(mode="json", by_alias=True)
    snap_error_ms = (event.acoustic_sample - event.chart_sample) * 1000.0
    # The persisted model receives sample_rate-aware snap_error_ms at publish
    # time; this placeholder is replaced before insertion.
    return CandidateEventModel(
        id=event.id,
        hit_point_id=None,
        sample=event.acoustic_sample,
        acoustic_sample=event.acoustic_sample,
        chart_sample=event.chart_sample,
        snap_error_ms=snap_error_ms,
        lane="vocals",
        source_evidence_json=dumps(
            {"vocals": 1.0, "melody": 0.0, "drums": 0.0, "mix": 0.0}
        ),
        semantic_evidence_json=dumps(
            {
                "lyricAlignment": event.evidence.hubert,
                "phonemeConfidence": event.evidence.hubert,
                "pitchConfidence": event.evidence.pitch,
                "beatConfidence": event.evidence.rhythm,
                "energyConfidence": event.evidence.energy,
                "spectralChangeConfidence": event.evidence.spectral_change,
                "rapDensity": event.evidence.rap_density,
                "longVowelSplit": event.evidence.long_vowel_split,
            }
        ),
        confidence=event.confidence,
        status=event.status,
        grid_type=(
            "straight_1_16"
            if event.chart_sample != event.acoustic_sample or event.evidence.rhythm > 0.0
            else "unsnapped"
        ),
        grid_confidence=event.evidence.rhythm,
        source=event.source,
        generator=event.generator,
        character=event.character,
        mora=event.mora,
        phoneme=event.phoneme,
        event_level=(
            "mora"
            if event.policy in {"mora", "long_vowel_pitch_split"}
            else event.policy
        ),
        event_policy=event.policy,
        alignment_unit_id=event.alignment_unit_id,
        alignment_unit_index=event.mora_index,
        alignment_run_id=run_id,
        character_indices_json=dumps(event.character_indices),
        phonemes_json=dumps(event.phonemes),
        aligned_sample=event.aligned_sample,
        refined_sample=event.refined_sample,
        evidence_json=dumps(evidence),
    )


def persist_hubert_candidates(
    bundle: HubertCandidateBundle,
    *,
    session: Session | None = None,
) -> int:
    """Upsert one HuBERT run without touching any other candidate generator."""

    owns_session = session is None
    active_session = session or SessionLocal()
    try:
        track = active_session.scalar(
            select(TrackModel)
            .where(TrackModel.id == bundle.track_id)
            .options(selectinload(TrackModel.candidate_events))
        )
        if track is None:
            return 0
        desired = {
            event.id: _candidate_model(event, bundle.run_id)
            for event in bundle.events
        }
        for model in desired.values():
            model.snap_error_ms = (
                (model.acoustic_sample - model.chart_sample)
                * 1000.0
                / bundle.sample_rate
            )
        existing = {
            candidate.id: candidate
            for candidate in track.candidate_events
            if candidate.generator == _GENERATOR and candidate.source == "vocals"
        }
        for candidate in list(track.candidate_events):
            if (
                candidate.generator == _GENERATOR
                and candidate.source == "vocals"
                and candidate.id not in desired
            ):
                track.candidate_events.remove(candidate)
        fields = (
            "hit_point_id",
            "sample",
            "acoustic_sample",
            "chart_sample",
            "snap_error_ms",
            "lane",
            "source_evidence_json",
            "semantic_evidence_json",
            "confidence",
            "status",
            "grid_type",
            "grid_confidence",
            "source",
            "generator",
            "character",
            "mora",
            "phoneme",
            "event_level",
            "event_policy",
            "alignment_unit_id",
            "alignment_unit_index",
            "alignment_run_id",
            "character_indices_json",
            "phonemes_json",
            "aligned_sample",
            "refined_sample",
            "evidence_json",
        )
        for candidate_id, replacement in desired.items():
            current = existing.get(candidate_id)
            if current is None:
                track.candidate_events.append(replacement)
                continue
            for field in fields:
                setattr(current, field, getattr(replacement, field))
        active_session.commit()
        return len(desired)
    finally:
        if owns_session:
            active_session.close()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def candidate_bundle_path(storage_dir: Path, track_id: str, run_id: str) -> Path:
    return storage_dir / "alignment" / track_id / "ctc" / f"{run_id}.candidate-events.json"


def hubert_report_path(storage_dir: Path, track_id: str, run_id: str) -> Path:
    return storage_dir / "alignment" / track_id / "ctc" / f"{run_id}.hubert-report.json"


def publish_hubert_artifacts(
    context: AlignmentContext,
    artifacts: HubertArtifacts,
    *,
    persist: bool = True,
) -> int:
    """Atomically publish per-run API artifacts and the required project report."""

    _atomic_json(
        candidate_bundle_path(
            context.storage_dir,
            context.track_id,
            artifacts.candidates.run_id,
        ),
        artifacts.candidates.model_dump(mode="json", by_alias=True),
    )
    report_payload = artifacts.report.model_dump(mode="json", by_alias=True)
    _atomic_json(
        hubert_report_path(
            context.storage_dir,
            context.track_id,
            artifacts.report.hubert.run_id,
        ),
        report_payload,
    )
    _atomic_json(context.project_root / "reports" / "hubert-alignment-report.json", report_payload)
    return persist_hubert_candidates(artifacts.candidates) if persist else 0


def load_hubert_candidates(
    storage_dir: Path,
    track_id: str,
    run_id: str,
) -> HubertCandidateBundle | None:
    try:
        return HubertCandidateBundle.model_validate_json(
            candidate_bundle_path(storage_dir, track_id, run_id).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None


def load_hubert_report(
    storage_dir: Path,
    track_id: str,
    run_id: str,
) -> HubertAlignmentReport | None:
    try:
        return HubertAlignmentReport.model_validate_json(
            hubert_report_path(storage_dir, track_id, run_id).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return None
