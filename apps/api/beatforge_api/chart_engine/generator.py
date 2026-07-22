from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from .footwork import LaneEvidenceKey, LaneProbabilities, lane_evidence_key
from .models import ChartDocument, ChartEvent, ChartNote
from .optimizer import optimize_events
from .statistics import chart_statistics
from .timing import TempoTimeline
from .validator import validate_chart

_DEFAULT_TRANSITIONS = (
    (0.03, 0.12, 0.30, 0.28, 0.27),
    (0.12, 0.03, 0.30, 0.27, 0.28),
    (0.24, 0.24, 0.04, 0.24, 0.24),
    (0.28, 0.27, 0.30, 0.03, 0.12),
    (0.27, 0.28, 0.30, 0.12, 0.03),
)
_COORDS = ((-1.0, -1.0), (-1.0, 1.0), (0.0, 0.0), (1.0, 1.0), (1.0, -1.0))
_BIG_SPIN = (3, 2, 4, 2, 1, 2, 0, 2)
_SMALL_SPIN = (3, 2, 4) * 3
_MODEL_EVENT_THRESHOLD = 0.5
_GENERATOR_VERSION = "1.7"


@dataclass(slots=True)
class _SourcePoint:
    beat: float
    time_sec: float
    score: float
    source_id: str | None
    source: str
    subdivision: int
    lane_probabilities: tuple[float, float, float, float, float] | None = None
    hold_probability: float | None = None
    source_event_ids: tuple[str, ...] = ()
    source_hit_point_ids: tuple[str, ...] = ()
    anchor_priority: int = 0
    is_full_band_accent: bool = False


def _value(item: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(item, dict) and name in item:
            return item[name]
        if hasattr(item, name):
            return getattr(item, name)
    return default


def _is_vocal_mora_anchor(item: Any) -> bool:
    """Return whether a published HuBERT mora is chart rhythm evidence.

    The vocal editor renders both accepted and uncertain HuBERT mora events as
    aligned timing markers.  ``uncertain`` describes confidence, not the
    absence of a vocal onset, so the chart model may choose its panel but must
    not delete its timing row.  Explicitly rejected events remain optional.
    """

    status = str(_value(item, "status", default="uncertain")).lower()
    generator = str(_value(item, "generator", default="")).lower()
    event_level = str(_value(item, "event_level", "eventLevel", default="")).lower()
    source = str(_value(item, "source", "lane", default="")).lower()
    alignment_unit_id = _value(
        item, "alignment_unit_id", "alignmentUnitId", default=None
    )
    alignment_unit_index = _value(
        item, "alignment_unit_index", "alignmentUnitIndex", default=None
    )
    alignment_run_id = _value(
        item, "alignment_run_id", "alignmentRunId", default=None
    )
    return (
        status != "rejected"
        and generator == "hubert_ctc"
        and event_level == "mora"
        and source == "vocals"
        and bool(alignment_unit_id)
        and alignment_unit_index is not None
        and bool(alignment_run_id)
    )


def _tempo_timeline(
    tempo_segments: Iterable[Any], sample_rate: int
) -> tuple[TempoTimeline, list[tuple[float, float]]]:
    segments = sorted(
        list(tempo_segments),
        key=lambda item: int(_value(item, "start_sample", "startSample", default=0)),
    )
    if not segments:
        raise ValueError("chart generation requires a BeatForge tempo map")
    first = segments[0]
    first_bpm = float(_value(first, "bpm", default=0.0))
    if first_bpm <= 0:
        raise ValueError("chart generation requires a positive BPM")
    beat_zero_sec = (
        float(_value(first, "beat_offset_sample", "beatOffsetSample", default=0)) / sample_rate
    )
    changes: list[tuple[float, float]] = [(0.0, first_bpm)]
    previous_time = 0.0
    previous_beat = -beat_zero_sec * first_bpm / 60.0
    previous_bpm = first_bpm
    for segment in segments[1:]:
        start_time = float(_value(segment, "start_sample", "startSample", default=0)) / sample_rate
        beat = previous_beat + (start_time - previous_time) * previous_bpm / 60.0
        bpm = float(_value(segment, "bpm", default=previous_bpm))
        if beat >= 0 and bpm > 0:
            changes.append((beat, bpm))
        previous_time = start_time
        previous_beat = beat
        previous_bpm = bpm
    return TempoTimeline(changes, offset_sec=-beat_zero_sec), changes


def _grid_subdivision(difficulty: int) -> int:
    if difficulty <= 3:
        return 4
    if difficulty <= 7:
        return 8
    # Lv.8-10 use at most sixteenths. Lv.11+ may additionally select
    # evidence-backed twenty-fourths, but inferred filler stays on sixteenths.
    return 16


def _grid_step(difficulty: int) -> float:
    return 4.0 / _grid_subdivision(difficulty)


def _snap_beat(beat: float, subdivision: int) -> float:
    step = 4.0 / subdivision
    return round(beat / step) * step


def _slot_key(beat: float) -> int:
    """Index the union of 1/16 and 1/24 grids without float-key drift."""

    # Sixteenth rows are multiples of 3/12 beat and twenty-fourth rows are
    # multiples of 2/12 beat, so 1/12 beat is their exact common lattice.
    return int(round(beat * 12.0))


def _source_grid(
    item: Any,
    *,
    kind: str,
    difficulty: int,
    timeline: TempoTimeline,
    sample_rate: int,
    chart_sample: int,
) -> tuple[float, int]:
    chart_beat = timeline.time_to_beat(chart_sample / sample_rate)
    base_subdivision = _grid_subdivision(difficulty)
    if difficulty <= 10:
        return _snap_beat(chart_beat, base_subdivision), base_subdivision

    grid_type = str(_value(item, "grid_type", "gridType", default="")).lower()
    grid_confidence = float(
        _value(item, "grid_confidence", "gridConfidence", default=0.0) or 0.0
    )
    if grid_confidence >= 0.5 and ("1_24" in grid_type or "twenty_fourth" in grid_type):
        return _snap_beat(chart_beat, 24), 24
    if grid_confidence >= 0.5 and ("1_16" in grid_type or "sixteenth" in grid_type):
        return _snap_beat(chart_beat, 16), 16

    # BeatForge currently emits predominantly 1/16 chart samples. At Lv.11+
    # only low-confidence/unsnapped candidate evidence may recover a 1/24 row
    # from its acoustic position. Confirmed hit points retain their edited chart
    # timing and are snapped to the nearest supported row.
    source_sample = chart_sample
    if kind == "candidate":
        source_sample = int(
            _value(
                item,
                "acoustic_sample",
                "acousticSample",
                "refined_sample",
                "refinedSample",
                "sample",
                default=chart_sample,
            )
        )
    source_beat = timeline.time_to_beat(source_sample / sample_rate)
    choices = [
        (abs(source_beat - _snap_beat(source_beat, subdivision)), subdivision)
        for subdivision in (16, 24)
    ]
    _error, subdivision = min(choices, key=lambda value: (value[0], value[1]))
    return _snap_beat(source_beat, subdivision), subdivision


def _merge_ids(*groups: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for group in groups for item in group if item))


def _merge_source_points(previous: _SourcePoint, point: _SourcePoint) -> _SourcePoint:
    winner, other = max(
        ((previous, point), (point, previous)),
        key=lambda pair: (pair[0].anchor_priority, pair[0].score),
    )
    source_event_ids = _merge_ids(previous.source_event_ids, point.source_event_ids)
    source_hit_point_ids = _merge_ids(
        previous.source_hit_point_ids, point.source_hit_point_ids
    )
    source_id = winner.source_id
    if source_event_ids and source_id not in source_event_ids:
        source_id = source_event_ids[0]
    return _SourcePoint(
        beat=winner.beat,
        time_sec=winner.time_sec,
        score=winner.score,
        source_id=source_id,
        source="hit_point" if source_hit_point_ids else winner.source,
        subdivision=min(previous.subdivision, point.subdivision),
        lane_probabilities=winner.lane_probabilities or other.lane_probabilities,
        hold_probability=(
            winner.hold_probability
            if winner.hold_probability is not None
            else other.hold_probability
        ),
        source_event_ids=source_event_ids,
        source_hit_point_ids=source_hit_point_ids,
        anchor_priority=max(previous.anchor_priority, point.anchor_priority),
        is_full_band_accent=(
            previous.is_full_band_accent or point.is_full_band_accent
        ),
    )


def _source_points(
    *,
    timeline: TempoTimeline,
    sample_rate: int,
    duration_sec: float,
    candidates: Iterable[Any],
    hit_points: Iterable[Any],
    difficulty: int,
    model_predictions: dict[str, dict[str, Any]] | None,
) -> list[_SourcePoint]:
    by_slot: dict[int, _SourcePoint] = {}
    candidate_items = list(candidates)
    hit_items = list(hit_points)
    candidate_by_hit_id = {
        str(hit_point_id): str(candidate_id)
        for candidate in candidate_items
        if (candidate_id := _value(candidate, "id"))
        and (hit_point_id := _value(candidate, "hit_point_id", "hitPointId", default=None))
    }

    def add(item: Any, kind: str, bonus: float) -> None:
        sample = int(
            _value(
                item,
                "chart_sample",
                "chartSample",
                "snapped_sample",
                "snappedSample",
                "sample",
                default=-1,
            )
        )
        time_sec = sample / sample_rate
        if sample < 0 or time_sec < 0 or time_sec > duration_sec:
            return
        beat = timeline.time_to_beat(time_sec)
        if beat < 0 and kind != "hit_point":
            return
        if beat < 0:
            # A confirmed marker can legitimately sit in the short pickup
            # before BeatForge's first beat. Chart rows cannot have a negative
            # beat, so preserve it at the beat-zero boundary instead of
            # silently deleting it.
            snapped_beat = 0.0
            subdivision = _grid_subdivision(difficulty)
        else:
            snapped_beat, subdivision = _source_grid(
                item,
                kind=kind,
                difficulty=difficulty,
                timeline=timeline,
                sample_rate=sample_rate,
                chart_sample=sample,
            )
        snapped_time = timeline.beat_to_time(snapped_beat)
        if snapped_time < 0 or snapped_time > duration_sec:
            return
        confidence = float(_value(item, "confidence", default=0.5) or 0.0)
        grid_confidence = float(
            _value(item, "grid_confidence", "gridConfidence", default=0.5) or 0.0
        )
        salience = float(_value(item, "salience", default=confidence) or 0.0)
        status = str(_value(item, "status", default="accepted"))
        primary = str(_value(item, "primary_stem", "primaryStem", "lane", "source", default="mix"))
        source_weight = {"drums": 0.12, "vocals": 0.08, "melody": 0.06, "other": 0.06}.get(
            primary, 0.03
        )
        score = (
            0.48 * confidence
            + 0.24 * grid_confidence
            + 0.18 * salience
            + source_weight
            + bonus
            + (0.06 if status == "accepted" else -0.08 if status == "rejected" else 0.0)
        )
        slot = _slot_key(snapped_beat)
        item_id = str(_value(item, "id", default="")) or None
        prediction_id = item_id
        if kind == "hit_point":
            prediction_id = (
                str(
                    _value(
                        item,
                        "candidate_event_id",
                        "candidateEventId",
                        default=candidate_by_hit_id.get(item_id or ""),
                    )
                    or ""
                )
                or None
            )
        prediction = (model_predictions or {}).get(prediction_id or "", {})
        raw_lanes = prediction.get("laneProbabilities")
        lane_probabilities = None
        anchor_priority = (
            2
            if kind == "hit_point"
            else 1
            if status == "accepted" or _is_vocal_mora_anchor(item)
            else 0
        )
        if isinstance(raw_lanes, list | tuple) and len(raw_lanes) == 5:
            lane_probabilities = tuple(min(1.0, max(0.0, float(value))) for value in raw_lanes)
            # The current model has five lane sigmoid heads, not an event-
            # presence head. Low absolute lane confidence can filter optional
            # proposals, but must never erase accepted/confirmed rhythm input.
            if (
                anchor_priority == 0
                and max(lane_probabilities) < _MODEL_EVENT_THRESHOLD
            ):
                return
            score += 0.35 * (max(lane_probabilities) - 0.5)
        raw_hold = prediction.get("holdProbability")
        hold_probability = min(1.0, max(0.0, float(raw_hold))) if raw_hold is not None else None
        point = _SourcePoint(
            beat=snapped_beat,
            time_sec=snapped_time,
            score=score,
            source_id=prediction_id if lane_probabilities is not None else item_id,
            source=kind,
            subdivision=subdivision,
            lane_probabilities=lane_probabilities,
            hold_probability=hold_probability,
            source_event_ids=(
                (item_id,)
                if kind == "candidate" and item_id
                else (prediction_id,)
                if prediction_id
                else ()
            ),
            source_hit_point_ids=(item_id,) if kind == "hit_point" and item_id else (),
            anchor_priority=anchor_priority,
            is_full_band_accent=(
                kind == "hit_point"
                and str(_value(item, "band", default="")) == "full_band_accent"
            ),
        )
        previous = by_slot.get(slot)
        if previous is None:
            by_slot[slot] = point
        else:
            by_slot[slot] = _merge_source_points(previous, point)

    for candidate in candidate_items:
        add(candidate, "candidate", 0.0)
    for hit in hit_items:
        add(hit, "hit_point", 0.10)
    return sorted(by_slot.values(), key=lambda point: (point.beat, -point.score))


def _choose_points(
    points: list[_SourcePoint],
    *,
    difficulty: int,
    duration_sec: float,
    timeline: TempoTimeline,
    rng: random.Random,
    fill_from_tempo_grid: bool,
) -> list[_SourcePoint]:
    if not points:
        raise ValueError("chart generation requires analyzed candidate events or hit points")
    target_nps = 0.45 + difficulty * 0.21
    active_duration = min(duration_sec, max(1.0, points[-1].time_sec - points[0].time_sec + 2.0))
    target = max(8, int(round(active_duration * target_nps)))
    protected = [point for point in points if point.anchor_priority > 0]
    optional = [point for point in points if point.anchor_priority == 0]
    optional_budget = max(0, target - len(protected))
    if len(optional) > optional_budget:
        ratio = optional_budget / len(optional) if optional else 0.0
        buckets: defaultdict[int, list[_SourcePoint]] = defaultdict(list)
        for point in optional:
            buckets[int(point.beat // 16.0)].append(point)
        chosen: list[_SourcePoint] = []
        for bucket in buckets.values():
            if optional_budget == 0:
                break
            quota = max(1, min(len(bucket), int(round(len(bucket) * ratio))))
            ranked = sorted(
                bucket,
                key=lambda point: point.score + rng.random() * 0.035,
                reverse=True,
            )
            chosen.extend(ranked[:quota])
        if len(chosen) > optional_budget:
            chosen = sorted(
                chosen, key=lambda point: point.score + rng.random() * 0.02, reverse=True
            )[:optional_budget]
        elif len(chosen) < optional_budget:
            chosen_ids = {id(point) for point in chosen}
            remaining = [point for point in optional if id(point) not in chosen_ids]
            chosen.extend(
                sorted(
                    remaining,
                    key=lambda point: point.score + rng.random() * 0.02,
                    reverse=True,
                )[: optional_budget - len(chosen)]
            )
        points = sorted(protected + chosen, key=lambda point: point.beat)
    if len(points) < target and fill_from_tempo_grid:
        # Fill only gaps inside analyzed coverage and only on the real tempo grid.
        step = _grid_step(difficulty)
        subdivision = _grid_subdivision(difficulty)
        occupied = {_slot_key(point.beat) for point in points}
        first_slot = int(math.ceil(points[0].beat / step - 1e-9))
        last_slot = int(math.floor(points[-1].beat / step + 1e-9))
        additions: list[_SourcePoint] = []
        for slot in range(first_slot, last_slot + 1):
            beat = slot * step
            if len(points) + len(additions) >= target or _slot_key(beat) in occupied:
                continue
            # Prefer downbeats and half-beats; finer inferred grid points receive
            # a lower score and are used only at high requested difficulties.
            phase = beat % 1.0
            if phase and difficulty < 8:
                continue
            time_sec = timeline.beat_to_time(beat)
            if 0 <= time_sec <= duration_sec:
                additions.append(
                    _SourcePoint(
                        beat=beat,
                        time_sec=time_sec,
                        score=0.34 if phase == 0 else 0.24,
                        source_id=None,
                        source="tempo_grid",
                        subdivision=subdivision,
                    )
                )
        points = sorted(points + additions, key=lambda point: point.beat)
    return points


def _weighted_lane(
    rng: random.Random,
    previous_lane: int | None,
    previous_time: float | None,
    time_sec: float,
    transitions: list[list[float]],
    previous_panel_side: str | None,
    model_probabilities: tuple[float, float, float, float, float] | None,
) -> int:
    weights = list(transitions[previous_lane] if previous_lane is not None else [1.0] * 5)
    if model_probabilities is not None:
        transition_total = max(sum(weights), 1e-9)
        weights = [
            0.25 * (weights[index] / transition_total) + 0.75 * model_probabilities[index]
            for index in range(5)
        ]
    interval = time_sec - previous_time if previous_time is not None else 1.0
    for lane in range(5):
        if lane == previous_lane:
            weights[lane] *= 0.04
        if previous_lane is not None and interval < 0.12:
            distance = math.dist(_COORDS[previous_lane], _COORDS[lane])
            if distance > 2.1:
                weights[lane] *= 0.02
        preferred = "left" if lane in {0, 1} else "right" if lane in {3, 4} else None
        if (
            interval < 0.24
            and preferred is not None
            and preferred == previous_panel_side
        ):
            weights[lane] *= 0.18
    total = sum(max(0.0, weight) for weight in weights)
    pick = rng.random() * total
    cursor = 0.0
    for lane, weight in enumerate(weights):
        cursor += max(0.0, weight)
        if pick <= cursor:
            return lane
    return 2


def _assign_events(
    points: list[_SourcePoint],
    *,
    difficulty: int,
    timeline: TempoTimeline,
    transitions: list[list[float]],
    rng: random.Random,
) -> tuple[
    list[ChartEvent],
    dict[LaneEvidenceKey, LaneProbabilities],
]:
    # Lv.6-10 model jumps must come from two independent lane heads. Randomly
    # decorating otherwise isolated rows creates tiring, unmusical jump spam;
    # a full-band BeatForge accent on an integer beat is the acoustic exception.
    # Expert charts may retain a very small corpus-style fallback probability.
    jump_probability = max(0.0, (difficulty - 10) * 0.008)
    hold_probability = 0.015 + difficulty * 0.0035
    events: list[ChartEvent] = []
    lane_evidence: dict[LaneEvidenceKey, LaneProbabilities] = {}
    previous_lane: int | None = None
    previous_time: float | None = None
    previous_panel_side: str | None = None
    for index, point in enumerate(points):
        step = 4.0 / point.subdivision
        lane = _weighted_lane(
            rng,
            previous_lane,
            previous_time,
            point.time_sec,
            transitions,
            previous_panel_side,
            point.lane_probabilities,
        )
        interval = point.time_sec - previous_time if previous_time is not None else 1.0
        previous_beat_gap = (
            point.beat - points[index - 1].beat if index > 0 else math.inf
        )
        next_beat_gap = (
            points[index + 1].beat - point.beat if index + 1 < len(points) else math.inf
        )
        jump_spacing_safe = min(previous_beat_gap, next_beat_gap) >= 0.5 - 1e-9
        predicted_jump_pair: tuple[int, int] | None = None
        if point.lane_probabilities is not None:
            ranked_lanes = sorted(
                range(5), key=lambda value: point.lane_probabilities[value], reverse=True
            )
            # Five sigmoid heads describe independent panels. A jump exists only
            # when the *second-highest* head is strong too; looking for a high
            # lane relative to a randomly sampled primary falsely turns a single
            # strong head into a jump.
            if point.lane_probabilities[ranked_lanes[1]] >= 0.58:
                predicted_jump_pair = (ranked_lanes[0], ranked_lanes[1])
                if jump_spacing_safe and lane not in predicted_jump_pair:
                    lane = predicted_jump_pair[0]
        note_type = "tap"
        end_beat = end_time = None
        if index + 1 < len(points):
            next_beat = points[index + 1].beat
            effective_hold_probability = hold_probability
            if point.hold_probability is not None:
                effective_hold_probability = max(
                    hold_probability * 0.5, min(0.25, point.hold_probability * 0.35)
                )
            if next_beat - point.beat >= 0.75 and rng.random() < effective_hold_probability:
                end_beat = min(point.beat + 1.0, next_beat - step)
                if end_beat - point.beat >= 0.5:
                    note_type = "hold"
                    end_time = timeline.beat_to_time(end_beat)
        notes = [
            ChartNote(
                lane=lane,
                type=note_type,
                end_time_sec=end_time,
                end_beat=end_beat,
                source=point.source,
                confidence=min(1.0, max(0.0, point.score)),
                foot=None,
            )
        ]
        predicted_jump_lane = None
        if predicted_jump_pair is not None and lane in predicted_jump_pair:
            predicted_jump_lane = next(
                value for value in predicted_jump_pair if value != lane
            )
        accent_jump = (
            difficulty >= 8
            and point.is_full_band_accent
            and math.isclose(point.beat, round(point.beat), abs_tol=1e-9)
        )
        if (
            difficulty >= 6
            and interval >= 0.13
            and jump_spacing_safe
            and note_type == "tap"
            and (
                predicted_jump_lane is not None
                or accent_jump
                or (jump_probability > 0.0 and rng.random() < jump_probability)
            )
        ):
            if predicted_jump_lane is not None:
                second = predicted_jump_lane
            else:
                choices = [value for value in range(5) if value != lane and abs(value - lane) >= 2]
                second = choices[int(rng.random() * len(choices))]
            notes.append(
                ChartNote(
                    lane=second,
                    source=point.source,
                    confidence=min(1.0, max(0.0, point.score)),
                    foot=None,
                )
            )
        measure = int(point.beat // 4.0)
        row_index = int(
            round((point.beat - measure * 4.0) * point.subdivision / 4.0)
        )
        event = ChartEvent(
            time_sec=point.time_sec,
            beat=point.beat,
            measure=measure,
            subdivision=point.subdivision,
            row_index=row_index,
            notes=notes,
            source_event_id=point.source_id,
            source_event_ids=list(point.source_event_ids),
            source_hit_point_ids=list(point.source_hit_point_ids),
            anchor_priority=point.anchor_priority,
        )
        events.append(event)
        if point.lane_probabilities is not None:
            lane_evidence[lane_evidence_key(event)] = point.lane_probabilities
        previous_lane = lane
        previous_time = point.time_sec
        previous_panel_side = (
            "left" if lane in {0, 1} else "right" if lane in {3, 4} else None
        )
    return events, lane_evidence


def _apply_spin(events: list[ChartEvent], difficulty: int, enabled: bool) -> list[ChartEvent]:
    if not enabled or difficulty < 11:
        return events
    output = list(events)

    def apply_pattern(start: int, pattern: tuple[int, ...], label: str) -> bool:
        window = output[start : start + len(pattern)]
        if len(window) != len(pattern):
            return False
        if any(
            event.pattern or len(event.notes) != 1 or event.notes[0].type != "tap"
            for event in window
        ):
            return False
        if any(
            current.beat - previous.beat > 0.5
            for previous, current in zip(window, window[1:], strict=False)
        ):
            return False
        for index, (event, lane) in enumerate(zip(window, pattern, strict=True)):
            foot = "left" if index % 2 == 0 else "right"
            output[start + index] = event.model_copy(
                update={
                    "notes": [event.notes[0].model_copy(update={"lane": lane, "foot": foot})],
                    "pattern": label,
                }
            )
        return True

    def apply_near(target: int, pattern: tuple[int, ...], label: str) -> bool:
        starts = range(0, max(0, len(output) - len(pattern)) + 1)
        for start in sorted(starts, key=lambda value: abs(value - target)):
            if apply_pattern(start, pattern, label):
                return True
        return False

    small_start = max(0, len(output) // 3 - len(_SMALL_SPIN) // 2)
    apply_near(small_start, _SMALL_SPIN, "small_spin")
    if difficulty >= 14:
        big_start = max(0, (len(output) * 2) // 3 - len(_BIG_SPIN) // 2)
        apply_near(big_start, _BIG_SPIN, "big_spin")
    return output


def generate_chart(
    *,
    track_id: str,
    title: str,
    artist: str,
    music: str,
    duration_sec: float,
    sample_rate: int,
    tempo_segments: Iterable[Any],
    candidates: Iterable[Any],
    hit_points: Iterable[Any],
    difficulty: int,
    enable_spin: bool = False,
    seed: int | None = None,
    transition_probabilities: list[list[float]] | None = None,
    model_predictions: dict[str, dict[str, Any]] | None = None,
    model_provenance: dict[str, Any] | None = None,
) -> ChartDocument:
    """Generate a deterministic five-panel chart from persisted BeatForge evidence."""

    difficulty = min(max(int(difficulty), 1), 15)
    if seed is None:
        key = (
            f"{track_id}:{difficulty}:{int(enable_spin)}:"
            f"beatforge-chart-v{_GENERATOR_VERSION}"
        )
        seed = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    rng = random.Random(seed)
    timeline, _changes = _tempo_timeline(tempo_segments, sample_rate)
    candidate_items = list(candidates)
    hit_point_items = list(hit_points)
    vocal_mora_marker_ids = {
        str(candidate_id)
        for candidate in candidate_items
        if _is_vocal_mora_anchor(candidate)
        and (candidate_id := _value(candidate, "id", default=None))
    }
    points = _source_points(
        timeline=timeline,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        candidates=candidate_items,
        hit_points=hit_point_items,
        difficulty=difficulty,
        model_predictions=model_predictions,
    )
    chosen = _choose_points(
        points,
        difficulty=difficulty,
        duration_sec=duration_sec,
        timeline=timeline,
        rng=rng,
        fill_from_tempo_grid=model_predictions is None,
    )
    matrix = transition_probabilities or [list(row) for row in _DEFAULT_TRANSITIONS]
    if len(matrix) != 5 or any(len(row) != 5 for row in matrix):
        matrix = [list(row) for row in _DEFAULT_TRANSITIONS]
    events, lane_evidence = _assign_events(
        chosen,
        difficulty=difficulty,
        timeline=timeline,
        transitions=matrix,
        rng=rng,
    )
    anchor_input = [event for event in events if event.anchor_priority > 0]
    accepted_input = [event for event in events if event.anchor_priority == 1]
    hit_point_input = [event for event in events if event.anchor_priority == 2]
    events, optimization = optimize_events(
        events,
        difficulty,
        bpm=timeline.primary_bpm,
        lane_probabilities=lane_evidence,
    )
    events = _apply_spin(events, difficulty, enable_spin)
    output_source_event_ids = {
        source_id for event in events for source_id in event.source_event_ids
    }
    optimization_payload = {
        **asdict(optimization),
        "protected_source_points_input": len(anchor_input),
        "protected_source_points_output": sum(
            event.anchor_priority > 0 for event in events
        ),
        "accepted_anchor_events_input": len(accepted_input),
        "accepted_anchor_events_output": sum(
            event.anchor_priority == 1 for event in events
        ),
        "hit_point_anchor_events_input": len(hit_point_input),
        "hit_point_anchor_events_output": sum(
            event.anchor_priority == 2 for event in events
        ),
        "vocal_mora_markers_input": len(vocal_mora_marker_ids),
        "vocal_mora_markers_output": len(
            vocal_mora_marker_ids & output_source_event_ids
        ),
    }
    generator = "local_chart_transformer" if model_predictions else "real_corpus_profile_rules"
    document_identity = (
        f"{track_id}:{difficulty}:{enable_spin}:{seed}:generator-{_GENERATOR_VERSION}"
    )
    if model_predictions:
        provenance = model_provenance or {}
        checkpoint_identity = ":".join(
            str(provenance.get(key) or "")
            for key in (
                "checkpointSha256",
                "architecture",
                "createdAt",
                "datasetFingerprint",
            )
        )
        document_identity = f"{document_identity}:{generator}:{checkpoint_identity}"
    document_id = hashlib.sha256(document_identity.encode()).hexdigest()[:20]
    chart = ChartDocument(
        id=document_id,
        title=title,
        artist=artist,
        music=music,
        source_group="BEATFORGE_GENERATED",
        mode="pump-single",
        lane_count=5,
        difficulty="Challenge" if difficulty >= 13 else "Hard",
        meter=difficulty,
        bpm=timeline.primary_bpm,
        offset_sec=timeline.offset_sec,
        duration_sec=duration_sec,
        measure_count=max((event.measure for event in events), default=0) + 1,
        tempo_map=timeline.points(),
        events=events,
        optimization=optimization_payload,
        model_provenance=model_provenance,
        generator=generator,
        generator_version=_GENERATOR_VERSION,
        seed=seed,
        spin_enabled=enable_spin,
    )
    chart = chart.model_copy(update={"statistics": chart_statistics(chart)})
    return chart.model_copy(update={"validation": validate_chart(chart)})
