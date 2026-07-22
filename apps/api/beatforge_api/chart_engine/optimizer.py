from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

from .footwork import LaneEvidenceKey, LaneProbabilities, repair_no_spin_footwork
from .models import ChartEvent
from .rhythm_policy import density_note_limit


@dataclass(frozen=True, slots=True)
class OptimizationReport:
    input_events: int
    output_events: int
    duplicate_notes_removed: int
    simultaneous_notes_removed: int
    density_events_removed: int
    footwork_lanes_reassigned: int
    footwork_feet_assigned: int
    footwork_segments_repaired: int


@dataclass(slots=True)
class _SelectedEvent:
    event: ChartEvent
    playable_notes: int
    anchor_priority: int
    confidence: float
    removed: bool = False


def _field_has_value(event: ChartEvent, name: str) -> bool:
    value: Any = getattr(event, name, None)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return bool(value)
    if value is None:
        return False
    try:
        return bool(len(value))
    except TypeError:
        return bool(value)


def _anchor_priority(event: ChartEvent) -> int:
    """Return the strongest available rhythm-anchor priority.

    New charts carry ``anchor_priority`` explicitly.  The remaining checks keep
    optimization safe for charts produced before that field existed and for
    partially migrated fixtures.
    """

    raw_priority = getattr(event, "anchor_priority", None)
    has_explicit_priority = raw_priority is not None
    try:
        priority = min(2, max(0, int(raw_priority or 0)))
    except (TypeError, ValueError):
        priority = 0
    if _field_has_value(event, "source_hit_point_ids") or any(
        note.source == "hit_point" for note in event.notes
    ):
        priority = 2
    if not has_explicit_priority and (
        _field_has_value(event, "protected_source_ids")
        or _field_has_value(event, "source_event_ids")
        or _field_has_value(event, "is_protected")
    ):
        priority = max(priority, 1)
    return priority


def _playable_notes(event: ChartEvent) -> list[Any]:
    return [note for note in event.notes if note.type != "mine"]


def _event_confidence(event: ChartEvent) -> float:
    notes = _playable_notes(event) or event.notes
    return sum(note.confidence for note in notes) / max(len(notes), 1)


def _downgrade_jump(event: ChartEvent) -> tuple[ChartEvent, int]:
    """Reduce a jump to its strongest playable note without losing its timing."""

    playable = _playable_notes(event)
    if len(playable) <= 1:
        return event, 0
    strongest = max(playable, key=lambda note: note.confidence)
    notes = [
        note
        for note in event.notes
        if note.type == "mine" or note is strongest
    ]
    return event.model_copy(update={"notes": notes}), len(playable) - 1


def _limit_disjoint_jump_transitions(
    events: list[ChartEvent], difficulty: int
) -> tuple[list[ChartEvent], int]:
    """Remove forced jump-to-jump repositioning from lower-level charts.

    Two nearby jumps that share no panel require both feet to leave and land on
    a different pair. At Lv.10 and below, keep both rhythm rows and downgrade
    the later row to its strongest panel. Shared-pivot pairs and isolated jumps
    remain intact.
    """

    if difficulty > 10:
        return events, 0
    output: list[ChartEvent] = []
    removed_notes = 0
    for event in events:
        previous = output[-1] if output else None
        previous_playable = _playable_notes(previous) if previous is not None else []
        playable = _playable_notes(event)
        if (
            previous is not None
            and len(previous_playable) > 1
            and len(playable) > 1
            and event.beat - previous.beat <= 1.0 + 1e-9
            and {note.lane for note in previous_playable}.isdisjoint(
                note.lane for note in playable
            )
        ):
            event, removed = _downgrade_jump(event)
            removed_notes += removed
        output.append(event)
    return output, removed_notes


def optimize_events(
    events: list[ChartEvent],
    difficulty: int,
    *,
    bpm: float | None = None,
    lane_probabilities: dict[LaneEvidenceKey, LaneProbabilities] | None = None,
) -> tuple[list[ChartEvent], OptimizationReport]:
    """Apply deterministic panel and density constraints before validation."""

    normalized: list[ChartEvent] = []
    duplicate_notes_removed = 0
    simultaneous_notes_removed = 0
    for event in sorted(events, key=lambda item: (item.time_sec, item.beat)):
        by_lane = {}
        for note in sorted(event.notes, key=lambda item: item.confidence, reverse=True):
            if note.lane in by_lane:
                duplicate_notes_removed += 1
                continue
            by_lane[note.lane] = note
        notes = list(by_lane.values())
        if len(notes) > 2:
            simultaneous_notes_removed += len(notes) - 2
            notes = notes[:2]
        if notes:
            normalized.append(event.model_copy(update={"notes": notes}))

    normalized, jump_notes_removed = _limit_disjoint_jump_transitions(
        normalized, difficulty
    )
    simultaneous_notes_removed += jump_notes_removed

    limit = density_note_limit(difficulty, bpm=bpm)
    recent: deque[_SelectedEvent] = deque()
    selected: list[_SelectedEvent] = []
    density_removed = 0
    for event in normalized:
        while recent and event.time_sec - recent[0].event.time_sec >= 2.0:
            recent.popleft()

        count = len(_playable_notes(event))
        current_notes = sum(
            item.playable_notes for item in recent if not item.removed
        )
        if current_notes + count > limit and count > 1:
            event, removed_notes = _downgrade_jump(event)
            count -= removed_notes
            simultaneous_notes_removed += removed_notes

        priority = _anchor_priority(event)
        if current_notes + count > limit and priority == 0:
            density_removed += 1
            continue

        while current_notes + count > limit:
            replaceable = [
                item
                for item in recent
                if not item.removed and item.anchor_priority == 0
            ]
            if not replaceable:
                # BeatForge rhythm markers (including aligned vocal morae) and
                # confirmed hit points are anchors. Keep every timing row and
                # let the validator describe any unavoidable density excess.
                break
            victim = min(
                replaceable,
                key=lambda item: (
                    item.anchor_priority,
                    item.confidence,
                    -item.playable_notes,
                    item.event.time_sec,
                ),
            )
            if victim.anchor_priority == 0 and victim.playable_notes > 1:
                downgraded, removed_notes = _downgrade_jump(victim.event)
                victim.event = downgraded
                victim.playable_notes -= removed_notes
                simultaneous_notes_removed += removed_notes
            else:
                victim.removed = True
                density_removed += 1
            current_notes = sum(
                item.playable_notes for item in recent if not item.removed
            )

        item = _SelectedEvent(
            event=event,
            playable_notes=count,
            anchor_priority=priority,
            confidence=_event_confidence(event),
        )
        selected.append(item)
        recent.append(item)

    output = [item.event for item in selected if not item.removed]
    output, footwork = repair_no_spin_footwork(
        output, lane_probabilities=lane_probabilities
    )
    simultaneous_notes_removed += footwork.notes_removed
    return output, OptimizationReport(
        input_events=len(events),
        output_events=len(output),
        duplicate_notes_removed=duplicate_notes_removed,
        simultaneous_notes_removed=simultaneous_notes_removed,
        density_events_removed=density_removed,
        footwork_lanes_reassigned=footwork.lanes_reassigned,
        footwork_feet_assigned=footwork.feet_assigned,
        footwork_segments_repaired=footwork.segments_repaired,
    )
