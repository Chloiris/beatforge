from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import permutations
from typing import Literal

from .models import ChartEvent

Foot = Literal["left", "right"]
LaneEvidenceKey = tuple[float, float]
LaneProbabilities = tuple[float, float, float, float, float]

_COORDS = ((-1.0, -1.0), (-1.0, 1.0), (0.0, 0.0), (1.0, 1.0), (1.0, -1.0))
_RESET_GAP_SEC = 0.60
_HEADING_LIMIT_DEGREES = 135.0
_MAX_HEADING_STEP_DEGREES = 180.0
_EPSILON = 1e-9
_SPIN_PATTERNS = {"small_spin", "big_spin"}


@dataclass(frozen=True, slots=True)
class FootPose:
    left_lane: int | None = None
    right_lane: int | None = None
    heading: float = 0.0
    last_strike: Foot | None = None
    left_hold_until: float = 0.0
    right_hold_until: float = 0.0


@dataclass(frozen=True, slots=True)
class FootworkViolation:
    event_index: int
    time_sec: float
    beat: float


@dataclass(frozen=True, slots=True)
class FootworkAnalysis:
    full_step_reachable: bool
    checked_events: int
    segment_count: int
    violations: tuple[FootworkViolation, ...]
    max_abs_heading: float
    crossover_count: int
    hold_forced_repeats: int


@dataclass(frozen=True, slots=True)
class FootworkRepairReport:
    lanes_reassigned: int
    feet_assigned: int
    notes_removed: int
    segments_repaired: int


@dataclass(frozen=True, slots=True)
class _Transition:
    pose: FootPose
    feet: tuple[Foot, ...]
    turn: float
    crossover: int
    hold_forced_repeat: int


@dataclass(frozen=True, slots=True)
class _Choice:
    playable_indices: tuple[int, ...]
    lanes: tuple[int, ...]
    feet: tuple[Foot, ...]


@dataclass(frozen=True, slots=True)
class _Path:
    # Keep removals and lane edits lexicographically ahead of style costs.
    score: tuple[int, int, int, float, float, float]
    previous: _Path | None
    choice: _Choice | None
    depth: int
    max_abs_heading: float
    crossover_count: int
    hold_forced_repeats: int


def lane_evidence_key(event: ChartEvent) -> LaneEvidenceKey:
    return (round(event.time_sec, 9), round(event.beat, 9))


def _playable_positions(event: ChartEvent) -> tuple[int, ...]:
    return tuple(index for index, note in enumerate(event.notes) if note.type != "mine")


def _released(pose: FootPose, time_sec: float) -> FootPose:
    left_hold = pose.left_hold_until if pose.left_hold_until > time_sec + _EPSILON else 0.0
    right_hold = pose.right_hold_until if pose.right_hold_until > time_sec + _EPSILON else 0.0
    if left_hold == pose.left_hold_until and right_hold == pose.right_hold_until:
        return pose
    return FootPose(
        left_lane=pose.left_lane,
        right_lane=pose.right_lane,
        heading=pose.heading,
        last_strike=pose.last_strike,
        left_hold_until=left_hold,
        right_hold_until=right_hold,
    )


def _reset_pose() -> FootPose:
    return FootPose()


def _heading_for_stance(
    left_lane: int | None,
    right_lane: int | None,
    previous_heading: float,
) -> tuple[float, float] | None:
    if left_lane is None or right_lane is None or left_lane == right_lane:
        return previous_heading, 0.0
    left = _COORDS[left_lane]
    right = _COORDS[right_lane]
    raw = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))
    candidates = []
    for winding in (-1, 0, 1):
        heading = raw + winding * 360.0
        turn = abs(heading - previous_heading)
        if (
            abs(heading) <= _HEADING_LIMIT_DEGREES + _EPSILON
            and turn <= _MAX_HEADING_STEP_DEGREES + _EPSILON
        ):
            candidates.append((turn, abs(heading), heading))
    if not candidates:
        return None
    turn, _absolute, heading = min(candidates)
    return heading, turn


def _candidate_feet(pose: FootPose) -> tuple[Foot, ...]:
    left_locked = pose.left_hold_until > 0.0
    right_locked = pose.right_hold_until > 0.0
    if left_locked and right_locked:
        return ()
    if left_locked:
        return ("right",)
    if right_locked:
        return ("left",)
    if pose.last_strike == "left":
        return ("right",)
    if pose.last_strike == "right":
        return ("left",)
    return ("left", "right")


def _single_transitions(
    pose: FootPose,
    event: ChartEvent,
    playable_index: int,
    lane: int,
) -> tuple[_Transition, ...]:
    pose = _released(pose, event.time_sec)
    note = event.notes[playable_index]
    output: list[_Transition] = []
    for foot in _candidate_feet(pose):
        left_lane = pose.left_lane
        right_lane = pose.right_lane
        left_hold = pose.left_hold_until
        right_hold = pose.right_hold_until
        forced_repeat = int(
            (left_hold > 0.0 or right_hold > 0.0) and pose.last_strike == foot
        )
        if foot == "left":
            if left_hold > 0.0:
                continue
            if right_lane == lane:
                if right_hold > 0.0:
                    continue
                right_lane = None
            left_lane = lane
            if note.type == "hold" and note.end_time_sec is not None:
                left_hold = note.end_time_sec
        else:
            if right_hold > 0.0:
                continue
            if left_lane == lane:
                if left_hold > 0.0:
                    continue
                left_lane = None
            right_lane = lane
            if note.type == "hold" and note.end_time_sec is not None:
                right_hold = note.end_time_sec
        heading_result = _heading_for_stance(left_lane, right_lane, pose.heading)
        if heading_result is None:
            continue
        heading, turn = heading_result
        output.append(
            _Transition(
                pose=FootPose(
                    left_lane=left_lane,
                    right_lane=right_lane,
                    heading=heading,
                    last_strike=foot,
                    left_hold_until=left_hold,
                    right_hold_until=right_hold,
                ),
                feet=(foot,),
                turn=turn,
                crossover=int(abs(heading) > 90.0 + _EPSILON),
                hold_forced_repeat=forced_repeat,
            )
        )
    return tuple(output)


def _jump_transitions(
    pose: FootPose,
    event: ChartEvent,
    playable_indices: tuple[int, int],
    lanes: tuple[int, int],
) -> tuple[_Transition, ...]:
    pose = _released(pose, event.time_sec)
    if pose.left_hold_until > 0.0 or pose.right_hold_until > 0.0:
        # A continuing hold plus two new panels would require three feet.
        return ()
    output: list[_Transition] = []
    for left_note_position, right_note_position in permutations((0, 1)):
        left_lane = lanes[left_note_position]
        right_lane = lanes[right_note_position]
        heading_result = _heading_for_stance(left_lane, right_lane, pose.heading)
        if heading_result is None:
            continue
        heading, turn = heading_result
        feet: list[Foot] = ["left", "left"]
        feet[left_note_position] = "left"
        feet[right_note_position] = "right"
        left_note = event.notes[playable_indices[left_note_position]]
        right_note = event.notes[playable_indices[right_note_position]]
        left_hold = (
            left_note.end_time_sec
            if left_note.type == "hold" and left_note.end_time_sec is not None
            else 0.0
        )
        right_hold = (
            right_note.end_time_sec
            if right_note.type == "hold" and right_note.end_time_sec is not None
            else 0.0
        )
        output.append(
            _Transition(
                pose=FootPose(
                    left_lane=left_lane,
                    right_lane=right_lane,
                    heading=heading,
                    last_strike=None,
                    left_hold_until=left_hold,
                    right_hold_until=right_hold,
                ),
                feet=(feet[0], feet[1]),
                turn=turn,
                crossover=int(abs(heading) > 90.0 + _EPSILON),
                hold_forced_repeat=0,
            )
        )
    return tuple(output)


def _fixed_transitions(pose: FootPose, event: ChartEvent) -> tuple[_Transition, ...]:
    playable = _playable_positions(event)
    if not playable:
        return (
            _Transition(
                pose=_released(pose, event.time_sec),
                feet=(),
                turn=0.0,
                crossover=0,
                hold_forced_repeat=0,
            ),
        )
    if len(playable) == 1:
        return _single_transitions(
            pose, event, playable[0], event.notes[playable[0]].lane
        )
    if len(playable) == 2:
        lanes = (event.notes[playable[0]].lane, event.notes[playable[1]].lane)
        return _jump_transitions(pose, event, (playable[0], playable[1]), lanes)
    return ()


def analyze_no_spin_footwork(events: list[ChartEvent]) -> FootworkAnalysis:
    ordered = sorted(enumerate(events), key=lambda item: (item[1].time_sec, item[1].beat))
    poses: dict[FootPose, tuple[float, int, int]] = {_reset_pose(): (0.0, 0, 0)}
    violations: list[FootworkViolation] = []
    checked_events = 0
    segment_count = 1 if ordered else 0
    previous_time: float | None = None

    for original_index, event in ordered:
        if event.pattern in _SPIN_PATTERNS:
            poses = {_reset_pose(): (0.0, 0, 0)}
            previous_time = event.time_sec
            continue
        if previous_time is not None and event.time_sec - previous_time >= _RESET_GAP_SEC:
            if all(
                pose.left_hold_until <= event.time_sec + _EPSILON
                and pose.right_hold_until <= event.time_sec + _EPSILON
                for pose in poses
            ):
                poses = {_reset_pose(): min(poses.values())}
                segment_count += 1
        previous_time = event.time_sec
        if _playable_positions(event):
            checked_events += 1

        next_poses: dict[FootPose, tuple[float, int, int]] = {}
        for pose, metrics in poses.items():
            for transition in _fixed_transitions(pose, event):
                candidate = (
                    max(metrics[0], abs(transition.pose.heading)),
                    metrics[1] + transition.crossover,
                    metrics[2] + transition.hold_forced_repeat,
                )
                previous = next_poses.get(transition.pose)
                if previous is None or candidate < previous:
                    next_poses[transition.pose] = candidate
        if next_poses:
            poses = next_poses
            continue

        violations.append(
            FootworkViolation(
                event_index=original_index,
                time_sec=event.time_sec,
                beat=event.beat,
            )
        )
        # Resume auditing at the failed row so one bad phrase does not make the
        # remainder of the song look like hundreds of independent failures.
        prior_metrics = min(poses.values())
        reset_transitions = _fixed_transitions(_reset_pose(), event)
        poses = {
            transition.pose: (
                max(prior_metrics[0], abs(transition.pose.heading)),
                prior_metrics[1] + transition.crossover,
                prior_metrics[2] + transition.hold_forced_repeat,
            )
            for transition in reset_transitions
        } or {_reset_pose(): (0.0, 0, 0)}
        segment_count += 1

    best_metrics = min(poses.values(), default=(0.0, 0, 0))
    return FootworkAnalysis(
        full_step_reachable=not violations,
        checked_events=checked_events,
        segment_count=segment_count,
        violations=tuple(violations),
        max_abs_heading=best_metrics[0],
        crossover_count=best_metrics[1],
        hold_forced_repeats=best_metrics[2],
    )


def _path_key(path: _Path) -> tuple[int, int, int, float, float, float]:
    return path.score


def _add_path(
    target: dict[FootPose, _Path],
    pose: FootPose,
    path: _Path,
) -> None:
    previous = target.get(pose)
    if previous is None or _path_key(path) < _path_key(previous):
        target[pose] = path


def _extended_path(
    path: _Path,
    transition: _Transition,
    choice: _Choice,
    *,
    removed_notes: int,
    changed_events: int,
    changed_notes: int,
    model_loss: float,
    distance: float,
) -> _Path:
    score = (
        path.score[0] + removed_notes,
        path.score[1] + changed_events,
        path.score[2] + changed_notes,
        path.score[3] + model_loss,
        path.score[4] + distance,
        path.score[5] + transition.turn,
    )
    return _Path(
        score=score,
        previous=path,
        choice=choice,
        depth=path.depth + 1,
        max_abs_heading=max(path.max_abs_heading, abs(transition.pose.heading)),
        crossover_count=path.crossover_count + transition.crossover,
        hold_forced_repeats=path.hold_forced_repeats + transition.hold_forced_repeat,
    )


def _lane_cost(
    original_lane: int,
    lane: int,
    probabilities: LaneProbabilities | None,
) -> tuple[int, int, float, float]:
    changed = int(original_lane != lane)
    model_loss = 0.0
    if probabilities is not None:
        model_loss = max(probabilities) - probabilities[lane]
    original = _COORDS[original_lane]
    candidate = _COORDS[lane]
    distance = math.hypot(candidate[0] - original[0], candidate[1] - original[1])
    return changed, changed, model_loss, distance


def repair_no_spin_footwork(
    events: list[ChartEvent],
    *,
    lane_probabilities: dict[LaneEvidenceKey, LaneProbabilities] | None = None,
) -> tuple[list[ChartEvent], FootworkRepairReport]:
    """Viterbi-decode a strict alternating, no-spin path without deleting rows."""

    ordered = sorted(events, key=lambda item: (item.time_sec, item.beat))
    paths: dict[FootPose, _Path] = {
        _reset_pose(): _Path(
            score=(0, 0, 0, 0.0, 0.0, 0.0),
            previous=None,
            choice=None,
            depth=0,
            max_abs_heading=0.0,
            crossover_count=0,
            hold_forced_repeats=0,
        )
    }
    previous_time: float | None = None
    for event in ordered:
        if event.pattern in _SPIN_PATTERNS:
            best = min(paths.values(), key=_path_key)
            paths = {_reset_pose(): best}
            previous_time = event.time_sec
        elif previous_time is not None and event.time_sec - previous_time >= _RESET_GAP_SEC:
            resettable = [
                path
                for pose, path in paths.items()
                if pose.left_hold_until <= event.time_sec + _EPSILON
                and pose.right_hold_until <= event.time_sec + _EPSILON
            ]
            if resettable:
                paths = {_reset_pose(): min(resettable, key=_path_key)}
        previous_time = event.time_sec
        playable = _playable_positions(event)
        probabilities = (lane_probabilities or {}).get(lane_evidence_key(event))
        next_paths: dict[FootPose, _Path] = {}

        if not playable:
            for pose, path in paths.items():
                transition = _fixed_transitions(pose, event)[0]
                _add_path(
                    next_paths,
                    transition.pose,
                    _extended_path(
                        path,
                        transition,
                        _Choice((), (), ()),
                        removed_notes=0,
                        changed_events=0,
                        changed_notes=0,
                        model_loss=0.0,
                        distance=0.0,
                    ),
                )
        elif len(playable) == 1:
            original_lane = event.notes[playable[0]].lane
            for pose, path in paths.items():
                for lane in range(5):
                    lane_cost = _lane_cost(original_lane, lane, probabilities)
                    for transition in _single_transitions(pose, event, playable[0], lane):
                        choice = _Choice((playable[0],), (lane,), transition.feet)
                        _add_path(
                            next_paths,
                            transition.pose,
                            _extended_path(
                                path,
                                transition,
                                choice,
                                removed_notes=0,
                                changed_events=lane_cost[0],
                                changed_notes=lane_cost[1],
                                model_loss=lane_cost[2],
                                distance=lane_cost[3],
                            ),
                        )
        elif len(playable) == 2:
            original_lanes = (
                event.notes[playable[0]].lane,
                event.notes[playable[1]].lane,
            )
            for pose, path in paths.items():
                for transition in _jump_transitions(
                    pose, event, (playable[0], playable[1]), original_lanes
                ):
                    choice = _Choice(playable, original_lanes, transition.feet)
                    _add_path(
                        next_paths,
                        transition.pose,
                        _extended_path(
                            path,
                            transition,
                            choice,
                            removed_notes=0,
                            changed_events=0,
                            changed_notes=0,
                            model_loss=0.0,
                            distance=0.0,
                        ),
                    )
                # If a continuing hold makes a jump impossible, preserve its
                # strongest rhythm note instead of deleting the timing row.
                if not _jump_transitions(
                    pose, event, (playable[0], playable[1]), original_lanes
                ):
                    retained = max(
                        playable,
                        key=lambda index: event.notes[index].confidence,
                    )
                    original_lane = event.notes[retained].lane
                    for lane in range(5):
                        lane_cost = _lane_cost(original_lane, lane, probabilities)
                        for transition in _single_transitions(pose, event, retained, lane):
                            choice = _Choice((retained,), (lane,), transition.feet)
                            _add_path(
                                next_paths,
                                transition.pose,
                                _extended_path(
                                    path,
                                    transition,
                                    choice,
                                    removed_notes=1,
                                    changed_events=lane_cost[0],
                                    changed_notes=lane_cost[1],
                                    model_loss=lane_cost[2],
                                    distance=lane_cost[3],
                                ),
                            )
        if not next_paths:
            raise ValueError(
                f"no full-step lane repair is possible at beat {event.beat:.6f}"
            )
        paths = next_paths

    best = min(paths.values(), key=_path_key)
    choices: list[_Choice] = []
    cursor: _Path | None = best
    while cursor is not None and cursor.choice is not None:
        choices.append(cursor.choice)
        cursor = cursor.previous
    choices.reverse()
    if best.depth != len(ordered) or len(choices) != len(ordered):
        raise ValueError("footwork decoder lost event alignment")
    output: list[ChartEvent] = []
    lanes_reassigned = 0
    feet_assigned = 0
    notes_removed = 0
    segments_repaired = 0
    segment_changed = False
    previous_time = None
    for event, choice in zip(ordered, choices, strict=True):
        if previous_time is not None and event.time_sec - previous_time >= _RESET_GAP_SEC:
            segments_repaired += int(segment_changed)
            segment_changed = False
        previous_time = event.time_sec
        selected = {
            index: (lane, foot)
            for index, lane, foot in zip(
                choice.playable_indices, choice.lanes, choice.feet, strict=True
            )
        }
        updated_notes = []
        for index, note in enumerate(event.notes):
            if note.type == "mine":
                updated_notes.append(note)
                continue
            if index not in selected:
                notes_removed += 1
                continue
            lane, foot = selected[index]
            lanes_reassigned += int(lane != note.lane)
            feet_assigned += int(note.foot != foot)
            segment_changed = segment_changed or lane != note.lane
            updated_notes.append(note.model_copy(update={"lane": lane, "foot": foot}))
        segment_changed = segment_changed or len(selected) < len(_playable_positions(event))
        output.append(event.model_copy(update={"notes": updated_notes}))
    segments_repaired += int(segment_changed)
    return output, FootworkRepairReport(
        lanes_reassigned=lanes_reassigned,
        feet_assigned=feet_assigned,
        notes_removed=notes_removed,
        segments_repaired=segments_repaired,
    )
