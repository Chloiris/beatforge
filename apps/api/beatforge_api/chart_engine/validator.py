from __future__ import annotations

import math
from collections import defaultdict, deque

from .footwork import analyze_no_spin_footwork
from .models import ChartDocument, ValidationIssue, ValidationResult
from .rhythm_policy import density_limit_nps, maximum_subdivision
from .statistics import chart_statistics

_COORDS = ((-1.0, -1.0), (-1.0, 1.0), (0.0, 0.0), (1.0, 1.0), (1.0, -1.0))


def _distance(first: int, second: int) -> float:
    a = _COORDS[first % 5]
    b = _COORDS[second % 5]
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _foot_for_lane(lane: int, center_foot: str) -> str:
    panel = lane % 5
    if panel in {0, 1}:
        return "left"
    if panel in {3, 4}:
        return "right"
    return center_foot


def validate_chart(chart: ChartDocument) -> ValidationResult:
    """Evaluate density, travel, hold conflicts, and rhythm-aware same-foot runs."""

    issues: list[ValidationIssue] = []
    events = sorted(chart.events, key=lambda item: (item.time_sec, item.beat))
    difficulty = min(max(chart.meter, 1), 15)
    density_limit = density_limit_nps(difficulty, bpm=chart.bpm)
    subdivision_limit = maximum_subdivision(difficulty)
    # Lv.11+ accepts the union of 1/16 (quarters of a beat) and 1/24
    # (sixths of a beat), whose common lattice is twelfths of a beat.
    lattice_rows_per_beat = 4.0 if subdivision_limit == 16 else 12.0
    too_fine = [
        event
        for event in events
        if not math.isclose(
            event.beat * lattice_rows_per_beat,
            round(event.beat * lattice_rows_per_beat),
            rel_tol=0.0,
            abs_tol=1e-6,
        )
    ]
    if too_fine:
        first = too_fine[0]
        issues.append(
            ValidationIssue(
                code="SUBDIVISION_TOO_FINE_FOR_LEVEL",
                severity="error",
                message=(
                    f"Level {difficulty} supports at most 1/{subdivision_limit} rhythm; "
                    f"found 1/{first.subdivision}."
                ),
                time_sec=first.time_sec,
                beat=first.beat,
                penalty=20,
            )
        )
    recent: deque[float] = deque()
    peak_two_second_nps = 0.0
    for event in events:
        playable = [note for note in event.notes if note.type != "mine"]
        if len(playable) > 2:
            issues.append(
                ValidationIssue(
                    code="TOO_MANY_SIMULTANEOUS_PANELS",
                    severity="error",
                    message="An event requires more than two simultaneous panels.",
                    time_sec=event.time_sec,
                    beat=event.beat,
                    penalty=20,
                )
            )
        for _note in playable:
            recent.append(event.time_sec)
        while recent and event.time_sec - recent[0] >= 2.0:
            recent.popleft()
        peak_two_second_nps = max(peak_two_second_nps, len(recent) / 2.0)
    if peak_two_second_nps > density_limit:
        issues.append(
            ValidationIssue(
                code="EXTREME_NPS",
                severity="error" if peak_two_second_nps > density_limit * 1.25 else "warning",
                message=(
                    f"Two-second density reaches {peak_two_second_nps:.2f} NPS; "
                    f"the level-{difficulty} limit is {density_limit:.2f}."
                ),
                penalty=min(25.0, (peak_two_second_nps - density_limit) * 4.0),
            )
        )

    holds_by_lane: defaultdict[int, list[tuple[float, float]]] = defaultdict(list)
    for event in events:
        for note in event.notes:
            if note.type != "hold" or note.end_time_sec is None:
                continue
            for start, end in holds_by_lane[note.lane]:
                if event.time_sec < end and note.end_time_sec > start:
                    issues.append(
                        ValidationIssue(
                            code="OVERLAPPING_HOLD",
                            severity="error",
                            message=f"Lane {note.lane + 1} contains overlapping holds.",
                            time_sec=event.time_sec,
                            beat=event.beat,
                            penalty=18,
                        )
                    )
                    break
            holds_by_lane[note.lane].append((event.time_sec, note.end_time_sec))

    disjoint_jump_transitions = 0
    if difficulty <= 10:
        for previous, event in zip(events, events[1:], strict=False):
            previous_playable = [note for note in previous.notes if note.type != "mine"]
            playable = [note for note in event.notes if note.type != "mine"]
            if (
                len(previous_playable) > 1
                and len(playable) > 1
                and event.beat - previous.beat <= 1.0 + 1e-9
                and {note.lane for note in previous_playable}.isdisjoint(
                    note.lane for note in playable
                )
            ):
                disjoint_jump_transitions += 1
                issues.append(
                    ValidationIssue(
                        code="DISJOINT_JUMP_TRANSITION",
                        severity="warning",
                        message=(
                            "Nearby jumps require both feet to reposition at once; "
                            "keep one row as a single panel."
                        ),
                        time_sec=event.time_sec,
                        beat=event.beat,
                        penalty=5,
                    )
                )

    single_events = [
        (event, [note for note in event.notes if note.type != "mine"][0])
        for event in events
        if len([note for note in event.notes if note.type != "mine"]) == 1
    ]
    for (previous_event, previous), (event, note) in zip(
        single_events, single_events[1:], strict=False
    ):
        interval = event.time_sec - previous_event.time_sec
        travel = _distance(previous.lane, note.lane)
        if interval < 0.095 and travel > 2.1:
            issues.append(
                ValidationIssue(
                    code="IMPOSSIBLE_TRAVEL",
                    severity="error",
                    message="A corner-to-corner move is too fast for reliable play.",
                    time_sec=event.time_sec,
                    beat=event.beat,
                    penalty=14,
                )
            )

    current_foot: str | None = None
    run = 0
    center_foot = "left"
    same_foot_violations = 0
    for event, note in single_events:
        foot = note.foot or _foot_for_lane(note.lane, center_foot)
        if note.foot is None and note.lane % 5 == 2:
            center_foot = "right" if center_foot == "left" else "left"
        run = run + 1 if foot == current_foot else 1
        current_foot = foot
        if run == 10 and event.subdivision >= 16:
            same_foot_violations += 1
            issues.append(
                ValidationIssue(
                    code="SUSTAINED_SAME_FOOT_16TH",
                    severity="warning",
                    message="Ten or more 16th-or-faster notes are assigned to the same foot.",
                    time_sec=event.time_sec,
                    beat=event.beat,
                    penalty=8,
                )
            )

    footwork = analyze_no_spin_footwork(events)
    enforce_full_step = not chart.spin_enabled and (
        chart.source_group == "BEATFORGE_GENERATED"
        or chart.generator in {"local_chart_transformer", "real_corpus_profile_rules"}
    )
    if enforce_full_step:
        for violation in footwork.violations:
            issues.append(
                ValidationIssue(
                    code="NO_FULL_ALTERNATING_PATH",
                    severity="error",
                    message=(
                        "Strict left-right alternation reaches a back-facing stance; "
                        "reassign a panel or enable an explicit spin pattern."
                    ),
                    time_sec=violation.time_sec,
                    beat=violation.beat,
                    penalty=20,
                )
            )

    statistics = chart.statistics or chart_statistics(chart)
    penalty = sum(issue.penalty for issue in issues)
    score = max(0.0, min(100.0, 100.0 - penalty))
    return ValidationResult(
        valid=not any(issue.severity == "error" for issue in issues),
        score=score,
        issues=issues,
        metrics={
            "difficulty": difficulty,
            "averageNps": statistics.nps_average,
            "peakNps": statistics.nps_peak,
            "peakTwoSecondNps": peak_two_second_nps,
            "densityLimit": density_limit,
            "densityNoteLimit": int(round(density_limit * 2.0)),
            "maximumSubdivision": subdivision_limit,
            "sameFootViolations": same_foot_violations,
            "holdCount": statistics.hold_count,
            "jumpCount": statistics.jump_count,
            "disjointJumpTransitions": disjoint_jump_transitions,
            "fullStepReachable": footwork.full_step_reachable,
            "unplayableAlternatingSegments": len(footwork.violations),
            "fullStepCheckedEvents": footwork.checked_events,
            "fullStepSegments": footwork.segment_count,
            "maxFacingAngle": footwork.max_abs_heading,
            "crossoverCount": footwork.crossover_count,
            "holdForcedRepeats": footwork.hold_forced_repeats,
        },
    )
