from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable

from .models import ChartDocument, ChartStatistics, CorpusStatistics

_BIG_SPIN = (3, 2, 4, 2, 1, 2, 0, 2)
_SMALL_SPIN = (3, 2, 4) * 3


def _peak_nps(times: list[float], window_sec: float = 1.0) -> float:
    if not times:
        return 0.0
    ordered = sorted(times)
    left = 0
    peak = 0
    for right, time_sec in enumerate(ordered):
        while time_sec - ordered[left] > window_sec:
            left += 1
        peak = max(peak, right - left + 1)
    return peak / window_sec


def _primary_lane_sequence(chart: ChartDocument) -> list[int]:
    return [
        event.notes[0].lane
        for event in chart.events
        if len(event.notes) == 1 and event.notes[0].type != "mine"
    ]


def _adjacent_single_lane_transitions(chart: ChartDocument) -> Iterable[tuple[int, int]]:
    """Yield only transitions that are adjacent in the original chart.

    Projecting a chart to single rows first used to bridge across a removed
    jump. That taught lower-level generation transitions which only existed
    because a two-foot reset row had been discarded.
    """

    for previous, current in zip(chart.events, chart.events[1:], strict=False):
        previous_playable = [note for note in previous.notes if note.type != "mine"]
        current_playable = [note for note in current.notes if note.type != "mine"]
        if len(previous_playable) == len(current_playable) == 1:
            yield previous_playable[0].lane, current_playable[0].lane


def _pattern_count(sequence: list[int], pattern: tuple[int, ...]) -> int:
    if len(sequence) < len(pattern):
        return 0
    return sum(
        tuple(sequence[index : index + len(pattern)]) == pattern
        for index in range(len(sequence) - len(pattern) + 1)
    )


def _same_foot_runs(chart: ChartDocument) -> list[int]:
    # Pump panels 0/1 are left, 3/4 are right, and center alternates. This is
    # an intentionally conservative proxy, not a claim about a player's feet.
    runs: list[int] = []
    current_foot: str | None = None
    current_length = 0
    center_next = "left"
    for event in chart.events:
        playable = [note for note in event.notes if note.type != "mine"]
        if len(playable) != 1:
            if current_length:
                runs.append(current_length)
            current_foot = None
            current_length = 0
            continue
        lane = playable[0].lane % 5
        foot = playable[0].foot
        if foot is None:
            if lane in {0, 1}:
                foot = "left"
            elif lane in {3, 4}:
                foot = "right"
            else:
                foot = center_next
                center_next = "right" if center_next == "left" else "left"
        if foot == current_foot:
            current_length += 1
        else:
            if current_length:
                runs.append(current_length)
            current_foot = foot
            current_length = 1
    if current_length:
        runs.append(current_length)
    return runs


def chart_statistics(chart: ChartDocument) -> ChartStatistics:
    lane_counts = [0] * chart.lane_count
    hold_count = jump_count = mine_count = note_count = 0
    note_times: list[float] = []
    measure_counts: Counter[int] = Counter()
    for event in chart.events:
        playable = 0
        for note in event.notes:
            lane_counts[note.lane] += 1
            measure_counts[event.measure] += 1
            note_count += 1
            note_times.append(event.time_sec)
            if note.type == "hold":
                hold_count += 1
                playable += 1
            elif note.type == "mine":
                mine_count += 1
            else:
                playable += 1
        if playable >= 2:
            jump_count += 1
    event_count = len(chart.events)
    duration = max(
        chart.duration_sec,
        max(note_times, default=0.0) - min(0.0, min(note_times, default=0.0)),
    )
    denominator = max(note_count, 1)
    sequence = _primary_lane_sequence(chart)
    foot_runs = _same_foot_runs(chart)
    foot_steps = sum(foot_runs)
    return ChartStatistics(
        note_count=note_count,
        event_count=event_count,
        hold_count=hold_count,
        jump_count=jump_count,
        mine_count=mine_count,
        duration_sec=max(0.0, duration),
        nps_average=note_count / duration if duration > 0 else 0.0,
        nps_peak=_peak_nps(note_times),
        single_ratio=sum(len(event.notes) == 1 for event in chart.events) / max(event_count, 1),
        jump_ratio=jump_count / max(event_count, 1),
        hold_ratio=hold_count / denominator,
        lane_counts=lane_counts,
        measure_densities=[
            float(measure_counts.get(index, 0)) for index in range(chart.measure_count)
        ],
        same_foot_runs=foot_runs,
        foot_switch_ratio=max(0, len(foot_runs) - 1) / max(foot_steps - 1, 1),
        small_spin_count=_pattern_count(sequence, _SMALL_SPIN),
        big_spin_count=_pattern_count(sequence, _BIG_SPIN),
    )


def corpus_statistics(charts: Iterable[ChartDocument]) -> CorpusStatistics:
    values = list(charts)
    single = [chart for chart in values if chart.mode == "pump-single"]
    double = [chart for chart in values if chart.mode == "pump-double"]
    transitions = [[1.0 for _ in range(5)] for _ in range(5)]
    groups: Counter[str] = Counter()
    songs: set[str] = set()
    meter_values: defaultdict[int, list[ChartStatistics]] = defaultdict(list)
    total_notes = 0
    total_duration = 0.0
    for chart in values:
        stats = chart.statistics or chart_statistics(chart)
        total_notes += stats.note_count
        total_duration += stats.duration_sec
        groups[chart.source_group or "UNKNOWN"] += 1
        songs.add(chart.music or chart.title)
        meter_values[chart.meter].append(stats)
        if chart.mode != "pump-single":
            continue
        for previous, current in _adjacent_single_lane_transitions(chart):
            transitions[previous][current] += 1.0
    for row in transitions:
        total = sum(row)
        for index in range(5):
            row[index] = row[index] / total
    profiles: dict[str, dict[str, float]] = {}
    for meter, items in sorted(meter_values.items()):
        profiles[str(meter)] = {
            "charts": float(len(items)),
            "averageNps": sum(item.nps_average for item in items) / len(items),
            "peakNps": sum(item.nps_peak for item in items) / len(items),
            "jumpRatio": sum(item.jump_ratio for item in items) / len(items),
            "holdRatio": sum(item.hold_ratio for item in items) / len(items),
        }
    meters = [chart.meter for chart in values]
    return CorpusStatistics(
        chart_count=len(values),
        song_count=len(songs),
        single_chart_count=len(single),
        single_song_count=len({chart.music or chart.title for chart in single}),
        double_chart_count=len(values) - len(single),
        double_song_count=len({chart.music or chart.title for chart in double}),
        difficulty_min=min(meters, default=0),
        difficulty_max=max(meters, default=0),
        total_notes=total_notes,
        total_duration_sec=total_duration,
        average_nps=total_notes / total_duration if total_duration > 0 else 0.0,
        groups=dict(groups),
        lane_transition_probabilities=transitions,
        meter_profiles=profiles,
    )
