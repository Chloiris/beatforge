from __future__ import annotations

from bisect import bisect_right

from .models import TempoPoint


class TempoTimeline:
    """Piecewise-constant StepMania tempo map with exact beat/time conversion."""

    def __init__(self, changes: list[tuple[float, float]], offset_sec: float = 0.0):
        normalized: dict[float, float] = {}
        for beat, bpm in changes:
            if beat < 0 or bpm <= 0:
                continue
            normalized[float(beat)] = float(bpm)
        if not normalized:
            raise ValueError("tempo map is empty")
        ordered = sorted(normalized.items())
        if ordered[0][0] != 0.0:
            ordered.insert(0, (0.0, ordered[0][1]))
        self.offset_sec = float(offset_sec)
        self.beats = [item[0] for item in ordered]
        self.bpms = [item[1] for item in ordered]
        self.times = [-self.offset_sec]
        for index in range(1, len(ordered)):
            beat_delta = self.beats[index] - self.beats[index - 1]
            self.times.append(self.times[-1] + beat_delta * 60.0 / self.bpms[index - 1])

    @property
    def primary_bpm(self) -> float:
        return self.bpms[0]

    def beat_to_time(self, beat: float) -> float:
        index = max(0, bisect_right(self.beats, float(beat)) - 1)
        return self.times[index] + (float(beat) - self.beats[index]) * 60.0 / self.bpms[index]

    def time_to_beat(self, time_sec: float) -> float:
        index = max(0, bisect_right(self.times, float(time_sec)) - 1)
        return self.beats[index] + (float(time_sec) - self.times[index]) * self.bpms[index] / 60.0

    def bpm_at_beat(self, beat: float) -> float:
        return self.bpms[max(0, bisect_right(self.beats, float(beat)) - 1)]

    def points(self) -> list[TempoPoint]:
        return [
            TempoPoint(beat=beat, bpm=bpm, time_sec=time_sec)
            for beat, bpm, time_sec in zip(self.beats, self.bpms, self.times, strict=True)
        ]
