from __future__ import annotations

from decimal import Decimal
from fractions import Fraction


def sample_to_time(sample: int, sample_rate: int) -> float:
    if sample < 0 or sample_rate <= 0:
        raise ValueError("sample must be non-negative and sample_rate must be positive")
    return sample / sample_rate


def time_to_sample(time_sec: float, sample_rate: int) -> int:
    if time_sec < 0 or sample_rate <= 0:
        raise ValueError("time_sec must be non-negative and sample_rate must be positive")
    value = Decimal(str(time_sec)) * sample_rate
    return int(value.to_integral_value(rounding="ROUND_HALF_UP"))


def map_sample_index(sample: int, source_rate: int, target_rate: int) -> int:
    if sample < 0 or source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample must be non-negative and sample rates must be positive")
    numerator = sample * target_rate
    return (2 * numerator + source_rate) // (2 * source_rate)


def samples_per_beat(sample_rate: int, bpm: float) -> Fraction:
    if sample_rate <= 0 or bpm <= 0:
        raise ValueError("sample_rate and bpm must be positive")
    bpm_fraction = Fraction(Decimal(str(bpm)))
    return Fraction(sample_rate * 60, 1) / bpm_fraction


def round_fraction(value: Fraction) -> int:
    if value >= 0:
        return (2 * value.numerator + value.denominator) // (2 * value.denominator)
    return -round_fraction(-value)


def grid_sample(
    *,
    beat_index: int,
    subdivision_index: int,
    subdivisions_per_beat: int,
    sample_rate: int,
    bpm: float,
    beat_offset_sample: int,
) -> int:
    if subdivisions_per_beat <= 0:
        raise ValueError("subdivisions_per_beat must be positive")
    position = Fraction(beat_offset_sample, 1) + samples_per_beat(sample_rate, bpm) * (
        Fraction(beat_index, 1) + Fraction(subdivision_index, subdivisions_per_beat)
    )
    return round_fraction(position)


def nearest_grid_sample(
    sample: int,
    *,
    sample_rate: int,
    bpm: float,
    beat_offset_sample: int,
    subdivisions_per_beat: int,
) -> int:
    step = samples_per_beat(sample_rate, bpm) / subdivisions_per_beat
    relative = Fraction(sample - beat_offset_sample, 1) / step
    index = round_fraction(relative)
    return round_fraction(Fraction(beat_offset_sample, 1) + index * step)
