from __future__ import annotations

from beatforge_api.timing import (
    grid_sample,
    map_sample_index,
    nearest_grid_sample,
    sample_to_time,
    samples_per_beat,
    time_to_sample,
)


def test_sample_time_round_trip() -> None:
    for sample_rate in (44_100, 48_000, 96_000):
        for sample in (0, 1, 22_050, 592_120, 12_345_678):
            assert time_to_sample(sample_to_time(sample, sample_rate), sample_rate) == sample


def test_maps_between_sample_rates_without_float_drift() -> None:
    assert map_sample_index(44_100, 44_100, 48_000) == 48_000
    mapped = map_sample_index(123_456_789, 48_000, 44_100)
    restored = map_sample_index(mapped, 44_100, 48_000)
    assert abs(restored - 123_456_789) <= 1


def test_bpm_grid_and_offset() -> None:
    assert grid_sample(
        beat_index=1,
        subdivision_index=0,
        subdivisions_per_beat=4,
        sample_rate=48_000,
        bpm=120,
        beat_offset_sample=321,
    ) == 24_321
    assert nearest_grid_sample(
        24_400,
        sample_rate=48_000,
        bpm=120,
        beat_offset_sample=321,
        subdivisions_per_beat=1,
    ) == 24_321


def test_long_grid_has_no_cumulative_drift() -> None:
    per_beat = samples_per_beat(44_100, 128.1)
    for beat_index in (1, 10, 1_000, 100_000):
        sample = grid_sample(
            beat_index=beat_index,
            subdivision_index=0,
            subdivisions_per_beat=4,
            sample_rate=44_100,
            bpm=128.1,
            beat_offset_sample=777,
        )
        expected = round(777 + beat_index * float(per_beat))
        assert abs(sample - expected) <= 1
