#!/usr/bin/env python3
"""Generate deterministic, copyright-free BeatForge demo tracks and onset truth.

Ground truth is produced by the synthesizer's score. The analyzer deliberately has
no dependency on this module or the generated truth files.
"""

from __future__ import annotations

import argparse
import json
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

SAMPLE_RATE = 44_100
CHANNELS = 2
SEED = 20_260_718


@dataclass
class Event:
    sample: int
    layers: set[str] = field(default_factory=set)
    strength: float = 0.0


@dataclass(frozen=True)
class DemoSpec:
    slug: str
    title: str
    english_title: str
    artist: str
    genre: str
    bpm: float
    bars: int
    seed_offset: int
    composer: Callable[[np.ndarray, np.random.Generator, list[Event], "DemoSpec"], None]

    @property
    def beat_samples(self) -> float:
        return SAMPLE_RATE * 60.0 / self.bpm

    @property
    def sample_count(self) -> int:
        return round(self.bars * 4 * self.beat_samples)


def add_event(events: list[Event], sample: int, layer: str, strength: float) -> None:
    sample = int(max(0, sample))
    for event in reversed(events[-8:]):
        if event.sample == sample:
            event.layers.add(layer)
            event.strength = max(event.strength, strength)
            return
    events.append(Event(sample=sample, layers={layer}, strength=float(strength)))


def place(signal: np.ndarray, sample: int, sound: np.ndarray, gain: float = 1.0) -> None:
    if sample >= signal.size or sample + sound.size <= 0:
        return
    source_start = max(0, -sample)
    target_start = max(0, sample)
    length = min(sound.size - source_start, signal.size - target_start)
    signal[target_start : target_start + length] += sound[source_start : source_start + length] * gain


def kick(rng: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    duration = 0.34
    length = round(SAMPLE_RATE * duration)
    time = np.arange(length) / SAMPLE_RATE
    phase = 2 * np.pi * (47 * time + (105 - 47) * (1 - np.exp(-time * 38)) / 38)
    body = np.sin(phase) * np.exp(-time * 13)
    click_noise = rng.normal(0, 1, length)
    click = np.concatenate(([click_noise[0]], np.diff(click_noise))) * np.exp(-time * 180)
    return strength * (0.88 * body + 0.13 * click)


def snare(rng: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    duration = 0.24
    length = round(SAMPLE_RATE * duration)
    time = np.arange(length) / SAMPLE_RATE
    noise = rng.normal(0, 1, length)
    high = np.concatenate(([noise[0]], np.diff(noise)))
    tone = np.sin(2 * np.pi * 190 * time) * np.exp(-time * 26)
    return strength * (0.52 * high * np.exp(-time * 22) + 0.38 * tone)


def hat(rng: np.random.Generator, strength: float = 1.0, open_hat: bool = False) -> np.ndarray:
    duration = 0.18 if open_hat else 0.065
    length = round(SAMPLE_RATE * duration)
    time = np.arange(length) / SAMPLE_RATE
    noise = rng.normal(0, 1, length)
    high = np.concatenate(([noise[0]], np.diff(noise)))
    metallic = sum(np.sin(2 * np.pi * f * time) for f in (6_700, 8_450, 10_300)) / 3
    decay = 24 if open_hat else 72
    return strength * (0.34 * high + 0.27 * metallic) * np.exp(-time * decay)


def chug(rng: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    duration = 0.2
    length = round(SAMPLE_RATE * duration)
    time = np.arange(length) / SAMPLE_RATE
    carrier = (
        np.sin(2 * np.pi * 82.4 * time)
        + 0.5 * np.sin(2 * np.pi * 164.8 * time)
        + 0.23 * np.sin(2 * np.pi * 329.6 * time)
    )
    noise = rng.normal(0, 0.12, length)
    return strength * np.tanh(2.8 * (carrier + noise)) * np.exp(-time * 18)


def glass(rng: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    duration = 0.55
    length = round(SAMPLE_RATE * duration)
    time = np.arange(length) / SAMPLE_RATE
    freqs = (2_350, 3_610, 5_420, 7_930, 10_170)
    phases = rng.uniform(0, 2 * np.pi, len(freqs))
    resonances = sum(
        np.sin(2 * np.pi * freq * time + phase) * np.exp(-time * (8 + index * 1.3))
        for index, (freq, phase) in enumerate(zip(freqs, phases))
    ) / len(freqs)
    attack = np.minimum(1.0, time * 4_000)
    click = rng.normal(0, 0.3, length) * np.exp(-time * 120)
    return strength * (resonances * attack + click)


def accent(rng: np.random.Generator, strength: float = 1.0) -> np.ndarray:
    components = [kick(rng, 0.95), snare(rng, 0.78), hat(rng, 0.8, True)]
    length = max(component.size for component in components)
    result = np.zeros(length, dtype=np.float64)
    for component in components:
        result[: component.size] += component
    return strength * result


def beat_sample(spec: DemoSpec, beat: float) -> int:
    return round(beat * spec.beat_samples)


def compose_neon(signal: np.ndarray, rng: np.random.Generator, events: list[Event], spec: DemoSpec) -> None:
    total_beats = spec.bars * 4
    for eighth in range(total_beats * 2):
        beat = eighth / 2
        sample = beat_sample(spec, beat)
        level = 0.38 if eighth % 2 else 0.5
        place(signal, sample, hat(rng, level, eighth % 8 == 7))
        add_event(events, sample, "hat", level)
    for beat in range(total_beats):
        sample = beat_sample(spec, beat)
        place(signal, sample, kick(rng, 0.82))
        add_event(events, sample, "kick", 0.86)
        if beat % 4 in (1, 3):
            place(signal, sample, snare(rng, 0.74))
            add_event(events, sample, "snare", 0.82)
    for bar in (3, 7, 11, 15):
        sample = beat_sample(spec, bar * 4)
        place(signal, sample, accent(rng, 0.8))
        add_event(events, sample, "accent", 1.0)
    for sixteenth in range(56 * 4, total_beats * 4):
        beat = sixteenth / 4
        if sixteenth % 2:
            sample = beat_sample(spec, beat)
            level = 0.48 + 0.08 * (sixteenth % 4)
            place(signal, sample, snare(rng, level))
            add_event(events, sample, "fill", 0.75)
    time = np.arange(signal.size) / SAMPLE_RATE
    signal += 0.045 * np.sin(2 * np.pi * 55 * time) * (0.65 + 0.35 * np.sin(2 * np.pi * 0.14 * time))


def compose_iron(signal: np.ndarray, rng: np.random.Generator, events: list[Event], spec: DemoSpec) -> None:
    total_beats = spec.bars * 4
    for eighth in range(total_beats * 2):
        beat = eighth / 2
        bar = int(beat // 4)
        if 16 <= bar < 20 and eighth % 2:
            continue
        sample = beat_sample(spec, beat)
        kick_level = 0.67 if eighth % 2 else 0.78
        place(signal, sample, kick(rng, kick_level))
        add_event(events, sample, "kick", 0.78 if eighth % 2 else 0.86)
        if eighth % 2 == 0 and int(beat) % 4 in (1, 3):
            place(signal, sample, snare(rng, 0.78))
            add_event(events, sample, "snare", 0.9)
        if eighth % 2 == 0 or (bar % 4 == 2):
            place(signal, sample, chug(rng, 0.44 + 0.1 * (eighth % 2 == 0)))
            add_event(events, sample, "chug", 0.74)
        place(signal, sample, hat(rng, 0.22 if eighth % 2 else 0.3))
        add_event(events, sample, "hat", 0.44)
    # Double-kick runs and syncopated chugs. The 16th-note spacing stays well above merge windows.
    for bar in (2, 3, 6, 10, 14, 20, 22, 23):
        for step in range(8, 16):
            beat = bar * 4 + step / 4
            sample = beat_sample(spec, beat)
            place(signal, sample, kick(rng, 0.69))
            add_event(events, sample, "double_kick", 0.82)
    for bar in (5, 9, 13, 21):
        for position in (0.75, 2.5, 3.25):
            sample = beat_sample(spec, bar * 4 + position)
            place(signal, sample, chug(rng, 0.68))
            add_event(events, sample, "syncopated_chug", 0.8)
    for bar in (0, 4, 8, 12, 16, 20, 23):
        sample = beat_sample(spec, bar * 4)
        place(signal, sample, accent(rng, 0.73))
        add_event(events, sample, "accent", 1.0)
    time = np.arange(signal.size) / SAMPLE_RATE
    signal += 0.038 * np.tanh(2 * np.sin(2 * np.pi * 41.2 * time))


def compose_glass(signal: np.ndarray, rng: np.random.Generator, events: list[Event], spec: DemoSpec) -> None:
    for bar in range(spec.bars):
        quiet = 0.28 if bar < 2 else 1.0
        positions = (0.0, 2.5) if bar % 2 == 0 else (1.0, 3.25)
        for index, position in enumerate(positions):
            sample = beat_sample(spec, bar * 4 + position)
            level = quiet * (0.48 + 0.2 * ((bar + index) % 3))
            if index == 0 and bar % 3 == 0:
                place(signal, sample, kick(rng, level * 0.72))
                add_event(events, sample, "soft_low", max(0.3, level))
            else:
                place(signal, sample, glass(rng, level))
                add_event(events, sample, "glass", max(0.28, level))
    for bar in (6, 9):
        start = bar * 4 + 2
        for triplet in range(6):
            sample = beat_sample(spec, start + triplet / 3)
            level = 0.5 + triplet * 0.055
            place(signal, sample, glass(rng, level))
            add_event(events, sample, "triplet_glass", 0.64 + triplet * 0.04)
    for bar in (4, 7, 10):
        sample = beat_sample(spec, bar * 4)
        place(signal, sample, glass(rng, 0.86))
        add_event(events, sample, "glass_accent", 0.91)
    final_sample = beat_sample(spec, spec.bars * 4 - 1)
    place(signal, final_sample, accent(rng, 0.95))
    place(signal, final_sample, glass(rng, 0.9))
    add_event(events, final_sample, "final_accent", 1.0)
    time = np.arange(signal.size) / SAMPLE_RATE
    pad = sum(np.sin(2 * np.pi * freq * time) for freq in (110, 164.81, 220)) / 3
    signal += 0.025 * pad * np.sin(np.pi * np.minimum(1, time / 4))


SPECS = (
    DemoSpec("neon-pulse", "霓虹脉冲", "Neon Pulse", "BeatForge Lab", "Electronic", 128.0, 16, 1, compose_neon),
    DemoSpec("iron-rift", "钢铁断层", "Iron Rift", "Synthetic Foundry", "Metal / Industrial", 174.0, 24, 2, compose_iron),
    DemoSpec("glass-tide", "玻璃潮汐", "Glass Tide", "Quiet Machines", "Ambient Percussion", 96.0, 12, 3, compose_glass),
)


def classify_event(layers: set[str]) -> str:
    low = any(name in layers for name in ("kick", "double_kick", "soft_low"))
    high = any("glass" in name or name in ("hat",) for name in layers)
    mid = any(name in layers for name in ("snare", "fill", "chug", "syncopated_chug"))
    if "accent" in " ".join(layers) or sum((low, mid, high)) >= 2:
        return "full_band_accent"
    if low:
        return "low_hit"
    if mid:
        return "mid_hit"
    return "high_hit"


def write_wav(path: Path, mono: np.ndarray) -> None:
    mono = mono - float(np.mean(mono))
    peak = float(np.max(np.abs(mono))) or 1.0
    mono = np.clip(mono * (0.92 / peak), -1.0, 1.0)
    # Deterministic stereo width without changing the onset sample.
    delayed = np.concatenate((np.zeros(13), mono[:-13]))
    stereo = np.column_stack((mono, 0.965 * mono + 0.035 * delayed))
    pcm = np.round(stereo * 32_767).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(CHANNELS)
        output.setsampwidth(2)
        output.setframerate(SAMPLE_RATE)
        output.writeframes(pcm.tobytes())


def cover_svg(spec: DemoSpec) -> str:
    palettes = {
        "neon-pulse": ("#07131d", "#1fd1c2", "#b76cff"),
        "iron-rift": ("#120e0d", "#f26b38", "#d7d1c5"),
        "glass-tide": ("#09131a", "#73b9d4", "#c9e7e4"),
    }
    background, primary, secondary = palettes[spec.slug]
    bars = "".join(
        f'<rect x="{94 + i * 32}" y="{170 - (i % 5) * 17}" width="12" height="{50 + (i % 5) * 34}" rx="6" fill="{primary}" opacity="{0.35 + i % 3 * 0.2}"/>'
        for i in range(12)
    )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="640" height="640" viewBox="0 0 640 640" role="img" aria-label="{spec.title} cover">
<rect width="640" height="640" rx="40" fill="{background}"/>
<circle cx="492" cy="132" r="188" fill="{primary}" opacity=".08"/>
<path d="M0 434 C112 366 204 502 328 422 S540 330 640 390 V640 H0Z" fill="{secondary}" opacity=".08"/>
{bars}
<text x="72" y="472" fill="#f4f7f7" font-family="system-ui,sans-serif" font-size="48" font-weight="700">{spec.title}</text>
<text x="74" y="518" fill="{secondary}" font-family="system-ui,sans-serif" font-size="24" letter-spacing="2">{spec.english_title.upper()}</text>
<text x="74" y="565" fill="#9aa9ad" font-family="system-ui,sans-serif" font-size="20">{spec.bpm:g} BPM · {spec.genre}</text>
</svg>'''


def generate(spec: DemoSpec, output_root: Path, force: bool) -> dict[str, object]:
    wav_path = output_root / "storage" / "demo" / f"{spec.slug}.wav"
    truth_path = output_root / "storage" / "demo" / f"{spec.slug}.ground-truth.json"
    cover_path = output_root / "storage" / "covers" / f"{spec.slug}.svg"
    if not force and wav_path.exists() and truth_path.exists() and cover_path.exists():
        return json.loads(truth_path.read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED + spec.seed_offset)
    signal = np.zeros(spec.sample_count, dtype=np.float64)
    events: list[Event] = []
    spec.composer(signal, rng, events, spec)
    # Layer composers intentionally run in independent passes. Consolidate every
    # simultaneous layer globally so one musical attack is one truth event.
    consolidated: dict[int, Event] = {}
    for event in events:
        existing = consolidated.get(event.sample)
        if existing is None:
            consolidated[event.sample] = Event(
                sample=event.sample,
                layers=set(event.layers),
                strength=event.strength,
            )
        else:
            existing.layers.update(event.layers)
            existing.strength = max(existing.strength, event.strength)
    events = sorted(consolidated.values(), key=lambda event: event.sample)
    write_wav(wav_path, signal)
    truth: dict[str, object] = {
        "schemaVersion": "1.0",
        "slug": spec.slug,
        "title": spec.title,
        "englishTitle": spec.english_title,
        "artist": spec.artist,
        "genre": spec.genre,
        "sampleRate": SAMPLE_RATE,
        "channels": CHANNELS,
        "sampleCount": spec.sample_count,
        "durationSec": spec.sample_count / SAMPLE_RATE,
        "bpm": spec.bpm,
        "beatOffsetSample": 0,
        "timeSignatureNumerator": 4,
        "timeSignatureDenominator": 4,
        "seed": SEED + spec.seed_offset,
        "audioFile": str(wav_path.relative_to(output_root)),
        "coverFile": str(cover_path.relative_to(output_root)),
        "onsets": [
            {
                "sample": event.sample,
                "timeSec": event.sample / SAMPLE_RATE,
                "band": classify_event(event.layers),
                "strength": round(event.strength, 6),
                "layers": sorted(event.layers),
            }
            for event in events
            if event.sample < spec.sample_count
        ],
    }
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.write_text(json.dumps(truth, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cover_path.parent.mkdir(parents=True, exist_ok=True)
    cover_path.write_text(cover_svg(spec) + "\n", encoding="utf-8")
    return truth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = {"schemaVersion": "1.0", "tracks": []}
    for spec in SPECS:
        truth = generate(spec, args.root, args.force)
        manifest["tracks"].append(
            {
                key: truth[key]
                for key in ("slug", "title", "englishTitle", "artist", "genre", "bpm", "durationSec", "sampleCount", "audioFile", "coverFile")
            }
        )
        print(f"{truth['title']}: {truth['durationSec']:.3f}s, {len(truth['onsets'])} scored onsets")
    manifest_path = args.root / "storage" / "demo" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
