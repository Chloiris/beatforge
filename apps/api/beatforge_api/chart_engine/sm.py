from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Any

from .models import ChartDocument, ChartEvent, ChartNote
from .timing import TempoTimeline

_HEADER = re.compile(r"^\s*#([A-Z0-9_]+)\s*:\s*(.*?)\s*;?\s*$", re.IGNORECASE)
_NOTES_START = re.compile(r"#NOTES\s*:", re.IGNORECASE)
_VALID_ROW = re.compile(r"^[0-4MFKL]+$", re.IGNORECASE)


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for encoding in ("utf-8-sig", "gb18030", "shift_jis"):
        try:
            return raw.decode(encoding).replace("\r\n", "\n").replace("\r", "\n")
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace").replace("\r", "\n")


def _header_tags(text: str) -> dict[str, str]:
    match = _NOTES_START.search(text)
    header = text[: match.start()] if match else text
    tags: dict[str, str] = {}
    for line in header.splitlines():
        parsed = _HEADER.match(line)
        if parsed:
            tags[parsed.group(1).upper()] = parsed.group(2).strip().rstrip(";").strip()
    return tags


def _number(value: str | None, fallback: float = 0.0) -> float:
    try:
        return float((value or "").strip())
    except ValueError:
        return fallback


def _tempo_changes(value: str) -> list[tuple[float, float]]:
    changes: list[tuple[float, float]] = []
    for item in value.rstrip(";, ").split(","):
        if "=" not in item:
            continue
        beat_value, bpm_value = item.split("=", 1)
        try:
            beat = float(beat_value.strip())
            bpm = float(bpm_value.strip().rstrip(";"))
        except ValueError:
            continue
        if beat >= 0 and 0 < bpm <= 10_000_000:
            changes.append((beat, bpm))
    if not changes:
        raise ValueError("SM chart has no valid #BPMS entries")
    return changes


def _notes_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    for match in _NOTES_START.finditer(text):
        start = match.end()
        end = text.find(";", start)
        if end < 0:
            end = len(text)
        blocks.append(text[start:end])
    return blocks


def _clean_notes_block(block: str) -> str:
    lines = []
    for raw_line in block.splitlines():
        line = raw_line.split("//", 1)[0].strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _parse_block_metadata(block: str) -> tuple[str, str, str, int, str]:
    parts = _clean_notes_block(block).split(":", 5)
    if len(parts) != 6:
        raise ValueError("SM #NOTES block is missing metadata fields")
    mode = parts[0].strip().lower()
    description = parts[1].strip()
    difficulty = parts[2].strip() or "Hard"
    try:
        meter = max(1, int(float(parts[3].strip())))
    except ValueError:
        meter = 1
    return mode, description, difficulty, meter, parts[5]


def _event_note_count(events: list[dict[str, Any]]) -> int:
    return sum(len(event["notes"]) for event in events)


def parse_sm(
    path: str | Path,
    *,
    chart_index: int = 0,
    source_group: str | None = None,
    document_id: str | None = None,
) -> ChartDocument:
    """Parse a real StepMania pump chart into absolute timestamped events."""

    source = Path(path).expanduser().resolve()
    text = _read_text(source)
    tags = _header_tags(text)
    changes = _tempo_changes(tags.get("BPMS", ""))
    offset = _number(tags.get("OFFSET"), 0.0)
    timeline = TempoTimeline(changes, offset)
    blocks = _notes_blocks(text)
    if not blocks:
        raise ValueError("SM file contains no #NOTES block")
    if chart_index < 0 or chart_index >= len(blocks):
        raise IndexError("SM chart index is out of range")
    mode, _description, difficulty, meter, note_data = _parse_block_metadata(blocks[chart_index])

    measures: list[list[str]] = []
    lane_count = 5 if mode == "pump-single" else 10 if mode == "pump-double" else 0
    for raw_measure in note_data.split(","):
        rows: list[str] = []
        for raw_line in raw_measure.splitlines():
            line = raw_line.split("//", 1)[0].strip().upper()
            if not line or not _VALID_ROW.fullmatch(line):
                continue
            if lane_count == 0:
                lane_count = len(line)
            if len(line) != lane_count:
                raise ValueError(
                    f"inconsistent SM row width in {source.name}: "
                    f"expected {lane_count}, got {len(line)}"
                )
            rows.append(line)
        measures.append(rows)
    if lane_count not in {5, 10}:
        raise ValueError(f"unsupported SM lane count: {lane_count}")
    normalized_mode = "pump-single" if lane_count == 5 else "pump-double"

    event_rows: list[dict[str, Any]] = []
    active_holds: dict[int, dict[str, Any]] = {}
    for measure_index, rows in enumerate(measures):
        row_count = len(rows)
        if row_count == 0:
            continue
        for row_index, row in enumerate(rows):
            beat = measure_index * 4.0 + row_index * 4.0 / row_count
            time_sec = timeline.beat_to_time(beat)
            row_notes: list[dict[str, Any]] = []
            for lane, symbol in enumerate(row):
                if symbol == "0":
                    continue
                if symbol in {"2", "4"}:
                    previous = active_holds.get(lane)
                    if previous is not None:
                        previous["type"] = "tap"
                        previous.pop("end_time_sec", None)
                        previous.pop("end_beat", None)
                    note = {
                        "lane": lane,
                        "type": "hold",
                        "end_time_sec": None,
                        "end_beat": None,
                        "source": "sm",
                        "confidence": 1.0,
                    }
                    row_notes.append(note)
                    active_holds[lane] = note
                elif symbol == "3":
                    active = active_holds.pop(lane, None)
                    if active is not None:
                        active["end_time_sec"] = max(time_sec, timeline.beat_to_time(beat))
                        active["end_beat"] = beat
                elif symbol == "M":
                    row_notes.append(
                        {"lane": lane, "type": "mine", "source": "sm", "confidence": 1.0}
                    )
                else:
                    row_notes.append(
                        {"lane": lane, "type": "tap", "source": "sm", "confidence": 1.0}
                    )
            if row_notes:
                event_rows.append(
                    {
                        "time_sec": time_sec,
                        "beat": beat,
                        "measure": measure_index,
                        "subdivision": row_count,
                        "row_index": row_index,
                        "notes": row_notes,
                    }
                )

    # Corrupt/unclosed hold starts should remain visible and playable as taps.
    for active in active_holds.values():
        active["type"] = "tap"
        active.pop("end_time_sec", None)
        active.pop("end_beat", None)

    events: list[ChartEvent] = []
    for raw in event_rows:
        notes = [ChartNote.model_validate(note) for note in raw.pop("notes")]
        events.append(ChartEvent(**raw, notes=notes))
    events.sort(key=lambda event: (event.time_sec, event.beat))
    duration = max(
        (
            max(
                [event.time_sec]
                + [note.end_time_sec for note in event.notes if note.end_time_sec is not None]
            )
            for event in events
        ),
        default=0.0,
    )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    chart = ChartDocument(
        id=document_id or digest,
        title=tags.get("TITLE", source.stem),
        artist=tags.get("ARTIST", ""),
        music=tags.get("MUSIC", ""),
        source_group=source_group,
        source_path=str(source),
        mode=normalized_mode,
        lane_count=lane_count,
        difficulty=difficulty,
        meter=meter,
        bpm=timeline.primary_bpm,
        offset_sec=offset,
        duration_sec=max(0.0, duration),
        measure_count=len(measures),
        tempo_map=timeline.points(),
        events=events,
        generator="sm_parser",
    )
    from .statistics import chart_statistics

    return chart.model_copy(update={"statistics": chart_statistics(chart)})


def _safe_tag(value: str) -> str:
    return value.replace(";", "").replace("\r", " ").replace("\n", " ").strip()


def _put_symbol(row: list[str], lane: int, symbol: str) -> None:
    priority = {"0": 0, "M": 1, "1": 2, "3": 3, "2": 4}
    if priority.get(symbol, 0) >= priority.get(row[lane], 0):
        row[lane] = symbol


def export_sm(chart: ChartDocument, *, rows_per_measure: int | None = None) -> str:
    """Export a chart as a deterministic StepMania SM document."""

    if rows_per_measure is None:
        rows_per_measure = 4
        for subdivision in {event.subdivision for event in chart.events}:
            rows_per_measure = math.lcm(rows_per_measure, subdivision)
    if rows_per_measure < 4 or rows_per_measure % 4:
        raise ValueError("rows_per_measure must be a multiple of four")
    last_beat = max(
        (
            max([event.beat] + [note.end_beat for note in event.notes if note.end_beat is not None])
            for event in chart.events
        ),
        default=0.0,
    )
    measure_count = max(1, int(math.floor(last_beat / 4.0)) + 1)
    grid = [
        [["0"] * chart.lane_count for _ in range(rows_per_measure)] for _ in range(measure_count)
    ]

    def location(beat: float) -> tuple[int, int]:
        absolute_row = max(0, int(round(beat * rows_per_measure / 4.0)))
        measure, row = divmod(absolute_row, rows_per_measure)
        while measure >= len(grid):
            grid.append([["0"] * chart.lane_count for _ in range(rows_per_measure)])
        return measure, row

    for event in chart.events:
        measure, row = location(event.beat)
        for note in event.notes:
            if note.type == "mine":
                _put_symbol(grid[measure][row], note.lane, "M")
            elif note.type == "hold" and note.end_beat is not None:
                _put_symbol(grid[measure][row], note.lane, "2")
                end_measure, end_row = location(note.end_beat)
                _put_symbol(grid[end_measure][end_row], note.lane, "3")
            else:
                _put_symbol(grid[measure][row], note.lane, "1")

    bpms = ",".join(f"{point.beat:.6f}={point.bpm:.6f}" for point in chart.tempo_map)
    display_values = [point.bpm for point in chart.tempo_map if point.bpm <= 500]
    display_min = min(display_values, default=chart.bpm)
    display_max = max(display_values, default=chart.bpm)
    display = (
        f"{display_min:.3f}"
        if math.isclose(display_min, display_max)
        else f"{display_min:.3f}:{display_max:.3f}"
    )
    measures = []
    for index, rows in enumerate(grid):
        body = "\n".join("".join(row) for row in rows)
        suffix = "," if index < len(grid) - 1 else ";"
        measures.append(f"  // measure {index + 1}\n{body}\n{suffix}")
    notes_body = "\n".join(measures)
    radar = "0.000,0.000,0.000,0.000,0.000"
    return (
        f"#TITLE:{_safe_tag(chart.title)};\n"
        f"#SUBTITLE:;\n"
        f"#ARTIST:{_safe_tag(chart.artist)};\n"
        f"#MUSIC:{_safe_tag(chart.music)};\n"
        f"#DISPLAYBPM:{display};\n"
        f"#OFFSET:{chart.offset_sec:.6f};\n"
        f"#BPMS:{bpms};\n"
        f"#NOTES:\n"
        f"     {chart.mode}:\n"
        f"     BeatForge AI:\n"
        f"     {chart.difficulty}:\n"
        f"     {chart.meter}:\n"
        f"     {radar}:\n"
        f"{notes_body}\n"
    )
