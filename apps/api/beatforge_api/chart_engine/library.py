from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import soundfile as sf

from .models import ChartDocument, ReferenceChartSummary
from .sm import parse_sm
from .statistics import chart_statistics, corpus_statistics

_CHART_CACHE_LOCK = threading.RLock()


@dataclass(frozen=True, slots=True)
class ReferenceAsset:
    id: str
    group: str
    chart_path: Path
    audio_path: Path


@lru_cache(maxsize=256)
def _cached_chart(
    chart_path: str,
    chart_mtime_ns: int,
    chart_size: int,
    audio_path: str,
    audio_mtime_ns: int,
    group: str,
    chart_id: str,
) -> ChartDocument:
    del chart_mtime_ns, chart_size, audio_mtime_ns
    chart = parse_sm(chart_path, source_group=group, document_id=chart_id)
    try:
        audio_duration = float(sf.info(audio_path).duration)
    except (RuntimeError, TypeError, ValueError):
        audio_duration = chart.duration_sec
    updated = chart.model_copy(update={"duration_sec": max(chart.duration_sec, audio_duration)})
    return updated.model_copy(update={"statistics": chart_statistics(updated)})


class ReferenceLibrary:
    """Read-only index over the verified SPEED_CLUB/DEVIL/REMIX corpus."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self._assets = self._scan()
        self._by_id = {asset.id: asset for asset in self._assets}

    def _scan(self) -> list[ReferenceAsset]:
        if not self.root.is_dir():
            return []
        assets: list[ReferenceAsset] = []
        for chart_path in sorted(self.root.glob("SPEED_*/*/*.sm")):
            relative = chart_path.relative_to(self.root).as_posix()
            group = chart_path.relative_to(self.root).parts[0]
            audio_files = sorted(chart_path.parent.glob("*.mp3"))
            if not audio_files:
                continue
            chart_id = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:20]
            assets.append(
                ReferenceAsset(
                    id=chart_id,
                    group=group,
                    chart_path=chart_path.resolve(),
                    audio_path=audio_files[0].resolve(),
                )
            )
        return assets

    def __len__(self) -> int:
        return len(self._assets)

    def assets(self) -> list[ReferenceAsset]:
        return list(self._assets)

    def asset(self, chart_id: str) -> ReferenceAsset:
        try:
            return self._by_id[chart_id]
        except KeyError as exc:
            raise KeyError(f"reference chart not found: {chart_id}") from exc

    def chart(self, chart_id: str) -> ChartDocument:
        asset = self.asset(chart_id)
        chart_stat = asset.chart_path.stat()
        audio_stat = asset.audio_path.stat()
        # Starlette executes synchronous list/statistics routes in worker
        # threads. On a cold start those requests can reach libsndfile/mpg123 at
        # the same time, and a few corpus MP3s are not safe to probe
        # concurrently. Serialize only cache misses/probes; warm reads remain
        # effectively free through the process-local LRU cache.
        with _CHART_CACHE_LOCK:
            chart = _cached_chart(
                str(asset.chart_path),
                chart_stat.st_mtime_ns,
                chart_stat.st_size,
                str(asset.audio_path),
                audio_stat.st_mtime_ns,
                asset.group,
                asset.id,
            )
        # ``parse_sm`` records the source it read for local diagnostics. Never
        # expose that absolute filesystem path through the reference-chart API.
        public_source = asset.chart_path.relative_to(self.root).as_posix()
        return chart.model_copy(update={"source_path": public_source})

    def charts(self, *, mode: str | None = None) -> list[ChartDocument]:
        charts = [self.chart(asset.id) for asset in self._assets]
        if mode:
            charts = [chart for chart in charts if chart.mode == mode]
        return charts

    def summaries(
        self,
        *,
        mode: str | None = None,
        group: str | None = None,
        search: str = "",
    ) -> list[ReferenceChartSummary]:
        needle = search.strip().casefold()
        output: list[ReferenceChartSummary] = []
        for asset in self._assets:
            if group and asset.group != group:
                continue
            chart = self.chart(asset.id)
            if mode and chart.mode != mode:
                continue
            if needle and needle not in f"{chart.title} {asset.group} {chart.music}".casefold():
                continue
            stats = chart.statistics or chart_statistics(chart)
            output.append(
                ReferenceChartSummary(
                    id=asset.id,
                    title=chart.title,
                    group=asset.group,
                    mode=chart.mode,
                    lane_count=chart.lane_count,
                    difficulty=chart.difficulty,
                    meter=chart.meter,
                    bpm=chart.bpm,
                    bpm_max=max(point.bpm for point in chart.tempo_map),
                    offset_sec=chart.offset_sec,
                    duration_sec=chart.duration_sec,
                    note_count=stats.note_count,
                    event_count=stats.event_count,
                    nps_average=stats.nps_average,
                    nps_peak=stats.nps_peak,
                    audio_url=f"/api/chart-engine/reference-charts/{asset.id}/audio",
                    chart_url=f"/api/chart-engine/reference-charts/{asset.id}",
                )
            )
        return output

    def statistics(self):
        return corpus_statistics(self.charts())
