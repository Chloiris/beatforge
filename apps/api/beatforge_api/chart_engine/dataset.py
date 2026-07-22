from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from ..audio.io import AudioDecodeError
from ..audio.pipeline import analyze_audio
from ..media import prepare_analysis_source
from .library import ReferenceAsset, ReferenceLibrary
from .statistics import corpus_statistics

AnalysisMode = Literal["recall", "balanced", "clean", "accurate"]


@dataclass(slots=True)
class DatasetBuildReport:
    source_chart_count: int
    completed: int = 0
    skipped: int = 0
    failed: int = 0
    analyzed_audio_count: int = 0
    reused_analysis_count: int = 0
    samples: list[str] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sourceChartCount": self.source_chart_count,
            "completed": self.completed,
            "skipped": self.skipped,
            "failed": self.failed,
            "analyzedAudioCount": self.analyzed_audio_count,
            "reusedAnalysisCount": self.reused_analysis_count,
            "samples": self.samples,
            "errors": self.errors,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _materialize_audio(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        try:
            target.symlink_to(source)
        except OSError:
            shutil.copy2(source, target)


def _split_for_hash(audio_hash: str) -> str:
    bucket = int(audio_hash[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _analysis_payload(asset: ReferenceAsset, mode: AnalysisMode) -> dict[str, Any]:
    decode_backend = "libsndfile"
    try:
        result = analyze_audio(asset.audio_path, mode=mode, sensitivity=0.5)
    except AudioDecodeError:
        # Some corpus MP3s expose valid metadata but fail during a complete
        # libsndfile read. Decode those exact source bytes through the same local
        # ffmpeg recovery path used by uploaded BeatForge tracks.
        with tempfile.TemporaryDirectory(prefix="beatforge-chart-decode-") as directory:
            decoded, _ = prepare_analysis_source(
                asset.audio_path, Path(directory), force_ffmpeg=True
            )
            result = analyze_audio(decoded, mode=mode, sensitivity=0.5)
        decode_backend = "ffmpeg_pcm_f32le"
    payload = result.to_dict()
    payload["schema_version"] = "beatforge.analysis.v1"
    payload["source_audio"] = asset.audio_path.name
    payload["source_decode_backend"] = decode_backend
    return payload


def build_dataset(
    library: ReferenceLibrary,
    output_dir: str | Path,
    *,
    mode: Literal["pump-single", "pump-double"] = "pump-single",
    analyze_missing: bool = False,
    analysis_mode: AnalysisMode = "balanced",
    limit: int | None = None,
    only_ids: set[str] | None = None,
) -> DatasetBuildReport:
    """Build only complete real `(audio, BeatForge, chart)` training triples.

    When analysis is absent, no placeholder is written. Callers must explicitly
    opt into running the production local analyzer with ``analyze_missing``.
    """

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache_dir = output / ".feature-cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    assets = [asset for asset in library.assets() if library.chart(asset.id).mode == mode]
    if only_ids is not None:
        assets = [asset for asset in assets if asset.id in only_ids]
    if limit is not None:
        assets = assets[: max(0, limit)]
    report = DatasetBuildReport(source_chart_count=len(assets))
    built_charts = []
    manifest_samples: list[dict[str, Any]] = []
    for asset in assets:
        chart = library.chart(asset.id)
        audio_hash = _sha256(asset.audio_path)
        feature_path = cache_dir / f"{audio_hash}.{analysis_mode}.json"
        try:
            if feature_path.is_file():
                analysis = json.loads(feature_path.read_text(encoding="utf-8"))
                # Caches created before decode provenance was added came only
                # from successful direct libsndfile analyses.
                analysis.setdefault("source_decode_backend", "libsndfile")
                report.reused_analysis_count += 1
            elif analyze_missing:
                analysis = _analysis_payload(asset, analysis_mode)
                _write_json(feature_path, analysis)
                report.analyzed_audio_count += 1
            else:
                report.skipped += 1
                report.errors.append(
                    {
                        "chartId": asset.id,
                        "code": "BEATFORGE_ANALYSIS_MISSING",
                        "message": (
                            "No real BeatForge analysis exists for this SPEED audio. "
                            "Re-run with analyze_missing enabled."
                        ),
                    }
                )
                continue
            sample_dir = output / asset.id
            sample_dir.mkdir(parents=True, exist_ok=True)
            _materialize_audio(asset.audio_path, sample_dir / "audio.mp3")
            _write_json(
                sample_dir / "beatforge.json",
                {
                    "schemaVersion": "beatforge.chart-training-features.v1",
                    "audioSha256": audio_hash,
                    "analysisMode": analysis_mode,
                    "analysis": analysis,
                },
            )
            _write_json(
                sample_dir / "chart.json",
                chart.model_dump(by_alias=True, mode="json"),
            )
            metadata = {
                "schemaVersion": "beatforge.chart-training-sample.v1",
                "songId": asset.id,
                "sourceGroup": asset.group,
                "title": chart.title,
                "mode": chart.mode,
                "difficulty": chart.meter,
                "chartBpm": chart.bpm,
                "chartOffsetSec": chart.offset_sec,
                "durationSec": chart.duration_sec,
                "analysisBpm": analysis.get("bpm"),
                "analysisBpmConfidence": analysis.get("bpm_confidence"),
                "audioSha256": audio_hash,
                "chartSha256": _sha256(asset.chart_path),
                "split": _split_for_hash(audio_hash),
                "audioFile": "audio.mp3",
                "beatforgeFile": "beatforge.json",
                "chartFile": "chart.json",
                "realData": True,
            }
            _write_json(sample_dir / "metadata.json", metadata)
            manifest_samples.append(metadata)
            built_charts.append(chart)
            report.samples.append(asset.id)
            report.completed += 1
        except Exception as exc:  # each real song should leave an auditable failure
            report.failed += 1
            report.errors.append(
                {"chartId": asset.id, "code": type(exc).__name__, "message": str(exc)}
            )
    if built_charts:
        statistics = corpus_statistics(built_charts)
        _write_json(
            output / "chart_statistics.json",
            statistics.model_dump(by_alias=True, mode="json"),
        )
    _write_json(
        output / "manifest.json",
        {
            "schemaVersion": "beatforge.chart-training-dataset.v1",
            "realDataOnly": True,
            "mode": mode,
            "analysisMode": analysis_mode,
            "sampleCount": len(manifest_samples),
            "uniqueAudioCount": len({sample["audioSha256"] for sample in manifest_samples}),
            "splits": {
                split: sum(sample["split"] == split for sample in manifest_samples)
                for split in ("train", "validation", "test")
            },
            "samples": manifest_samples,
        },
    )
    _write_json(output / "build_report.json", report.to_dict())
    return report
