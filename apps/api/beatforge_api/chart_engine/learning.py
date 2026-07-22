from __future__ import annotations

import bisect
import copy
import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .model import (
    MODEL_ARCHITECTURE,
    ChartTransformer,
    ChartTransformerConfig,
    require_torch,
)
from .models import ChartDocument

FEATURE_SCHEMA_VERSION = "beatforge.chart-candidate-features.v1"
CHECKPOINT_SCHEMA_VERSION = "beatforge.chart-transformer.checkpoint.v1"
TRAINING_SAMPLE_SCHEMA_VERSION = "beatforge.chart-training-sample.v1"
TRAINING_FEATURE_SCHEMA_VERSION = "beatforge.chart-training-features.v1"

FEATURE_NAMES = (
    "timePosition",
    "previousGap2Sec",
    "nextGap2Sec",
    "candidateConfidence",
    "gridConfidence",
    "snapError100Ms",
    "statusAccepted",
    "statusUncertain",
    "statusRejected",
    "laneVocals",
    "laneMelody",
    "laneDrums",
    "laneMix",
    "sourceVocals",
    "sourceMelody",
    "sourceDrums",
    "sourceMix",
    "semanticLyricAlignment",
    "semanticPhonemeConfidence",
    "semanticPitchConfidence",
    "semanticBeatConfidence",
    "generatorAnalysis",
    "generatorHubertCtc",
    "localDensityHalfSecond",
    "tempoBpm300",
    "tempoConfidence",
)

SplitName = Literal["train", "validation", "test"]


class DatasetContractError(ValueError):
    """Raised when a training directory is not a complete real dataset sample."""


class CheckpointContractError(ValueError):
    """Raised when a local checkpoint does not match the supported model contract."""


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetContractError(f"cannot read dataset JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DatasetContractError(f"dataset JSON must contain an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _value(item: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in item:
            return item[name]
    return default


def _number(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def _unit(value: Any) -> float:
    return min(1.0, max(0.0, _number(value)))


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _safe_sample_asset(sample_dir: Path, relative_name: Any, label: str) -> Path:
    if not isinstance(relative_name, str) or not relative_name.strip():
        raise DatasetContractError(f"{sample_dir.name} has no {label} file name")
    candidate = (sample_dir / relative_name).resolve()
    if not candidate.is_relative_to(sample_dir.resolve()):
        raise DatasetContractError(f"{sample_dir.name} {label} path escapes the sample directory")
    if not candidate.is_file() or candidate.stat().st_size <= 0:
        raise DatasetContractError(f"{sample_dir.name} {label} file is missing or empty")
    return candidate


@dataclass(frozen=True, slots=True)
class RealDatasetSample:
    sample_id: str
    split: SplitName
    source_group: str
    source_difficulty: int
    training_difficulty: int
    difficulty_was_clamped: bool
    audio_sha256: str
    metadata: dict[str, Any]
    beatforge: dict[str, Any]
    chart: ChartDocument
    sample_dir: Path


def load_completed_dataset_samples(
    dataset_dir: str | Path,
    *,
    split: SplitName | None = None,
    verify_audio_hashes: bool = True,
) -> list[RealDatasetSample]:
    """Load complete real samples emitted by :mod:`chart_engine.dataset` only.

    No arrays, targets, or placeholder candidates are synthesized when files are
    absent. Any directory declaring itself as a sample must satisfy the complete
    audio/analysis/chart contract before it can enter training.
    """

    root = Path(dataset_dir).expanduser().resolve()
    if not root.is_dir():
        raise DatasetContractError(f"chart dataset directory does not exist: {root}")
    samples: list[RealDatasetSample] = []
    for sample_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        metadata_path = sample_dir / "metadata.json"
        if not metadata_path.is_file():
            continue
        metadata = _json_object(metadata_path)
        if metadata.get("schemaVersion") != TRAINING_SAMPLE_SCHEMA_VERSION:
            raise DatasetContractError(
                f"{sample_dir.name} uses an unsupported training sample schema"
            )
        if metadata.get("realData") is not True:
            raise DatasetContractError(f"{sample_dir.name} is not marked as real data")
        sample_split = metadata.get("split")
        if sample_split not in {"train", "validation", "test"}:
            raise DatasetContractError(f"{sample_dir.name} has an invalid dataset split")
        if split is not None and sample_split != split:
            continue
        if metadata.get("mode") != "pump-single":
            raise DatasetContractError(
                f"{sample_dir.name} is not a five-lane pump-single training sample"
            )
        sample_id = str(metadata.get("songId") or "")
        if not sample_id or sample_id != sample_dir.name:
            raise DatasetContractError(f"{sample_dir.name} has an inconsistent songId")

        audio_path = _safe_sample_asset(sample_dir, metadata.get("audioFile"), "audio")
        beatforge_path = _safe_sample_asset(
            sample_dir, metadata.get("beatforgeFile"), "BeatForge analysis"
        )
        chart_path = _safe_sample_asset(sample_dir, metadata.get("chartFile"), "chart")
        expected_audio_hash = str(metadata.get("audioSha256") or "").lower()
        if len(expected_audio_hash) != 64:
            raise DatasetContractError(f"{sample_dir.name} has no valid audio SHA256")
        if verify_audio_hashes and _sha256(audio_path) != expected_audio_hash:
            raise DatasetContractError(f"{sample_dir.name} audio SHA256 does not match metadata")

        beatforge = _json_object(beatforge_path)
        if beatforge.get("schemaVersion") != TRAINING_FEATURE_SCHEMA_VERSION:
            raise DatasetContractError(
                f"{sample_dir.name} uses an unsupported BeatForge feature schema"
            )
        if str(beatforge.get("audioSha256") or "").lower() != expected_audio_hash:
            raise DatasetContractError(
                f"{sample_dir.name} BeatForge features refer to a different audio file"
            )
        analysis = beatforge.get("analysis")
        if not isinstance(analysis, dict):
            raise DatasetContractError(f"{sample_dir.name} has no completed analysis payload")
        candidates = _value(analysis, "candidate_events", "candidateEvents", default=[])
        if not isinstance(candidates, list) or not candidates:
            raise DatasetContractError(f"{sample_dir.name} has no real candidate events")

        try:
            chart = ChartDocument.model_validate(_json_object(chart_path))
        except Exception as exc:
            raise DatasetContractError(f"{sample_dir.name} chart is invalid: {exc}") from exc
        if chart.id != sample_id or chart.mode != "pump-single" or chart.lane_count != 5:
            raise DatasetContractError(
                f"{sample_dir.name} chart does not match its five-lane sample metadata"
            )
        source_difficulty = int(metadata.get("difficulty") or chart.meter)
        training_difficulty = min(15, max(1, source_difficulty))
        samples.append(
            RealDatasetSample(
                sample_id=sample_id,
                split=sample_split,
                source_group=str(metadata.get("sourceGroup") or "UNKNOWN"),
                source_difficulty=source_difficulty,
                training_difficulty=training_difficulty,
                difficulty_was_clamped=training_difficulty != source_difficulty,
                audio_sha256=expected_audio_hash,
                metadata=metadata,
                beatforge=beatforge,
                chart=chart,
                sample_dir=sample_dir,
            )
        )
    if not samples:
        suffix = f" for split {split}" if split else ""
        raise DatasetContractError(f"no complete real chart samples found{suffix} in {root}")
    return samples


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    candidate_id: str
    time_sec: float
    acoustic_sample: int
    chart_sample: int
    features: tuple[float, ...]


def candidate_records(beatforge_payload: dict[str, Any]) -> list[CandidateRecord]:
    """Convert a real BeatForge candidate bundle to the stable model feature schema."""

    analysis_value = beatforge_payload.get("analysis", beatforge_payload)
    if not isinstance(analysis_value, dict):
        raise DatasetContractError("BeatForge inference payload has no analysis object")
    candidates_value = _value(analysis_value, "candidate_events", "candidateEvents", default=[])
    if not isinstance(candidates_value, list) or not candidates_value:
        raise DatasetContractError("BeatForge inference requires real candidate events")
    sample_rate = int(
        _value(
            analysis_value,
            "original_sample_rate",
            "originalSampleRate",
            "sample_rate",
            "sampleRate",
            default=0,
        )
    )
    sample_count = int(_value(analysis_value, "sample_count", "sampleCount", default=0))
    if sample_rate <= 0 or sample_count <= 0:
        raise DatasetContractError("BeatForge analysis has an invalid sample timeline")
    duration_sec = _number(
        _value(analysis_value, "duration_sec", "durationSec"),
        sample_count / sample_rate,
    )
    if duration_sec <= 0:
        duration_sec = sample_count / sample_rate
    bpm = _number(_value(analysis_value, "bpm", "estimatedBpm"), 120.0)
    bpm_confidence = _unit(_value(analysis_value, "bpm_confidence", "bpmConfidence", default=0.0))

    raw_records: list[tuple[dict[str, Any], str, float, int, int]] = []
    for item in candidates_value:
        if not isinstance(item, dict):
            raise DatasetContractError("candidate events must be JSON objects")
        candidate_id = str(item.get("id") or "")
        if not candidate_id:
            raise DatasetContractError("candidate event is missing its provenance ID")
        acoustic_sample = int(
            _value(item, "acoustic_sample", "acousticSample", "sample", default=-1)
        )
        chart_sample = int(_value(item, "chart_sample", "chartSample", default=acoustic_sample))
        time_sec = _number(_value(item, "time_sec", "timeSec"), acoustic_sample / sample_rate)
        if acoustic_sample < 0 or acoustic_sample >= sample_count or time_sec < 0:
            raise DatasetContractError(
                f"candidate {candidate_id} lies outside the real audio timeline"
            )
        raw_records.append((item, candidate_id, time_sec, acoustic_sample, chart_sample))
    raw_records.sort(key=lambda value: (value[2], value[3], value[1]))
    times = [item[2] for item in raw_records]
    records: list[CandidateRecord] = []
    for index, (item, candidate_id, time_sec, acoustic_sample, chart_sample) in enumerate(
        raw_records
    ):
        previous_gap = time_sec - times[index - 1] if index else 2.0
        next_gap = times[index + 1] - time_sec if index + 1 < len(times) else 2.0
        status = str(item.get("status") or "uncertain")
        lane = str(item.get("lane") or "mix")
        source = _mapping(_value(item, "source_evidence", "sourceEvidence", default={}))
        semantic = _mapping(_value(item, "semantic_evidence", "semanticEvidence", default={}))
        generator = str(item.get("generator") or "analysis")
        local_count = bisect.bisect_right(times, time_sec + 0.5) - bisect.bisect_left(
            times, time_sec - 0.5
        )
        feature_values = (
            min(1.0, max(0.0, time_sec / duration_sec)),
            min(1.0, max(0.0, previous_gap / 2.0)),
            min(1.0, max(0.0, next_gap / 2.0)),
            _unit(item.get("confidence")),
            _unit(_value(item, "grid_confidence", "gridConfidence")),
            min(
                1.0,
                max(
                    -1.0,
                    _number(_value(item, "snap_error_ms", "snapErrorMs")) / 100.0,
                ),
            ),
            float(status == "accepted"),
            float(status == "uncertain"),
            float(status == "rejected"),
            float(lane == "vocals"),
            float(lane == "melody"),
            float(lane == "drums"),
            float(lane == "mix"),
            _unit(source.get("vocals")),
            _unit(source.get("melody")),
            _unit(source.get("drums")),
            _unit(source.get("mix")),
            _unit(_value(semantic, "lyricAlignment", "lyric_alignment")),
            _unit(_value(semantic, "phonemeConfidence", "phoneme_confidence")),
            _unit(_value(semantic, "pitchConfidence", "pitch_confidence")),
            _unit(_value(semantic, "beatConfidence", "beat_confidence")),
            float(generator == "analysis"),
            float(generator == "hubert_ctc"),
            min(1.0, local_count / 16.0),
            min(1.0, max(0.0, bpm / 300.0)),
            bpm_confidence,
        )
        if len(feature_values) != len(FEATURE_NAMES):
            raise AssertionError("candidate feature schema is inconsistent")
        records.append(
            CandidateRecord(
                candidate_id=candidate_id,
                time_sec=time_sec,
                acoustic_sample=acoustic_sample,
                chart_sample=chart_sample,
                features=feature_values,
            )
        )
    return records


@dataclass(frozen=True, slots=True)
class SequenceExample:
    sample_id: str
    split: SplitName
    source_difficulty: int
    difficulty: int
    audio_sha256: str
    records: tuple[CandidateRecord, ...]
    lane_targets: tuple[tuple[float, float, float, float, float], ...]
    hold_targets: tuple[float, ...]
    matched_event_count: int


def sequence_example(
    sample: RealDatasetSample, *, match_tolerance_ms: float = 80.0
) -> SequenceExample:
    if match_tolerance_ms <= 0:
        raise ValueError("match_tolerance_ms must be positive")
    records = candidate_records(sample.beatforge)
    playable_events = [
        event
        for event in sample.chart.events
        if any(note.type != "mine" and note.lane < 5 for note in event.notes)
    ]
    event_times = [event.time_sec for event in playable_events]
    tolerance = match_tolerance_ms / 1000.0
    pairs: list[tuple[float, int, int]] = []
    for candidate_index, record in enumerate(records):
        left = bisect.bisect_left(event_times, record.time_sec - tolerance)
        right = bisect.bisect_right(event_times, record.time_sec + tolerance)
        pairs.extend(
            (abs(record.time_sec - event_times[event_index]), candidate_index, event_index)
            for event_index in range(left, right)
        )
    pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    matched_candidates: set[int] = set()
    matched_events: set[int] = set()
    assignments: dict[int, int] = {}
    for _distance, candidate_index, event_index in pairs:
        if candidate_index in matched_candidates or event_index in matched_events:
            continue
        matched_candidates.add(candidate_index)
        matched_events.add(event_index)
        assignments[candidate_index] = event_index

    lane_targets: list[tuple[float, float, float, float, float]] = []
    hold_targets: list[float] = []
    for candidate_index in range(len(records)):
        lanes = [0.0] * 5
        hold = 0.0
        event_index = assignments.get(candidate_index)
        if event_index is not None:
            for note in playable_events[event_index].notes:
                if note.type == "mine" or note.lane >= 5:
                    continue
                lanes[note.lane] = 1.0
                if note.type == "hold":
                    hold = 1.0
        lane_targets.append((lanes[0], lanes[1], lanes[2], lanes[3], lanes[4]))
        hold_targets.append(hold)
    if not assignments:
        raise DatasetContractError(
            f"real sample {sample.sample_id} has no chart events within "
            f"{match_tolerance_ms:g} ms of its BeatForge candidates"
        )
    return SequenceExample(
        sample_id=sample.sample_id,
        split=sample.split,
        source_difficulty=sample.source_difficulty,
        difficulty=sample.training_difficulty,
        audio_sha256=sample.audio_sha256,
        records=tuple(records),
        lane_targets=tuple(lane_targets),
        hold_targets=tuple(hold_targets),
        matched_event_count=len(assignments),
    )


@dataclass(frozen=True, slots=True)
class SequenceChunk:
    sample_id: str
    difficulty: int
    features: tuple[tuple[float, ...], ...]
    lane_targets: tuple[tuple[float, float, float, float, float], ...]
    hold_targets: tuple[float, ...]


def sequence_chunks(
    examples: list[SequenceExample], *, length: int, stride: int | None = None
) -> list[SequenceChunk]:
    if length <= 0:
        raise ValueError("sequence length must be positive")
    active_stride = stride or length
    if active_stride <= 0 or active_stride > length:
        raise ValueError("sequence stride must be in the range 1..length")
    chunks: list[SequenceChunk] = []
    for example in examples:
        for start in range(0, len(example.records), active_stride):
            end = min(start + length, len(example.records))
            if end <= start:
                continue
            chunks.append(
                SequenceChunk(
                    sample_id=example.sample_id,
                    difficulty=example.difficulty,
                    features=tuple(record.features for record in example.records[start:end]),
                    lane_targets=example.lane_targets[start:end],
                    hold_targets=example.hold_targets[start:end],
                )
            )
            if end == len(example.records):
                break
    if not chunks:
        raise DatasetContractError("real training samples produced no candidate sequences")
    return chunks


@dataclass(frozen=True, slots=True)
class FeatureNormalization:
    means: tuple[float, ...]
    scales: tuple[float, ...]

    @classmethod
    def fit(cls, examples: list[SequenceExample]) -> FeatureNormalization:
        rows = [record.features for example in examples for record in example.records]
        if not rows:
            raise DatasetContractError("cannot normalize an empty real candidate set")
        count = len(rows)
        means = tuple(
            sum(row[index] for row in rows) / count for index in range(len(FEATURE_NAMES))
        )
        variances = tuple(
            sum((row[index] - means[index]) ** 2 for row in rows) / count
            for index in range(len(FEATURE_NAMES))
        )
        scales = tuple(max(math.sqrt(value), 1e-6) for value in variances)
        return cls(means=means, scales=scales)

    def normalize(self, row: tuple[float, ...]) -> tuple[float, ...]:
        if len(row) != len(self.means):
            raise ValueError("candidate feature count does not match checkpoint normalization")
        return tuple(
            (value - self.means[index]) / self.scales[index] for index, value in enumerate(row)
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {"means": list(self.means), "scales": list(self.scales)}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> FeatureNormalization:
        means = tuple(float(item) for item in value.get("means", []))
        scales = tuple(float(item) for item in value.get("scales", []))
        if len(means) != len(FEATURE_NAMES) or len(scales) != len(FEATURE_NAMES):
            raise CheckpointContractError("checkpoint feature normalization is incomplete")
        if any(scale <= 0 or not math.isfinite(scale) for scale in scales):
            raise CheckpointContractError("checkpoint feature scales are invalid")
        return cls(means=means, scales=scales)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    epochs: int = 12
    batch_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    match_tolerance_ms: float = 80.0
    sequence_length: int = 512
    sequence_stride: int | None = None
    seed: int = 20260721
    device: str = "auto"
    validation_split: SplitName | None = "validation"
    verify_audio_hashes: bool = True
    max_batches_per_epoch: int | None = None

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer parameters are invalid")
        if self.gradient_clip <= 0 or self.match_tolerance_ms <= 0:
            raise ValueError("gradient_clip and match_tolerance_ms must be positive")
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if (
            self.sequence_stride is not None
            and not 0 < self.sequence_stride <= self.sequence_length
        ):
            raise ValueError("sequence_stride must be in the range 1..sequence_length")
        if self.max_batches_per_epoch is not None and self.max_batches_per_epoch <= 0:
            raise ValueError("max_batches_per_epoch must be positive when provided")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _device(torch_module: Any, requested: str) -> Any:
    if requested != "auto":
        return torch_module.device(requested)
    if hasattr(torch_module.backends, "mps") and torch_module.backends.mps.is_available():
        return torch_module.device("mps")
    if torch_module.cuda.is_available():
        return torch_module.device("cuda")
    return torch_module.device("cpu")


def _normalized_chunks(
    chunks: list[SequenceChunk], normalization: FeatureNormalization
) -> list[SequenceChunk]:
    return [
        SequenceChunk(
            sample_id=chunk.sample_id,
            difficulty=chunk.difficulty,
            features=tuple(normalization.normalize(row) for row in chunk.features),
            lane_targets=chunk.lane_targets,
            hold_targets=chunk.hold_targets,
        )
        for chunk in chunks
    ]


class _ChunkDataset:
    def __init__(self, chunks: list[SequenceChunk]) -> None:
        self.chunks = chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> SequenceChunk:
        return self.chunks[index]


def _collate(torch_module: Any, chunks: list[SequenceChunk]) -> dict[str, Any]:
    batch_size = len(chunks)
    maximum = max(len(chunk.features) for chunk in chunks)
    features = torch_module.zeros(
        (batch_size, maximum, len(FEATURE_NAMES)), dtype=torch_module.float32
    )
    lane_targets = torch_module.zeros((batch_size, maximum, 5), dtype=torch_module.float32)
    hold_targets = torch_module.zeros((batch_size, maximum), dtype=torch_module.float32)
    padding_mask = torch_module.ones((batch_size, maximum), dtype=torch_module.bool)
    difficulties = torch_module.empty((batch_size,), dtype=torch_module.long)
    for index, chunk in enumerate(chunks):
        length = len(chunk.features)
        features[index, :length] = torch_module.tensor(chunk.features, dtype=torch_module.float32)
        lane_targets[index, :length] = torch_module.tensor(
            chunk.lane_targets, dtype=torch_module.float32
        )
        hold_targets[index, :length] = torch_module.tensor(
            chunk.hold_targets, dtype=torch_module.float32
        )
        padding_mask[index, :length] = False
        difficulties[index] = chunk.difficulty
    return {
        "features": features,
        "lane_targets": lane_targets,
        "hold_targets": hold_targets,
        "padding_mask": padding_mask,
        "difficulties": difficulties,
    }


def _positive_weights(torch_module: Any, chunks: list[SequenceChunk]) -> tuple[Any, Any]:
    total = sum(len(chunk.lane_targets) for chunk in chunks)
    lane_positive = [
        sum(row[lane] for chunk in chunks for row in chunk.lane_targets) for lane in range(5)
    ]
    hold_positive = sum(value for chunk in chunks for value in chunk.hold_targets)
    if not any(lane_positive):
        raise DatasetContractError("real training candidates contain no matched lane targets")
    lane_weights = [
        min(50.0, max(1.0, (total - positive) / max(positive, 1.0))) for positive in lane_positive
    ]
    hold_weight = min(50.0, max(1.0, (total - hold_positive) / max(hold_positive, 1.0)))
    return (
        torch_module.tensor(lane_weights, dtype=torch_module.float32),
        torch_module.tensor(hold_weight, dtype=torch_module.float32),
    )


def _batch_loss(
    torch_module: Any,
    model: Any,
    batch: dict[str, Any],
    device: Any,
    lane_positive_weights: Any,
    hold_positive_weight: Any,
) -> Any:
    features = batch["features"].to(device)
    lanes = batch["lane_targets"].to(device)
    holds = batch["hold_targets"].to(device)
    mask = (~batch["padding_mask"]).to(device)
    difficulties = batch["difficulties"].to(device)
    output = model(features, difficulties, padding_mask=~mask)
    lane_loss = torch_module.nn.functional.binary_cross_entropy_with_logits(
        output["lane_logits"],
        lanes,
        pos_weight=lane_positive_weights.to(device),
        reduction="none",
    )
    hold_loss = torch_module.nn.functional.binary_cross_entropy_with_logits(
        output["hold_logits"],
        holds,
        pos_weight=hold_positive_weight.to(device),
        reduction="none",
    )
    lane_mask = mask.unsqueeze(-1).expand_as(lane_loss)
    lane_value = lane_loss[lane_mask].mean()
    hold_value = hold_loss[mask].mean()
    return lane_value + 0.45 * hold_value


def _evaluate(
    torch_module: Any,
    model: Any,
    loader: Any | None,
    device: Any,
    lane_positive_weights: Any,
    hold_positive_weight: Any,
) -> float | None:
    if loader is None:
        return None
    model.eval()
    losses: list[float] = []
    with torch_module.inference_mode():
        for batch in loader:
            loss = _batch_loss(
                torch_module,
                model,
                batch,
                device,
                lane_positive_weights,
                hold_positive_weight,
            )
            losses.append(float(loss.detach().cpu().item()))
    return sum(losses) / len(losses) if losses else None


def _dataset_fingerprint(samples: list[RealDatasetSample]) -> str:
    identity = [
        {
            "sampleId": sample.sample_id,
            "split": sample.split,
            "audioSha256": sample.audio_sha256,
            "chartSha256": sample.metadata.get("chartSha256"),
        }
        for sample in sorted(samples, key=lambda item: (item.sample_id, item.split))
    ]
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _save_checkpoint(path: Path, payload: dict[str, Any], torch_module: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        torch_module.save(payload, temporary_name)
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


@dataclass(frozen=True, slots=True)
class TrainingResult:
    checkpoint_path: Path
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"checkpointPath": str(self.checkpoint_path), "metadata": self.metadata}


def train_chart_transformer(
    dataset_dir: str | Path,
    checkpoint_path: str | Path,
    *,
    training: TrainingConfig | None = None,
    model_config: ChartTransformerConfig | None = None,
) -> TrainingResult:
    """Train and atomically save a local Transformer from complete real triples."""

    torch_module = require_torch()
    active = training or TrainingConfig()
    random.seed(active.seed)
    torch_module.manual_seed(active.seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(active.seed)

    train_samples = load_completed_dataset_samples(
        dataset_dir,
        split="train",
        verify_audio_hashes=active.verify_audio_hashes,
    )
    validation_samples: list[RealDatasetSample] = []
    if active.validation_split is not None:
        try:
            validation_samples = load_completed_dataset_samples(
                dataset_dir,
                split=active.validation_split,
                verify_audio_hashes=active.verify_audio_hashes,
            )
        except DatasetContractError as exc:
            if "no complete real chart samples found" not in str(exc):
                raise
    train_examples = [
        sequence_example(sample, match_tolerance_ms=active.match_tolerance_ms)
        for sample in train_samples
    ]
    validation_examples = [
        sequence_example(sample, match_tolerance_ms=active.match_tolerance_ms)
        for sample in validation_samples
    ]
    normalization = FeatureNormalization.fit(train_examples)
    train_chunks = _normalized_chunks(
        sequence_chunks(
            train_examples,
            length=active.sequence_length,
            stride=active.sequence_stride,
        ),
        normalization,
    )
    validation_chunks = (
        _normalized_chunks(
            sequence_chunks(
                validation_examples,
                length=active.sequence_length,
                stride=active.sequence_stride,
            ),
            normalization,
        )
        if validation_examples
        else []
    )
    configured_model = model_config or ChartTransformerConfig(
        input_dim=len(FEATURE_NAMES), max_sequence_length=active.sequence_length
    )
    if configured_model.input_dim != len(FEATURE_NAMES):
        raise ValueError("model input_dim does not match the candidate feature schema")
    if configured_model.max_sequence_length < active.sequence_length:
        raise ValueError("model max_sequence_length is smaller than the training sequence length")

    device = _device(torch_module, active.device)
    model = ChartTransformer(configured_model).to(device)
    optimizer = torch_module.optim.AdamW(
        model.parameters(),
        lr=active.learning_rate,
        weight_decay=active.weight_decay,
    )
    generator = torch_module.Generator()
    generator.manual_seed(active.seed)
    collate = lambda chunks: _collate(torch_module, chunks)  # noqa: E731
    train_loader = torch_module.utils.data.DataLoader(
        _ChunkDataset(train_chunks),
        batch_size=active.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        collate_fn=collate,
    )
    validation_loader = (
        torch_module.utils.data.DataLoader(
            _ChunkDataset(validation_chunks),
            batch_size=active.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
        )
        if validation_chunks
        else None
    )
    lane_positive_weights, hold_positive_weight = _positive_weights(torch_module, train_chunks)
    history: list[dict[str, float | int | None]] = []
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    for epoch in range(1, active.epochs + 1):
        model.train()
        losses: list[float] = []
        for batch_index, batch in enumerate(train_loader):
            optimizer.zero_grad(set_to_none=True)
            loss = _batch_loss(
                torch_module,
                model,
                batch,
                device,
                lane_positive_weights,
                hold_positive_weight,
            )
            loss.backward()
            torch_module.nn.utils.clip_grad_norm_(model.parameters(), active.gradient_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
            if (
                active.max_batches_per_epoch is not None
                and batch_index + 1 >= active.max_batches_per_epoch
            ):
                break
        train_loss = sum(losses) / len(losses)
        validation_loss = _evaluate(
            torch_module,
            model,
            validation_loader,
            device,
            lane_positive_weights,
            hold_positive_weight,
        )
        selection_loss = validation_loss if validation_loss is not None else train_loss
        history.append(
            {
                "epoch": epoch,
                "trainLoss": train_loss,
                "validationLoss": validation_loss,
            }
        )
        if selection_loss < best_loss:
            best_loss = selection_loss
            best_state = copy.deepcopy(model.state_dict())
    if best_state is None:
        raise RuntimeError("training completed without producing model weights")
    model.load_state_dict(best_state)
    all_samples = train_samples + validation_samples
    all_examples = train_examples + validation_examples
    clamped = [sample.sample_id for sample in all_samples if sample.difficulty_was_clamped]
    metadata: dict[str, Any] = {
        "schemaVersion": CHECKPOINT_SCHEMA_VERSION,
        "createdAt": datetime.now(UTC).isoformat(),
        "architecture": MODEL_ARCHITECTURE,
        "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
        "featureNames": list(FEATURE_NAMES),
        "trainingSampleSchemaVersion": TRAINING_SAMPLE_SCHEMA_VERSION,
        "trainingFeatureSchemaVersion": TRAINING_FEATURE_SCHEMA_VERSION,
        "realDataOnly": True,
        "datasetFingerprint": _dataset_fingerprint(all_samples),
        "sampleCount": len(all_samples),
        "trainSampleCount": len(train_samples),
        "validationSampleCount": len(validation_samples),
        "sequenceCount": len(train_chunks) + len(validation_chunks),
        "candidateCount": sum(len(example.records) for example in all_examples),
        "matchedEventCount": sum(example.matched_event_count for example in all_examples),
        "sampleIds": [sample.sample_id for sample in all_samples],
        "audioSha256": [sample.audio_sha256 for sample in all_samples],
        "sourceDifficulties": {
            sample.sample_id: sample.source_difficulty for sample in all_samples
        },
        "difficultyClampedSampleIds": clamped,
        "modelConfig": configured_model.to_dict(),
        "trainingConfig": active.to_dict(),
        "normalization": normalization.to_dict(),
        "positiveWeights": {
            "lanes": lane_positive_weights.tolist(),
            "hold": float(hold_positive_weight.item()),
        },
        "history": history,
        "bestLoss": best_loss,
        "torchVersion": str(torch_module.__version__),
        "trainingDevice": str(device),
    }
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "metadata": metadata,
        "model_config": configured_model.to_dict(),
        "feature_names": list(FEATURE_NAMES),
        "normalization": normalization.to_dict(),
        "model_state_dict": model.state_dict(),
    }
    destination = Path(checkpoint_path).expanduser().resolve()
    _save_checkpoint(destination, checkpoint, torch_module)
    return TrainingResult(checkpoint_path=destination, metadata=metadata)


@dataclass(frozen=True, slots=True)
class CandidatePrediction:
    candidate_id: str
    time_sec: float
    acoustic_sample: int
    chart_sample: int
    lane_probabilities: tuple[float, float, float, float, float]
    hold_probability: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidateId": self.candidate_id,
            "timeSec": self.time_sec,
            "acousticSample": self.acoustic_sample,
            "chartSample": self.chart_sample,
            "laneProbabilities": list(self.lane_probabilities),
            "holdProbability": self.hold_probability,
        }


@dataclass(frozen=True, slots=True)
class InferenceResult:
    difficulty: int
    checkpoint_metadata: dict[str, Any]
    predictions: tuple[CandidatePrediction, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "difficulty": self.difficulty,
            "checkpoint": self.checkpoint_metadata,
            "predictions": [prediction.to_dict() for prediction in self.predictions],
        }


class LocalChartModel:
    """Loaded local checkpoint with deterministic candidate-level inference APIs."""

    def __init__(
        self,
        model: Any,
        normalization: FeatureNormalization,
        metadata: dict[str, Any],
        device: Any,
    ) -> None:
        self.model = model
        self.normalization = normalization
        self.metadata = metadata
        self.device = device

    @classmethod
    def load(cls, checkpoint_path: str | Path, *, device: str = "auto") -> LocalChartModel:
        torch_module = require_torch()
        active_device = _device(torch_module, device)
        source = Path(checkpoint_path).expanduser().resolve()
        if not source.is_file():
            raise CheckpointContractError(f"chart checkpoint does not exist: {source}")
        try:
            payload = torch_module.load(source, map_location=active_device, weights_only=True)
        except Exception as exc:
            raise CheckpointContractError(f"cannot load chart checkpoint: {exc}") from exc
        if not isinstance(payload, dict):
            raise CheckpointContractError("chart checkpoint payload must be a dictionary")
        if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointContractError("unsupported chart checkpoint schema")
        if payload.get("feature_names") != list(FEATURE_NAMES):
            raise CheckpointContractError("checkpoint feature schema does not match this runtime")
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("realDataOnly") is not True:
            raise CheckpointContractError("checkpoint lacks real-data training provenance")
        model_config_value = payload.get("model_config")
        normalization_value = payload.get("normalization")
        state = payload.get("model_state_dict")
        if not isinstance(model_config_value, dict) or not isinstance(normalization_value, dict):
            raise CheckpointContractError("checkpoint configuration is incomplete")
        if not isinstance(state, dict):
            raise CheckpointContractError("checkpoint contains no model state")
        config = ChartTransformerConfig.from_dict(model_config_value)
        model = ChartTransformer(config).to(active_device)
        try:
            model.load_state_dict(state, strict=True)
        except Exception as exc:
            raise CheckpointContractError(f"checkpoint model state is incompatible: {exc}") from exc
        model.eval()
        return cls(
            model=model,
            normalization=FeatureNormalization.from_dict(normalization_value),
            metadata=metadata,
            device=active_device,
        )

    def predict(self, beatforge_payload: dict[str, Any], *, difficulty: int) -> InferenceResult:
        if difficulty < 1 or difficulty > 15:
            raise ValueError("difficulty must be between 1 and 15")
        torch_module = require_torch()
        records = candidate_records(beatforge_payload)
        maximum = self.model.config.max_sequence_length
        predictions: list[CandidatePrediction] = []
        with torch_module.inference_mode():
            for start in range(0, len(records), maximum):
                chunk = records[start : start + maximum]
                normalized = [self.normalization.normalize(record.features) for record in chunk]
                features = torch_module.tensor(
                    normalized, dtype=torch_module.float32, device=self.device
                ).unsqueeze(0)
                difficulties = torch_module.tensor(
                    [difficulty], dtype=torch_module.long, device=self.device
                )
                output = self.model(features, difficulties)
                lane_values = torch_module.sigmoid(output["lane_logits"])[0].cpu().tolist()
                hold_values = torch_module.sigmoid(output["hold_logits"])[0].cpu().tolist()
                for record, lane_row, hold_value in zip(
                    chunk, lane_values, hold_values, strict=True
                ):
                    predictions.append(
                        CandidatePrediction(
                            candidate_id=record.candidate_id,
                            time_sec=record.time_sec,
                            acoustic_sample=record.acoustic_sample,
                            chart_sample=record.chart_sample,
                            lane_probabilities=(
                                float(lane_row[0]),
                                float(lane_row[1]),
                                float(lane_row[2]),
                                float(lane_row[3]),
                                float(lane_row[4]),
                            ),
                            hold_probability=float(hold_value),
                        )
                    )
        return InferenceResult(
            difficulty=difficulty,
            checkpoint_metadata=self.metadata,
            predictions=tuple(predictions),
        )


def predict_candidate_probabilities(
    beatforge_payload: dict[str, Any],
    checkpoint_path: str | Path,
    *,
    difficulty: int,
    device: str = "auto",
) -> InferenceResult:
    """Convenience API for one-shot, entirely local chart-model inference."""

    return LocalChartModel.load(checkpoint_path, device=device).predict(
        beatforge_payload, difficulty=difficulty
    )
