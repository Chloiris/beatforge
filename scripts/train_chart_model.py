#!/usr/bin/env python3
"""Train the local BeatForge five-lane Transformer from completed real dataset triples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))

from beatforge_api.chart_engine.learning import (  # noqa: E402
    FEATURE_NAMES,
    TrainingConfig,
    train_chart_transformer,
)
from beatforge_api.chart_engine.model import ChartTransformerConfig  # noqa: E402
from beatforge_api.config import get_settings  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--sequence-stride", type=int)
    parser.add_argument("--match-tolerance-ms", type=float, default=80.0)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-validation", action="store_true")
    parser.add_argument("--max-batches-per-epoch", type=int)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--feedforward", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser


def main() -> int:
    args = _parser().parse_args()
    settings = get_settings()
    dataset = (args.dataset or settings.chart_dataset_dir).expanduser().resolve()
    output = (
        (args.output or settings.chart_models_dir / "chart-transformer.pt")
        .expanduser()
        .resolve()
    )
    training = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        match_tolerance_ms=args.match_tolerance_ms,
        sequence_length=args.sequence_length,
        sequence_stride=args.sequence_stride,
        seed=args.seed,
        device=args.device,
        validation_split=None if args.no_validation else "validation",
        max_batches_per_epoch=args.max_batches_per_epoch,
    )
    model = ChartTransformerConfig(
        input_dim=len(FEATURE_NAMES),
        d_model=args.d_model,
        nhead=args.heads,
        num_layers=args.layers,
        dim_feedforward=args.feedforward,
        dropout=args.dropout,
        max_sequence_length=args.sequence_length,
    )
    result = train_chart_transformer(
        dataset,
        output,
        training=training,
        model_config=model,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
