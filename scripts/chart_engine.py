#!/usr/bin/env python3
"""Build and inspect the local real-data BeatForge chart engine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "api"))

from beatforge_api.chart_engine.dataset import build_dataset  # noqa: E402
from beatforge_api.chart_engine.library import ReferenceLibrary  # noqa: E402
from beatforge_api.config import get_settings  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inventory", help="Inspect the local SPEED reference corpus")

    build = subparsers.add_parser(
        "build-dataset", help="Build complete real training triples"
    )
    build.add_argument("--output", type=Path)
    build.add_argument(
        "--mode", choices=("pump-single", "pump-double"), default="pump-single"
    )
    build.add_argument(
        "--analysis-mode",
        choices=("recall", "balanced", "clean", "accurate"),
        default="balanced",
    )
    build.add_argument("--analyze-missing", action="store_true")
    build.add_argument("--limit", type=int)
    return parser


def main() -> int:
    args = _parser().parse_args()
    settings = get_settings()
    library = ReferenceLibrary(settings.speed_charts_dir)
    if args.command == "inventory":
        stats = library.statistics()
        print(
            json.dumps(
                stats.model_dump(by_alias=True, mode="json"),
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0
    report = build_dataset(
        library,
        args.output or settings.chart_dataset_dir,
        mode=args.mode,
        analyze_missing=args.analyze_missing,
        analysis_mode=args.analysis_mode,
        limit=args.limit,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
