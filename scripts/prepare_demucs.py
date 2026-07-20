#!/usr/bin/env python3
"""Explicitly download and verify the optional local Demucs model.

BeatForge never downloads model weights from an analysis request. This command is
the deliberate, observable preparation step for accurate mode.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and verify the optional htdemucs four-stem model."
    )
    parser.add_argument(
        "--model",
        default="htdemucs",
        choices=("htdemucs",),
        help="Demucs model to prepare (default: htdemucs).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if importlib.util.find_spec("demucs") is None or importlib.util.find_spec("torch") is None:
        print(
            "Accurate dependencies are missing. Run "
            "`python scripts/beatforge.py install-accurate` first.",
            file=sys.stderr,
        )
        return 2

    import torch
    from demucs.pretrained import get_model

    print(f"Preparing {args.model}; a model download may start now...")
    model = get_model(args.model)
    sources = tuple(str(source) for source in getattr(model, "sources", ()))
    required_sources = {"vocals", "drums", "bass", "other"}
    if not required_sources.issubset(sources):
        print(f"Unexpected model sources: {sources}", file=sys.stderr)
        return 3

    checkpoint_dir = Path(torch.hub.get_dir()) / "checkpoints"
    checkpoints = sorted(checkpoint_dir.glob("*.th"))
    if not checkpoints:
        print("Model loaded, but no local checkpoint was found.", file=sys.stderr)
        return 4

    print(f"Model ready: {args.model}")
    print(f"Sources: {', '.join(sources)}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
