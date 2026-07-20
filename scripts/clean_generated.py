#!/usr/bin/env python3
"""Remove disposable analysis artifacts while retaining original and demo audio."""

from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for directory in (root / "storage/waveform", root / "storage/analyses/decoded"):
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True, exist_ok=True)
    for path in (root / "storage/analyses").glob("*.json"):
        path.unlink()
    print("Removed regenerable waveform and analysis artifacts; audio files were retained.")


if __name__ == "__main__":
    main()
