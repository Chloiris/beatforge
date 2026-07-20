"""Explicitly download the optional local vocal models into BeatForge storage."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download
import pyopenjtalk

MODELS = {
    "asr": "Qwen/Qwen3-ASR-1.7B",
    "aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
}


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=project_root / "storage" / "models",
    )
    parser.add_argument("--only", choices=("all", "asr", "aligner"), default="all")
    args = parser.parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)

    selected = MODELS.items() if args.only == "all" else [(args.only, MODELS[args.only])]
    for kind, repository in selected:
        target = args.models_dir / repository.rsplit("/", 1)[-1]
        if (target / "config.json").is_file():
            print(f"{kind}: already available at {target}")
            continue
        print(f"{kind}: downloading {repository} to {target}")
        snapshot_download(repo_id=repository, local_dir=target)
        print(f"{kind}: ready at {target}")

    # pyopenjtalk downloads its modified-BSD Open JTalk dictionary on the first
    # conversion. Do that only in this explicit preparation command so analysis
    # jobs remain fully offline and never surprise the user with a network call.
    reading = pyopenjtalk.g2p("日本語", kana=True)
    dictionary_path = Path(os.fsdecode(pyopenjtalk.OPEN_JTALK_DICT_DIR))
    print(f"open-jtalk: ready at {dictionary_path} ({reading})")


if __name__ == "__main__":
    main()
