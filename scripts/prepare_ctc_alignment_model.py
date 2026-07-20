#!/usr/bin/env python3
"""Prepare the pinned public Japanese phoneme CTC checkpoint for offline use."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

MODEL_ID = "prj-beatrice/japanese-hubert-base-phoneme-ctc-v4"
MODEL_REVISION = "f5fe07043bcb0b77a86faf72ac6d8fc1ae558f99"
MODEL_LICENSE = "Apache-2.0"
MODEL_DIRECTORY_NAME = "japanese-hubert-base-phoneme-ctc-v4"
EXPECTED_WEIGHT_BYTES = 377_659_928
FILES = (
    "config.json",
    "model.safetensors",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "vocab.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_manifest(model_dir: Path) -> Path:
    records = []
    for name in FILES:
        path = model_dir / name
        if not path.is_file():
            raise RuntimeError(f"Pinned snapshot is incomplete: {name} is missing")
        records.append(
            {
                "path": name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    weight_path = model_dir / "model.safetensors"
    if weight_path.stat().st_size != EXPECTED_WEIGHT_BYTES:
        raise RuntimeError(
            "Pinned checkpoint size mismatch: "
            f"expected {EXPECTED_WEIGHT_BYTES}, got {weight_path.stat().st_size}"
        )
    manifest = {
        "schemaVersion": 1,
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "license": MODEL_LICENSE,
        "source": f"https://huggingface.co/{MODEL_ID}/tree/{MODEL_REVISION}",
        "preparedAt": datetime.now(timezone.utc).isoformat(),
        "automaticRuntimeDownloads": False,
        "files": records,
    }
    path = model_dir / "beatforge-model.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    storage_value = Path(
        os.environ.get("BEATFORGE_STORAGE_DIR", str(project_root / "storage"))
    ).expanduser()
    storage_dir = (
        storage_value.resolve()
        if storage_value.is_absolute()
        else (project_root / storage_value).resolve()
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=storage_dir / "models" / MODEL_DIRECTORY_NAME,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise SystemExit(
            "huggingface_hub is missing; run "
            "`python scripts/beatforge.py install-vocal` first"
        ) from error

    model_dir = args.output_dir.expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=model_dir,
        allow_patterns=list(FILES),
    )
    manifest = _write_manifest(model_dir)
    print(f"Prepared {MODEL_ID}@{MODEL_REVISION}")
    print(f"Model directory: {model_dir}")
    print(f"Pinned manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
