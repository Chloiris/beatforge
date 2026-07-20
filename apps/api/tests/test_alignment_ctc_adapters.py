from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from scripts.ctc_phoneme_align import ctc_viterbi

from beatforge_api.audio.alignment.base import (
    AdapterDiagnostics,
    AlignmentAdapterError,
    AlignmentContext,
)
from beatforge_api.audio.alignment.ctc_adapter import CTCAlignmentAdapter
from beatforge_api.audio.alignment.singing_adapter import SingingAlignmentAdapter


def _context(tmp_path: Path) -> AlignmentContext:
    vocals = tmp_path / "vocals.wav"
    sf.write(vocals, np.zeros(1_600, dtype=np.float32), 16_000)
    return AlignmentContext(
        track_id="track-ctc",
        lyrics="未来",
        lyrics_format="japanese",
        vocals_path=vocals,
        sample_rate=16_000,
        sample_count=1_600,
        tempo_map=(),
        models_dir=tmp_path / "models",
        storage_dir=tmp_path / "storage",
        project_root=tmp_path,
    )


def test_ctc_viterbi_uses_observed_label_frames() -> None:
    log_probs = np.full((5, 3), -10.0, dtype=np.float32)
    for frame, token_id in enumerate((0, 1, 0, 2, 0)):
        log_probs[frame, token_id] = 0.0

    paths, score = ctc_viterbi(log_probs, [1, 2], blank_id=0)

    assert score == pytest.approx(0.0)
    assert [(item.first_frame, item.last_frame) for item in paths] == [(1, 1), (3, 3)]
    assert [item.confidence for item in paths] == pytest.approx([1.0, 1.0])


def test_ctc_adapter_preserves_phone_surface_and_real_spans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    adapter = CTCAlignmentAdapter()
    monkeypatch.setattr(
        CTCAlignmentAdapter,
        "diagnostics",
        lambda self, context=None: AdapterDiagnostics(available=True),
    )
    monkeypatch.setattr(
        CTCAlignmentAdapter,
        "_paths",
        staticmethod(lambda context=None: (Path(sys.executable), tmp_path / "helper.py", tmp_path)),
    )

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        output = Path(command[command.index("--output") + 1])
        output.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "source_sample_rate": 16_000,
                    "phones": [
                        {
                            "surface": "未来",
                            "phoneme": "m",
                            "start_sample": 100,
                            "end_sample": 200,
                            "confidence": 0.8,
                        },
                        {
                            "surface": "未来",
                            "phoneme": "i",
                            "start_sample": 300,
                            "end_sample": 400,
                            "confidence": 0.7,
                        },
                    ],
                    "metadata": {"surfaceSequence": ["未来"]},
                    "warnings": [],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    output = adapter.run(context)

    assert [(item.text, item.phoneme) for item in output.tokens] == [
        ("未来", "m"),
        ("未来", "i"),
    ]
    assert [(item.start_sample, item.end_sample) for item in output.tokens] == [
        (100, 200),
        (300, 400),
    ]
    assert output.metadata["alignedText"] == "未来"
    assert "no lyric timestamps" in output.metadata["timestampProvenance"]


def test_singing_adapter_records_unsupported_japanese(tmp_path: Path) -> None:
    context = _context(tmp_path)
    adapter = SingingAlignmentAdapter()

    diagnostics = adapter.diagnostics(context)

    assert diagnostics.available is False
    assert diagnostics.details["failureCode"] == "UNSUPPORTED_LANGUAGE"
    assert diagnostics.details["approximatePhoneMappingAllowed"] is False
    with pytest.raises(AlignmentAdapterError) as captured:
        adapter.run(context)
    assert captured.value.status == "unavailable"
    assert captured.value.code == "SINGING_MODEL_UNSUPPORTED_LANGUAGE"
