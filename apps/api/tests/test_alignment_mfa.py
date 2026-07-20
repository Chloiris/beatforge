from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from beatforge_api.audio.alignment.base import (
    AdapterDiagnostics,
    AlignmentAdapterError,
    AlignmentContext,
)
from beatforge_api.audio.alignment.mfa_adapter import (
    MFAAlignmentAdapter,
    _G2PWord,
    _openjtalk_phones_to_mfa,
    _parse_long_textgrid,
)
from beatforge_api.platform_paths import venv_executable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
QWEN_PYTHON = venv_executable(PROJECT_ROOT, ".venv-qwen")


def _context(tmp_path: Path, *, sample_rate: int = 1_000) -> AlignmentContext:
    vocals_path = tmp_path / "vocals.flac"
    sf.write(
        vocals_path,
        np.zeros(sample_rate, dtype=np.float32),
        sample_rate,
        format="FLAC",
    )
    return AlignmentContext(
        track_id="track",
        lyrics="未来",
        lyrics_format="japanese",
        vocals_path=vocals_path,
        sample_rate=sample_rate,
        sample_count=sample_rate,
        tempo_map=(),
        models_dir=tmp_path / "models",
        storage_dir=tmp_path / "storage",
        project_root=tmp_path,
    )


def _textgrid() -> str:
    return '''File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 0.1
            text = ""
        intervals [2]:
            xmin = 0.1
            xmax = 0.5
            text = "bfw00000"
        intervals [3]:
            xmin = 0.5
            xmax = 0.9
            text = "bfw00001"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1
        intervals: size = 6
        intervals [1]:
            xmin = 0
            xmax = 0.1
            text = "sil"
        intervals [2]:
            xmin = 0.1
            xmax = 0.2
            text = "m"
        intervals [3]:
            xmin = 0.2
            xmax = 0.5
            text = "i"
        intervals [4]:
            xmin = 0.5
            xmax = 0.65
            text = "dʑ"
        intervals [5]:
            xmin = 0.65
            xmax = 0.9
            text = "ɯ"
        intervals [6]:
            xmin = 0.9
            xmax = 1
            text = "sp"
'''


def test_openjtalk_phone_conversion_uses_japanese_mfa_inventory() -> None:
    assert _openjtalk_phones_to_mfa(
        ["k", "I", "sh", "i", "cl", "k", "a", "N"]
    ) == ("k", "i̥", "ɕ", "i", "kː", "a", "ɴ")

    with pytest.raises(AlignmentAdapterError) as raised:
        _openjtalk_phones_to_mfa(["not-a-real-openjtalk-phone"])

    assert raised.value.code == "MFA_G2P_UNSUPPORTED_PHONE"


def test_textgrid_phone_intervals_become_real_alignment_tokens(tmp_path: Path) -> None:
    context = _context(tmp_path)
    path = tmp_path / "alignment.TextGrid"
    path.write_text(_textgrid(), encoding="utf-8")
    words = [
        _G2PWord("bfw00000", "未", ("m", "i")),
        _G2PWord("bfw00001", "来", ("dʑ", "ɯ")),
    ]

    tiers = _parse_long_textgrid(path.read_text(encoding="utf-8"))
    tokens, metadata = MFAAlignmentAdapter()._tokens_from_textgrid(context, path, words)

    assert set(tiers) == {"words", "phones"}
    assert [(token.text, token.phoneme) for token in tokens] == [
        ("未", "m"),
        ("未", "i"),
        ("来", "dʑ"),
        ("来", "ɯ"),
    ]
    assert [(token.start_sample, token.end_sample) for token in tokens] == [
        (100, 200),
        (200, 500),
        (500, 650),
        (650, 900),
    ]
    assert all(token.confidence == 0.0 for token in tokens)
    assert metadata["skippedSilencePhoneCount"] == 2


def test_unavailable_mfa_never_returns_tokens(tmp_path: Path) -> None:
    class UnavailableAdapter(MFAAlignmentAdapter):
        def diagnostics(self, context=None):  # type: ignore[no-untyped-def]
            return AdapterDiagnostics(
                available=False,
                reason="missing for test",
                details={"issues": ["mfa_cli_missing_or_broken"]},
            )

    with pytest.raises(AlignmentAdapterError) as raised:
        UnavailableAdapter().run(_context(tmp_path))

    assert raised.value.status == "unavailable"
    assert raised.value.code == "MFA_RUNTIME_UNAVAILABLE"


def test_model_download_is_confined_to_storage_models(tmp_path: Path, monkeypatch) -> None:
    context = _context(tmp_path)
    executable = tmp_path / "mfa"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    captured_root: list[str] = []

    def fake_run(command, **kwargs):  # type: ignore[no-untyped-def]
        assert command[:4] == [str(executable), "model", "download", "--version"]
        captured_root.append(kwargs["env"]["MFA_ROOT_DIR"])
        target = context.models_dir / "mfa" / "pretrained_models" / "acoustic"
        target.mkdir(parents=True)
        (target / "japanese_mfa.zip").write_bytes(b"real model placeholder")
        return subprocess.CompletedProcess(command, 0, stdout="downloaded", stderr="")

    monkeypatch.setattr(
        "beatforge_api.audio.alignment.mfa_adapter.subprocess.run",
        fake_run,
    )

    model, downloaded = MFAAlignmentAdapter()._ensure_acoustic_model(context, executable)

    assert downloaded is True
    assert model.is_relative_to(context.models_dir)
    assert captured_root == [str(context.models_dir / "mfa")]


def test_run_uses_only_textgrid_boundaries(tmp_path: Path, monkeypatch) -> None:
    context = _context(tmp_path)
    executable = tmp_path / "mfa"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    model = tmp_path / "japanese_mfa.zip"
    model.write_bytes(b"model")

    class FakeAdapter(MFAAlignmentAdapter):
        def diagnostics(self, context=None):  # type: ignore[no-untyped-def]
            return AdapterDiagnostics(available=True)

        @classmethod
        def _mfa_executable(cls, context=None):  # type: ignore[no-untyped-def]
            return executable

        def _ensure_acoustic_model(self, context, mfa):  # type: ignore[no-untyped-def]
            return model, False

        def _generate_pronunciations(self, context, lyrics):  # type: ignore[no-untyped-def]
            return [
                _G2PWord("bfw00000", "未", ("m", "i")),
                _G2PWord("bfw00001", "来", ("dʑ", "ɯ")),
            ]

        def _legacy_align_command(self, mfa, environment):  # type: ignore[no-untyped-def]
            return "align_one"

    def fake_run(command, **_kwargs):  # type: ignore[no-untyped-def]
        Path(command[-1]).write_text(_textgrid(), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="aligned", stderr="")

    monkeypatch.setattr(
        "beatforge_api.audio.alignment.mfa_adapter.subprocess.run",
        fake_run,
    )

    output = FakeAdapter().run(context)

    assert [token.start_sample for token in output.tokens] == [100, 200, 500, 650]
    assert output.metadata["timestampProvenance"] == "MFA TextGrid phone intervals"
    assert output.metadata["confidenceProvenance"] == "unavailable_from_mfa_textgrid"


@pytest.mark.skipif(
    not QWEN_PYTHON.is_file(),
    reason="optional pyopenjtalk runtime is not installed",
)
def test_real_pyopenjtalk_g2p_is_used(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("BEATFORGE_PYOPENJTALK_PYTHON", str(QWEN_PYTHON))
    context = _context(tmp_path)
    words = MFAAlignmentAdapter()._generate_pronunciations(context, "未来")

    assert words
    assert "".join(word.text for word in words) == "未来"
    assert [phone for word in words for phone in word.phones]
    assert all("pau" not in word.phones for word in words)
