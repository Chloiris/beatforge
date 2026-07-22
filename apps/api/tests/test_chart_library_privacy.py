from __future__ import annotations

from pathlib import Path

import pytest

from beatforge_api.chart_engine import library as library_module
from beatforge_api.chart_engine.library import ReferenceLibrary
from beatforge_api.chart_engine.models import ChartDocument, TempoPoint


def test_reference_chart_never_exposes_the_corpus_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus_root = tmp_path / "licensed-corpus"
    song_dir = corpus_root / "SPEED_CLUB" / "synthetic-song"
    song_dir.mkdir(parents=True)
    chart_path = song_dir / "synthetic_Lv5.sm"
    audio_path = song_dir / "synthetic.mp3"
    chart_path.write_text("synthetic parser input", encoding="utf-8")
    audio_path.write_bytes(b"synthetic audio placeholder")

    parsed = ChartDocument(
        id="synthetic-chart",
        title="Synthetic chart",
        artist="BeatForge Test Lab",
        music=audio_path.name,
        source_group="SPEED_CLUB",
        source_path=str(chart_path.resolve()),
        meter=5,
        bpm=120,
        duration_sec=8,
        tempo_map=[TempoPoint(beat=0, bpm=120, time_sec=0)],
        events=[],
    )
    monkeypatch.setattr(library_module, "_cached_chart", lambda *args: parsed)

    library = ReferenceLibrary(corpus_root)
    result = library.chart(library.assets()[0].id)

    assert result.source_path == "SPEED_CLUB/synthetic-song/synthetic_Lv5.sm"
    assert not Path(result.source_path).is_absolute()
    assert str(tmp_path) not in result.model_dump_json()
