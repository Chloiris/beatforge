from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from scripts.ctc_phoneme_align import (
    AlignmentScriptError,
    load_phone_targets_from_plan,
)

from beatforge_api.audio.alignment.base import (
    AdapterOutput,
    AlignmentContext,
)
from beatforge_api.audio.alignment.hubert_ctc import HubertCTCAlignmentAdapter
from beatforge_api.audio.alignment.hubert_engine import build_hubert_artifacts
from beatforge_api.audio.alignment.lyric_processor import (
    JapaneseG2PBackend,
    ProcessedLyrics,
    map_characters_to_moras_dp,
    map_phoneme_sequences_dp,
    map_phones_to_moras_dp,
    process_japanese_lyrics,
    tokenize_moras,
)
from beatforge_api.audio.alignment.schema import AlignmentResult, AlignmentToken

_LYRICS = "仮名例(デジャヴ) きゅーっと 仮名例(デジャヴ) きゅーっと"
_PHONE_MAP = {
    "デ": ("d", "e"),
    "ジャ": ("j", "a"),
    "ヴ": ("v", "u"),
    "きゅ": ("ky", "u"),
    "と": ("t", "o"),
    "か": ("k", "a"),
    "が": ("g", "a"),
    "や": ("y", "a"),
    "い": ("i",),
    "て": ("t", "e"),
    "カ": ("k", "a"),
    "ガ": ("g", "a"),
    "ヤ": ("y", "a"),
}
_READING_MAP = {"仮": "デ", "名": "ジャ", "例": "ヴ", "輝": "カガヤ"}


class _DeterministicJapaneseBackend:
    """Tiny lexical backend; planning and mapping remain the production algorithms."""

    @property
    def engine_name(self) -> str:
        return "deterministic-japanese-g2p-test-1"

    def frontend(self, text: str) -> list[dict[str, Any]]:
        return [
            {"string": surface, "read": self.kana(surface)}
            for surface in text.split()
        ]

    def kana(self, text: str) -> str:
        return "".join(_READING_MAP.get(character, character) for character in text)

    def phones(self, text: str) -> tuple[str, ...]:
        phones: list[str] = []
        previous_vowel = "u"
        for mora in tokenize_moras(text):
            if mora.kind == "long_vowel":
                values = (previous_vowel,)
            elif mora.kind == "sokuon":
                values = ("cl",)
            else:
                values = _PHONE_MAP[mora.kana]
            phones.extend(values)
            for value in reversed(values):
                if value in {"a", "i", "u", "e", "o"}:
                    previous_vowel = value
                    break
        return tuple(phones)


@pytest.fixture
def backend() -> JapaneseG2PBackend:
    return _DeterministicJapaneseBackend()


@pytest.fixture
def repeated_plan(backend: JapaneseG2PBackend) -> ProcessedLyrics:
    return process_japanese_lyrics(_LYRICS, backend=backend)


def _contains_timing_field(value: object) -> bool:
    if isinstance(value, dict):
        return any(
            "sample" in str(key).casefold()
            or "timestamp" in str(key).casefold()
            or _contains_timing_field(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_timing_field(item) for item in value)
    return False


def _write_plan(path: Path, plan: ProcessedLyrics) -> dict[str, Any]:
    payload = {"status": "ok", **plan.to_dict()}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


def test_japanese_plan_handles_ruby_mora_kinds_and_repeated_occurrences(
    repeated_plan: ProcessedLyrics,
) -> None:
    plan = repeated_plan

    assert plan.spoken_text == "デジャヴ きゅーっと デジャヴ きゅーっと"
    assert [(item.base, item.reading) for item in plan.annotations] == [
        ("仮名例", "デジャヴ"),
        ("仮名例", "デジャヴ"),
    ]
    assert "仮名例" not in plan.spoken_text
    assert {mora.kind for mora in plan.moras} >= {
        "long_vowel",
        "sokuon",
        "yoon",
    }
    assert [(item.text, item.occurrence) for item in plan.characters if item.text == "仮"] == [
        ("仮", 0),
        ("仮", 1),
    ]

    # Repeated lyrics retain identical pronunciation but distinct occurrence-stable indices.
    assert plan.phone_sequence[:12] == plan.phone_sequence[12:]
    assert plan.characters[0].phoneme_indices != plan.characters[8].phoneme_indices
    assert plan.characters[0].id != plan.characters[8].id
    assert len({item.id for item in plan.phonemes}) == len(plan.phonemes)
    assert _contains_timing_field(plan.to_dict()) is False


def test_mora_phone_character_and_observed_phone_mapping_use_dp(
    backend: JapaneseG2PBackend,
) -> None:
    moras = tokenize_moras("きゅーっと")
    assert [(item.kana, item.kind) for item in moras] == [
        ("きゅ", "yoon"),
        ("ー", "long_vowel"),
        ("っ", "sokuon"),
        ("と", "mora"),
    ]

    mora_mapping = map_phones_to_moras_dp(
        moras,
        ("ky", "u", "u", "cl", "t", "o"),
        backend,
    )
    assert mora_mapping.assignments == ((0, 1), (2,), (3,), (4, 5))
    assert mora_mapping.cost == pytest.approx(0.0)

    character_mapping = map_characters_to_moras_dp(
        ["仮", "名", "例"],
        tokenize_moras("デジャヴ"),
        backend,
    )
    assert character_mapping.assignments == ((0,), (1,), (2,))
    assert character_mapping.cost == pytest.approx(0.0)

    observed_mapping = map_phoneme_sequences_dp(
        ("ky", "u", "cl", "t", "o"),
        ("ky", "u", "x", "cl", "d", "o"),
    )
    assert observed_mapping.inserted_observed_indices == (2,)
    assert [item.operation for item in observed_mapping.matches] == [
        "match",
        "match",
        "match",
        "substitute",
        "match",
    ]
    assert [item.observed_index for item in observed_mapping.matches] == [0, 1, 3, 4, 5]


def test_ctc_helper_accepts_only_hashed_no_time_g2p_plan(
    tmp_path: Path,
    repeated_plan: ProcessedLyrics,
) -> None:
    plan_path = tmp_path / "plan.json"
    payload = _write_plan(plan_path, repeated_plan)

    targets, metadata = load_phone_targets_from_plan(plan_path, _LYRICS)

    assert len(targets) == len(repeated_plan.phonemes)
    assert [item.plan_phone_index for item in targets] == list(range(len(targets)))
    assert [item.mora_index for item in targets] == [
        item.mora_index for item in repeated_plan.phonemes
    ]
    assert [item.character_indices for item in targets] == [
        item.character_indices for item in repeated_plan.phonemes
    ]
    assert metadata["lyricPlanHash"] == payload["ctcPlan"]["planHash"]
    expected_plan = {
        key: value for key, value in payload.items() if key != "status"
    }
    assert metadata["lyricPlan"] == expected_plan
    assert _contains_timing_field(metadata["lyricPlan"]) is False


@pytest.mark.parametrize("forbidden_key", ["alignedSample", "wordTimestamp"])
def test_ctc_helper_rejects_any_timing_field_in_g2p_plan(
    tmp_path: Path,
    repeated_plan: ProcessedLyrics,
    forbidden_key: str,
) -> None:
    plan_path = tmp_path / f"forbidden-{forbidden_key}.json"
    payload = {"status": "ok", **copy.deepcopy(repeated_plan.to_dict())}
    payload["phonemes"][0][forbidden_key] = 123
    plan_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    with pytest.raises(AlignmentScriptError) as captured:
        load_phone_targets_from_plan(plan_path, _LYRICS)

    assert captured.value.code == "CTC_G2P_PLAN_HAS_TIMESTAMPS"


def _synthetic_context(
    tmp_path: Path,
    phone_count: int,
    lyrics: str = "きゅーっと",
) -> tuple[AlignmentContext, list[tuple[int, int]]]:
    sample_rate = 16_000
    sample_count = max(32_000, 1_000 + max(0, phone_count - 1) * 4_800 + 3_500)
    time_axis = np.arange(sample_count, dtype=np.float32) / sample_rate
    audio = np.asarray(0.008 * np.sin(2 * np.pi * 220.0 * time_axis), dtype=np.float32)
    spans: list[tuple[int, int]] = []
    for index in range(phone_count):
        start = 1_000 + index * 4_800
        end = start + 3_000
        spans.append((start, end))
        for center in (start + 500, end - 500):
            burst_start = center - 100
            burst_end = center + 100
            burst_phase = np.arange(burst_end - burst_start, dtype=np.float32)
            burst = np.asarray(
                0.8
                * np.sin(2 * np.pi * 950.0 * burst_phase / sample_rate)
                * np.hanning(burst_end - burst_start),
                dtype=np.float32,
            )
            audio[burst_start:burst_end] += burst
    vocals = tmp_path / "synthetic-vocals.wav"
    sf.write(vocals, audio, sample_rate)
    return (
        AlignmentContext(
            track_id="synthetic-hubert",
            lyrics=lyrics,
            lyrics_format="japanese",
            vocals_path=vocals,
            sample_rate=sample_rate,
            sample_count=sample_count,
            tempo_map=(),
            models_dir=tmp_path / "models",
            storage_dir=tmp_path / "storage",
            project_root=tmp_path,
        ),
        spans,
    )


def test_hubert_adapter_refines_synthetic_vocals_and_aggregates_hierarchy(
    tmp_path: Path,
    backend: JapaneseG2PBackend,
) -> None:
    processed = process_japanese_lyrics("きゅーっと", backend=backend)
    context, raw_spans = _synthetic_context(tmp_path, len(processed.phonemes))
    raw_tokens = tuple(
        AlignmentToken(
            id=f"raw-{index}",
            text=phone.text,
            phoneme=phone.phoneme,
            start_sample=start,
            end_sample=end,
            confidence=0.7,
            method="ctc",
        )
        for index, (phone, (start, end)) in enumerate(
            zip(processed.phonemes, raw_spans, strict=True)
        )
    )
    base_output = AdapterOutput(tokens=raw_tokens, metadata={"totalElapsedSec": 1.0})
    payload = {"metadata": {"lyricPlan": processed.to_dict()}}

    output = HubertCTCAlignmentAdapter()._postprocess_output(
        context,
        payload,
        base_output,
    )

    assert output.hierarchy is not None
    hierarchy = output.hierarchy
    assert len(hierarchy.phonemes) == len(output.tokens) == len(processed.phonemes)
    assert len(hierarchy.moras) == len(processed.moras)
    assert len(hierarchy.characters) == len(processed.characters)
    assert output.metadata["dynamicProgramming"]["expectedObservedPhoneCost"] == 0.0
    assert output.metadata["acousticRefinement"]["features"] == [
        "vocalRms",
        "spectralChange",
        "pitchChange",
    ]
    assert output.metadata["acousticRefinement"]["tempoUsed"] is False
    assert output.metadata["acousticRefinement"]["changedBoundaryCount"] > 0

    changed = 0
    for index, (raw, refined, phone) in enumerate(
        zip(raw_tokens, output.tokens, hierarchy.phonemes, strict=True)
    ):
        assert phone.match_operation == "match"
        assert phone.observed_token_index == index
        assert phone.aligned_start_sample == raw.start_sample
        assert phone.aligned_end_sample == raw.end_sample
        assert phone.aligned_sample == raw.start_sample
        assert phone.refined_start_sample == refined.start_sample
        assert phone.refined_end_sample == refined.end_sample
        assert phone.refined_sample == refined.start_sample
        assert phone.evidence is not None
        assert 0.0 <= phone.evidence.energy <= 1.0
        assert 0.0 <= phone.evidence.spectral_change <= 1.0
        assert 0.0 <= phone.evidence.pitch_change <= 1.0
        serialized = phone.model_dump(by_alias=True)
        assert serialized["alignedSample"] == raw.start_sample
        assert serialized["refinedSample"] == refined.start_sample
        assert serialized["evidence"] == {
            "energy": phone.evidence.energy,
            "spectralChange": phone.evidence.spectral_change,
            "pitchChange": phone.evidence.pitch_change,
        }
        changed += int(
            raw.start_sample != refined.start_sample or raw.end_sample != refined.end_sample
        )
    assert changed > 0

    for aggregate in (*hierarchy.moras, *hierarchy.characters):
        members = [hierarchy.phonemes[index] for index in aggregate.phoneme_indices]
        assert aggregate.aligned_start_sample == min(
            item.aligned_start_sample for item in members
        )
        assert aggregate.aligned_end_sample == max(item.aligned_end_sample for item in members)
        assert aggregate.refined_start_sample == min(
            item.refined_start_sample for item in members
        )
        assert aggregate.refined_end_sample == max(item.refined_end_sample for item in members)
        assert aggregate.evidence is not None
        assert aggregate.evidence.energy == max(
            item.evidence.energy for item in members if item.evidence is not None
        )


def test_kagayaite_decodes_five_moras_and_five_chart_candidates(
    tmp_path: Path,
    backend: JapaneseG2PBackend,
) -> None:
    lyrics = "かがやいて"
    processed = process_japanese_lyrics(lyrics, backend=backend)
    context, raw_spans = _synthetic_context(
        tmp_path,
        len(processed.phonemes),
        lyrics,
    )
    base_output = AdapterOutput(
        tokens=tuple(
            AlignmentToken(
                id=f"raw-kagayaite-{index}",
                text=phone.text,
                phoneme=phone.phoneme,
                start_sample=start,
                end_sample=end,
                confidence=0.88 - index * 0.01,
                method="ctc",
            )
            for index, (phone, (start, end)) in enumerate(
                zip(processed.phonemes, raw_spans, strict=True)
            )
        ),
        metadata={"totalElapsedSec": 1.0},
    )
    output = HubertCTCAlignmentAdapter()._postprocess_output(
        context,
        {"metadata": {"lyricPlan": processed.to_dict()}},
        base_output,
    )
    assert output.hierarchy is not None
    now = datetime.now(UTC)
    result = AlignmentResult(
        run_id="kagayaite-mora-run",
        track_id=context.track_id,
        method="ctc",
        status="completed",
        sample_rate=context.sample_rate,
        sample_count=context.sample_count,
        tokens=list(output.tokens),
        hierarchy=output.hierarchy,
        warnings=list(output.warnings),
        metadata=output.metadata,
        created_at=now,
        updated_at=now,
    )

    artifacts = build_hubert_artifacts(context, result)
    mora_events = artifacts.candidates.mora_events
    base_candidates = [
        event for event in artifacts.candidates.events if event.policy == "mora"
    ]

    assert [event.mora for event in mora_events] == ["か", "が", "や", "い", "て"]
    assert [event.text for event in mora_events] == ["か", "が", "や", "い", "て"]
    assert [event.phonemes for event in mora_events] == [
        ["k", "a"],
        ["g", "a"],
        ["y", "a"],
        ["i"],
        ["t", "e"],
    ]
    assert len(base_candidates) == len(mora_events) == 5
    assert [event.mora_index for event in base_candidates] == list(range(5))
    assert [event.refined_sample for event in base_candidates] == [
        event.refined_start_sample for event in mora_events
    ]
    assert all(event.character == event.mora for event in base_candidates)
    assert artifacts.report.counts["baseMoraCandidates"] == 5
    assert artifacts.report.details["chartCandidateSource"] == "MoraEvent"

    # Run the same observed phone spans through the complete mixed-kanji path:
    # 輝 is display-only while its カ/ガ/ヤ morae each become a chart candidate.
    mixed_plan = process_japanese_lyrics("輝いて", backend=backend)
    mixed_context = replace(context, lyrics="輝いて")
    mixed_output = HubertCTCAlignmentAdapter()._postprocess_output(
        mixed_context,
        {"metadata": {"lyricPlan": mixed_plan.to_dict()}},
        base_output,
    )
    assert mixed_output.hierarchy is not None
    mixed_result = AlignmentResult(
        run_id="kagayaite-mixed-kanji-run",
        track_id=mixed_context.track_id,
        method="ctc",
        status="completed",
        sample_rate=mixed_context.sample_rate,
        sample_count=mixed_context.sample_count,
        tokens=list(mixed_output.tokens),
        hierarchy=mixed_output.hierarchy,
        warnings=list(mixed_output.warnings),
        metadata=mixed_output.metadata,
        created_at=now,
        updated_at=now,
    )
    mixed_artifacts = build_hubert_artifacts(mixed_context, mixed_result)
    mixed_events = mixed_artifacts.candidates.mora_events
    mixed_candidates = [
        event
        for event in mixed_artifacts.candidates.events
        if event.policy == "mora"
    ]

    assert [event.mora for event in mixed_events] == ["カ", "ガ", "ヤ", "い", "て"]
    assert [event.text for event in mixed_events] == ["カ", "ガ", "ヤ", "い", "て"]
    assert [
        [parent.text for parent in event.parent_characters]
        for event in mixed_events[:3]
    ] == [["輝"], ["輝"], ["輝"]]
    assert [(event.character, event.mora) for event in mixed_candidates] == [
        ("輝", "カ"),
        ("輝", "ガ"),
        ("輝", "ヤ"),
        ("い", "い"),
        ("て", "て"),
    ]
    assert [event.mora_index for event in mixed_candidates] == list(range(5))
    assert len({event.alignment_unit_id for event in mixed_candidates}) == 5
    assert [event.refined_sample for event in mixed_candidates] == [
        event.refined_start_sample for event in mixed_events
    ]
