from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import inspect, select

from beatforge_api import routes
from beatforge_api.audio.alignment.base import (
    AdapterDiagnostics,
    AdapterOutput,
    AlignmentAdapter,
    AlignmentContext,
    TempoReference,
)
from beatforge_api.audio.alignment.hubert_engine import (
    HubertAlignmentReport,
    HubertCandidateBundle,
    build_hubert_artifacts,
    persist_hubert_candidates,
    publish_hubert_artifacts,
)
from beatforge_api.audio.alignment.hybrid import HybridAlignmentAdapter
from beatforge_api.audio.alignment.runner import AlignmentRunner
from beatforge_api.audio.alignment.schema import (
    AlignmentAcousticEvidence,
    AlignmentHierarchy,
    AlignmentHierarchyUnit,
    AlignmentReport,
    AlignmentResult,
    AlignmentToken,
)
from beatforge_api.database import SessionLocal, engine
from beatforge_api.models import CandidateEventModel, ProjectModel, TrackModel
from beatforge_api.serialization import candidate_event_dict, dumps


class StubAdapter(AlignmentAdapter):
    name = "stub"

    def __init__(self, method: str) -> None:
        self.method = method  # type: ignore[assignment]

    def diagnostics(self, context: AlignmentContext | None = None) -> AdapterDiagnostics:
        return AdapterDiagnostics(available=True, model="stub")

    def run(self, context: AlignmentContext) -> AdapterOutput:
        if self.method == "ctc":
            result = _result()
            return AdapterOutput(
                tokens=tuple(result.tokens),
                hierarchy=result.hierarchy,
                metadata={"engineVersion": "0.6", "totalElapsedSec": 5.25},
            )
        return AdapterOutput(
            tokens=(
                AlignmentToken(
                    id=f"{self.method}-token",
                    text="歌",
                    start_sample=100,
                    end_sample=180,
                    confidence=0.8,
                    method=self.method,  # type: ignore[arg-type]
                ),
            )
        )


class StubEvaluator:
    def evaluate(self, context: AlignmentContext, result: AlignmentResult) -> AlignmentReport:
        return AlignmentReport(
            run_id=result.run_id,
            track_id=context.track_id,
            method=result.method,
            score=0.8,
            coverage=1.0,
            acoustic=0.7,
            rhythm=0.6,
            stability=1.0,
            lyric_token_count=8,
            aligned_token_count=len(result.tokens),
            created_at=datetime.now(UTC),
        )


class ArtifactRunnerStub:
    def __init__(
        self,
        candidates: HubertCandidateBundle | None,
        report: HubertAlignmentReport | None,
    ) -> None:
        self.candidates = candidates
        self.report = report

    def get_hubert_candidates(self, track_id: str) -> HubertCandidateBundle | None:
        assert track_id == "track-hubert"
        return self.candidates

    def get_hubert_report(self, track_id: str) -> HubertAlignmentReport | None:
        assert track_id == "track-hubert"
        return self.report


def _context(tmp_path: Path, *, tempo: bool = True) -> AlignmentContext:
    vocals = tmp_path / "storage" / "stems" / "track-hubert" / "vocals.flac"
    vocals.parent.mkdir(parents=True, exist_ok=True)
    vocals.touch()
    return AlignmentContext(
        track_id="track-hubert",
        lyrics="遠ーくへ歌う",
        lyrics_format="japanese",
        vocals_path=vocals,
        sample_rate=8_000,
        sample_count=16_000,
        tempo_map=(TempoReference(0, 120.0, 0),) if tempo else (),
        models_dir=tmp_path / "storage" / "models",
        storage_dir=tmp_path / "storage",
        project_root=tmp_path,
        song="Synthetic Vocal Demo",
        artist="Demo Artist",
    )


def _phone(index: int, sample: int, *, pitch: float = 0.2) -> AlignmentHierarchyUnit:
    owner_index = 1 if index == 3 else index if index < 3 else index - 1
    return AlignmentHierarchyUnit(
        id=f"phone-{index}",
        index=index,
        level="phoneme",
        text=f"字{index}",
        kana=f"かな{index}",
        phoneme="u" if index in {1, 3, 8} else "k",
        character_indices=[owner_index],
        mora_indices=[owner_index],
        aligned_start_sample=sample,
        aligned_end_sample=sample + 80,
        refined_start_sample=sample + 10,
        refined_end_sample=sample + 90,
        aligned_sample=sample,
        refined_sample=sample + 10,
        confidence=0.9,
        observed_token_index=index,
        match_operation="match",
        evidence=AlignmentAcousticEvidence(
            energy=0.7,
            spectral_change=0.6,
            pitch_change=pitch,
        ),
    )


def _unit(
    kind: str,
    index: int,
    sample: int,
    *,
    text: str,
    phone_indices: list[int],
    mora_indices: list[int],
    unit_kind: str | None = None,
) -> AlignmentHierarchyUnit:
    return AlignmentHierarchyUnit(
        id=f"{kind}-{index}",
        index=index,
        level=kind,  # type: ignore[arg-type]
        text=text,
        kana=text,
        phoneme="",
        kind=unit_kind,
        character_indices=[index],
        mora_indices=mora_indices,
        phoneme_indices=phone_indices,
        aligned_start_sample=sample,
        aligned_end_sample=sample + 120,
        refined_start_sample=sample + 10,
        refined_end_sample=sample + 130,
        aligned_sample=sample,
        refined_sample=sample + 10,
        confidence=0.9,
    )


def _result() -> AlignmentResult:
    samples = [100, 500, 900, 1_300, 1_700, 2_100, 2_500, 2_900]
    phone_samples = [100, 500, 900, 1_200, 1_300, 1_700, 2_100, 2_500, 2_900]
    character_texts = ["遠", "ー", "く", "へ", "歌", "う", "空", "雨"]
    phones = [_phone(index, sample) for index, sample in enumerate(phone_samples)]
    # A second observed phone inside the long-vowel character has a measured
    # pitch change. It is the only source of the extra long-vowel candidate.
    phones[3] = _phone(3, 1_200, pitch=0.8)
    moras = [
        _unit(
            "mora",
            index,
            sample,
            text="ー" if index == 1 else f"も{index}",
            phone_indices=(
                [1, 3]
                if index == 1
                else [index if index < 3 else index + 1]
            ),
            mora_indices=[index],
            unit_kind="long_vowel" if index == 1 else "mora",
        )
        for index, sample in enumerate(samples)
    ]
    characters = [
        _unit(
            "character",
            index,
            sample,
            text=character_texts[index],
            phone_indices=(
                [1, 3]
                if index == 1
                else [index if index < 3 else index + 1]
            ),
            mora_indices=[index],
        )
        for index, sample in enumerate(samples)
    ]
    return AlignmentResult(
        run_id="hubert-run",
        track_id="track-hubert",
        method="ctc",
        status="completed",
        sample_rate=8_000,
        sample_count=16_000,
        tokens=[
            AlignmentToken(
                id=f"token-{phone.index}",
                text=phone.text,
                phoneme=phone.phoneme,
                start_sample=phone.refined_start_sample,
                end_sample=phone.refined_end_sample,
                confidence=phone.confidence,
                method="ctc",
            )
            for phone in phones
        ],
        hierarchy=AlignmentHierarchy(
            phonemes=phones,
            moras=moras,
            characters=characters,
        ),
        metadata={"engineVersion": "0.6", "totalElapsedSec": 5.25},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _qwen_report() -> AlignmentReport:
    return AlignmentReport(
        run_id="qwen-run",
        track_id="track-hubert",
        method="qwen",
        score=0.7,
        coverage=0.625,
        acoustic=0.6,
        rhythm=0.5,
        stability=1.0,
        lyric_token_count=8,
        aligned_token_count=5,
        created_at=datetime.now(UTC),
    )


def test_hierarchy_candidates_keep_real_samples_and_tempo_never_adds_events(
    tmp_path: Path,
) -> None:
    result = _result()
    with_tempo = build_hubert_artifacts(
        _context(tmp_path),
        result,
        qwen_report=_qwen_report(),
    )
    without_tempo = build_hubert_artifacts(_context(tmp_path, tempo=False), result)

    assert len(with_tempo.candidates.events) == len(without_tempo.candidates.events) == 9
    assert [event.acoustic_sample for event in with_tempo.candidates.events] == [
        event.refined_sample for event in with_tempo.candidates.events
    ]
    assert {
        (event.character, event.aligned_sample, event.refined_sample)
        for event in with_tempo.candidates.events
    } == {
        (event.character, event.aligned_sample, event.refined_sample)
        for event in without_tempo.candidates.events
    }
    assert any(
        event.chart_sample != event.acoustic_sample for event in with_tempo.candidates.events
    )
    assert all(
        event.chart_sample == event.acoustic_sample
        for event in without_tempo.candidates.events
    )
    split = next(
        event
        for event in with_tempo.candidates.events
        if event.policy == "long_vowel_pitch_split"
    )
    assert split.character == "ー"
    assert split.refined_sample == 1_210
    assert split.evidence.pitch == 0.8
    assert split.evidence.long_vowel_split == 1.0
    assert all(event.evidence.rap_policy == 1.0 for event in with_tempo.candidates.events)


@pytest.mark.parametrize(
    ("phoneme", "confidence"),
    (("s", 0.9), ("u", 0.19)),
)
def test_long_vowel_split_requires_voiced_vowel_and_hubert_confidence(
    tmp_path: Path,
    phoneme: str,
    confidence: float,
) -> None:
    result = _result()
    assert result.hierarchy is not None
    split_phone = result.hierarchy.phonemes[3]
    split_phone.phoneme = phoneme
    split_phone.confidence = confidence

    artifacts = build_hubert_artifacts(_context(tmp_path), result)

    assert not any(
        event.policy == "long_vowel_pitch_split"
        for event in artifacts.candidates.events
    )


def test_hierarchy_requires_nonempty_reciprocal_observed_relations(
    tmp_path: Path,
) -> None:
    result = _result()
    assert result.hierarchy is not None
    payload = result.hierarchy.model_dump(mode="python")
    payload["characters"][0]["phoneme_indices"] = []

    with pytest.raises(
        ValueError,
        match="relations must be reciprocal|every character must map",
    ):
        AlignmentHierarchy.model_validate(payload)

    payload = result.hierarchy.model_dump(mode="python")
    payload["phonemes"][0]["observed_token_index"] = None
    with pytest.raises(ValueError, match="observed CTC token index"):
        AlignmentHierarchy.model_validate(payload)

    # Defence in depth: even a post-validation in-memory mutation cannot turn
    # a text span without an observed CTC token into a CandidateEvent.
    result.hierarchy.phonemes[0].observed_token_index = None
    artifacts = build_hubert_artifacts(_context(tmp_path), result)
    assert not any(event.character == "遠" for event in artifacts.candidates.events)


def test_hubert_report_has_real_runtime_hierarchy_coverage_and_qwen_delta(
    tmp_path: Path,
) -> None:
    result = _result()
    qwen = AlignmentResult(
        run_id="qwen-run",
        track_id="track-hubert",
        method="qwen",
        status="completed",
        sample_rate=8_000,
        sample_count=16_000,
        tokens=[
            AlignmentToken(
                id="qwen-token",
                text="遠",
                start_sample=90,
                end_sample=190,
                confidence=0.8,
                method="qwen",
            )
        ],
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    artifacts = build_hubert_artifacts(
        _context(tmp_path),
        result,
        qwen_result=qwen,
        qwen_report=_qwen_report(),
    )

    assert artifacts.report.hubert.character_coverage == 1.0
    assert artifacts.report.hubert.mora_coverage == 1.0
    assert artifacts.report.hubert.phoneme_coverage == 1.0
    assert artifacts.report.hubert.forced_character_coverage == 1.0
    assert artifacts.report.hubert.coverage_confidence_threshold == 0.2
    assert artifacts.report.hubert.runtime_sec == 5.25
    assert artifacts.report.hubert.runtime_source == "ctc_metadata"
    assert artifacts.report.qwen_coverage == pytest.approx(1 / 8)
    assert artifacts.report.qwen_proxy_coverage == 0.625
    assert artifacts.report.coverage_delta == pytest.approx(7 / 8)
    assert artifacts.report.details["canonicalLyricUnitCount"] == 8
    assert artifacts.report.details["canonicalLyricUnitSource"] == (
        "hubert_typed_character_hierarchy"
    )
    assert artifacts.report.run_ids == {"hubert": "hubert-run", "qwen": "qwen-run"}

    published = publish_hubert_artifacts(_context(tmp_path), artifacts, persist=False)
    assert published == 0
    payload = json.loads(
        (tmp_path / "reports" / "hubert-alignment-report.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["hubert"]["characterCoverage"] == 1.0
    assert payload["hubert"]["forcedCharacterCoverage"] == 1.0
    assert payload["hubert"]["runtimeSec"] == 5.25
    assert payload["qwenCoverage"] == pytest.approx(1 / 8)
    assert payload["qwenProxyCoverage"] == 0.625
    assert payload["coverageDelta"] == pytest.approx(7 / 8)
    candidate_payload = json.loads(
        (
            tmp_path
            / "storage"
            / "alignment"
            / "track-hubert"
            / "ctc"
            / "hubert-run.candidate-events.json"
        ).read_text(encoding="utf-8")
    )
    assert candidate_payload["events"][0]["evidence"]["spectralChange"] == 0.6


def test_v06_run_rejects_missing_hierarchy_instead_of_treating_phones_as_characters(
    tmp_path: Path,
) -> None:
    plain = AlignmentResult(
        run_id="broken-v06",
        track_id="track-hubert",
        method="ctc",
        status="completed",
        sample_rate=8_000,
        sample_count=16_000,
        tokens=[
            AlignmentToken(
                id="phone-token",
                text="遠くへ",
                phoneme="m",
                start_sample=100,
                end_sample=180,
                confidence=0.9,
                method="ctc",
            )
        ],
        metadata={"engineVersion": "0.6", "totalElapsedSec": 1.0},
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    with pytest.raises(ValueError, match="require a typed non-empty hierarchy"):
        build_hubert_artifacts(_context(tmp_path), plain)


def test_runner_publishes_and_loads_hubert_outputs(
    client: TestClient,
    tmp_path: Path,
) -> None:
    assert client.get("/api/health").status_code == 200
    context = _context(tmp_path)
    with SessionLocal() as session:
        project = ProjectModel(
            id="runner-project", title="Synthetic Vocal Demo", artist="Demo Artist"
        )
        project.track = TrackModel(
            id=context.track_id,
            original_file_name="runner.wav",
            stored_file_name="runner.wav",
            file_path="audio/runner.wav",
            format="wav",
            original_sample_rate=context.sample_rate,
            channels=1,
            sample_count=context.sample_count,
            duration_sec=context.sample_count / context.sample_rate,
        )
        session.add(project)
        session.commit()
    adapters = {
        "qwen": StubAdapter("qwen"),
        "mfa": StubAdapter("mfa"),
        "ctc": StubAdapter("ctc"),
        "singing": StubAdapter("singing"),
        "hybrid": HybridAlignmentAdapter(),
    }
    runner = AlignmentRunner(
        context.storage_dir,
        context.project_root,
        adapters=adapters,  # type: ignore[arg-type]
        evaluator=StubEvaluator(),  # type: ignore[arg-type]
    )
    try:
        runner.submit(context, "ctc")
        deadline = time.monotonic() + 5.0
        completed: AlignmentResult | None = None
        while time.monotonic() < deadline:
            completed = runner.get_result(context.track_id, "ctc")
            if completed is not None and completed.status in {"completed", "failed"}:
                break
            time.sleep(0.01)
        assert completed is not None and completed.status == "completed"
        assert completed.hierarchy is not None
        assert completed.metadata["hubertCandidateEventCount"] == 9
        assert completed.metadata["hubertPersistedCandidateCount"] == 9
        candidates = runner.get_hubert_candidates(context.track_id)
        report = runner.get_hubert_report(context.track_id)
        assert candidates is not None and len(candidates.events) == 9
        assert report is not None and report.hubert.runtime_sec == 5.25
        assert (tmp_path / "reports" / "hubert-alignment-report.json").is_file()
    finally:
        runner._executor.shutdown(wait=True)  # noqa: SLF001


def test_runner_retains_alignment_when_hubert_publication_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context(tmp_path)
    completed = _result()
    runner = AlignmentRunner(context.storage_dir, context.project_root)

    def fail_publish(*_args: object, **_kwargs: object) -> int:
        raise OSError("disk full")

    monkeypatch.setattr(
        "beatforge_api.audio.alignment.runner.publish_hubert_artifacts",
        fail_publish,
    )
    try:
        retained = runner._publish_hubert_outputs(context, completed)  # noqa: SLF001
    finally:
        runner._executor.shutdown(wait=True)  # noqa: SLF001
    assert retained.status == "completed"
    assert retained.hierarchy == completed.hierarchy
    assert retained.tokens == completed.tokens
    assert retained.metadata["hubertPostprocessError"]["message"] == "disk full"
    assert any("publishing failed" in warning for warning in retained.warnings)


def _legacy_candidate(candidate_id: str) -> CandidateEventModel:
    return CandidateEventModel(
        id=candidate_id,
        sample=50,
        acoustic_sample=50,
        chart_sample=50,
        snap_error_ms=0.0,
        lane="mix",
        source_evidence_json=dumps({"mix": 1.0}),
        semantic_evidence_json=dumps({}),
        confidence=0.5,
        status="uncertain",
        grid_type="unsnapped",
        grid_confidence=0.0,
        source="mix",
        generator="analysis",
        aligned_sample=50,
        refined_sample=50,
    )


def test_persistence_replaces_only_hubert_generator_and_serializes_provenance(
    client: TestClient,
    tmp_path: Path,
) -> None:
    assert client.get("/api/health").status_code == 200
    artifacts = build_hubert_artifacts(_context(tmp_path), _result())
    with SessionLocal() as session:
        project = ProjectModel(
            id="project-hubert", title="Synthetic Vocal Demo", artist="Demo Artist"
        )
        track = TrackModel(
            id="track-hubert",
            project=project,
            original_file_name="sample.wav",
            stored_file_name="sample.wav",
            file_path="audio/sample.wav",
            format="wav",
            original_sample_rate=8_000,
            channels=1,
            sample_count=16_000,
            duration_sec=2.0,
        )
        track.candidate_events.append(_legacy_candidate("analysis-candidate"))
        non_vocal_hubert = _legacy_candidate("other-source-hubert-candidate")
        non_vocal_hubert.generator = "hubert_ctc"
        non_vocal_hubert.source = "other"
        track.candidate_events.append(non_vocal_hubert)
        session.add(project)
        session.commit()

        assert persist_hubert_candidates(artifacts.candidates, session=session) == 9
        rows = list(
            session.scalars(
                select(CandidateEventModel).where(
                    CandidateEventModel.track_id == "track-hubert"
                )
            )
        )
        assert len(rows) == 11
        assert sum(
            row.generator == "hubert_ctc" and row.source == "vocals" for row in rows
        ) == 9
        assert any(row.id == "analysis-candidate" for row in rows)
        assert any(row.id == "other-source-hubert-candidate" for row in rows)
        hubert = next(
            row
            for row in rows
            if row.generator == "hubert_ctc" and row.source == "vocals"
        )
        serialized = candidate_event_dict(hubert, 8_000)
        assert serialized["source"] == "vocals"
        assert serialized["generator"] == "hubert_ctc"
        assert serialized["acoustic_sample"] == serialized["refined_sample"]
        assert serialized["aligned_sample"] >= 0
        assert "spectralChange" in serialized["evidence"]
        assert serialized["event_level"] == "character"
        assert serialized["event_policy"] in {
            "character",
            "long_vowel_pitch_split",
        }
        assert serialized["alignment_unit_id"]
        assert serialized["alignment_run_id"] == "hubert-run"
        assert serialized["character_indices"]
        assert serialized["phonemes"]

        rerun = artifacts.candidates.model_copy(update={"run_id": "hubert-run-2"})
        assert persist_hubert_candidates(rerun, session=session) == 9
        assert hubert.alignment_run_id == "hubert-run-2"

        reduced = HubertCandidateBundle(
            **artifacts.candidates.model_dump(exclude={"events"}),
            events=artifacts.candidates.events[:1],
        )
        assert persist_hubert_candidates(reduced, session=session) == 1
        remaining = list(
            session.scalars(
                select(CandidateEventModel).where(
                    CandidateEventModel.track_id == "track-hubert"
                )
            )
        )
        assert len(remaining) == 3
        assert any(row.id == "analysis-candidate" for row in remaining)
        assert any(row.id == "other-source-hubert-candidate" for row in remaining)
        session.delete(
            next(row for row in remaining if row.id == "other-source-hubert-candidate")
        )
        session.commit()

    tempo = client.patch(
        "/api/tracks/track-hubert/tempo-map",
        json={
            "tempoMap": [
                {
                    "startSample": 0,
                    "bpm": 120.0,
                    "beatOffsetSample": 0,
                    "timeSignatureNumerator": 4,
                    "timeSignatureDenominator": 4,
                }
            ]
        },
    )
    assert tempo.status_code == 200, tempo.text
    candidates = client.get("/api/tracks/track-hubert/candidate-events").json()
    assert len(candidates) == 2  # tempo policy did not create a grid event
    hubert_payload = next(item for item in candidates if item["generator"] == "hubert_ctc")
    assert hubert_payload["acousticSample"] == hubert_payload["refinedSample"]
    assert hubert_payload["evidence"]["rhythm"] == pytest.approx(
        hubert_payload["gridConfidence"]
    )


def test_hubert_candidate_and_report_api_are_typed(
    client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert client.get("/api/health").status_code == 200
    with SessionLocal() as session:
        project = ProjectModel(
            id="project-hubert", title="Synthetic Vocal Demo", artist="Demo Artist"
        )
        project.track = TrackModel(
            id="track-hubert",
            original_file_name="sample.wav",
            stored_file_name="api-sample.wav",
            file_path="audio/api-sample.wav",
            format="wav",
            original_sample_rate=8_000,
            channels=1,
            sample_count=16_000,
            duration_sec=2.0,
        )
        session.add(project)
        session.commit()

    artifacts = build_hubert_artifacts(_context(tmp_path), _result())
    stub = ArtifactRunnerStub(artifacts.candidates, artifacts.report)
    monkeypatch.setattr(routes, "_alignment_runner", stub)

    candidates = client.get(
        "/api/tracks/track-hubert/alignment/ctc/candidate-events"
    )
    assert candidates.status_code == 200, candidates.text
    candidate_payload = candidates.json()
    assert candidate_payload["runId"] == "hubert-run"
    assert candidate_payload["events"][0]["source"] == "vocals"
    assert candidate_payload["events"][0]["generator"] == "hubert_ctc"
    assert candidate_payload["events"][0]["acousticSample"] == 110
    assert candidate_payload["events"][0]["evidence"]["hubert"] == 0.9

    report = client.get("/api/tracks/track-hubert/alignment/ctc/hubert-report")
    assert report.status_code == 200, report.text
    assert report.json()["hubert"]["runtimeSec"] == 5.25

    monkeypatch.setattr(routes, "_alignment_runner", ArtifactRunnerStub(None, None))
    missing = client.get(
        "/api/tracks/track-hubert/alignment/ctc/candidate-events"
    )
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "HUBERT_CANDIDATES_NOT_FOUND"
    missing_report = client.get(
        "/api/tracks/track-hubert/alignment/ctc/hubert-report"
    )
    assert missing_report.status_code == 404
    assert missing_report.json()["error"]["code"] == "HUBERT_REPORT_NOT_FOUND"


def test_candidate_event_schema_columns_are_additive(client: TestClient) -> None:
    assert client.get("/api/health").status_code == 200
    columns = {column["name"] for column in inspect(engine).get_columns("candidate_events")}
    assert {
        "source",
        "generator",
        "character",
        "mora",
        "phoneme",
        "event_level",
        "event_policy",
        "alignment_unit_id",
        "alignment_unit_index",
        "alignment_run_id",
        "character_indices_json",
        "phonemes_json",
        "aligned_sample",
        "refined_sample",
        "evidence_json",
    } <= columns
