from __future__ import annotations

import json

from beatforge_api.jobs import _candidate_event_from_payload


def test_persisted_candidate_follows_final_hit_chart_position_after_tempo_policy() -> None:
    candidate = _candidate_event_from_payload(
        {
            "id": "candidate-1",
            "acoustic_sample": 1_000,
            "chart_sample": 1_100,
            "snap_error_ms": -10.0,
            "lane": "vocals",
            "source_evidence": {"vocals": 0.9},
            "semantic_evidence": {"lyricAlignment": 0.7, "beatConfidence": 0.9},
            "confidence": 0.8,
            "status": "accepted",
            "grid_type": "straight_1_16",
        },
        {
            "id": "hit-1",
            "acoustic_sample": 1_000,
            "chart_sample": 1_250,
            "snap_error_ms": -25.0,
        },
    )

    assert candidate.hit_point_id == "hit-1"
    assert candidate.acoustic_sample == 1_000
    assert candidate.chart_sample == 1_250
    assert candidate.snap_error_ms == -25.0
    semantic = json.loads(candidate.semantic_evidence_json)
    assert semantic["lyricAlignment"] == 0.7
    assert 0 < semantic["beatConfidence"] < 1
