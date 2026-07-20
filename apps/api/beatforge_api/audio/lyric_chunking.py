"""Monotonic matching between singing-ASR chunks and known lyric lines.

Qwen's forced aligner is intended for speech and becomes unstable when a whole
song is submitted at once.  This module keeps the song-level orchestration pure
and testable: singing ASR first identifies which lyric lines belong to each
short chunk, then only those lines are passed to the forced aligner.
"""

from __future__ import annotations

import difflib
import math
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LyricChunkAssignment:
    chunk_index: int
    start_line: int
    end_line: int
    text: str
    transcript: str
    similarity: float
    confidence: float


def normalize_lyric_text(text: str) -> str:
    """Normalize Japanese, Latin and numeric lyric text for fuzzy matching."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return "".join(character for character in normalized if character.isalnum())


def _match_quality(transcript: str, candidate: str) -> tuple[float, float, float]:
    observed = normalize_lyric_text(transcript)
    expected = normalize_lyric_text(candidate)
    if not observed or not expected:
        return 0.0, 0.0, float("inf")
    matcher = difflib.SequenceMatcher(None, observed, expected, autojunk=False)
    similarity = float(matcher.ratio())
    matched = sum(block.size for block in matcher.get_matching_blocks())
    overlap = matched / max(1, min(len(observed), len(expected)))
    length_penalty = abs(math.log((len(expected) + 1) / (len(observed) + 1)))
    return similarity, float(overlap), float(length_penalty)


def assign_lyric_lines(
    transcripts: list[str],
    lyric_lines: list[str],
    *,
    max_lines_per_chunk: int = 10,
    minimum_similarity: float = 0.20,
) -> tuple[list[LyricChunkAssignment], list[int]]:
    """Assign contiguous lyric ranges to chunks with a monotonic Viterbi pass.

    Both chunks and lyric lines may be skipped.  A hard similarity floor is
    important: it is safer to leave an instrumental or unrecognized passage
    without chart points than to force unrelated lyrics into silence.
    """

    lines = [line.strip() for line in lyric_lines if line.strip()]
    chunk_count = len(transcripts)
    line_count = len(lines)
    if chunk_count == 0 or line_count == 0:
        return [], list(range(line_count))
    if max_lines_per_chunk <= 0:
        raise ValueError("max_lines_per_chunk must be positive")

    negative_infinity = float("-inf")
    scores = [negative_infinity] * (line_count + 1)
    scores[0] = 0.0
    # Backpointers are indexed by (processed chunk count, consumed line count).
    parents: dict[tuple[int, int], tuple[int, int, str, tuple[int, int, float, float] | None]] = {}

    for chunk_index, transcript in enumerate(transcripts):
        # Allow lyric lines to be skipped before processing this chunk.  This
        # relaxation is monotonic because skipped lines only increase the index.
        relaxed = scores[:]
        skip_parents: dict[int, tuple[int, int, str, tuple[int, int, float, float] | None]] = {}
        for consumed in range(line_count):
            candidate = relaxed[consumed] - 0.065
            if candidate > relaxed[consumed + 1]:
                relaxed[consumed + 1] = candidate
                skip_parents[consumed + 1] = (
                    chunk_index,
                    consumed,
                    "skip_line",
                    None,
                )

        next_scores = [negative_infinity] * (line_count + 1)
        next_parents: dict[int, tuple[int, int, str, tuple[int, int, float, float] | None]] = {}
        for consumed, base_score in enumerate(relaxed):
            if not math.isfinite(base_score):
                continue
            # A chunk with no trustworthy transcript remains unassigned.
            if base_score > next_scores[consumed]:
                next_scores[consumed] = base_score
                next_parents[consumed] = (
                    chunk_index,
                    consumed,
                    "skip_chunk",
                    None,
                )
            for line_total in range(1, min(max_lines_per_chunk, line_count - consumed) + 1):
                end = consumed + line_total
                candidate_text = "\n".join(lines[consumed:end])
                similarity, overlap, length_penalty = _match_quality(
                    transcript,
                    candidate_text,
                )
                if similarity < minimum_similarity:
                    continue
                relative_midpoint = ((consumed + end) / 2.0) / line_count
                chunk_midpoint = (chunk_index + 0.5) / chunk_count
                position_penalty = abs(relative_midpoint - chunk_midpoint)
                match_score = (
                    2.45 * similarity
                    + 0.55 * overlap
                    - 0.38 * length_penalty
                    - 0.34 * position_penalty
                    - 0.20
                    - 0.018 * (line_total - 1)
                )
                total_score = base_score + match_score
                if total_score <= next_scores[end]:
                    continue
                confidence = max(
                    0.0,
                    min(
                        1.0,
                        0.68 * similarity
                        + 0.32 * overlap
                        - 0.16 * min(length_penalty, 1.5),
                    ),
                )
                next_scores[end] = total_score
                next_parents[end] = (
                    chunk_index,
                    consumed,
                    "assign",
                    (consumed, end, similarity, confidence),
                )

        # Materialize skip-line chains as same-layer parents so reconstruction
        # can cross them before stepping into the previous chunk.
        for consumed, parent in skip_parents.items():
            parents[(chunk_index, consumed)] = parent
        for consumed, parent in next_parents.items():
            parents[(chunk_index + 1, consumed)] = parent
        scores = next_scores

    # Skipping unmatched trailing lyric lines is allowed and explicitly reported.
    final_scores = scores[:]
    for consumed in range(line_count):
        candidate = final_scores[consumed] - 0.065
        if candidate > final_scores[consumed + 1]:
            final_scores[consumed + 1] = candidate
            parents[(chunk_count, consumed + 1)] = (
                chunk_count,
                consumed,
                "skip_line",
                None,
            )

    state = (chunk_count, line_count)
    assignments: list[LyricChunkAssignment] = []
    assigned_lines: set[int] = set()
    safety = (chunk_count + 1) * (line_count + 1) + 1
    while state != (0, 0) and safety > 0:
        safety -= 1
        parent = parents.get(state)
        if parent is None:
            break
        previous_chunk, previous_line, action, detail = parent
        if action == "assign" and detail is not None:
            start, end, similarity, confidence = detail
            assignments.append(
                LyricChunkAssignment(
                    chunk_index=state[0] - 1,
                    start_line=start,
                    end_line=end,
                    text="\n".join(lines[start:end]),
                    transcript=transcripts[state[0] - 1],
                    similarity=similarity,
                    confidence=confidence,
                )
            )
            assigned_lines.update(range(start, end))
        state = (previous_chunk, previous_line)

    assignments.reverse()
    unassigned = [index for index in range(line_count) if index not in assigned_lines]
    return assignments, unassigned
