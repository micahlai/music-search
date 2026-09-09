"""Natural-language CLAP retrieval and transparent track ranking."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any, Protocol
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from music_search.db import SegmentMatch
from music_search.query_parser import parse_query


class TextEmbedder(Protocol):
    model_name: str

    def embed_text(self, texts: str | Sequence[str]) -> NDArray[np.float32]: ...


class SegmentRepository(Protocol):
    def search_nearest_segments(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int = 20,
        embedding_model: str | None = None,
    ) -> list[SegmentMatch]: ...


@dataclass(frozen=True, slots=True)
class RankedTrack:
    """One unique track ranked by its strongest retrieved segment."""

    track_id: UUID
    title: str
    artist: str | None
    album: str | None
    source_uri: str
    score: float
    best_start_seconds: float
    best_end_seconds: float
    rms_energy: float
    estimated_bpm: float | None
    bpm_source: str | None
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["track_id"] = str(self.track_id)
        return value


def _format_timestamp(seconds: float) -> str:
    whole_seconds = max(0, round(seconds))
    minutes, remainder = divmod(whole_seconds, 60)
    return f"{minutes}:{remainder:02d}"


def _explain(match: SegmentMatch) -> str:
    evidence = [
        "Best CLAP semantic match at "
        f"{_format_timestamp(match.start_seconds)}-{_format_timestamp(match.end_seconds)}",
        f"RMS {match.rms_energy:.3f}",
    ]
    bpm = match.segment_bpm or match.track_bpm
    if bpm is not None and bpm > 0:
        source = "segment" if match.segment_bpm else "track"
        evidence.append(f"estimated {bpm:.0f} BPM ({source})")
    return "; ".join(evidence)


class SearchService:
    """Embed a prompt, retrieve segments, and collapse them into unique tracks."""

    def __init__(
        self,
        repository: SegmentRepository,
        embedder: TextEmbedder,
        *,
        segment_overfetch: int = 10,
        minimum_segment_candidates: int = 50,
    ) -> None:
        if segment_overfetch < 1 or minimum_segment_candidates < 1:
            raise ValueError("candidate controls must be positive")
        self.repository = repository
        self.embedder = embedder
        self.segment_overfetch = segment_overfetch
        self.minimum_segment_candidates = minimum_segment_candidates

    def search(self, prompt: str, *, limit: int = 10) -> list[RankedTrack]:
        if limit < 1:
            raise ValueError("limit must be at least 1")

        query = parse_query(prompt)
        vector = np.asarray(self.embedder.embed_text(query.semantic_text), dtype=np.float32)
        if vector.ndim != 1:
            raise RuntimeError(f"text embedder returned an unexpected shape: {vector.shape}")

        candidate_limit = max(self.minimum_segment_candidates, limit * self.segment_overfetch)
        matches = self.repository.search_nearest_segments(
            vector.tolist(), limit=candidate_limit, embedding_model=self.embedder.model_name
        )

        best_by_track: dict[UUID, SegmentMatch] = {}
        for match in matches:
            current = best_by_track.get(match.track_id)
            if current is None or (
                -match.cosine_similarity,
                match.start_seconds,
                str(match.segment_id),
            ) < (
                -current.cosine_similarity,
                current.start_seconds,
                str(current.segment_id),
            ):
                best_by_track[match.track_id] = match

        ranked_matches = sorted(
            best_by_track.values(),
            key=lambda match: (
                -match.cosine_similarity,
                match.title.casefold(),
                str(match.track_id),
            ),
        )[:limit]

        return [
            RankedTrack(
                track_id=match.track_id,
                title=match.title,
                artist=match.artist,
                album=match.album,
                source_uri=match.source_uri,
                score=match.cosine_similarity,
                best_start_seconds=match.start_seconds,
                best_end_seconds=match.end_seconds,
                rms_energy=match.rms_energy,
                estimated_bpm=match.segment_bpm or match.track_bpm,
                bpm_source=(
                    "segment" if match.segment_bpm else "track" if match.track_bpm else None
                ),
                explanation=_explain(match),
            )
            for match in ranked_matches
        ]
