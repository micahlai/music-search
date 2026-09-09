"""Typed, transaction-safe persistence operations for ingestion and retrieval."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, cast
from uuid import UUID, uuid4

from sqlalchemy import Select, Table, delete, func, select
from sqlalchemy.dialects.postgresql import insert as postgresql_insert
from sqlalchemy.orm import Session, selectinload, sessionmaker

from music_search.db.models import Segment, Track

EMBEDDING_DIMENSIONS = 512
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_TRACK_TABLE = cast(Table, Track.__table__)
_SEGMENT_TABLE = cast(Table, Segment.__table__)


@dataclass(frozen=True, slots=True)
class SegmentWrite:
    """Analyzed segment data accepted by the ingestion write path."""

    segment_index: int
    start_seconds: float
    end_seconds: float
    rms_energy: float
    bpm: float | None
    embedding: Sequence[float]


@dataclass(frozen=True, slots=True)
class TrackWrite:
    """A complete track analysis; writes always replace its segment set atomically."""

    source_uri: str
    content_sha256: str
    title: str
    duration_seconds: float
    sample_rate: int
    channels: int
    bpm: float | None
    embedding_model: str
    analysis_version: str
    segments: Sequence[SegmentWrite]
    artist: str | None = None
    album: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SegmentRecord:
    id: UUID
    track_id: UUID
    segment_index: int
    start_seconds: float
    end_seconds: float
    rms_energy: float
    bpm: float | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class TrackRecord:
    id: UUID
    source_uri: str
    content_sha256: str
    title: str
    artist: str | None
    album: str | None
    duration_seconds: float
    sample_rate: int
    channels: int
    bpm: float | None
    embedding_model: str
    analysis_version: str
    metadata: Mapping[str, Any]
    created_at: datetime
    updated_at: datetime
    segments: tuple[SegmentRecord, ...]


@dataclass(frozen=True, slots=True)
class SegmentMatch:
    """One nearest segment plus enough parent-track context for CLI rendering."""

    segment_id: UUID
    segment_index: int
    start_seconds: float
    end_seconds: float
    rms_energy: float
    segment_bpm: float | None
    track_id: UUID
    source_uri: str
    content_sha256: str
    title: str
    artist: str | None
    album: str | None
    track_bpm: float | None
    cosine_distance: float
    cosine_similarity: float


class MusicRepository:
    """Short-lived-session repository for ingest and vector retrieval operations."""

    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

    def find_track_by_content_hash(self, content_sha256: str) -> TrackRecord | None:
        """Return an existing track and its segments, keyed by normalized SHA-256."""

        normalized_hash = _normalize_sha256(content_sha256)
        statement = (
            select(Track)
            .where(Track.content_sha256 == normalized_hash)
            .options(selectinload(Track.segments))
        )
        with self._session_factory() as session:
            track = session.scalar(statement)
            return _track_record(track) if track is not None else None

    def replace_track(self, track: TrackWrite) -> TrackRecord:
        """Upsert track metadata and replace all segments in one database transaction."""

        normalized_hash, normalized_segments = _validate_track_write(track)
        track_values = {
            "id": uuid4(),
            "source_uri": track.source_uri,
            "content_sha256": normalized_hash,
            "title": track.title,
            "artist": track.artist,
            "album": track.album,
            "duration_seconds": float(track.duration_seconds),
            "sample_rate": track.sample_rate,
            "channels": track.channels,
            "bpm": float(track.bpm) if track.bpm is not None else None,
            "embedding_model": track.embedding_model,
            "analysis_version": track.analysis_version,
            "metadata": dict(track.metadata),
        }

        # Use the table form so the physical ``metadata`` column does not collide
        # with DeclarativeBase.metadata during ORM-enabled INSERT key resolution.
        insert_statement = postgresql_insert(_TRACK_TABLE).values(**track_values)
        update_columns = {
            key: getattr(insert_statement.excluded, key) for key in track_values if key != "id"
        }
        update_columns["updated_at"] = func.now()
        upsert_statement = insert_statement.on_conflict_do_update(
            index_elements=[_TRACK_TABLE.c.content_sha256],
            set_=update_columns,
        ).returning(_TRACK_TABLE.c.id)

        # sessionmaker.begin() owns commit/rollback, so no partially replaced segment
        # collection can escape if vector insertion or validation at PostgreSQL fails.
        with self._session_factory.begin() as session:
            track_id = session.scalar(upsert_statement)
            if track_id is None:  # pragma: no cover - defensive SQL contract guard
                raise RuntimeError("track upsert did not return an id")

            session.execute(delete(Segment).where(Segment.track_id == track_id))
            if normalized_segments:
                session.execute(
                    postgresql_insert(_SEGMENT_TABLE),
                    [
                        {
                            "id": uuid4(),
                            "track_id": track_id,
                            "segment_index": segment.segment_index,
                            "start_seconds": float(segment.start_seconds),
                            "end_seconds": float(segment.end_seconds),
                            "rms_energy": float(segment.rms_energy),
                            "bpm": float(segment.bpm) if segment.bpm is not None else None,
                            "embedding": embedding,
                        }
                        for segment, embedding in normalized_segments
                    ],
                )

            stored_track = session.scalar(
                select(Track)
                .where(Track.id == track_id)
                .options(selectinload(Track.segments))
                .execution_options(populate_existing=True)
            )
            if stored_track is None:  # pragma: no cover - same-transaction invariant
                raise RuntimeError(f"track {track_id} disappeared during replacement")
            return _track_record(stored_track)

    def search_nearest_segments(
        self,
        query_embedding: Sequence[float],
        *,
        limit: int = 20,
        embedding_model: str | None = None,
    ) -> list[SegmentMatch]:
        """Find nearest CLAP segment vectors using pgvector cosine distance."""

        vector = _normalize_embedding(query_embedding, field_name="query_embedding")
        if limit < 1:
            raise ValueError("limit must be at least 1")

        statement = _nearest_segment_statement(vector, limit=limit, embedding_model=embedding_model)
        with self._session_factory() as session:
            rows = session.execute(statement).mappings().all()

        matches: list[SegmentMatch] = []
        for row in rows:
            distance = float(row["cosine_distance"])
            matches.append(
                SegmentMatch(
                    segment_id=row["segment_id"],
                    segment_index=row["segment_index"],
                    start_seconds=row["start_seconds"],
                    end_seconds=row["end_seconds"],
                    rms_energy=row["rms_energy"],
                    segment_bpm=row["segment_bpm"],
                    track_id=row["track_id"],
                    source_uri=row["source_uri"],
                    content_sha256=row["content_sha256"],
                    title=row["title"],
                    artist=row["artist"],
                    album=row["album"],
                    track_bpm=row["track_bpm"],
                    cosine_distance=distance,
                    cosine_similarity=1.0 - distance,
                )
            )
        return matches


def _nearest_segment_statement(
    vector: list[float], *, limit: int, embedding_model: str | None = None
) -> Select[Any]:
    distance = Segment.embedding.cosine_distance(vector).label("cosine_distance")
    statement = (
        select(
            Segment.id.label("segment_id"),
            Segment.segment_index,
            Segment.start_seconds,
            Segment.end_seconds,
            Segment.rms_energy,
            Segment.bpm.label("segment_bpm"),
            Track.id.label("track_id"),
            Track.source_uri,
            Track.content_sha256,
            Track.title,
            Track.artist,
            Track.album,
            Track.bpm.label("track_bpm"),
            distance,
        )
        .join(Track, Track.id == Segment.track_id)
        # Keep the distance operator as the sole ascending sort key so PostgreSQL
        # can satisfy the top-k query with the HNSW cosine index.
        .order_by(distance.asc())
        .limit(limit)
    )
    if embedding_model is not None:
        statement = statement.where(Track.embedding_model == embedding_model)
    return statement


def _normalize_sha256(content_sha256: str) -> str:
    normalized = content_sha256.strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError("content_sha256 must be exactly 64 hexadecimal characters")
    return normalized


def _normalize_embedding(
    embedding: Sequence[float],
    *,
    field_name: str,
) -> list[float]:
    vector = [float(value) for value in embedding]
    if len(vector) != EMBEDDING_DIMENSIONS:
        raise ValueError(f"{field_name} must have {EMBEDDING_DIMENSIONS} values; got {len(vector)}")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{field_name} must contain only finite values")
    norm = math.hypot(*vector)
    if norm == 0 or not math.isfinite(norm):
        raise ValueError(f"{field_name} must have a finite nonzero norm")
    return [value / norm for value in vector]


def _require_finite_positive(value: float, *, field_name: str) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{field_name} must be finite and greater than zero")


def _validate_track_write(
    track: TrackWrite,
) -> tuple[str, list[tuple[SegmentWrite, list[float]]]]:
    normalized_hash = _normalize_sha256(track.content_sha256)
    if not track.source_uri.strip():
        raise ValueError("source_uri must not be empty")
    if not track.title.strip():
        raise ValueError("title must not be empty")
    if not track.embedding_model.strip():
        raise ValueError("embedding_model must not be empty")
    if not track.analysis_version.strip():
        raise ValueError("analysis_version must not be empty")
    _require_finite_positive(track.duration_seconds, field_name="duration_seconds")
    if track.sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero")
    if track.channels <= 0:
        raise ValueError("channels must be greater than zero")
    if track.bpm is not None:
        _require_finite_positive(track.bpm, field_name="bpm")
    if not track.segments:
        raise ValueError("segments must contain at least one analyzed window")

    indexes: set[int] = set()
    normalized_segments: list[tuple[SegmentWrite, list[float]]] = []
    for segment in track.segments:
        prefix = f"segments[{segment.segment_index}]"
        if segment.segment_index < 0:
            raise ValueError(f"{prefix}.segment_index must be nonnegative")
        if segment.segment_index in indexes:
            raise ValueError(f"duplicate segment_index {segment.segment_index}")
        indexes.add(segment.segment_index)
        if not math.isfinite(float(segment.start_seconds)) or segment.start_seconds < 0:
            raise ValueError(f"{prefix}.start_seconds must be finite and nonnegative")
        if (
            not math.isfinite(float(segment.end_seconds))
            or segment.end_seconds <= segment.start_seconds
        ):
            raise ValueError(f"{prefix}.end_seconds must be finite and after its start")
        if segment.end_seconds > track.duration_seconds + 1e-6:
            raise ValueError(f"{prefix}.end_seconds exceeds track duration")
        if not math.isfinite(float(segment.rms_energy)) or segment.rms_energy < 0:
            raise ValueError(f"{prefix}.rms_energy must be finite and nonnegative")
        if segment.bpm is not None:
            _require_finite_positive(segment.bpm, field_name=f"{prefix}.bpm")
        normalized_segments.append(
            (
                segment,
                _normalize_embedding(segment.embedding, field_name=f"{prefix}.embedding"),
            )
        )

    return normalized_hash, normalized_segments


def _track_record(track: Track) -> TrackRecord:
    return TrackRecord(
        id=track.id,
        source_uri=track.source_uri,
        content_sha256=track.content_sha256,
        title=track.title,
        artist=track.artist,
        album=track.album,
        duration_seconds=track.duration_seconds,
        sample_rate=track.sample_rate,
        channels=track.channels,
        bpm=track.bpm,
        embedding_model=track.embedding_model,
        analysis_version=track.analysis_version,
        metadata=dict(track.metadata_json),
        created_at=track.created_at,
        updated_at=track.updated_at,
        segments=tuple(
            SegmentRecord(
                id=segment.id,
                track_id=segment.track_id,
                segment_index=segment.segment_index,
                start_seconds=segment.start_seconds,
                end_seconds=segment.end_seconds,
                rms_energy=segment.rms_energy,
                bpm=segment.bpm,
                created_at=segment.created_at,
                updated_at=segment.updated_at,
            )
            for segment in track.segments
        ),
    )
