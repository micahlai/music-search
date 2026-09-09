"""Database-free tests for repository validation and pgvector query construction."""

from __future__ import annotations

from typing import Any, cast

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session, sessionmaker

from music_search.db.repository import (
    MusicRepository,
    SegmentWrite,
    TrackWrite,
    _nearest_segment_statement,
)


def _track(*, embedding_size: int = 512) -> TrackWrite:
    return TrackWrite(
        source_uri="file:///authorized/example.wav",
        content_sha256="A" * 64,
        title="example",
        duration_seconds=12.0,
        sample_rate=48_000,
        channels=2,
        bpm=120.0,
        embedding_model="laion/clap-htsat-unfused",
        analysis_version="1",
        segments=(
            SegmentWrite(
                segment_index=0,
                start_seconds=0.0,
                end_seconds=10.0,
                rms_energy=0.25,
                bpm=120.0,
                embedding=[1.0] + [0.0] * (embedding_size - 1),
            ),
        ),
    )


class _DatabaseMustNotBeReached:
    def begin(self) -> Any:
        raise AssertionError("validation should fail before a database transaction starts")


def test_replace_rejects_wrong_embedding_dimension_before_database_access() -> None:
    repository = MusicRepository(cast(sessionmaker[Session], _DatabaseMustNotBeReached()))

    with pytest.raises(ValueError, match="must have 512 values"):
        repository.replace_track(_track(embedding_size=511))


def test_replace_rejects_duplicate_segment_indexes_before_database_access() -> None:
    original = _track().segments[0]
    invalid = TrackWrite(
        **{
            **{field: getattr(_track(), field) for field in _track().__dataclass_fields__},
            "segments": (original, original),
        }
    )
    repository = MusicRepository(cast(sessionmaker[Session], _DatabaseMustNotBeReached()))

    with pytest.raises(ValueError, match="duplicate segment_index 0"):
        repository.replace_track(invalid)


def test_valid_replace_reaches_transaction_after_building_upsert() -> None:
    repository = MusicRepository(cast(sessionmaker[Session], _DatabaseMustNotBeReached()))

    with pytest.raises(AssertionError, match="database transaction"):
        repository.replace_track(_track())


def test_nearest_segment_statement_uses_pgvector_cosine_operator() -> None:
    statement = _nearest_segment_statement([0.0] * 512, limit=7)
    compiled = str(statement.compile(dialect=postgresql.dialect()))

    assert "<=>" in compiled
    assert "ORDER BY cosine_distance ASC" in compiled
    assert "LIMIT" in compiled


def test_search_filters_model_identity() -> None:
    statement = _nearest_segment_statement([1.0] + [0.0] * 511, limit=7, embedding_model="model-b")
    compiled = statement.compile(dialect=postgresql.dialect())
    assert "tracks.embedding_model =" in str(compiled)
    assert "model-b" in compiled.params.values()


def test_zero_vector_is_rejected_before_querying() -> None:
    repository = MusicRepository(cast(sessionmaker[Session], _DatabaseMustNotBeReached()))
    with pytest.raises(ValueError, match="nonzero"):
        repository.search_nearest_segments([0.0] * 512)
