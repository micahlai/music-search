"""SQLAlchemy models for tracks and fixed-size audio segments."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PostgreSQLUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base shared by the runtime models and Alembic."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Track(Base):
    """One authorized local audio file and its analysis provenance."""

    __tablename__ = "tracks"
    __table_args__ = (
        CheckConstraint(
            "content_sha256 ~ '^[0-9a-f]{64}$'",
            name="content_sha256_lower_hex",
        ),
        CheckConstraint("duration_seconds > 0", name="duration_positive"),
        CheckConstraint("sample_rate > 0", name="sample_rate_positive"),
        CheckConstraint("channels > 0", name="channels_positive"),
        CheckConstraint("bpm IS NULL OR bpm > 0", name="bpm_positive"),
        Index("ix_tracks_source_uri", "source_uri"),
        Index("ix_tracks_title", "title"),
    )

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=text("gen_random_uuid()"),
    )
    source_uri: Mapped[str] = mapped_column(Text, nullable=False)
    content_sha256: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    artist: Mapped[str | None] = mapped_column(Text)
    album: Mapped[str | None] = mapped_column(Text)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    sample_rate: Mapped[int] = mapped_column(Integer, nullable=False)
    channels: Mapped[int] = mapped_column(Integer, nullable=False)
    bpm: Mapped[float | None] = mapped_column(Float)
    embedding_model: Mapped[str] = mapped_column(Text, nullable=False)
    analysis_version: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
        server_default=text("'{}'::jsonb"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    segments: Mapped[list[Segment]] = relationship(
        back_populates="track",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Segment.segment_index",
    )


class Segment(Base):
    """A time-bounded track window with DSP features and a CLAP vector."""

    __tablename__ = "segments"
    __table_args__ = (
        UniqueConstraint("track_id", "segment_index", name="uq_segments_track_segment_index"),
        CheckConstraint("segment_index >= 0", name="segment_index_nonnegative"),
        CheckConstraint("start_seconds >= 0", name="start_nonnegative"),
        CheckConstraint("end_seconds > start_seconds", name="end_after_start"),
        CheckConstraint("rms_energy >= 0", name="rms_nonnegative"),
        CheckConstraint("bpm IS NULL OR bpm > 0", name="bpm_positive"),
        Index("ix_segments_track_start", "track_id", "start_seconds"),
        Index(
            "ix_segments_embedding_hnsw_cosine",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
            postgresql_with={"m": 16, "ef_construction": 64},
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
        server_default=text("gen_random_uuid()"),
    )
    track_id: Mapped[UUID] = mapped_column(
        PostgreSQLUUID(as_uuid=True),
        ForeignKey("tracks.id", ondelete="CASCADE"),
        nullable=False,
    )
    segment_index: Mapped[int] = mapped_column(Integer, nullable=False)
    start_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    end_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    rms_energy: Mapped[float] = mapped_column(Float, nullable=False)
    bpm: Mapped[float | None] = mapped_column(Float)
    embedding: Mapped[list[float]] = mapped_column(VECTOR(512), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    track: Mapped[Track] = relationship(back_populates="segments")
