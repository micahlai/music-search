"""Create track and CLAP segment vector tables.

Revision ID: 0001_initial_schema
Revises: None
Create Date: 2026-09-09
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import VECTOR
from sqlalchemy.dialects import postgresql

revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Extension creation requires a database role allowed to install extensions.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "tracks",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("source_uri", sa.Text(), nullable=False),
        sa.Column("content_sha256", sa.String(length=64), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("artist", sa.Text(), nullable=True),
        sa.Column("album", sa.Text(), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=False),
        sa.Column("sample_rate", sa.Integer(), nullable=False),
        sa.Column("channels", sa.Integer(), nullable=False),
        sa.Column("bpm", sa.Float(), nullable=True),
        sa.Column("embedding_model", sa.Text(), nullable=False),
        sa.Column("analysis_version", sa.Text(), nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "content_sha256 ~ '^[0-9a-f]{64}$'",
            name="content_sha256_lower_hex",
        ),
        sa.CheckConstraint("duration_seconds > 0", name="duration_positive"),
        sa.CheckConstraint("sample_rate > 0", name="sample_rate_positive"),
        sa.CheckConstraint("channels > 0", name="channels_positive"),
        sa.CheckConstraint("bpm IS NULL OR bpm > 0", name="bpm_positive"),
        sa.PrimaryKeyConstraint("id", name="pk_tracks"),
        sa.UniqueConstraint("content_sha256", name="uq_tracks_content_sha256"),
    )
    op.create_index("ix_tracks_source_uri", "tracks", ["source_uri"], unique=False)
    op.create_index("ix_tracks_title", "tracks", ["title"], unique=False)

    op.create_table(
        "segments",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("track_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("segment_index", sa.Integer(), nullable=False),
        sa.Column("start_seconds", sa.Float(), nullable=False),
        sa.Column("end_seconds", sa.Float(), nullable=False),
        sa.Column("rms_energy", sa.Float(), nullable=False),
        sa.Column("bpm", sa.Float(), nullable=True),
        sa.Column("embedding", VECTOR(dim=512), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "segment_index >= 0",
            name="segment_index_nonnegative",
        ),
        sa.CheckConstraint("start_seconds >= 0", name="start_nonnegative"),
        sa.CheckConstraint("end_seconds > start_seconds", name="end_after_start"),
        sa.CheckConstraint("rms_energy >= 0", name="rms_nonnegative"),
        sa.CheckConstraint("bpm IS NULL OR bpm > 0", name="bpm_positive"),
        sa.ForeignKeyConstraint(
            ["track_id"],
            ["tracks.id"],
            name="fk_segments_track_id_tracks",
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name="pk_segments"),
        sa.UniqueConstraint(
            "track_id",
            "segment_index",
            name="uq_segments_track_segment_index",
        ),
    )
    op.create_index(
        "ix_segments_track_start",
        "segments",
        ["track_id", "start_seconds"],
        unique=False,
    )
    op.create_index(
        "ix_segments_embedding_hnsw_cosine",
        "segments",
        ["embedding"],
        unique=False,
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
        postgresql_with={"m": 16, "ef_construction": 64},
    )


def downgrade() -> None:
    op.drop_index("ix_segments_embedding_hnsw_cosine", table_name="segments")
    op.drop_index("ix_segments_track_start", table_name="segments")
    op.drop_table("segments")
    op.drop_index("ix_tracks_title", table_name="tracks")
    op.drop_index("ix_tracks_source_uri", table_name="tracks")
    op.drop_table("tracks")
    # Keep the database-wide vector extension: another schema may depend on it.
