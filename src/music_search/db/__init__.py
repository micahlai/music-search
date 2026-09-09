"""PostgreSQL persistence primitives for analyzed music tracks."""

from music_search.db.models import Base, Segment, Track
from music_search.db.repository import (
    MusicRepository,
    SegmentMatch,
    SegmentRecord,
    SegmentWrite,
    TrackRecord,
    TrackWrite,
)
from music_search.db.session import (
    DEFAULT_DATABASE_URL,
    create_engine_from_url,
    create_session_factory,
    resolve_database_url,
    session_scope,
)

__all__ = [
    "DEFAULT_DATABASE_URL",
    "Base",
    "MusicRepository",
    "Segment",
    "SegmentMatch",
    "SegmentRecord",
    "SegmentWrite",
    "Track",
    "TrackRecord",
    "TrackWrite",
    "create_engine_from_url",
    "create_session_factory",
    "resolve_database_url",
    "session_scope",
]
