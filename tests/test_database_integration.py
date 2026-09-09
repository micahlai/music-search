"""Opt-in live pgvector tests, isolated to a fresh schema per run.

MUSIC_SEARCH_TEST_DATABASE_URL must explicitly name a disposable test database.
The fixture creates/removes only its own random schema, retaining the vector extension.
"""

import os
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import delete, event, func, select, text

from music_search.db import MusicRepository, create_engine_from_url, create_session_factory
from music_search.db.models import Segment, Track
from music_search.db.repository import SegmentWrite, TrackWrite

pytestmark = pytest.mark.integration


@pytest.fixture
def database():
    url = os.getenv("MUSIC_SEARCH_TEST_DATABASE_URL")
    if not url:
        pytest.skip("Set MUSIC_SEARCH_TEST_DATABASE_URL to a disposable pgvector database")
    admin = create_engine_from_url(url)
    schema = "music_search_test_" + uuid4().hex
    with admin.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public"))
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_engine_from_url(url)

    @event.listens_for(engine, "connect")
    def set_schema(connection, _record):
        connection.autocommit = True
        with connection.cursor() as cursor:
            cursor.execute(f'SET search_path TO "{schema}", public')
        connection.autocommit = False

    try:
        root = Path(__file__).resolve().parents[1]
        config = Config(str(root / "alembic.ini"))
        config.set_main_option("script_location", str(root / "migrations"))
        with engine.begin() as connection:
            config.attributes["connection"] = connection
            command.upgrade(config, "head")
        yield engine, config
    finally:
        engine.dispose()
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


def test_live_migration_upsert_search_rollback_and_cascade(database) -> None:
    engine, config = database
    repository = MusicRepository(create_session_factory(engine))
    vector = [1.0] + [0.0] * 511
    segment = SegmentWrite(0, 0, 10, 0.2, None, vector)
    track = TrackWrite(
        source_uri="file:///synthetic.wav",
        content_sha256="a" * 64,
        title="Synthetic",
        duration_seconds=10,
        sample_rate=48000,
        channels=1,
        bpm=None,
        embedding_model="model-a",
        analysis_version="test",
        segments=[segment],
    )
    first = repository.replace_track(track)
    assert len(first.segments) == 1
    assert repository.find_track_by_content_hash("a" * 64).id == first.id
    found = repository.search_nearest_segments(vector, embedding_model="model-a")
    assert found[0].cosine_similarity == pytest.approx(1.0)
    assert repository.search_nearest_segments(vector, embedding_model="model-b") == []
    updated = repository.replace_track(replace(track, title="Reindexed"))
    assert updated.id == first.id
    assert updated.segments[0].id != first.segments[0].id

    # Force a database-side segment failure after the upsert and segment delete.
    def reject_segment_insert(_conn, _cursor, statement, _params, _context, _many):
        if statement.lstrip().startswith("INSERT INTO segments"):
            raise RuntimeError("simulated segment persistence failure")

    event.listen(engine, "before_cursor_execute", reject_segment_insert)
    try:
        with pytest.raises(RuntimeError, match="simulated"):
            repository.replace_track(replace(track, title="Must roll back"))
    finally:
        event.remove(engine, "before_cursor_execute", reject_segment_insert)
    restored = repository.find_track_by_content_hash("a" * 64)
    assert restored.title == "Reindexed"
    assert restored.segments[0].id == updated.segments[0].id
    with engine.begin() as connection:
        connection.execute(delete(Track).where(Track.id == first.id))
        assert connection.scalar(select(func.count()).select_from(Segment)) == 0
        config.attributes["connection"] = connection
        command.downgrade(config, "base")
        assert connection.scalar(text("SELECT to_regclass('tracks')")) is None
