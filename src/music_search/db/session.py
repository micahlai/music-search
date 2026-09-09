"""Engine and transaction helpers for the synchronous CLI backend."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

DEFAULT_DATABASE_URL = "postgresql+psycopg://music_search:music_search@127.0.0.1:5432/music_search"


def resolve_database_url(database_url: str | None = None) -> str:
    """Resolve an explicit URL, environment/.env settings, then the local default."""

    resolved = database_url or os.getenv("MUSIC_SEARCH_DATABASE_URL")
    if resolved is None:
        try:
            from music_search.config import get_settings
        except ImportError:  # Allows the database package to remain usable in isolation.
            resolved = DEFAULT_DATABASE_URL
        else:
            resolved = get_settings().database_url
    # Make a common driver-less PostgreSQL URL select the project's psycopg v3 driver.
    if resolved.startswith("postgresql://"):
        return resolved.replace("postgresql://", "postgresql+psycopg://", 1)
    return resolved


def create_engine_from_url(
    database_url: str | None = None,
    *,
    echo: bool = False,
    pool_pre_ping: bool = True,
) -> Engine:
    """Create a sync SQLAlchemy engine suitable for CLI and batch ingestion work."""

    return create_engine(
        resolve_database_url(database_url),
        echo=echo,
        pool_pre_ping=pool_pre_ping,
    )


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    """Create the factory expected by :class:`MusicRepository`."""

    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    """Yield a session and commit or roll it back as one transaction."""

    with factory() as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
