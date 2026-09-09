from __future__ import annotations

import pytest
from pydantic import ValidationError

from music_search.config import Settings


def test_settings_use_prototype_defaults() -> None:
    settings = Settings(_env_file=None)

    assert settings.sample_rate == 48_000
    assert settings.window_seconds == 10.0
    assert settings.stride_seconds == 5.0
    assert settings.embedding_dimension == 512


def test_stride_cannot_exceed_window() -> None:
    with pytest.raises(ValidationError, match="stride_seconds"):
        Settings(window_seconds=5, stride_seconds=10, _env_file=None)


def test_settings_accept_environment_strings(monkeypatch) -> None:
    monkeypatch.setenv("MUSIC_SEARCH_SAMPLE_RATE", "48000")
    monkeypatch.setenv("MUSIC_SEARCH_EMBEDDING_DIMENSION", "512")
    settings = Settings(_env_file=None)
    assert settings.sample_rate == 48000
    assert settings.embedding_dimension == 512
