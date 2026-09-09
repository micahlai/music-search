"""CLI failures should be concise and leave inference untouched."""

import json
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

import music_search.cli as cli
from music_search.config import get_settings

runner = CliRunner()


def test_help_is_independent_of_settings_and_inference(monkeypatch) -> None:
    settings = Mock(side_effect=AssertionError("no settings for help"))
    monkeypatch.setattr(cli, "get_settings", settings)
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0
    assert "db-upgrade" in result.output
    settings.assert_not_called()


@pytest.mark.parametrize("command", ["search", "db-upgrade", "ingest"])
def test_bad_configuration_is_reported_without_traceback(command, tmp_path, monkeypatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("MUSIC_SEARCH_EMBEDDING_DIMENSION", "1024")
    args = [command]
    if command == "search":
        args += ["piano"]
    elif command == "ingest":
        args += [str(tmp_path)]
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 1
    assert "512" in result.output
    assert "Traceback" not in result.output
    get_settings.cache_clear()


def test_empty_json_is_valid_json(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_repository", lambda _: Mock())
    monkeypatch.setattr(cli, "_embedder", lambda _: Mock())
    monkeypatch.setattr("music_search.search.SearchService.search", lambda *a, **k: [])
    result = runner.invoke(cli.app, ["search", "piano", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.output) == []


def test_blank_prompt_is_rejected_before_setup(monkeypatch) -> None:
    repository = Mock(side_effect=AssertionError("no setup for blank input"))
    monkeypatch.setattr(cli, "_repository", repository)
    result = runner.invoke(cli.app, ["search", " "])
    assert result.exit_code == 1
    repository.assert_not_called()


@pytest.mark.parametrize("quiet", [False, True])
def test_ingestion_progress_is_live_by_default_and_quiet_is_optional(
    quiet, tmp_path, monkeypatch
) -> None:
    (tmp_path / "[red]example.wav").touch()
    repo = Mock()

    def fail_lookup(_digest):
        # Verify progress was printed before the blocking database operation.
        if not quiet:
            assert any("Hashing" in message for message in observed)
        raise ValueError("database unavailable")

    repo.find_track_by_content_hash.side_effect = fail_lookup
    monkeypatch.setattr(cli, "_repository", lambda _: repo)
    monkeypatch.setattr(cli, "_embedder", lambda _: Mock())
    observed: list[str] = []
    original_print = cli.console.print

    def capture(message, *args, **kwargs):
        if isinstance(message, str):
            observed.append(message)
        original_print(message, *args, **kwargs)

    monkeypatch.setattr(cli.console, "print", capture)
    args = ["ingest", str(tmp_path)] + (["--quiet"] if quiet else [])
    result = runner.invoke(cli.app, args)
    assert result.exit_code == 1
    assert "database unavailable" in result.output
    assert "failed 1" in result.output
    assert ("Found 1 supported audio files" in result.output) is not quiet
    if not quiet:
        assert "[red]example.wav" in result.output
        assert "FAILED:" in result.output
