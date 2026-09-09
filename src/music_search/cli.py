"""Typer command-line interface for local ingestion and semantic search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from music_search.config import Settings, get_settings

if TYPE_CHECKING:
    from music_search.db import MusicRepository
    from music_search.embeddings import ClapEmbedder

app = typer.Typer(
    name="music-search",
    help="Discover authorized local music with natural-language CLAP search.",
    no_args_is_help=True,
)
console = Console()
error_console = Console(stderr=True)


def _repository(settings: Settings) -> MusicRepository:
    from music_search.db import MusicRepository, create_engine_from_url, create_session_factory

    engine = create_engine_from_url(settings.database_url)
    return MusicRepository(create_session_factory(engine))


def _embedder(settings: Settings) -> ClapEmbedder:
    from music_search.embeddings import ClapEmbedder

    return ClapEmbedder(
        model_name=settings.clap_model_name,
        device=settings.device,
        batch_size=settings.embedding_batch_size,
    )


@app.command("db-upgrade")
def db_upgrade() -> None:
    """Create or migrate the configured PostgreSQL/pgvector schema."""

    from alembic import command
    from alembic.config import Config

    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "alembic.ini"
    migration_path = project_root / "migrations"
    if not config_path.is_file():
        config_path = Path(__file__).parent / "_alembic.ini"
        migration_path = Path(__file__).parent / "_migrations"

    try:
        settings = get_settings()
        config = Config(str(config_path))
        config.set_main_option("script_location", str(migration_path))
        config.set_main_option("sqlalchemy.url", settings.database_url.replace("%", "%%"))
        command.upgrade(config, "head")
    except Exception as exc:
        error_console.print(f"[red]Database migration failed:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc
    console.print("[green]Database is at the latest migration.[/]")


@app.command()
def ingest(
    path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            readable=True,
            resolve_path=True,
            help="Authorized local audio file or directory.",
        ),
    ],
    artist: Annotated[
        str | None,
        typer.Option(help="Artist override applied to every discovered file."),
    ] = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Reanalyze content already stored by SHA-256."),
    ] = False,
    fail_fast: Annotated[
        bool,
        typer.Option(help="Stop after the first file that fails analysis."),
    ] = False,
) -> None:
    """Analyze a folder of audio that you are authorized to process."""

    from music_search.ingestion import IngestionService

    console.print(
        "Analyzing local files only. The operator is responsible for supplying "
        "audio authorized for analysis."
    )
    try:
        settings = get_settings()
        service = IngestionService(
            _repository(settings),
            _embedder(settings),
            sample_rate=settings.sample_rate,
            window_seconds=settings.window_seconds,
            stride_seconds=settings.stride_seconds,
        )
        summary = service.ingest_path(
            path,
            artist=artist,
            force=force,
            fail_fast=fail_fast,
        )
    except Exception as exc:
        error_console.print(f"[red]Ingestion could not start:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    if summary.discovered_count == 0:
        error_console.print(f"[yellow]No supported audio files found in {path}.[/]")
        raise typer.Exit(1)

    table = Table(title="Ingestion results")
    table.add_column("Status")
    table.add_column("File")
    table.add_column("Segments", justify="right")
    table.add_column("Detail")
    for result in summary.results:
        color = {"ingested": "green", "skipped": "yellow", "failed": "red"}[result.status]
        table.add_row(
            f"[{color}]{result.status}[/]",
            escape(str(result.path)),
            str(result.segment_count),
            escape(result.error or ""),
        )
    console.print(table)
    console.print(
        f"Ingested {summary.ingested_count}; skipped {summary.skipped_count}; "
        f"failed {summary.failed_count}."
    )
    if summary.failed_count:
        raise typer.Exit(1)


@app.command()
def search(
    prompt: Annotated[
        str | None,
        typer.Argument(help="Natural-language description; omit for an interactive prompt."),
    ] = None,
    limit: Annotated[
        int,
        typer.Option(min=1, max=100, help="Maximum number of unique tracks."),
    ] = 10,
    json_output: Annotated[
        bool,
        typer.Option("--json", help="Emit machine-readable JSON instead of a table."),
    ] = False,
) -> None:
    """Return tracks whose audio windows best match a natural-language prompt."""

    from music_search.search import SearchService

    query = prompt if prompt is not None else typer.prompt("Search")
    try:
        from music_search.query_parser import parse_query

        query = parse_query(query).semantic_text
        settings = get_settings()
        service = SearchService(_repository(settings), _embedder(settings))
        results = service.search(query, limit=limit)
    except Exception as exc:
        error_console.print(f"[red]Search failed:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    if json_output:
        console.print_json(json.dumps([result.to_dict() for result in results]))
        return

    if not results:
        console.print("No indexed tracks matched. Ingest authorized audio first.")
        return

    table = Table(title=f"Music matches for: {query.strip()}")
    table.add_column("#", justify="right")
    table.add_column("Track")
    table.add_column("Similarity", justify="right")
    table.add_column("Why it matched")
    table.add_column("Source")
    for rank, result in enumerate(results, start=1):
        display_name = result.title
        if result.artist:
            display_name = f"{result.title} — {result.artist}"
        table.add_row(
            str(rank),
            escape(display_name),
            f"{result.score:.3f}",
            result.explanation,
            escape(result.source_uri),
        )
    console.print(table)


if __name__ == "__main__":
    app()
