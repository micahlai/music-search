# Music discovery engine

A Python CLI for semantic music discovery over **authorized local audio files**.
The planned product is a Next.js website hosted on Vercel, supported by Python
audio-analysis and search services. This starter proves retrieval before building
the website or adding platform metadata connectors.

## What works in this starter

- Recursive audio discovery, SHA-256 deduplication, and atomic per-track reindexing.
- SoundFile decoding, mono conversion, and 48 kHz polyphase resampling.
- 10-second windows with 5-second stride; one padded tail with true timestamps.
- Lazy, batched CLAP audio/text inference with normalized 512-dimensional vectors.
- Window RMS energy and estimated window/track BPM.
- PostgreSQL + pgvector storage, Alembic migration, and cosine segment retrieval.
- Unique ranked tracks, best matching timestamps, and table/JSON CLI output.

The project helps people find existing music. Licensing, purchasing, rights
management, scraping, streaming downloads, and DRM bypass are outside scope.
Spotify and YouTube modules are future **metadata-only** connector placeholders.

## Quick start

On this Mac, PostgreSQL + pgvector has also been configured directly with Homebrew.
Use [local database instructions](docs/LOCAL_DATABASE.md) for that setup; Docker is
not required when the native service is running.

Use Python 3.11–3.13, [uv](https://docs.astral.sh/uv/), and Docker Compose.
Run these commands from the repository root:

```bash
uv sync --group dev
cp .env.example .env
docker compose up -d --wait
uv run music-search db-upgrade
uv run music-search ingest /absolute/path/to/authorized/audio
uv run music-search search "dreamy guitar with a cinematic build" --limit 10
```

`uv.lock` records resolved versions. The default database credentials are for local
development. Compose binds PostgreSQL to localhost and persists it in a named volume.
An existing PostgreSQL 16+ service with pgvector can be used instead by setting
`MUSIC_SEARCH_DATABASE_URL` in `.env`. Its migration role must be able to create the
`vector` extension. `docker compose down` stops the local service and retains data.

The first ingestion/search downloads the CLAP processor and weights from Hugging
Face. Allow disk space and time for PyTorch and model weights; CPU inference works
but indexing can be slow. A populated Hugging Face cache supports offline operation
with `HF_HUB_OFFLINE=1`. Audio files stay local and are not uploaded to Hugging Face.
Set `MUSIC_SEARCH_DEVICE=cpu` if an accelerator reports unsupported operations.

SoundFile uses libsndfile; codec support depends on its build. WAV/FLAC are good
initial inputs. MP3, Ogg, AIFF, and other recognized formats work when supported by
the installed decoder. AAC/M4A are not included; convert authorized files separately
using a decoder you control. Corrupt files are reported individually.

### Commands

```bash
# Override inferred artist; filename convention is Artist - Title.ext.
uv run music-search ingest ./audio --artist "Example Artist"

# Replace stored analysis for duplicate file bytes.
uv run music-search ingest ./audio --force

# Stop after the first failed file.
uv run music-search ingest ./audio --fail-fast

# Ingestion is verbose by default: file/stage/batch progress and elapsed times.
# Use --quiet to keep only the final result table and totals.
uv run music-search ingest ./audio --quiet

# Interactive prompt and JSON output for scripts.
uv run music-search search
uv run music-search search "restrained piano and strings" --limit 5 --json

# Inspect/apply migrations directly.
uv run alembic current
uv run alembic upgrade head
```

A non-zero ingestion exit status means at least one file failed, no supported files
were found, or setup failed. Previously successful files remain committed. A rerun
skips identical bytes regardless of filename; `--force` updates the existing track
and replaces its segments. Empty search results are successful and emit `[]` in JSON.

### Configuration

See [.env.example](.env.example). Defaults are 48,000 Hz, 10/5-second window/stride,
batch size 8, automatic device selection (CUDA, MPS, then CPU), and
`laion/clap-htsat-unfused`. The starter schema fixes the embedding dimension at 512.
The default checkpoint expects 48 kHz and at most 10 seconds per input window.
Changing model identity requires reanalysis; embeddings from different models must
not be compared. CLI search filters tracks by the configured model identifier. To
switch models, set `MUSIC_SEARCH_CLAP_MODEL_NAME` and rerun ingestion with `--force`
on the full catalog. Until reanalysis completes, search sees only the updated subset.
Checkpoint revision pinning is deferred; never change weights under the same model
identifier during a run. Shorter windows/stride can be configured for experiments.

## Architecture and ranking

```text
local file → SHA-256 → decode/resample → windows → CLAP + DSP → PostgreSQL
prompt → CLAP text encoder → pgvector cosine search → best segment per track → CLI
```

`tracks` stores content identity, file URI, inferred metadata, analysis parameters,
model identity, duration, and estimated BPM. `segments` stores time bounds, RMS,
estimated BPM, and `vector(512)`. HNSW supports cosine retrieval.

Track score is its strongest retrieved segment's cosine similarity. This is not a
calibrated match percentage. Search retrieves extra segments before grouping tracks;
candidate retrieval is bounded and approximate, so it may return fewer than the
requested number of unique tracks. Whole-song arrangement and “no vocals” constraints
are not enforced yet. The explanation reports the matching time window and measured
features; it does not invent musical tags. Unknown BPM is stored as SQL `NULL`.

## Repository map

```text
src/music_search/
  audio.py, features.py, embeddings.py   decoding, DSP, CLAP
  ingestion.py, search.py, cli.py        orchestration and commands
  config.py                            environment configuration
  db/                                  SQLAlchemy models and repository
  query_parser.py, mert.py              future analysis/query boundaries
  connectors/                          Spotify/YouTube metadata TODOs
migrations/                            versioned PostgreSQL schema
tests/                                 offline unit tests and opt-in DB tests
docs/                                  scope, specification, milestones, architecture
frontend/                              future Next.js/Vercel placeholder
docker-compose.yml                     local pgvector service
pyproject.toml, uv.lock                 package, tooling, resolved dependencies
```

## Verification

```bash
uv run pytest
uv run ruff check .
uv run mypy
uv run alembic upgrade head --sql
```

Run the opt-in integration test against a disposable test database with:

```bash
MUSIC_SEARCH_TEST_DATABASE_URL=postgresql+psycopg://music_search:music_search@localhost:5432/music_search uv run pytest -m integration
```

It creates and removes a uniquely named test schema, retaining the `vector` extension.
It verifies live migrations, atomic reindex rollback, model filtering, and cascade
deletion. Without the explicit test URL this test is skipped.

Unit tests use synthetic audio and mocked inference; they do not download model
weights or contact PostgreSQL. SQL compilation alone does not prove live migrations
or semantic relevance. See [milestone gates](docs/MILESTONES.md) for live database,
real-model smoke testing, and the 100–1,000-track relevance benchmark.

## Product plan

- [Project scope](docs/PROJECT_SCOPE.md): users, product promise, included/excluded work.
- [Detailed specification](docs/SPECIFICATION.md): requirements and acceptance criteria.
- [Architecture](docs/ARCHITECTURE.md): component boundaries and Python/Vercel decisions.
- [Milestones](docs/MILESTONES.md): staged deliverables and validation gates.
- [Roadmap](docs/ROADMAP.md): LLM parsing, MERT, structure, connectors, and website.
- [Retrieval design](docs/RETRIEVAL_DESIGN.md): measured ranking diagnosis, checkpoint
  and calibration ablation, analysis layers, and vector storage budget.
- [Validation record](docs/VALIDATION.md): Astra review, executed checks, and pending live gates.

Python remains the intended implementation and test language for ML/audio services.
The later Next.js/TypeScript frontend will run on Vercel; its unit/browser tests and
API contract tests will complement Python tests. Backend hosting will be chosen with
measured memory, model-load time, inference latency, and job-duration evidence.
