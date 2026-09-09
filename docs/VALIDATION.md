# Starter validation record

Date: 2026-09-09. This record covers the initial local prototype and planning
documents. It does not assert production readiness or measured music relevance.

**Current update:** the previously pending live gates below are now exercised.
PostgreSQL 17.11/pgvector 0.8.6 are running locally, all 53 tests passed, and real
CLAP ingestion/search succeeded on 20 tracks (881 segments). See the subsequent
[live smoke-test report](SMOKE_TEST.md) and [local setup](LOCAL_DATABASE.md).
The initial review/check results below are retained as historical evidence.

## Independent documentation review

The scope, specification, architecture, milestones, roadmap, and README were reviewed
against the code by **gpt-6-astra**, explicitly requested by the project owner.
Its initial review accepted product direction and identified implementation/contract
gaps. These were corrected and independently reviewed again.

The final review concluded: **Pass; no remaining must-fix doc/code contradictions.**
The reviewer inspected source and tests read-only; test execution evidence below
comes from the implementation session, not from the reviewer.

Corrections verified:

- CLI search filters by the stored embedding-model identifier.
- Incompatible vector dimensions, sample rates, and long CLAP windows are rejected.
- Configuration and initialization errors pass through concise CLI error handlers.
- Nonfinite decoded PCM is rejected before analysis.
- BPM output identifies the segment or whole-track estimate it uses.
- The future vocal query example specifies interval, allowed state, and strength.
- The specification matches reindex UUID behavior, segment-only RMS persistence,
  nullable BPM, and possible underfilling from bounded segment candidates.

The final optional wording suggestion, “positive estimate or NULL” for stored
segment BPM, was also applied.

## Executed verification

| Check | Result |
| --- | --- |
| `uv sync --group dev` | Successful isolated install and resolved lock file |
| `uv run pytest -q` | 52 passed, 1 skipped |
| `uv run ruff check .` | Passed |
| `uv run mypy` | Passed; 18 source files |
| `uv run alembic upgrade head --sql` | PostgreSQL/pgvector migration SQL compiled |
| `uv run music-search --help` | CLI entry point worked |
| `uv build` | Wheel and source distribution built successfully |
| Wheel contents | Alembic config, migration scripts, and type marker included |
| Relative Markdown links | No broken links in README/docs |

Tests cover synthetic decoding/resampling/windowing, real installed CLAP feature
preprocessing, mocked model loading/inference, normalization, database validation and
SQL construction, ingest failures/restarts, track grouping/model filtering, BPM
provenance, and CLI behavior. No test downloaded a pretrained checkpoint.

## Live gates pending at the initial review (now exercised)

The live database integration test is skipped unless
`MUSIC_SEARCH_TEST_DATABASE_URL` explicitly names a disposable pgvector database.
Docker and PostgreSQL executables were not available on this host. The test creates
and removes an isolated random schema and covers migration, upsert, vector search,
rollback, and cascade deletion.

Pretrained CLAP inference and a complete real-audio → database → CLI-search run have
not been validated. A real installed feature extractor and mocked encoder tests do
not substitute for that evidence. No relevance or latency benchmark has been run
on the proposed 100–1,000 authorized tracks.

M0 is documentation-validated. M1 is implemented with offline checks passed, while
live PostgreSQL, pretrained model, and small-catalog smoke gates remain pending.
M2 and later milestones remain planned. See [MILESTONES.md](MILESTONES.md).
