# Architecture

## Current implementation

The CLI composes three separable layers: decoded-audio analysis, ingestion/search
services, and a PostgreSQL repository. Python protocols allow tests to replace the
embedder and repository without a database or model download.

```text
CLI ── Settings
 ├─ IngestionService
 │   ├─ content hash → duplicate lookup
 │   ├─ SoundFile → mono PCM → scipy resample_poly
 │   ├─ overlapping windows → CLAP audio encoder
 │   ├─ librosa RMS/BPM
 │   └─ MusicRepository → transaction → tracks + segments
 └─ SearchService
     ├─ pass-through query → CLAP text encoder
     ├─ MusicRepository → cosine segment candidates
     └─ max segment similarity per track → timestamp/features → CLI output
```

An entire file is decoded in memory and its windows are materialized. Inference is
batched, but this is not a streaming ingester. Files are processed serially; each
complete analysis is one transaction. Failures do not discard other files.

The first migration establishes tracks, segments, foreign keys, content-hash and
segment-index uniqueness, scalar checks, and a 512-dimensional HNSW cosine index.
Track RMS is not a dedicated column. Segment RMS and track/segment BPM are persisted;
unknown BPM is NULL. Raw waveforms are never database fields.

## Decisions

| Decision | Reason and cost |
| --- | --- |
| Python backend | Existing audio and PyTorch ecosystem; avoid a rewrite between prototype and service. |
| CLAP audio/text encoder | Shared retrieval space without custom training; subtle film language needs evaluation. |
| 10-second windows, 5-second stride | Local evidence with overlap; does not model long-range arrangement on its own. |
| Mono 48 kHz | Default CLAP input contract; stereo spatial information is discarded. |
| PostgreSQL + pgvector | One transactional store for metadata and vectors; index/filter recall needs measurement. |
| Best-segment track score | Easy to inspect; long tracks have more opportunities to match. |
| SHA-256 identity | Restartable deduplication; different encodings of one recording remain separate tracks. |
| Nullable BPM | Silence and uncertain tempo remain unknown, not falsely zero-tempo music. |

Model identity belongs to the analysis provenance. Embedding dimension alone does
not establish compatibility. Later model variants/revisions need separate versioned
indexes or filtering, and controlled reindex migrations.

## Planned website and deployment

The product is planned as a **Next.js/TypeScript website hosted on Vercel**. The
website will submit queries, render results and matching moments, and link to source
platforms. The current milestone does not create or deploy that website.

```text
browser → Next.js on Vercel → Python search API → PostgreSQL/pgvector
                                  │
                                  └─ resident CLAP text inference

authorized assets → job queue → Python analysis workers → database
official metadata APIs → metadata connectors → canonical source records
```

Python is the intended production language for audio ingestion, DSP, CLAP/MERT
inference, and initial ranking. A FastAPI adapter can wrap the existing services.
The API and long-running analysis workers should be deployable independently, with
model caching, job retries, and optional accelerators selected after measurement.

Vercel supports Python/FastAPI as well as the planned Next.js frontend. The choice
to use a separate inference service is an engineering proposal based on model
residency and batch jobs, not an assertion that Python cannot run on Vercel. Before
hosting selection, benchmark package/model size, cold starts, peak memory, duration,
concurrency, cost, and available hardware against current platform limits. See
[Vercel Python runtime](https://vercel.com/docs/functions/runtimes/python) and
[function limits](https://vercel.com/docs/functions/limitations).

Keep development, preview, staging, and production credentials/catalogs separate.
Use pooled database connections, authenticated service calls, and object-storage
references for future authorized uploads. The frontend should receive public result
DTOs, never local filesystem paths or database credentials.

## Testing language decision

Python tests remain appropriate for the Python backend in production: unit tests
for transforms/ranking, database integration tests, and measured retrieval quality
on labeled audio. TypeScript tests cover the future Next.js components and routes;
browser tests cover complete search interactions. A versioned HTTP schema and
cross-language contract tests connect the two. Reconsider a TypeScript API only if
operational evidence favors it; Python inference and analysis can remain separate.

## Source versus audio capability

A platform track ID or public playback URL is descriptive identity, not an audio
asset. Spotify/YouTube connectors will normalize official metadata and source links.
They cannot enqueue waveform analysis. A separately authorized asset is required.
Later canonical tables will separate `tracks`, `track_sources`, `audio_assets`,
`embeddings`, and `structural_events`; there is no licensing table or pricing logic.

## Known limits and evolution

V1 has bounded segment candidates, approximate retrieval, serial/in-memory ingestion,
filename-derived metadata, no catalog-wide arrangement model, and no relevance
benchmark result yet. The next improvements should be driven by retrieval failures:
more candidate coverage, explicit model provenance, human labels, richer features,
and validated constraints before production scale.
