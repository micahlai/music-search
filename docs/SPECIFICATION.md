# Music discovery engine specification

Status: starter-prototype and evolutionary product specification

Version: 0.2

Normative words: **must**, **should**, and **may** describe required, recommended,
and optional behavior respectively.

## 1. Purpose

This specification defines a local Python prototype that proves text-to-music
retrieval over authorized audio. It also fixes the boundaries and data contracts
needed to evolve that prototype into a multi-source discovery product.

The system is not a licensing platform. The intended product is a website hosted on
Vercel that returns discovery evidence and links to an original source; listening
rights, acquisition, and licensing remain between the user and that source. The
website is deferred until the local retrieval hypothesis has been evaluated.

## 2. System context

### 2.1 V1 context

```text
authorized local folder
        │
        ▼
decode → mono/resample → overlapping windows
        │                         │
        │                         ├─ CLAP audio vectors
        │                         └─ RMS + BPM
        ▼
PostgreSQL + pgvector
        ▲
        │ cosine nearest segments
CLAP text vector ← natural-language CLI prompt
        │
        ▼
ranked tracks + best moments + measured evidence
```

### 2.2 Later context

```text
official metadata connectors ─────────┐
partner/artist authorized assets ─────┼─→ canonical catalog + analysis jobs
operator-owned authorized files ──────┘

prompt → structured query parser → hybrid candidate retrieval
       → deterministic filters → feature-aware reranker
       → grounded explanations → service API
       → Next.js website on Vercel → original source link
```

Metadata connectors and analyzable-audio providers are separate interfaces.

### 2.3 Deployment boundary

The planned Vercel deployment owns the browser-facing Next.js application and may
own a thin web/API edge if a deployment spike validates that choice. It does not,
by default, own audio decoding, resampling, long-running CLAP/MERT inference, or
bulk reindexing. Those jobs require a separate Python worker/service with suitable
CPU/GPU, memory, timeout, queue, and storage characteristics. The exact provider
and protocol between the web edge, search API, job system, and worker are open
decisions; [ARCHITECTURE.md](ARCHITECTURE.md) defines the constraints without
claiming a final topology.

## 3. V1 functional requirements

### 3.1 Configuration

- **CFG-001** The application must read configuration from environment variables
  prefixed with `MUSIC_SEARCH_` and may load a local `.env` file.
- **CFG-002** Defaults must be 48,000 Hz, 10-second windows, 5-second stride,
  512-dimensional embeddings, and `laion/clap-htsat-unfused`.
- **CFG-003** Window and stride values must be positive, and stride must not exceed
  window length.
- **CFG-004** The database URL, model, inference device, embedding batch size, and
  window/stride parameters must be configurable. This CLAP starter fixes 48 kHz and
  limits windows to at most 10 seconds, rejecting incompatible settings before work.
- **CFG-005** The schema's vector dimension and configured embedding dimension must
  agree. A different dimension requires a migration and reindex, not a silent cast.

### 3.2 Local file discovery

- **ING-001** The ingest command must accept a file or directory path on the local
  filesystem.
- **ING-002** Directory discovery must be recursive, deterministic, and
  case-insensitive by file extension.
- **ING-003** Supported extensions should cover formats available through the local
  libsndfile build, including WAV, FLAC, AIFF, Ogg/Opus, and MP3 where supported.
- **ING-004** Unsupported files must be ignored during directory discovery.
- **ING-005** An empty discovery result must produce a clear non-success message.
- **ING-006** This interface must not accept remote URLs.
- **ING-007** The CLI and documentation must state that the operator is responsible
  for supplying audio authorized for analysis.

### 3.3 Content identity and metadata

- **ING-010** The ingester must compute SHA-256 from the original local file bytes.
- **ING-011** SHA-256 must be unique in the V1 catalog and serve as the duplicate
  identity even when the same bytes appear at another path.
- **ING-012** A normal ingest must skip content already present.
- **ING-013** A force/reindex operation may replace derived analysis atomically.
- **ING-014** V1 must infer title from the filename stem. It should recognize the
  common `Artist - Title` filename pattern and allow an operator-supplied artist
  override.
- **ING-015** Stored track metadata must include source path/URI, hash, title,
  optional artist and album, duration, sample rate, channel count after analysis,
  track BPM estimate, analysis/model version, arbitrary JSON metadata, and
  timestamps. Segment records carry segment RMS and segment BPM estimates.
- **ING-016** Local path information must not be exposed outside an operator-owned
  deployment without an explicit privacy decision.

### 3.4 Decode and standardization

- **AUD-001** A supported file must decode to finite floating-point PCM.
- **AUD-002** Multi-channel audio must be downmixed to mono before embedding and DSP
  analysis.
- **AUD-003** PCM must be resampled to the configured rate with an anti-aliasing
  resampler.
- **AUD-004** V1 must store the analysis sample rate and the standardized channel
  count (`1`).
- **AUD-005** A corrupt, unreadable, or empty file must fail that file with a clear
  error; by default it must not roll back successfully ingested sibling files.
- **AUD-006** The pipeline must not write decoded waveform bytes to PostgreSQL.

### 3.5 Windowing

- **SEG-001** Default windows must be 10.0 seconds long with starts separated by a
  5.0-second stride.
- **SEG-002** Timestamps must refer to positions in the decoded source timeline.
- **SEG-003** A window that extends beyond EOF must be right-padded with zeroes for
  model input while retaining its true, unpadded end time.
- **SEG-004** At most one partial trailing window may be emitted.
- **SEG-005** If a full window reaches EOF exactly, the pipeline must not add a
  redundant trailing partial window.
- **SEG-006** Segment indexes must be zero-based, ordered, and unique per track.
- **SEG-007** DSP values for a partial tail should use only valid source samples so
  padding does not lower energy or alter BPM.

Examples at the default settings:

| Track duration | Windows |
| ---: | --- |
| 7 s | `0–7` (padded to 10 s for CLAP) |
| 10 s | `0–10` |
| 12 s | `0–10`, `5–12` (second padded) |
| 20 s | `0–10`, `5–15`, `10–20` |

### 3.6 DSP features

- **DSP-001** V1 must compute RMS energy for every segment. Whole-track RMS is not
  part of the V1 persistence contract.
- **DSP-002** RMS must be finite and non-negative; silence must return zero.
- **DSP-003** V1 must estimate whole-track and segment BPM using a deterministic
  library call and the analysis sample rate.
- **DSP-004** The DSP boundary must return a finite, non-negative value. Silence,
  clips too short for a meaningful estimate, or indeterminate tempo return `0.0`
  rather than NaN; the ingestion boundary must convert that sentinel to `NULL`
  before persistence. Persisted BPM values are therefore either positive estimates
  or `NULL`/unknown, never zero.
- **DSP-005** BPM is an estimate, not ground truth, and should be labeled as such in
  user-facing output.
- **DSP-006** Future feature additions must be versioned so a catalog can be
  reprocessed reproducibly.

### 3.7 CLAP embeddings

- **EMB-001** V1 must use one audio-text model for both audio-window and query-text
  embeddings. CLI search filters by stored model identifier. Immutable checkpoint
  revision pinning is deferred; replacing weights under an unchanged identifier
  requires explicit full reanalysis and must not occur during indexing/search.
- **EMB-002** Model loading must be lazy so configuration and help commands do not
  initialize PyTorch or download weights.
- **EMB-003** The default model must be `laion/clap-htsat-unfused`.
- **EMB-004** The first model use may download weights from the configured model
  registry; offline operators must be able to provide a pre-populated model cache.
- **EMB-005** Audio embeddings must be batched with a configurable batch size.
- **EMB-006** Device selection must support CPU and, when available, CUDA or Apple
  MPS. `auto` must choose a supported device without assuming one exists.
- **EMB-007** Every stored embedding must have exactly 512 finite float values.
- **EMB-008** Audio and text embeddings must be L2-normalized before storage/search.
- **EMB-009** Empty prompts must be rejected before model inference.
- **EMB-010** The model identifier must be stored with track analysis metadata so a
  later model change can trigger a complete, explicit reindex.

### 3.8 Persistence

- **DB-001** PostgreSQL with the pgvector extension is the V1 datastore.
- **DB-002** Schema changes must be expressed as ordered Alembic migrations.
- **DB-003** IDs must be UUIDs. Track IDs survive forced reanalysis of identical
  bytes. Replaced segments receive new IDs; segment IDs identify one analysis run.
- **DB-004** Deleting a track must cascade to its segments.
- **DB-005** Segment embeddings must use `vector(512)` and cosine distance.
- **DB-006** The schema must include a cosine HNSW index for approximate nearest
  neighbors and B-tree indexes for normal relationship/look-up fields.
- **DB-007** Track replacement and all of its new segments must commit atomically.
- **DB-008** A failed analysis must not leave a track with a partial new segment set.
- **DB-009** JSON metadata must default to an empty object and must not carry secrets.
- **DB-010** Timestamps must be timezone-aware and assigned by the database.

### 3.9 Query and retrieval

- **QRY-001** The search command must accept a natural-language prompt either as an
  argument or interactively.
- **QRY-002** V1 must trim the prompt, reject blank input, and otherwise pass the
  semantic text directly to the CLAP text encoder.
- **QRY-003** V1 must retrieve nearest audio segments using pgvector cosine distance.
- **QRY-004** The search service must request `max(50, 10 * track_limit)` segment
  candidates before grouping by track. This bounded approximate search can underfill
  the requested track count when one track dominates or index filtering loses recall.
- **QRY-005** V1 track score must be transparent. The default is the maximum cosine
  similarity among retrieved segments for that track.
- **QRY-006** Result ordering must be deterministic for equal scores.
- **QRY-007** Each result must include track identity, score, source path or URI,
  best matching start/end time, and the best segment's RMS. BPM uses the segment
  estimate when available, otherwise the whole-track estimate, otherwise unknown;
  output must label this provenance (`bpm_source` in JSON).
- **QRY-008** V1 explanations must use only measured/stored evidence and may read,
  for example, “best semantic match at 0:45–0:55; estimated 121 BPM (segment).”
- **QRY-009** No V1 output may imply license availability or permission to use a
  track.
- **QRY-010** Zero indexed segments must produce a clear empty-result response.

### 3.10 CLI

The executable name is `music-search`. It must expose:

```text
music-search db-upgrade
music-search ingest PATH [--artist NAME] [--force] [--fail-fast]
music-search search [PROMPT] [--limit N] [--json]
```

- **CLI-001** `db-upgrade` must apply migrations to the configured database.
- **CLI-002** `ingest` must report ingested, skipped, and failed counts and identify
  per-file failures without printing tracebacks during normal errors.
- **CLI-003** `ingest --fail-fast` must stop after the first failed file.
- **CLI-004** `search` without a prompt argument must request one interactively.
- **CLI-005** Human output should be a readable ranked table; JSON output must be
  stable enough for local scripting.
- **CLI-006** A command that encounters a configuration, model, or database error
  must return a non-zero exit status.
- **CLI-007** Help output must remain usable without a database connection or loaded
  CLAP model.

## 4. V1 data model

### 4.1 Track

| Field | Type | Rule |
| --- | --- | --- |
| `id` | UUID | Primary key. |
| `content_sha256` | 64-char text | Unique identity of original file bytes. |
| `source_uri` | text | Local file URI/path in V1; outbound URL later. |
| `title` | text | Required inferred/operator metadata. |
| `artist`, `album` | text/null | Optional. |
| `duration_seconds` | float | Positive decoded duration. |
| `sample_rate` | integer | Analysis sample rate. |
| `channels` | integer | `1` after V1 standardization. |
| `bpm` | float/null | Estimated whole-track tempo. |
| `analysis_version` | text | Pipeline contract version. |
| `embedding_model` | text | Exact CLAP model identifier. |
| `metadata` | JSONB | Non-secret extension fields. |
| `created_at`, `updated_at` | timestamptz | Database timestamps. |

### 4.2 Segment

| Field | Type | Rule |
| --- | --- | --- |
| `id` | UUID | Primary key. |
| `track_id` | UUID | Required track FK, cascade delete. |
| `segment_index` | integer | Zero-based, unique within track. |
| `start_seconds`, `end_seconds` | float | `0 <= start < end <= duration`. |
| `rms_energy` | float | Finite, non-negative. |
| `bpm` | float/null | Finite positive estimate, or NULL/unknown. |
| `embedding` | vector(512) | Normalized CLAP audio embedding. |
| `created_at`, `updated_at` | timestamptz | Database timestamps. |

### 4.3 Future canonical-source separation

The V1 table is deliberately compact. Before adding external sources, migrate toward
separate canonical concepts:

```text
tracks
  ├── track_sources       # provider ID, URL, source metadata
  ├── audio_assets        # explicit analysis capability/authorization
  ├── audio_segments
  ├── embeddings          # model/version/level-specific vectors
  ├── track_features
  └── structural_events
```

An `audio_asset` must carry a capability state such as `authorized_full_audio`,
`authorized_preview`, or `metadata_only`. A metadata-only record cannot enter the
audio job queue.

## 5. Ranking contract

### 5.1 V1

For normalized query vector `q` and segment vector `s`:

```text
segment_similarity = 1 - cosine_distance(q, s)
track_score = max(segment_similarity for candidate segments of track)
```

The implementation retrieves more than `track_limit` segments, groups them by track,
sorts by `track_score`, and then applies `track_limit`. The CLI exposes the best
matching segment as evidence. It does not label the raw cosine score as a calibrated
probability.

### 5.2 Planned hybrid ranking

The future score will combine versioned, inspectable signals. An initial experiment
may use:

```text
0.45 semantic audio-text similarity
+ 0.20 mood match
+ 0.15 structure/trajectory match
+ 0.10 instrumentation match
+ 0.10 event timing/rhythm match
- explicit mismatch penalties
```

Weights are hypotheses, not fixed product truth. They must be evaluated on labeled
queries. Hard constraints such as an excluded instrument or a requested time range
must be validated separately from soft similarity. Licensing and price are not
ranking signals in this project.

## 6. Future query contract

The later LLM's role is to translate creative language into validated data; it does
not listen to every track or freely choose result order. A target structure is:

```json
{
  "semantic_text": "dark electronic music for a product film",
  "moods": ["dark"],
  "instruments": {
    "required": [],
    "preferred": ["synth bass"],
    "excluded": []
  },
  "vocals": {
    "interval_seconds": [0, 45],
    "interval_end_exclusive": true,
    "allowed": false,
    "strength": "hard"
  },
  "events": [
    {
      "type": "drop",
      "time_range_seconds": [45, 60],
      "strength": "high",
      "attributes": ["bass-heavy"]
    }
  ],
  "trajectory": ["restrained intro", "large build"]
}
```

Unknown or ambiguous fields must remain unset rather than be invented. The raw prompt
must remain available as the semantic embedding input. Parser output must be schema
validated, versioned, and safe to ignore if parsing fails.

## 7. Failure and recovery behavior

| Failure | Required behavior |
| --- | --- |
| Folder/path missing | Reject before model load and identify the path. |
| No supported files | Return a clear empty-ingest result. |
| Codec unsupported/corrupt file | Record/report one failure; continue unless fail-fast. |
| Empty decoded audio | Do not create a zero-segment track. |
| Model unavailable/offline cache miss | Fail clearly; do not persist incomplete analysis. |
| Wrong embedding dimension/NaN | Reject before repository write. |
| Database unavailable | Fail the command with a concise connection error. |
| One segment insert fails | Roll back that track replacement. |
| Duplicate hash | Skip normally; reanalyze only when explicitly forced. |
| Blank search prompt | Reject before model or database work. |
| No search matches | Return success with an empty result set. |

Ingestion should be restartable: successfully committed files remain available, and
a rerun skips them by hash.

## 8. Nonfunctional requirements

### 8.1 Reproducibility

- Pin compatible dependency ranges and record exact resolved versions in a lock file
  when preparing repeatable deployments.
- Store pipeline and model identifiers with every analysis.
- Tests must not download model weights or require PostgreSQL unless explicitly
  marked as integration tests.
- Migrations, not ORM auto-create, define production schema.

### 8.2 Performance targets for prototype evaluation

- Batch CLAP audio inference to use the selected device efficiently.
- Index a 3-minute track as approximately 35 default windows.
- Return a top-10 search from 1,000 tracks within one second after query embedding on
  a typical developer machine/database; model text inference is measured separately.
- Keep database search sublinear with an ANN index once the collection is large
  enough for it to matter.

These are evaluation targets, not service-level commitments.

### 8.3 Observability

The CLI should report files discovered, current file, segment count, elapsed time,
skipped duplicates, and failures. A later service must add structured logs and job
metrics without logging raw prompts or local paths by default in shared deployments.

### 8.4 Privacy and security

- Never read remote URLs in the audio decoder.
- Treat filenames, paths, prompts, and source metadata as untrusted data.
- Use parameterized SQL through SQLAlchemy/psycopg.
- Keep database credentials in environment configuration, not committed files.
- Do not execute metadata, model output, or query text.
- Do not store raw audio in PostgreSQL.
- Before multi-user deployment, define retention and access control for derived
  embeddings, prompts, and private catalog metadata.

### 8.5 Model trust

- Cosine similarity is a retrieval signal, not a truth probability.
- BPM and future tags/events must identify their estimator/model version.
- LLM prose must not add unsupported claims.
- A later UI must distinguish “audio analyzed” from “matched using source metadata.”

## 9. Testing requirements

### 9.1 Unit tests

- deterministic recursive discovery;
- stereo downmix and sample-rate conversion;
- exact, overlapping, short, and partial-tail window cases;
- RMS/BPM behavior for tone, click track, short audio, and silence;
- lazy CLAP loading, device selection, batching, normalization, prompt validation,
  and dimension/finite-value checks with fakes;
- filename parsing and SHA-256 identity;
- duplicate/force/failure ingestion behavior using a fake repository and embedder;
- track collapsing and deterministic ranking;
- CLI help and validation without loading the model.

### 9.2 Database integration tests

A separately marked suite should run against the Docker Compose pgvector service and
verify migration up/down, vector insert/search, duplicate replacement, transaction
rollback, and cascade delete. It must not run implicitly against an arbitrary
developer database.

### 9.3 Retrieval evaluation

Create a versioned evaluation set containing:

- 100–1,000 authorized tracks;
- filmmaker-written prompts;
- one or more relevant tracks per prompt;
- hard negatives with similar broad tags but wrong structure/emotion;
- time-range labels when a prompt refers to a moment.

Report Recall@1/5/10, MRR, nDCG@10, segment timestamp error for temporal prompts,
indexing failures, and latency. Compare against filename/tag text search and random
ranking baselines.

## 10. V1 acceptance criteria

The starter prototype is complete when all of the following are true:

1. A fresh developer can follow the README to start pgvector, install the package,
   migrate the database, ingest a folder, and search it.
2. Ingestion operates only on local files and contains no platform-download code.
3. Audio is mono/48 kHz by default and windowed at 10/5 seconds with correct tails.
4. CLAP audio/text embeddings are normalized 512-value vectors from the same model.
5. Segment RMS plus whole-track and segment BPM are stored; indeterminate BPM is
   persisted as `NULL`.
6. The migration creates extension, tables, constraints, and vector index.
7. A rerun skips duplicate bytes; force reanalysis replaces all segments atomically.
8. Search returns unique tracks in ranked order with best timestamps and grounded
   feature evidence.
9. Unit tests pass without downloading CLAP weights or connecting to PostgreSQL.
10. Scope, architecture, milestones, setup, limitations, and next work are documented.

## 11. Deferred requirements

The following are specified as boundaries/TODOs, not implemented in V1:

- schema-constrained LLM prompt parsing;
- MERT embeddings and multi-vector fusion;
- track/section/event hierarchy and drop/build detection;
- vocals, instrumentation, mood, key, onset, spectral, and energy-curve analysis;
- metadata-only YouTube and Spotify connectors;
- other source/partner connectors with explicit capabilities;
- FastAPI search/ingestion service and job queue;
- Next.js website hosted on Vercel with outbound source cards;
- a deployment decision for the service API, PostgreSQL access, asset storage,
  queue, and separately hosted heavyweight Python analysis worker;
- tapped rhythm, hummed query, and authorized reference-clip retrieval;
- relevance-feedback collection, reranking experiments, and model fine-tuning;
- production authentication, tenant isolation, storage, telemetry, and scaling.

### 11.1 Query understanding and retrieval evolution

- **FUT-QRY-001** A later LLM parser must emit schema-constrained data, preserve the
  raw prompt for semantic retrieval, and leave unknown fields unset.
- **FUT-QRY-002** Parser failure must degrade to CLAP semantic search rather than
  prevent a query.
- **FUT-QRY-003** Deterministic code, not unconstrained generated prose, must apply
  hard exclusions, event-time ranges, and numeric filters.
- **FUT-EMB-001** MERT or another music-specific embedding may be added only as a
  separately versioned signal whose incremental value is measured against the
  frozen CLAP baseline.
- **FUT-EVT-001** Structural events such as builds, drops, transitions, sparse
  passages, and energy changes must retain start/end times, detector version, and
  confidence. A UI explanation may not claim an event without such evidence.

### 11.2 Source connector contract

- **FUT-SRC-001** Every connector must declare independent `metadata`,
  `outbound_link`, `authorized_preview`, and `authorized_full_audio` capabilities.
- **FUT-SRC-002** Spotify and YouTube integrations must use supported official
  interfaces for metadata and outbound links only; neither may download or extract
  platform audio.
- **FUT-SRC-003** A metadata-only source record must not create an audio-analysis
  job or be presented as audio-analyzed.
- **FUT-SRC-004** Musicbed, Artlist, Epidemic Sound, Pixabay, Fesliyan Studios,
  SoundCloud, Bandcamp, partner catalogs, and participating artists are candidates,
  not promised integrations. Each requires a terms, permissions, API, identity,
  freshness, and rate-limit review before implementation.
- **FUT-SRC-005** The product must preserve provider IDs and outbound URLs separately
  from canonical track identity so one work can have multiple source records.

### 11.3 Vercel website contract

- **FUT-WEB-001** The product website must use Next.js and be hosted on Vercel unless
  a later recorded architecture decision supersedes that requirement.
- **FUT-WEB-002** The primary flow must be prompt entry, ranked results, match
  evidence, matching timestamp, source identity, and an outbound source action.
- **FUT-WEB-003** Web output must not expose an operator's local filesystem path.
- **FUT-WEB-004** Result cards must distinguish audio-derived matches from
  metadata-only matches and must not imply license availability.
- **FUT-WEB-005** Playback may be shown only through a source-supported embed or an
  asset explicitly authorized for that use. Search must remain useful without
  in-product playback.
- **FUT-WEB-006** The website must preserve keyboard access, responsive layouts,
  explicit loading/empty/error states, and outbound-link safety before beta.
- **FUT-WEB-007** The browser must never receive database credentials, connector
  secrets, model credentials, private asset locations, or unrestricted ingestion
  endpoints.

### 11.4 Service and environment contract

- **FUT-DEP-001** Development, Vercel preview, and production must use separate
  configuration and data boundaries. Preview builds must not silently access the
  production catalog or private audio assets.
- **FUT-DEP-002** Search requests and long-running analysis jobs must have distinct
  execution paths, scaling policies, timeouts, and observability.
- **FUT-DEP-003** The default architecture must place CLAP/MERT inference, decode,
  and reindex jobs in a separately deployable Python worker/service. Running them
  on Vercel requires an evidence-backed deployment decision demonstrating runtime
  compatibility, bounded duration, model caching, memory, and acceptable cost.
- **FUT-DEP-004** Analysis jobs must be idempotent by asset hash plus pipeline/model
  version, retryable, and unable to enqueue metadata-only source records.
- **FUT-DEP-005** The deployment decision must document API ownership, PostgreSQL
  connection strategy, queue semantics, authorized asset storage, region/data
  residency, secret management, model artifact caching, and failure recovery.
- **FUT-DEP-006** Public search latency and offline indexing throughput must be
  measured separately; one must not be hidden inside the other's metric.

Implementation sequencing and exit gates are in [MILESTONES.md](MILESTONES.md).
