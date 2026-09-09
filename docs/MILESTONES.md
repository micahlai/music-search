# Milestones and acceptance gates

Milestones are dependency-based; no delivery dates are promised. “Implemented” means
code exists; “validated” requires the exit evidence. This distinction prevents unit
test success from being mistaken for music-search quality.

| ID | Milestone | Depends on | Initial status |
| --- | --- | --- | --- |
| M0 | Scope and technical contract | None | Validated with Astra |
| M1 | Local Python retrieval prototype | M0 | Validated: 53 tests and real CLAP/pgvector on 20 tracks |
| M2 | Real-catalog relevance baseline | M1 | Planned |
| M3 | Structured LLM queries | M2 | Planned |
| M4 | Music representation experiments | M2 | Planned |
| M5 | Structural/event retrieval | M3, M4 experiments | Planned |
| M6 | Canonical catalog and metadata connectors | M2 | Planned |
| M7 | API and deployment spike | M2, source DTO design | Planned |
| M8 | Next.js website on Vercel | M6, M7 | Planned |
| M9 | Feedback, ranking improvements, operations | M8 | Planned |

## M0 — Establish the product contract

Deliver scope, detailed requirements, architecture decisions, source-access
boundaries, and this milestone plan. Explicitly supersede licensing-oriented ideas
from early planning. Exit: documents agree about V1 versus future functionality,
Vercel is the website target, Python remains the audio/backend language, and every
claimed implemented requirement maps to code or a documented gap.

## M1 — Authorized local retrieval

Entry: M0 scope. Deliver package/CLI, configuration, decoding/resampling, 10/5-second
windows, CLAP embeddings, RMS/BPM, atomic database writes, migrations, pgvector
search, ranked unique tracks, setup instructions, and offline tests.

Exit evidence:

- Tests and lint/type checks pass in the declared environment.
- A disposable pgvector database accepts the migration, insert, duplicate reindex,
  cosine search, rollback, and cascade behavior.
- Real CLAP encodes synthetic or authorized audio and text with finite 512D vectors.
- A fresh setup ingests and searches a small authorized catalog using README commands.
- Failure and restart behavior are demonstrated; no platform audio acquisition exists.

Risks: model download/compute footprint, decoder availability, stale model identity,
approximate index recall, and silent/short audio. Unit tests do not close the real
database/model gates. Missing runtime infrastructure must be reported explicitly.

The live gates were exercised on 2026-09-09: 20 files from `audio-small` produced
881 vectors; real CLAP searches and a duplicate-ingestion rerun passed. All 53
tests passed with the separate pgvector test database enabled. See
[SMOKE_TEST.md](SMOKE_TEST.md). This closes the starter runtime gates, not the M2
relevance target.

## M2 — Prove relevance on 100–1,000 tracks

Entry: executable M1 pipeline and an authorized catalog. Deliver a versioned dataset
manifest, 30–100 filmmaker prompts, graded relevant tracks, matching moments where
applicable, and hard negatives. Split evaluation data by track/artist where possible
to avoid near-duplicate leakage. Include “hopeful without triumphant” and gradual
build examples that cannot be solved from filenames alone.

Measure Recall@1/5/10, MRR, nDCG@10, indexing error rate, cold/warm query latency,
index size, and peak inference memory. Compare against random and filename/tag
baselines. Initial proposed gate: Recall@10 ≥ 0.70 on at least 30 held-out prompts
and a clear improvement over metadata-only search. This is a proposed target to
revisit after labels, not a claimed result. Exit also requires a failure taxonomy
(mood, instrumentation, vocals, structure, timing, or candidate recall).

## M3 — Parse creative language into constraints

Entry: M2 failure examples. Deliver schema-constrained LLM parser, original-prompt
fallback, hard/soft distinction, ambiguity handling, and parser evaluation fixtures.
Exit: invalid output cannot reach SQL, parser failure still permits semantic search,
and unsupported constraints are disclosed instead of represented as satisfied.
Risks: hallucinated fields, negation errors, latency, and API costs. A parsed event
cannot be enforced until an evaluated detector exists.

## M4 — Evaluate richer music representations

Entry: M2 benchmark. Deliver versioned MERT experiments, music feature/tag candidates,
and ablations against CLAP alone. Test mood, genre, instruments, vocals, and
reference-audio similarity only where supported by a trained downstream method.
Exit: promote a representation only if it improves held-out relevance enough to
justify compute/storage. Do not compare raw MERT vectors to CLAP text vectors.

## M5 — Structure, events, and advanced prompts

Entry: validated query contracts and useful M4 features. Deliver energy trajectories,
onset/bass/loudness changes, event candidates, confidence, section boundaries, and
temporal filters. Later add tapped rhythm, humming, or authorized reference clips
as separately evaluated inputs.

Exit: report event precision/recall and timestamp error on labeled transitions;
demonstrate a restrained intro followed by a drop in a requested interval. Evaluate
negative constraints and whole-track coverage separately. An RMS jump alone must
not be treated as proof of a drop. Risks: genre dependence and uncertain boundaries.

## M6 — Canonical identity and official metadata connectors

Entry: M2 retrieval demonstrated. Deliver track/source/asset separation and explicit
metadata/audio capabilities, then Spotify and YouTube official metadata adapters.
Exit: metadata-only records cannot enqueue analysis, duplicate-source matching is
auditable, credentials/rate limits/errors are handled, and source links are verified.
No scraping or stream extraction. Additional catalogs require individual supported
access decisions; licensing data remains excluded.

## M7 — API and deployment evidence

Entry: stable search/result DTOs. Deliver FastAPI adapter, authenticated requests,
versioned OpenAPI contract, queue/worker design, and a measured hosting spike.
Benchmark resident inference versus function execution, package/model size, cold
start, memory, latency, concurrency, and cost. Choose backend hosting using evidence
and current Vercel capabilities; the website target is Vercel.

Exit: staging search works through the API; analysis jobs can retry safely;
preview/staging/production data are separated; secrets and local paths do not leak
to frontend DTOs. Python remains the default backend and backend-test language.

## M8 — Website hosted on Vercel

Entry: M6 source links and M7 staging API. Deliver a Next.js/TypeScript website with
prompt entry, ranked cards, matching moments, evidence, loading/error/empty states,
and outbound source actions. Playback only uses supported, authorized mechanisms.
Exit: a Vercel preview passes component, API-contract, browser, accessibility, and
responsive-layout tests. Production release is a separate deliberate deployment.
No licensing, checkout, or DAW workflows are added.

## M9 — Relevance feedback and operations

Entry: users exercising M8. Deliver explicit relevance labels, prompt-positive-
negative examples, ranking experiments, privacy/retention choices, backups, monitoring,
and cost/performance reporting. Fine-tuning follows a sufficiently large, authorized
annotation dataset. Exit: held-out gains are repeatable, regressions can be rolled
back, and operational targets are based on observed usage rather than guesses.
