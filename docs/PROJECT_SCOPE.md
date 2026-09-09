# Project scope

## Product definition

This project is an AI-assisted **music discovery and retrieval engine** for
filmmakers, editors, and other people searching for existing music with creative
language. It is not a music licensing marketplace, media downloader, or audio
generation product.

The product answers questions such as:

- “Ambient synth with a huge cinematic build but no vocals.”
- “A beat drop that happens around 45 seconds.”
- “Music that feels like driving through Tokyo at night.”
- “Slow emotional piano that becomes orchestral.”
- “Something hopeful without sounding triumphant.”

The long-term product ends when a user opens a result on its original source.
How that user listens, acquires, negotiates, or licenses the track is outside this
project.

The intended product surface is a website hosted on Vercel. That website is a
post-prototype milestone, not part of V1. It will provide the search and result
experience while heavyweight Python audio decoding, CLAP inference, and catalog
indexing run behind a separately deployed worker/service boundary unless a later
deployment spike demonstrates that an appropriate Vercel runtime can satisfy the
model, memory, execution-time, and cost requirements.

## Product promise

> Let a filmmaker describe mood, instrumentation, rhythm, structure, transitions,
> and emotional direction in ordinary language, then return existing tracks whose
> measured audio and source metadata best fit that description.

The first proof is intentionally smaller: given a collection of authorized local
audio files, retrieve the right tracks for unusual natural-language prompts that
do not merely repeat file names or tags.

## Primary users and jobs

### Filmmakers and editors

- Find music for a scene before knowing an artist, title, or genre label.
- Search for a usable moment, not only for a whole-song category.
- Locate transitions, builds, drops, sparse passages, and changes in energy.
- Understand why a result matched before opening its source.

### Music supervisors and creative researchers

- Explore a permitted catalog with consistent language across sources.
- Compare candidates using semantic, musical, and structural evidence.
- Save time spent translating a creative brief into rigid catalog tags.

### Catalog partners and participating artists (later)

- Attach supported source metadata to audio they have authorized for analysis.
- Make tracks discoverable through descriptions that are richer than supplied tags.

## In scope

### Discovery inputs

- Natural-language descriptions of mood, genre, instrumentation, sonic texture,
  energy, tempo, arrangement, events, intended scene, and emotional progression.
- Later: a tapped rhythm, hummed or imitated phrase, and an authorized short
  reference clip for rhythm/audio-to-audio retrieval.
- Later: structured constraints parsed from language, such as a transition between
  45 and 60 seconds or no vocals before a specified moment.

### Searchable representation

- Audio-text embeddings that place a prompt and audio window in a shared vector
  space. CLAP is the V1 model family.
- Overlapping time windows so important moments are not lost in a whole-track
  average.
- Measured DSP features, beginning with RMS energy and estimated BPM.
- Later: MERT or another music-specific representation; track/section/event-level
  embeddings; onset, spectral, loudness, key, vocals, instrumentation, and energy
  trajectory features.
- Source-supplied metadata such as title, artist, duration, source ID, source URL,
  thumbnail, and public tags.

### Retrieval and ranking

- V1 cosine-nearest-neighbor retrieval over CLAP segment embeddings.
- Track ranking based on the best matching segment, with supporting timestamps and
  measured features available to explain the result.
- Later hybrid retrieval across audio vectors, music vectors, text metadata,
  rhythm patterns, and structural events.
- Deterministic filtering and reranking from validated query constraints.
- Grounded result explanations derived from stored evidence.

### Source integration

- V1: files already present in a user-provided local folder and authorized for
  analysis.
- Later: separately implemented source connectors that normalize supported public
  metadata and outbound links.
- Spotify and YouTube are planned as metadata/link connectors only. They are not
  audio acquisition mechanisms.
- Possible later sources include participating artists, partner catalogs,
  Musicbed, Artlist, Epidemic Sound, Pixabay, Fesliyan Studios, SoundCloud, and
  Bandcamp, subject to each source's supported API, permissions, and terms.

### Result experience

The eventual Vercel-hosted website will accept creative-language searches and
present source-linked result cards. A mature result may contain:

- track and artist;
- match score;
- the matching time range;
- a concise, evidence-backed explanation;
- source/platform identity;
- an authorized playback mechanism, when separately supplied by that source; and
- an outbound “Open source” action.

The discovery engine does not assert that a result is licensed or suitable for a
particular use.

## V1 local prototype scope

The current milestone includes only:

1. Recursively discover supported audio files in a local folder.
2. Compute a SHA-256 content identity for idempotent ingestion.
3. Decode to floating-point PCM, downmix to mono, and resample to 48 kHz.
4. Split audio into 10-second windows with a 5-second stride.
5. Right-pad one trailing partial window while preserving its true end time.
6. Compute one normalized 512-dimensional CLAP embedding per window.
7. Compute RMS energy for each window and estimated BPM for tracks and windows.
8. Store track metadata, segment metadata, features, and vectors in PostgreSQL with
   pgvector.
9. Embed a natural-language prompt with the same CLAP model.
10. Retrieve matching segments, collapse them into ranked tracks, and display the
    best timestamp and grounded feature evidence in a CLI.

V1 has no LLM dependency. The natural-language prompt goes directly to CLAP's text
encoder. A later LLM parser will create structured constraints without replacing
the underlying retrieval evidence.

## Explicit non-goals

### Licensing and commerce

- No license verification or legal advice.
- No commercial-use determination.
- No license-price indexing, budget filters, or price comparison.
- No checkout, payments, subscription management, or brokerage.
- No guarantee that a source link confers permission to use a track.

These exclusions supersede any earlier licensing-oriented planning. Users must
evaluate terms with the original provider outside this product.

### Audio acquisition and platform circumvention

- No scraping to obtain audio.
- No Spotify or YouTube stream downloading.
- No DRM bypass, decryption, stream separation, or format circumvention.
- No pretending that public playback implies permission to analyze or redistribute.
- No ingestion of a metadata-only connector's media.

### Rights management

- No copyright ownership verification.
- No Content ID claim handling.
- No rights clearance, territory checks, or usage monitoring.

### Music creation and editing

- No generative music.
- No remixing, stem extraction, source separation, or mastering.
- No DAW or timeline integration in the scoped product.

### V1 application surface

- No website, user accounts, saved searches, collaborative projects, or public API.
- No distributed job queue, object storage, or multi-tenant catalog.
- No LLM query parser, MERT, structural-event detector, or custom model training.

These are V1 exclusions, not a rejection of the planned product surface. A
Next.js website hosted on Vercel, a service API, and separated analysis workers are
explicit later milestones.

## Source and analysis policy

Metadata access and audio-analysis access are separate capabilities. A canonical
track may have several metadata records but no analyzable waveform. Every future
connector must declare capabilities such as:

| Capability | Meaning |
| --- | --- |
| `metadata` | The connector may read and normalize supported descriptive fields. |
| `outbound_link` | The connector may provide a URL back to the original source. |
| `authorized_preview` | A specific preview is explicitly permitted for the intended analysis. |
| `authorized_full_audio` | A specific full asset is explicitly permitted for analysis. |

Only the final two capabilities can create audio-analysis work, and they must be
backed by explicit authorization. Metadata-only tracks remain searchable only from
their metadata and must be labeled accordingly in a later UI.

V1 makes this boundary concrete by accepting only local file paths supplied by the
operator. It stores derived vectors and features in PostgreSQL; it does not copy raw
audio into the database.

## Research questions

1. **Audio representation:** Which representations make “sounds like” searchable?
2. **Text-to-music alignment:** How well does a general CLAP model understand
   filmmaker language such as “bittersweet but not triumphant”?
3. **Musical structure:** Can builds, drops, transitions, instrumentation changes,
   and emotional arcs be represented and queried reliably?
4. **Ranking:** How should semantic relevance, timing, structure, instrumentation,
   and measured features be combined without opaque model judgment?
5. **Explanation:** Can every user-facing claim be traced to a retrieval score,
   metadata field, classifier output, or measured event?
6. **Deployment:** Which boundary between Vercel, the search API, PostgreSQL, asset
   storage, and heavyweight analysis workers meets latency, privacy, cost, and model
   runtime constraints without coupling web requests to audio ingestion?

## Product principles

1. **Retrieval before scale.** Prove relevance on 100–1,000 authorized tracks before
   indexing large catalogs.
2. **Moments matter.** Preserve segment and event timelines rather than reducing a
   track to one average label.
3. **Use the right intelligence.** CLAP aligns language and audio; MERT represents
   music; DSP measures signals; an LLM interprets creative language.
4. **Measured ranking, generated prose.** Models may phrase explanations, but stored
   evidence controls retrieval and factual claims.
5. **Capability-gated sources.** A metadata connector never silently becomes an
   audio downloader.
6. **Original source is the destination.** The discovery engine links out and stops.
7. **Separate interactive and batch workloads.** Keep the Vercel web experience
   responsive while authorized-audio analysis runs as an observable, retryable
   background workload.

## Success definition

The core idea is validated when, over a labeled set of 100–1,000 authorized tracks,
a user can write a filmmaker-specific prompt that is absent from title and supplied
tags and still find the intended track near the top of the results. The benchmark
must include hard negatives: tracks that share broad genre/instrumentation but have
the wrong emotional direction or arrangement.

The detailed behavioral contract is in [SPECIFICATION.md](SPECIFICATION.md), the
component boundaries are in [ARCHITECTURE.md](ARCHITECTURE.md), stage gates are in
[MILESTONES.md](MILESTONES.md), and post-MVP sequencing is in
[ROADMAP.md](ROADMAP.md).
