# Roadmap

## Product destination

A music discovery website built with Next.js and hosted on Vercel. Users describe
music, inspect matching tracks and moments, and open original sources. Python
services perform audio understanding and retrieval. Acquisition and licensing remain
outside the product. The current deliverable is a local CLI prototype.

## Next work after the starter

1. Complete live PostgreSQL and real-CLAP smoke validation, then index a small
   authorized catalog and document first-query setup cost.
2. Build the 100–1,000-track evaluation corpus before expanding sources. Record
   relevance and latency baselines, including hard negatives and moment labels.
3. Add LLM parsing with schema validation and semantic fallback. Preserve hard/soft
   intent, and identify requests that lack an implemented detector.
4. Experiment with MERT and musical features using separate versioned representations.
   Keep only improvements supported by evaluation.
5. Detect builds, drops, transitions, vocal/instrument entrances, and emotional
   trajectories with confidence and time-bound evidence. Improve hybrid ranking.
6. Separate canonical tracks, source identities, and authorized audio assets. Add
   official Spotify/YouTube metadata and outbound links without waveform downloads.
7. Wrap the Python services in an API, benchmark backend hosting, and build the
   Next.js/Vercel website. Add cross-language contract and browser tests.
8. Collect relevance feedback, improve reranking, and consider fine-tuning only when
   a labeled dataset and reproducible benchmark justify it.

## Research backlog

Possible later inputs include tapped rhythms, humming/imitations, musical references,
and authorized short reference clips. Possible source adapters include partners,
participating artists, Musicbed, Artlist, Epidemic Sound, Pixabay, Fesliyan Studios,
SoundCloud, and Bandcamp. Each is a separate access/integration decision, not a
promise of a universal crawler.

Investigate section/event embeddings, global track summaries, edit points, endings,
rhythm patterns, vocal probability, onset density, spectral descriptors, loudness,
key, and energy curves. Generated descriptions supplement numerical evidence; they
do not replace it or guarantee hard constraints.

## Deferred infrastructure

Add object storage, job queues, cache, model serving, tenant isolation, telemetry,
and alternative search stores only when workload evidence supports them. PostgreSQL
and pgvector remain the initial catalog/search store. The language split is Python
for ML/backend tests and TypeScript for the website; an API rewrite is not required
to deploy a Vercel frontend.

See [milestones](MILESTONES.md) for dependencies and exit criteria and
[architecture](ARCHITECTURE.md) for deployment decisions.
