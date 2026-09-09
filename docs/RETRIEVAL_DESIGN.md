# Retrieval design

Measured diagnosis of V1 CLAP retrieval quality and the design that follows from it.
Every number below was produced against the local catalog described in
[measurement method](#measurement-method). Sample sizes are small; treat the
direction as established and the magnitudes as provisional.

## Diagnosis

Coarse retrieval works better than it feels. "aggressive fast punk drums" returns
the punk section of a Backseat Lovers track; "solo acoustic piano, sad and slow"
returns piano tracks. Recall is not the primary defect.

The primary defect is that **similarity scores are not comparable across queries**,
so a query with no valid answer is indistinguishable from one with a good answer.
Raw cosine maxima over 22,415 indexed segments:

| max cosine | query | has a real answer in catalog |
| --- | --- | --- |
| 0.541 | traditional indian sitar and tabla raga | no |
| 0.511 | eerie ambiance that slowly transitions to a dreamy tone | yes |

A nonsense control outranks a genuine query. There is no threshold that separates
a match from a fabrication, so every search returns a full page of confident
results. This is the behavior reported as hallucinative.

Two hypotheses were tested and rejected:

- **Hubness.** Unrelated queries share only 0.8 of their top-50 segments on
  average. No small set of segments dominates the index.
- **Embedding corruption.** Vectors are L2-normalized, finite, and
  well-distributed. The write path is sound.

One secondary defect is real. With 10-second windows at 5-second stride, the top
50 segments collapse to only 12–27 distinct tracks, because each musical moment
appears 2–4 times. Roughly half the candidate budget is spent on duplicates of
the same moment.

A third limitation is structural rather than a defect. Compositional queries are
not representable as one vector. In "lot of synth maj9 over house beat" the
harmony term contributes nothing measurable to ranking; "house beat" carries the
result. CLAP has no chord vocabulary, no duration vocabulary, and no way to
express a transition between two states.

## Ablation

Eight queries were labeled with a known correct track and evaluated against an
18-track subset re-embedded with both CLAP checkpoints.

| configuration | mean rank | MRR | top-1 |
| --- | --- | --- | --- |
| V1 baseline (general checkpoint, max pooling) | 4.88 | 0.374 | 1/8 |
| + top-3 mean pooling | 5.12 | 0.457 | 2/8 |
| + background z-scoring | 4.75 | 0.540 | 3/8 |
| + prompt ensembling | 3.75 | 0.586 | 3/8 |
| music checkpoint alone | 3.12 | 0.532 | 2/8 |
| music checkpoint + ensembling + z-scoring | **1.88** | **0.740** | **5/8** |

### Checkpoint

`laion/clap-htsat-unfused` is trained predominantly on AudioSet-style general
audio. It discriminates dogs and jet engines, not chord voicings.
`laion/larger_clap_music_and_speech` raises MRR from 0.374 to 0.532 on its own
and improves score separation: the sitar control falls from 0.527 to 0.379 while
real queries hold near 0.57. This is a configuration change plus a re-ingest.

### Background z-scoring

Embed a fixed bank of roughly 30 generic prompts ("rock music", "piano ballad",
"silence") once. At ingest, store each segment's mean and standard deviation
against that bank as two float16 values, **4 bytes per segment**. At query time
score `(cosine - mean) / stdev`.

| query class | raw max | z max |
| --- | --- | --- |
| answerable | 0.51–0.68 | +2.9 to +4.1 |
| nonsense control | 0.29–0.54 | +1.7 to +2.1 |

The z-score yields a usable "no confident match" threshold and is the single
largest precision contributor in the ablation. Note that it does not repair every
case: "sitar and tabla" still scores high against a guitar-heavy track, because
the underlying confusion is acoustic rather than statistical.

### Prompt ensembling

Averaging the unit vectors of 3–4 paraphrases of a query clause improves ranking
at no storage cost. The query parser should emit paraphrases as part of its
output rather than leaving generation to call sites.

### Pooling

Ranking a track by the mean of its top-3 segment scores rather than by its single
best segment reduces the chance that one lucky window promotes an otherwise
unrelated track. Adjacent segments should be deduplicated before pooling so that
overlapping windows of one moment do not count three times.

## Query decomposition

`parse_query` is currently a pass-through and is the highest-leverage unimplemented
component. Encoding "eerie ambiance that slowly transitions to a dreamy tone" as a
single vector averages two distinct states into a description of neither.

Decomposing into clauses and scoring `min(sim_A[t], sim_B[t + delta])` across each
track's segment timeline recovers the intended structure:

```text
+0.471  Terekke - unother       eerie@15s  -> dreamy@20s
+0.454  Edgehill - Lookaround   dist@10s   -> clean@15s
```

The parser should emit a structured plan rather than a string: weighted semantic
clauses with paraphrases, hard filters, transition constraints with maximum gaps,
and duration constraints. A hybrid scorer then combines the CLAP score with the
tag, harmony, and sequence layers below, and each contributing term becomes one
line of the existing `RankedTrack.explanation` field.

Requests that name no implemented detector must be reported as unsupported rather
than silently reduced to semantic similarity.

## Analysis layers

CLAP cannot answer harmonic, durational, or production-detail queries at any
model size. These require explicit features, all of which are cheap to compute
and to store.

| layer | contents | bytes/section |
| --- | --- | --- |
| Harmony | key, mode, chord-quality histogram (maj/min/dom7/maj7/add9-ext/sus/dim), harmonic rhythm, mean chroma | ~30 |
| Tags | genre, mood, and instrument probabilities over fixed vocabularies | ~200 |
| Timbre | spectral flatness, crest factor, odd/even harmonic ratio, spectral contrast | ~16 |
| Calibration | background mean and stdev | 4 |

**Harmony** comes from beat-synchronous CQT chroma and chord recognition
(`madmom` or Essentia). It converts "maj9" from an ignored token into an exact
filter.

**Tags** come from pretrained auto-tagging heads such as Essentia's
`discogs-effnet` family. They are small ONNX models, fast on CPU, and
interpretable: "synth over house" becomes a SQL predicate over named columns
rather than a fuzzy vector match, which is both more accurate and directly
explainable.

**Timbre** descriptors make distortion versus clean tone measurable rather than
inferred. Harmonic spreading and crest factor separate the two reliably.

**Structure.** Replace fixed 10s/5s windows with novelty-based section boundaries
(`librosa.segment`). This yields roughly 12 musical sections instead of 42
windows, which simultaneously eliminates duplicate-window flooding, supplies real
section durations for queries like "a solo around 20 seconds long", and reduces
storage 3.5x.

**MERT.** Do not persist MERT vectors; they are too large for the deployment
budget. Use MERT at ingest as a feature extractor for small probe heads (key,
chord, instrument, mood) and persist only the probe outputs. This is how
music-specific representation learning fits inside the size constraint, and it
supersedes the placeholder in `mert.py`.

## Storage

Anticipated cloud deployment constrains per-track footprint. Measured over all
22,415 segments, with fp32 512-dimensional ranking as ground truth:

| scheme | bytes/vector | recall@20 |
| --- | --- | --- |
| fp32 512d (current) | 2048 | 1.000 |
| float16 512d | 1024 | 1.000 |
| int8 512d | 516 | 0.975 |
| PCA 256d int8 | 260 | 0.617 |
| PCA 128d int8 | 132 | 0.600 |
| binary 512-bit, top-200 candidates | 64 | 0.983 |

**Dimensionality reduction is not viable.** PCA to 256 dimensions retains 99.68%
of variance and still loses 40% of recall. CLAP's discriminative signal lives in
low-variance directions, so variance-preserving projection discards exactly the
information retrieval depends on. This result is counterintuitive enough to be
worth re-verifying before any future attempt.

The viable design is two-tier retrieval, both tiers natively supported by the
installed pgvector 0.8.6:

1. `bit(512)` with Hamming distance for candidate generation — 64 bytes/vector,
   98.3% of true top-20 recovered within 200 candidates.
2. `halfvec` or int8 rerank over those candidates — 97.5% recall.

Combined recall is approximately 0.96 at a fraction of current storage.

Per 3.5-minute track, 12 sections at roughly 64 + 516 + 250 auxiliary bytes is
about **10 KB, against 86 KB today**, with substantially more expressive power.
At one million tracks that is 10 GB rather than 86 GB.

## Sequenced work

1. Switch to `laion/larger_clap_music_and_speech` and re-ingest.
2. Add background z-scoring at ingest and a "no confident match" threshold in
   `SearchService`.
3. Deduplicate adjacent segments and switch to top-3 mean pooling.
4. Replace fixed windows with structural segmentation.
5. Implement `parse_query`: clauses, paraphrases, filters, transitions, durations.
6. Add tag, harmony, and timbre layers with hybrid scoring and per-term evidence.
7. Move to two-tier binary plus int8 storage before cloud deployment.

Steps 1–3 are approximately one day of work and account for most of the ablated
gain.

## Measurement method

Catalog: 531 tracks, 22,415 segments, the local PostgreSQL instance.

- Score-calibration and hubness figures use all 22,415 segments against 6
  answerable queries and 4 nonsense controls.
- The checkpoint A/B and configuration ablation use 8 queries labeled with a
  known correct track, over an 18-track subset re-embedded under both
  checkpoints. Tracks are collapsed by their best segment unless pooling is
  under test.
- Quantization figures use 6 queries over all segments, scoring agreement with
  fp32 ranking.

Both caveats matter. Eight labeled queries is too few to trust any individual
number, and quantization recall measures agreement with fp32 ranking rather than
human relevance. Building the 100–1,000-track evaluation corpus with roughly 50
labeled queries, already item 2 in the [roadmap](ROADMAP.md), should precede
acting on any single figure here.
