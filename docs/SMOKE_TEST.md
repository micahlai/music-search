# Live database and audio-small smoke test

Executed 2026-09-09 after local PostgreSQL setup. Audio files were read from the
operator-provided `./audio-small/` folder. No files were acquired from platforms.

## Environment

- PostgreSQL 17.11 (Homebrew), pgvector 0.8.6, localhost port 5432.
- Application database `music_search`; separate test database `music_search_test`.
- Migration `0001_initial_schema` at head, including the HNSW cosine index.
- Real `laion/clap-htsat-unfused` pretrained weights and processor.
- Cached Hugging Face revision `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a`.
- Apple MPS device selected by automatic device detection.
- Default 48 kHz mono, 10-second windows, 5-second stride, batch size 8.

## Results

| Check | Observed result |
| --- | --- |
| Full pytest suite with live test database enabled | **53 passed**, 0 skipped, 4.62 seconds |
| Ruff | Passed |
| Strict mypy | Passed, 18 source files |
| Discovered/decoded audio | 20 MP3 files, 4,457.2645 seconds (74.29 minutes) |
| First ingestion | 20 ingested, 0 skipped, 0 failed |
| Segments stored | 881, matching the expected window count |
| Vector dimensions | All 512 |
| Vector norms | 0.9999999709 to 1.0000000298 |
| Invalid timestamps/start strides | 0 |
| Second ingestion | 0 ingested, 20 skipped, 0 failed |
| Live test schema cleanup | No leftover test schemas |
| CLI JSON search | Returned five ranked, unique tracks with timestamps/features |
| Additional real-model queries | Three queries passed uniqueness, finite-score, and ordering checks |

The live integration test exercised migration up/down, content lookup, atomic
replacement, model compatibility filtering, cosine search, simulated mid-write
failure with rollback, and cascade deletion. Its temporary schema was removed from
the separate test database; the 20-track application catalog remains available.

## Sample search observations

These are observed rankings, not human relevance judgments. Similarity is a cosine
score, not a percentage or probability.

| Prompt | Top returned track | Best moment | Similarity |
| --- | --- | --- | ---: |
| warm acoustic guitar with male vocals | Cardinal Bloom — To Love Someone | 0:30–0:40 | 0.5790 |
| energetic indie rock with electric guitar and drums | Cardinal Bloom — To Love Someone | 4:10–4:20 | 0.6921 |
| gentle piano with an intimate male vocal | Cardinal Bloom — To Love Someone | 0:20–0:30 | 0.4032 |

Additional returned tracks included Bird and Byron — I'll Always Be There, Ax and
the Hatchetmen — Utah, Cardinal Bloom — She’s Just a Friend, and Bird and Byron —
In the Clear. Each query returned five unique tracks in descending score order.

In a process with the model already loaded, the three measured search calls took
0.266, 0.056, and 0.020 seconds (query encoding plus database/ranking). These are
single-run smoke measurements on 20 tracks, exclude model load time, and are not
a latency benchmark or service commitment.

The same track ranked first for different prompts at different moments. The current
best-segment scoring can favor tracks with more matching opportunities. Whether
these results are creatively useful requires listening and labels in M2.

## Reproduce

With the native database service running, from the project root:

```bash
MUSIC_SEARCH_TEST_DATABASE_URL=postgresql+psycopg://music_search:music_search@localhost:5432/music_search_test uv run pytest -q
uv run music-search ingest ./audio-small/
uv run music-search search "warm acoustic guitar with male vocals" --limit 5 --json
```

The existing 20 files will be skipped unless `--force` is supplied. The real audio
run emitted a Transformers deprecation warning for the still-supported `audios`
processor argument; it did not cause failures. A future Transformers upgrade should
migrate that argument and its compatibility tests together.

## Remaining work

M1's local runtime gates are complete. M2 still needs the larger authorized catalog,
human relevance labels, hard negatives, baseline comparisons, and repeatable quality
metrics. Whole-song structure, negative constraints such as “no vocals,” and beat-drop
timing filters remain future functionality.
