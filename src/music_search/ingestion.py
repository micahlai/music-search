"""Orchestration for analyzing authorized local audio and persisting it."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal, Protocol
from uuid import UUID

import numpy as np
from numpy.typing import NDArray

from music_search.audio import decode_audio, discover_audio_files, split_audio
from music_search.db import SegmentWrite, TrackRecord, TrackWrite
from music_search.features import extract_dsp_features

ANALYSIS_VERSION = "local-clap-v1"


class AudioEmbedder(Protocol):
    """Minimal CLAP interface used by the ingestion pipeline."""

    model_name: str

    def embed_audio(
        self,
        audios: Sequence[NDArray[np.float32]],
        sample_rate: int,
    ) -> NDArray[np.float32]: ...


class TrackRepository(Protocol):
    """Persistence operations required by ingestion."""

    def find_track_by_content_hash(self, content_sha256: str) -> TrackRecord | None: ...

    def replace_track(self, track: TrackWrite) -> TrackRecord: ...


@dataclass(frozen=True, slots=True)
class FileIngestionResult:
    """Outcome for one discovered file."""

    path: Path
    status: Literal["ingested", "skipped", "failed"]
    track_id: UUID | None = None
    segment_count: int = 0
    error: str | None = None


@dataclass(frozen=True, slots=True)
class IngestionSummary:
    """Aggregate result for one file-or-folder ingestion request."""

    results: tuple[FileIngestionResult, ...]

    @property
    def discovered_count(self) -> int:
        return len(self.results)

    @property
    def ingested_count(self) -> int:
        return sum(result.status == "ingested" for result in self.results)

    @property
    def skipped_count(self) -> int:
        return sum(result.status == "skipped" for result in self.results)

    @property
    def failed_count(self) -> int:
        return sum(result.status == "failed" for result in self.results)


def content_sha256(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash original file bytes without loading the entire file into memory."""

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")

    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def infer_title_artist(path: Path, artist_override: str | None = None) -> tuple[str, str | None]:
    """Infer lightweight metadata from `Artist - Title.ext` or the filename stem."""

    stem = path.stem.strip()
    if not stem:
        raise ValueError(f"Cannot infer a title from filename: {path.name}")

    inferred_artist: str | None = None
    title = stem
    if " - " in stem:
        candidate_artist, candidate_title = stem.split(" - ", 1)
        if candidate_artist.strip() and candidate_title.strip():
            inferred_artist = candidate_artist.strip()
            title = candidate_title.strip()

    override = artist_override.strip() if artist_override is not None else None
    return title, override or inferred_artist


def _persisted_bpm(value: float) -> float | None:
    """Translate the DSP indeterminate sentinel into the database's unknown value."""

    return value if value > 0 else None


class IngestionService:
    """Analyze files one at a time, committing each complete track atomically."""

    def __init__(
        self,
        repository: TrackRepository,
        embedder: AudioEmbedder,
        *,
        sample_rate: int = 48_000,
        window_seconds: float = 10.0,
        stride_seconds: float = 5.0,
        analysis_version: str = ANALYSIS_VERSION,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if window_seconds <= 0 or stride_seconds <= 0:
            raise ValueError("window_seconds and stride_seconds must be positive")
        if stride_seconds > window_seconds:
            raise ValueError("stride_seconds must not exceed window_seconds")
        if not analysis_version.strip():
            raise ValueError("analysis_version must not be empty")

        self.repository = repository
        self.embedder = embedder
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self.stride_seconds = stride_seconds
        self.analysis_version = analysis_version

    def ingest_path(
        self,
        path: str | Path,
        *,
        artist: str | None = None,
        force: bool = False,
        fail_fast: bool = False,
    ) -> IngestionSummary:
        """Discover and ingest files, isolating normal failures to one file."""

        files = discover_audio_files(path)
        results: list[FileIngestionResult] = []
        for audio_path in files:
            try:
                results.append(self.ingest_file(audio_path, artist=artist, force=force))
            except Exception as exc:  # one bad asset should not discard its siblings
                results.append(
                    FileIngestionResult(
                        path=audio_path,
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                if fail_fast:
                    break
        return IngestionSummary(tuple(results))

    def ingest_file(
        self,
        path: str | Path,
        *,
        artist: str | None = None,
        force: bool = False,
    ) -> FileIngestionResult:
        """Analyze and persist one local file known to be authorized by the operator."""

        audio_path = Path(path).expanduser().resolve()
        digest = content_sha256(audio_path)
        existing = self.repository.find_track_by_content_hash(digest)
        if existing is not None and not force:
            return FileIngestionResult(
                path=audio_path,
                status="skipped",
                track_id=existing.id,
                segment_count=len(existing.segments),
            )

        audio = decode_audio(audio_path, target_sample_rate=self.sample_rate)
        if audio.samples.size == 0:
            raise ValueError("decoded audio is empty")

        windows = split_audio(
            audio.samples,
            audio.sample_rate,
            window_seconds=self.window_seconds,
            stride_seconds=self.stride_seconds,
        )
        if not windows:
            raise ValueError("audio produced no analysis windows")

        vectors = self.embedder.embed_audio(
            [window.samples for window in windows],
            sample_rate=audio.sample_rate,
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(windows):
            raise RuntimeError(
                "embedder returned an unexpected batch shape: "
                f"expected {len(windows)} rows, got {vectors.shape}"
            )

        segments: list[SegmentWrite] = []
        for window, vector in zip(windows, vectors, strict=True):
            valid_samples = window.samples[: window.valid_sample_count]
            features = extract_dsp_features(valid_samples, audio.sample_rate)
            segments.append(
                SegmentWrite(
                    segment_index=window.index,
                    start_seconds=window.start_seconds,
                    end_seconds=window.end_seconds,
                    rms_energy=features.rms_energy,
                    bpm=_persisted_bpm(features.bpm),
                    embedding=vector.tolist(),
                )
            )

        track_features = extract_dsp_features(audio.samples, audio.sample_rate)
        title, inferred_artist = infer_title_artist(audio_path, artist)
        stored = self.repository.replace_track(
            TrackWrite(
                source_uri=audio_path.as_uri(),
                content_sha256=digest,
                title=title,
                artist=inferred_artist,
                album=None,
                duration_seconds=audio.duration_seconds,
                sample_rate=audio.sample_rate,
                channels=1,
                bpm=_persisted_bpm(track_features.bpm),
                embedding_model=self.embedder.model_name,
                analysis_version=self.analysis_version,
                metadata={
                    "analysis_source": "authorized_local_file",
                    "original_filename": audio_path.name,
                    "window_seconds": self.window_seconds,
                    "stride_seconds": self.stride_seconds,
                },
                segments=segments,
            )
        )
        return FileIngestionResult(
            path=audio_path,
            status="ingested",
            track_id=stored.id,
            segment_count=len(stored.segments),
        )
