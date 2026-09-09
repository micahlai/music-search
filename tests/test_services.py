"""Regression tests for persistence boundaries and ranking without external services."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID

import numpy as np
import pytest

from music_search.audio import AudioData
from music_search.db import SegmentMatch
from music_search.features import DSPFeatures
from music_search.ingestion import IngestionService, content_sha256, infer_title_artist
from music_search.search import SearchService


def test_hash_and_filename_metadata(tmp_path: Path) -> None:
    path = tmp_path / "Artist - Track.wav"
    path.write_bytes(b"abc")
    assert content_sha256(path, chunk_size=1) == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    assert infer_title_artist(path) == ("Track", "Artist")
    assert infer_title_artist(path, "Override") == ("Track", "Override")


def test_duplicate_skip_then_force_replaces_complete_tail(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "Track.wav"
    path.write_bytes(b"synthetic")
    repository = Mock()
    repository.find_track_by_content_hash.return_value = SimpleNamespace(
        id=UUID(int=1), segments=[]
    )
    embedder = Mock(model_name="test-clap")
    decode = Mock(return_value=AudioData(np.ones(24, dtype=np.float32), 2))
    monkeypatch.setattr("music_search.ingestion.decode_audio", decode)
    measured_lengths = []

    def measure(samples, sample_rate):
        measured_lengths.append(len(samples))
        return DSPFeatures(rms_energy=1.0, bpm=0.0)

    monkeypatch.setattr("music_search.ingestion.extract_dsp_features", measure)
    embedder.embed_audio.return_value = np.ones((2, 512), dtype=np.float32)
    repository.replace_track.side_effect = lambda track: SimpleNamespace(
        id=UUID(int=1), segments=track.segments
    )
    progress: list[str] = []
    service = IngestionService(repository, embedder, sample_rate=2, on_progress=progress.append)
    assert service.ingest_file(path).status == "skipped"
    decode.assert_not_called()
    embedder.embed_audio.assert_not_called()
    assert any("Duplicate SHA-256" in message for message in progress)
    assert not any("Decoding" in message for message in progress)

    progress.clear()
    assert service.ingest_file(path, force=True).segment_count == 2
    written = repository.replace_track.call_args.args[0]
    assert written.bpm is None
    assert written.channels == 1
    assert written.source_uri == path.as_uri()
    assert [(s.start_seconds, s.end_seconds) for s in written.segments] == [(0, 10), (5, 12)]
    assert measured_lengths == [20, 14, 24]  # tail DSP must exclude model-input padding
    assert all(s.bpm is None for s in written.segments)
    assert any("Created 2 windows" in message for message in progress)
    assert any("DSP: 2/2" in message for message in progress)
    assert "Saved track" in progress[-1]
    assert embedder.embed_audio.call_args.kwargs["on_progress"] == progress.append


def test_corrupt_file_isolation_and_fail_fast(tmp_path: Path, monkeypatch) -> None:
    for name in ("a.wav", "b.wav"):
        (tmp_path / name).touch()
    service = IngestionService(Mock(), Mock())
    ingest = Mock(side_effect=ValueError("bad codec"))
    monkeypatch.setattr(service, "ingest_file", ingest)
    assert service.ingest_path(tmp_path).failed_count == 2
    assert service.ingest_path(tmp_path, fail_fast=True).failed_count == 1


def test_embedding_failure_never_writes_partial_track(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "bad.wav"
    path.write_bytes(b"test")
    repository = Mock()
    repository.find_track_by_content_hash.return_value = None
    embedder = Mock()
    embedder.embed_audio.return_value = np.zeros((0, 512), dtype=np.float32)
    monkeypatch.setattr(
        "music_search.ingestion.decode_audio",
        lambda *args, **kwargs: AudioData(np.ones(24, dtype=np.float32), 2),
    )
    with pytest.raises(RuntimeError, match="batch shape"):
        IngestionService(repository, embedder).ingest_file(path)
    repository.replace_track.assert_not_called()


def _match(track: int, score: float, **kwargs) -> SegmentMatch:
    base = SegmentMatch(
        segment_id=UUID(int=track + 100),
        segment_index=0,
        start_seconds=0,
        end_seconds=10,
        rms_energy=0.2,
        segment_bpm=None,
        track_id=UUID(int=track),
        source_uri=f"file:///track{track}.wav",
        content_sha256="a" * 64,
        title=f"Track {track}",
        artist=None,
        album=None,
        track_bpm=120,
        cosine_distance=1 - score,
        cosine_similarity=score,
    )
    return replace(base, **kwargs)


def test_search_groups_tracks_and_labels_bpm_provenance() -> None:
    repository = Mock()
    repository.search_nearest_segments.return_value = [
        _match(1, 0.7),
        _match(2, 0.8),
        _match(1, 0.9, start_seconds=5, end_seconds=15, segment_bpm=100),
    ]
    embedder = Mock(model_name="test-clap")
    embedder.embed_text.return_value = np.ones(512, dtype=np.float32)
    results = SearchService(repository, embedder).search(" dreamy guitar ", limit=2)
    assert [r.track_id for r in results] == [UUID(int=1), UUID(int=2)]
    assert results[0].best_start_seconds == 5
    assert results[0].score == 0.9
    assert results[0].bpm_source == "segment"
    assert results[1].bpm_source == "track"
    assert "(track)" in results[1].explanation
    assert results[0].to_dict()["track_id"] == str(UUID(int=1))
    embedder.embed_text.assert_called_once_with("dreamy guitar")
    assert repository.search_nearest_segments.call_args.kwargs == {
        "limit": 50,
        "embedding_model": "test-clap",
    }


def test_search_blank_rejected_before_inference_and_ties_stable() -> None:
    repository, embedder = Mock(), Mock(model_name="test-clap")
    service = SearchService(repository, embedder)
    with pytest.raises(ValueError, match="empty"):
        service.search(" ")
    embedder.embed_text.assert_not_called()
    embedder.embed_text.return_value = np.ones(512)
    matches = [_match(2, 0.8), _match(1, 0.8)]
    repository.search_nearest_segments.return_value = matches
    first = service.search("piano")
    repository.search_nearest_segments.return_value = list(reversed(matches))
    assert first == service.search("piano")
