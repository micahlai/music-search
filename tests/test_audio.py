from pathlib import Path

import numpy as np
import pytest

from music_search.audio import AudioData, decode_audio, discover_audio_files, split_audio


def test_decode_rejects_nonfinite_pcm(tmp_path: Path) -> None:
    import soundfile as sf

    path = tmp_path / "nonfinite.wav"
    sf.write(path, np.array([0.0, np.nan], dtype=np.float32), 48000, subtype="FLOAT")
    with pytest.raises(ValueError, match="non-finite"):
        decode_audio(path)


def test_discover_audio_files_is_recursive_case_insensitive_and_sorted(
    tmp_path: Path,
) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    expected = [tmp_path / "A.WAV", nested / "b.flac", nested / "C.Mp3"]
    for path in [*expected, nested / "notes.txt"]:
        path.touch()

    discovered = discover_audio_files(tmp_path)

    assert discovered == sorted(expected, key=lambda path: path.as_posix().casefold())
    assert discover_audio_files(expected[0]) == [expected[0]]
    assert discover_audio_files(nested / "notes.txt") == []


def test_discover_audio_files_accepts_extensions_without_a_dot(tmp_path: Path) -> None:
    wav = tmp_path / "track.wav"
    flac = tmp_path / "track.flac"
    wav.touch()
    flac.touch()

    assert discover_audio_files(tmp_path, extensions={"WAV"}) == [wav]


def test_decode_audio_downmixes_stereo_to_contiguous_float32(tmp_path: Path) -> None:
    soundfile = pytest.importorskip("soundfile")
    sample_rate = 8_000
    left = np.linspace(-0.8, 0.8, 80, dtype=np.float32)
    right = np.linspace(0.2, -0.2, 80, dtype=np.float32)
    stereo = np.column_stack((left, right))
    path = tmp_path / "stereo.wav"
    soundfile.write(path, stereo, sample_rate, subtype="FLOAT")

    audio = decode_audio(path, target_sample_rate=sample_rate)

    assert isinstance(audio, AudioData)
    assert audio.path == path
    assert audio.sample_rate == sample_rate
    assert audio.samples.dtype == np.float32
    assert audio.samples.ndim == 1
    assert audio.samples.flags.c_contiguous
    np.testing.assert_allclose(audio.samples, stereo.mean(axis=1), atol=1e-6)
    assert audio.duration_seconds == pytest.approx(0.01)


def test_decode_audio_resamples_to_requested_rate(tmp_path: Path) -> None:
    soundfile = pytest.importorskip("soundfile")
    source_rate = 8_000
    target_rate = 16_000
    duration_seconds = 0.05
    times = np.arange(int(source_rate * duration_seconds), dtype=np.float32) / source_rate
    samples = np.sin(2 * np.pi * 440 * times).astype(np.float32)
    path = tmp_path / "tone.wav"
    soundfile.write(path, samples, source_rate, subtype="FLOAT")

    audio = decode_audio(path, target_sample_rate=target_rate)

    assert audio.sample_rate == target_rate
    assert len(audio.samples) == int(target_rate * duration_seconds)
    assert audio.duration_seconds == pytest.approx(duration_seconds)


def test_split_audio_overlaps_and_pads_only_the_final_window() -> None:
    samples = np.arange(14, dtype=np.float32)

    windows = split_audio(
        samples,
        sample_rate=2,
        window_seconds=4.0,
        stride_seconds=2.0,
    )

    assert [window.index for window in windows] == [0, 1, 2]
    assert [window.start_seconds for window in windows] == [0.0, 2.0, 4.0]
    assert [window.end_seconds for window in windows] == [4.0, 6.0, 7.0]
    assert [window.valid_sample_count for window in windows] == [8, 8, 6]
    assert [window.is_padded for window in windows] == [False, False, True]
    np.testing.assert_array_equal(windows[0].samples, np.arange(8, dtype=np.float32))
    np.testing.assert_array_equal(windows[1].samples, np.arange(4, 12, dtype=np.float32))
    np.testing.assert_array_equal(
        windows[2].samples,
        np.concatenate((np.arange(8, 14, dtype=np.float32), np.zeros(2, dtype=np.float32))),
    )


def test_split_audio_does_not_add_a_tail_after_an_exact_full_window() -> None:
    windows = split_audio(
        np.arange(8, dtype=np.float32),
        sample_rate=2,
        window_seconds=4.0,
        stride_seconds=2.0,
    )

    assert len(windows) == 1
    assert not windows[0].is_padded


@pytest.mark.parametrize(
    ("samples", "sample_rate", "window_seconds", "stride_seconds"),
    [
        (np.zeros(4, dtype=np.float32), 0, 1.0, 1.0),
        (np.zeros(4, dtype=np.float32), 4, 0.0, 1.0),
        (np.zeros(4, dtype=np.float32), 4, 1.0, 0.0),
        (np.zeros((2, 2), dtype=np.float32), 4, 1.0, 1.0),
    ],
)
def test_split_audio_rejects_invalid_inputs(
    samples: np.ndarray,
    sample_rate: int,
    window_seconds: float,
    stride_seconds: float,
) -> None:
    with pytest.raises(ValueError):
        split_audio(samples, sample_rate, window_seconds, stride_seconds)
