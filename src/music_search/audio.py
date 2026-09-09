"""Utilities for reading and windowing authorized local audio files.

The functions in this module deliberately operate on decoded PCM only.  They do
not download audio or interact with streaming services.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from math import gcd
from os import PathLike
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

DEFAULT_SAMPLE_RATE = 48_000

# Formats commonly handled by libsndfile.  Actual codec availability is
# determined by the libsndfile build installed on the host.
AUDIO_EXTENSIONS = frozenset(
    {
        ".aif",
        ".aifc",
        ".aiff",
        ".au",
        ".caf",
        ".flac",
        ".mp3",
        ".oga",
        ".ogg",
        ".opus",
        ".rf64",
        ".snd",
        ".voc",
        ".w64",
        ".wav",
    }
)


@dataclass(frozen=True, slots=True)
class AudioData:
    """Decoded, mono, floating-point PCM audio."""

    samples: NDArray[np.float32]
    sample_rate: int
    path: Path | None = None

    @property
    def duration_seconds(self) -> float:
        """Duration represented by the decoded samples."""

        return len(self.samples) / self.sample_rate


@dataclass(frozen=True, slots=True)
class AudioWindow:
    """A fixed-size model input with timestamps from the unpadded audio.

    ``samples`` always has the configured window length.  ``end_seconds`` and
    ``valid_sample_count`` describe the source audio before right-padding, so a
    caller never needs to infer real duration from the model input length.
    """

    index: int
    samples: NDArray[np.float32]
    sample_rate: int
    start_seconds: float
    end_seconds: float
    valid_sample_count: int

    @property
    def is_padded(self) -> bool:
        """Whether this window contains right-padding."""

        return self.valid_sample_count < len(self.samples)


def discover_audio_files(
    root: str | PathLike[str],
    extensions: Collection[str] = AUDIO_EXTENSIONS,
) -> list[Path]:
    """Recursively find supported audio files below ``root``.

    Discovery is case-insensitive and deterministic.  A supported file may be
    supplied directly in place of a directory.
    """

    path = Path(root).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Audio path does not exist: {path}")

    normalized_extensions = {
        extension.casefold() if extension.startswith(".") else f".{extension.casefold()}"
        for extension in extensions
    }

    if path.is_file():
        return [path] if path.suffix.casefold() in normalized_extensions else []
    if not path.is_dir():
        return []

    files = (
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.casefold() in normalized_extensions
    )
    return sorted(files, key=lambda candidate: candidate.as_posix().casefold())


def decode_audio(
    path: str | PathLike[str],
    target_sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> AudioData:
    """Decode an audio file to mono float32 PCM at ``target_sample_rate``.

    ``soundfile`` performs decoding and ``scipy.signal.resample_poly`` performs
    sample-rate conversion.  Importing both inside the function keeps simple
    metadata-only commands from paying their import cost.
    """

    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be a positive integer")

    source_path = Path(path).expanduser()
    if not source_path.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {source_path}")

    import soundfile as sf  # type: ignore[import-untyped]

    decoded, source_sample_rate = sf.read(
        source_path,
        dtype="float32",
        always_2d=True,
    )
    if source_sample_rate <= 0:
        raise ValueError(
            f"Decoder returned an invalid sample rate for {source_path}: {source_sample_rate}"
        )

    decoded_array = np.asarray(decoded, dtype=np.float32)
    if decoded_array.ndim == 1:  # Defensive support for alternate sf.read wrappers.
        samples = decoded_array
    elif decoded_array.ndim == 2 and decoded_array.shape[1] > 0:
        # soundfile returns (frames, channels).  Averaging channels avoids
        # selecting one side of a stereo recording and stays in float32.
        samples = decoded_array.mean(axis=1, dtype=np.float32)
    else:
        raise ValueError(
            f"Decoder returned an invalid audio shape for {source_path}: {decoded_array.shape}"
        )

    if source_sample_rate != target_sample_rate and samples.size:
        from scipy.signal import resample_poly  # type: ignore[import-untyped]

        divisor = gcd(int(source_sample_rate), int(target_sample_rate))
        samples = resample_poly(
            samples,
            up=target_sample_rate // divisor,
            down=source_sample_rate // divisor,
        ).astype(np.float32, copy=False)

    if not np.all(np.isfinite(samples)):
        raise ValueError(f"Decoded audio contains non-finite samples: {source_path}")

    return AudioData(
        samples=np.ascontiguousarray(samples, dtype=np.float32),
        sample_rate=target_sample_rate,
        path=source_path,
    )


def split_audio(
    samples: NDArray[np.float32],
    sample_rate: int,
    window_seconds: float = 10.0,
    stride_seconds: float = 5.0,
) -> list[AudioWindow]:
    """Split mono PCM into overlapping, fixed-size windows.

    At most one partial window is emitted.  It is right-padded with zeros and
    processing then stops.  If a full window already ends exactly at EOF, no
    additional overlapping partial window is added.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive")

    mono = np.asarray(samples, dtype=np.float32)
    if mono.ndim != 1:
        raise ValueError(f"samples must be one-dimensional mono audio; got {mono.shape}")
    if mono.size == 0:
        return []

    window_size = round(window_seconds * sample_rate)
    stride_size = round(stride_seconds * sample_rate)
    if window_size < 1:
        raise ValueError("window_seconds is too small for the sample rate")
    if stride_size < 1:
        raise ValueError("stride_seconds is too small for the sample rate")

    total_samples = len(mono)
    windows: list[AudioWindow] = []
    start_sample = 0

    while start_sample < total_samples:
        true_end_sample = min(start_sample + window_size, total_samples)
        valid_sample_count = true_end_sample - start_sample
        window = np.zeros(window_size, dtype=np.float32)
        window[:valid_sample_count] = mono[start_sample:true_end_sample]

        windows.append(
            AudioWindow(
                index=len(windows),
                samples=window,
                sample_rate=sample_rate,
                start_seconds=start_sample / sample_rate,
                end_seconds=true_end_sample / sample_rate,
                valid_sample_count=valid_sample_count,
            )
        )

        # A partial tail is the final window.  Equality also stops here: the
        # source has already been covered through EOF by a full window.
        if true_end_sample == total_samples:
            break
        start_sample += stride_size

    return windows
