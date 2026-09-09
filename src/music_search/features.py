"""Lightweight DSP features for decoded audio windows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class DSPFeatures:
    """Scalar features persisted alongside a segment embedding."""

    rms_energy: float
    bpm: float


def _mono_float32(samples: NDArray[np.float32]) -> NDArray[np.float32]:
    mono = np.asarray(samples, dtype=np.float32)
    if mono.ndim != 1:
        raise ValueError(f"samples must be one-dimensional mono audio; got {mono.shape}")
    # A damaged sample should not make the whole ingestion run produce NaN
    # metadata.  Decoders normally return finite PCM, but callers may not.
    return np.nan_to_num(mono, copy=True, nan=0.0, posinf=1.0, neginf=-1.0)


def rms_energy(samples: NDArray[np.float32]) -> float:
    """Return mean frame-wise root-mean-square energy using librosa."""

    mono = _mono_float32(samples)
    if mono.size == 0:
        return 0.0

    import librosa

    frame_rms = np.asarray(librosa.feature.rms(y=mono), dtype=np.float64)
    finite_values = frame_rms[np.isfinite(frame_rms)]
    if finite_values.size == 0:
        return 0.0
    return max(0.0, float(finite_values.mean()))


def estimate_bpm(samples: NDArray[np.float32], sample_rate: int) -> float:
    """Estimate tempo in beats per minute, returning zero when indeterminate."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")

    mono = _mono_float32(samples)
    # Very short clips do not contain enough temporal context for a meaningful
    # tempo estimate.  This also avoids edge-case failures inside beat tracking.
    if mono.size < max(2, sample_rate // 2):
        return 0.0
    if not np.any(np.abs(mono) > np.finfo(np.float32).eps):
        return 0.0

    import librosa

    try:
        tempo, _beat_frames = librosa.beat.beat_track(y=mono, sr=sample_rate)
    except (IndexError, ValueError, FloatingPointError):
        return 0.0

    values = np.asarray(tempo, dtype=np.float64).reshape(-1)
    finite_positive = values[np.isfinite(values) & (values > 0)]
    if finite_positive.size == 0:
        return 0.0
    return float(finite_positive[0])


def extract_dsp_features(
    samples: NDArray[np.float32],
    sample_rate: int,
) -> DSPFeatures:
    """Compute all scalar DSP metadata for one audio segment."""

    return DSPFeatures(
        rms_energy=rms_energy(samples),
        bpm=estimate_bpm(samples, sample_rate),
    )
