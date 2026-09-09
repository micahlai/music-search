import sys
from types import SimpleNamespace

import numpy as np
import pytest

import music_search.features as features_module
from music_search.features import DSPFeatures, estimate_bpm, extract_dsp_features, rms_energy


def test_rms_energy_uses_mono_float32_samples(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, np.ndarray] = {}

    def fake_rms(*, y: np.ndarray) -> np.ndarray:
        observed["samples"] = y
        return np.array([[np.sqrt(np.mean(np.square(y, dtype=np.float64)))]])

    fake_librosa = SimpleNamespace(feature=SimpleNamespace(rms=fake_rms))
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)
    samples = np.array([3.0, 4.0], dtype=np.float64)

    result = rms_energy(samples)

    assert result == pytest.approx(np.sqrt(12.5))
    assert observed["samples"].dtype == np.float32
    np.testing.assert_array_equal(observed["samples"], samples.astype(np.float32))


def test_rms_energy_returns_zero_for_empty_audio() -> None:
    assert rms_energy(np.array([], dtype=np.float32)) == 0.0


def test_estimate_bpm_converts_librosa_array_result_to_scalar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sample_rate = 100
    samples = np.zeros(sample_rate, dtype=np.float32)
    samples[::25] = 1.0

    def fake_beat_track(*, y: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        np.testing.assert_array_equal(y, samples)
        assert sr == sample_rate
        return np.array([120.0]), np.array([0, 25, 50, 75])

    fake_librosa = SimpleNamespace(beat=SimpleNamespace(beat_track=fake_beat_track))
    monkeypatch.setitem(sys.modules, "librosa", fake_librosa)

    assert estimate_bpm(samples, sample_rate) == pytest.approx(120.0)


@pytest.mark.parametrize(
    "samples",
    [
        np.zeros(100, dtype=np.float32),
        np.ones(10, dtype=np.float32),
    ],
)
def test_estimate_bpm_returns_zero_when_tempo_is_indeterminate(samples: np.ndarray) -> None:
    assert estimate_bpm(samples, sample_rate=100) == 0.0


def test_extract_dsp_features_combines_feature_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    samples = np.ones(32, dtype=np.float32)
    monkeypatch.setattr(features_module, "rms_energy", lambda value: 0.75)
    monkeypatch.setattr(features_module, "estimate_bpm", lambda value, sample_rate: 98.0)

    result = extract_dsp_features(samples, sample_rate=48_000)

    assert result == DSPFeatures(rms_energy=0.75, bpm=98.0)


def test_feature_functions_reject_multichannel_audio() -> None:
    stereo = np.zeros((8, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="one-dimensional"):
        rms_energy(stereo)
    with pytest.raises(ValueError, match="one-dimensional"):
        estimate_bpm(stereo, sample_rate=48_000)
