import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from music_search.embeddings import CLAP_EMBEDDING_DIM, ClapEmbedder, resolve_device


class _FakeBatch(dict[str, Any]):
    def __init__(self, batch_size: int) -> None:
        super().__init__(batch_size=batch_size)
        self.moved_to: str | None = None

    def to(self, device: str) -> "_FakeBatch":
        self.moved_to = device
        return self


def _install_fake_hugging_face(
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    state = SimpleNamespace(
        processor_names=[],
        model_names=[],
        processor_calls=[],
        model_device=None,
        model_eval_called=False,
        audio_batch_sizes=[],
        text_batch_sizes=[],
    )

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_name: str) -> "FakeProcessor":
            state.processor_names.append(model_name)
            return cls()

        def __call__(self, **kwargs: Any) -> _FakeBatch:
            values = kwargs.get("audios", kwargs.get("text"))
            state.processor_calls.append(kwargs)
            return _FakeBatch(len(values))

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_name: str) -> "FakeModel":
            state.model_names.append(model_name)
            return cls()

        def to(self, device: str) -> "FakeModel":
            state.model_device = device
            return self

        def eval(self) -> None:
            state.model_eval_called = True

        def get_audio_features(self, *, batch_size: int) -> np.ndarray:
            state.audio_batch_sizes.append(batch_size)
            vectors = np.zeros((batch_size, CLAP_EMBEDDING_DIM), dtype=np.float32)
            vectors[:, :2] = (3.0, 4.0)
            return vectors

        def get_text_features(self, *, batch_size: int) -> np.ndarray:
            state.text_batch_sizes.append(batch_size)
            vectors = np.zeros((batch_size, CLAP_EMBEDDING_DIM), dtype=np.float32)
            vectors[:, :2] = (5.0, 12.0)
            return vectors

    fake_torch = ModuleType("torch")
    fake_torch.cuda = SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
    fake_torch.backends = SimpleNamespace(  # type: ignore[attr-defined]
        mps=SimpleNamespace(is_available=lambda: False)
    )
    fake_torch.inference_mode = nullcontext  # type: ignore[attr-defined]

    fake_transformers = ModuleType("transformers")
    fake_transformers.ClapProcessor = FakeProcessor  # type: ignore[attr-defined]
    fake_transformers.ClapModel = FakeModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    return state


def test_clap_embedder_loads_mocked_hugging_face_components_lazily(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_fake_hugging_face(monkeypatch)
    embedder = ClapEmbedder(model_name="local-test-clap")

    assert not embedder.is_loaded
    assert state.processor_names == []
    assert state.model_names == []

    assert embedder.device == "cpu"
    assert embedder.is_loaded
    assert state.processor_names == ["local-test-clap"]
    assert state.model_names == ["local-test-clap"]
    assert state.model_device == "cpu"
    assert state.model_eval_called


def test_embed_audio_batches_normalizes_and_forwards_sample_rate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_fake_hugging_face(monkeypatch)
    embedder = ClapEmbedder(device="cpu", batch_size=2)
    audios = [np.linspace(-1.0, 1.0, length, dtype=np.float32) for length in (16, 24, 32)]

    progress: list[str] = []
    vectors = embedder.embed_audio(audios, sample_rate=16_000, on_progress=progress.append)

    assert vectors.shape == (3, CLAP_EMBEDDING_DIM)
    assert vectors.dtype == np.float32
    assert vectors.flags.c_contiguous
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(vectors[:, :2], np.array([[0.6, 0.8]] * 3), atol=1e-6)
    assert state.audio_batch_sizes == [2, 1]
    assert "Loading CLAP" in progress[0]
    assert any("ready on cpu" in message for message in progress)
    assert [
        message.split(" windows")[0].strip() for message in progress if " windows" in message
    ] == ["CLAP: 2/3", "CLAP: 3/3"]
    assert [call["sampling_rate"] for call in state.processor_calls] == [16_000, 16_000]
    assert [len(call["audios"]) for call in state.processor_calls] == [2, 1]


def test_embed_text_preserves_single_and_sequence_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _install_fake_hugging_face(monkeypatch)
    embedder = ClapEmbedder(batch_size=2)

    single = embedder.embed_text("nocturnal ambient synth")
    multiple = embedder.embed_text(["quiet piano", "driving pulse", "cinematic build"])

    assert single.shape == (CLAP_EMBEDDING_DIM,)
    assert multiple.shape == (3, CLAP_EMBEDDING_DIM)
    np.testing.assert_allclose(np.linalg.norm(single), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(multiple, axis=1), 1.0, atol=1e-6)
    assert state.text_batch_sizes == [1, 2, 1]


def test_empty_batches_do_not_load_the_model() -> None:
    embedder = ClapEmbedder()

    assert embedder.embed_audio([]).shape == (0, CLAP_EMBEDDING_DIM)
    assert embedder.embed_text([]).shape == (0, CLAP_EMBEDDING_DIM)
    assert not embedder.is_loaded


@pytest.mark.parametrize("bad_text", ["", "   ", ["valid", ""]])
def test_embed_text_rejects_blank_values_without_loading(bad_text: Any) -> None:
    embedder = ClapEmbedder()

    with pytest.raises(ValueError, match="non-empty"):
        embedder.embed_text(bad_text)
    assert not embedder.is_loaded


def test_embed_audio_rejects_non_mono_input_without_loading() -> None:
    embedder = ClapEmbedder()

    with pytest.raises(ValueError, match="one-dimensional"):
        embedder.embed_audio([np.zeros((8, 2), dtype=np.float32)])
    assert not embedder.is_loaded


def test_resolve_device_rejects_an_unavailable_accelerator() -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )

    assert resolve_device("auto", fake_torch) == "cpu"
    with pytest.raises(RuntimeError, match="not available"):
        resolve_device("cuda", fake_torch)


def test_clap_embedding_dimension_is_validated() -> None:
    wrong_dimension = np.ones((1, CLAP_EMBEDDING_DIM - 1), dtype=np.float32)

    with pytest.raises(RuntimeError, match="embedding dimension"):
        ClapEmbedder._normalize_and_validate(wrong_dimension, expected_rows=1)


def test_actual_clap_feature_extractor_accepts_fixed_and_short_windows() -> None:
    # Exercise the real installed preprocessing contract without fetching a checkpoint.
    from transformers import ClapFeatureExtractor

    extractor = ClapFeatureExtractor()
    features = extractor(
        [np.zeros(480000, dtype=np.float32), np.zeros(240000, dtype=np.float32)],
        sampling_rate=48000,
        padding="pad",
        truncation="rand_trunc",
        return_tensors="np",
    )
    assert features["input_features"].shape[0] == 2
    assert np.isfinite(features["input_features"]).all()
