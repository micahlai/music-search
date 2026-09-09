"""Audio/text embeddings backed by LAION CLAP.

PyTorch and Transformers are intentionally imported only when an embedding is
first requested.  Commands that only inspect metadata therefore remain light
and do not initialize an accelerator or download a model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from threading import Lock
from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from music_search.audio import DEFAULT_SAMPLE_RATE

CLAP_MODEL_NAME = "laion/clap-htsat-unfused"
CLAP_EMBEDDING_DIM = 512


def _accelerator_available(torch_module: Any, name: str) -> bool:
    if name == "cuda":
        cuda = getattr(torch_module, "cuda", None)
        return bool(cuda and cuda.is_available())
    if name == "mps":
        backends = getattr(torch_module, "backends", None)
        mps = getattr(backends, "mps", None)
        return bool(mps and mps.is_available())
    return name == "cpu"


def resolve_device(requested_device: str, torch_module: Any) -> str:
    """Resolve ``auto`` or validate an explicit CLAP execution device."""

    requested = requested_device.casefold().strip()
    if requested == "auto":
        for candidate in ("cuda", "mps"):
            if _accelerator_available(torch_module, candidate):
                return candidate
        return "cpu"

    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be one of: auto, cpu, cuda, mps")
    if not _accelerator_available(torch_module, requested):
        raise RuntimeError(f"Requested CLAP device is not available: {requested}")
    return requested


class ClapEmbedder:
    """Lazy, batched wrapper around ``transformers.ClapModel``.

    Audio and text vectors share CLAP's 512-dimensional retrieval space.  Every
    returned vector is L2-normalized, making cosine similarity equivalent to a
    dot product in pgvector.
    """

    def __init__(
        self,
        model_name: str = CLAP_MODEL_NAME,
        device: str = "auto",
        batch_size: int = 8,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name must not be empty")
        if batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if device.casefold().strip() not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")

        self.model_name = model_name
        self.requested_device = device.casefold().strip()
        self.batch_size = batch_size
        self.embedding_dimension = CLAP_EMBEDDING_DIM

        self._torch: Any | None = None
        self._processor: Any | None = None
        self._model: Any | None = None
        self._device: str | None = None
        self._load_lock = Lock()

    @property
    def is_loaded(self) -> bool:
        """Whether model weights and their processor are currently loaded."""

        return self._model is not None

    @property
    def device(self) -> str:
        """Resolved execution device, loading CLAP on first access."""

        self._ensure_loaded()
        assert self._device is not None
        return self._device

    @property
    def processor(self) -> Any:
        """The lazy-loaded Hugging Face ``ClapProcessor`` instance."""

        self._ensure_loaded()
        return self._processor

    @property
    def model(self) -> Any:
        """The lazy-loaded Hugging Face ``ClapModel`` instance."""

        self._ensure_loaded()
        return self._model

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import (
                    ClapModel,
                    ClapProcessor,
                )
            except ImportError as exc:  # pragma: no cover - depends on environment
                raise RuntimeError(
                    "CLAP embeddings require the 'torch' and 'transformers' "
                    "packages. Install the project's embedding dependencies."
                ) from exc

            resolved_device = resolve_device(self.requested_device, torch)
            processor = ClapProcessor.from_pretrained(self.model_name)
            model = ClapModel.from_pretrained(self.model_name)
            model.to(resolved_device)
            model.eval()

            # Assign only after every load step succeeds, so a failed download or
            # device move does not leave a half-initialized wrapper.
            self._torch = torch
            self._processor = processor
            self._model = model
            self._device = resolved_device

    def _move_inputs(self, inputs: Any) -> dict[str, Any]:
        assert self._device is not None
        if hasattr(inputs, "to"):
            moved = inputs.to(self._device)
            if moved is not None:
                inputs = moved
        if not isinstance(inputs, Mapping):
            raise TypeError("CLAP processor output must be a mapping of model inputs")

        return {
            key: value.to(self._device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def _inference_context(self) -> Any:
        assert self._torch is not None
        context_factory = getattr(self._torch, "inference_mode", None)
        if context_factory is None:
            context_factory = getattr(self._torch, "no_grad", None)
        return context_factory() if context_factory is not None else nullcontext()

    @staticmethod
    def _normalize_and_validate(
        features: Any,
        expected_rows: int,
    ) -> NDArray[np.float32]:
        if hasattr(features, "detach"):
            features = features.detach()
        if hasattr(features, "float"):
            features = features.float()
        if hasattr(features, "cpu"):
            features = features.cpu()
        if hasattr(features, "numpy"):
            features = features.numpy()

        vectors = np.asarray(features, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2:
            raise RuntimeError(f"CLAP returned an invalid embedding shape: {vectors.shape}")
        if vectors.shape[0] != expected_rows:
            raise RuntimeError(
                "CLAP returned a different number of embeddings than inputs: "
                f"expected {expected_rows}, got {vectors.shape[0]}"
            )
        if vectors.shape[1] != CLAP_EMBEDDING_DIM:
            raise RuntimeError(
                "Unexpected CLAP embedding dimension: "
                f"expected {CLAP_EMBEDDING_DIM}, got {vectors.shape[1]}"
            )
        if not np.all(np.isfinite(vectors)):
            raise RuntimeError("CLAP returned non-finite embedding values")

        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms <= np.finfo(np.float32).eps):
            raise RuntimeError("CLAP returned a zero-norm embedding")
        return np.ascontiguousarray(vectors / norms, dtype=np.float32)

    def embed_audio(
        self,
        audios: Sequence[NDArray[np.float32]],
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        *,
        on_progress: Callable[[str], None] | None = None,
    ) -> NDArray[np.float32]:
        """Embed mono audio arrays, returning shape ``(n_audio, 512)``."""

        if sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer")

        prepared: list[NDArray[np.float32]] = []
        for audio in audios:
            mono = np.asarray(audio, dtype=np.float32)
            if mono.ndim != 1:
                raise ValueError(f"Each audio input must be one-dimensional; got {mono.shape}")
            if mono.size == 0:
                raise ValueError("Audio inputs must not be empty")
            prepared.append(
                np.ascontiguousarray(
                    np.nan_to_num(
                        mono,
                        copy=True,
                        nan=0.0,
                        posinf=1.0,
                        neginf=-1.0,
                    ),
                    dtype=np.float32,
                )
            )

        if not prepared:
            return np.empty((0, CLAP_EMBEDDING_DIM), dtype=np.float32)

        if on_progress is not None and not self.is_loaded:
            on_progress(
                f"  Loading CLAP {self.model_name}; downloading weights if not cached "
                f"(requested device: {self.requested_device})"
            )
        self._ensure_loaded()
        assert self._processor is not None
        assert self._model is not None
        if on_progress is not None:
            on_progress(f"  CLAP ready on {self._device}; batch size {self.batch_size}")

        batches: list[NDArray[np.float32]] = []
        for offset in range(0, len(prepared), self.batch_size):
            batch_started = perf_counter()
            batch = prepared[offset : offset + self.batch_size]
            inputs = self._processor(
                audios=batch,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding="pad",
                truncation="rand_trunc",
            )
            model_inputs = self._move_inputs(inputs)
            with self._inference_context():
                features = self._model.get_audio_features(**model_inputs)
            batches.append(self._normalize_and_validate(features, len(batch)))
            if on_progress is not None:
                on_progress(
                    f"  CLAP: {offset + len(batch)}/{len(prepared)} windows "
                    f"({perf_counter() - batch_started:.2f}s for this batch)"
                )

        return np.concatenate(batches, axis=0)

    def embed_text(
        self,
        texts: str | Sequence[str],
    ) -> NDArray[np.float32]:
        """Embed search text in the same space as audio.

        A single string returns shape ``(512,)``.  A sequence returns
        ``(n_texts, 512)`` and uses the configured batch size.
        """

        single = isinstance(texts, str)
        prepared = [texts] if single else list(texts)
        if any(not isinstance(text, str) or not text.strip() for text in prepared):
            raise ValueError("Text inputs must be non-empty strings")
        if not prepared:
            return np.empty((0, CLAP_EMBEDDING_DIM), dtype=np.float32)

        self._ensure_loaded()
        assert self._processor is not None
        assert self._model is not None

        batches: list[NDArray[np.float32]] = []
        for offset in range(0, len(prepared), self.batch_size):
            batch = prepared[offset : offset + self.batch_size]
            inputs = self._processor(
                text=batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            model_inputs = self._move_inputs(inputs)
            with self._inference_context():
                features = self._model.get_text_features(**model_inputs)
            batches.append(self._normalize_and_validate(features, len(batch)))

        vectors = np.concatenate(batches, axis=0)
        return vectors.reshape(CLAP_EMBEDDING_DIM) if single else vectors
