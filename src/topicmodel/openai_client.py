"""Lightweight wrapper around the OpenAI embeddings API."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The `openai` package is required for OpenAIClient. "
        "Install it via `pip install openai`."
    ) from exc


class OpenAIClient:
    """Minimal helper that owns a configured OpenAI SDK client."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        self._ensure_local_env_loaded()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Please export your API key "
                "before requesting embeddings."
            )
        self.embedding_model = embedding_model
        self._client = OpenAI(api_key=self.api_key)

    def embed_texts(
        self,
        texts: Iterable[str],
        *,
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """Return embeddings for each input string."""
        model_name = model or self.embedding_model
        response = self._client.embeddings.create(
            model=model_name,
            input=list(texts),
        )
        return [item.embedding for item in response.data]

    @staticmethod
    def _ensure_local_env_loaded() -> None:
        """Load OPENAI_API_KEY from a .env sitting next to this module."""
        env_path = Path(__file__).with_name(".env")
        if not env_path.exists():
            return

        for line in env_path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


class OpenAIEmbedder:
    """Adapter that mimics the SentenceTransformer.encode interface."""

    def __init__(
        self,
        *,
        embedding_model: str = "text-embedding-3-small",
        batch_size: int = 128,
    ) -> None:
        self.client = OpenAIClient(embedding_model=embedding_model)
        self.embedding_model = embedding_model
        self.batch_size = batch_size

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: Optional[int] = None,
        show_progress_bar: bool = False,  # noqa: ARG002 (kept for API parity)
        **_: object,
    ) -> np.ndarray:
        """Return numpy matrix of embeddings for the given sentences."""
        bs = batch_size or self.batch_size
        embeddings: List[List[float]] = []
        n_batches = math.ceil(len(sentences) / bs) if sentences else 0
        for batch_idx in range(n_batches):
            start = batch_idx * bs
            end = start + bs
            batch = sentences[start:end]
            embeddings.extend(self.client.embed_texts(batch))
        return np.asarray(embeddings, dtype=np.float32)


def wants_openai_embeddings(model_name: str) -> bool:
    """Return True if the requested embedding model should use OpenAI."""
    return model_name.startswith("text-embedding")

