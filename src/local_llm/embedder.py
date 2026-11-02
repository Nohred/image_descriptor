import numpy as np
from ollama import Client

from .config import (
    OLLAMA_ENDPOINT,
    EMBED_MODEL,
    EMBED_DIM,
    EMBED_BATCH_SIZE,
    TIMEOUT,
)

_client = Client(
    host=OLLAMA_ENDPOINT,
    timeout=TIMEOUT,
)


def _embed(texts: list[str], prefix: str) -> np.ndarray:
    if not texts:
        return np.empty((0, EMBED_DIM), dtype=np.float32)

    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Cada entrada debe ser un texto no vacío.")

    batches = []

    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]

        response = _client.embed(
            model=EMBED_MODEL,
            input=[prefix + text.strip() for text in batch],
            truncate=False,
        )

        vectors = np.asarray(
            response.embeddings,
            dtype=np.float32,
        )

        expected_shape = (len(batch), EMBED_DIM)

        if vectors.shape != expected_shape:
            raise ValueError(
                f"Forma inesperada: {vectors.shape}; "
                f"se esperaba {expected_shape}."
            )

        batches.append(vectors)

    return np.concatenate(batches, axis=0)


def embed_texts(texts: list[str]) -> np.ndarray:
    return _embed(texts, prefix="search_document: ")


def embed_query(text: str) -> np.ndarray:
    return _embed([text], prefix="search_query: ")[0]