"""Texto -> vectores normalizados con el modelo de embeddings local (Ollama)."""
import numpy as np
from ollama import Client

from .config import (
    EMBED_BATCH_SIZE,
    EMBED_DIM,
    EMBED_MODEL,
    OLLAMA_ENDPOINT,
    TIMEOUT,
)

DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

_client = Client(host=OLLAMA_ENDPOINT, timeout=TIMEOUT)


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
            truncate=False,  # falla en vez de cortar en silencio textos de mas de 512 tokens
        )
        vectors = np.asarray(response.embeddings, dtype=np.float32)
        expected_shape = (len(batch), EMBED_DIM)
        if vectors.shape != expected_shape:
            raise ValueError(f"Forma inesperada: {vectors.shape}; se esperaba {expected_shape}.")
        batches.append(vectors)

    vectors = np.concatenate(batches, axis=0)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.all(np.isfinite(vectors)) or np.any(norms == 0):
        raise ValueError("El modelo devolvió vectores inválidos (NaN, infinitos o nulos).")
    return vectors / norms  # normalizados: la similitud coseno es un producto punto


def embed_texts(texts: list[str]) -> np.ndarray:
    return _embed(texts, prefix=DOC_PREFIX)


def embed_query(text: str) -> np.ndarray:
    return _embed([text], prefix=QUERY_PREFIX)[0]