"""Funcion publica de busqueda: search(query, mode, limit)."""
from time import perf_counter
from typing import Protocol

from retrieval.query_parser import parse_query
from retrieval.rules import apply_rules
from retrieval.schemas import (
    DEFAULT_LIMIT, IMPLEMENTED_MODES, MAX_LIMIT, MODES, SearchResult,
)


class GeneratedSource(Protocol):
    """DECLARADO, sin implementar: lo cumplira database/generated_features/repository.py."""

    def knn(self, query_vector, limit, allowed_ids=None):
        """Devuelve list[Candidate] ordenados por cercania, opcionalmente restringidos a allowed_ids."""
        ...


class SearchService:
    def __init__(self, fixed_source, chat_fn, generated_source=None, embedder=None):
        self.fixed = fixed_source
        self.chat_fn = chat_fn
        self.generated = generated_source  # sin uso hasta implementar generated_features
        self.embedder = embedder           # idem

    def search(self, query, mode="fixed", limit=DEFAULT_LIMIT):
        if mode not in MODES:
            raise ValueError(f"Modo desconocido {mode!r}. Opciones: {MODES}")
        if mode not in IMPLEMENTED_MODES:
            raise NotImplementedError(f"El modo {mode!r} requiere generated_features (pendiente).")
        start = perf_counter()
        # parsed = parse_query(query, self.chat_fn)
        parsed = apply_rules(query, parse_query(query, self.chat_fn))
        if not parsed.has_filters():
            parsed.warnings.append("No se detecto ningun filtro: se devuelven registros sin restriccion.")
        effective = max(1, min(parsed.limit or limit, MAX_LIMIT))
        candidates, total = self.fixed.search(parsed, effective)
        return SearchResult(mode=mode, query=query, parsed=parsed, candidates=candidates,
                            total=total, elapsed_ms=(perf_counter() - start) * 1000)
