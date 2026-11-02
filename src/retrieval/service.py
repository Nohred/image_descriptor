"""Funcion publica de busqueda: search(query, mode, limit).

Modos:
- fixed:    LLM -> filtros -> SQL sobre vehicles.sqlite3.
- semantic: embedding de la consulta completa -> similitud sobre florence.sqlite3 (sin LLM de chat).
- hybrid:   LLM separa filtros y texto semantico; los filtros acotan las imagenes y el texto las ordena.
En semantic/hybrid, SearchResult.total es el numero de imagenes consideradas, no el de coincidencias.
"""
from time import perf_counter
from typing import Protocol

from retrieval.query_parser import parse_query
from retrieval.rules import apply_rules
from retrieval.schemas import DEFAULT_LIMIT, MAX_LIMIT, MODES, ParsedQuery, SearchResult

# Configuracion
HYBRID_USE_FULL_QUERY = False  # True: en hybrid se embebe la consulta completa; False: solo el texto semantico residual


class GeneratedSource(Protocol):
    """Contrato que cumple database/generated_features/repository.py."""

    def search(self, query_vector, limit, allowed_ids=None):
        """Devuelve (list[Candidate] por similitud descendente, imagenes consideradas)."""
        ...


def _default_embedder():
    from local_llm.embedder import embed_query
    return embed_query


class SearchService:
    def __init__(self, fixed_source, chat_fn, generated_source=None, embedder=None):
        self.fixed = fixed_source
        self.chat_fn = chat_fn
        self.generated = generated_source
        self.embedder = embedder  # callable texto -> vector; por defecto local_llm.embedder.embed_query

    def search(self, query, mode="fixed", limit=DEFAULT_LIMIT):
        if mode not in MODES:
            raise ValueError(f"Modo desconocido {mode!r}. Opciones: {MODES}")
        if mode in ("semantic", "hybrid") and self.generated is None:
            raise RuntimeError(f"El modo {mode!r} requiere generated_source (GeneratedSource).")
        start = perf_counter()
        if mode == "fixed":
            parsed, candidates, total = self._fixed(query, limit)
        elif mode == "semantic":
            parsed, candidates, total = self._semantic(query, limit)
        else:
            parsed, candidates, total = self._hybrid(query, limit)
        return SearchResult(mode=mode, query=query, parsed=parsed, candidates=candidates,
                            total=total, elapsed_ms=(perf_counter() - start) * 1000)

    @staticmethod
    def _clamp(requested, default):
        return max(1, min(requested or default, MAX_LIMIT))

    def _parse(self, query):
        return apply_rules(query, parse_query(query, self.chat_fn))

    def _embed(self, text):
        embed = self.embedder or _default_embedder()
        return embed(text)

    def _fixed(self, query, limit):
        parsed = self._parse(query)
        if not parsed.has_filters():
            parsed.warnings.append("No se detecto ningun filtro: se devuelven registros sin restriccion.")
        candidates, total = self.fixed.search(parsed, self._clamp(parsed.limit, limit))
        return parsed, candidates, total

    def _semantic(self, query, limit):
        parsed = ParsedQuery(semantic_text=query)
        candidates, considered = self.generated.search(self._embed(query), self._clamp(None, limit))
        return parsed, candidates, considered

    def _hybrid(self, query, limit):
        parsed = self._parse(query)
        effective = self._clamp(parsed.limit, limit)
        allowed, attributes, fixed_candidates, fixed_total = None, {}, [], 0
        if parsed.has_filters():
            fixed_total = self.fixed.search(parsed, 1)[1]
            if fixed_total == 0:
                return parsed, [], 0
            fixed_candidates, _ = self.fixed.search(parsed, fixed_total)
            allowed = {c.image_id for c in fixed_candidates}
            attributes = {c.image_id: c.attributes for c in fixed_candidates}

        text = query if HYBRID_USE_FULL_QUERY else parsed.semantic_text
        if not text:
            if allowed is not None:
                parsed.warnings.append("Sin texto semantico: se devuelve el resultado de los filtros (modo fixed).")
                return parsed, fixed_candidates[:effective], fixed_total
            parsed.warnings.append("Sin filtros ni texto semantico: se usa la consulta completa.")
            text = query

        candidates, considered = self.generated.search(self._embed(text), effective, allowed_ids=allowed)
        if allowed is not None and considered < len(allowed):
            parsed.warnings.append(
                f"{len(allowed) - considered} de {len(allowed)} imagenes filtradas no tienen descripcion y no se consideraron.")
        for candidate in candidates:
            candidate.attributes = {**attributes.get(candidate.image_id, {}), **candidate.attributes}
        return parsed, candidates, considered