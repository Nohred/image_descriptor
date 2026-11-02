"""Consulta en lenguaje natural -> ParsedQuery validado."""
from datetime import date
import json
import re

from local_llm.prompts import build_output_schema, build_system_prompt
from retrieval.schemas import (
    COLOR_ALIASES, COLOR_NAMES, MAX_LIMIT, PERIODS, TYPE_ALIASES, TYPE_NAMES,
    ParsedQuery, map_values,
)

TIME_RE = re.compile(r"^(?:[01]?\d|2[0-3]):[0-5]\d$")


class QueryParseError(Exception):
    pass


def _clean_time(value):
    if value in (None, ""):
        return None
    value = str(value).strip()
    if value == "24:00":
        return "00:00"
    if not TIME_RE.match(value):
        return False
    hour, minute = value.split(":")
    return f"{int(hour):02d}:{minute}"


def _clean_date(value):
    if value in (None, ""):
        return None
    try:
        return date.fromisoformat(str(value).strip()).isoformat()
    except ValueError:
        return False


def validate(raw):
    """Normaliza el JSON del LLM. Lo invalido se descarta y se registra en warnings."""
    if not isinstance(raw, dict):
        raise QueryParseError("La respuesta del LLM no es un objeto JSON.")
    parsed = ParsedQuery()
    unsupported = [str(t) for t in raw.get("unsupported_terms") or []]
    for field_name, valid, aliases in (
        ("types", TYPE_NAMES, TYPE_ALIASES), ("colors", COLOR_NAMES, COLOR_ALIASES),
        ("exclude_types", TYPE_NAMES, TYPE_ALIASES), ("exclude_colors", COLOR_NAMES, COLOR_ALIASES),
    ):
        ok, bad = map_values(raw.get(field_name), valid, aliases)
        setattr(parsed, field_name, ok)
        unsupported.extend(bad)
        if bad:
            parsed.warnings.append(f"{field_name}: valores no reconocidos {bad}")
    for name in ("types", "colors"):
        both = set(getattr(parsed, name)) & set(getattr(parsed, "exclude_" + name))
        if both:
            parsed.warnings.append(f"{name}: {sorted(both)} aparece como incluido y excluido; se ignora la exclusion")
            setattr(parsed, "exclude_" + name, [v for v in getattr(parsed, "exclude_" + name) if v not in both])

    t_from, t_to = _clean_time(raw.get("time_from")), _clean_time(raw.get("time_to"))
    if t_from is False or t_to is False:
        parsed.warnings.append("hora con formato invalido; se ignora el filtro horario")
    elif (t_from is None) != (t_to is None):
        parsed.warnings.append("rango horario incompleto; se ignora")
    elif t_from is not None and t_from == t_to:
        parsed.warnings.append("time_from igual a time_to; se ignora")
    else:
        parsed.time_from, parsed.time_to = t_from, t_to

    d_from, d_to = _clean_date(raw.get("date_from")), _clean_date(raw.get("date_to"))
    if d_from is False or d_to is False:
        parsed.warnings.append("fecha con formato invalido; se ignora el filtro de fecha")
    elif d_from and d_to and d_from > d_to:
        parsed.warnings.append("date_from posterior a date_to; se ignora el filtro de fecha")
    else:
        parsed.date_from, parsed.date_to = d_from, d_to

    limit = raw.get("limit")
    if isinstance(limit, int) and not isinstance(limit, bool) and limit > 0:
        parsed.limit = min(limit, MAX_LIMIT)
    text = raw.get("semantic_text")
    parsed.semantic_text = str(text).strip() if text and str(text).strip() else None
    parsed.unsupported_terms = list(dict.fromkeys(unsupported))
    return parsed


def parse_query(query, chat_fn, today=None):
    """chat_fn(system_prompt, user_text, json_schema) -> str con JSON."""
    today = today or date.today().isoformat()
    system = build_system_prompt(COLOR_NAMES, TYPE_NAMES, PERIODS, COLOR_ALIASES, TYPE_ALIASES, today)
    schema = build_output_schema(COLOR_NAMES, TYPE_NAMES)
    text = chat_fn(system, query, schema)
    try:
        raw = json.loads(text)
    except (json.JSONDecodeError, TypeError) as error:
        raise QueryParseError(f"El LLM no devolvio JSON valido: {error}") from error
    return validate(raw)