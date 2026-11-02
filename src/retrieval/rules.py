"""Correcciones deterministas sobre la salida del LLM (modelos pequenos fallan en lo repetitivo)."""
import re

from retrieval.schemas import COLOR_ALIASES, PERIODS, TYPE_ALIASES, strip_accents

PERIOD_PATTERNS = {
    "madrugada": r"\bmadrugada\b",
    "manana": r"\b(?:en|por|de|durante)\s+la\s+manana\b",
    "tarde": r"\b(?:en|por|de|durante)\s+la\s+tarde\b",
    "noche": r"\b(?:de\s+noche|(?:en|por|de|durante)\s+la\s+noche)\b",
}
# Si el usuario dio una hora explicita ("a las 6", "10 de la noche", "18:00"), no se aplica el periodo.
EXPLICIT_HOUR = re.compile(r"\b\d{1,2}:\d{2}\b|\b\d{1,2}\s*(?:de la|am|pm|hrs|horas|h)\b|\blas?\s+\d{1,2}\b")
NEGATORS = {"no", "sin", "excepto", "menos", "ni"}
CONJUNCTIONS = {"y", "o", "e", "u", "pero"}


def _find(key, text):
    return re.finditer(rf"\b{re.escape(key)}(?:s|es)?\b", text)


def _negated(text, start):
    """Camina hacia atras hasta 3 palabras: gana lo que aparezca primero (negador o conjuncion)."""
    for word in reversed(re.findall(r"[a-z]+", text[:start])[-3:]):
        if word in NEGATORS:
            return True
        if word in CONJUNCTIONS:
            return False
    return False


def _scan(text, aliases):
    """Devuelve (valores_positivos_inequivocos, valores_ambiguos, valores_negados)."""
    direct, ambiguous, negated = [], [], []
    for key, values in aliases.items():
        for match in _find(key, text):
            bucket = negated if _negated(text, match.start()) else (direct if len(values) == 1 else ambiguous)
            for value in values:
                if value not in bucket:
                    bucket.append(value)
    return direct, ambiguous, negated


def _apply_attribute(text, parsed, aliases, include, exclude):
    direct, ambiguous, negated = _scan(text, aliases)
    current_in, current_out = list(getattr(parsed, include)), list(getattr(parsed, exclude))
    for value in negated:
        if value not in current_out:
            current_out.append(value)
            parsed.warnings.append(f"{exclude}: se agrego {value!r} por negacion detectada en el texto")
        if value in current_in:
            current_in.remove(value)
    if ambiguous:
        wanted = ambiguous + [v for v in direct if v not in ambiguous]
        if sorted(wanted) != sorted(current_in):
            parsed.warnings.append(f"{include}: {current_in} reemplazado por {wanted} (alias ambiguo)")
            current_in = wanted
    setattr(parsed, include, current_in)
    setattr(parsed, exclude, current_out)


def _apply_period(text, parsed):
    if EXPLICIT_HOUR.search(text):
        return
    hits = {name: re.search(pattern, text) for name, pattern in PERIOD_PATTERNS.items()}
    hits = {name: m for name, m in hits.items() if m}
    if len(hits) != 1:
        return
    name, match = next(iter(hits.items()))
    start, end = PERIODS[name]
    if (parsed.time_from, parsed.time_to) != (start, end):
        parsed.warnings.append(f"horario fijado por el periodo {name!r}: {start}-{end}")
    parsed.time_from, parsed.time_to = start, end
    if parsed.semantic_text:
        left = re.sub(PERIOD_PATTERNS[name], " ", strip_accents(parsed.semantic_text))
        if len(re.sub(r"[^a-z]+", "", left)) < 3:
            parsed.semantic_text = None


def apply_rules(query, parsed):
    text = strip_accents(query)
    _apply_attribute(text, parsed, TYPE_ALIASES, "types", "exclude_types")
    _apply_attribute(text, parsed, COLOR_ALIASES, "colors", "exclude_colors")
    _apply_period(text, parsed)
    return parsed
