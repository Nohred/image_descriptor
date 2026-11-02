"""Vocabulario, aliases y objetos compartidos del modulo retrieval."""
from dataclasses import dataclass, field, asdict
import unicodedata

COLOR_NAMES = ["yellow", "orange", "green", "gray", "red", "blue", "white", "golden", "brown", "black"]
TYPE_NAMES = ["sedan", "suv", "van", "hatchback", "mpv", "pickup", "bus", "truck", "estate"]

# Alias en espanol (sin acentos, minusculas). Cada alias apunta a una lista de valores validos.
# Los alias ambiguos (p. ej. "camioneta") devuelven varios tipos: editalos segun tu criterio.
COLOR_ALIASES = {
    "amarillo": ["yellow"], "amarilla": ["yellow"],
    "naranja": ["orange"], "anaranjado": ["orange"], "anaranjada": ["orange"],
    "verde": ["green"],
    "gris": ["gray"], "plata": ["gray"], "plateado": ["gray"], "plateada": ["gray"],
    "rojo": ["red"], "roja": ["red"],
    "azul": ["blue"],
    "blanco": ["white"], "blanca": ["white"],
    "dorado": ["golden"], "dorada": ["golden"], "oro": ["golden"],
    "cafe": ["brown"], "marron": ["brown"],
    "negro": ["black"], "negra": ["black"],
}
TYPE_ALIASES = {
    "sedan": ["sedan"],
    "suv": ["suv"],
    "van": ["van"], "furgoneta": ["van"], "furgon": ["van"],
    "hatchback": ["hatchback"], "compacto": ["hatchback"],
    "mpv": ["mpv"], "minivan": ["mpv"], "monovolumen": ["mpv"],
    "pickup": ["pickup"], "pick-up": ["pickup"],
    "camioneta": ["suv", "pickup"],
    "bus": ["bus"], "autobus": ["bus"],
    "truck": ["truck"], "camion": ["truck"],
    "estate": ["estate"], "station wagon": ["estate"], "familiar": ["estate"],
}

# Periodos del dia: [desde, hasta). Un periodo que cruza medianoche termina en 00:00.
PERIODS = {
    "madrugada": ("00:00", "06:00"),
    "manana": ("06:00", "12:00"),
    "tarde": ("12:00", "19:00"),
    "noche": ("19:00", "00:00"),
}

DEFAULT_LIMIT = 20
MAX_LIMIT = 100

MODES = ("fixed", "semantic", "hybrid")
IMPLEMENTED_MODES = ("fixed",)  # semantic e hybrid dependen de generated_features


def strip_accents(text):
    text = unicodedata.normalize("NFKD", text.strip().lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def map_values(values, valid, aliases):
    """Devuelve (valores_validos_sin_duplicar, rechazados)."""
    accepted, rejected = [], []
    for raw in values or []:
        key = strip_accents(str(raw))
        if key in valid:
            found = [key]
        else:
            found = aliases.get(key) or aliases.get(key.rstrip("s")) or []
        if not found:
            rejected.append(str(raw))
        for value in found:
            if value not in accepted:
                accepted.append(value)
    return accepted, rejected


@dataclass
class ParsedQuery:
    types: list = field(default_factory=list)
    colors: list = field(default_factory=list)
    exclude_types: list = field(default_factory=list)
    exclude_colors: list = field(default_factory=list)
    time_from: str = None        # "HH:MM", hora del dia, inclusiva
    time_to: str = None          # "HH:MM", exclusiva; si time_from > time_to cruza medianoche
    date_from: str = None        # "YYYY-MM-DD", inclusiva
    date_to: str = None          # "YYYY-MM-DD", inclusiva
    semantic_text: str = None    # DECLARADO: lo usaran los modos semantic/hybrid (generated_features)
    unsupported_terms: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    limit: int = None

    def has_filters(self):
        return any([self.types, self.colors, self.exclude_types, self.exclude_colors,
                    self.time_from, self.date_from, self.date_to])

    def to_dict(self):
        return asdict(self)


@dataclass
class Candidate:
    image_id: str
    image_path: str
    source: str                  # "fixed" o "generated"
    score: float = None          # None en fixed; distancia/similitud en generated
    attributes: dict = field(default_factory=dict)


@dataclass
class SearchResult:
    mode: str
    query: str
    parsed: ParsedQuery
    candidates: list
    total: int
    elapsed_ms: float