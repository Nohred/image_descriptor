"""Prueba manual de los tres modos (fixed, semantic, hybrid). Ejecutar: python src/retrieval/run_test.py

- fixed:    compara filtros esperados y totales contra la base.
- semantic: precision@K por color y tipo (usando las etiquetas de la base fija como referencia),
            variantes full/sentences y consultas en ingles vs espanol; mas detalles para revision manual.
- hybrid:   compara filtros esperados y verifica que cada resultado cumpla las restricciones.
"""
from contextlib import closing
from pathlib import Path
import json
import sqlite3
import sys
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # permite importar retrieval, local_llm y database

from database.generated_features.repository import GeneratedSource
from retrieval import fixed_source as fs
from retrieval.fixed_source import FixedSource
from retrieval.schemas import COLOR_NAMES, TYPE_NAMES, ParsedQuery
from retrieval.service import SearchService

# Configuracion
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:3b"   # nombre exacto segun `ollama list`
THINK = None                  # False para desactivar razonamiento en modelos que lo soportan
TIMEOUT_S = 120
RUN_FIXED = True
RUN_SEMANTIC = True
RUN_HYBRID = True
SEMANTIC_VARIANTS = ["full", "sentences"]   # variantes a comparar (deben existir en la base)
HYBRID_VARIANT = "full"
PRECISION_K = 10              # K para precision@K en semantic
SHOW_RESULTS = 3              # candidatos a imprimir por consulta
SNIPPET_CHARS = 150           # largo del texto de descripcion que se imprime

# ---------------------------------------------------------------- consultas
# (consulta, filtros esperados). Los totales esperados se calculan con la propia base.
FIXED_QUERIES = [
    ("un camion amarillo", {"types": ["truck"], "colors": ["yellow"]}),
    ("camiones amarillos entre las 18:00 y las 21:00",
     {"types": ["truck"], "colors": ["yellow"], "time_from": "18:00", "time_to": "21:00"}),
    ("camionetas verdes", {"types": ["suv", "pickup"], "colors": ["green"]}),
    ("buses azules o verdes", {"types": ["bus"], "colors": ["blue", "green"]}),
    ("autos rojos que no sean hatchback", {"colors": ["red"], "exclude_types": ["hatchback"]}),
    ("vans que no sean grises", {"types": ["van"], "exclude_colors": ["gray"]}),
    ("sedanes negros de noche", {"types": ["sedan"], "colors": ["black"],
                                 "time_from": "19:00", "time_to": "00:00"}),
    ("suv blancos en la manana", {"types": ["suv"], "colors": ["white"],
                                  "time_from": "06:00", "time_to": "12:00"}),
    ("pickups verdes de 10 de la noche a 2 de la manana",
     {"types": ["pickup"], "colors": ["green"], "time_from": "22:00", "time_to": "02:00"}),
    ("los 3 camiones mas recientes", {"types": ["truck"], "limit": 3}),
    ("hatchback rojos con el espejo roto", {"types": ["hatchback"], "colors": ["red"]}),
    ("sedanes blancos del 5 al 10 de octubre",
     {"types": ["sedan"], "colors": ["white"], "date_from": "2026-10-05", "date_to": "2026-10-10"}),
    ("una van amarilla", {"types": ["van"], "colors": ["yellow"]}),                      # 0 resultados reales
    ("camiones verdes en la madrugada", {"types": ["truck"], "colors": ["green"],
                                          "time_from": "00:00", "time_to": "06:00"}),    # 0 resultados reales
]

# Pares ingles/espanol con el mismo significado. Referencia: etiquetas de la base fija.
SEMANTIC_PAIRS = [
    (("a black sedan on the road", "un sedan negro en la carretera"), {"type": "sedan", "color": "black"}),
    (("a white suv", "una suv blanca"), {"type": "suv", "color": "white"}),
    (("a green pickup truck", "una pickup verde"), {"type": "pickup", "color": "green"}),
    (("a blue bus", "un autobus azul"), {"type": "bus", "color": "blue"}),
    (("a red hatchback", "un hatchback rojo"), {"type": "hatchback", "color": "red"}),
    (("a yellow truck", "un camion amarillo"), {"type": "truck", "color": "yellow"}),
]
# Detalles sin etiqueta: se revisan a ojo (descripcion y, si puedes, la imagen).
SEMANTIC_DETAILS = [
    "a car with a spoiler on the back",
    "un carro con spoiler trasero",
    "a wet road after rain",
    "carretera mojada despues de la lluvia",
    "trees and bushes on both sides of the road",
    "una camioneta con portaequipajes en el techo",
    "a blurred car moving at high speed",
]

# (consulta, filtros esperados). Los resultados deben cumplir estos filtros.
HYBRID_QUERIES = [
    ("sedanes negros de noche con spoiler trasero",
     {"types": ["sedan"], "colors": ["black"], "time_from": "19:00", "time_to": "00:00"}),
    ("camionetas verdes con arboles a los lados", {"types": ["suv", "pickup"], "colors": ["green"]}),
    ("autobuses azules en la manana con la carretera mojada",
     {"types": ["bus"], "colors": ["blue"], "time_from": "06:00", "time_to": "12:00"}),
    ("suv blancos por la tarde en una carretera vacia",
     {"types": ["suv"], "colors": ["white"], "time_from": "12:00", "time_to": "19:00"}),
    ("camiones amarillos entre las 18:00 y las 21:00 con la carretera mojada",
     {"types": ["truck"], "colors": ["yellow"], "time_from": "18:00", "time_to": "21:00"}),
    ("autos rojos que no sean sedan con diseno deportivo",
     {"colors": ["red"], "exclude_types": ["sedan"]}),
    ("vans que no sean grises", {"types": ["van"], "exclude_colors": ["gray"]}),           # sin texto semantico
    ("una van amarilla con spoiler", {"types": ["van"], "colors": ["yellow"]}),             # filtros sin resultados
]


# ---------------------------------------------------------------- utilidades
def chat_ollama(system, user, schema):
    """Adaptador. Sustituyelo por una llamada a local_llm/client.py si prefieres tu cliente."""
    body = {"model": OLLAMA_MODEL, "stream": False, "format": schema,
            "options": {"temperature": 0},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if THINK is not None:
        body["think"] = THINK
    request = urllib.request.Request(OLLAMA_URL, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return json.load(response)["message"]["content"]
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code}: {error.read().decode(errors='replace')}") from error


def load_labels():
    """image_id -> (tipo, color) leyendo la base fija; sirve de referencia para medir semantic."""
    with closing(sqlite3.connect(f"file:{fs.DB_PATH}?mode=ro", uri=True)) as conn:
        rows = conn.execute(f'SELECT "{fs.IMAGE_COLUMN}", "{fs.TYPE_COLUMN}", "{fs.COLOR_COLUMN}" '
                            f'FROM "{fs.TABLE}"').fetchall()
    def name(value, names):
        position = int(value) - fs.ID_BASE
        return names[position] if 0 <= position < len(names) else None
    return {r[0]: (name(r[1], TYPE_NAMES), name(r[2], COLOR_NAMES)) for r in rows}


def _norm(value):
    return sorted(value) if isinstance(value, list) else value


def compare(parsed, expected):
    got = parsed.to_dict()
    return {k: _norm(got.get(k)) == _norm(v) for k, v in expected.items()}


def satisfies(attributes, expected):
    """Verifica que un candidato cumpla los filtros esperados (tipo, color, exclusiones, hora)."""
    if not attributes or "type" not in attributes:
        return None  # sin atributos: no se puede verificar
    t, c = attributes["type"], attributes["color"]
    if expected.get("types") and t not in expected["types"]:
        return False
    if expected.get("colors") and c not in expected["colors"]:
        return False
    if t in expected.get("exclude_types", []) or c in expected.get("exclude_colors", []):
        return False
    start, end = expected.get("time_from"), expected.get("time_to")
    if start:
        hhmm = attributes["captured_at"][11:16]
        inside = (start <= hhmm < end) if start < end else (hhmm >= start or hhmm < end)
        if not inside:
            return False
    return True


def snippet(candidate):
    text = candidate.attributes.get("matched_text") or candidate.attributes.get("description", "")
    text = " ".join(text.split())
    return text[:SNIPPET_CHARS] + ("..." if len(text) > SNIPPET_CHARS else "")


def show(candidates, with_description=True):
    for c in candidates[:SHOW_RESULTS]:
        extra = {k: v for k, v in c.attributes.items() if k in ("type", "color", "captured_at")}
        score = f"{c.score:.3f}" if c.score is not None else "-"
        print(f"    {c.image_id} | sim {score} | {extra}")
        if with_description and c.source == "generated":
            print(f"      {snippet(c)}")


def header(title):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


# ---------------------------------------------------------------- fixed
def run_fixed(service, source):
    header("MODO FIXED")
    fields_ok = fields_total = totals_ok = 0
    for query, expected in FIXED_QUERIES:
        print(f"\n> {query}")
        expected_total = source.search(ParsedQuery(**expected), 1)[1]
        fields_total += len(expected)
        try:
            result = service.search(query, mode="fixed")
        except Exception as error:
            print(f"  ERROR: {type(error).__name__}: {error}")
            continue
        wrong = [k for k, ok in compare(result.parsed, expected).items() if not ok]
        fields_ok += len(expected) - len(wrong)
        totals_ok += result.total == expected_total
        shown = {k: v for k, v in result.parsed.to_dict().items() if v and k != "warnings"}
        print(f"  filtros: {json.dumps(shown, ensure_ascii=False)}")
        if result.parsed.warnings:
            print(f"  avisos: {result.parsed.warnings}")
        print(f"  interpretacion: {'OK' if not wrong else 'DIFERENTE en ' + str(wrong)}")
        print(f"  total: {result.total} (esperado {expected_total}) | {result.elapsed_ms:.0f} ms")
        show(result.candidates, with_description=False)
    print(f"\n[fixed] campos correctos: {fields_ok}/{fields_total} | totales correctos: {totals_ok}/{len(FIXED_QUERIES)}")


# ---------------------------------------------------------------- semantic
def run_semantic(fixed, generated, chat_fn, labels):
    header(f"MODO SEMANTIC (precision@{PRECISION_K} contra las etiquetas de la base fija)")
    summary = {}
    for variant, source in generated.items():
        service = SearchService(fixed, chat_fn, generated_source=source)
        covered = sum(1 for i in source.image_ids if i in labels)
        print(f"\n--- variante {variant!r}: {len(source)} imagenes con descripcion, {covered} con etiqueta fija ---")
        stats = {"en": [], "es": []}
        for (english, spanish), expected in SEMANTIC_PAIRS:
            for lang, query in (("en", english), ("es", spanish)):
                result = service.search(query, mode="semantic", limit=PRECISION_K)
                labeled = [labels[c.image_id] for c in result.candidates if c.image_id in labels]
                if not labeled:
                    print(f"\n> [{lang}] {query}\n  sin resultados con etiqueta para evaluar")
                    continue
                type_hit = sum(t == expected["type"] for t, _ in labeled) / len(labeled)
                color_hit = sum(c == expected["color"] for _, c in labeled) / len(labeled)
                both = sum(t == expected["type"] and c == expected["color"] for t, c in labeled) / len(labeled)
                stats[lang].append((color_hit, type_hit, both))
                print(f"\n> [{lang}] {query}\n  P@{len(labeled)}: color {color_hit:.2f} | tipo {type_hit:.2f} | ambos {both:.2f}"
                      f" | {result.elapsed_ms:.0f} ms")
                show(result.candidates)
        summary[variant] = stats
        print(f"\n--- detalles para revision manual ({variant}) ---")
        for query in SEMANTIC_DETAILS:
            result = service.search(query, mode="semantic", limit=PRECISION_K)
            print(f"\n> {query}")
            show(result.candidates)
    print("\n[semantic] promedios (color / tipo / ambos):")
    for variant, stats in summary.items():
        for lang, values in stats.items():
            if values:
                avg = [sum(v[i] for v in values) / len(values) for i in range(3)]
                print(f"  {variant:10s} {lang}: {avg[0]:.2f} / {avg[1]:.2f} / {avg[2]:.2f}  ({len(values)} consultas)")


# ---------------------------------------------------------------- hybrid
def run_hybrid(fixed, generated, chat_fn):
    header(f"MODO HYBRID (variante {HYBRID_VARIANT!r})")
    service = SearchService(fixed, chat_fn, generated_source=generated)
    parse_ok = parse_total = valid = checked = empty = 0
    for query, expected in HYBRID_QUERIES:
        print(f"\n> {query}")
        parse_total += len(expected)
        try:
            result = service.search(query, mode="hybrid", limit=PRECISION_K)
        except Exception as error:
            print(f"  ERROR: {type(error).__name__}: {error}")
            continue
        wrong = [k for k, ok in compare(result.parsed, expected).items() if not ok]
        parse_ok += len(expected) - len(wrong)
        shown = {k: v for k, v in result.parsed.to_dict().items() if v and k != "warnings"}
        print(f"  filtros: {json.dumps(shown, ensure_ascii=False)}")
        if result.parsed.warnings:
            print(f"  avisos: {result.parsed.warnings}")
        print(f"  interpretacion: {'OK' if not wrong else 'DIFERENTE en ' + str(wrong)}")
        checks = [satisfies(c.attributes, expected) for c in result.candidates]
        checks = [c for c in checks if c is not None]
        valid += sum(checks)
        checked += len(checks)
        empty += not result.candidates
        status = f"{sum(checks)}/{len(checks)} cumplen los filtros esperados" if checks else "sin resultados"
        print(f"  imagenes consideradas: {result.total} | devueltas: {len(result.candidates)} | {status} | {result.elapsed_ms:.0f} ms")
        show(result.candidates)
    print(f"\n[hybrid] campos correctos: {parse_ok}/{parse_total} | resultados que cumplen filtros: {valid}/{checked}"
          f" | consultas vacias: {empty}/{len(HYBRID_QUERIES)}")


# ---------------------------------------------------------------- main
def main(chat_fn=chat_ollama, fixed=None, generated=None):
    fixed = fixed or FixedSource()
    if generated is None:
        generated = {}
        for variant in dict.fromkeys(SEMANTIC_VARIANTS + [HYBRID_VARIANT]):
            try:
                generated[variant] = GeneratedSource(variant=variant)
            except Exception as error:
                print(f"Variante {variant!r} no disponible: {error}")
    labels = load_labels() if RUN_SEMANTIC else {}
    if RUN_FIXED:
        run_fixed(SearchService(fixed, chat_fn), fixed)
    if RUN_SEMANTIC:
        run_semantic(fixed, {v: generated[v] for v in SEMANTIC_VARIANTS if v in generated}, chat_fn, labels)
    if RUN_HYBRID:
        if HYBRID_VARIANT in generated:
            run_hybrid(fixed, generated[HYBRID_VARIANT], chat_fn)
        else:
            print(f"\nHybrid omitido: la variante {HYBRID_VARIANT!r} no esta disponible.")


if __name__ == "__main__":
    main()