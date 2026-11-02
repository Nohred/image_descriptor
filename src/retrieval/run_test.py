"""Prueba manual del pipeline en modo fixed. Ejecutar: python src/retrieval/run_test.py

Por cada consulta compara (1) los filtros que interpreta el LLM contra los esperados y
(2) el total de coincidencias contra el total que da la base con los filtros esperados.
"""
from pathlib import Path
import json
import sys
import urllib.error
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # permite importar retrieval y local_llm

from retrieval.fixed_source import FixedSource
from retrieval.schemas import ParsedQuery
from retrieval.service import SearchService

# Configuracion
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5:3b"   # nombre exacto segun `ollama list`
THINK = None                  # False para desactivar razonamiento en modelos que lo soportan
TIMEOUT_S = 120
SHOW_RESULTS = 3              # candidatos a imprimir por consulta
MODE = "fixed"

# (consulta, filtros esperados). Los totales esperados se calculan con la propia base.
# Las combinaciones con resultados salen de la tabla tipo x color de check_fixed_db.py.
TEST_QUERIES = [
    ("un camion amarillo", {"types": ["truck"], "colors": ["yellow"]}),
    ("camiones amarillos entre las 18:00 y las 21:00",
     {"types": ["truck"], "colors": ["yellow"], "time_from": "18:00", "time_to": "21:00"}),
    ("camionetas verdes", {"types": ["suv", "pickup"], "colors": ["green"]}),
    ("buses azules o verdes", {"types": ["bus"], "colors": ["blue", "green"]}),
    ("autos rojos que no sean sedan", {"colors": ["red"], "exclude_types": ["sedan"]}),
    ("vans que no sean grises", {"types": ["van"], "exclude_colors": ["gray"]}),
    ("sedanes negros de noche", {"types": ["sedan"], "colors": ["black"],
                                 "time_from": "19:00", "time_to": "00:00"}),
    ("suv blancos en la manana", {"types": ["suv"], "colors": ["white"],
                                  "time_from": "06:00", "time_to": "12:00"}),
    ("pickups verdes de 10 de la noche a 2 de la manana",
     {"types": ["pickup"], "colors": ["green"], "time_from": "22:00", "time_to": "02:00"}),
    ("los 5 buses mas recientes", {"types": ["bus"], "limit": 5}),
    ("hatchback rojos con el espejo roto", {"types": ["hatchback"], "colors": ["red"]}),
    ("sedanes blancos del 5 al 10 de octubre",
     {"types": ["sedan"], "colors": ["white"], "date_from": "2026-10-05", "date_to": "2026-10-10"}),
    # Casos sin resultados: la combinacion no existe en la base (total esperado: 0).
    ("una van amarilla", {"types": ["van"], "colors": ["yellow"]}),
    ("camiones verdes en la madrugada", {"types": ["truck"], "colors": ["green"],
                                          "time_from": "00:00", "time_to": "06:00"}),
]


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


def _norm(value):
    return sorted(value) if isinstance(value, list) else value


def compare(parsed, expected):
    got = parsed.to_dict()
    return {k: _norm(got.get(k)) == _norm(v) for k, v in expected.items()}


def main(chat_fn=chat_ollama, source=None):
    source = source or FixedSource()
    service = SearchService(source, chat_fn)
    fields_ok = fields_total = totals_ok = 0
    for query, expected in TEST_QUERIES:
        print(f"\n> {query}")
        expected_total = source.search(ParsedQuery(**expected), 1)[1]
        fields_total += len(expected)
        try:
            result = service.search(query, mode=MODE)
        except Exception as error:
            print(f"  ERROR: {type(error).__name__}: {error}")
            continue
        checks = compare(result.parsed, expected)
        fields_ok += sum(checks.values())
        wrong = [k for k, ok in checks.items() if not ok]
        totals_ok += result.total == expected_total
        shown = {k: v for k, v in result.parsed.to_dict().items() if v and k != "warnings"}
        print(f"  filtros: {json.dumps(shown, ensure_ascii=False)}")
        if result.parsed.warnings:
            print(f"  avisos: {result.parsed.warnings}")
        print(f"  interpretacion: {'OK' if not wrong else 'DIFERENTE en ' + str(wrong)}")
        print(f"  total: {result.total} (esperado {expected_total}) | mostrando {len(result.candidates)} | {result.elapsed_ms:.0f} ms")
        for c in result.candidates[:SHOW_RESULTS]:
            print(f"    {c.image_id} | {c.attributes}")
    print(f"\nCampos interpretados como se esperaba: {fields_ok}/{fields_total}")
    print(f"Consultas con el total esperado: {totals_ok}/{len(TEST_QUERIES)}")


if __name__ == "__main__":
    main()