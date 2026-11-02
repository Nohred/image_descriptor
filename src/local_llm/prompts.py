"""Prompt y esquema de salida para convertir una consulta en filtros."""
import json


def build_output_schema(colors, types):
    str_or_null = {"type": ["string", "null"]}
    return {
        "type": "object",
        "properties": {
            "types": {"type": "array", "items": {"type": "string", "enum": list(types)}},
            "colors": {"type": "array", "items": {"type": "string", "enum": list(colors)}},
            "exclude_types": {"type": "array", "items": {"type": "string", "enum": list(types)}},
            "exclude_colors": {"type": "array", "items": {"type": "string", "enum": list(colors)}},
            "time_from": str_or_null,
            "time_to": str_or_null,
            "date_from": str_or_null,
            "date_to": str_or_null,
            "semantic_text": str_or_null,
            "unsupported_terms": {"type": "array", "items": {"type": "string"}},
            "limit": {"type": ["integer", "null"]},
        },
        "required": ["types", "colors", "exclude_types", "exclude_colors", "time_from", "time_to",
                     "date_from", "date_to", "semantic_text", "unsupported_terms", "limit"],
    }


def _hints(aliases):
    return "\n".join(f"- {k} -> {json.dumps(v)}" for k, v in aliases.items())


def _example(query, **fields):
    base = {"types": [], "colors": [], "exclude_types": [], "exclude_colors": [], "time_from": None,
            "time_to": None, "date_from": None, "date_to": None, "semantic_text": None,
            "unsupported_terms": [], "limit": None}
    base.update(fields)
    return f"Consulta: {query}\n{json.dumps(base, ensure_ascii=False)}"


def build_system_prompt(colors, types, periods, color_aliases, type_aliases, today):
    period_lines = "\n".join(f"- {name}: {a} a {b}" for name, (a, b) in periods.items())
    examples = "\n\n".join([
        _example("una van amarilla entre las 6 y las 9 de la noche",
                 types=["van"], colors=["yellow"], time_from="18:00", time_to="21:00"),
        _example("autos rojos que no sean sedan", colors=["red"], exclude_types=["sedan"]),
        _example("camionetas negras de noche", types=["suv", "pickup"], colors=["black"],
                 time_from="19:00", time_to="00:00"),
        _example("los 5 buses mas recientes", types=["bus"], limit=5),
        _example("una van blanca con el parabrisas roto", types=["van"], colors=["white"],
                 semantic_text="con el parabrisas roto", unsupported_terms=["parabrisas roto"]),
        _example("vehiculos azules de 10 de la noche a 2 de la manana", colors=["blue"],
                 time_from="22:00", time_to="02:00"),
    ])
    return f"""Eres un interprete de consultas para un buscador de imagenes de vehiculos.
Conviertes la consulta del usuario (en espanol) en un objeto JSON con filtros. No respondes la consulta ni inventas datos.

Fecha de hoy: {today}

Valores permitidos
- Colores: {", ".join(colors)}
- Tipos: {", ".join(types)}

Equivalencias de colores (espanol -> valores permitidos)
{_hints(color_aliases)}

Equivalencias de tipos (espanol -> valores permitidos)
{_hints(type_aliases)}

Reglas
1. "types" y "colors" listan lo que el usuario quiere. "exclude_types" y "exclude_colors" lo que NO quiere ("que no sean sedan", "pero no negras"); un valor excluido nunca va tambien en la lista de incluidos.
2. Usa siempre las equivalencias de arriba. Si un tipo es ambiguo (camioneta), incluye todos los valores de su equivalencia.
3. Hora del dia en HH:MM, 24 horas. time_from es inclusiva y time_to exclusiva. Si el rango cruza medianoche, time_from es mayor que time_to.
4. Si el usuario nombra un periodo del dia sin horas exactas, usa estos rangos exactos:
{period_lines}
5. "date_from" y "date_to" son fechas YYYY-MM-DD inclusivas, calculadas con la fecha de hoy. Si no hay fecha, null.
6. Si el usuario no menciona un filtro, usa lista vacia o null. No supongas filtros que no pidio.
7. Atributos que no sean color, tipo, hora o fecha (abolladuras, marca, placas, objetos) van en "unsupported_terms", y su descripcion en "semantic_text". Si no queda ninguna descripcion visual extra, "semantic_text" es null. Nunca copies en "semantic_text" la consulta completa, las horas ni el periodo del dia.
8. "limit" solo si el usuario pide una cantidad ("los 5", "hasta 10"). Si no, null.
9. Responde unicamente el JSON.

Ejemplos
{examples}"""
