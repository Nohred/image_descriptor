"""Acceso de solo lectura a la base de fixed features con filtros parametrizados.

La tabla guarda colorID y typeID como enteros; aqui se traducen desde y hacia los nombres de
COLOR_NAMES / TYPE_NAMES (schemas.py) usando la posicion en cada lista (+ ID_BASE).
"""
from contextlib import closing
from pathlib import Path
import re
import sqlite3

from retrieval.schemas import COLOR_NAMES, TYPE_NAMES, Candidate

# Configuracion
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "src/database/fixed_features/data/vehicles.sqlite3"
IMAGE_DIR = PROJECT_ROOT / "VeRi/image_test"  # carpeta donde estan los recortes; ajustala
TABLE = "vehicles"
IMAGE_COLUMN = "imageName"
COLOR_COLUMN = "colorID"
TYPE_COLUMN = "typeID"
TIME_COLUMN = "captured_at"  # texto ISO: 2026-10-17T18:42:09-06:00
ID_BASE = 0  # 0 si colorID/typeID empiezan en 0 (indice de la lista); 1 si empiezan en 1

_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _ident(name):
    if not _IDENT.match(name):
        raise ValueError(f"Identificador SQL invalido: {name!r}")
    return f'"{name}"'


def _to_ids(names, vocabulary):
    return [vocabulary.index(name) + ID_BASE for name in names]


def _to_name(value, vocabulary):
    position = int(value) - ID_BASE
    return vocabulary[position] if 0 <= position < len(vocabulary) else f"unknown({value})"


class FixedSource:
    def __init__(self, db_path=DB_PATH, image_dir=IMAGE_DIR):
        self.db_path = Path(db_path)
        self.image_dir = Path(image_dir)
        self.t, self.i = _ident(TABLE), _ident(IMAGE_COLUMN)
        self.ty, self.co, self.ts = _ident(TYPE_COLUMN), _ident(COLOR_COLUMN), _ident(TIME_COLUMN)

    def _where(self, q):
        clauses, params = [], []

        def membership(column, ids, negate):
            marks = ",".join("?" * len(ids))
            clauses.append(f"{column} {'NOT IN' if negate else 'IN'} ({marks})")
            params.extend(ids)

        if q.types:
            membership(self.ty, _to_ids(q.types, TYPE_NAMES), False)
        if q.exclude_types:
            membership(self.ty, _to_ids(q.exclude_types, TYPE_NAMES), True)
        if q.colors:
            membership(self.co, _to_ids(q.colors, COLOR_NAMES), False)
        if q.exclude_colors:
            membership(self.co, _to_ids(q.exclude_colors, COLOR_NAMES), True)
        if q.date_from:
            clauses.append(f"substr({self.ts}, 1, 10) >= ?")
            params.append(q.date_from)
        if q.date_to:
            clauses.append(f"substr({self.ts}, 1, 10) <= ?")
            params.append(q.date_to)
        if q.time_from:
            hhmm = f"substr({self.ts}, 12, 5)"
            if q.time_from < q.time_to:
                clauses.append(f"({hhmm} >= ? AND {hhmm} < ?)")
            else:
                clauses.append(f"({hhmm} >= ? OR {hhmm} < ?)")
            params.extend([q.time_from, q.time_to])
        return (" WHERE " + " AND ".join(clauses)) if clauses else "", params

    def search(self, parsed, limit):
        where, params = self._where(parsed)
        uri = f"file:{self.db_path}?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            total = conn.execute(f"SELECT COUNT(*) FROM {self.t}{where}", params).fetchone()[0]
            rows = conn.execute(
                f"SELECT {self.i}, {self.ty}, {self.co}, {self.ts} FROM {self.t}{where} "
                f"ORDER BY {self.ts} DESC, {self.i} LIMIT ?", params + [limit]).fetchall()
        candidates = [
            Candidate(
                image_id=str(name), image_path=str(self.image_dir / str(name)), source="fixed",
                attributes={"type": _to_name(type_id, TYPE_NAMES), "color": _to_name(color_id, COLOR_NAMES),
                            "captured_at": stamp})
            for name, type_id, color_id, stamp in rows
        ]
        return candidates, total
