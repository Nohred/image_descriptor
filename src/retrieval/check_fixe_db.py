"""Diagnostico de la base fixed: distribucion tipo x color y rango de fechas. Ejecutar: python src/retrieval/check_fixed_db.py"""
from contextlib import closing
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval.fixed_source import DB_PATH, TABLE, IMAGE_COLUMN, COLOR_COLUMN, TYPE_COLUMN, TIME_COLUMN, ID_BASE
from retrieval.schemas import COLOR_NAMES, TYPE_NAMES

# Configuracion
SAMPLES_PER_CASE = 5
CASES_TO_SAMPLE = [("bus", "blue"), ("van", "white"), ("truck", "gray")]  # (tipo, color) a inspeccionar visualmente


def main():
    with closing(sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)) as conn:
        total, first, last = conn.execute(f"SELECT COUNT(*), MIN({TIME_COLUMN}), MAX({TIME_COLUMN}) FROM {TABLE}").fetchone()
        print(f"Filas: {total} | desde {first} hasta {last}")
        counts = {(t, c): n for t, c, n in conn.execute(
            f"SELECT {TYPE_COLUMN}, {COLOR_COLUMN}, COUNT(*) FROM {TABLE} GROUP BY 1, 2")}
        ids_t = sorted({t for t, _ in counts}); ids_c = sorted({c for _, c in counts})
        print("\nIDs presentes -> tipos:", ids_t, "| colores:", ids_c)
        print(f"Rango esperado con ID_BASE={ID_BASE}: tipos {ID_BASE}-{ID_BASE + len(TYPE_NAMES) - 1}, "
              f"colores {ID_BASE}-{ID_BASE + len(COLOR_NAMES) - 1}")
        header = "tipo \\ color".ljust(14) + "".join(n[:6].rjust(8) for n in COLOR_NAMES)
        print("\n" + header)
        for i, tname in enumerate(TYPE_NAMES):
            row = "".join(str(counts.get((i + ID_BASE, j + ID_BASE), 0)).rjust(8) for j in range(len(COLOR_NAMES)))
            print(tname.ljust(14) + row)
        for tname, cname in CASES_TO_SAMPLE:
            rows = conn.execute(
                f"SELECT {IMAGE_COLUMN} FROM {TABLE} WHERE {TYPE_COLUMN}=? AND {COLOR_COLUMN}=? LIMIT ?",
                (TYPE_NAMES.index(tname) + ID_BASE, COLOR_NAMES.index(cname) + ID_BASE, SAMPLES_PER_CASE)).fetchall()
            print(f"\nMuestras de {tname}/{cname}:", [r[0] for r in rows] or "ninguna")


if __name__ == "__main__":
    main()
