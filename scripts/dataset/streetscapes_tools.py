#!/usr/bin/env python3
"""Herramientas para preparar fondos de Global Streetscapes.

Subcomandos:
  inspect : muestra los valores reales de las columnas de contextual.csv
  select  : filtra los CSV y escribe selected.csv (uuid, source, orig_id, ...)
  verify  : revisa las imagenes descargadas y escribe downloaded.csv
"""
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

CHUNK = 500_000
KEY = ["uuid", "source", "orig_id"]
CTX_COLS = ["platform", "weather", "view_direction", "lighting_condition",
            "glare", "quality", "reflection", "pano_status"]
DEFAULT_FILTERS = ["lighting_condition~day", "quality~good", "platform~driving"]
VEHICLE_COLS = ["Bus", "Car", "Caravan", "Motorcycle", "Other-Vehicle",
                "Trailer", "Truck"]

# Configuración editable.
# Elige una operación: "inspect", "select" o "verify".
COMMAND = "select"

# Configuración de inspect.
INSPECT_CSV_DIR = "data/raw/streetscapes/csv"
INSPECT_ROWS = 1_000_000

# Configuración de select.
SELECT_CSV_DIR = "data/raw/streetscapes/csv"
SELECT_OUTPUT = "data/raw/streetscapes/selected.csv"
SELECT_N = 2000
SELECT_OVERSAMPLE = 1.25
SELECT_SEED = 0
SELECT_CANDIDATE_FRAC = 0.01
SELECT_MIN_SHORT_SIDE = 640
SELECT_FILTERS = []  # Ejemplo: ["weather~clear", "projection_type~perspective"].
SELECT_NO_DEFAULTS = False  # True usa únicamente SELECT_FILTERS.
SELECT_MAX_VEHICLE_SHARE = None  # None no usa segmentation.csv.

# Configuración de verify.
VERIFY_SELECTED = "data/raw/streetscapes/selected.csv"
VERIFY_IMAGES = "data/raw/streetscapes/images"
VERIFY_OUTPUT = "data/raw/streetscapes/downloaded.csv"
VERIFY_KEEP = 2000
VERIFY_MIN_SHORT_SIDE = 640


def read_chunks(path, usecols):
    return pd.read_csv(path, usecols=usecols, chunksize=CHUNK, dtype=str)


def parse_filter(text):
    col, _, val = text.partition("~")
    if col not in CTX_COLS or not val:
        raise SystemExit(f"Filtro invalido '{text}'. Formato: columna~texto, "
                         f"columnas validas: {CTX_COLS}")
    return col, val.lower()


def cmd_inspect(a):
    frames, seen = [], 0
    for chunk in read_chunks(Path(a.csv_dir) / "contextual.csv", CTX_COLS):
        frames.append(chunk)
        seen += len(chunk)
        if seen >= a.rows:
            break
    df = pd.concat(frames)
    print(f"Filas leidas: {len(df)} (las primeras del archivo; pueden no ser representativas)")
    for col in CTX_COLS:
        print(f"\n== {col}")
        print(df[col].str.lower().value_counts(dropna=False).head(15))


def cmd_select(a):
    rng = np.random.default_rng(a.seed)
    csv_dir = Path(a.csv_dir)
    filters = [] if a.no_defaults else [parse_filter(f) for f in DEFAULT_FILTERS]
    filters += [parse_filter(f) for f in a.filter]
    print("Filtros contextual:", filters, "| sin panoramicas")

    cands, total, passed = [], 0, 0
    for chunk in read_chunks(csv_dir / "contextual.csv", KEY + CTX_COLS):
        total += len(chunk)
        mask = chunk["pano_status"].str.lower().ne("true")
        for col, val in filters:
            mask &= chunk[col].str.lower().str.contains(val, na=False, regex=False)
        sub = chunk[mask]
        passed += len(sub)
        if len(sub):
            cands.append(sub[rng.random(len(sub)) < a.candidate_frac])
    print(f"contextual.csv: {total} filas, {passed} pasan los filtros")
    if not cands:
        raise SystemExit("Ninguna fila paso los filtros; revisa con 'inspect'.")
    cand = pd.concat(cands, ignore_index=True)
    print(f"Candidatos tras muestreo (frac={a.candidate_frac}): {len(cand)}")

    ids, meta = set(cand["uuid"]), []
    cols = ["uuid", "width", "height", "projection_type"]
    for chunk in read_chunks(csv_dir / "metadata_common_attributes.csv", cols):
        sub = chunk[chunk["uuid"].isin(ids)].copy()
        if sub.empty:
            continue
        for c in ("width", "height"):
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
        ok = sub[["width", "height"]].min(axis=1) >= a.min_short_side
        ok &= sub["projection_type"].str.lower().str.contains("perspective", na=False)
        meta.append(sub[ok])
    if not meta:
        raise SystemExit("Ningun candidato cumple tamano/proyeccion.")
    cand = cand.merge(pd.concat(meta), on="uuid")
    print(f"Tras tamano (lado menor >= {a.min_short_side}) y proyeccion perspectiva: {len(cand)}")

    if a.max_vehicle_share is not None:
        ids, rows = set(cand["uuid"]), []
        for chunk in read_chunks(csv_dir / "segmentation.csv",
                                 ["uuid", "Total"] + VEHICLE_COLS):
            sub = chunk[chunk["uuid"].isin(ids)].copy()
            if sub.empty:
                continue
            vals = sub[VEHICLE_COLS + ["Total"]].apply(pd.to_numeric, errors="coerce")
            sub["vehicle_share"] = vals[VEHICLE_COLS].sum(axis=1) / vals["Total"]
            rows.append(sub[["uuid", "vehicle_share"]])
        if not rows:
            raise SystemExit("segmentation.csv no contiene los candidatos.")
        cand = cand.merge(pd.concat(rows), on="uuid")
        cand = cand[cand["vehicle_share"] <= a.max_vehicle_share]
        print(f"Tras vehicle_share <= {a.max_vehicle_share}: {len(cand)}")

    n_req = math.ceil(a.n * a.oversample)
    if len(cand) < n_req:
        print(f"AVISO: solo hay {len(cand)} candidatos (pedidos {n_req}); "
              "sube --candidate-frac o relaja los filtros.")
    out = cand.drop_duplicates("uuid").sample(
        n=min(len(cand), n_req), random_state=a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(f"Escrito {a.out}: {len(out)} filas ({a.n} objetivo + margen para fallos)")
    print(out["source"].value_counts().to_string())


def cmd_verify(a):
    from PIL import Image
    sel = pd.read_csv(a.selected, dtype=str)
    root = Path(a.images)
    files = {p.stem: p for ext in ("*.jpeg", "*.jpg") for p in root.rglob(ext)}
    rows, missing, corrupt = [], 0, 0
    for uuid in sel["uuid"]:
        path = files.get(uuid)
        if path is None:
            missing += 1
            continue
        try:
            with Image.open(path) as im:
                im.verify()
            with Image.open(path) as im:
                w, h = im.size
        except Exception:
            corrupt += 1
            continue
        rows.append((uuid, str(path), w, h))
    df = pd.DataFrame(rows, columns=["uuid", "path", "width", "height"])
    print(f"Seleccionadas: {len(sel)} | validas: {len(df)} | "
          f"faltan: {missing} | corruptas: {corrupt}")
    if len(df):
        short = df[["width", "height"]].min(axis=1)
        q = np.percentile(short, [0, 5, 25, 50, 75, 95, 100])
        print("Lado menor [min,p5,p25,p50,p75,p95,max]:", " ".join(f"{x:.0f}" for x in q))
        print(f"Con lado menor < {a.min_short_side}: {(short < a.min_short_side).sum()}")
    df = df.head(a.keep)
    df.to_csv(a.out, index=False)
    print(f"Escrito {a.out}: {len(df)} filas")
    if len(df) < a.keep:
        print("AVISO: menos imagenes validas que el objetivo; repite la descarga.")


def main():
    if COMMAND == "inspect":
        args = SimpleNamespace(csv_dir=INSPECT_CSV_DIR, rows=INSPECT_ROWS)
        cmd_inspect(args)
    elif COMMAND == "select":
        args = SimpleNamespace(
            csv_dir=SELECT_CSV_DIR,
            out=SELECT_OUTPUT,
            n=SELECT_N,
            oversample=SELECT_OVERSAMPLE,
            seed=SELECT_SEED,
            candidate_frac=SELECT_CANDIDATE_FRAC,
            min_short_side=SELECT_MIN_SHORT_SIDE,
            filter=SELECT_FILTERS,
            no_defaults=SELECT_NO_DEFAULTS,
            max_vehicle_share=SELECT_MAX_VEHICLE_SHARE,
        )
        cmd_select(args)
    elif COMMAND == "verify":
        args = SimpleNamespace(
            selected=VERIFY_SELECTED,
            images=VERIFY_IMAGES,
            out=VERIFY_OUTPUT,
            keep=VERIFY_KEEP,
            min_short_side=VERIFY_MIN_SHORT_SIDE,
        )
        cmd_verify(args)
    else:
        raise ValueError("COMMAND debe ser 'inspect', 'select' o 'verify'")


if __name__ == "__main__":
    main()
