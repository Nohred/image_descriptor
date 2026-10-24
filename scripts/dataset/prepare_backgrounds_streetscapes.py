"""Prepare a JSONL index and contact sheets for Global Streetscapes."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw
import pandas as pd


# Configuración editable.
DOWNLOADED_CSV = "data/raw/streetscapes/downloaded.csv"
SELECTED_CSV = "data/raw/streetscapes/selected.csv"
EXCLUSION_FILE = None  # Ruta a un archivo con un UUID por línea; None no excluye.
MANIFEST_PATH = "data/backgrounds/streetscapes_manifest.jsonl"
PREVIEW_DIR = "artifacts/background_preview/streetscapes"
SAMPLE_SIZE = 24  # Número de fondos por source en cada hoja de contactos.
SEED = 0


def main() -> None:
    downloaded = Path(DOWNLOADED_CSV)
    selected = Path(SELECTED_CSV)
    for path in (downloaded, selected):
        if not path.is_file():
            raise FileNotFoundError(f"No se encontró el CSV de Streetscapes: {path}")
    downloads = pd.read_csv(downloaded, dtype={"uuid": str})
    contexts = pd.read_csv(selected, dtype={"uuid": str})
    required = {"uuid", "path", "width", "height"}
    missing = required - set(downloads.columns)
    if missing:
        raise ValueError(f"Faltan columnas en downloaded.csv: {sorted(missing)}")
    if "uuid" not in contexts:
        raise ValueError("selected.csv no contiene la columna uuid")
    excluded = set()
    if EXCLUSION_FILE:
        exclusion_path = Path(EXCLUSION_FILE)
        if not exclusion_path.is_file():
            raise FileNotFoundError(f"No se encontró el archivo de exclusión: {exclusion_path}")
        excluded = {line.strip() for line in exclusion_path.read_text().splitlines() if line.strip()}
    merged = downloads.merge(contexts, on="uuid", how="inner", suffixes=("_downloaded", ""))
    records = []
    for row in merged.to_dict("records"):
        if row["uuid"] in excluded:
            continue
        path = Path(row["path"])
        if not path.is_file():
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = image.size
        except (OSError, ValueError):
            continue
        attributes = {
            key: value
            for key, value in row.items()
            if key not in {"uuid", "source", "orig_id", "path", "width", "height",
                           "width_downloaded", "height_downloaded"}
            and pd.notna(value)
        }
        records.append({
            "uuid": row["uuid"],
            "source": row["source"],
            "orig_id": str(row["orig_id"]),
            "path": str(path),
            "width": width,
            "height": height,
            "attributes": attributes,
            "license_note": "pendiente de verificar",
        })
    manifest = Path(MANIFEST_PATH)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    rng = random.Random(SEED)
    by_source: dict[str, list[dict]] = {}
    for record in records:
        by_source.setdefault(record["source"], []).append(record)
    for source, items in by_source.items():
        selected_items = rng.sample(items, min(SAMPLE_SIZE, len(items)))
        sheet = Image.new("RGB", (720, 4 * 200), "white")
        draw = ImageDraw.Draw(sheet)
        for index, item in enumerate(selected_items):
            with Image.open(item["path"]) as image:
                thumb = image.convert("RGB")
            thumb.thumbnail((240, 170))
            x = (index % 3) * 240
            y = (index // 3) * 200
            sheet.paste(thumb, (x + (240 - thumb.width) // 2, y))
            draw.text((x + 4, y + 174), item["uuid"], fill="black")
        output = Path(PREVIEW_DIR)
        output.mkdir(parents=True, exist_ok=True)
        sheet.save(output / f"{source.lower()}.jpg", quality=95)
        print(f"{source}: {len(items)}")
    print(f"Fondos válidos: {len(records)}")
    print(f"Excluidos explícitamente: {len(excluded)}")
    print(f"Manifiesto: {manifest}")


if __name__ == "__main__":
    main()
