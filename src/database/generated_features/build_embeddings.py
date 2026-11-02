"""florence_results.json -> florence.sqlite3 (descripciones + embeddings).

Ejecutar: python src/database/generated_features/build_embeddings.py
Es incremental: no vuelve a embeber descripciones que no cambiaron.
"""
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
import sqlite3
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from local_llm.config import EMBED_DIM, EMBED_MODEL
from local_llm.embedder import DOC_PREFIX, QUERY_PREFIX, embed_texts

# Configuracion
INPUT_JSON = PROJECT_ROOT / "src/feature_generator/data/florence_results.json"
OUTPUT_DB = PROJECT_ROOT / "src/database/generated_features/data/florence.sqlite3"
VARIANTS = ["full", "sentences"]   # full: un vector por imagen; sentences: un vector por oracion
MIN_SENTENCE_CHARS = 15
BLOCK_TEXTS = 128                  # textos por bloque; cada bloque se confirma en la base
REBUILD = False                    # True: borra todo y re-embebe (obligatorio si cambias de modelo o dimension)

SCHEMA = """
CREATE TABLE IF NOT EXISTS descriptions (
    image_id TEXT PRIMARY KEY, text TEXT NOT NULL, text_sha256 TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS embeddings (
    image_id TEXT NOT NULL, variant TEXT NOT NULL, chunk_idx INTEGER NOT NULL,
    text TEXT NOT NULL, vector BLOB NOT NULL,
    PRIMARY KEY (image_id, variant, chunk_idx));
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def load_descriptions():
    with INPUT_JSON.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("florence_results.json debe ser un objeto {image_id: descripcion}.")
    descriptions, skipped = {}, []
    for image_id, text in data.items():
        if isinstance(text, str) and text.strip():
            descriptions[image_id] = text.strip()
        else:
            skipped.append(image_id)
    if not descriptions:
        raise ValueError("No hay descripciones validas en el JSON.")
    return descriptions, skipped


def make_chunks(variant, text):
    if variant == "full":
        return [text]
    if variant == "sentences":
        parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)]
        parts = [s for s in parts if len(s) >= MIN_SENTENCE_CHARS]
        return parts or [text]
    raise ValueError(f"Variante desconocida: {variant!r}")


def check_meta(conn):
    expected = {"embedding_model": EMBED_MODEL, "dim": str(EMBED_DIM),
                "doc_prefix": DOC_PREFIX, "query_prefix": QUERY_PREFIX}
    stored = dict(conn.execute("SELECT key, value FROM meta"))
    if not stored:
        conn.executemany("INSERT INTO meta VALUES (?, ?)",
                         list(expected.items()) + [("created_at", datetime.now(timezone.utc).isoformat())])
        return
    changed = {k: (stored.get(k), v) for k, v in expected.items() if stored.get(k) != v}
    if changed:
        raise RuntimeError(f"La base fue creada con otra configuracion {changed}. Pon REBUILD = True para rehacerla.")


def sync_descriptions(conn, descriptions):
    existing = dict(conn.execute("SELECT image_id, text_sha256 FROM descriptions"))
    hashes = {i: hashlib.sha256(t.encode("utf-8")).hexdigest() for i, t in descriptions.items()}
    changed = [i for i, h in hashes.items() if existing.get(i) != h]
    stale = [i for i in existing if i not in descriptions]
    for image_id in changed:
        conn.execute("DELETE FROM embeddings WHERE image_id = ?", (image_id,))
        conn.execute("INSERT OR REPLACE INTO descriptions VALUES (?, ?, ?)",
                     (image_id, descriptions[image_id], hashes[image_id]))
    for image_id in stale:
        conn.execute("DELETE FROM embeddings WHERE image_id = ?", (image_id,))
        conn.execute("DELETE FROM descriptions WHERE image_id = ?", (image_id,))
    conn.commit()
    print(f"Descripciones: {len(descriptions)} | nuevas o modificadas: {len(changed)} | eliminadas: {len(stale)}")


def safe_embed(texts):
    """Devuelve (vectores_o_None, errores). Si el lote falla, aisla el texto problematico."""
    try:
        return list(embed_texts(texts)), []
    except Exception:
        results, errors = [], []
        for index, text in enumerate(texts):
            try:
                results.append(embed_texts([text])[0])
            except Exception as error:
                results.append(None)
                errors.append((index, str(error)[:200]))
        return results, errors


def flush(conn, variant, rows, failures):
    vectors, errors = safe_embed([text for _, _, text in rows])
    failed_images = set()
    for index, message in errors:
        failures.append((variant, rows[index][0], rows[index][1], message))
        failed_images.add(rows[index][0])
    # Todo o nada por imagen: una imagen con fragmentos faltantes se reintenta completa en la siguiente ejecucion.
    conn.executemany(
        "INSERT OR REPLACE INTO embeddings VALUES (?, ?, ?, ?, ?)",
        [(image_id, variant, idx, text, vector.astype("float32").tobytes())
         for (image_id, idx, text), vector in zip(rows, vectors) if image_id not in failed_images])
    conn.commit()


def embed_variant(conn, variant, descriptions, failures):
    done = {r[0] for r in conn.execute("SELECT DISTINCT image_id FROM embeddings WHERE variant = ?", (variant,))}
    pending = [i for i in descriptions if i not in done]
    print(f"[{variant}] imagenes pendientes: {len(pending)}")
    rows, processed = [], 0
    for image_id in pending:  # bloques de imagenes completas: un corte no deja imagenes a medias
        rows.extend((image_id, idx, text) for idx, text in enumerate(make_chunks(variant, descriptions[image_id])))
        processed += 1
        if len(rows) >= BLOCK_TEXTS:
            flush(conn, variant, rows, failures)
            rows = []
            print(f"[{variant}] {processed}/{len(pending)}")
    if rows:
        flush(conn, variant, rows, failures)
        print(f"[{variant}] {processed}/{len(pending)}")


def main():
    descriptions, skipped = load_descriptions()
    if skipped:
        print(f"Entradas sin texto valido (omitidas): {len(skipped)} {skipped[:3]}")
    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    with closing(sqlite3.connect(OUTPUT_DB)) as conn:
        conn.executescript(SCHEMA)
        if REBUILD:
            conn.executescript("DELETE FROM embeddings; DELETE FROM descriptions; DELETE FROM meta;")
        check_meta(conn)
        sync_descriptions(conn, descriptions)
        for variant in VARIANTS:
            embed_variant(conn, variant, descriptions, failures)
        conn.commit()
        for variant in VARIANTS:
            images, chunks = conn.execute(
                "SELECT COUNT(DISTINCT image_id), COUNT(*) FROM embeddings WHERE variant = ?", (variant,)).fetchone()
            print(f"Resultado [{variant}]: {images}/{len(descriptions)} imagenes, {chunks} vectores")
    if failures:
        print(f"Fallos de embedding: {len(failures)} (se reintentan en la siguiente ejecucion)")
        for variant, image_id, idx, message in failures[:5]:
            print(f"  {variant} | {image_id} | chunk {idx} | {message}")
    print(f"Base generada: {OUTPUT_DB}")


if __name__ == "__main__":
    main()
