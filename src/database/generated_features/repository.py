"""Busqueda por similitud sobre florence.sqlite3 (exacta, en memoria con NumPy).

Los vectores estan normalizados, asi que la similitud coseno es un producto punto.
Con la variante "sentences" cada imagen puntua con su mejor oracion.
"""
from contextlib import closing
from pathlib import Path
import sqlite3

import numpy as np

from retrieval.fixed_source import IMAGE_DIR  # misma carpeta de recortes que usa la base fija
from retrieval.schemas import Candidate

# Configuracion
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = PROJECT_ROOT / "src/database/generated_features/data/florence.sqlite3"
VARIANT = "full"  # "full" o "sentences"; debe existir en la base (ver VARIANTS en build_embeddings.py)


class GeneratedSource:
    def __init__(self, db_path=DB_PATH, variant=VARIANT, image_dir=IMAGE_DIR):
        self.db_path = Path(db_path)
        self.variant = variant
        self.image_dir = Path(image_dir)
        self._load()

    def _load(self):
        with closing(sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)) as conn:
            meta = dict(conn.execute("SELECT key, value FROM meta"))
            rows = conn.execute(
                "SELECT image_id, text, vector FROM embeddings WHERE variant = ? ORDER BY image_id, chunk_idx",
                (self.variant,)).fetchall()
        if not rows:
            raise RuntimeError(f"No hay embeddings de la variante {self.variant!r} en {self.db_path}.")
        self.dim = int(meta["dim"])
        if any(len(r[2]) != self.dim * 4 for r in rows):
            raise RuntimeError("Hay vectores con tamano distinto a la dimension registrada en meta.")
        self.meta = meta
        self.matrix = np.frombuffer(b"".join(r[2] for r in rows), dtype=np.float32).reshape(len(rows), self.dim)
        self.chunk_texts = [r[1] for r in rows]
        owners = [r[0] for r in rows]
        starts = [i for i in range(len(owners)) if i == 0 or owners[i] != owners[i - 1]]
        self.image_ids = [owners[i] for i in starts]
        self.starts = np.asarray(starts, dtype=np.int64)
        self.ends = np.append(self.starts[1:], len(rows))
        self.index = {image_id: i for i, image_id in enumerate(self.image_ids)}

    def __len__(self):
        return len(self.image_ids)

    def search(self, query_vector, limit, allowed_ids=None):
        """Devuelve (candidatos, considerados). Considerados = imagenes sobre las que se busco."""
        query = np.asarray(query_vector, dtype=np.float32).ravel()
        if query.shape != (self.dim,):
            raise ValueError(f"El vector de consulta tiene forma {query.shape}; se esperaba ({self.dim},).")
        norm = np.linalg.norm(query)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("Vector de consulta invalido.")
        query = query / norm

        chunk_scores = self.matrix @ query
        scores = np.maximum.reduceat(chunk_scores, self.starts)  # mejor fragmento por imagen
        if allowed_ids is None:
            considered = len(self.image_ids)
        else:
            mask = np.zeros(len(self.image_ids), dtype=bool)
            for image_id in allowed_ids:
                position = self.index.get(image_id)
                if position is not None:
                    mask[position] = True
            considered = int(mask.sum())
            if considered == 0:
                return [], 0
            scores = np.where(mask, scores, -np.inf)

        order = np.argsort(-scores, kind="stable")[:max(0, min(limit, considered))]
        top_ids = [self.image_ids[i] for i in order]
        descriptions = self._descriptions(top_ids)
        candidates = []
        for position, image_id in zip(order, top_ids):
            best_chunk = self.starts[position] + int(np.argmax(chunk_scores[self.starts[position]:self.ends[position]]))
            attributes = {"description": descriptions.get(image_id, "")}
            if self.variant != "full":
                attributes["matched_text"] = self.chunk_texts[best_chunk]
            candidates.append(Candidate(image_id=image_id, image_path=str(self.image_dir / image_id),
                                        source="generated", score=float(scores[position]), attributes=attributes))
        return candidates, considered

    def _descriptions(self, image_ids):
        if not image_ids:
            return {}
        marks = ",".join("?" * len(image_ids))
        with closing(sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)) as conn:
            return dict(conn.execute(f"SELECT image_id, text FROM descriptions WHERE image_id IN ({marks})", image_ids))
