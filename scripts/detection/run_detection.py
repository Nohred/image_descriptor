#!/usr/bin/env python3
"""Ejecuta YOLO26 (n y/o s) sobre el dataset sintetico y evalua.

Modifica las variables de configuracion de este archivo antes de ejecutar:
    python scripts/run_detection.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.yolo_detector import load_detector  # noqa: E402
from src.evaluation.detection_eval import evaluate_synthetic, save_report  # noqa: E402

COLUMNS = ["detection_rate", "recall@0.5", "recall@0.75", "mean_iou_all",
           "AP50", "mAP50-95", "extra_detections_mean", "inference_ms_mean"]

# Configuracion editable.
MODELS = ["n", "s"]  # Opciones: ["n"], ["s"] o ["n", "s"].
DATA_VARIANTS = ["matched", "control"]  # Opciones: ["matched"], ["control"] o ambas.
DATA_ROOT = "data/synthetic"
OUTPUT_DIR = "artifacts/detection"
WEIGHTS = None  # Ruta a pesos propios; requiere seleccionar un solo modelo.
MIN_CONF = 0.25  # Confianza minima usada para evaluar una deteccion.
CONF_FLOOR = 0.05  # Umbral inferior de confianza para curvas y metricas.
IMAGE_SIZE = 640  # Tamano de entrada de YOLO en pixeles.
BATCH_SIZE = 16
DEVICE = None  # Opciones: None (automatico), "cpu", "0" o "cuda:0".
LIMIT = None  # Numero maximo de imagenes; None procesa todo el dataset.
NMS_FREE = False  # True desactiva la supresion no maxima durante la evaluacion.


def run(models=("n", "s"), data_variants=("matched", "control"), root="data/synthetic",
        out_dir="artifacts/detection", weights=None, **eval_kwargs) -> list[dict]:
    """Funcion reutilizable: devuelve una lista de informes (y los guarda en disco)."""
    reports = []
    for m in models:
        model = load_detector(m, weights=weights)
        for v in data_variants:
            rep = evaluate_synthetic(model, root=root, variant=v, **eval_kwargs)
            rep["model"] = f"yolo26{m}"
            save_report(rep, Path(out_dir) / f"yolo26{m}_{v}")
            reports.append(rep)
    return reports


def print_table(reports: list[dict]) -> None:
    head = f"{'modelo':9}{'datos':9}" + "".join(f"{c:>22}" for c in COLUMNS) + f"{'vram_mb':>10}"
    print(head)
    for r in reports:
        s = r["summary"]
        vals = "".join(f"{s[c]:>22.4f}" for c in COLUMNS)
        vram = r["vram_peak_mb"]
        print(f"{r['model']:9}{r['data_variant']:9}{vals}{(f'{vram:.0f}' if vram else 'n/a'):>10}")


def main() -> None:
    if WEIGHTS and len(MODELS) != 1:
        raise ValueError("WEIGHTS requiere seleccionar un solo modelo en MODELS")
    reports = run(
        MODELS,
        DATA_VARIANTS,
        DATA_ROOT,
        OUTPUT_DIR,
        WEIGHTS,
        min_conf=MIN_CONF,
        conf_floor=CONF_FLOOR,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        limit=LIMIT,
        nms_free=NMS_FREE,
    )
    print_table(reports)


if __name__ == "__main__":
    main()
