#!/usr/bin/env python3
"""Ejecuta YOLO26 (n y/o s) sobre el dataset sintetico y evalua.

Ejemplo:
    python scripts/run_detection.py --models n s --data-variants matched control
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detection.yolo_detector import load_detector  # noqa: E402
from src.evaluation.detection_eval import evaluate_synthetic, save_report  # noqa: E402

COLUMNS = ["detection_rate", "recall@0.5", "recall@0.75", "mean_iou_all",
           "AP50", "mAP50-95", "extra_detections_mean", "inference_ms_mean"]


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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models", nargs="+", default=["n", "s"], choices=["n", "s"])
    p.add_argument("--data-variants", nargs="+", default=["matched", "control"])
    p.add_argument("--root", default="data/synthetic")
    p.add_argument("--out", default="artifacts/detection")
    p.add_argument("--weights", default=None, help="pesos propios (solo con un modelo)")
    p.add_argument("--min-conf", type=float, default=0.25)
    p.add_argument("--conf-floor", type=float, default=0.05)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--nms-free", action="store_true")
    a = p.parse_args()
    if a.weights and len(a.models) != 1:
        p.error("--weights requiere un solo valor en --models")
    reports = run(a.models, a.data_variants, a.root, a.out, a.weights,
                  min_conf=a.min_conf, conf_floor=a.conf_floor, imgsz=a.imgsz,
                  batch=a.batch, device=a.device, limit=a.limit, nms_free=a.nms_free)
    print_table(reports)


if __name__ == "__main__":
    main()
