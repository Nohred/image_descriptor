"""Evaluacion de deteccion sobre el dataset sintetico.

Protocolo: por imagen se conserva SOLO el vehiculo detectado mas grande
(entre las detecciones con conf >= min_conf) y se compara con la caja
etiquetada (si hubiera varias, la mas grande). Las detecciones extra no se
descartan de la informacion: se cuentan aparte (n_extra).
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from src.detection.yolo_detector import (
    ImageDetections,
    detect_images,
    peak_vram_mb,
    reset_peak_vram,
    select_largest,
)

IOU_THRESHOLDS = tuple(round(float(t), 2) for t in np.arange(0.5, 0.96, 0.05))
SCALE_BUCKETS = (("s<=1", 0.0, 1.0), ("1<s<=2", 1.0, 2.0), ("2<s<=3", 2.0, 3.0),
                 ("3<s<=4", 3.0, 4.0), ("s>4", 4.0, math.inf))


def scale_bucket(scale: float | None) -> str:
    if scale is None:
        return "unknown"
    for name, lo, hi in SCALE_BUCKETS:
        if (scale > lo or lo == 0.0) and scale <= hi:
            return name
    return "unknown"


@dataclass
class Sample:
    image_id: str
    image_path: Path
    label_path: Path
    scale: float | None = None
    blur_factor: float | None = None


@dataclass
class PerImage:
    image_id: str
    scale: float | None
    scale_bucket: str
    blur_factor: float | None
    gt_xyxy: tuple | None
    pred_xyxy: tuple | None
    pred_conf: float | None
    pred_class: str | None
    iou: float
    n_candidates: int   # detecciones de vehiculo con conf >= min_conf
    n_extra: int        # candidatas menos la conservada
    inference_ms: float | None
    total_ms: float | None


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def read_gt_largest(label_path: Path, width: int, height: int):
    """Caja (x1, y1, x2, y2) en pixeles de la etiqueta YOLO mas grande."""
    boxes = []
    for line in Path(label_path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = (float(v) for v in parts[1:5])
        boxes.append(((cx - w / 2) * width, (cy - h / 2) * height,
                      (cx + w / 2) * width, (cy + h / 2) * height))
    if not boxes:
        return None
    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


def load_samples(root: str | Path, variant: str) -> list[Sample]:
    """Lee <root>/<variant>/manifest.jsonl y localiza imagenes y etiquetas."""
    base = Path(root) / variant
    samples = []
    for line in (base / "manifest.jsonl").read_text().splitlines():
        item = json.loads(line)
        image_id = item["image_id"]
        image_path = Path(item.get("file", ""))
        if not image_path.is_file():
            image_path = base / "images" / (image_id + ".png")
        samples.append(Sample(
            image_id=image_id,
            image_path=image_path,
            label_path=base / "labels" / f"{image_id}.txt",
            scale=(item.get("vehicle") or {}).get("scale"),
            blur_factor=item.get("background_blur_factor"),
        ))
    return samples


def match_largest(samples: Sequence[Sample], imgdets: Sequence[ImageDetections],
                  min_conf: float = 0.25) -> list[PerImage]:
    if len(samples) != len(imgdets):
        raise ValueError("samples e imgdets deben tener la misma longitud")
    rows = []
    for s, d in zip(samples, imgdets):
        gt = read_gt_largest(s.label_path, d.width, d.height)
        best = select_largest(d.detections, min_conf)
        n_cand = sum(x.conf >= min_conf for x in d.detections)
        iou = iou_xyxy(gt, best.xyxy) if (gt is not None and best is not None) else 0.0
        inf = d.speed_ms.get("inference")
        tot = sum(d.speed_ms.values()) if d.speed_ms else None
        rows.append(PerImage(
            image_id=s.image_id, scale=s.scale, scale_bucket=scale_bucket(s.scale),
            blur_factor=s.blur_factor, gt_xyxy=gt,
            pred_xyxy=best.xyxy if best else None,
            pred_conf=best.conf if best else None,
            pred_class=best.class_name if best else None,
            iou=iou, n_candidates=n_cand, n_extra=max(0, n_cand - 1),
            inference_ms=inf, total_ms=tot))
    return rows


def average_precision(conf: Sequence[float], tp: Sequence[bool], n_gt: int) -> float:
    """AP con interpolacion de 101 puntos (estilo COCO)."""
    if n_gt == 0:
        return float("nan")
    if len(conf) == 0:
        return 0.0
    order = np.argsort(-np.asarray(conf, float), kind="stable")
    tp = np.asarray(tp, float)[order]
    ctp, cfp = np.cumsum(tp), np.cumsum(1.0 - tp)
    recall = ctp / n_gt
    precision = ctp / np.maximum(ctp + cfp, 1e-9)
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    q = np.where(idx < len(recall), envelope[np.minimum(idx, len(recall) - 1)], 0.0)
    return float(q.mean())


def _pct(values, q):
    values = [v for v in values if v is not None]
    return float(np.percentile(values, q)) if values else float("nan")


def summarize(rows: Sequence[PerImage], iou_thresholds=IOU_THRESHOLDS) -> dict:
    n = len(rows)
    if n == 0:
        return {"n_images": 0}
    with_gt = [r for r in rows if r.gt_xyxy is not None]
    n_gt = len(with_gt)
    preds = [r for r in with_gt if r.pred_xyxy is not None]
    ious_all = [r.iou for r in with_gt]
    ap = {}
    for t in iou_thresholds:
        ap[t] = average_precision([r.pred_conf for r in preds],
                                  [r.iou >= t for r in preds], n_gt)

    def recall(t):
        return sum(r.iou >= t for r in with_gt) / n_gt if n_gt else float("nan")

    tp50 = sum(r.iou >= 0.5 for r in preds)
    inf = [r.inference_ms for r in rows]
    tot = [r.total_ms for r in rows]
    return {
        "n_images": n,
        "n_with_prediction": len(preds),
        "detection_rate": len(preds) / n_gt if n_gt else float("nan"),
        "mean_iou_all": float(np.mean(ious_all)) if ious_all else float("nan"),
        "mean_iou_detected": float(np.mean([r.iou for r in preds])) if preds else float("nan"),
        "recall@0.5": recall(0.5),
        "recall@0.75": recall(0.75),
        "precision@0.5": tp50 / len(preds) if preds else float("nan"),
        "AP50": ap.get(0.5, float("nan")),
        "AP75": ap.get(0.75, float("nan")),
        "mAP50-95": float(np.nanmean(list(ap.values()))) if ap else float("nan"),
        "extra_detections_mean": float(np.mean([r.n_extra for r in rows])),
        "images_with_extra_pct": 100.0 * sum(r.n_extra > 0 for r in rows) / n,
        "inference_ms_mean": float(np.nanmean([x for x in inf if x is not None])) if any(x is not None for x in inf) else float("nan"),
        "inference_ms_p95": _pct(inf, 95),
        "total_ms_mean": float(np.nanmean([x for x in tot if x is not None])) if any(x is not None for x in tot) else float("nan"),
    }


def summarize_by(rows: Sequence[PerImage], key: Callable[[PerImage], str] = lambda r: r.scale_bucket) -> dict:
    groups: dict[str, list[PerImage]] = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    order = [b[0] for b in SCALE_BUCKETS] + ["unknown"]
    keys = sorted(groups, key=lambda k: order.index(k) if k in order else 99)
    return {k: summarize(groups[k]) for k in keys}


def evaluate_synthetic(model, root: str | Path = "data/synthetic", variant: str = "matched",
                       *, min_conf: float = 0.25, conf_floor: float = 0.05,
                       imgsz: int = 640, batch: int = 16, device=None,
                       limit: int | None = None, nms_free: bool = False) -> dict:
    """Ejecuta el detector sobre una variante y evalua con el vehiculo mas grande."""
    if conf_floor > min_conf:
        raise ValueError("conf_floor debe ser <= min_conf")
    samples = load_samples(root, variant)
    if limit:
        samples = samples[:limit]
    reset_peak_vram()
    imgdets = detect_images(model, [s.image_path for s in samples], conf=conf_floor,
                            imgsz=imgsz, batch=batch, device=device, nms_free=nms_free)
    rows = match_largest(samples, imgdets, min_conf)
    return {
        "data_variant": variant,
        "min_conf": min_conf,
        "conf_floor": conf_floor,
        "imgsz": imgsz,
        "nms_free": nms_free,
        "vram_peak_mb": peak_vram_mb(),
        "summary": summarize(rows),
        "by_scale": summarize_by(rows),
        "per_image": [asdict(r) for r in rows],
    }


def save_report(report: dict, out_dir: str | Path) -> Path:
    """Escribe summary.json y per_image.csv en out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {k: v for k, v in report.items() if k != "per_image"}
    (out / "summary.json").write_text(json.dumps(meta, indent=2, default=str))
    rows = report["per_image"]
    if rows:
        with open(out / "per_image.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return out
