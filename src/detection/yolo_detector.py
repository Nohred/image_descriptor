"""Deteccion de vehiculos con YOLO26 (variantes n y s).

Pensado para reutilizarse en el pipeline final:

    model = load_detector("n")
    dets = detect_images(model, ["a.png", "b.png"])
    best = select_largest(dets[0].detections, min_conf=0.25)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence, Union

import numpy as np

VARIANT_WEIGHTS = {"n": "yolo26n.pt", "s": "yolo26s.pt"}
VEHICLE_NAMES = ("car", "bus", "truck")  # clases COCO tratadas como "vehicle"

Source = Union[str, Path, np.ndarray]  # ruta o arreglo BGR (alto, ancho, 3)


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    class_id: int
    class_name: str

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ImageDetections:
    source: str
    width: int
    height: int
    detections: list[Detection] = field(default_factory=list)
    speed_ms: dict = field(default_factory=dict)  # preprocess/inference/postprocess


def load_detector(variant: str = "n", weights: str | Path | None = None):
    """Carga YOLO26. 'weights' permite usar un archivo propio (p. ej. ajustado).

    Si los pesos no existen localmente, Ultralytics los descarga (requiere internet).
    """
    from ultralytics import YOLO

    if weights is None:
        if variant not in VARIANT_WEIGHTS:
            raise ValueError(f"variant debe ser una de {list(VARIANT_WEIGHTS)}")
        weights = VARIANT_WEIGHTS[variant]
    return YOLO(str(weights))


def resolve_device(device: str | int | None = None) -> str:
    if device is not None:
        return str(device)
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


def detect_images(
    model,
    sources: Sequence[Source],
    *,
    conf: float = 0.05,
    iou: float = 0.7,
    imgsz: int = 640,
    batch: int = 16,
    device: str | int | None = None,
    half: bool | None = None,
    vehicle_names: Sequence[str] = VEHICLE_NAMES,
    nms_free: bool = False,
    max_det: int = 300,
) -> list[ImageDetections]:
    """Detecta vehiculos en cada fuente y devuelve TODAS las detecciones >= conf.

    Usa un 'conf' bajo para poder variar el umbral despues (select_largest)
    sin volver a ejecutar el modelo. nms_free=True activa la cabeza sin NMS.
    """
    names = model.names
    keep_ids = {int(i) for i, n in names.items() if n in vehicle_names}
    if not keep_ids:
        raise ValueError(f"El modelo no tiene las clases {tuple(vehicle_names)}")

    device = resolve_device(device)
    if half is None:
        half = device.startswith("cuda") or device.isdigit()
    kwargs = dict(conf=conf, iou=iou, imgsz=imgsz, device=device, half=half,
                  max_det=max_det, batch=batch, verbose=False)
    if nms_free:
        kwargs["nms"] = False

    sources = list(sources)
    out: list[ImageDetections] = []
    for start in range(0, len(sources), batch):
        chunk = [str(s) if isinstance(s, (str, Path)) else s
                 for s in sources[start:start + batch]]
        results = model.predict(source=chunk, **kwargs)
        for k, res in enumerate(results):
            idx = start + k
            src = sources[idx]
            name = str(src) if isinstance(src, (str, Path)) else f"array_{idx}"
            h, w = res.orig_shape[:2]
            boxes = res.boxes
            xyxy = _to_numpy(boxes.xyxy).reshape(-1, 4)
            confs = _to_numpy(boxes.conf).reshape(-1)
            clss = _to_numpy(boxes.cls).reshape(-1).astype(int)
            dets = [
                Detection(float(b[0]), float(b[1]), float(b[2]), float(b[3]),
                          float(c), int(k_), names[int(k_)])
                for b, c, k_ in zip(xyxy, confs, clss) if int(k_) in keep_ids
            ]
            out.append(ImageDetections(name, int(w), int(h), dets,
                                       dict(getattr(res, "speed", {}) or {})))
    return out


def select_largest(detections: Sequence[Detection], min_conf: float = 0.25):
    """Devuelve la deteccion de mayor area con conf >= min_conf (o None)."""
    candidates = [d for d in detections if d.conf >= min_conf]
    return max(candidates, key=lambda d: (d.area, d.conf), default=None)


def detect_largest(model, sources: Sequence[Source], *, min_conf: float = 0.25, **kwargs):
    """Atajo para el pipeline: [(ImageDetections, Detection | None), ...]."""
    kwargs.setdefault("conf", min(min_conf, 0.05))
    dets = detect_images(model, sources, **kwargs)
    return [(d, select_largest(d.detections, min_conf)) for d in dets]


def reset_peak_vram() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def peak_vram_mb() -> float | None:
    """Pico de VRAM asignada por PyTorch desde el ultimo reset (None sin GPU)."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1024 ** 2
    except ImportError:
        pass
    return None
