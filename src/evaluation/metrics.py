"""Utilidades pequeñas para evaluación de detección sintética."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any


COCO_VEHICLE_CLASSES = {"car", "bus", "truck"}


def map_coco_vehicle_class(name: str) -> str | None:
    """Map COCO vehicle classes used by YOLO26 to the synthetic class."""
    return "vehicle" if name in COCO_VEHICLE_CLASSES else None


def group_detection_results(
    results: Iterable[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Group per-image results by background group and size category."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        objects = result.get("objects")
        if objects is None:
            objects = [result]
        for obj in objects:
            background = result.get("background", {})
            vehicle = result.get("vehicle", obj)
            grouped[
                (
                    background.get("group", background.get("source", "unknown")),
                    vehicle.get("size_category", "area_fraction"),
                )
            ].append(result)
    return dict(grouped)


def scale_range(scale: float) -> str:
    if scale <= 1:
        return "s<=1"
    if scale <= 2:
        return "1<s<=2"
    if scale <= 3:
        return "2<s<=3"
    if scale <= 4:
        return "3<s<=4"
    return "s>4"


def group_by_scale_and_variant(
    results: Iterable[dict[str, Any]],
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[(result["variant"], scale_range(result["vehicle"]["scale"]))].append(result)
    return dict(grouped)


def vehicle_primary_protocol(
    ground_truth: Iterable[dict[str, Any]],
    detections_by_image: dict[str, list[dict[str, float]]],
    iou_threshold: float = 0.5,
) -> dict[str, float]:
    """Report primary-box recall and extra detections separately."""
    total = matched = extras = 0
    for item in ground_truth:
        total += 1
        vehicle = item["vehicle"]
        truth = {
            "x": vehicle["position"]["x"],
            "y": vehicle["position"]["y"],
            "width": vehicle["final_size"]["width"],
            "height": vehicle["final_size"]["height"],
        }
        detections = detections_by_image.get(item["image_id"], [])
        if any(_box_iou(truth, detection) >= iou_threshold for detection in detections):
            matched += 1
        extras += max(0, len(detections) - 1)
    return {
        "images": total,
        "primary_recall": matched / total if total else 0.0,
        "extra_detections": extras,
        "iou_threshold": iou_threshold,
    }


def _box_iou(first: dict[str, float], second: dict[str, float]) -> float:
    left = max(first["x"], second["x"])
    top = max(first["y"], second["y"])
    right = min(first["x"] + first["width"], second["x"] + second["width"])
    bottom = min(first["y"] + first["height"], second["y"] + second["height"])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first["width"] * first["height"] + second["width"] * second["height"] - intersection
    return intersection / union if union else 0.0
