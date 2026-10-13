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
        for obj in result.get("objects", []):
            grouped[(result["background"]["group"], obj["size_category"])].append(result)
    return dict(grouped)
