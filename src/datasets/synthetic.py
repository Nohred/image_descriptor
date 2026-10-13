"""Generación reproducible de imágenes sintéticas para detección."""

from __future__ import annotations

import json
import math
import random
from io import BytesIO
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from src.datasets.veri import VeriRecord, load_veri_records


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    def iou(self, other: "Box") -> float:
        x1, y1 = max(self.x, other.x), max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = self.area + other.area - intersection
        return intersection / union if union else 0.0


def _size_category(area: int, thresholds: dict[str, int]) -> str:
    if area < thresholds["small_max"]:
        return "pequeño"
    if area <= thresholds["medium_max"]:
        return "mediano"
    return "grande"


def _sample_background(path: Path, size: tuple[int, int], rng: random.Random) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    side = min(image.size)
    left = rng.randint(0, image.width - side)
    top = rng.randint(0, image.height - side)
    image = image.crop((left, top, left + side, top + side)).resize(
        size, Image.Resampling.LANCZOS
    )
    return image


def _apply_background_options(image: Image.Image, options: dict[str, Any]) -> Image.Image:
    if options.get("background_degradation", False):
        image = image.filter(ImageFilter.GaussianBlur(options["background_blur_radius"]))
        if options.get("background_jpeg_quality"):
            buffer = BytesIO()
            image.save(buffer, format="JPEG", quality=options["background_jpeg_quality"])
            buffer.seek(0)
            image = Image.open(buffer).convert("RGB")
    if options.get("full_image_brightness_contrast", False):
        image = ImageEnhance.Brightness(image).enhance(options["brightness_factor"])
        image = ImageEnhance.Contrast(image).enhance(options["contrast_factor"])
    return image


def _transform_vehicle(
    record: VeriRecord,
    target_side: int,
    flip: bool,
    options: dict[str, Any],
) -> tuple[Image.Image, float]:
    with Image.open(record.image_path) as source:
        vehicle = source.convert("RGB")
    original_w, original_h = vehicle.size
    scale = min(target_side / max(original_w, original_h), 1.0)
    final_size = (
        max(1, round(original_w * scale)),
        max(1, round(original_h * scale)),
    )
    vehicle = vehicle.resize(final_size, Image.Resampling.LANCZOS)
    if flip:
        vehicle = ImageOps.mirror(vehicle)
    if options.get("edge_smoothing", False):
        vehicle = vehicle.filter(ImageFilter.GaussianBlur(radius=options["edge_radius"]))
    return vehicle, scale


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _yolo_line(box: Box, canvas: tuple[int, int]) -> str:
    width, height = canvas
    return "0 {:.8f} {:.8f} {:.8f} {:.8f}".format(
        (box.x + box.width / 2) / width,
        (box.y + box.height / 2) / height,
        box.width / width,
        box.height / height,
    )


def generate(config: dict[str, Any]) -> list[dict[str, Any]]:
    rng = random.Random(config["seed"])
    canvas = (config["canvas"]["width"], config["canvas"]["height"])
    output = Path(config["output"]["root"])
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "labels").mkdir(parents=True, exist_ok=True)
    records = load_veri_records(config["veri_root"])
    allowed_splits = ["test"]
    if config["source"]["allow_train_identities"]:
        allowed_splits.append("train")
    sources = [record for record in records if record.split in allowed_splits]
    backgrounds: dict[str, list[Path]] = {}
    for group in config["backgrounds"]["groups"]:
        group_dir = Path(config["backgrounds"]["output_root"]) / group
        backgrounds[group] = sorted(group_dir.glob("*"))
        if not backgrounds[group]:
            raise FileNotFoundError(
                f"No hay fondos en {group_dir}. Ejecuta prepare_backgrounds.py primero."
            )
    weights = config["backgrounds"]["weights"]
    groups = list(weights)
    manifest: list[dict[str, Any]] = []
    thresholds = config["size_categories"]
    for index in range(config["count"]):
        group = rng.choices(groups, weights=[weights[g] for g in groups])[0]
        background_path = rng.choice(backgrounds[group])
        image = _apply_background_options(
            _sample_background(background_path, canvas, rng), config["options"]
        )
        boxes: list[Box] = []
        objects = rng.randint(1, config["objects"]["max_per_image"])
        image_entries: list[dict[str, Any]] = []
        for _ in range(objects):
            record = rng.choice(sources)
            target = round(math.exp(rng.uniform(
                math.log(config["scale"]["target_side_min"]),
                math.log(config["scale"]["target_side_max"]),
            )))
            flip = rng.random() < config["objects"]["horizontal_flip_probability"]
            vehicle, scale = _transform_vehicle(record, target, flip, config["options"])
            if vehicle.width > canvas[0] or vehicle.height > canvas[1]:
                continue
            placed = None
            for _attempt in range(config["objects"]["placement_attempts"]):
                candidate = Box(
                    rng.randint(0, canvas[0] - vehicle.width),
                    rng.randint(0, canvas[1] - vehicle.height),
                    vehicle.width,
                    vehicle.height,
                )
                if all(candidate.iou(previous) <= config["objects"]["max_iou"] for previous in boxes):
                    placed = candidate
                    break
            if placed is None:
                continue
            image.paste(vehicle, (placed.x, placed.y))
            boxes.append(placed)
            image_entries.append({
                "veri_image": record.image_name,
                "identity": record.vehicle_id,
                "color": record.color,
                "type": record.vehicle_type,
                "original_size": {
                    "width": _image_size(record.image_path)[0],
                    "height": _image_size(record.image_path)[1],
                },
                "scale": scale,
                "final_size": {"width": placed.width, "height": placed.height, "area": placed.area},
                "size_category": _size_category(placed.area, thresholds),
                "position": {"x": placed.x, "y": placed.y},
                "horizontal_flip": flip,
            })
        image_id = f"synthetic_{index:06d}"
        image_path = output / "images" / f"{image_id}.png"
        label_path = output / "labels" / f"{image_id}.txt"
        image.save(image_path, format="PNG")
        label_path.write_text("\n".join(_yolo_line(box, canvas) for box in boxes) + "\n", encoding="utf-8")
        manifest.append({
            "image_id": image_id,
            "file": str(image_path),
            "background": {"file": str(background_path), "group": group},
            "objects": image_entries,
            "source_split": allowed_splits,
            "options": config["options"],
            "seed": config["seed"],
        })
    (output / "data.yaml").write_text(
        "path: .\ntrain: images\nval: images\nnames:\n  0: vehicle\n",
        encoding="utf-8",
    )
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for item in manifest:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return manifest
