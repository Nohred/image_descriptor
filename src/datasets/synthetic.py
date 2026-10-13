"""Generación reproducible de las variantes V2 de Streetscapes."""

from __future__ import annotations

import csv
from io import BytesIO
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from src.datasets.veri import VeriRecord, load_veri_records


VARIANTS = ("matched", "control")


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class Background:
    uuid: str
    source: str
    orig_id: str
    path: Path
    width: int
    height: int
    attributes: dict[str, Any]


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"No se encontró {description}: {path}")


def load_backgrounds(manifest_path: str | Path) -> list[Background]:
    _require_file(Path(manifest_path), "el manifiesto de Streetscapes")
    backgrounds = []
    for line in Path(manifest_path).read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        path = Path(item["path"])
        _require_file(path, f"el fondo {item['uuid']}")
        backgrounds.append(
            Background(
                uuid=item["uuid"],
                source=item["source"],
                orig_id=str(item["orig_id"]),
                path=path,
                width=int(item["width"]),
                height=int(item["height"]),
                attributes=item.get("attributes", {}),
            )
        )
    if not backgrounds:
        raise ValueError(f"El manifiesto no contiene fondos: {manifest_path}")
    return backgrounds


def _load_contextual(selected_path: Path) -> dict[str, dict[str, Any]]:
    with selected_path.open(newline="", encoding="utf-8") as handle:
        return {
            row["uuid"]: {
                key: value
                for key, value in row.items()
                if key not in {"uuid", "source", "orig_id", "width", "height"}
            }
            for row in csv.DictReader(handle)
        }


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _prepare_background(
    background: Background,
    canvas: tuple[int, int],
    mode: str,
    rng: random.Random,
) -> tuple[Image.Image, dict[str, int]]:
    with Image.open(background.path) as source:
        image = source.convert("RGB")
    if mode == "resize_crop":
        scale = max(canvas[0] / image.width, canvas[1] / image.height)
        resized = image.resize(
            (math.ceil(image.width * scale), math.ceil(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
        window_width, window_height = canvas
        max_left = resized.width - window_width
        max_top = resized.height - window_height
        left = rng.randint(0, max_left)
        top = rng.randint(0, max_top)
        window = resized.crop((left, top, left + window_width, top + window_height))
        return window, {
            "mode": mode,
            "source_width": image.width,
            "source_height": image.height,
            "resized_width": resized.width,
            "resized_height": resized.height,
            "left": left,
            "top": top,
        }
    if mode == "native_crop":
        if min(image.size) < min(canvas):
            raise ValueError(f"El fondo es menor que el lienzo: {background.path}")
        left = rng.randint(0, image.width - canvas[0])
        top = rng.randint(0, image.height - canvas[1])
        return image.crop((left, top, left + canvas[0], top + canvas[1])), {
            "mode": mode,
            "source_width": image.width,
            "source_height": image.height,
            "resized_width": image.width,
            "resized_height": image.height,
            "left": left,
            "top": top,
        }
    raise ValueError(f"Modo de fondo no soportado: {mode}")


def _size_category(area_fraction: float) -> str:
    if area_fraction < 0.4:
        return "pequeño"
    if area_fraction <= 0.6:
        return "mediano"
    return "grande"


def _gradient_mask(size: tuple[int, int], width: int, smoothstep: bool) -> Image.Image:
    image = Image.new("L", size, 255)
    pixels = image.load()
    for y in range(size[1]):
        for x in range(size[0]):
            distance = min(x, y, size[0] - 1 - x, size[1] - 1 - y)
            value = min(1.0, distance / width) if width else 1.0
            if smoothstep:
                value = value * value * (3.0 - 2.0 * value)
            pixels[x, y] = round(value * 255)
    return image


def _transform_vehicle(
    record: VeriRecord,
    scale: float,
    flip: bool,
    gradient_width: int,
    smoothstep: bool,
) -> tuple[Image.Image, Image.Image]:
    with Image.open(record.image_path) as source:
        vehicle = source.convert("RGB")
    final_size = (
        max(1, round(vehicle.width * scale)),
        max(1, round(vehicle.height * scale)),
    )
    vehicle = vehicle.resize(final_size, Image.Resampling.LANCZOS)
    if flip:
        vehicle = ImageOps.mirror(vehicle)
    return vehicle, _gradient_mask(final_size, gradient_width, smoothstep)


def _blur_background(image: Image.Image, blur_factor: float) -> Image.Image:
    if blur_factor <= 1:
        return image
    reduced = image.resize(
        (max(1, round(image.width / blur_factor)), max(1, round(image.height / blur_factor))),
        Image.Resampling.LANCZOS,
    )
    return reduced.resize(image.size, Image.Resampling.LANCZOS)


def _apply_background_options(image: Image.Image, options: dict[str, Any]) -> Image.Image:
    if options.get("background_noise", False):
        noise = Image.effect_noise(image.size, float(options.get("noise_sigma", 4.0))).convert("RGB")
        image = Image.blend(image, noise, 0.08)
    if options.get("background_jpeg", False):
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=int(options.get("jpeg_quality", 75)))
        buffer.seek(0)
        image = Image.open(buffer).convert("RGB")
    if options.get("full_image_brightness_contrast", False):
        image = ImageEnhance.Brightness(image).enhance(float(options.get("brightness_factor", 1.0)))
        image = ImageEnhance.Contrast(image).enhance(float(options.get("contrast_factor", 1.0)))
    return image


def _yolo_line(box: Box, canvas: tuple[int, int]) -> str:
    return "0 {:.8f} {:.8f} {:.8f} {:.8f}".format(
        (box.x + box.width / 2) / canvas[0],
        (box.y + box.height / 2) / canvas[1],
        box.width / canvas[0],
        box.height / canvas[1],
    )


def _write_data_yaml(path: Path) -> None:
    path.write_text("path: .\ntrain: images\nval: images\nnames:\n  0: vehicle\n", encoding="utf-8")


def _select_background(backgrounds: list[Background], rng: random.Random) -> Background:
    return backgrounds[rng.randrange(len(backgrounds))]


def generate(config: dict[str, Any]) -> list[dict[str, Any]]:
    canvas = (int(config["canvas"]["width"]), int(config["canvas"]["height"]))
    backgrounds = load_backgrounds(config["backgrounds"]["manifest"])
    all_records = load_veri_records(config["veri_root"])
    splits = ["test"] + (["train"] if config["source"]["allow_train_identities"] else [])
    sources = [record for record in all_records if record.split in splits]
    if not sources:
        raise ValueError("No hay recortes de VeRi disponibles para los splits configurados")
    output_root = Path(config["output"]["root"])
    for variant in VARIANTS:
        (output_root / variant / "images").mkdir(parents=True, exist_ok=True)
        (output_root / variant / "labels").mkdir(parents=True, exist_ok=True)
        _write_data_yaml(output_root / variant / "data.yaml")
    used_images: set[str] = set()
    manifests: dict[str, list[dict[str, Any]]] = {variant: [] for variant in VARIANTS}
    rejections_total = 0
    size_mode = config["size"]["size_mode"]
    index = 0
    while index < int(config["count"]):
        rng = random.Random(f"{config['seed']}:{index}")
        background = _select_background(backgrounds, rng)
        background_image, crop = _prepare_background(
            background, canvas, config["backgrounds"]["mode"], rng
        )
        background_image = _apply_background_options(background_image, config["options"])
        while True:
            available = [record for record in sources if record.image_name not in used_images]
            if not available:
                raise ValueError("No hay suficientes recortes únicos de VeRi")
            record = available[rng.randrange(len(available))]
            original_width, original_height = _image_size(record.image_path)
            original_area = original_width * original_height
            target_fraction = rng.uniform(*config["size"]["area_fraction_range"])
            if size_mode == "area_fraction":
                scale = math.sqrt(target_fraction * canvas[0] * canvas[1] / original_area)
            elif size_mode == "target_side":
                target_side = rng.uniform(*config["size"]["target_side_range"])
                scale = target_side / max(original_width, original_height)
            else:
                raise ValueError(f"Modo de tamaño no soportado: {size_mode}")
            capped = False
            max_upscale = config["size"].get("max_upscale")
            if max_upscale is not None and scale > float(max_upscale):
                scale = float(max_upscale)
                capped = True
            final_width = max(1, round(original_width * scale))
            final_height = max(1, round(original_height * scale))
            if final_width <= canvas[0] and final_height <= canvas[1]:
                break
            rejections_total += 1
        used_images.add(record.image_name)
        position = (
            rng.randint(0, canvas[0] - final_width),
            rng.randint(0, canvas[1] - final_height),
        )
        flip = rng.random() < config["vehicle"]["horizontal_flip_probability"]
        gradient_max = min(
            int(config["vehicle"]["gradient_max_px"]),
            max(1, round(min(final_width, final_height) * config["vehicle"]["gradient_side_fraction"])),
        )
        gradient_min = min(int(config["vehicle"]["gradient_min_px"]), gradient_max)
        gradient_width = rng.randint(gradient_min, gradient_max)
        blur_factor = min(scale, float(config["background"]["blur_factor_cap"]))
        if config["background"]["blur_mode"] == "off":
            blur_factor = 1.0
        vehicle, alpha = _transform_vehicle(
            record, scale, flip, gradient_width, config["vehicle"]["gradient_smoothstep"]
        )
        box = Box(position[0], position[1], vehicle.width, vehicle.height)
        actual_fraction = box.area / (canvas[0] * canvas[1])
        entry = {
            "image_id": f"synthetic_{index:06d}",
            "variant": None,
            "background": {
                "uuid": background.uuid,
                "source": background.source,
                "orig_id": background.orig_id,
                "path": str(background.path),
                "original_size": {"width": background.width, "height": background.height},
                "crop": crop,
                "attributes": background.attributes,
                "license_note": "pendiente de verificar",
            },
            "vehicle": {
                "veri_image": record.image_name,
                "identity": record.vehicle_id,
                "color": record.color,
                "type": record.vehicle_type,
                "original_size": {"width": original_width, "height": original_height},
                "final_size": {"width": box.width, "height": box.height, "area": box.area},
                "area_fraction": actual_fraction,
                "requested_area_fraction": target_fraction,
                "scale": scale,
                "scale_cap_applied": capped,
                "position": {"x": box.x, "y": box.y},
                "horizontal_flip": flip,
                "gradient_width": gradient_width,
            },
            "seed": config["seed"],
            "index": index,
            "rejections": rejections_total,
            "source_splits": splits,
        }
        for variant in VARIANTS:
            variant_entry = json.loads(json.dumps(entry))
            variant_entry["variant"] = variant
            image = background_image.copy()
            if variant == "matched":
                image = _blur_background(image, blur_factor)
            image.paste(vehicle, position, alpha)
            image_path = output_root / variant / "images" / f"{entry['image_id']}.png"
            label_path = output_root / variant / "labels" / f"{entry['image_id']}.txt"
            image.save(image_path, format="PNG")
            label_path.write_text(_yolo_line(box, canvas) + "\n", encoding="utf-8")
            variant_entry["file"] = str(image_path)
            variant_entry["label_file"] = str(label_path)
            variant_entry["background_blur_factor"] = blur_factor if variant == "matched" else 1.0
            manifests[variant].append(variant_entry)
        index += 1
    for variant in VARIANTS:
        with (output_root / variant / "manifest.jsonl").open("w", encoding="utf-8") as handle:
            for item in manifests[variant]:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return manifests["matched"]
