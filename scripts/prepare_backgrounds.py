"""Select and prepare COCO validation backgrounds without downloading data."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic.yaml")
    parser.add_argument("--sample-size", type=int, default=12)
    return parser.parse_args()


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"No se encontró {description}: {path}. "
            "Descarga COCO manualmente en las rutas configuradas."
        )


def contact_sheet(paths: list[Path], output: Path, title: str) -> None:
    if not paths:
        return
    thumb_w, thumb_h = 240, 180
    sheet = Image.new("RGB", (thumb_w * 3, (thumb_h + 28) * 4), "white")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths[:12]):
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb_w, thumb_h))
            x = (index % 3) * thumb_w
            y = (index // 3) * (thumb_h + 28)
            sheet.paste(image, (x + (thumb_w - image.width) // 2, y))
            draw.text((x + 4, y + thumb_h + 4), path.name, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    coco = config["backgrounds"]
    image_dir = Path(coco["images"])
    annotation_path = Path(coco["annotations"])
    require_file(annotation_path, "las anotaciones COCO")
    if not image_dir.is_dir():
        raise FileNotFoundError(
            f"No se encontró el directorio de imágenes COCO: {image_dir}. "
            "Descarga val2017 manualmente en data/raw/coco/val2017/."
        )

    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {item["id"]: item["name"] for item in data["categories"]}
    images = {item["id"]: item for item in data["images"]}
    image_categories: dict[int, set[str]] = {image_id: set() for image_id in images}
    for annotation in data["annotations"]:
        image_categories[annotation["image_id"]].add(
            categories[annotation["category_id"]]
        )
    forbidden = set(coco["excluded_classes"])
    urban = set(coco["groups_classes"]["urbano"])
    difficult = set(coco["groups_classes"]["negativo_dificil"])
    minimum_side = int(config["canvas"]["width"])
    output_root = Path(coco["output_root"])
    records: list[dict] = []
    grouped: dict[str, list[Path]] = {name: [] for name in coco["groups"]}
    rng = random.Random(int(config["seed"]))

    for image_id, image in images.items():
        present = image_categories[image_id]
        if present & forbidden:
            continue
        if min(image["width"], image["height"]) < minimum_side:
            continue
        if present & urban:
            group = "urbano"
        elif present & difficult:
            group = "negativo_dificil"
        else:
            group = "generico"
        source = image_dir / image["file_name"]
        require_file(source, f"la imagen COCO {image['file_name']}")
        destination = output_root / group / image["file_name"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        license_id = image.get("license")
        license_record = next(
            (item for item in data.get("licenses", []) if item["id"] == license_id),
            None,
        )
        record = {
            "file": str(destination),
            "source_file": str(source),
            "group": group,
            "width": image["width"],
            "height": image["height"],
            "categories": sorted(present),
            "license": license_record,
        }
        records.append(record)
        grouped[group].append(destination)

    manifest = output_root / "manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    preview_root = Path(coco["preview_root"])
    for group, paths in grouped.items():
        contact_sheet(
            rng.sample(paths, min(args.sample_size, len(paths))),
            preview_root / f"{group}.png",
            group,
        )
        print(f"{group}: {len(paths)}")
    print(f"Fondos seleccionados: {len(records)}")
    print(f"Manifiesto: {manifest}")


if __name__ == "__main__":
    main()
