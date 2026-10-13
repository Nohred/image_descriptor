"""Draw labels over representative synthetic images."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/synthetic/manifest.jsonl")
    parser.add_argument("--output", default="artifacts/synthetic_preview")
    parser.add_argument("--per-group", type=int, default=3)
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No se encontró el manifiesto sintético: {manifest_path}")
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        for obj in item["objects"]:
            key = (item["background"]["group"], obj["size_category"])
            if len(groups[key]) < args.per_group:
                groups[key].append(item)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    written: set[str] = set()
    for items in groups.values():
        for item in items:
            if item["image_id"] in written:
                continue
            written.add(item["image_id"])
            with Image.open(item["file"]) as source:
                image = source.convert("RGB")
            draw = ImageDraw.Draw(image)
            for obj in item["objects"]:
                x, y = obj["position"]["x"], obj["position"]["y"]
                width, height = obj["final_size"]["width"], obj["final_size"]["height"]
                draw.rectangle((x, y, x + width, y + height), outline="red", width=3)
                draw.text((x + 3, y + 3), obj["size_category"], fill="yellow")
            image.save(output / f"{item['image_id']}.png")
    print(f"Vistas previas: {len(written)}")


if __name__ == "__main__":
    main()
