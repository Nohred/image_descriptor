"""Create side-by-side previews for matched and control V2 variants."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw


# Configuración editable.
DATA_ROOT = "data/synthetic"
OUTPUT_DIR = "artifacts/synthetic_preview_v2"
PER_CASE = 3  # Número de ejemplos por caso: s_bajo, s_medio y s_alto.


def _read(path: Path) -> dict[str, dict]:
    return {
        item["image_id"]: item
        for item in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
    }


def main() -> None:
    root = Path(DATA_ROOT)
    matched = _read(root / "matched/manifest.jsonl")
    control = _read(root / "control/manifest.jsonl")
    cases = {"s_bajo": [], "s_medio": [], "s_alto": []}
    for image_id, item in matched.items():
        scale = item["vehicle"]["scale"]
        key = "s_bajo" if scale <= 1 else "s_medio" if scale <= 4 else "s_alto"
        if len(cases[key]) < PER_CASE:
            cases[key].append(image_id)
    output = Path(OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    for case, image_ids in cases.items():
        for image_id in image_ids:
            left_item, right_item = matched[image_id], control[image_id]
            with Image.open(left_item["file"]) as left_source:
                left = left_source.convert("RGB")
            with Image.open(right_item["file"]) as right_source:
                right = right_source.convert("RGB")
            for image, item in ((left, left_item), (right, right_item)):
                draw = ImageDraw.Draw(image)
                vehicle = item["vehicle"]
                x, y = vehicle["position"]["x"], vehicle["position"]["y"]
                width, height = vehicle["final_size"]["width"], vehicle["final_size"]["height"]
                draw.rectangle((x, y, x + width, y + height), outline="red", width=3)
                draw.text((x + 4, y + 4), f"s={vehicle['scale']:.2f}", fill="yellow")
            combined = Image.new("RGB", (left.width * 2, left.height))
            combined.paste(left, (0, 0))
            combined.paste(right, (left.width, 0))
            combined.save(output / f"{case}_{image_id}.png")
    print({case: len(image_ids) for case, image_ids in cases.items()})


if __name__ == "__main__":
    main()
