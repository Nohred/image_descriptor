"""Compute VeRi, Streetscapes and synthetic dataset statistics."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.veri import load_veri_records


# Configuración editable.
CONFIG_PATH = "configs/synthetic.yaml"
OUTPUT_DIR = "artifacts/dataset_stats"


def _percentiles(values: list[float]) -> dict[str, float]:
    names = ("min", "p1", "p5", "p25", "median", "p75", "p95", "p99", "max")
    return dict(zip(names, np.percentile(values, [0, 1, 5, 25, 50, 75, 95, 99, 100]).tolist()))


def _write_tex(path: Path, title: str, rows: dict[str, object]) -> None:
    lines = [f"\\begin{{table}}[ht]\\centering\\caption{{{title}}}", "\\begin{tabular}{ll}", "\\toprule"]
    for key, value in rows.items():
        escaped = str(value).replace("_", "\\_")
        lines.append(f"{key} & {escaped} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    config = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    output = Path(OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)
    records = load_veri_records(config["veri_root"])
    veri = {}
    for split in ("train", "test", "query"):
        subset = [record for record in records if record.split == split]
        dimensions = []
        for record in subset:
            from PIL import Image
            with Image.open(record.image_path) as image:
                width, height = image.size
            dimensions.append((width, height))
        veri[split] = {
            "images": len(subset),
            "identities": len({record.vehicle_id for record in subset}),
            "sizes": {
                name: _percentiles([value[index] for value in dimensions])
                for name, index in (("width", 0), ("height", 1))
            },
        }
        veri[split]["sizes"]["area"] = _percentiles([width * height for width, height in dimensions])
        veri[split]["sizes"]["aspect_ratio"] = _percentiles([width / height for width, height in dimensions])
    test = [record for record in records if record.split == "test"]
    veri["test"]["color"] = dict(Counter(record.color for record in test))
    veri["test"]["type"] = dict(Counter(record.vehicle_type for record in test))
    streets = []
    manifest_path = Path(config["backgrounds"]["manifest"])
    if manifest_path.is_file():
        streets = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]
    short = [min(item["width"], item["height"]) for item in streets]
    streetscapes = {
        "count": len(streets),
        "source": dict(Counter(item["source"] for item in streets)),
        "short_side": _percentiles(short) if short else {},
        "attributes": {
            key: dict(Counter(item.get("attributes", {}).get(key) for item in streets))
            for key in ("platform", "weather", "view_direction", "lighting_condition",
                        "glare", "quality", "reflection", "pano_status", "projection_type")
        },
    }
    synthetic = {}
    for variant in ("matched", "control"):
        path = Path(config["output"]["root"]) / variant / "manifest.jsonl"
        items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.is_file() else []
        scales = [item["vehicle"]["scale"] for item in items]
        fractions = [item["vehicle"]["area_fraction"] for item in items]
        synthetic[variant] = {
            "count": len(items),
            "scale": _percentiles(scales) if scales else {},
            "area_fraction": _percentiles(fractions) if fractions else {},
            "scale_ranges": dict(Counter(
                "s<=1" if value <= 1 else "1<s<=2" if value <= 2 else "2<s<=3" if value <= 3 else "3<s<=4" if value <= 4 else "s>4"
                for value in scales
            )),
            "blur_capped": sum(item["background_blur_factor"] >= config["background"]["blur_factor_cap"] for item in items),
        }
    result = {"veri": veri, "streetscapes": streetscapes, "synthetic": synthetic}
    (output / "dataset_stats.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_tex(output / "veri.tex", "Estadísticas de VeRi", {"particiones": veri})
    _write_tex(output / "streetscapes.tex", "Estadísticas de Streetscapes", streetscapes)
    _write_tex(output / "synthetic.tex", "Estadísticas sintéticas", synthetic)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
