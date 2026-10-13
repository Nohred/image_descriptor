"""CLI for generating the synthetic YOLO dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.synthetic import generate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/synthetic.yaml")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    manifest = generate(config)
    print(f"Imágenes generadas: {len(manifest)}")


if __name__ == "__main__":
    main()
