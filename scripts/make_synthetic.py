"""Genera las variantes sintéticas pareadas de Streetscapes."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.synthetic import generate


# Configuración editable.
CONFIG_PATH = "configs/synthetic.yaml"


def main() -> None:
    config = yaml.safe_load(Path(CONFIG_PATH).read_text(encoding="utf-8"))
    manifest = generate(config)
    print(f"Imágenes matched generadas: {len(manifest)}")


if __name__ == "__main__":
    main()
