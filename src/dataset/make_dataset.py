"""Genera un dataset sintético (VeRi sobre fondos de calles) en formato YOLO
y guarda un CSV de metadata por split."""

import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

import cv2
import pandas as pd

from src.dataset.create_synthetic import create_synthetic_image

# ----------------------------------------------------------------------------
# CONFIGURACIÓN
# ----------------------------------------------------------------------------
PROJECT = Path("/home/ulises-calderon/Projects/ImageDescriptorsv2")
FONDOS_DIR = PROJECT / "streetscapes"
VERI_DIR = PROJECT / "VeRi"
OUTPUT_DIR = PROJECT / "synthetic_dataset"

CANTIDAD_TRAIN = 5000          # imágenes de train a muestrear
CANTIDAD_VAL = 1000           # imágenes de validación a muestrear
AREA_RANGE = [0.15, 0.3]       # fracción del fondo que ocupa el vehículo
LOG_EVERY = 500                # cada cuántas imágenes imprimir progreso
SEED = 42                      # reproducibilidad

# Nombre de split en YOLO -> nombre del CSV que se guarda
CSV_NAMES = {
    "train": "train_metadata.csv",
    "val": "val_metadata.csv",
    "test": "test_metadata.csv",
}


# ----------------------------------------------------------------------------
# LECTURA DE ARCHIVOS
# ----------------------------------------------------------------------------
def collect_images(*directories: str | Path) -> list[Path]:
    """Devuelve todos los .jpg/.jpeg (búsqueda recursiva, ordenados)."""
    images: list[Path] = []
    for directory in directories:
        path = Path(directory)
        if not path.is_dir():
            raise NotADirectoryError(f"No es un directorio: {path}")
        images.extend(
            f for f in path.rglob("*")
            if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg"}
        )
    return sorted(images)


def read_labels(label_path: str | Path) -> pd.DataFrame:
    """Lee un XML de etiquetas de VeRi y devuelve solo imageName, colorID y typeID."""
    columns = ["imageName", "colorID", "typeID"]
    xml_bytes = Path(label_path).read_bytes()

    # El XML de VeRi puede venir en UTF-8 o UTF-16
    try:
        xml_text = xml_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        xml_text = xml_bytes.decode("utf-16")

    root = ET.fromstring(xml_text)
    rows = [{c: item.attrib.get(c) for c in columns} for item in root.findall(".//Item")]
    return pd.DataFrame(rows, columns=columns).set_index("imageName")


# ----------------------------------------------------------------------------
# UTILIDADES
# ----------------------------------------------------------------------------
def cyclic_backgrounds(backgrounds: list[Path]) -> Iterator[Path]:
    """Entrega fondos al azar sin repetir; al agotarlos, vuelve a barajar."""
    pool = list(backgrounds)
    while True:
        random.shuffle(pool)
        yield from pool


def make_yolo_dirs(base_dir: Path) -> None:
    """Crea images/{train,val,text} y labels/{train,val,text}."""
    for kind in ("images", "labels"):
        for split in CSV_NAMES:
            (base_dir / kind / split).mkdir(parents=True, exist_ok=True)


def parse_yolo_label(label: str, img_w: int, img_h: int) -> dict:
    """Convierte 'clase cx cy w h' (normalizado) a bbox en píxeles + valores YOLO."""
    cls, cx, cy, w, h = label.strip().splitlines()[0].split()
    cx, cy, w, h = map(float, (cx, cy, w, h))

    return {
        "yolo_class": int(cls),
        "x_center": cx, "y_center": cy, "width": w, "height": h,   # normalizados
        "x_min": round((cx - w / 2) * img_w),                       # píxeles
        "y_min": round((cy - h / 2) * img_h),
        "x_max": round((cx + w / 2) * img_w),
        "y_max": round((cy + h / 2) * img_h),
    }


def save_synthetic_sample(img, label: str, img_path: Path, txt_path: Path) -> None:
    """Guarda la imagen sintética y su .txt de YOLO."""
    cv2.imwrite(str(img_path), img)
    txt_path.write_text(label.strip() + "\n")


# ----------------------------------------------------------------------------
# GENERACIÓN DEL DATASET
# ----------------------------------------------------------------------------
def generate_split(
    vehicles: list[Path],
    labels_df: pd.DataFrame,
    bg_gen: Iterator[Path],
    out_dir: Path,
    split: str,
) -> pd.DataFrame:
    """Genera imágenes + txt YOLO para un split y devuelve su metadata como DataFrame."""
    img_dir = out_dir / "images" / split
    lbl_dir = out_dir / "labels" / split
    records: list[dict] = []

    for i, vehicle_path in enumerate(vehicles, start=1):
        bg_path = next(bg_gen)
        stem = vehicle_path.stem
        synth_name = f"{stem}_synth.jpg"
        img_path = img_dir / synth_name
        txt_path = lbl_dir / f"{stem}_synth.txt"

        # 1) Crear la imagen sintética (imagen BGR + etiqueta YOLO en texto)
        img, yolo_label = create_synthetic_image(
            background_path=bg_path,
            overlay_path=vehicle_path,
            output_path=img_path,
            area_range=AREA_RANGE,
        )

        # 2) Guardar imagen y txt
        save_synthetic_sample(img, yolo_label, img_path, txt_path)

        # 3) Armar la fila de metadata
        img_h, img_w = img.shape[:2]
        meta = labels_df.loc[vehicle_path.name]  # colorID y typeID originales
        records.append({
            "split": split,
            "image_name": synth_name,
            "original_image": vehicle_path.name,
            "background": bg_path.name,
            "colorID": meta["colorID"],
            "typeID": meta["typeID"],
            "img_width": img_w,
            "img_height": img_h,
            **parse_yolo_label(yolo_label, img_w, img_h),
        })

        if i % LOG_EVERY == 0:
            print(f"[{split.upper()}] {i}/{len(vehicles)} imágenes procesadas...")

    return pd.DataFrame(records)


def run_split(split, vehicles, labels_df, bg_gen) -> None:
    """Genera un split completo y guarda su CSV."""
    print(f"\nGenerando split '{split}' ({len(vehicles)} imágenes)...")
    df = generate_split(vehicles, labels_df, bg_gen, OUTPUT_DIR, split)
    csv_path = OUTPUT_DIR / CSV_NAMES[split]
    df.to_csv(csv_path, index=False)
    print(f"CSV guardado: {csv_path}")


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main() -> None:
    random.seed(SEED)

    # Imágenes
    fondos = collect_images(FONDOS_DIR)
    veh_train = collect_images(VERI_DIR / "image_train")
    veh_test = collect_images(VERI_DIR / "image_test")
    print(f"Fondos: {len(fondos)} | Vehículos train: {len(veh_train)} | test: {len(veh_test)}")

    # Etiquetas (indexadas por imageName para buscar rápido)
    train_labels = read_labels(VERI_DIR / "train_label.xml")
    test_labels = read_labels(VERI_DIR / "test_label.xml")

    # Conservar solamente imágenes presentes en sus XML
    veh_train = [p for p in veh_train if p.name in train_labels.index]
    veh_test = [p for p in veh_test if p.name in test_labels.index]


    # Preparar salida y generador de fondos
    make_yolo_dirs(OUTPUT_DIR)
    bg_gen = cyclic_backgrounds(fondos)

    # Reservar imágenes de train para train y validación
    total_train = min(
        len(veh_train),
        CANTIDAD_TRAIN + CANTIDAD_VAL,
    )
    train_pool = random.sample(veh_train, total_train)

    train_sample = train_pool[:CANTIDAD_TRAIN]
    val_sample = train_pool[CANTIDAD_TRAIN:]

    # Test se mantiene independiente
    run_split("test", veh_test, test_labels, bg_gen)

    # Train y validación salen de image_train
    run_split("train", train_sample, train_labels, bg_gen)
    run_split("val", val_sample, train_labels, bg_gen)

    print("\n¡Generación de dataset sintético finalizada!")


if __name__ == "__main__":
    main()