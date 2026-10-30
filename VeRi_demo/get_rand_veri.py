"""Save a random subset of VeRi test images and their labels."""

import csv
import random
import shutil
import re
import xml.etree.ElementTree as ET
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def find_test_images(dataset_dir: Path):
    """Find images in the usual VeRi image_test directory."""
    candidates = [dataset_dir / "image_test", dataset_dir / "test"]
    image_dir = next((path for path in candidates if path.is_dir()), None)
    if image_dir is None:
        raise FileNotFoundError(
            f"No se encontró image_test en {dataset_dir}. "
            "Verifica la ruta de la variable dataset_dir."
        )
    return sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

def read_labels(dataset_dir: Path):
    """Return labels indexed by image filename from VeRi's test_label.xml."""
    xml_path = dataset_dir / "test_label.xml"
    if not xml_path.is_file():
        raise FileNotFoundError(f"No se encontró el archivo de labels: {xml_path}")

    # Solución al ValueError: leer como texto con gbk/gb2312 y quitar la declaración XML
    with open(xml_path, 'r', encoding='gbk', errors='ignore') as f:
        xml_string = f.read()
    
    xml_string = re.sub(r'<\?xml[^>]+\?>', '', xml_string)
    root = ET.fromstring(xml_string)

    labels = {}
    for element in root.iter():
        attrs = dict(element.attrib)
        image_name = (
            attrs.get("imageName") or attrs.get("image_name") or
            attrs.get("name") or attrs.get("filename")
        )
        if image_name:
            labels[Path(image_name).name] = attrs
    return labels

def main():
    # Variables de configuración
    dataset_dir = Path("/home/ulises-calderon/Projects/ImageDescriptors/VeRi")
    n = 100
    output_dir = Path(__file__).resolve().parent / "random_test_veri"
    seed = None

    if n < 1:
        raise ValueError("La variable 'n' debe ser mayor que cero")

    images = find_test_images(dataset_dir)
    labels = read_labels(dataset_dir)
    available = [image for image in images if image.name in labels]
    
    if n > len(available):
        raise ValueError(
            f"Solo hay {len(available)} imágenes de test con label; se pidieron {n}."
        )

    selected = random.Random(seed).sample(available, n)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for image in selected:
        shutil.copy2(image, output_dir / image.name)
        row = {"image": image.name, **labels[image.name]}
        rows.append(row)

    fields = ["image"]
    fields.extend(sorted({key for row in rows for key in row if key != "image"}))
    with (output_dir / "labels.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Guardadas {len(selected)} imágenes y sus labels en {output_dir}")

if __name__ == "__main__":
    main()