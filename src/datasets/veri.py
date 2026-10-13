"""Lectura de anotaciones y particiones del dataset VeRi."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree


@dataclass(frozen=True)
class VeriRecord:
    image_name: str
    split: str
    image_path: Path
    vehicle_id: str
    camera_id: str
    color: str
    vehicle_type: str


def _read_mapping(path: Path) -> dict[int, str]:
    return {
        int(parts[0]): parts[1]
        for line in path.read_text(encoding="utf-8").splitlines()
        if (parts := line.split()) and len(parts) >= 2
    }


def _read_xml(path: Path) -> dict[str, dict[str, str]]:
    root = ElementTree.fromstring(path.read_bytes().decode("gb2312"))
    return {item.attrib["imageName"]: item.attrib for item in root.findall(".//Item")}


def load_veri_records(root: str | Path = "VeRi") -> list[VeriRecord]:
    """Load all records listed by VeRi's official train/test/query files."""
    root = Path(root)
    colors = _read_mapping(root / "list_color.txt")
    types = _read_mapping(root / "list_type.txt")
    annotations: dict[str, dict[str, str]] = {}
    for annotation_file in ("train_label.xml", "test_label.xml"):
        annotations.update(_read_xml(root / annotation_file))

    records: list[VeriRecord] = []
    for split in ("train", "test", "query"):
        names = (root / f"name_{split}.txt").read_text(encoding="utf-8").split()
        image_dir = root / f"image_{split}"
        for name in names:
            try:
                attrs = annotations[name]
            except KeyError as exc:
                raise ValueError(f"Missing VeRi annotation for {name}") from exc
            records.append(
                VeriRecord(
                    image_name=name,
                    split=split,
                    image_path=image_dir / name,
                    vehicle_id=attrs["vehicleID"],
                    camera_id=attrs["cameraID"],
                    color=colors[int(attrs["colorID"])],
                    vehicle_type=types[int(attrs["typeID"])],
                )
            )
    return records


def test_identity_count(root: str | Path = "VeRi") -> int:
    """Return the number of distinct identities in the official test list."""
    return len({
        record.vehicle_id
        for record in load_veri_records(root)
        if record.split == "test"
    })
