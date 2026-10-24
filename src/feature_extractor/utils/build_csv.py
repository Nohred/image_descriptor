import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
import os

def xml_to_dataframe(xml_path):
    """Lee el XML de VeRi y devuelve un DataFrame con IDs en base 0."""
    xml_bytes = Path(xml_path).read_bytes()
    try:
        xml_text = xml_bytes.decode("utf-8-sig")
    except UnicodeDecodeError:
        xml_text = xml_bytes.decode("utf-16")

    root = ET.fromstring(xml_text)
    
    records = []
    for item in root.findall(".//Item"):
        # Extraer atributos
        img_name = item.attrib.get("imageName")
        color_id = int(item.attrib.get("colorID"))
        type_id = int(item.attrib.get("typeID"))
        
        # Ajuste Crítico para PyTorch: Convertir a Base 0
        records.append({
            "imageName": img_name,
            "colorID": color_id - 1, 
            "typeID": type_id - 1
        })
        
    return pd.DataFrame(records)

if __name__ == "__main__":
    # Rutas base
    veri_dir = Path("../VeRi")
    output_dir = Path("data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Archivos de entrada
    train_xml = veri_dir / "train_label.xml"
    test_xml = veri_dir / "test_label.xml"

    # Procesar
    print("Procesando conjunto de entrenamiento...")
    df_train = xml_to_dataframe(train_xml)
    df_train.to_csv(output_dir / "train_features.csv", index=False)
    print(f"Guardado: {len(df_train)} registros en train_features.csv")

    print("Procesando conjunto de prueba...")
    df_test = xml_to_dataframe(test_xml)
    df_test.to_csv(output_dir / "test_features.csv", index=False)
    print(f"Guardado: {len(df_test)} registros en test_features.csv")