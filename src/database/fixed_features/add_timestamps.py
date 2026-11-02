import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import json

# Configuración
PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_CSV = PROJECT_ROOT / "src/feature_extractor/data/test_features.csv"
OUTPUT_CSV = PROJECT_ROOT / "src/database/fixed_features/data/test_features_timestamped.csv"

IMAGE_COLUMN = "imageName"  # Actualizado según tu CSV
START = "2026-10-01T00:00:00-06:00" 
END = "2026-11-01T00:00:00-06:00"  
SEED = 42

def calcular_timestamp(image_key, start_dt, total_seconds):
    """Genera un timestamp determinista basado en el nombre de la imagen."""
    payload = json.dumps(["capture-time-v1", str(SEED), str(image_key)], ensure_ascii=False)
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    offset = int.from_bytes(digest, "big") % total_seconds
    return (start_dt + timedelta(seconds=offset)).isoformat(timespec="seconds")

def main():
    # 1. Preparar el rango de fechas
    start_dt = datetime.fromisoformat(START)
    end_dt = datetime.fromisoformat(END)
    total_seconds = int((end_dt - start_dt).total_seconds())

    # 2. Cargar el CSV con pandas
    df = pd.read_csv(INPUT_CSV)

    # 3. Aplicar la función a cada fila para crear la nueva columna
    df["captured_at"] = df[IMAGE_COLUMN].apply(
        lambda x: calcular_timestamp(x, start_dt, total_seconds)
    )

    # 4. Guardar el nuevo CSV
    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # 5. Resumen
    print(f"CSV generado: {output_path}")
    print(f"Filas: {len(df)}")
    print(f"Imágenes únicas: {df[IMAGE_COLUMN].nunique()}")
    print(f"Intervalo: [{START}, {END})")

if __name__ == "__main__":
    main()