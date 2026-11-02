import pandas as pd
import sqlite3
from pathlib import Path

# Configuración de rutas
PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_CSV = PROJECT_ROOT / "src/database/fixed_features/data/test_features_timestamped.csv"
OUTPUT_DB = PROJECT_ROOT / "src/database/fixed_features/data/vehicles.sqlite3"

def main():
    # 1. Cargar el CSV con los timestamps
    df = pd.read_csv(INPUT_CSV)
    
    # 2. Asegurar que el directorio exista y conectar a SQLite
    OUTPUT_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(OUTPUT_DB)
    
    # 3. Volcar los datos a SQLite (crea la tabla 'vehicles' automáticamente)
    df.to_sql("vehicles", conn, if_exists="replace", index=False)
    
    # 4. Crear índices para optimizar las consultas generadas por el LLM
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_color ON vehicles(colorID);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_type ON vehicles(typeID);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_time ON vehicles(captured_at);")
    
    conn.commit()
    conn.close()

    print(f"Base de datos SQLite generada: {OUTPUT_DB}")
    print(f"Registros insertados en la tabla 'vehicles': {len(df)}")

if __name__ == "__main__":
    main()