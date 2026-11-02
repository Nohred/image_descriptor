import sqlite3
from pathlib import Path
from typing import List, Dict, Optional

# Configuración de ruta
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = PROJECT_ROOT / "src/database/fixed_features/data/vehicles.sqlite3"

def get_connection():
    """Crea la conexión y configura el formato de salida."""
    conn = sqlite3.connect(DB_PATH)
    # Esto permite que las filas se comporten como diccionarios en lugar de tuplas
    conn.row_factory = sqlite3.Row
    return conn

def search_vehicles(
    color_id: Optional[int] = None,
    type_id: Optional[int] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None
) -> List[Dict]:
    """
    Busca vehículos basados en filtros dinámicos.
    Los parámetros None son ignorados en la consulta.
    """
    # 1=1 es un truco para poder concatenar los 'AND' dinámicamente sin romper la sintaxis
    query = "SELECT * FROM vehicles WHERE 1=1"
    params = []

    if color_id is not None:
        query += " AND colorID = ?"
        params.append(color_id)
        
    if type_id is not None:
        query += " AND typeID = ?"
        params.append(type_id)
        
    if start_time is not None:
        query += " AND captured_at >= ?"
        params.append(start_time)
        
    if end_time is not None:
        query += " AND captured_at <= ?"
        params.append(end_time)

    # Bloque 'with' maneja el cierre automático de la base de datos
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        # Convertimos sqlite3.Row a diccionarios estándar de Python
        return [dict(row) for row in rows]

# Bloque de prueba rápida
if __name__ == "__main__":
    # Ejemplo: Si el LLM te dice que el usuario quiere autos color 0
    resultados = search_vehicles(color_id=2, type_id=1, start_time="2026-10-15T00:00:00-06:00", end_time="2026-10-20T23:59:59-06:00")
    print(f"Se encontraron {len(resultados)} resultados.")
    if resultados:
        print("Ejemplo del primer registro:", resultados[0])