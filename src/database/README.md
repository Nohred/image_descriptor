# Bases de datos de descriptores

Este módulo convierte las salidas del pipeline de visión en fuentes SQLite
consultables. Hay dos bases con propósitos diferentes:

1. `fixed_features/data/vehicles.sqlite3`: atributos estructurados de VeRi
   (color, tipo y tiempo de captura).
2. `generated_features/data/florence.sqlite3`: descripciones textuales de
   Florence-2 y sus embeddings para búsquedas semánticas.

Los archivos de datos y las bases generadas están excluidos por `.gitignore`.
Deben reconstruirse después de clonar el repositorio.

## Atributos fijos

### `fixed_features/add_timestamps.py`

Lee `src/feature_extractor/data/test_features.csv` y agrega `captured_at`.
El timestamp se genera de manera determinista a partir de `imageName`, una
semilla y el intervalo configurado (`2026-10-01` a `2026-11-01`). La misma
imagen produce siempre el mismo valor.

```bash
python src/database/fixed_features/add_timestamps.py
```

Salida:

```text
src/database/fixed_features/data/test_features_timestamped.csv
```

### `fixed_features/build_database.py`

Carga el CSV timestamped en la tabla `vehicles` de SQLite y crea índices para
`colorID`, `typeID` y `captured_at`.

```bash
python src/database/fixed_features/build_database.py
```

### `fixed_features/repository.py`

Expone `search_vehicles(color_id, type_id, start_time, end_time)`. Los filtros
son opcionales y se combinan con consultas parametrizadas. La capa de
retrieval traduce los nombres (`red`, `suv`, etc.) a los IDs de VeRi.

## Descripciones y embeddings

### `generated_features/build_embeddings.py`

Convierte `src/feature_generator/data/florence_results.json`, con formato
`{image_id: descripcion}`, en `florence.sqlite3`. Para cada imagen crea dos
variantes:

- `full`: un embedding para la descripción completa;
- `sentences`: un embedding por oración de al menos 15 caracteres.

Los vectores se generan con Ollama usando `nomic-embed-text-v2-moe`, tienen 768
dimensiones, se normalizan y se almacenan como `float32` en SQLite. El proceso
es incremental: compara SHA-256 de los textos, elimina embeddings obsoletos y
solo procesa imágenes nuevas o modificadas. Los bloques se confirman
progresivamente y los textos que fallan se aíslan para reintento posterior.

```bash
python src/database/generated_features/build_embeddings.py
```

Para reconstruir completamente la base, cambie `REBUILD = True` en el script.
Debe hacerse si cambia el modelo o la dimensión del embedding. La configuración
utilizada queda registrada en la tabla `meta` y se valida al iniciar.

### `generated_features/repository.py`

`GeneratedSource` carga una variante en memoria y calcula similitud coseno
mediante producto punto. Para `sentences`, la puntuación de una imagen es la
mejor coincidencia entre sus oraciones. También permite restringir la búsqueda
por un conjunto de `allowed_ids`, que es lo que usa el modo híbrido.

## Orden de construcción

```bash
# 1. Dataset de features supervisados disponible
python src/database/fixed_features/add_timestamps.py
python src/database/fixed_features/build_database.py

# 2. florence_results.json generado previamente
ollama pull nomic-embed-text-v2-moe
python src/database/generated_features/build_embeddings.py
```

La base de embeddings requiere que Ollama esté activo y que el modelo
configurado en `src/local_llm/config.py` esté descargado. Consulte también
[local_llm](../local_llm/README.md) y [retrieval](../retrieval/README.md).

