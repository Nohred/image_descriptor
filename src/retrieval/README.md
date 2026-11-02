# Recuperación de vehículos

Este módulo ofrece una API de búsqueda en lenguaje natural sobre los
descriptores de vehículos. El punto de entrada es
`SearchService.search(query, mode, limit)`.

## Modos

### `fixed`

El modelo de chat interpreta la consulta, `query_parser.py` valida el JSON y
`rules.py` aplica correcciones deterministas. Después `FixedSource` consulta
`vehicles.sqlite3` con filtros de:

- color y tipo;
- inclusiones y exclusiones;
- hora del día, incluyendo rangos que cruzan medianoche;
- fechas;
- límite solicitado.

Es el modo que funciona con la base de atributos fijos.

### `semantic`

Embebe la consulta completa con Ollama y busca similitud coseno en
`florence.sqlite3`. No utiliza el modelo de chat. El resultado incluye score y
la descripción asociada.

### `hybrid`

Divide la consulta en filtros estructurados y texto semántico. Primero reduce
los candidatos con SQLite y después ordena por similitud sobre los embeddings.
Si no existe texto semántico residual, devuelve el resultado filtrado. La
configuración `HYBRID_USE_FULL_QUERY` permite decidir si se embebe solo el
residuo o la consulta completa.

En `semantic` y `hybrid`, `SearchResult.total` representa las imágenes
consideradas para el ranking, no necesariamente el número final devuelto.

## Normalización de consultas

`schemas.py` define los catálogos de 10 colores y 9 tipos, aliases en español,
periodos del día y límites (`20` por defecto, `100` máximo). `query_parser.py`
valida formatos de hora/fecha, descarta valores inválidos y registra warnings.
`rules.py` corrige casos frecuentes de modelos pequeños: negaciones, aliases
ambiguos como `camioneta` y periodos como `mañana`, `tarde` o `noche`.

Los términos no soportados (marca, placas, daños, accesorios, etc.) se
conservan como `unsupported_terms` y pueden alimentar `semantic_text`; no se
convierten en filtros SQL inventados.

## Preparar las fuentes

Construya las dos bases siguiendo
[database/README.md](../database/README.md). La base fija debe contener la
tabla `vehicles`; la semántica requiere embeddings generados con la misma
dimensión y modelo configurados en `local_llm`.

## Uso desde Python

```python
from retrieval.fixed_source import FixedSource
from retrieval.service import SearchService
from local_llm.client import chat_json
from database.generated_features.repository import GeneratedSource
from local_llm.embedder import embed_query

fixed = FixedSource()
generated = GeneratedSource(variant="full")
service = SearchService(fixed, chat_json, generated_source=generated,
                        embedder=embed_query)

result = service.search("autos rojos que no sean sedan", mode="hybrid",
                        limit=10)
for candidate in result.candidates:
    print(candidate.image_id, candidate.score)
```

## Pruebas manuales

`run_test.py` compara los modos fixed, semantic e hybrid con consultas
predefinidas y calcula comprobaciones de precisión@K y cumplimiento de filtros:

```bash
python src/retrieval/run_test.py
```

El script espera que las bases y los modelos locales estén disponibles. Para
usar solo filtros fijos, basta con construir `vehicles.sqlite3`.

