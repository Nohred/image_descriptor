# Aplicación web local

La aplicación de [`web_app.py`](../../web_app.py) expone una interfaz mínima
para buscar imágenes de VeRi con lenguaje natural. Usa un servidor HTTP de la
biblioteca estándar y sirve [`web/index.html`](../../web/index.html).

## Inicio

Desde la raíz del proyecto:

```bash
python web_app.py
```

Abra <http://127.0.0.1:8765/>.

La aplicación utiliza `VeRi/image_test` como carpeta de imágenes, limita cada
consulta a 100 resultados y carga automáticamente las fuentes disponibles. Si
una base o una dependencia no está disponible, el servidor puede iniciar y
reporta el modo afectado en `/api/status`.

## Modos de búsqueda

La interfaz permite seleccionar:

- `fixed`: filtros estructurados sobre la base SQLite;
- `semantic`: ranking por embeddings;
- `hybrid`: filtros más ranking semántico.

El campo de consulta acepta frases como `autos rojos que no sean hatchback`.
Cada tarjeta muestra la imagen, el origen (`fixed` o `generated`), score,
atributos y descripción cuando existe.

## API HTTP

### Estado

```bash
curl http://127.0.0.1:8765/api/status
```

### Búsqueda

```bash
curl -X POST http://127.0.0.1:8765/api/search \
  -H 'Content-Type: application/json' \
  -d '{"query":"autos rojos","mode":"fixed","limit":10}'
```

`limit` debe ser un entero entre 1 y 100. Las respuestas contienen `mode`,
`query`, `parsed`, `candidates`, `total` y `elapsed_ms`.

### Imágenes

Las imágenes se sirven mediante `/images/<image_id>`. El servidor resuelve la
ruta dentro de `VeRi/image_test` y rechaza rutas fuera de esa carpeta.

## Dependencias

Para `fixed` se necesita la base de atributos y el cliente de chat local.
`semantic` y `hybrid` requieren además `florence.sqlite3`, Ollama y
`nomic-embed-text-v2-moe`. Consulte [local_llm](../local_llm/README.md),
[database](../database/README.md) y [retrieval](../retrieval/README.md).

