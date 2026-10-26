# Generación de descriptores con modelos multimodales

Este módulo prueba si un modelo visión-lenguaje puede describir dos atributos
de un vehículo de VeRi:

- **color**: `yellow`, `orange`, `green`, `gray`, `red`, `blue`, `white`,
  `golden`, `brown`, `black`;
- **tipo**: `sedan`, `suv`, `van`, `hatchback`, `mpv`, `pickup`, `bus`, `truck`,
  `estate`.

Se comparan dos modelos:

- **Florence-2-base** (`microsoft/Florence-2-base`);
- **Qwen2-VL-2B-Instruct** (`Qwen/Qwen2-VL-2B-Instruct`) para el benchmark
  común; `run_qwen.py` también conserva una función opcional para
  `Qwen2-VL-7B-Instruct`.

El modelo recibe una imagen y una instrucción/pregunta. La respuesta libre se
normaliza mediante catálogos y sinónimos en `parse_outputs.py`, para convertir
variantes como `silver`/`grey` en `gray` y producir predicciones comparables.

## Archivos

- `run_florence.py`: carga processor/model de Florence-2, ejecuta VQA por
  imagen y guarda respuestas crudas JSON.
- `run_qwen.py`: carga Qwen-VL y ejecuta el prompt de atributos.
- `generate_predictions.py`: interfaz común para procesar un directorio,
  limitar el número de imágenes y guardar
  `data/generative_predictions_{florence,qwen}.csv`.
- `parse_outputs.py`: extrae color y tipo, aplica mapeos y marca resultados
  inválidos.
- `evaluate.py`: calcula reportes, matrices de confusión y métricas a partir de
  los CSV generados.
- `run_all.sh`: lanza el benchmark de ambos modelos.
- `auditor/`: aplicación y material de inspección manual de resultados.

## Entornos

Use `env_florence.yml` para Florence y `env_qwen.yml` para Qwen:

```bash
conda env create -f env_florence.yml
conda env create -f env_qwen.yml
```

Florence usa `transformers==4.47.1`; Qwen usa `transformers==4.51.3` y
`qwen-vl-utils`. No mezcle ambos entornos si aparecen incompatibilidades de
Transformers. Se recomienda GPU CUDA; el código selecciona CPU si CUDA no está
disponible, pero la inferencia será mucho más lenta.

## Uso

Ejecute desde `src/feature_generator`:

```bash
cd src/feature_generator

# Florence-2
conda activate image-descriptors-florence
python generate_predictions.py \
  --model florence \
  --num_images 100

# Qwen-VL
conda activate image-descriptors-qwen
python generate_predictions.py \
  --model qwen \
  --num_images 100
```

El primer uso descarga el modelo de Hugging Face. `--num_images` permite hacer
una prueba pequeña antes de procesar todo el conjunto. La ruta de `VeRi` está
definida en `test_directory` dentro del bloque principal de
`generate_predictions.py`; edítela si el repositorio está en otra ubicación.

Para lanzar ambos benchmarks desde un entorno ya preparado:

```bash
cd src/feature_generator
NUM_IMAGES=100 bash run_all.sh
```

Las predicciones se guardan en `data/`. Para evaluarlas:

```bash
python evaluate.py
```

La evaluación necesita que los CSV contengan las etiquetas reales esperadas por
el script. Genera reportes de clasificación y figuras de matriz de confusión.

## Flujo de datos

```text
imagen VeRi
  -> Florence-2/Qwen-VL
  -> texto libre
  -> parse_outputs.py
  -> CSV color/tipo
  -> evaluate.py
```

Las respuestas de modelos generativos no son garantías de detección: este
módulo clasifica la imagen que recibe. Para describir únicamente el vehículo,
combine la ROI producida por YOLO con este módulo o suministre recortes de
vehículos.
