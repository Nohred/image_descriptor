# ImageDescriptors

Pipeline de visión por computador para describir vehículos en imágenes de
vigilancia. El repositorio combina datos reales de **VeRi**, fondos de calles,
generación de imágenes sintéticas, detección de vehículos y clasificación de
atributos visuales.

## Qué hace el proyecto

El flujo completo está dividido en cuatro etapas:

1. **Dataset sintético** (`src/dataset/`): toma recortes de vehículos de VeRi y
   los compone sobre imágenes de `streetscapes`. El resultado queda en formato
   YOLO, con una caja por vehículo y metadata CSV.
2. **Detección** (`src/Yolo/`): ajusta un modelo YOLO sobre el dataset sintético,
   calcula métricas y permite obtener cajas o regiones de interés (ROI).
3. **Generación de descriptores** (`src/feature_generator/`): consulta modelos
   multimodales Florence-2 o Qwen-VL para proponer el color y tipo del vehículo.
4. **Extracción/clasificación supervisada** (`src/feature_extractor/`): entrena
   una red multitarea con backbone EfficientNet-B0 o MobileNetV3-Small para
   clasificar color y tipo a partir de VeRi.

El objetivo es comparar descriptores generados por modelos multimodales con un
clasificador supervisado y usar YOLO como etapa de localización.

## Estructura

```text
.
├── VeRi/                         # Dataset real (imágenes y XML de etiquetas)
├── streetscapes/                 # Fondos para composición sintética
├── synthetic_dataset/            # Salida YOLO (ignorada por Git)
├── src/
│   ├── dataset/                  # Construcción de imágenes y etiquetas sintéticas
│   ├── Yolo/                     # Fine-tuning, inferencia y métricas
│   ├── feature_generator/        # Florence-2, Qwen-VL y normalización de respuestas
│   └── feature_extractor/        # Clasificador multitarea supervisado
├── env_florence.yml
└── env_qwen.yml
```

Los pesos (`.pt`, `.pth`), datasets grandes, resultados CSV/figuras y salidas
de entrenamiento están excluidos por `.gitignore`. Por tanto, una clonación
limpia necesita que los datasets y pesos se obtengan o generen por separado.

## Datos

### VeRi

`VeRi/` debe conservar la estructura original:

- `image_train/`, `image_test/` e `image_query/`;
- `train_label.xml` y `test_label.xml`;
- archivos auxiliares de VeRi (`name_*.txt`, `list_color.txt`,
  `list_type.txt`, etc.).

El proyecto usa los XML para leer `imageName`, `colorID` y `typeID`. VeRi tiene
condiciones de uso propias; consulte `VeRi/ReadMe.txt` antes de redistribuirlo.

### Fondos y dataset sintético

`streetscapes/` contiene los fondos. `src/dataset/make_dataset.py` produce:

```text
synthetic_dataset/
├── images/{train,val,test}/*.jpg
├── labels/{train,val,test}/*.txt
└── metadata_{train,val,test}.csv
```

Los `.txt` siguen el formato YOLO normalizado y usan una única clase:
`0 vehicle`.

## Instalación

Se usan entornos Conda separados porque Florence-2 y Qwen-VL requieren
versiones distintas de `transformers`:

```bash
conda env create -f env_florence.yml
conda env create -f env_qwen.yml
conda activate image-descriptors-florence
```

Para el dataset y YOLO también se necesitan OpenCV, pandas y Ultralytics. El
entorno de Qwen incluye OpenCV, pero puede instalarse Ultralytics en el entorno
que se vaya a usar:

```bash
python -m pip install ultralytics
```

Los entornos están preparados para PyTorch con CUDA 12.4. Si se ejecuta en
CPU, instale la variante de PyTorch compatible con la máquina y espere una
ejecución considerablemente más lenta.

## Flujo recomendado

Ejecutar los comandos desde la raíz del repositorio:

```bash
# 1. Crear imágenes, etiquetas YOLO y metadata
python -m src.dataset.make_dataset

# 2. Ajustar YOLO (desde src/Yolo, porque el script usa synthetic.yaml)
cd src/Yolo
python finetuning.py
python metrics.py
cd ../..

# 3. Generar atributos con un modelo multimodal
conda activate image-descriptors-florence
cd src/feature_generator
python generate_predictions.py --model florence --num_images 100
# o, en el entorno de Qwen:
python generate_predictions.py --model qwen --num_images 100
python evaluate.py
cd ../..

# 4. Entrenar el clasificador multitarea
cd src/feature_extractor
python train.py
python evaluation.py
cd ../..
```

`run_all.sh` ejecuta el benchmark de Florence y Qwen de forma consecutiva,
pero debe lanzarse desde `src/feature_generator` y requiere que ambos modelos
estén disponibles en el entorno activo.

## Notas importantes

- Algunos scripts conservan rutas absolutas de una instalación anterior
  (`/home/ulises-calderon/Projects/ImageDescriptorsv2`). Antes de ejecutar
  `src/dataset/make_dataset.py`, `src/Yolo/metrics.py` o los bloques de prueba
  de inferencia, revise esas constantes y cámbielas por rutas de esta copia.
- `src/Yolo/synthetic.yaml` también contiene una ruta absoluta. Edite `path`
  para que apunte a `<raíz>/synthetic_dataset`.
- Los scripts de `feature_generator` importan módulos locales como
  `parse_outputs`; por eso se recomienda ejecutarlos desde
  `src/feature_generator`.
- `generate_predictions.py` recibe `--model` y `--num_images`, pero no expone
  la ruta del dataset como argumento: revise `test_directory` en el bloque
  principal si su copia no está en la ruta configurada.
- Los pesos ya entrenados, si existen, se encuentran en `src/Yolo/*.pt` y
  `src/feature_extractor/weights/*.pth`, pero Git los ignora.
- El primer uso de Florence-2 o Qwen-VL descarga los modelos desde Hugging
  Face y requiere conexión a Internet y suficiente VRAM/disco.

## Documentación por módulo

- [Dataset sintético](src/dataset/README.md)
- [Detección YOLO](src/Yolo/README.md)
- [Generación con Florence-2 y Qwen-VL](src/feature_generator/README.md)
- [Extractor/clasificador supervisado](src/feature_extractor/README.md)
