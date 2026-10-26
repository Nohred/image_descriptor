# Dataset sintético

Este módulo construye un dataset de detección de vehículos en formato YOLO a
partir de dos fuentes:

- recortes de vehículos de `VeRi/image_train` y `VeRi/image_test`;
- fondos de calles en `streetscapes/`.

El vehículo se escala a un área aleatoria del fondo, se coloca en una posición
aleatoria, se suavizan los bordes mediante alpha blending, se degrada el fondo
según la escala del objeto y finalmente se aplica letterbox a `640x640`.
También se transforma la caja resultante a
`class x_center y_center width height`, con valores normalizados.

## Archivos

### `create_synthetic.py`

Implementa las operaciones de una muestra individual:

- `insertar_con_gradiente`: mezcla el recorte con el fondo para evitar bordes
  artificiales;
- `degradar_fondo_dinamicamente`: ajusta el fondo alrededor del vehículo;
- `aplicar_letterbox`: conserva la proporción y rellena hasta `640x640`;
- `create_synthetic_image`: coordina lectura, resize, composición y cálculo de
  la caja YOLO.

La función principal devuelve `(imagen, etiqueta_yolo)`. El bloque de ejemplo
al final del archivo usa rutas absolutas y debe adaptarse antes de ejecutarlo
directamente.

### `make_dataset.py`

Es el generador por lotes. Lee los XML de VeRi, descarta imágenes que no tengan
entrada en el XML, y crea los splits:

- `test`: todas las imágenes de `VeRi/image_test` con etiqueta válida;
- `train`: una muestra de `image_train`;
- `val`: el resto de la reserva de entrenamiento.

La configuración actual reserva hasta `CANTIDAD_TRAIN` y `CANTIDAD_VAL`
imágenes, usa una semilla fija y cicla los fondos para poder reutilizarlos.
Cada split guarda un CSV de metadata con el nombre de la imagen sintética,
imagen original, fondo, color/tipo de VeRi, dimensiones y caja.

## Uso

Desde la raíz:

```bash
python -m src.dataset.make_dataset
```

El script espera:

```text
VeRi/image_train/
VeRi/image_test/
VeRi/train_label.xml
VeRi/test_label.xml
streetscapes/
```

La salida es:

```text
synthetic_dataset/
├── images/train|val|test/*.jpg
├── labels/train|val|test/*.txt
└── metadata_train.csv
   metadata_val.csv
   metadata_test.csv
```

Cada etiqueta contiene una sola clase (`0`, vehículo). El archivo
`src/Yolo/synthetic.yaml` apunta a estos splits.

## Ajustes frecuentes

La ruta del proyecto, las carpetas de entrada/salida, el tamaño de los splits,
la semilla, `AREA_RANGE` y la frecuencia de logging están definidos como
constantes al inicio de `make_dataset.py`. Revise esas constantes antes de
generar un dataset grande. La generación puede sobrescribir archivos
existentes en `synthetic_dataset/`.

