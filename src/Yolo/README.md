# Detección de vehículos con YOLO

Este módulo localiza vehículos en las imágenes sintéticas y entrega cajas
dibujadas o regiones de interés para las etapas posteriores. El dataset tiene
una sola clase:

```yaml
0: vehicle
```

## Archivos

- `synthetic.yaml`: configuración de Ultralytics con `train`, `val` y `test`.
- `finetuning.py`: carga `yolo26s.pt` y hace fine-tuning durante 50 épocas,
  con imágenes de `640`, batch `16`, GPU `0` y seis workers.
- `detect.py`: define `detectar_vehiculos`, que devuelve la imagen anotada y
  un JSON con clase, confianza y coordenadas en píxeles. `extract_roi` toma la
  primera detección y devuelve su recorte.
- `metrics.py`: valida en el split `test` e imprime precision, recall,
  mAP@50, mAP@50-95, tiempos y FPS.

Los pesos base `yolo26n.pt` y `yolo26s.pt` son variantes YOLO26; el entrenamiento
actual usa `yolo26s.pt`. El peso final se guarda normalmente bajo
`runs/detect/.../weights/best.pt`.

## Preparar el dataset

Primero genere `synthetic_dataset`:

```bash
python -m src.dataset.make_dataset
```

Después edite `synthetic.yaml`: su campo `path` contiene una ruta absoluta de
otra instalación y debe ser la ruta de `synthetic_dataset` en esta copia.

## Entrenar y evaluar

Ejecute desde este directorio, porque los scripts usan los archivos locales:

```bash
cd src/Yolo
python finetuning.py
python metrics.py
```

`metrics.py` también contiene una ruta absoluta al peso `best.pt`; actualícela
al resultado de la corrida que quiera evaluar. Las métricas se calculan sobre
`split='test'`, separado de train/val.

## Inferencia

Como función Python:

```python
from detect import detectar_vehiculos

imagen, detecciones_json = detectar_vehiculos(
    "ruta/a/imagen.jpg",
    model_path="runs/detect/train/weights/best.pt",
    conf_threshold=0.25,
)
```

Para obtener una ROI:

```python
from ultralytics import YOLO
from detect import extract_roi

modelo = YOLO("runs/detect/train/weights/best.pt")
roi = extract_roi("ruta/a/imagen.jpg", modelo, conf_threshold=0.25)
```

`extract_roi` actualmente usa la primera caja detectada. Si no hay detecciones,
la función no tiene una ruta de manejo para una lista vacía; valide el
resultado antes de usarla en producción. El bloque `__main__` de `detect.py`
contiene rutas de ejemplo que deben reemplazarse.

