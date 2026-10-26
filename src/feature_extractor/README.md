# Extractor y clasificador multitarea

Este módulo entrena una red supervisada para predecir simultáneamente el color
y el tipo de un vehículo de VeRi. La arquitectura `MultiTaskVehicleNet` usa un
backbone de torchvision y dos cabezas de clasificación:

- 10 clases de color;
- 9 clases de tipo.

Backbones soportados:

- `efficientnet_b0`;
- `mobilenet_v3_small`.

La red comparte la representación visual y calcula una salida independiente
para cada atributo. El entrenamiento usa pesos de clase para compensar el
desequilibrio y guarda el mejor checkpoint en `weights/`.

## Archivos

- `models.py`: define `MultiTaskVehicleNet`, reemplaza las cabezas del
  backbone y devuelve logits de color y tipo.
- `dataset.py`: lee los XML de VeRi, crea transformaciones de torchvision y
  dataloaders para train/validación/test.
- `train.py`: entrena con Adam, pérdida CrossEntropy ponderada, registra
  accuracy y guarda pesos e historial.
- `evaluation.py`: carga un checkpoint, obtiene predicciones, imprime
  `classification_report`, accuracy/F1 y genera matrices de confusión.
- `utils/build_csv.py`: utilidad para construir CSV de etiquetas/características.
- `utils/analyze_data.py`: inspecciona distribución de clases y genera la
  gráfica de imbalance.
- `utils/plotting.py`: guarda curvas del historial de entrenamiento.
- `data/` y `weights/`: resultados generados y checkpoints; muchos están
  excluidos de Git.

## Datos esperados

La configuración actual busca VeRi desde las rutas relativas usadas por
`dataset.py`. La estructura mínima es:

```text
VeRi/
├── image_train/
├── image_test/
├── train_label.xml
└── test_label.xml
```

Las clases se codifican según los catálogos definidos en los scripts:
`COLOR_NAMES` y `TYPE_NAMES`. Verifique que el orden coincida con los IDs del
XML antes de reutilizar pesos o etiquetas externas.

## Entrenamiento

Ejecute desde este directorio para que funcionen los imports locales:

```bash
cd src/feature_extractor
python train.py
```

El bloque principal llama a `entrenar_modelo` con sus valores por defecto:
10 épocas, batch de 32, learning rate `0.001` y backbone
`efficientnet_b0`. Para probar MobileNet o cambiar hiperparámetros, modifique
la llamada al final de `train.py` o invoque la función desde un script:

```python
from train import entrenar_modelo

entrenar_modelo(epochs=20, batch_size=32, lr=1e-3,
                backbone="mobilenet_v3_small")
```

Los checkpoints se escriben en `weights/`, por ejemplo
`best_efficientnet_b0.pth` o `best_mobilenet_v3_small.pth`, y las curvas en
`data/`.

## Evaluación

```bash
cd src/feature_extractor
python evaluation.py
```

`evaluation.py` debe apuntar al checkpoint que se desea evaluar. Produce
reportes separados para color y tipo y matrices de confusión. Si el checkpoint
fue entrenado con otro backbone, use el mismo `backbone_name` al construir el
modelo.

## Preprocesamiento y salidas

Las imágenes pasan por las transformaciones definidas en `dataset.py`; el
loader devuelve la imagen junto con sus etiquetas de color y tipo. La salida
del modelo son dos tensores de logits, no un descriptor vectorial aislado.
Para usarlo como descriptor, tome la representación antes de las cabezas o
adapte `models.py` de forma compatible con los checkpoints existentes.

