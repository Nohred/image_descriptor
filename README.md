# Recuperación de imágenes por características

Módulos:
- src/detection/   : localización con YOLO26
- src/descriptors/ : atributos fijos (fixed) y descripción generada (generative)
- src/database/    : MongoDB
- experiments/locateanything/ : prueba secundaria

Datos (fuera de Git): ver .gitignore.
La versión activa del generador usa Global Streetscapes y produce
`data/synthetic/matched/` y `data/synthetic/control/`.
La prueba histórica con COCO se conserva únicamente en la documentación.
Planeación: ver carpeta de planeación.
