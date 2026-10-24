from ultralytics import YOLO

# 1. Cargar el modelo base
modelo = YOLO('yolo26s.pt') 

# 2. Iniciar el fine-tuning
print("Iniciando entrenamiento...")
resultados = modelo.train(
    data='synthetic.yaml',  # Tu archivo de configuración
    epochs=50,              # Pasadas completas por el dataset
    imgsz=640,              # Tamaño de imagen (coincide con tu letterbox)
    batch=16,               # Imágenes por lote procesadas al mismo tiempo
    device=0,               # '0' fuerza el uso de la GPU dedicada
    workers=6               # Hilos de CPU para cargar las imágenes más rápido
)

print("¡Entrenamiento finalizado!")