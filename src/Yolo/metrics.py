from ultralytics import YOLO

# 1. Cargar el mejor modelo resultante de tu entrenamiento
modelo = YOLO('/home/ulises-calderon/Projects/ImageDescriptorsv2/Yolo/runs/detect/train-2/weights/best.pt')

# 2. Ejecutar la validación sobre el conjunto de test
# El parámetro split='test' obliga a YOLO a usar esa carpeta específica
print("Evaluando el modelo en el dataset de Test...")
metricas = modelo.val(data='synthetic.yaml', split='test')

# 3. Imprimir las métricas principales de forma limpia
print("\n--- RESULTADOS DE DETECCIÓN (SET DE TEST) ---")
print(f"Precisión (Precision):  {metricas.box.mp:.4f}")
print(f"Exhaustividad (Recall): {metricas.box.mr:.4f}")
print(f"mAP@50:                 {metricas.box.map50:.4f}")
print(f"mAP@50-95:              {metricas.box.map:.4f}")

# 4. Extraer e imprimir los tiempos de ejecución promedio por imagen
promedio_pre = metricas.speed['preprocess']
promedio_inf = metricas.speed['inference']
promedio_post = metricas.speed['postprocess']
promedio_total = promedio_pre + promedio_inf + promedio_post

# Calcular los FPS (1000 milisegundos divididos entre el tiempo total que toma 1 imagen)
fps = 1000 / promedio_total if promedio_total > 0 else 0

print("\n--- VELOCIDAD PROMEDIO POR IMAGEN ---")
print(f"Pre-procesamiento:  {promedio_pre:.2f} ms")
print(f"Inferencia pura:    {promedio_inf:.2f} ms")
print(f"Post-procesamiento: {promedio_post:.2f} ms")
print(f"Tiempo total:       {promedio_total:.2f} ms")
print(f"Rendimiento global: {fps:.1f} FPS")