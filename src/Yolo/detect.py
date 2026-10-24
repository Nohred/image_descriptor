from ultralytics import YOLO
import cv2
import json

def detectar_vehiculos(image_path, model_path='runs/detect/train/weights/best.pt', conf_threshold=0.25):
    """
    Realiza la detección sobre una imagen usando tu modelo fine-tuneado.
    Devuelve la imagen anotada (array de numpy) y un JSON con las detecciones.
    """
    # 1. Cargar tu modelo fine-tuneado
    modelo = YOLO(model_path)
    
    # 2. Realizar la predicción
    # conf=0.25 filtra las detecciones con baja seguridad. Puedes subirlo (ej. 0.50) si detecta ruido.
    resultados = modelo(image_path, conf=conf_threshold)
    
    # El modelo puede procesar batches, pero como le pasamos 1 imagen, tomamos el índice 0
    resultado_img = resultados[0]
    
    # 3. Obtener la imagen ya dibujada por YOLO (con cajas y etiquetas)
    imagen_anotada = resultado_img.plot()
    
    # 4. Extraer la data estructurada
    detecciones = []
    
    # Iterar sobre cada objeto detectado en la imagen
    for box in resultado_img.boxes:
        # Extraer coordenadas absolutas (píxeles): x_min, y_min, x_max, y_max
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        
        # Extraer confianza y clase
        confianza = box.conf[0].item()
        clase_id = int(box.cls[0].item())
        nombre_clase = modelo.names[clase_id] # YOLO guarda los nombres en el modelo
        
        detecciones.append({
            "clase_id": clase_id,
            "nombre": nombre_clase,
            "confianza": round(confianza, 4),
            "bbox_pixeles": {
                "x_min": int(x1),
                "y_min": int(y1),
                "x_max": int(x2),
                "y_max": int(y2)
            }
        })
        
    # Convertir el diccionario a un string JSON formateado
    json_detecciones = json.dumps(detecciones, indent=4)
    
    return imagen_anotada, json_detecciones

def extract_roi(image, modelo, conf_threshold=0.25):
    resultados = modelo(image, conf=conf_threshold)

    resultado_img = resultados[0]
    bbox = resultado_img.boxes.xyxy[0].tolist()  # Tomamos la primera detección
    # Get roi image
    x1, y1, x2, y2 = map(int, bbox)
    roi = resultado_img.orig_img[y1:y2, x1:x2]
    #print roi shape
    print(f"ROI shape: {roi.shape}")
    return roi



# --- CÓMO USARLO ---
if __name__ == "__main__":
    ruta_prueba = '/home/ulises-calderon/Projects/ImageDescriptorsv2/synthetic_dataset/images/test/0002_c002_00030615_1_synth.jpg'
    model_path = '/home/ulises-calderon/Projects/ImageDescriptorsv2/Yolo/runs/detect/train-3/weights/best.pt'
    
    # img_dibujada, json_datos = detectar_vehiculos(ruta_prueba, model_path=model_path, conf_threshold=0.1)
    modelo = YOLO(model_path)
    resultado = extract_roi(ruta_prueba, modelo, conf_threshold=0.1)
    
    # Guardar la imagen para verla
    cv2.imwrite('resultado_inferencia.jpg', resultado)
    
    # Imprimir el JSON en la terminal
    # print("Detecciones encontradas:")
    # print(json_datos)