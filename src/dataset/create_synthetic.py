import cv2
import numpy as np
import random


def insertar_con_gradiente(img_fondo, img_pequena, x, y, grosor_borde=8):
    """
    Inserta img_pequena en img_fondo usando un gradiente en los bordes (Alpha Blending)
    para evitar líneas duras, manteniendo intacto el color interior.
    """
    h_peq, w_peq = img_pequena.shape[:2]
    
    # 1. Crear una máscara de 1 canal (escala de grises), todo en negro (0)
    mascara = np.zeros((h_peq, w_peq), dtype=np.float32)
    
    # 2. Dibujar un rectángulo blanco (1.0) en el centro, dejando un margen negro
    # Ajustamos el grosor por si la imagen es muy pequeña
    borde = min(grosor_borde, h_peq // 4, w_peq // 4)
    mascara[borde:h_peq-borde, borde:w_peq-borde] = 1.0
    
    # 3. Desenfocar la máscara agresivamente para crear el gradiente suave
    # El tamaño del kernel debe ser impar y proporcional al borde
    kernel_size = (borde * 2) + 1
    mascara = cv2.GaussianBlur(mascara, (kernel_size, kernel_size), 0)
    
    # Convertir la máscara a 3 canales para multiplicar con BGR (forma: H, W, 3)
    mascara_3c = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
    
    # 4. Extraer el ROI (Región de Interés) del fondo donde irá la imagen
    roi_fondo = img_fondo[y:y+h_peq, x:x+w_peq].astype(np.float32)
    img_pequena_float = img_pequena.astype(np.float32)
    
    # 5. Aplicar la fórmula de Alpha Blending
    # Resultado = (Carro * Máscara) + (Fondo * (1 - Máscara))
    roi_fusionado = (img_pequena_float * mascara_3c) + (roi_fondo * (1.0 - mascara_3c))
    
    # 6. Volver a formato uint8 e insertar en la imagen original
    img_fondo[y:y+h_peq, x:x+w_peq] = roi_fusionado.astype(np.uint8)
    
    return img_fondo

def degradar_fondo_dinamicamente(img_fondo, scale_factor, umbral_minimo=1.2, agresividad=5):
    """
    Aplica un desenfoque a TODA la imagen de fondo de forma proporcional 
    a cuánto tuvimos que estirar la imagen pequeña.
    """
    # Si la imagen se redujo, mantuvo su tamaño, o creció muy poco (por debajo del umbral), no hacemos nada
    if scale_factor <= umbral_minimo:
        return img_fondo

    # 1. Calcular el tamaño base del kernel. 
    # Le restamos 1 al scale_factor para que el cálculo empiece a partir del crecimiento real.
    # Ej con agresividad=5: 
    # scale_factor 1.5 -> (0.5 * 5) = 2.5
    # scale_factor 3.0 -> (2.0 * 5) = 10.0
    kernel_crudo = int((scale_factor - 1.0) * agresividad)

    # 2. Asegurarnos de que sea un número IMPAR (requisito estricto de OpenCV)
    if kernel_crudo % 2 == 0:
        kernel_size = kernel_crudo + 1
    else:
        kernel_size = kernel_crudo

    # (Opcional) Poner un límite máximo para no destruir el fondo por completo si el scale_factor es gigante
    kernel_size = min(kernel_size, 15)
    
    # Nos aseguramos de que el kernel sea al menos de 3x3 si pasó el umbral
    kernel_size = max(kernel_size, 3)

    # print(f"[Global Blur] Factor de estiramiento: {scale_factor:.2f}. Aplicando desenfoque Gaussiano de {kernel_size}x{kernel_size}")

    # 3. Aplicar el desenfoque a todo el fondo
    fondo_degradado = cv2.GaussianBlur(img_fondo, (kernel_size, kernel_size), 0)

    return fondo_degradado

def aplicar_letterbox(img, tamano_objetivo=(640, 640), color_relleno=(114, 114, 114)):
    """
    Redimensiona una imagen manteniendo su relación de aspecto y agrega
    padding (relleno) gris para alcanzar la resolución objetivo exacta.
    """
    h, w = img.shape[:2]
    w_obj, h_obj = tamano_objetivo

    # 1. Calcular el factor de escala manteniendo las proporciones
    escala = min(w_obj / w, h_obj / h)
    
    # Nuevas dimensiones (redimensionado)
    nuevo_w = int(w * escala)
    nuevo_h = int(h * escala)
    
    # 2. Redimensionar la imagen original
    # Es una buena práctica usar INTER_AREA si la imagen se hace más pequeña, 
    # y LINEAR o CUBIC si se hace más grande.
    interpolacion = cv2.INTER_AREA if escala < 1 else cv2.INTER_LINEAR
    img_redimensionada = cv2.resize(img, (nuevo_w, nuevo_h), interpolation=interpolacion)
    
    # 3. Calcular cuánto relleno (padding) falta para llegar a 640x640
    pad_w = w_obj - nuevo_w
    pad_h = h_obj - nuevo_h
    
    # Dividir el relleno a la mitad para centrar la imagen (usamos // para enteros)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    # 4. Aplicar el borde con el color gris neutro
    img_con_borde = cv2.copyMakeBorder(
        img_redimensionada, 
        top, bottom, left, right, 
        cv2.BORDER_CONSTANT, 
        value=color_relleno
    )
    
    return img_con_borde

def create_synthetic_image(background_path, overlay_path, output_path, area_range=[0.15, 0.3]):
    """
    Crea una imagen sintética insertando una imagen más pequeña sobre una imagen de fondo.
    
    :param background_path: Ruta de la imagen de fondo.
    :param overlay_path: Ruta de la imagen que se va a superponer.
    :param output_path: Ruta donde se guardará la imagen resultante.
    :param area_range: Rango de área relativa que la imagen pequeña puede ocupar en la grande (porcentaje).
    """

    # Cargar las imágenes
    img_grande = cv2.imread(background_path)
    img_pequena = cv2.imread(overlay_path)

    # Redimensionar la imagen de fondo para tener un tamaño máximo de 640 píxeles en su dimensión más grande
    h_grande, w_grande = img_grande.shape[:2]

    if w_grande > h_grande:
        img_grande = cv2.resize(img_grande, (640, int(640 * h_grande / w_grande))) # Resize to width 640 
    else:
        img_grande = cv2.resize(img_grande, (int(640 * w_grande / h_grande), 640)) # Resize to height 640

    # Obtener dimensiones
    h_pequena, w_pequena = img_pequena.shape[:2]

    # Dimensiones originales
    # print(f"Dimensiones imagen pequeña original: {w_pequena}x{h_pequena}")
    # print(f"Dimensiones imagen grande original: {w_grande}x{h_grande}")

    h_grande, w_grande = img_grande.shape[:2]

    # Calcular el área máxima permitida para la imagen pequeña
    max_area = (w_grande * h_grande) * area_range[1]
    min_area = (w_grande * h_grande) * area_range[0]
    obj_area = random.randint(int(min_area), int(max_area))
    ## Porcentaje de area objetivo respecto a la imagen grande
    
    # print(f"Area objetivo: {obj_area} ({obj_area / (w_grande * h_grande) * 100:.2f}%)")

    scale_factor = (obj_area / (w_pequena * h_pequena)) ** 0.5

    # print(f'Factor de escala pequena= {scale_factor}')

    # Limitar la escala para que el vehículo quepa completamente en el fondo
    max_scale = min(
        w_grande / w_pequena,
        h_grande / h_pequena,
    )
    scale_factor = min(scale_factor, max_scale)

    new_w = max(1, int(w_pequena * scale_factor))
    new_h = max(1, int(h_pequena * scale_factor))
    
    img_pequena = cv2.resize(img_pequena, (new_w, new_h))
    h_pequena, w_pequena = img_pequena.shape[:2]
    

    # Calcular los límites máximos para x e y
    max_y = h_grande - h_pequena
    max_x = w_grande - w_pequena

    # Generar coordenadas aleatorias dentro de los límites
    y_aleatorio = random.randint(0, max_y)
    x_aleatorio = random.randint(0, max_x)


    img_grande = degradar_fondo_dinamicamente(img_grande, scale_factor, umbral_minimo=1.2, agresividad=5)

    img_grande = insertar_con_gradiente(img_grande, img_pequena, x_aleatorio, y_aleatorio, grosor_borde=8)

    img_grande = aplicar_letterbox(img_grande, tamano_objetivo=(640,640))

    # --- CALCULAR BOUNDING BOX FORMATO YOLO ---
    # 1. Calcular el padding que letterbox agregó
    pad_x = (640 - w_grande) // 2
    pad_y = (640 - h_grande) // 2

    # 2. Desplazar las coordenadas originales sumando el padding
    x_final = x_aleatorio + pad_x
    y_final = y_aleatorio + pad_y

    # 3. Calcular el centro de la bounding box
    x_centro = x_final + (w_pequena / 2.0)
    y_centro = y_final + (h_pequena / 2.0)

    # 4. Normalizar dividiendo entre el tamaño final (640)
    x_centro_norm = x_centro / 640.0
    y_centro_norm = y_centro / 640.0
    w_norm = w_pequena / 640.0
    h_norm = h_pequena / 640.0

    # Crear el string en formato YOLO (asumiendo que la clase del vehículo es 0)
    clase_id = 0
    yolo_annotation = f"{clase_id} {x_centro_norm:.6f} {y_centro_norm:.6f} {w_norm:.6f} {h_norm:.6f}"

    # print(f"YOLO Annotation: {yolo_annotation}")

    # Guardar la imagen resultante
    # cv2.imwrite(output_path, img_grande)
    
    # Devolver los datos por si quieres procesarlos o guardarlos en un .txt fuera de la función
    return img_grande, yolo_annotation


if __name__ == "__main__":

    # 1. Cargar las imágenes
    path_grande = '/home/ulises-calderon/Projects/ImageDescriptorsv2/streetscapes/0a0f8f0b-5fe4-459f-996b-c4508897745c.jpeg'
    path_pequena = "/home/ulises-calderon/Projects/ImageDescriptorsv2/VeRi/image_train/0001_c012_00014680_0.jpg"


    create_synthetic_image(
        background_path=path_grande,
        overlay_path=path_pequena,
        output_path='/home/ulises-calderon/Projects/ImageDescriptorsv2/synthetic_image.jpg'
    )

