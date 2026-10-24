import re
import json

# ==========================================
# 1. CATÁLOGOS ESTRICTOS Y MAPEOS
# ==========================================
VALID_COLORS = ['yellow', 'orange', 'green', 'gray', 'red', 'blue', 'white', 'golden', 'brown', 'black']
VALID_TYPES = ['sedan', 'suv', 'van', 'hatchback', 'mpv', 'pickup', 'bus', 'truck', 'estate']

# Mapeos para sinónimos o palabras frecuentes 
COLOR_MAPPING = {
    'silver': 'gray',
    'grey': 'gray',
    'gold': 'golden',
    'navy': 'blue',
    'maroon': 'red'
}

TYPE_MAPPING = {
    'car': 'sedan',      # Asunción por defecto si solo dicen "car"
    'jeep': 'suv',
    'lorry': 'truck',
    'minivan': 'van'
}

# ==========================================
# 2. FUNCIONES AUXILIARES
# ==========================================
def normalize_attribute(found_words, valid_list, mapping_dict):
    """Convierte sinónimos a clases válidas y asegura que pertenezcan al catálogo estricto."""
    for word in found_words:
        word = word.lower().strip()
        # Aplicar mapeo si existe (ej. silver -> gray)
        mapped_word = mapping_dict.get(word, word)
        
        # Validar si pertenece a la lista estricta
        if mapped_word in valid_list:
            return mapped_word
    return "unknown"

# ==========================================
# 3. PARSER PARA FLORENCE (Texto libre)
# ==========================================
def parse_florence_output(text):
    text_lower = text.lower()
    
    # Extraer colores (buscando válidos + llaves de mapeo)
    search_colors = VALID_COLORS + list(COLOR_MAPPING.keys())
    found_colors = [c for c in search_colors if re.search(rf'\b{c}\b', text_lower)]
    final_color = normalize_attribute(found_colors, VALID_COLORS, COLOR_MAPPING)
    
    # Extraer tipos
    search_types = VALID_TYPES + list(TYPE_MAPPING.keys())
    found_types = [t for t in search_types if re.search(rf'\b{t}\b', text_lower)]
    
    # Prioridad: Si encuentra "pickup" y "truck", queremos "pickup" que es más específico
    if 'pickup' in found_types: 
        found_types.insert(0, 'pickup')
        
    final_type = normalize_attribute(found_types, VALID_TYPES, TYPE_MAPPING)

    # Atributos de escena y accesorios (Regex básicos)
    scene_keywords = ['low angle', 'rear', 'front', 'side', 'night', 'day']
    acc_keywords = ['roof rack', 'spoiler', 'flatbed trailer', 'cargo container', 'alloy wheels']
    
    return {
        "source_model": "florence",
        "features": {
            "color": final_color,
            "vehicle_type": final_type,
        },
        "scene": [kw for kw in scene_keywords if kw in text_lower],
        "accessories": [kw for kw in acc_keywords if kw in text_lower]
    }

# ==========================================
# 4. PARSER PARA QWEN (JSON Strings)
# ==========================================
def parse_qwen_output(json_string):
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError:
        print("Error: El string proporcionado no es un JSON válido.")
        return None

    fixed = data.get("fixed_attributes", {})
    dynamic = data.get("dynamic_attributes", {})
    
    # Extraer valores crudos
    raw_color = fixed.get("color", "")
    raw_type = fixed.get("vehicle_type", "")
    
    # Normalizar (Pasamos como lista de una sola palabra para reusar la función)
    final_color = normalize_attribute([raw_color] if raw_color else [], VALID_COLORS, COLOR_MAPPING)
    final_type = normalize_attribute([raw_type] if raw_type else [], VALID_TYPES, TYPE_MAPPING)

    # Limpiar accesorios de alucinaciones del prompt (ej: "list visible external accessories")
    raw_accessories = dynamic.get("accessories", [])
    clean_accessories = []
    for acc in raw_accessories:
        # Si el modelo escupe la instrucción textual en lugar de un accesorio, la ignoramos
        if "list visible" not in acc.lower():
            # Si el modelo las juntó en un string separadas por coma, las separamos
            parts = [p.strip() for p in acc.split(',')]
            clean_accessories.extend(parts)

    return {
        "source_model": "qwen",
        "features": {
            "color": final_color,
            "vehicle_type": final_type,
        },
        "scene": {
            "time_of_day": dynamic.get("time_of_day", "unknown"),
            "view_angle": dynamic.get("view_angle", "unknown")
        },
        "accessories": clean_accessories
    }


# ==========================================
# 5. EJEMPLO DE USO CON TUS DATOS DE QWEN
# ==========================================
if __name__ == "__main__":

    florence_descriptions = {
    "0698_c014_00049240_0.jpg": "The image is a blurred view of a black car driving on a road. The car appears to be a newer model with a sleek design and a modern look. It has a large front grille with the logo of the car manufacturer, Holden, in the center. The headlights are angular and the car has a sporty appearance. The road is empty and there are no other cars or people visible in the image. The image is taken from a low angle, looking up at the car.",
    "0392_c012_00027770_0.jpg": "The image shows a large yellow dump truck driving on a highway. The truck has a flatbed trailer attached to the back, which appears to be made of metal. The trailer has a textured surface with a pattern of small squares and rectangles. The text on the trailer reads \"BASE078\". The truck is moving at a high speed, as there are other vehicles visible in the background. The image is taken from a low angle, looking up at the truck.",
    "0113_c010_00076745_1.jpg": "The image shows a silver Toyota Land Cruiser Prado pickup truck driving on a highway. The truck has a roof rack on top, which is secured to the roof with a metal bracket. The front of the truck has the Toyota logo and the license plate reads \"Toyota Land Cruiser\". The truck appears to be in motion, as it is moving at a high speed. The background is blurred, but it seems to be a highway with white lines marking the lanes."
}
    # Procesar y simular inserción en MongoDB
    mongodb_documents = []

    for image_id, description in florence_descriptions.items():
        doc = parse_florence_output(description)
        doc["image_id"] = image_id # Llave principal
        mongodb_documents.append(doc)

    # Mostrar resultado estructurado
    print("\n[+] Resultados procesados de Florence:")
    print(json.dumps(mongodb_documents, indent=4, ensure_ascii=False))
        
    # Tu diccionario original con strings JSON de Qwen
    qwen_outputs = {
        "0698_c014_00049240_0.jpg": "{\n  \"fixed_attributes\": {\n    \"color\": \"orange\",\n    \"vehicle_type\": \"suv\"\n  },\n  \"dynamic_attributes\": {\n    \"time_of_day\": \"day\",\n    \"view_angle\": \"front-side\",\n    \"accessories\": [\"list visible external accessories\"]\n  }\n}",
        "0294_c007_00061840_1.jpg": "{\n  \"fixed_attributes\": {\n    \"color\": \"silver\",\n    \"vehicle_type\": \"car\"\n  },\n  \"dynamic_attributes\": {\n    \"time_of_day\": \"day\",\n    \"view_angle\": \"front-side\",\n    \"accessories\": [\"roof rack, spoiler\"]\n  }\n}"
    }

    mongodb_documents = []

    for image_id, qwen_json_string in qwen_outputs.items():
        # Procesar con el parser de Qwen
        doc = parse_qwen_output(qwen_json_string)
        if doc:
            doc["image_id"] = image_id # Llave principal para MongoDB
            mongodb_documents.append(doc)

    # Imprimir el resultado limpio y estructurado
    print("\n[+] Resultados procesados de Qwen:")
    print(json.dumps(mongodb_documents, indent=4, ensure_ascii=False))


