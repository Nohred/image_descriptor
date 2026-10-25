import torch
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig
from PIL import Image

# 1. CARGA DEL MODELO 
model_id = "nvidia/LocateAnything-3B"
print(f"[*] Cargando {model_id} en 4-bits...")

# Usamos la nueva configuración oficial para evitar los warnings de deprecación
quant_config = BitsAndBytesConfig(load_in_4bit=True)

model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map="cuda",
    torch_dtype=torch.float16,
    quantization_config=quant_config 
).eval()

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

# 2. FUNCIÓN DE LA PRUEBA (El Test de la Ausencia)
def ejecutar_prueba(image_path, test_name, prompt_phrase):
    print(f"\n{'='*50}\n[>] {test_name}\n[?] Buscando: '{prompt_phrase}'")
    
    try:
        image = Image.open(image_path).convert("RGB")
        
        prompt = f"Locate the region that matches the following description: {prompt_phrase}."
        
        # CORRECCIÓN CLAVE: Envolvemos 'image' y 'prompt' en listas
        inputs = processor(images=[image], text=[prompt], return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=64)
            
        # Limpiar el output para leer la respuesta cruda
        result_text = processor.decode(outputs[0], skip_special_tokens=True)
        
        # Quitamos el prompt original del resultado impreso
        respuesta = result_text.replace(prompt, "").strip()
        print(f"[*] Respuesta cruda del modelo:\n{respuesta}")
        
        if "[" in respuesta or "<loc_" in respuesta:
            print("🚨 CONCLUSIÓN: ALUCINACIÓN (Falso Positivo). El modelo forzó la detección ciegamente.")
        else:
            print("✅ CONCLUSIÓN: PASÓ. No dibujó cajas para algo inexistente.")
            
    except Exception as e:
        print(f"[-] Error procesando la imagen {image_path}: {e}")

# ==========================================
# 3. EJECUCIÓN DE TUS GRUPOS EXPERIMENTALES
# ==========================================
if __name__ == "__main__":
    
    # 1. El Test del Objeto Fantasma
    ejecutar_prueba(
        image_path="sedan_liso.jpg", 
        test_name="TEST 1: Objeto Fantasma (Contexto Forzado)", 
        prompt_phrase="roof rack"
    )
    
    # 2. El Test de Contradicción de Atributos
    ejecutar_prueba(
        image_path="auto_rojo.jpg", 
        test_name="TEST 2: Contradicción (Ceguera de Color)", 
        prompt_phrase="blue car"
    )
    
    # 3. El Test de Falla Espacial
    ejecutar_prueba(
        image_path="suv_llanta_trasera.jpg", 
        test_name="TEST 3: Falla Espacial (Relaciones de Posición)", 
        prompt_phrase="spare tire on the roof"
    )