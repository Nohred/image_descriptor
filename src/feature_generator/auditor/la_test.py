import torch
import traceback
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from transformers import AutoModel, AutoProcessor, BitsAndBytesConfig, AutoTokenizer
from qwen_vl_utils import process_vision_info

# 1. CARGA DEL MODELO 
model_id = "nvidia/LocateAnything-3B"
print(f"[*] Cargando {model_id} en 4-bits con SDPA...")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16 
)

model = AutoModel.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map="cuda",
    torch_dtype=torch.float16,
    quantization_config=quant_config,
    attn_implementation="sdpa" 
).eval()

for name, module in model.named_modules():
    if hasattr(module, "_attn_implementation"):
        module._attn_implementation = "sdpa"

if hasattr(model.config, "_attn_implementation"):
    model.config._attn_implementation = "sdpa"
if hasattr(model, "language_model") and hasattr(model.language_model.config, "_attn_implementation"):
    model.language_model.config._attn_implementation = "sdpa"
if hasattr(model, "vision_model") and hasattr(model.vision_model.config, "_attn_implementation"):
    model.vision_model.config._attn_implementation = "sdpa"

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


# =====================================================================
# FUNCIÓN AUXILIAR: Dibujar y guardar las cajas
# =====================================================================
def guardar_imagen_con_cajas(image_path, respuesta, test_name):
    # Encontrar todas las cajas con formato <box><Y1><X1><Y2><X2></box>
    # Extraemos solo los números usando expresiones regulares
    cajas = re.findall(r'<box><(\d+)><(\d+)><(\d+)><(\d+)></box>', respuesta)
    
    if not cajas:
        return
        
    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size
    
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    
    for y1, x1, y2, x2 in cajas:
        # Los números del modelo están normalizados de 0 a 1000
        # Convertimos esas proporciones a píxeles reales de la imagen
        y1_px = (int(y1) / 1000.0) * img_h
        x1_px = (int(x1) / 1000.0) * img_w
        y2_px = (int(y2) / 1000.0) * img_h
        x2_px = (int(x2) / 1000.0) * img_w
        
        ancho = x2_px - x1_px
        alto = y2_px - y1_px
        
        # Dibujamos un rectángulo rojo
        rect = patches.Rectangle((x1_px, y1_px), ancho, alto, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        
    plt.axis('off')
    
    # Limpiamos el nombre del test para usarlo como nombre de archivo
    nombre_archivo = f"{test_name.split(':')[0].strip().replace(' ', '_')}_resultado.jpg"
    plt.savefig(nombre_archivo, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"[+] Imagen guardada como: {nombre_archivo}")


# 2. FUNCIÓN DE LA PRUEBA
def ejecutar_prueba(image_path, test_name, prompt_phrase):
    print(f"\n{'='*50}\n[>] {test_name}\n[?] Buscando: '{prompt_phrase}'")
    
    try:
        img = Image.open(image_path).convert("RGB")
        # Mantuvimos tu redimensionamiento original
        img.thumbnail((800, 800)) 
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image", 
                        "image": img, 
                        "resized_height": 512, 
                        "resized_width": 512
                    },
                    {"type": "text", "text": f"Locate the region that matches the following description: {prompt_phrase}."}
                ]
            }
        ]
        
        text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to("cuda")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=64, 
                use_cache=True,
                tokenizer=tokenizer
            )
            
        if isinstance(outputs, torch.Tensor):
            prompt_len = inputs.input_ids.shape[1]
            respuesta_cruda = tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True)
        elif isinstance(outputs, list):
            respuesta_cruda = str(outputs[0])
        else:
            respuesta_cruda = str(outputs)
        
        prompt_oficial = f"Locate the region that matches the following description: {prompt_phrase}."
        respuesta = respuesta_cruda.replace(prompt_oficial, "").replace("<|im_end|>", "").strip()
        
        print(f"[*] Respuesta cruda del modelo:\n{respuesta}")
        
        guardar_imagen_con_cajas(image_path, respuesta, test_name)

            
    except Exception as e:
        print(f"[-] Error procesando la imagen {image_path}:\n{traceback.format_exc()}")

# ==========================================
# 3. EJECUCIÓN DE TUS GRUPOS EXPERIMENTALES
# ==========================================
if __name__ == "__main__":
    
    ejecutar_prueba(
        image_path="sedan_liso.jpg", 
        test_name="TEST 1: Objeto Fantasma (Contexto Forzado)", 
        prompt_phrase="roof rack"
    )
    
    ejecutar_prueba(
        image_path="auto_rojo.jpg", 
        test_name="TEST 2: Contradicción (Ceguera de Color)", 
        prompt_phrase="blue car"
    )
    
    ejecutar_prueba(
        image_path="suv_llanta_trasera.jpg", 
        test_name="TEST 3: Falla Espacial (Relaciones de Posición)", 
        prompt_phrase="spare tire on the roof"
    )