import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import os
import json
from tqdm import tqdm
import time

def load_qwen_model():
    """Carga Qwen2-VL-7B cuantizado a 8-bits usando bitsandbytes."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Cargando Qwen2-VL-7B en 8-bits ({device}). Esto puede tardar...")


    # Cambiamos al modelo de 7B
    model_id = "Qwen/Qwen2-VL-7B-Instruct" 
    
    # Cargamos en 8-bit en lugar de 4-bit
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_8bit=True # Requiere pip install bitsandbytes
    )

    processor = AutoProcessor.from_pretrained(model_id)

    return model, processor

def load_light_qwen_model():
    """Carga Qwen2.5-VL-3B cuantizado a 4-bits usando bitsandbytes."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Cargando Qwen2.5-VL-3B en 4-bits ({device}). Esto puede tardar...")
    
    model_id = "Qwen/Qwen2-VL-2B-Instruct" # Usamos la versión de 2B que es la oficial más ligera
    
    # Configuramos la cuantización a 4-bits para salvar VRAM
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True # Requiere bitsandbytes
    )
    processor = AutoProcessor.from_pretrained(model_id)
    
    return model, processor

def run_qwen_inference(model, processor, image_path, prompt):
    """Ejecuta inferencia conversacional con Qwen."""
    # Estructura de chat estándar de Qwen-VL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    # Procesar imagen y texto
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    ).to(model.device)
    
    # Generar respuesta
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    
    # Recortar el prompt de la salida para obtener solo la respuesta nueva
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    return output_text

def process_dataset_qwen(test_dir, num_images=50):

    # model, processor = load_qwen_model()
    model, processor = load_light_qwen_model()  # Usamos la versión ligera de 2B para ahorrar VRAM
    
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))][:num_images]
    results = {}
    
    # PROMPT ENGINEERING CRÍTICO
    system_prompt = """
    Analyze the vehicle in the image. Return ONLY a valid JSON with two main keys: 'fixed_attributes' and 'dynamic_attributes'.
    Do not include markdown tags, just the JSON.

    Use this schema:
    {
    "fixed_attributes": {
        "color": "choose one: yellow, orange, green, gray, red, blue, white, golden, brown, black",
        "vehicle_type": "choose one: sedan, suv, van, hatchback, mpv, pickup, bus, truck, estate"
    },
    "dynamic_attributes": {
        "time_of_day": "choose: day, night, unknown",
        "view_angle": "choose: front, rear, side, front-side, rear-side",
        "accessories": ["list visible external accessories "]
    }
    }
    """

    # like: roof rack, spare tire, pickup cover, spoiler, none
    
    print(f"\n[*] Iniciando procesamiento de {num_images} imágenes con Qwen-VL...")
    start_time = time.time()
    
    for img_name in tqdm(image_files, desc="Generando JSONs"):
        img_path = os.path.join(test_dir, img_name)
        try:
            response = run_qwen_inference(model, processor, img_path, system_prompt)
            results[img_name] = response
        except Exception as e:
            print(f"[-] Error procesando {img_name}: {e}")
            
    end_time = time.time()
    total_time = end_time - start_time
    
    print("\n" + "="*50)
    print(f"RENDIMIENTO QWEN-VL-2B (4-BIT)")
    print(f"Tiempo Total : {total_time:.2f} segundos")
    print(f"Por Imagen   : {total_time/num_images:.2f} s/img")
    print("="*50)
    
    os.makedirs('data', exist_ok=True)
    out_path = 'data/qwen_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\n[+] Resultados guardados en {out_path}")

if __name__ == "__main__":
    test_directory = "/home/ulises-calderon/Projects/ImageDescriptors/VeRi/image_test" 
    process_dataset_qwen(test_directory, num_images=20)