import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import os
import json
from tqdm import tqdm
import time

def load_florence_model():
    """Carga Florence-2-base. Entra perfectamente en 8GB de VRAM sin cuantizar."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Cargando Florence-2 en {device}...")
    
    model_id = 'microsoft/Florence-2-base'
    # trust_remote_code=True es obligatorio para modelos de Microsoft en HuggingFace
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    return model, processor, device

def run_florence_inference(model, processor, device, image_path, task="<VQA>", question=""):
    """Pasa una imagen por Florence y devuelve el texto generado."""
    image = Image.open(image_path).convert("RGB")

    prompt = task + question
    
    # Procesar entrada
    inputs = processor(
        text=prompt, 
        images=image, 
        return_tensors="pt").to(device)
    
    # Generar texto
    # max_new_tokens=1024 permite descripciones largas sin cortarse
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3 # Mejora la coherencia del texto
    )
    
    # Decodificar salida
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    
    # Limpiar el formato de salida de Florence
    parsed_answer = processor.post_process_generation(generated_text, task=task, image_size=(image.width, image.height))
    return parsed_answer[task]



def process_dataset_florence(test_dir, num_images=50):
    """Procesa un batch de prueba y guarda los resultados en JSON."""
    model, processor, device = load_florence_model()

    task_token = "<MORE_DETAILED_CAPTION>"
    # task_token = "<VQA>"
    # question_text = " What is the vehicle color, vehicle type, view angle (front, rear, side), time of day, and any external accessories like roof racks or spare tires?"
    # question_text = ' Vehicle color?'
    question_text =''
    # Tomar solo un subset para no tardar horas en la prueba inicial
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))][:num_images]
    
    results = {}
    print(f"\n[*] Iniciando procesamiento de {num_images} imágenes con Florence-2...")
    
    start_time = time.time()
    
    for img_name in tqdm(image_files, desc="Generando Descripciones"):
        img_path = os.path.join(test_dir, img_name)
        try:
            description = run_florence_inference(model, processor, device, img_path, task=task_token, question=question_text)
            results[img_name] = description
        except Exception as e:
            print(f"[-] Error procesando {img_name}: {e}")
            
    end_time = time.time()
    
    # Cálculos de latencia
    total_time = end_time - start_time
    ms_per_image = (total_time / num_images) * 1000
    
    print("\n" + "="*50)
    print(f"RENDIMIENTO FLORENCE-2-BASE")
    print(f"Tiempo Total : {total_time:.2f} segundos")
    print(f"Por Imagen   : {ms_per_image:.2f} ms ({num_images/total_time:.2f} FPS)")
    print("="*50)
    
    # Guardar resultados
    os.makedirs('data', exist_ok=True)
    out_path = 'data/florence_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"\n[+] Resultados guardados en {out_path}")

if __name__ == "__main__":
    # Ajusta la ruta a tu carpeta de test real de VeRi
    test_directory = "/home/ulises-calderon/Projects/ImageDescriptors/VeRi/image_test" 
    
    # Ejecutamos con 20 imágenes para medir velocidad real en tu PC
    process_dataset_florence(test_directory, num_images=20)