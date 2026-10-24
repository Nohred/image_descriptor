import os
import time
import argparse
import pandas as pd
from tqdm import tqdm

# parse_outputs solo usa librerías estándar (json, re), puede importarse en ambos entornos
from parse_outputs import parse_florence_output, parse_qwen_output

def run_generative_benchmark(test_dir, target_model, num_images):
    os.makedirs('data', exist_ok=True)
    output_csv = f"data/generative_predictions_{target_model}.csv"
    
    # 1. Cargar el modelo solicitado condicionalmente
    if target_model == "florence":
        print("\n[*] Inicializando Florence-2...")
        from run_florence import load_florence_model, run_florence_inference
        model, processor, device = load_florence_model()
        florence_task = "<MORE_DETAILED_CAPTION>"
        florence_question = ""
    
    elif target_model == "qwen":
        print("\n[*] Inicializando Qwen-VL...")
        from run_qwen import load_light_qwen_model, run_qwen_inference
        model, processor = load_light_qwen_model()
        qwen_prompt = """
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
    
    # 2. Preparar Dataset
    image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))][:num_images]
    print(f"\n[*] Iniciando Benchmark ({target_model}) con {len(image_files)} imágenes.")
    print(f"[*] Se guardará progresivamente en: {output_csv}")
    
    results_list = []
    start_time = time.time()
    
    # 3. Bucle Principal de Inferencia
    for img_name in tqdm(image_files, desc=f"Evaluando {target_model}"):
        img_path = os.path.join(test_dir, img_name)
        
        row_data = {
            "imageName": img_name,
            f"{target_model}_color": "error", 
            f"{target_model}_type": "error"
        }
        
        try:
            if target_model == "florence":
                raw_text = run_florence_inference(
                    model, processor, device, img_path, 
                    task=florence_task, question=florence_question
                )
                parsed = parse_florence_output(raw_text)
            
            elif target_model == "qwen":
                raw_text = run_qwen_inference(model, processor, img_path, qwen_prompt)
                parsed = parse_qwen_output(raw_text)

            if parsed:
                row_data[f"{target_model}_color"] = parsed["features"]["color"]
                row_data[f"{target_model}_type"] = parsed["features"]["vehicle_type"]

        except Exception as e:
            print(f"\n[-] Error en {img_name}: {e}")

        results_list.append(row_data)
        
        # Guardado progresivo cada 50 imágenes
        if len(results_list) % 50 == 0:
            pd.DataFrame(results_list).to_csv(output_csv, index=False)

    # 4. Guardado Final y Métricas
    pd.DataFrame(results_list).to_csv(output_csv, index=False)
    total_time = time.time() - start_time
    
    print("\n" + "="*50)
    print(f"BENCHMARK COMPLETADO: {target_model.upper()}")
    print(f"Tiempo Total : {total_time/3600:.2f} horas")
    print(f"Promedio     : {total_time/len(image_files):.2f} seg/imagen")
    print("="*50)

if __name__ == "__main__":
    test_directory = "/home/ulises-calderon/Projects/ImageDescriptors/VeRi/image_test"  # Cambia según tu entorno

    parser = argparse.ArgumentParser(description="Benchmark Generativo para VeRi")
    parser.add_argument("--model", type=str, choices=["florence", "qwen"], required=True, help="Modelo a evaluar")
    parser.add_argument("--num_images", type=int, default=5400, help="Cantidad de imágenes a procesar")    
    args = parser.parse_args()
    
    run_generative_benchmark(test_directory, args.model, args.num_images)