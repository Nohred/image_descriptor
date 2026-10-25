import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from tqdm import tqdm
import os
import time

from dataset import get_dataloaders
from models import MultiTaskVehicleNet

COLOR_NAMES = ['yellow', 'orange', 'green', 'gray', 'red', 'blue', 'white', 'golden', 'brown', 'black']
TYPE_NAMES = ['sedan', 'suv', 'van', 'hatchback', 'mpv', 'pickup', 'bus', 'truck', 'estate']

def plot_confusion_matrices(true_color, pred_color, true_type, pred_type, backbone_name):
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f'Matrices de Confusión: {backbone_name}', fontsize=16, fontweight='bold')

    cm_color = confusion_matrix(true_color, pred_color)
    sns.heatmap(cm_color, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=COLOR_NAMES, yticklabels=COLOR_NAMES)
    axes[0].set_title('Predicción de Color')
    axes[0].set_ylabel('Etiqueta Real')
    axes[0].set_xlabel('Predicción del Modelo')
    axes[0].tick_params(axis='x', rotation=45)

    cm_type = confusion_matrix(true_type, pred_type)
    sns.heatmap(cm_type, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                xticklabels=TYPE_NAMES, yticklabels=TYPE_NAMES)
    axes[1].set_title('Predicción de Tipo de Vehículo')
    axes[1].set_ylabel('Etiqueta Real')
    axes[1].set_xlabel('Predicción del Modelo')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    os.makedirs('data', exist_ok=True)
    save_path = f'data/confusion_matrix_{backbone_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[+] Matriz de confusión guardada en: {save_path}")
    plt.close()

def evaluar_modelo(backbone='efficientnet_b0', batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights_path = f"weights/best_{backbone}.pth"
    
    if not os.path.exists(weights_path):
        print(f"[-] Error: No se encontraron los pesos en {weights_path}.")
        return

    print(f"\n[*] Evaluando modelo: {backbone} en {device}")
    
    train_csv = "data/train_features.csv"
    train_dir = "/home/ulises-calderon/Projects/ImageDescriptors/VeRi/image_train"
    test_csv = "data/test_features.csv"
    test_dir = "/home/ulises-calderon/Projects/ImageDescriptors/VeRi/image_test"

    _, _, test_loader = get_dataloaders(
        train_csv, train_dir, test_csv, test_dir, batch_size=batch_size
    )

    model = MultiTaskVehicleNet(backbone_name=backbone, num_colors=10, num_types=9).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    all_true_color, all_pred_color = [], []
    all_true_type, all_pred_type = [], []

    print("[*] Iniciando inferencia para cálculo de métricas y latencia...")
    
    # Sincronizar CUDA antes de medir para evitar contar tiempos de transferencia previos
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    start_time = time.time()

    with torch.no_grad():
        for images, labels_color, labels_type in tqdm(test_loader, desc="Procesando Test Set"):
            images = images.to(device)
            
            preds_color, preds_type, _ = model(images)
            
            _, pred_c = torch.max(preds_color, 1)
            _, pred_t = torch.max(preds_type, 1)
            
            all_true_color.extend(labels_color.cpu().numpy())
            all_pred_color.extend(pred_c.cpu().numpy())
            
            all_true_type.extend(labels_type.cpu().numpy())
            all_pred_type.extend(pred_t.cpu().numpy())

    # Sincronizar de nuevo para medir el tiempo real de la GPU
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    # Cálculos de latencia
    total_time = end_time - start_time
    total_images = len(all_true_color)
    ms_per_image = (total_time / total_images) * 1000
    fps = total_images / total_time

    print("\n" + "="*50)
    print(f"  MÉTRICAS FINALES (TEST SET) - {backbone.upper()}")
    print("="*50)
    
    print(f"\n⏱  RENDIMIENTO (INFERENCIA PURA)")
    print(f"Tiempo Total Test Set : {total_time:.2f} segundos")
    print(f"Tiempo por Imagen     : {ms_per_image:.2f} ms")
    print(f"Fotogramas por Seg.   : {fps:.2f} FPS")

    print("\n---  RESULTADOS: COLOR ---")
    # Filtrar clases presentes en el test set
    present_color_classes = sorted(list(set(all_true_color)))
    present_color_names = [COLOR_NAMES[i] for i in present_color_classes]
    
    acc_color = accuracy_score(all_true_color, all_pred_color)
    # Calcular el F1 Macro ignorando clases que no existen en y_true
    f1_macro_color = f1_score(all_true_color, all_pred_color, average='macro', labels=present_color_classes)
    
    print(f"Accuracy Global : {acc_color*100:.2f}%")
    print(f"F1-Score Macro  : {f1_macro_color*100:.2f}% (Ajustado a clases presentes)")
    print("\nReporte Detallado por Color:")
    print(classification_report(all_true_color, all_pred_color, target_names=present_color_names, labels=present_color_classes, zero_division=0))

    print("\n---  RESULTADOS: TIPO DE VEHÍCULO ---")
    # Filtrar clases presentes en el test set (Esto quita el MPV si tiene 0 muestras)
    present_type_classes = sorted(list(set(all_true_type)))
    present_type_names = [TYPE_NAMES[i] for i in present_type_classes]

    acc_type = accuracy_score(all_true_type, all_pred_type)
    # Calcular el F1 Macro ignorando clases que no existen en y_true
    f1_macro_type = f1_score(all_true_type, all_pred_type, average='macro', labels=present_type_classes)
    
    print(f"Accuracy Global : {acc_type*100:.2f}%")
    print(f"F1-Score Macro  : {f1_macro_type*100:.2f}% (Ajustado a clases presentes)")
    print("\nReporte Detallado por Tipo:")
    print(classification_report(all_true_type, all_pred_type, target_names=present_type_names, labels=present_type_classes, zero_division=0))

    plot_confusion_matrices(all_true_color, all_pred_color, all_true_type, all_pred_type, backbone)


if __name__ == "__main__":
    evaluar_modelo(backbone='efficientnet_b0', batch_size=32)
    # evaluar_modelo(backbone='mobilenet_v3_small', batch_size=32)