import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np

# Importar tus módulos
from dataset import get_dataloaders
from models import MultiTaskVehicleNet
from plotting import plot_training_history  # <-- NUEVO IMPORT

def entrenar_modelo(epochs=10, batch_size=32, lr=0.001, backbone='efficientnet_b0'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Entrenando en: {device}")
    
    train_csv = "data/train_features.csv"
    train_dir = "/home/ulises-calderon/Projects/ImageDescriptorsv2/VeRi/image_train"
    test_csv = "data/test_features.csv"
    test_dir = "/home/ulises-calderon/Projects/ImageDescriptorsv2/VeRi/image_test"

    print("[*] Cargando DataLoaders...")
    train_loader, val_loader, _test_loader = get_dataloaders(
        train_csv, train_dir, test_csv, test_dir, batch_size
    )

    color_weights = torch.tensor([2.8792, 23.5912, 2.1350, 0.4153, 0.8661, 1.1639, 0.6285, 2.1907, 4.2796, 0.4100], dtype=torch.float32).to(device)
    type_weights = torch.tensor([0.2106, 1.0077, 2.0793, 2.0907, 4.5987, 1.7352, 1.5075, 1.2267, 34.9500], dtype=torch.float32).to(device)

    print(f"[*] Inicializando arquitectura: {backbone}")
    model = MultiTaskVehicleNet(backbone_name=backbone, num_colors=10, num_types=9).to(device)

    criterion_color = nn.CrossEntropyLoss(weight=color_weights)
    criterion_type = nn.CrossEntropyLoss(weight=type_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    os.makedirs('weights', exist_ok=True)
    best_loss = float('inf')
    
    # --- NUEVO: Diccionario para guardar el historial ---
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc_color': [],
        'val_acc_type': []
    }

    for epoch in range(epochs):
        print(f"\n--- Época {epoch+1}/{epochs} ---")
        
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc="Entrenando")
        
        for images, labels_color, labels_type in progress_bar:
            images = images.to(device)
            labels_color = labels_color.to(device)
            labels_type = labels_type.to(device)
            
            optimizer.zero_grad()
            preds_color, preds_type, _ = model(images)
            loss_c = criterion_color(preds_color, labels_color)
            loss_t = criterion_type(preds_type, labels_type)
            loss_total = loss_c + loss_t
            loss_total.backward()
            optimizer.step()
            
            train_loss += loss_total.item()
            progress_bar.set_postfix({'loss': f"{loss_total.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)
        print(f"[*] Loss Promedio Entrenamiento: {avg_train_loss:.4f}")

        model.eval()
        val_loss = 0.0
        correct_color, correct_type = 0, 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels_color, labels_type in tqdm(val_loader, desc="Validando"):
                images = images.to(device)
                labels_color = labels_color.to(device)
                labels_type = labels_type.to(device)
                
                preds_color, preds_type, _ = model(images)
                
                loss_c = criterion_color(preds_color, labels_color)
                loss_t = criterion_type(preds_type, labels_type)
                val_loss += (loss_c + loss_t).item()
                
                _, pred_c = torch.max(preds_color, 1)
                _, pred_t = torch.max(preds_type, 1)
                
                correct_color += (pred_c == labels_color).sum().item()
                correct_type += (pred_t == labels_type).sum().item()
                total_samples += labels_color.size(0)

        avg_val_loss = val_loss / len(val_loader)
        acc_color = (correct_color / total_samples) * 100
        acc_type = (correct_type / total_samples) * 100
        
        print(f"[*] Loss Validación: {avg_val_loss:.4f}")
        print(f"[*] Acc Color: {acc_color:.2f}% | Acc Tipo: {acc_type:.2f}%")

        # --- Guardar métricas de la época actual ---
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc_color'].append(acc_color)
        history['val_acc_type'].append(acc_type)

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            save_path = f"weights/best_{backbone}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"[!] Nuevo mejor modelo guardado en {save_path}")

    # --- Retornar historial al finalizar el ciclo de épocas ---
    return history

if __name__ == "__main__":
    # Entrenar MobileNet y graficar
    hist_mobile = entrenar_modelo(epochs=15, batch_size=32, lr=0.0003, backbone='mobilenet_v3_small')
    plot_training_history(hist_mobile, backbone_name='mobilenet_v3_small')

    # Entrenar EfficientNet y graficar
    hist_eff = entrenar_modelo(epochs=15, batch_size=32, lr=0.0003, backbone='efficientnet_b0')
    plot_training_history(hist_eff, backbone_name='efficientnet_b0')