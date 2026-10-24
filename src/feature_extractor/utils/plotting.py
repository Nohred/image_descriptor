import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_training_history(history, backbone_name, save_dir="data"):
    """
    Genera y guarda gráficos de las curvas de entrenamiento.
    """
    os.makedirs(save_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Extraer datos
    epochs = range(1, len(history['train_loss']) + 1)
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    val_acc_color = history['val_acc_color']
    val_acc_type = history['val_acc_type']

    # Crear figura con 2 subplots (1 fila, 2 columnas)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Resultados de Entrenamiento: {backbone_name}', fontsize=16, fontweight='bold')

    # Gráfico 1: Curvas de Loss
    axes[0].plot(epochs, train_loss, label='Train Loss', marker='o', color='tab:blue')
    axes[0].plot(epochs, val_loss, label='Val Loss', marker='o', color='tab:orange')
    axes[0].set_title('Pérdida (Loss) por Época', fontsize=14)
    axes[0].set_xlabel('Épocas')
    axes[0].set_ylabel('Loss (CrossEntropy)')
    axes[0].legend()

    # Gráfico 2: Curvas de Accuracy
    axes[1].plot(epochs, val_acc_color, label='Acc Color', marker='s', color='tab:green')
    axes[1].plot(epochs, val_acc_type, label='Acc Tipo', marker='^', color='tab:red')
    axes[1].set_title('Precisión (Accuracy) de Validación', fontsize=14)
    axes[1].set_xlabel('Épocas')
    axes[1].set_ylabel('Precisión (%)')
    axes[1].legend()

    plt.tight_layout()
    
    # Guardar gráfico
    save_path = os.path.join(save_dir, f'history_{backbone_name}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[+] Gráfica de entrenamiento guardada en: {save_path}")
    
    # Cerrar figura para liberar memoria
    plt.close()