import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix


# Los catálogos oficiales (Deben coincidir con los IDs del test_features.csv)
VALID_COLORS = ['yellow', 'orange', 'green', 'gray', 'red', 'blue', 'white', 'golden', 'brown', 'black']
VALID_TYPES = ['sedan', 'suv', 'van', 'hatchback', 'mpv', 'pickup', 'bus', 'truck', 'estate']

def plot_confusion_matrices(true_color, pred_color, true_type, pred_type, model_name):
    """Genera y guarda las matrices de confusión."""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f'Matrices de Confusión (Generativo): {model_name.upper()}', fontsize=16, fontweight='bold')

    # Obtenemos todas las etiquetas únicas presentes (incluyendo "unknown" o "error" si los hubo)
    color_labels = sorted(list(set(true_color + pred_color)))
    type_labels = sorted(list(set(true_type + pred_type)))

    # Matriz Color
    cm_color = confusion_matrix(true_color, pred_color, labels=color_labels)
    sns.heatmap(cm_color, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=color_labels, yticklabels=color_labels)
    axes[0].set_title('Predicción de Color')
    axes[0].set_ylabel('Etiqueta Real')
    axes[0].set_xlabel('Predicción del VLM')
    axes[0].tick_params(axis='x', rotation=45)

    # Matriz Tipo
    cm_type = confusion_matrix(true_type, pred_type, labels=type_labels)
    sns.heatmap(cm_type, annot=True, fmt='d', cmap='Oranges', ax=axes[1],
                xticklabels=type_labels, yticklabels=type_labels)
    axes[1].set_title('Predicción de Tipo de Vehículo')
    axes[1].set_ylabel('Etiqueta Real')
    axes[1].set_xlabel('Predicción del VLM')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    os.makedirs('data', exist_ok=True)
    save_path = f'data/confusion_matrix_{model_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[+] Matriz de confusión guardada en: {save_path}")
    plt.close()

def evaluate_model_csv(true_csv_path, pred_csv_path, model_name):
    print("\n" + "="*50)
    print(f"  MÉTRICAS FINALES (TEST SET) - {model_name.upper()}")
    print("="*50)
    
    # 1. Validar que el archivo exista
    if not os.path.exists(pred_csv_path):
        print(f"[-] No se encontró el CSV de predicciones: {pred_csv_path}")
        return

    # 2. Cargar los Datos
    df_true = pd.read_csv(true_csv_path)
    df_pred = pd.read_csv(pred_csv_path)
    
    # 3. Mapear los IDs numéricos del CSV real a texto
    df_true['true_color'] = df_true['colorID'].apply(lambda x: VALID_COLORS[int(x)])
    df_true['true_type'] = df_true['typeID'].apply(lambda x: VALID_TYPES[int(x)])
    
    # 4. Cruzar los datos reales con las predicciones usando el imageName
    df_merged = pd.merge(df_true, df_pred, on='imageName', how='inner')
    
    if len(df_merged) == 0:
        print("[-] Error: No hay imágenes coincidentes entre las etiquetas reales y las predicciones.")
        return
        
    print(f"[*] Imágenes evaluadas: {len(df_merged)}")
    
    # 5. Extraer listas para Scikit-Learn
    pred_col_color = f"{model_name}_color"
    pred_col_type = f"{model_name}_type"
    
    y_true_c = df_merged['true_color'].tolist()
    y_pred_c = df_merged[pred_col_color].tolist()
    
    y_true_t = df_merged['true_type'].tolist()
    y_pred_t = df_merged[pred_col_type].tolist()
    
    # --- RESULTADOS COLOR ---
    print("\n--- RESULTADOS: COLOR ---")
    present_colors = sorted(list(set(y_true_c)))
    
    acc_c = accuracy_score(y_true_c, y_pred_c)
    # labels=present_colors asegura que excluyamos clases reales con 0 ejemplos
    f1_macro_c = f1_score(y_true_c, y_pred_c, average='macro', labels=present_colors)
    
    print(f"Accuracy Global : {acc_c*100:.2f}%")
    print(f"F1-Score Macro  : {f1_macro_c*100:.2f}% (Ajustado a clases presentes)\n")
    print("Reporte Detallado por Color:")
    print(classification_report(y_true_c, y_pred_c, labels=present_colors, target_names=present_colors, zero_division=0))
    
    # --- RESULTADOS TIPO ---
    print("\n--- RESULTADOS: TIPO DE VEHÍCULO ---")
    present_types = sorted(list(set(y_true_t)))
    
    acc_t = accuracy_score(y_true_t, y_pred_t)
    f1_macro_t = f1_score(y_true_t, y_pred_t, average='macro', labels=present_types)
    
    print(f"Accuracy Global : {acc_t*100:.2f}%")
    print(f"F1-Score Macro  : {f1_macro_t*100:.2f}% (Ajustado a clases presentes)\n")
    print("Reporte Detallado por Tipo:")
    print(classification_report(y_true_t, y_pred_t, labels=present_types, target_names=present_types, zero_division=0))
    
    # 6. Generar Gráficas
    plot_confusion_matrices(y_true_c, y_pred_c, y_true_t, y_pred_t, model_name)

if __name__ == "__main__":
    # Ruta al CSV de validación real (Asumiendo tu estructura de carpetas)
    true_labels_csv = "../feature_extractor/data/test_features.csv" 
    
    # Rutas a los CSVs generados por tu bash script
    florence_csv = "data/generative_predictions_florence.csv"
    qwen_csv = "data/generative_predictions_qwen.csv"
    
    # Evaluar ambos
    evaluate_model_csv(true_labels_csv, florence_csv, "florence")
    evaluate_model_csv(true_labels_csv, qwen_csv, "qwen")