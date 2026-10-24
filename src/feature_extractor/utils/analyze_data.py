import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_and_get_weights(df, column_name, class_names):
    """Imprime la distribución y devuelve los pesos para PyTorch."""
    total_samples = len(df)
    counts = df[column_name].value_counts().sort_index()
    
    print(f"\n--- Análisis de {column_name.upper()} ---")
    print(f"{'Clase':<12} | {'Cantidad':<8} | {'Porcentaje':<10} | {'Peso PyTorch':<10}")
    print("-" * 50)
    
    weights = []
    for i in range(len(class_names)):
        count = counts.get(i, 0)
        pct = (count / total_samples) * 100 if total_samples > 0 else 0
        
        # Fórmula de peso inverso
        weight = total_samples / (len(class_names) * (count + 1e-6))
        weights.append(weight)
        
        name = class_names[i]
        print(f"{i}: {name:<9} | {count:<8} | {pct:>6.2f} %   | {weight:>8.4f}")
        
    return np.array(weights)

def plot_imbalance(df, color_names, type_names):
    """Genera y guarda un gráfico atractivo mostrando el desbalance de ambas características."""
    # Configurar el estilo visual (limpio y moderno)
    sns.set_theme(style="whitegrid")
    
    # Crear una figura con 1 fila y 2 columnas
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ---------------- GRÁFICO 1: COLOR ----------------
    color_counts = df['colorID'].value_counts().sort_index()
    x_colors = [color_names[i] for i in color_counts.index]
    
    sns.barplot(x=x_colors, y=color_counts.values, ax=axes[0], hue=x_colors, palette="viridis", legend=False)
    axes[0].set_title("Distribución de Colores (ColorID)", fontsize=16, fontweight='bold', pad=15)
    axes[0].set_xlabel("Clase de Color", fontsize=12)
    axes[0].set_ylabel("Cantidad de Imágenes", fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    
    # Agregar los números arriba de cada barra
    for p in axes[0].patches:
        axes[0].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5),
                         textcoords='offset points')

    # ---------------- GRÁFICO 2: TIPO ----------------
    type_counts = df['typeID'].value_counts().sort_index()
    x_types = [type_names[i] for i in type_counts.index]
    
    sns.barplot(x=x_types, y=type_counts.values, ax=axes[1], hue=x_types, palette="mako", legend=False)
    axes[1].set_title("Distribución de Tipos (TypeID)", fontsize=16, fontweight='bold', pad=15)
    axes[1].set_xlabel("Clase de Vehículo", fontsize=12)
    axes[1].set_ylabel("Cantidad de Imágenes", fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)

    # Agregar los números arriba de cada barra
    for p in axes[1].patches:
        axes[1].annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='baseline', fontsize=10, color='black', xytext=(0, 5),
                         textcoords='offset points')

    # Ajustar el espaciado para que no se corten los textos
    plt.tight_layout()
    
    # Guardar en alta calidad y mostrar
    output_path = "data/imbalance_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[+] Gráfico guardado exitosamente en: {output_path}")
    
    # Mostrar la ventana interactiva
    plt.show()

if __name__ == "__main__":
    COLOR_NAMES = ['yellow', 'orange', 'green', 'gray', 'red', 'blue', 'white', 'golden', 'brown', 'black']
    TYPE_NAMES = ['sedan', 'suv', 'van', 'hatchback', 'mpv', 'pickup', 'bus', 'truck', 'estate']
    
    try:
        df_train = pd.read_csv("data/train_features.csv")
    except FileNotFoundError:
        print("Error: Ejecuta build_csv.py primero.")
        exit()

    print(f"Total de imágenes en entrenamiento: {len(df_train)}")
    
    # Análisis en consola y extracción de pesos
    color_weights = analyze_and_get_weights(df_train, 'colorID', COLOR_NAMES)
    type_weights = analyze_and_get_weights(df_train, 'typeID', TYPE_NAMES)
    
    print("\n--- Arrays de Pesos para PyTorch ---")
    print(f"color_weights = {list(np.round(color_weights, 4))}")
    print(f"type_weights = {list(np.round(type_weights, 4))}")
    
    # Generar el gráfico
    plot_imbalance(df_train, COLOR_NAMES, TYPE_NAMES)