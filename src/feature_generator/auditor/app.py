import streamlit as st
import pandas as pd
import json
import os
import sys

# Agregar la ruta principal de feature_generator para poder importar el parser
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from parse_outputs import parse_florence_output, parse_qwen_output

# ==========================================
# CONFIGURACIÓN INICIAL
# ==========================================
st.set_page_config(page_title="VLM Auditor", layout="wide")

# Rutas a los datos (Ajusta la ruta de imágenes según tu entorno)
IMG_DIR = "/home/ulises-calderon/Projects/ImageDescriptors/VeRi/image_test"
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

# ==========================================
# MANEJO DEL ESTADO DE LA APLICACIÓN
# ==========================================
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0

if 'model_choice' not in st.session_state:
    st.session_state.model_choice = "qwen"  # Valor por defecto

if 'audit_results' not in st.session_state:
    st.session_state.audit_results = []

def load_data(model_choice):
    """Carga el JSON original con los textos/respuestas crudas del modelo."""
    file_path = os.path.join(RESULTS_DIR, f"{model_choice}_results.json")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"No se encontró el archivo: {file_path}")
        return {}

def save_audit(image_name, evaluation, model_name):
    """Guarda la evaluación y avanza a la siguiente imagen."""
    st.session_state.audit_results.append({
        "imageName": image_name,
        "model": model_name,
        "evaluation": evaluation
    })
    
    # Guardar en CSV en tiempo real
    df = pd.DataFrame(st.session_state.audit_results)
    df.to_csv(os.path.join(RESULTS_DIR, f"audit_report_{model_name}.csv"), index=False)
    
    st.session_state.current_index += 1

# ==========================================
# INTERFAZ DE USUARIO (SIDEBAR)
# ==========================================
st.sidebar.title("🛠️ Configuración")
new_model = st.sidebar.radio("Selecciona el modelo a evaluar:", ["qwen", "florence"])

# Si el usuario cambia de modelo, reiniciar el índice
if new_model != st.session_state.model_choice:
    st.session_state.model_choice = new_model
    st.session_state.current_index = 0
    # Cargar resultados previos si existen para no sobreescribir
    csv_path = os.path.join(RESULTS_DIR, f"audit_report_{new_model}.csv")
    if os.path.exists(csv_path):
        df_prev = pd.read_csv(csv_path)
        st.session_state.audit_results = df_prev.to_dict('records')
        st.session_state.current_index = len(df_prev) # Continuar donde nos quedamos
    else:
        st.session_state.audit_results = []

data = load_data(st.session_state.model_choice)
image_keys = list(data.keys())
total_images = len(image_keys)

st.sidebar.markdown("---")
st.sidebar.write(f"**Progreso:** {st.session_state.current_index} / {total_images}")

# Generar reporte resumen en el sidebar
if st.session_state.audit_results:
    df_report = pd.DataFrame(st.session_state.audit_results)
    summary = df_report['evaluation'].value_counts()
    st.sidebar.markdown("### Resumen Actual")
    st.sidebar.dataframe(summary)

# ==========================================
# INTERFAZ PRINCIPAL
# ==========================================
if not data:
    st.warning("No hay datos para mostrar. Genera los JSONs primero.")
    st.stop()

if st.session_state.current_index >= total_images:
    st.success("¡Auditoría Completada! Revisa el CSV en la carpeta data.")
    st.stop()

# Obtener datos de la imagen actual
current_image = image_keys[st.session_state.current_index]
raw_output = data[current_image]
img_path = os.path.join(IMG_DIR, current_image)

st.title(f"Auditoría Visual: {st.session_state.model_choice.upper()}")
st.write(f"**Archivo:** `{current_image}`")

col1, col2 = st.columns([1, 1])

# COLUMNA 1: La Imagen
with col1:
    try:
        st.image(img_path, use_container_width=True)
    except FileNotFoundError:
        st.error("No se encontró la imagen en el directorio especificado.")

# COLUMNA 2: Resultados y Botones
with col2:
    st.subheader("1. Resultado Final (Base de Datos)")
    
    # Procesar con el parser en tiempo real
    if st.session_state.model_choice == "florence":
        parsed_result = parse_florence_output(raw_output)
    else:
        parsed_result = parse_qwen_output(raw_output)
        
    st.json(parsed_result)
    
    with st.expander("2. Texto Original Crudo (Del Modelo)"):
        st.text(raw_output)
        
    st.markdown("---")
    st.subheader("Calificar Pipeline Completo")
    st.write("¿Cómo describirías el resultado del JSON comparado con la imagen real?")
    
    # Botones de Evaluación
    bcol1, bcol2 = st.columns(2)
    
    with bcol1:
        if st.button("✅ Perfecto (0 Errores)", use_container_width=True):
            save_audit(current_image, "Perfecto", st.session_state.model_choice)
            st.rerun()
            
        if st.button("🙈 Ceguera (Faltó algo obvio)", use_container_width=True):
            save_audit(current_image, "Falso Negativo (Ceguera)", st.session_state.model_choice)
            st.rerun()
            
    with bcol2:
        if st.button("👻 Alucinación (Inventó cosas)", use_container_width=True):
            save_audit(current_image, "Falso Positivo (Alucinacion)", st.session_state.model_choice)
            st.rerun()
            
        if st.button("💔 Falla del Parser (El modelo acertó)", use_container_width=True):
            save_audit(current_image, "Falla de Parser", st.session_state.model_choice)
            st.rerun()