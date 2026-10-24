#!/bin/bash

# Este comando permite que `conda activate` funcione dentro de un script bash
eval "$(conda shell.bash hook)"

NUM_IMAGES=5400

echo "================================================="
echo " INICIANDO PIPELINE DE BENCHMARK GENERATIVO"
echo "================================================="

# 1. Correr Florence-2
echo -e "\n[*] Activando entorno de Florence..."
conda activate image-descriptors-florence

echo "[*] Ejecutando benchmark de Florence-2..."
python generate_predictions.py --model florence --num_images $NUM_IMAGES

# 2. Correr Qwen-VL
echo -e "\n[*] Activando entorno de Qwen..."
conda activate image-descriptors-qwen

echo "[*] Ejecutando benchmark de Qwen-VL..."
python generate_predictions.py --model qwen --num_images $NUM_IMAGES

echo -e "\n================================================="
echo " BENCHMARK TOTAL FINALIZADO EXITOSAMENTE"
echo "================================================="