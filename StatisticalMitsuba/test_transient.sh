#!/bin/bash

# Número de muestras por píxel
SPP=2048

# Ruta del script Python
PYTHON_SCRIPT="transient.py"

# Ruta a las escenas
SCENE_DIR="../scenes/transient/kitchen"
# Directorio de salida
OUTPUT_DIR="./io/transient/kitchen/"

# Asegúrate de que el script existe
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "No se encontró el script $PYTHON_SCRIPT"
    exit 1
fi
# Recorrer archivos XML de transformaciones en la carpeta indicada
for scene_file in "$SCENE_DIR"/scene_bc_*.xml "$SCENE_DIR"/scene_yj_*.xml; do
    # Verificar que el archivo existe (por si no matchea nada)
    [ -e "$scene_file" ] || continue

    # Extraer nombre base (ej. scene_bc_1.2.xml)
    base_name=$(basename "$scene_file")

    # Extraer tipo de transformación
    if [[ "$base_name" == scene_bc_* ]]; then
        transform="bc"
    elif [[ "$base_name" == scene_yj_* ]]; then
        transform="yj"
    else
        echo "Archivo no reconocido: $scene_file"
        continue
    fi

    # Extraer lambda del nombre del archivo
   
    lambda_val=$(echo "$base_name" | sed -E "s/scene_${transform}_(-?[0-9.]+)\.xml/\1/")


    # Ejecutar el script Python
    echo "Ejecutando: $scene_file con transform=$transform, lambda=$lambda_val"
    python3 "$PYTHON_SCRIPT" "$SPP" "$scene_file" "$OUTPUT_DIR" "$transform" "$lambda_val"
done
