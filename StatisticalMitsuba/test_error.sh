#!/bin/bash

GROUND_TRUTH="./io/transient/kitchen/transient_data/transient_data_50000.npy"
RESULTS_DIR="./io/transient/kitchen/denoised_results"

if [[ ! -f "$GROUND_TRUTH" ]]; then
    echo "Ground truth no encontrado en $GROUND_TRUTH"
    exit 1
fi

# Elimina CSV anterior si existe
rm -f metrics.csv

for TRANSFORM in bc yj; do
    echo "Procesando transformacion: $TRANSFORM"
    for FILE in "$RESULTS_DIR/$TRANSFORM"/*.npy; do
        echo "→ Comparando $FILE"
        python3 errorScript.py "$TRANSFORM" "$GROUND_TRUTH" "$FILE"
    done
done

echo "Resultados guardados en metrics.csv"
