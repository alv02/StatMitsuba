#!/bin/bash

# Carpetas
TRANSFORMATIONS=("bc" "yj")
OUTPUT_ROOT="./io/transient/kitchen/denoised_results"
AOVS_PATH="./io/steady/kitchen/imagen.exr"
SPP=2048
SPATIAL_RADIUS=5
TEMPORAL_RADIUS=3

ALPHAS=(0.005 0.05 0.1 0.2 0.5 0.8)



for TRANSFORM in "${TRANSFORMATIONS[@]}"; do
    echo "=== Procesando transformación: $TRANSFORM ==="

    STATS_DIR="./io/transient/kitchen/transient_stats/$TRANSFORM"
    TRANSIENT_PATH="./io/transient/kitchen/transient_data/transient_data_${SPP}.npy"
    OUT_DIR="$OUTPUT_ROOT/$TRANSFORM"


    # Para cada estimands encontrado
    for ESTIMANDS_FILE in "$STATS_DIR"/estimands_*_${SPP}.npy; do
        FILENAME=$(basename "$ESTIMANDS_FILE")
        LAMBDA=$(echo "$FILENAME" | sed -E "s/estimands_(-?[0-9.]+)_${SPP}\.npy/\1/")

        ESTIMANDS_PATH="$STATS_DIR/estimands_${LAMBDA}_${SPP}.npy"
        VARIANCE_PATH="$STATS_DIR/estimands_variance_${LAMBDA}_${SPP}.npy"

        if [[ ! -f "$VARIANCE_PATH" ]]; then
            echo ">> Falta varianza para lambda=$LAMBDA en $TRANSFORM, saltando..."
            continue
        fi

        for ALPHA in "${ALPHAS[@]}"; do
            OUTPUT_FILE="$OUT_DIR/denoised_lambda_${LAMBDA}_alpha_${ALPHA}.npy"
            echo ">> Ejecutando lambda=$LAMBDA alpha=$ALPHA → $OUTPUT_FILE"

            python denoiserTransient.py \
                "$TRANSIENT_PATH" "$AOVS_PATH" "$ESTIMANDS_PATH" "$VARIANCE_PATH" \
                "$SPP" "$SPATIAL_RADIUS" "$TEMPORAL_RADIUS" "$ALPHA" "$OUTPUT_FILE"
        done
    done
done
