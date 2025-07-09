#!/bin/bash

ESCENA="cbox"
SPP=2048

GROUND_TRUTH="io/transient/${ESCENA}/transient_data_1000000.npy"

SPATIAL_RADIUS_LIST=(3 5 10)
TEMPORAL_RADIUS_LIST=(0 1 3 5)
ALPHA_LIST=(0.005 0.05 0.1 0.2 0.5)

for spatial in "${SPATIAL_RADIUS_LIST[@]}"; do
  for temporal in "${TEMPORAL_RADIUS_LIST[@]}"; do
    for alpha in "${ALPHA_LIST[@]}"; do

      DENOISED="io/transient/${ESCENA}/denoised_transient_${SPP}_${spatial}_${temporal}_${alpha}.npy"

      echo "📊 Evaluando con spatial=$spatial, temporal=$temporal, alpha=$alpha" 

      python error.py "$GROUND_TRUTH" "$DENOISED" 
      
      echo "-----------------------------"
    done
  done
done

echo "✅ Evaluación completada."
