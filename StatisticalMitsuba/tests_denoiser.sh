#!/bin/bash

# Parámetros fijos
ESCENA="cbox"
SPP=2048

# Listas de parámetros a probar
SPATIAL_RADIUS_LIST=(3 5 10)
TEMPORAL_RADIUS_LIST=(0)
ALPHA_LIST=(0.005  0.05 0.1 0.2 0.5)


# Bucle por todas las combinaciones
for spatial in "${SPATIAL_RADIUS_LIST[@]}"; do
  for temporal in "${TEMPORAL_RADIUS_LIST[@]}"; do
    for alpha in "${ALPHA_LIST[@]}"; do

      echo "🧪 Ejecutando con spatial=$spatial, temporal=$temporal, alpha=$alpha"

      # Ejecutar el script Python
      python denoiserTransient.py "$ESCENA" "$SPP" "$spatial" "$temporal" "$alpha"
    done
  done
done

echo "✅ Banco de pruebas completado."
