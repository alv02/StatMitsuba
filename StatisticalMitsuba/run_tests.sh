#!/bin/bash

# Array con los valores de spp
spps=(128 256 512 1024)

# Ejecutar el script Python para cada spp
for spp in "${spps[@]}"; do
    python render_transient.py "$spp"
done
