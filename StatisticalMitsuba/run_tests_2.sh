#!/bin/bash

# Lista de spp
spp_values=(10000 100000 1000000 10000000)

for spp in "${spp_values[@]}"
do
    echo "Renderizando con spp = $spp"
    python3 cbox_transient.py $spp
done
