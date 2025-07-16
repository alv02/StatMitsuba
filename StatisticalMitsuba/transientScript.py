import os
import sys

import mitsuba as mi

mi.set_variant("cuda_ad_rgb")
import mitransient as mitr
import numpy as np

# Comprobar que se han pasado los parámetros spp, escena, output_dir, transform y lambda
if len(sys.argv) != 6:
    print(
        "Uso: python transient.py <spp> <ruta_escena.xml> <output_dir> <transform: bc|yj> <lambda>"
    )
    sys.exit(1)

# Obtener los parámetros desde los argumentos
spp = int(sys.argv[1])
scene_path = sys.argv[2]
output_dir = sys.argv[3]
transform = sys.argv[4]
lambda_val = sys.argv[5]

# Validar transformación
if transform not in ("bc", "yj"):
    print("Transformación no válida. Usa 'bc' para Box-Cox o 'yj' para Yeo-Johnson.")
    sys.exit(1)

# Cargar escena
scene = mi.load_file(scene_path)

# Renderizar
data_steady, data_transient, stats = mi.render(scene, spp=spp)

# Crear directorios de salida si no existen
os.makedirs(output_dir, exist_ok=True)
stats_dir = os.path.join(output_dir, "transient_stats", transform)
os.makedirs(stats_dir, exist_ok=True)

# Guardar resultados
np.save(f"{output_dir}/transient_data_{spp}.npy", data_transient)
np.save(f"{stats_dir}/estimands_{lambda_val}_{spp}.npy", stats[0])
np.save(f"{stats_dir}/estimands_variance_{lambda_val}_{spp}.npy", stats[1])
