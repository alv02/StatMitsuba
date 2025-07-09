import os
import sys

import mitsuba as mi

mi.set_variant("cuda_ad_rgb")
import mitransient as mitr
import numpy as np

# Comprobar que se han pasado los parámetros spp, escena, lambda_bc y output_dir
if len(sys.argv) != 4:
    print("Uso: python render_transient.py <spp> <ruta_escena.xml> <output_dir>")
    sys.exit(1)

# Obtener los parámetros desde los argumentos
spp = int(sys.argv[1])
scene_path = sys.argv[2]
output_dir = sys.argv[3]

# Cargar escena
scene = mi.load_file(scene_path)

# Renderizar con spp y lambda_bc especificados
data_steady, data_transient, stats = mi.render(scene, spp=spp)

# Crear directorio de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Guardar los resultados
np.save(f"{output_dir}/transient_data_{spp}.npy", data_transient)
np.save(f"{output_dir}/transient_stats_{spp}.npy", stats)
