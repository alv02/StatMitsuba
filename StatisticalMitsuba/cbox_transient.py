import os
import sys

import mitsuba as mi

mi.set_variant("cuda_ad_rgb")
import mitransient as mitr
import numpy as np

# Comprobar que se ha pasado el parámetro spp
if len(sys.argv) != 2:
    print("Uso: python render_transient.py <spp>")
    sys.exit(1)

# Obtener el spp desde los argumentos
spp = int(sys.argv[1])

# Cargar escena
scene = mi.load_file("../scenes/transient/cornell-box/cbox_diffuse.xml")

# Renderizar con spp especificado
data_steady, data_transient = mi.render(scene, spp=spp)

# Guardar con spp en el nombre del archivo
output_path = f"./io/transient/transient_spp{spp}.npy"
np.save(output_path, data_transient)

print(f"Guardado resultado en {output_path}")
