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
data_steady, data_transient, stats = mi.render(scene, spp=spp)
np.save(f"./io/transient/transient_data_{spp}.npy", data_transient)
np.save(f"./io/transient/transient_stats_{spp}.npy", stats)
