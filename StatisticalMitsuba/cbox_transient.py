import os
import sys

import mitransient as mitr
import mitsuba as mi
import numpy as np

# Asegúrate de haber seteado la variante de Mitsuba que necesitas
mi.set_variant("llvm_ad_rgb")

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
