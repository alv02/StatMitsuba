from typing import cast

import matplotlib
import mitsuba as mi

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


mi.set_variant("llvm_ad_rgb")

scene = cast(mi.Scene, mi.load_file("../scenes/cbox.xml"))

image = mi.render(scene, spp=256)
mi.util.write_bitmap("my_first_render.exr", image)