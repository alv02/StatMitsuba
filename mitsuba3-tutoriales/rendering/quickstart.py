import mitsuba as mi
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt




mi.set_variant('llvm_ad_rgb')

scene = mi.load_file('../scenes/cbox.xml')

image = mi.render(scene, spp=256)

plt.axis('off')
plt.imshow(image ** (1.0/2.2))
plt.show()