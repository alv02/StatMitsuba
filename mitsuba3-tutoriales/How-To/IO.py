import mitsuba as mi
import matplotlib.pyplot as plt
import numpy as np

mi.set_variant('scalar_rgb')
bmp_exr = mi.Bitmap('../scenes/textures/multi_channels.exr')

print(bmp_exr)
# Here we convert the list of pairs into a dict for easier use
res = dict(bmp_exr.split())

# Plot the image, shading normal and depth buffer
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(res['image']);     axs[0].axis('off'); axs[0].set_title('image')
axs[1].imshow(res['sh_normal']); axs[1].axis('off'); axs[1].set_title('sh_normal')
axs[2].imshow(res['depth']);     axs[2].axis('off'); axs[2].set_title('depth')

plt.show()

data = np.zeros((64, 64, 5))

# .. process the data tensor ..

# Construct a bitmap object giving channel names
bmp_multi = mi.Bitmap(data, channel_names=['A.x', 'A.y', 'B.x', 'B.y', 'B.z'])

print(bmp_multi)

# Set a `cuda` variant
mi.set_variant('cuda_ad_rgb')

# Use the `box` reconstruction filter
scene_description = mi.cornell_box()
scene_description['sensor']['film']['rfilter']['type'] = 'box'
scene = mi.load_dict(scene_description)

noisy = mi.render(scene, spp=1)

# Denoise the rendered image
denoiser = mi.OptixDenoiser(input_size=noisy.shape[:2], albedo=False, normals=False, temporal=False)
denoised = denoiser(noisy)

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].imshow(mi.util.convert_to_bitmap(noisy));     axs[0].axis('off'); axs[0].set_title('noisy (1 spp)');
axs[1].imshow(mi.util.convert_to_bitmap(denoised));  axs[1].axis('off'); axs[1].set_title('denoised');