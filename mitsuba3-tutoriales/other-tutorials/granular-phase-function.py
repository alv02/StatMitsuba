import drjit as dr
import mitsuba as mi

mi.set_variant('llvm_ad_rgb')


scene = mi.load_dict({
    'type' : 'scene',
    'grain' : {
        'type' : 'sphere',
    }
})

bsdf = mi.load_dict({
    'type' : 'dielectric',
    'int_ior' : 'water', # We can also use float values
    'ext_ior' : 'air',
})

sampler = mi.load_dict({'type' : 'independent'})
sampler.seed(0, wavefront_size=int(1e7))

# Sample ray directions
d = mi.warp.square_to_uniform_sphere(sampler.next_2d())

# Construct coordinate frame object
frame = mi.Frame3f(d)

# Sample ray origins
xy_local = 2.0 * sampler.next_2d() - 1.0
local_o = mi.Vector3f(xy_local.x, xy_local.y, -1.0)
world_o = frame.to_world(local_o)

# Move ray origin according to scene bounding sphere
bsphere = scene.bbox().bounding_sphere()
o = world_o * bsphere.radius + bsphere.center

# Construct rays
rays = mi.Ray3f(o, d)

@dr.syntax
def sample(rays: mi.Ray3f, sampler: mi.Sampler):

    # Find first ray intersection with the object
    si = scene.ray_intersect(rays)
    valid = si.is_valid()

    # Maximum number of bounces
    max_bounces = 10

    # Loop state variables
    throughput = mi.Spectrum(1.0)
    active = mi.Bool(valid)
    i = mi.UInt32(0)

    while active & (i < max_bounces):
        # Sample new direction
        ctx = mi.BSDFContext()
        bs, bsdf_val = bsdf.sample(ctx, si, sampler.next_1d(), sampler.next_2d(), active)

        # Update throughput and rays for next bounce
        throughput[active] *= bsdf_val
        rays[active] = si.spawn_ray(si.to_world(bs.wo))

        # Find next intersection
        si = scene.ray_intersect(rays, active)
        active &= si.is_valid()

        # Increase loop iteration counter
        i += 1

    # Only account for rays that have escaped
    valid &= ~active

    # We don't care about a specific color for this tutorial
    return rays, mi.luminance(throughput), valid

rays, throughput, valid = sample(rays, sampler)

# Resolution of the histogram
histogram_size = 512

# Convert escaping directions into histogram bin indices
cos_theta = mi.Frame3f.cos_theta(frame.to_local(rays.d))
theta = dr.acos(cos_theta)
theta_idx = mi.UInt32(theta / dr.pi * histogram_size)

# Account for projection jacobian
throughput *= 1.0 / dr.sqrt(1 - cos_theta**2)

# Accumulate values into the histogram
histogram = dr.zeros(mi.Float, histogram_size)
dr.scatter_reduce(dr.ReduceOp.Add, histogram, throughput, theta_idx, valid)

# Execute the kernel by evaluating the histogram
dr.eval(histogram)

from matplotlib import pyplot as plt
import numpy as np

x = dr.linspace(mi.Float, 0, dr.pi, len(histogram), endpoint=True)
histogram_log = np.array(dr.log(histogram))

fig = plt.figure(figsize = (10, 5))

ax = fig.add_subplot(121, title='Angular distribution')
ax.plot(np.array(dr.cos(x)), histogram_log, color='C0')
ax.set_xlabel(r'$\cos(\theta)$', size=15)
ax.set_ylabel(r'$p(\cos(\theta))$', size=15)

ax = fig.add_subplot(122, polar=True, title='polar plot')
ax.plot(np.array(x), histogram_log, color='C0')
ax.plot(np.array(-x), histogram_log, color='C0', label='test')
plt.show()