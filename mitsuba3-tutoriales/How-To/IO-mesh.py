import mitsuba as mi
import drjit as dr

mi.set_variant('llvm_ad_rgb')

bunny = mi.load_dict({
    "type": "ply",
    "filename": "../scenes/meshes/bunny.ply",
    "face_normals": False,
    "to_world": mi.ScalarTransform4f().rotate([0, 0, 1], angle=10),
})

print(bunny)

# Wavy disk construction
#
# Let N define the total number of vertices, the first N-1 vertices will compose
# the fringe of the disk, while the last vertex should be placed at the center.
# The first N-1 vertices must have their height modified such that they oscillate
# with some given frequency and amplitude. To compute the face indices, we define
# the first vertex of every face to be the vertex at the center (idx=N-1) and the
# other two can be assigned sequentially (modulo N-2).

# Disk with a wavy fringe parameters
N = 100
frequency = 12.0
amplitude = 0.4

# Generate the vertex positions
theta = dr.linspace(mi.Float, 0.0, dr.two_pi, N)
x, y = dr.sincos(theta)
z = amplitude * dr.sin(theta * frequency)
vertex_pos = mi.Point3f(x, y, z)

# Move the last vertex to the center
vertex_pos[dr.arange(mi.UInt32, N) == N - 1] = 0.0

# Generate the face indices
idx = dr.arange(mi.UInt32, N - 1)
face_indices = mi.Vector3u(N - 1, (idx + 1) % (N - 2), idx % (N - 2))

# Create an empty mesh (allocates buffers of the correct size)
mesh = mi.Mesh(
    "wavydisk",
    vertex_count=N,
    face_count=N - 1,
    has_vertex_normals=False,
    has_vertex_texcoords=False,
)

mesh_params = mi.traverse(mesh)
mesh_params["vertex_positions"] = dr.ravel(vertex_pos)
mesh_params["faces"] = dr.ravel(face_indices)
print(mesh_params.update())

scene = mi.load_dict({
    "type": "scene",
    "integrator": {"type": "path"},
    "light": {"type": "constant"},
    "sensor": {
        "type": "perspective",
        "to_world": mi.ScalarTransform4f().look_at(
            origin=[0, -5, 5], target=[0, 0, 0], up=[0, 0, 1]
        ),
    },
    "wavydisk": mesh,
})

img = mi.render(scene)

from matplotlib import pyplot as plt

plt.axis("off")
plt.imshow(mi.util.convert_to_bitmap(img));
plt.show()
mesh.write_ply("wavydisk.ply")

mesh = mi.load_dict({
    "type": "ply",
    "filename": "wavydisk.ply",
    "bsdf": {
        "type": "diffuse",
        "reflectance": {
            "type": "mesh_attribute",
            "name": "vertex_color",  # This will be used to visualize our attribute
        },
    },
})

# Needs to start with vertex_ or face_
attribute_size = mesh.vertex_count() * 3
mesh.add_attribute(
    "vertex_color", 3, [0] * attribute_size
)  # Add 3 floats per vertex (initialized at 0)

mesh_params = mi.traverse(mesh)
print(mesh_params)

N = mesh.vertex_count()

vertex_colors = dr.zeros(mi.Float, 3 * N)
fringe_vertex_indices = dr.arange(mi.UInt, N - 1)
dr.scatter(vertex_colors, 1, fringe_vertex_indices * 3)  # Fringe is red
dr.scatter(vertex_colors, 1, [(N - 1) * 3 + 2])  # Center is blue

mesh_params["vertex_color"] = vertex_colors
mesh_params.update()

scene = mi.load_dict(
    {
        "type": "scene",
        "integrator": {"type": "path"},
        "light": {"type": "constant"},
        "sensor": {
            "type": "perspective",
            "to_world": mi.ScalarTransform4f().look_at(
                origin=[0, -5, 5], target=[0, 0, 0], up=[0, 0, 1]
            ),
        },
        "wavydisk": mesh,
    }
)

img = mi.render(scene)

plt.axis("off")
plt.imshow(mi.util.convert_to_bitmap(img))
plt.show()