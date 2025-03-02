import mitsuba as mi

mi.set_variant("scalar_rgb")

mi.Frame3f()  # Empty frame

mi.Frame3f(
    [1, 0, 0],  # s
    [0, 1, 0],  # t
    [0, 0, 1],  # n
)

mi.Frame3f([0, 1, 0])  # n

frame = mi.Frame3f(
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
)

world_vector = mi.Vector3f([3, 2, 1])  # In world frame
local_vector = frame.to_local(world_vector)
print(local_vector)

import numpy as np

# Default constructor is identity matrix
identity = mi.Transform4f()

np_mat = np.array(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
)
mi_mat = mi.Matrix3f(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]
)
list_mat = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
]

# Build from different types
t_from_np = mi.Transform3f(np_mat)
t_from_mi = mi.Transform3f(mi_mat)
t_from_list = mi.Transform3f(list_mat)

# Broadcasting
t_from_value = mi.Transform3f(3)  # Scaled identity matrix
t_from_row = mi.Transform3f([3, 2, 3])  # Broadcast over matrix columns
print(t_from_row)

t = mi.Transform4f().translate([0, 1, 2])
t = t @ mi.Transform4f().scale([1, 2, 3])
v = mi.Vector3f([3, 4, 5])
p = mi.Point3f([3, 4, 5])
n = mi.Normal3f([1, 0, 0])

print(f"{t @ v=}")
print(f"{t @ p=}")
print(f"{t @ n=}")