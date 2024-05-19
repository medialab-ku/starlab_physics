import random
from framework.mesh import Mesh
import taichi as ti

enable_profiler = False
ti.init(arch=ti.cuda, device_memory_GB=3, kernel_profiler=enable_profiler)

meshes_dynamic = []
mesh_dynamic_1 = Mesh("../models/OBJ/plane.obj", scale=5.0, trans=ti.math.vec3(0.0, 4.0, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0))
# mesh_dynamic_2 = Mesh("../models/OBJ/square_big.obj", scale=0.3, trans=ti.math.vec3(0.0, 5.0, 0.0), rot=ti.math.vec3(0.0, 0.0, 40.0))
# mesh_dynamic_3 = Mesh("../models/OBJ/square_huge.obj", scale=0.25, trans=ti.math.vec3(0.0, 0.6, 0.0), rot=ti.math.vec3(0.0, 0.0, 60.0))


meshes_dynamic.append(mesh_dynamic_1)
# meshes_dynamic.append(mesh_dynamic_2)
# meshes_dynamic.append(mesh_dynamic_3)

tet_meshes_dynamic = []

meshes_static = []
mesh_static_1 = Mesh("../models/OBJ/circle3K.obj", scale=5, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)
meshes_static.append(mesh_static_1)

particles = []

colors_tet_dynamic = []

for tid in range(len(tet_meshes_dynamic)):
    color = (random.randrange(0, 255) / 256, random.randrange(0, 255) / 256, random.randrange(0, 255) / 256)
    colors_tet_dynamic.append(color)


colors_tri_dynamic = []
for mid in range(len(meshes_dynamic)):
    color = (random.randrange(0, 255) / 256, random.randrange(0, 255) / 256, random.randrange(0, 255) / 256)
    colors_tri_dynamic.append(color)