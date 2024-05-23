import random
from framework.mesh import Mesh
import taichi as ti

enable_profiler = False
ti.init(arch=ti.cuda, device_memory_GB=3, kernel_profiler=enable_profiler)

meshes_dynamic = []
mesh_dynamic_1 = Mesh("../models/OBJ/square_big.obj", scale=0.2, trans=ti.math.vec3(0.0, 1.0, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0))
mesh_dynamic_2 = Mesh("../models/OBJ/square_big.obj", scale=0.2, trans=ti.math.vec3(0.0, 1.3, 0.0), rot=ti.math.vec3(0.0, 0.0, 20.0))
mesh_dynamic_3 = Mesh("../models/OBJ/square_big.obj", scale=0.2, trans=ti.math.vec3(0.0, 1.6, 0.0), rot=ti.math.vec3(0.0, 0.0, 40.0))

meshes_dynamic.append(mesh_dynamic_1)
meshes_dynamic.append(mesh_dynamic_2)
meshes_dynamic.append(mesh_dynamic_3)
# meshes_dynamic.append(mesh_dynamic_4)
# meshes_dynamic.append(mesh_dynamic_5)

tet_meshes_dynamic = []

meshes_static = []
mesh_static_1 = Mesh("../models/OBJ/circle3K.obj", scale=6, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)
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