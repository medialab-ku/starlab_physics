import random
from framework.meshio.meshTaichiWrapper import MeshTaichiWrapper
import taichi as ti

ti.init(arch=ti.cuda, device_memory_GB=3,  kernel_profiler=True)

meshes_dynamic = []
mesh_dynamic_1 = MeshTaichiWrapper("../models/OBJ/simpleCube.obj", scale=1.0, trans=ti.math.vec3(0.0, 1.0, 0.0), rot=ti.math.vec3(0.0, 0.0, 45.0))
mesh_dynamic_2 = MeshTaichiWrapper("../models/OBJ/simpleCube.obj", scale=1.0, trans=ti.math.vec3(0.0, 4.0, 0.0), rot=ti.math.vec3(0.0, .0, 0.0))
mesh_dynamic_3 = MeshTaichiWrapper("../models/OBJ/simpleCube.obj", scale=1.0, trans=ti.math.vec3(0.0, 7.0, 0.0), rot=ti.math.vec3(0.0, .0, 45.0))


meshes_dynamic.append(mesh_dynamic_1)
meshes_dynamic.append(mesh_dynamic_2)
meshes_dynamic.append(mesh_dynamic_3)

tet_meshes_dynamic = []

meshes_static = []
mesh_static_1 = MeshTaichiWrapper("../models/OBJ/simpleCube.obj", scale=1.0, trans=ti.math.vec3(0.0, -2.0, 0.0), rot=ti.math.vec3(0.0, 90.0, 0.0))
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