import random
from framework.deprecated.particle import Particle
from framework.meshTaichiWrapper import MeshTaichiWrapper
import taichi as ti


ti.init(arch=ti.cuda, device_memory_GB=3, kernel_profiler=True)

meshes_dynamic = []


tet_meshes_dynamic = []


meshes_static = []
mesh_static_1 = MeshTaichiWrapper("../models/OBJ/sphere1K.obj", scale=7, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)
# mesh_static_2 = Mesh("../models/OBJ/square_big.obj", scale=0.15, trans=ti.math.vec3(0.0, -0.5, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)
# mesh_static_3 = Mesh("../models/OBJ/square_huge.obj", scale=0.25, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(0.0, 15.0, 0.0), is_static=True)
# mesh_static_4 = Mesh("../models/OBJ/tet.obj", scale=0.0, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)

meshes_static.append(mesh_static_1)

particles = []
# particle_1 = Particle('../models/VTK/cube87K.vtk', trans=ti.math.vec3(0.0, 1.0, 0.0), scale=1.2, radius=0.01)
particle_2 = Particle('../models/VTK/cube87K.vtk', trans=ti.math.vec3(0.0, 0.0, 0), scale=1.8, radius=0.005)
# particle_3 = Particle('../models/VTK/cube1K.vtk', trans=ti.math.vec3(0.0, 2.0, 0), scale=4, radius=0.01)


# particles.append(particle_1)
particles.append(particle_2)
# particles.append(particle_3)

colors_tet_dynamic = []
for tid in range(len(tet_meshes_dynamic)):
    color = (random.randrange(0, 255) / 256, random.randrange(0, 255) / 256, random.randrange(0, 255) / 256)
    colors_tet_dynamic.append(color)


colors_tri_dynamic = []
for mid in range(len(meshes_dynamic)):
    color = (random.randrange(0, 255) / 256, random.randrange(0, 255) / 256, random.randrange(0, 255) / 256)
    colors_tri_dynamic.append(color)