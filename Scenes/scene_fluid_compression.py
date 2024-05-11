import random
from framework.particle import Particle
from framework.mesh import Mesh
import taichi as ti
from framework.TetMesh import TetMesh


# ti.init(arch=ti.cuda, device_memory_GB=3, kernel_profiler=True)
ti.init(arch=ti.cuda, device_memory_GB=11)

gravity = ti.math.vec3(0.0, -9.8, 0.0)
dt = 0.03
grid_size = ti.math.vec3(7, 7, 7)
particle_radius = 0.02
dHat = 6e-3

meshes_dynamic = []

tet_meshes_dynamic = []
tet_mesh_dynamic_1 = TetMesh("../models/MESH/cube.1.node",  scale=1.6, trans=ti.math.vec3(0.0, 2.5, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0))
# tet_meshes_dynamic.append(tet_mesh_dynamic_1)


meshes_static = []
mesh_static_1 = Mesh("../models/OBJ/hollow_box.obj", scale=5.0, trans=ti.math.vec3(0.0, -0.5, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)
# mesh_static_2 = Mesh("../models/OBJ/hollow_box2.obj", scale=2.0, trans=ti.math.vec3(0.0, -1.5, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)
# mesh_static_3 = Mesh("../models/OBJ/hollow_box3.obj", scale=2.0, trans=ti.math.vec3(0.0, -1.5, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)
meshes_static.append(mesh_static_1)

particles = []
particle_2 = Particle('../models/VTK/stackedcube.vtk', trans=ti.math.vec3(0.0, 0.0, 0), scale=1, radius=0.005)
particles.append(particle_2)


colors_tet_dynamic = []
for tid in range(len(tet_meshes_dynamic)):
    color = (random.randrange(0, 255) / 256, random.randrange(0, 255) / 256, random.randrange(0, 255) / 256)
    colors_tet_dynamic.append(color)


colors_tri_dynamic = []
for mid in range(len(meshes_dynamic)):
    color = (random.randrange(0, 255) / 256, random.randrange(0, 255) / 256, random.randrange(0, 255) / 256)
    colors_tri_dynamic.append(color)