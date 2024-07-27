import random
from framework.meshio.meshTaichiWrapper import MeshTaichiWrapper
import taichi as ti
from framework.meshio.TetMesh import TetMesh


ti.init(arch=ti.cuda, device_memory_GB=6)

enable_profiler = False
gravity = ti.math.vec3(0.0, 0.0, 0.0)
dt = 0.03
grid_size = ti.math.vec3(3.5, 3.5, 3.5)
particle_radius = 0.02
dHat = 1e-3

meshes_dynamic = []

tet_meshes_dynamic = []
tet_mesh_dynamic_1 = TetMesh("../models/MESH/torus3k.1.node", scale=2.0, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(0, 90, 90), is_static=True) # 10k element
# tet_mesh_dynamic_2 = TetMesh("../models/MESH/torus3k.1.node", scale=2.0, trans=ti.math.vec3(0.0, -6.0, 0.0), rot=ti.math.vec3(0,0,0), is_static=True) # 10k element
# tet_mesh_dynamic_3 = TetMesh("../models/MESH/torus3k.1.node", scale=2.0, trans=ti.math.vec3(0.0, -9.0, 0.0), rot=ti.math.vec3(0,0,90), is_static=True) # 10k element
tet_meshes_dynamic.append(tet_mesh_dynamic_1)
# tet_meshes_dynamic.append(tet_mesh_dynamic_2)
# tet_meshes_dynamic.append(tet_mesh_dynamic_3)


meshes_static = []
mesh_static_1 = MeshTaichiWrapper("../models/OBJ/torus3K.obj", scale=2.0, trans=ti.math.vec3(3.0, 0.0, 0.0), rot=ti.math.vec3(90.0, 0.0, 90.0), is_static=True)
mesh_static_2 = MeshTaichiWrapper("../models/OBJ/torus3K.obj", scale=2.0, trans=ti.math.vec3(-3.0, 0.0, 0.0), rot=ti.math.vec3(90.0, 0.0, 90.0), is_static=True)
# mesh_static_3 = Mesh("../models/OBJ/square_huge.obj", scale=0.25, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(0.0, 15.0, 0.0), is_static=True)
# mesh_static_4 = Mesh("../models/OBJ/tet.obj", scale=0.0, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)

meshes_static.append(mesh_static_1)
meshes_static.append(mesh_static_2)
# meshes_static.append(mesh_static_3)
# meshes_static.append(mesh_static_4)
# meshes_static.append(mesh_static_2)

particles = []
# particle_1 = Particle('../models/VTK/cube87K.vtk', trans=ti.math.vec3(0.0, 1.0, 0.0), scale=1.2, radius=0.01)
# particle_2 = Particle('../models/VTK/cube1K.vtk', trans=ti.math.vec3(-0.8, 2.0, 0.8), scale=1.2, radius=0.01)
# particle_2 = Particle('../models/VTK/bunny.vtk', trans=ti.math.vec3(1.0, 0.0, 0.0), radius=0.01)


# particles.append(particle_1)
# particles.append(particle_2)

colors_tet_dynamic = []

for tid in range(len(tet_meshes_dynamic)):
    color = (random.randrange(0, 255) / 256, random.randrange(0, 255) / 256, random.randrange(0, 255) / 256)
    colors_tet_dynamic.append(color)


colors_tri_dynamic = []
for mid in range(len(meshes_dynamic)):
    color = (random.randrange(0, 255) / 256, random.randrange(0, 255) / 256, random.randrange(0, 255) / 256)
    colors_tri_dynamic.append(color)