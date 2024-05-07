import random
from Test.mesh import Mesh
from Test.particle import Particle
import taichi as ti
from TetMesh import TetMesh


ti.init(arch=ti.cuda, device_memory_GB=3)

meshes_dynamic = []
# mesh_dynamic_1 = Mesh("../models/OBJ/square_big.obj", scale=0.1, trans=ti.math.vec3(0.0, 1.0, 0.0), rot=ti.math.vec3(0.0, 0.0, 45.0))
# mesh_static_1 = Mesh("../models/OBJ/square_big.obj", scale=0.15, trans=ti.math.vec3(0.0, -0.6, 0.0), rot=ti.math.vec3(0.0, 10.0, 0.0), is_static=True)
# mesh_dynamic_2 = Mesh("../models/OBJ/square_huge.obj", scale=0.2, trans=ti.math.vec3(0.0, 1.3, 0.0), rot=ti.math.vec3(0.0, 0.0, 45.0))
# mesh_dynamic_2 = Mesh("../models/OBJ/square_huge.obj", scale=0.2, trans=ti.math.vec3(0.0, 1.3, 0.0), rot=ti.math.vec3(0.0, 0.0, 45.0))
mesh_dynamic_3 = Mesh("../models/OBJ/square_huge.obj", scale=0.2, trans=ti.math.vec3(0.0, 1.5, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0))
mesh_dynamic_4 = Mesh("../models/OBJ/square_huge.obj", scale=0.2, trans=ti.math.vec3(0.0, 1.5, 0.0), rot=ti.math.vec3(0.0, 0.0, 15.0))
mesh_dynamic_5 = Mesh("../models/OBJ/square_huge.obj", scale=0.2, trans=ti.math.vec3(0.0, 1.7, 0.0), rot=ti.math.vec3(0.0, 0.0, 30.0))
# mesh_dynamic_4 = Mesh("../models/OBJ/triangle.obj", scale=1.0, trans=ti.math.vec3(0.0, 1.5, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0))
# mesh_dynamic_5 = Mesh("../models/OBJ/triangle.obj", scale=1.0, trans=ti.math.vec3(0.0, 1.5, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0))


# meshes_dynamic.append(mesh_dynamic_1)
meshes_dynamic.append(mesh_dynamic_3)
# meshes_dynamic.append(mesh_dynamic_4)
# meshes_dynamic.append(mesh_dynamic_5)
# meshes.append(mesh_4)

tet_meshes_dynamic = []

# tet_mesh_dynamic_1 = TetMesh("../models/MESH/bunny_tiny.1.node",  scale=0.1, trans=ti.math.vec3(0.8, 0.2, 0.8), rot=ti.math.vec3(0.0, 0.0, 0.0))
# tet_mesh_dynamic_2 = TetMesh("../models/MESH/bunny_tiny.1.node", scale=0.1, trans=ti.math.vec3(-0.8, 0.2, -0.8), rot=ti.math.vec3(0.0, 0.0, 0.0))
# tet_mesh_dynamic_3 = TetMesh("../models/MESH/bunny_tiny.1.node", scale=0.1, trans=ti.math.vec3(0.0, 0.2, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0))
# tet_mesh_dynamic_4 = TetMesh("../models/MESH/dragon.1.1.node", scale=0.1, trans=ti.math.vec3(0.0, 0.2, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0))
# tet_meshes_dynamic.append(tet_mesh_dynamic_1)
# tet_meshes_dynamic.append(tet_mesh_dynamic_2)
# tet_meshes_dynamic.append(tet_mesh_dynamic_3)
# tet_meshes_dynamic.append(tet_mesh_dynamic_4)

meshes_static = []
mesh_static_1 = Mesh("../models/OBJ/sphere1K.obj", scale=2.5, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)
# mesh_static_2 = Mesh("../models/OBJ/square_big.obj", scale=0.15, trans=ti.math.vec3(0.0, -0.5, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)
# mesh_static_3 = Mesh("../models/OBJ/square_huge.obj", scale=0.25, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(0.0, 15.0, 0.0), is_static=True)
# mesh_static_4 = Mesh("../models/OBJ/triangle.obj", scale=0.0, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)

meshes_static.append(mesh_static_1)
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