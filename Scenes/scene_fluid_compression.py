import random
from framework.particle import Particle
from framework.meshtaichiwrapper import MeshTaichiWrapper
import taichi as ti
from framework.TetMesh import TetMesh


# ti.init(arch=ti.cuda, device_memory_GB=3, kernel_profiler=True)
ti.init(arch=ti.cuda, device_memory_GB=9)

gravity = ti.math.vec3(0.0, -9.8, 0.0)
dt = 0.03
grid_size = ti.math.vec3(7, 7, 7)
particle_radius = 0.02
dHat = 6e-3

meshes_dynamic = []

tet_meshes_dynamic = []
tet_mesh_dynamic_1 = TetMesh("../models/MESH/cube.1.node",  scale=1.0, trans=ti.math.vec3(5, 5,5 ), rot=ti.math.vec3(0.0, 0.0, 0.0))
tet_meshes_dynamic.append(tet_mesh_dynamic_1)


meshes_static = []
mesh_static_1 = MeshTaichiWrapper("../models/OBJ/cuboid112.obj", scale=5.0, trans=ti.math.vec3(0.0, -0.5, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)         # 1. blender 기준 x,y,z (1,1,2) 정도로 하고 여기처럼 scale 5로 하면 이정도.
# mesh_static_2 = Mesh("../models/OBJ/hollow_box2.obj", scale=2.0, trans=ti.math.vec3(0.0, -1.5, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)
# mesh_static_3 = Mesh("../models/OBJ/hollow_box3.obj", scale=2.0, trans=ti.math.vec3(0.0, -1.5, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)
meshes_static.append(mesh_static_1)


particles = []
particle_2 = Particle('../models/VTK/cuboid224_170K.vtk', trans=ti.math.vec3(0.0, 0.0, 0), scale=1.5, radius=0.005)                                            # 2. blender 기준 x,y,z (2m,2m,4m) 직육면체인데. vtk 돌리면 170K정도. 그리고 크기도 잘 봐야됨
particles.append(particle_2)


colors_tet_dynamic = []
for tid in range(len(tet_meshes_dynamic)):
    color = (random.randrange(0, 255) / 256, random.randrange(0, 255) / 256, random.randrange(0, 255) / 256)
    colors_tet_dynamic.append(color)


colors_tri_dynamic = []
for mid in range(len(meshes_dynamic)):
    color = (random.randrange(0, 255) / 256, random.randrange(0, 255) / 256, random.randrange(0, 255) / 256)
    colors_tri_dynamic.append(color)