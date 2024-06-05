import random
from framework.meshtaichiwrapper import MeshTaichiWrapper
import taichi as ti
from framework.TetMesh import TetMesh

import os


ti.init(arch=ti.cuda, device_memory_GB=3)

gravity = ti.math.vec3(0.0, 0.0, 0.0)
dt = 0.03
grid_size = ti.math.vec3(3.5, 3.5, 3.5)
particle_radius = 0.02
dHat = 1e-3


scalef = 3.0

meshes_dynamic = []
mesh_dynamic_1 = MeshTaichiWrapper("../models/OBJ/clubbing_dress.obj", scalef=3.0, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(90.0, 0.0, 0.0))

meshes_dynamic.append(mesh_dynamic_1)

tet_meshes_dynamic = []

meshes_static = []
mesh_static_1 = MeshTaichiWrapper("../models/Arabesque/Kyra_DVStandClubbing_0000.obj", scalef=3.0, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(90, 0, 0), is_static=True)
meshes_static.append(mesh_static_1)


mesh_seq = []
for mid in range(350) :
    mm = MeshTaichiWrapper("../models/Arabesque/Kyra_DVStandClubbing_" + (str(mid)).zfill(4) + ".obj", scalef=3.0, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(90, 0, 0), is_static=True)
    mesh_seq.append(mm)



for mesh_idx in range(11 + 1) :
    pass

particles = []



colors_tet_dynamic = []

for tid in range(len(tet_meshes_dynamic)):
    color = (random.randrange(0, 255) / 256, random.randrange(0, 255) / 256, random.randrange(0, 255) / 256)
    colors_tet_dynamic.append(color)


colors_tri_dynamic = []
for mid in range(len(meshes_dynamic)):
    color = (random.randrange(0, 255) / 256, random.randrange(0, 255) / 256, random.randrange(0, 255) / 256)
    colors_tri_dynamic.append(color)