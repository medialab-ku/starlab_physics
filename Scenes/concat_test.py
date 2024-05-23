import random
import numpy as np
import framework.concat as concat
from framework.mesh import Mesh
import taichi as ti

enable_profiler = True
ti.init(arch=ti.cuda, device_memory_GB=3, kernel_profiler=enable_profiler)

model_dir = "../models/OBJ/"
model_names = []
model_names.append("plane.obj")
# model_names.append("plane.obj")
# model_names.append("plane.obj")
# model_names.append("plane.obj")
# model_names.append("plane.obj")

trans_list = []
trans_list.append(np.array([0.0, 1.0, 0.0]))
# trans_list.append(np.array([0.0, 1.3, 0.0]))
# trans_list.append(np.array([0.0, 1.6, 0.0]))
# trans_list.append(np.array([0.0, 1.9, 0.0]))
# trans_list.append(np.array([0.0, 2.2, 0.0]))

scale_list = []
scale_list.append(5.0)
# scale_list.append(5.0)
# scale_list.append(5.0)
# scale_list.append(5.0)
# scale_list.append(5.0)

concat.concat_mesh(model_dir, model_names, trans_list, scale_list)

# meshes_dynamic = []
# mesh_dy = Mesh("../models/concat.obj", scale=1, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0))
mesh_dy = Mesh("../models/OBJ/plane.obj", scale=1, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0))
# meshes_dynamic.append(mesh_dynamic_1)


# mesh_test = mesh_dynamic_1

tet_meshes_dynamic = []

# meshes_static = []
# mesh_st = Mesh("../models/OBJ/circle3K.obj", scale=6, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)
# meshes_static.append(mesh_static_1)

particles = []

# colors_tet_dynamic = []
#
# for tid in range(len(tet_meshes_dynamic)):
#     color = (random.randrange(0, 255) / 256, random.randrange(0, 255) / 256, random.randrange(0, 255) / 256)
#     colors_tet_dynamic.append(color)
#
#
# colors_tri_dynamic = []
# for mid in range(len(meshes_dynamic)):
#     color = (random.randrange(0, 255) / 256, random.randrange(0, 255) / 256, random.randrange(0, 255) / 256)
#     colors_tri_dynamic.append(color)