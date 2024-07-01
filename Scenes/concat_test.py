import random
import numpy as np
import framework.concat as concat
from framework.meshTaichiWrapper import MeshTaichiWrapper
import taichi as ti

enable_profiler = True
ti.init(arch=ti.cuda, device_memory_GB=6, kernel_profiler=enable_profiler)

model_dir = "../models/OBJ/"
model_names = []
trans_list = []
scale_list = []

for i in range(64):
    for j in range(1):
        for k in range(64):
            model_names.append("square.obj")
            trans_list.append(np.array([1.0 * i - 32, 1.0 * j, 1.0 * k - 32]))
            scale_list.append(0.5)

# for i in range(1):
#     for j in range(1):
#         for k in range(1):
#             model_names.append("plane_small.obj")
#             trans_list.append(np.array([1.0 * i, 2.0 * j, 1.0 * k]))
#             scale_list.append(0.5)

# model_names.append("square.obj")
# model_names.append("square.obj")
# model_names.append("plane.obj")
# trans_list = []
# trans_list.append(np.array([0.0, 0.5, 0.0]))
# trans_list.append(np.array([10., 10.5, 10.5]))
# trans_list.append(np.array([0.0, 1.0, 0.0]))

# scale_list.append(1.)
# scale_list.append(1.)
# scale_list.append(5.)

concat.concat_mesh(model_dir, model_names, trans_list, scale_list)

mesh_dy = MeshTaichiWrapper("../models/OBJ/plane.obj", scale=30.0, trans=ti.math.vec3(0, 10.0, 0), rot=ti.math.vec3(0.0, 0.0, 0.0))
mesh_st = MeshTaichiWrapper("../models/concat.obj", scale=1, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)