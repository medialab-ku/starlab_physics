import random
import numpy as np
import framework.concat as concat
from framework.meshTaichiWrapper import MeshTaichiWrapper
import taichi as ti

enable_profiler = False
ti.init(arch=ti.cuda, device_memory_GB=6, kernel_profiler=enable_profiler)

model_dir = "../models/OBJ/"
model_names = []
trans_list = []
scale_list = []

# for i in range(8):
#     for j in range(1):
#         for k in range(8):
#             model_names.append("square.obj")
#             trans_list.append(np.array([1.0 * i - 4, 1.0 * j, 1.0 * k - 4]))
#             scale_list.append(0.5)

# for i in range(1):
#     for j in range(1):
#         for k in range(1):
#             model_names.append("plane_3.obj")
#             trans_list.append(np.array([1.0 * i, 2.0 * j, 1.0 * k]))
#             scale_list.append(5.0)

model_names.append("plane.obj")
trans_list.append(np.array([0.0, 0.0, 0.0]))
scale_list.append(1.0)

# model_names.append("plane.obj")
# trans_list.append(np.array([0.0, 0.2, 0.0]))
# scale_list.append(0.8)

# model_names.append("square.obj")
# model_names.append("plane.obj")
# trans_list = []
# trans_list.append(np.array([0.0, 0.5, 0.0]))
# trans_list.append(np.array([10., 10.5, 10.5]))
# trans_list.append(np.array([0.0, 1.0, 0.0]))

# scale_list.append(1.)
# scale_list.append(1.)
# scale_list.append(5.)

concat_model_name = "plane_stack.obj"
offsets = concat.concat_mesh(concat_model_name, model_dir, model_names, trans_list, scale_list)
mesh_dy = MeshTaichiWrapper("../models/plane_stack.obj", offsets=offsets, scale=5.0, trans=ti.math.vec3(0, 4.5, 0), rot=ti.math.vec3(0.0, 0.0, 0.0))
mesh_st = None
mesh_st = MeshTaichiWrapper("../models/OBJ/plane.obj", offsets=[0], scale=6, trans=ti.math.vec3(0.0, -9.5, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)