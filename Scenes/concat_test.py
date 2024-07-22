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

for i in range(2):
    for j in range(2):
        for k in range(2):
            model_names.append("square.obj")
            trans_list.append(np.array([1. * i, 1.0 * j, 1. * k]))
            scale_list.append(0.5)


# model_names.append("cylinder3.3k.obj")
# trans_list.append(np.array([0.0, 0.0, 0.0]))
# scale_list.append(1.0)
#
# model_names.append("sphere1K.obj")
# trans_list.append(np.array([0.0, 0.0, 0.0]))
# scale_list.append(3.0)
#
# model_names.append("sphere1K.obj")
# trans_list.append(np.array([0.0, -2.5, 0.0]))
# scale_list.append(3.0)
# #
# # model_names.append("sphere1K.obj")
# # trans_list.append(np.array([0.0, -2.0, 0.0]))
# # scale_list.append(2.0)
#
# model_names.append("sphere1K.obj")
# trans_list.append(np.array([1.0, 1.0, 0.0]))
# scale_list.append(1.5)
#
# model_names.append("sphere1K.obj")
# trans_list.append(np.array([1.6, 0.6, 0.0]))
# scale_list.append(1.3)
#
#
# model_names.append("sphere1K.obj")
# trans_list.append(np.array([-1.0, 1.0, 0.0]))
# scale_list.append(1.5)
#
# model_names.append("sphere1K.obj")
# trans_list.append(np.array([-1.6, 0.6, 0.0]))
# scale_list.append(1.3)


# for i in range(1):
#     for j in range(1):
#         for k in range(1):
#             model_names.append("plane_8.obj")
#             trans_list.append(np.array([1.0 * i, 2.0 * j, 1.0 * k]))
#             scale_list.append(5.0)


# model_names.append("BasicTShirt.obj")
# # model_names.append("JinhoOBJ.obj")
# trans_list.append(np.array([0.0, 0.0, 0.0]))
# trans_list.append(np.array([0.0, 0.0, 0.0]))
# scale_list.append(1.)
# scale_list.append(1.)

# model_names.append("square.obj")
# model_names.append("plane.obj")
# trans_list = []
# trans_list.append(np.array([0.0, 0.5, 0.0]))
# trans_list.append(np.array([10., 10.5, 10.5]))
# trans_list.append(np.array([0.0, 1.0, 0.0]))

# scale_list.append(1.)
# scale_list.append(1.)
# scale_list.append(5.)
concat_model_name = "concat.obj"
offsets = concat.concat_mesh(concat_model_name, model_dir, model_names, trans_list, scale_list)

#dynamic mesh
# mesh_dy = MeshTaichiWrapper("../models/concat.obj", offsets=offsets, scale=0.005, trans=ti.math.vec3(0, -6.0, 0), rot=ti.math.vec3(0.0, 0.0, 0.0))
mesh_dy = MeshTaichiWrapper("../models/OBJ/square.obj", offsets=[0], scale=1.0, trans=ti.math.vec3(0, 2.0, 0), rot=ti.math.vec3(0.0, 0.0, 0.0))
# mesh_dy = MeshTaichiWrapper("../models/OBJ/BasicTShirt.obj", offsets=[0], scale=0.001, trans=ti.math.vec3(0, -13.0, 0), rot=ti.math.vec3(0.0, 0.0, 0.0))
mesh_st = None

#static mesh
# mesh_st = MeshTaichiWrapper("../models/OBJ/square_big.obj",  offsets=[0], scale=1.5, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)
# mesh_st = MeshTaichiWrapper("../models/OBJ/APoseSMPL.obj",  offsets=[0], scale=4.0, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)
mesh_st = MeshTaichiWrapper("../models/concat.obj",  offsets=[0], scale=1.0, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)