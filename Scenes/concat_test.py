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

trans_list = []
trans_list.append(np.array([0.0, 0.5, 0.0]))
# trans_list.append(np.array([0.0, 1.0, 0.0]))

scale_list = []
scale_list.append(5.)
# scale_list.append(5.)

concat.concat_mesh(model_dir, model_names, trans_list, scale_list)

mesh_dy = Mesh("../models/concat.obj", scale=1, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0))
mesh_st = Mesh("../models/OBJ/circle3K.obj", scale=5.0, trans=ti.math.vec3(0.0, -1.0, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)
