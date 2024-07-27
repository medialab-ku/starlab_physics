import numpy as np
from framework.meshio.meshTaichiWrapper import MeshTaichiWrapper
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
            model_names.append("sphere1K.obj")
            trans_list.append(np.array([0. * i, 0.0 * j, 0. * k]))
            scale_list.append(3.0)

# concat_model_name = "concat.obj"
# offsets = concat.concat_mesh(concat_model_name, model_dir, model_names, trans_list, scale_list)

#dynamic mesh
mesh_dy = MeshTaichiWrapper("../models/OBJ/poncho_8K.obj", offsets=[0], scale=3.4, trans=ti.math.vec3(0, 5.0, 0), rot=ti.math.vec3(0.0, 0.0, 0.0))

#static mesh
mesh_st = MeshTaichiWrapper("../models/OBJ/APoseSMPL.obj",  offsets=[0], scale=12.0, trans=ti.math.vec3(0.0, 0.0, 0.01), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)
