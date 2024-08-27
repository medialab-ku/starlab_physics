from framework.meshio.TriMesh import TriMesh
from framework.meshio.concat import concat_mesh
import taichi as ti
import numpy as np
from pathlib import Path

enable_profiler = False
ti.init(arch=ti.cuda, device_memory_GB=8, default_fp=ti.f64, kernel_profiler=enable_profiler)

model_path = Path(__file__).resolve().parent.parent / "models"
OBJ = "OBJ"
model_dir = str(model_path) + "/OBJ"
# print(model_dir)

# model_names = []
# trans_list = []
# scale_list = []
#
# concat_model_name = "concat.obj"
# #
# model_names.append("poncho_8K.obj")
# trans_list.append([0.0, 7.0, 0.0])
# scale_list.append(3.4)
# # #
# model_names.append("poncho_8K.obj")
# trans_list.append([0.0, 7.5, 0.0])
# scale_list.append(3.0)
#
# model_names.append("poncho_8K.obj")
# trans_list.append([0.0, 8.0, 0.0])
# scale_list.append(3.0)

# append more meshes
# model_names.append("your-obj-name.obj")
# trans_list.append([x, y, z])
# scale_list.append(size)

mesh_dy = TriMesh(
    model_dir,
    model_name_list=["triangle.obj"],
    trans_list=[(0.0, 5.0, 0.0)],
    scale_list=[10.0],
    rot_list=[], # (axis.x, axis.y, axis.z, radian)
    is_static=False)

# mesh_st = None
mesh_st = TriMesh(
    model_dir,
    model_name_list=["square.obj"],
    trans_list=[(0.0, -20.0, 0.0)],
    scale_list=[1.5],
    rot_list=[],
    is_static=True
)