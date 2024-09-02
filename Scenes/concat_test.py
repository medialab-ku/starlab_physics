from framework.meshio.TriMesh import TriMesh
from framework.meshio.TetMesh import TetMesh
# from framework.meshio.concat import concat_mesh
import taichi as ti
# import numpy as np
from pathlib import Path

enable_profiler = False
ti.init(arch=ti.cuda, device_memory_GB=8, default_fp=ti.f32, kernel_profiler=enable_profiler)

model_path = Path(__file__).resolve().parent.parent / "models"
OBJ = "OBJ"
obj_model_dir = str(model_path) + "/OBJ"

obj_mesh_dy = TriMesh(
    obj_model_dir,
    model_name_list=["plane_64.obj"],
    trans_list=[(0.0, 5.0, 0.0)],
    scale_list=[10.0],
    rot_list=[], # (axis.x, axis.y, axis.z, radian)
    is_static=False)

# mesh_st = None
obj_mesh_st = TriMesh(
    obj_model_dir,
    model_name_list=["square.obj"],
    trans_list=[(0.0, -20.0, 0.0)],
    scale_list=[1.5],
    rot_list=[],
    is_static=True
)

msh_model_dir = str(model_path) + "/MSH"
#dynamic mesh
msh_mesh_dy = TetMesh(msh_model_dir,
                      model_name_list=["sphere19K.msh"],
                      trans_list=[(0.0, 0.0, 0.0)],
                      scale_list=[30.0],
                      rot_list=[])
