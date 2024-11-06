from framework.meshio.TriMesh import TriMesh
from framework.meshio.TetMesh import TetMesh
from framework.meshio.particle import Particle
# from framework.meshio.concat import concat_mesh
import taichi as ti
import numpy as np
# import numpy as np
from pathlib import Path

enable_profiler = False
ti.init(arch=ti.cuda, device_memory_GB=8, default_fp=ti.f32, kernel_profiler=enable_profiler)

model_path = Path(__file__).resolve().parent.parent / "models"
OBJ = "OBJ"
CSV = "CSV"
obj_model_dir = str(model_path) + "/OBJ"
# obj_mesh_dy = TriMesh(
#     obj_model_dir,
#     model_name_list=[
#                      # "hood_modified.obj",
#                      "dress_modified.obj"
#                     ],
#     trans_list=[
#                 # (0.0, -2.4, 0.0),
#                 (0.0, 0.0, 0.0)
#                ],
#     scale_list=[
#                 # 2.3,
#                 1.0
#                ],
#     rot_list=[
#               (1.0, 0.0, 0.0, 0.0),
#               # (1.0, 0.0, 0.0, -3.14 / 2.0)
#                 ], # (axis.x, axis.y, axis.z, radian)
#     is_static=False)

obj_mesh_dy = TriMesh(
    obj_model_dir,
    model_name_list=[
                     # "tshirt.obj",
                     "plane.obj"
                    ],
    trans_list=[
                # (0.0, -2.4, 0.0),
                (0.0, 1.0, 0.0)
               ],
    scale_list=[
                # 2.3,
                3.0
               ],
    rot_list=[
              (1.0, 0.0, 0.0, 0.0),
              # (1.0, 0.0, 0.0, -3.14 / 2.0)
                ], # (axis.x, axis.y, axis.z, radian)
    is_static=False)

# obj_mesh_dy = TriMesh(
#     obj_model_dir,
#     model_name_list=[
#                      # "tshirt.obj",
#                      "plane.obj"
#                     ],
#     trans_list=[
#                 # (0.0, -2.4, 0.0),
#                 (0.0, 4.0, 0.0)
#                ],
#     scale_list=[
#                 # 2.3,
#                 2.0
#                ],
#     rot_list=[
#               # (1.0, 0.0, 0.0, 0.0),
#               (1.0, 0.0, 0.0, 0)], # (axis.x, axis.y, axis.z, radian)
#     is_static=False)


# mesh_st = None
# obj_mesh_st = TriMesh(
#     obj_model_dir,
#     model_name_list=["plane_8.obj"],
#     trans_list=[(0.0, 0.0, -3.0)],
#     scale_list=[0.1],
#     rot_list=[(1.0, 3.0, 5.0, 3.14/2)],
#     is_static=True
# )

msh_model_dir = str(model_path) + "/MSH"
#dynamic mesh
msh_mesh_dy = TetMesh(msh_model_dir,
                      model_name_list=["Armadillo54K.msh"],
                      trans_list=[(0.0, -2.0, 0.0)],
                      scale_list=[0.3],
                      rot_list=[0.0, 90.0, 0.0])


# model_path = Path(__file__).resolve().parent.parent / "models"
vtk_model_dir = str(model_path) + "/VTK"
particle_names = []
trans = []
scale = []
rot = []
is_static = []
rho0 = []
particle_names.append("smpl_neutral_new.vtk")
trans.append([0.0, 0.0, 0.0])
scale.append(0.0)
is_static.append(True)
rho0.append(1.0)


particles_st = Particle(vtk_model_dir, particle_names, translations=trans, scales=scale, rotations=[(1.0, 0.0, 0.0, 0.0)], is_static=is_static, rho0=rho0)
