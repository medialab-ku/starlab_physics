from framework.meshio.TetMesh import TetMeshWrapper as tm
import taichi as ti
from pathlib import Path

enable_profiler = False
ti.init(arch=ti.cuda, device_memory_GB=8, kernel_profiler=enable_profiler, default_fp=ti.float32)

model_path = Path(__file__).resolve().parent.parent / "models"
model_dir = str(model_path) + "/MSH"

model_name_list = []
model_name_list.append("cube.msh")
# model_name_list.append("tet.msh")

scale_list = []
scale_list.append(10.0)
# scale_list.append(5.0)

trans_list = []
trans_list.append([0.0, 0.0, 0.0])
# trans_list.append([0.0, 0.0, 0.0])

#dynamic mesh
mesh_dy = tm(str(model_dir), model_name_list, scale_list=scale_list, trans_list=trans_list, rot=ti.math.vec3(0.0, 0.0, 0.0))
#static mesh
# mesh_st = MeshTaichiWrapper(model_dir, "SMPL_APose.obj",  offsets=[0], scale=12.0, trans=ti.math.vec3(0.0, 0.0, 0.01), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)

# if you want to use another mesh as a static object...
# mesh_st = MeshTaichiWrapper(str(model_path) , "OBJ/plane_8.obj",  offsets=[0], scale=50.0, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)