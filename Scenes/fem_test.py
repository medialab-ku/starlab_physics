from framework.meshio.TetMesh import TetMesh
import taichi as ti
from pathlib import Path

enable_profiler = False
ti.init(arch=ti.cuda, device_memory_GB=8, kernel_profiler=enable_profiler, default_fp=ti.float32)

model_path = Path(__file__).resolve().parent.parent / "models"
msh_model_dir = str(model_path) + "/MSH"
#dynamic mesh
msh_mesh_dy = TetMesh(msh_model_dir, model_name_list=["tet.msh"], scale_list=[30, 0], trans_list=[(0.0, 0.0, 0.0)], rot=ti.math.vec3(0.0, 0.0, 0.0))
