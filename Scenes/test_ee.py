from framework.meshio.TriMesh import TriMeshWrapper
import taichi as ti
from pathlib import Path

ti.init(arch=ti.cuda, device_memory_GB=6)

model_path = Path(__file__).resolve().parent.parent / "models"

#dynamic mesh
mesh_dy = TriMeshWrapper(str(model_path / "OBJ/square.obj"), offsets=[0], scale=2.0, trans=ti.math.vec3(1.0, 4.0, 1.0), rot=ti.math.vec3(0.0, 0.0, 40.0))

#static mesh
mesh_st = TriMeshWrapper(str(model_path / "OBJ/square.obj"), offsets=[0], scale=2.0, trans=ti.math.vec3(0.0, 0.0, 0.), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)