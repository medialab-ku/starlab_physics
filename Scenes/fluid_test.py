from framework.meshio.meshTaichiWrapper import MeshTaichiWrapper
from framework.meshio.particle import Particle
import taichi as ti
from pathlib import Path

enable_profiler = False
ti.init(arch=ti.cuda, device_memory_GB=8, kernel_profiler=enable_profiler, default_fp=ti.float32)

model_path = Path(__file__).resolve().parent.parent / "models"
model_dir = str(model_path) + "/VTK"
# model_dir = model_dir + "/cube87K.vtk"

particle_names = []
trans = []
scale = []
rot = []
is_static = []
rho0 = []
particle_names.append("very_thin_sheet.vtk")
trans.append([0, 35.0, 0.0])
scale.append(2.0)
is_static.append(False)
rho0.append(1.0)

particles_dy = Particle(model_dir, particle_names, translations=trans, scales=scale, is_static=is_static, rho0=rho0)
particle_names.clear()
trans.clear()
scale.clear()
is_static.clear()

particle_names.append("smpl_neutral.vtk")
trans.append([0, 0.0, 0.0])
scale.append(20.0)
is_static.append(True)
rho0.append(1.0)
particles_st = Particle(model_dir, particle_names, translations=trans, scales=scale, is_static=is_static, rho0=rho0)