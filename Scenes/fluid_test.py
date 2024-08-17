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

particle_names.append("cube.vtk")
# particle_names.append("sphere5K.vtk")
# particle_names.append("cube.vtk")
# particle_names.append("cube.vtk")
# particle_names.append("cube.vtk")
# particle_names.append("cube.vtk")
# particle_names.append("cube.vtk")
# trans.append([0, -5.0, 0.0])
trans.append([0, 20.0, 0.0])
# trans.append([0, -20.0, 0.0])
# trans.append([0, -5.0, 0.0])
# trans.append([0.0, 5.0, 0.0])
# trans.append([5.0, 5.0, 5.0])
# trans.append([5.0, -5.0, 0.0])
# scale.append(5.0)
scale.append(20.0)
# scale.append(20.0)
# scale.append(10.0)
# scale.append(5.0)
# scale.append(5.0)
# scale.append(8.0)
is_static.append(False)
# is_static.append(False)
# is_static.append(False)
# is_static.append(False)
# is_static.append(False)
# is_static.append(True)

rho0.append(1.0)
# rho0.append(1e5)
# rho0.append(1000.0)

particles = Particle(model_dir, particle_names, translations=trans, scales=scale, is_static=is_static, rho0=rho0)
