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
particle_names.append("sphere5K.vtk")
# particle_names.append("cube.vtk")
# particle_names.append("cube.vtk")
# particle_names.append("cube.vtk")
# particle_names.append("cube.vtk")
# particle_names.append("cube.vtk")
# trans.append([0, -5.0, 0.0])
trans.append([0, 20.0, 0.0])
trans.append([0, -20.0, 20.0])
# trans.append([0, -5.0, 0.0])
# trans.append([0.0, 5.0, 0.0])
# trans.append([5.0, 5.0, 5.0])
# trans.append([5.0, -5.0, 0.0])
# scale.append(5.0)
scale.append(20.0)
scale.append(20.0)
# scale.append(10.0)
# scale.append(5.0)
# scale.append(5.0)
# scale.append(8.0)
is_static.append(False)
is_static.append(False)
# is_static.append(False)
# is_static.append(False)
# is_static.append(False)
# is_static.append(True)

rho0.append(1.0)
rho0.append(1e5)
# rho0.append(1000.0)

particles_dy = Particle(model_dir, particle_names, translations=trans, scales=scale, is_static=is_static, rho0=rho0)
# model_dir = model_dir + "/cuboid224_170K.vtk"
particles_st = None
particle_names.clear()
trans.clear()
scale.clear()

particle_names.append("cube.vtk")
# particle_names.append("cube.vtk")
trans.append([0.0, 0.0, 0.0])
# trans.append([0.0, 5.0, 0.0])
# scale.append(1.0)
scale.append(20.0)
# scale.append(5.0)
# particles_st = Particle(model_dir, particle_names, translations=trans, scales=scale,)

# model_names = []
# trans_list = []
# scale_list = []
#
# concat_model_name = "concat.obj"
#
# model_names.append("poncho_8K.obj")
# trans_list.append([0.0, 8.0, 0.0])
# scale_list.append(3.0)
#

# append more meshes
# model_names.append("your-obj-name.obj")
# trans_list.append([x, y, z])
# scale_list.append(size)

# offsets = concat_mesh(concat_model_name, model_dir, model_names, trans_list, scale_list)
# offsets = []

#dynamic mesh
# mesh_dy = MeshTaichiWrapper(str(model_path), "OBJ/torus.obj", offsets=offsets, scale=1.0, trans=ti.math.vec3(0, 0.0, 5555.0), rot=ti.math.vec3(0.0, 0.0, 0.0))
#static mesh
# mesh_st = MeshTaichiWrapper(model_dir, "SMPL_APose.obj",  offsets=[0], scale=12.0, trans=ti.math.vec3(0.0, 0.0, 0.01), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)

# if you want to use another mesh as a static object...
# mesh_st = MeshTaichiWrapper(str(model_path) , "OBJ/plane_8.obj",  offsets=[0], scale=50.0, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)