from framework.meshio.meshTaichiWrapper import MeshTaichiWrapper
from framework.meshio.particle import Particle
import taichi as ti
from pathlib import Path

enable_profiler = False
ti.init(arch=ti.cuda, device_memory_GB=8, kernel_profiler=enable_profiler,default_fp=ti.float64)

gravity = ti.math.vec3(0.0, -9.8, 0.0)
dt = 0.03
grid_size = ti.math.vec3(7, 7, 7)
# particle_radius = 0.02

model_path = Path(__file__).resolve().parent.parent / "models"
model_dir = str(model_path) + "/VTK"
# print(model_dir)

model_dir = model_dir+"/cuboid224_170K.vtk"

particles = Particle(model_dir,trans=ti.math.vec3(0,50.0,0.0),scale=ti.math.vec3(1,1,1) * 20,)


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
offsets = []

#dynamic mesh
mesh_dy = MeshTaichiWrapper(str(model_path), "OBJ/torus.obj", offsets=offsets, scale=1.0, trans=ti.math.vec3(0, 0.0, 5555.0), rot=ti.math.vec3(0.0, 0.0, 0.0))
#static mesh
# mesh_st = MeshTaichiWrapper(model_dir, "SMPL_APose.obj",  offsets=[0], scale=12.0, trans=ti.math.vec3(0.0, 0.0, 0.01), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)

# if you wan to use another mesh as a static object...
mesh_st = MeshTaichiWrapper(str(model_path) , "OBJ/plane_8.obj",  offsets=[0], scale=50.0, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)