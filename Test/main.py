import random
import taichi as ti
from mesh import Mesh
from particle import Particle
from TetMesh import TetMesh
import XPBD as xpbd

ti.init(arch=ti.cuda, device_memory_GB=3)

meshes_dynamic = []
# mesh_dynamic_1 = Mesh("../models/OBJ/square_big.obj", scale=0.1, trans=ti.math.vec3(0.0, 1.0, 0.0), rot=ti.math.vec3(0.0, 0.0, 45.0))
# mesh_static_1 = Mesh("../models/OBJ/square_big.obj", scale=0.15, trans=ti.math.vec3(0.0, -0.6, 0.0), rot=ti.math.vec3(0.0, 10.0, 0.0), is_static=True)
# mesh_dynamic_2 = Mesh("../models/OBJ/square_huge.obj", scale=0.2, trans=ti.math.vec3(0.0, 1.3, 0.0), rot=ti.math.vec3(0.0, 0.0, 45.0))
# mesh_dynamic_2 = Mesh("../models/OBJ/square_huge.obj", scale=0.2, trans=ti.math.vec3(0.0, 1.3, 0.0), rot=ti.math.vec3(0.0, 0.0, 45.0))
mesh_dynamic_3 = Mesh("../models/OBJ/square_huge.obj", scale=0.15, trans=ti.math.vec3(0.0, 1.3, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0))
mesh_dynamic_4 = Mesh("../models/OBJ/square_huge.obj", scale=0.15, trans=ti.math.vec3(0.0, 1.5, 0.0), rot=ti.math.vec3(0.0, 0.0, 10.0))
mesh_dynamic_5 = Mesh("../models/OBJ/square_huge.obj", scale=0.15, trans=ti.math.vec3(0.0, 1.7, 0.0), rot=ti.math.vec3(0.0, 0.0, 20.0))
# mesh_dynamic_4 = Mesh("../models/OBJ/triangle.obj", scale=1.0, trans=ti.math.vec3(0.0, 1.5, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0))
# mesh_dynamic_5 = Mesh("../models/OBJ/triangle.obj", scale=1.0, trans=ti.math.vec3(0.0, 1.5, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0))


# meshes_dynamic.append(mesh_dynamic_1)
# meshes_dynamic.append(mesh_dynamic_3)
# meshes_dynamic.append(mesh_dynamic_4)
meshes_dynamic.append(mesh_dynamic_5)
# meshes.append(mesh_4)

tet_meshes_dynamic = []

# tet_mesh_dynamic_1 = TetMesh("../models/MESH/bunny_tiny.1.node",  scale=0.1, trans=ti.math.vec3(0.8, 0.2, 0.8), rot=ti.math.vec3(0.0, 0.0, 0.0))
# tet_mesh_dynamic_2 = TetMesh("../models/MESH/bunny_tiny.1.node", scale=0.1, trans=ti.math.vec3(-0.8, 0.2, -0.8), rot=ti.math.vec3(0.0, 0.0, 0.0))
# tet_mesh_dynamic_3 = TetMesh("../models/MESH/bunny_tiny.1.node", scale=0.1, trans=ti.math.vec3(0.0, 0.2, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0))
# tet_mesh_dynamic_4 = TetMesh("../models/MESH/dragon.1.1.node", scale=0.1, trans=ti.math.vec3(0.0, 0.2, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0))
# tet_meshes_dynamic.append(tet_mesh_dynamic_1)
# tet_meshes_dynamic.append(tet_mesh_dynamic_2)
# tet_meshes_dynamic.append(tet_mesh_dynamic_3)
# tet_meshes_dynamic.append(tet_mesh_dynamic_4)

meshes_static = []
mesh_static_1 = Mesh("../models/OBJ/sphere1K.obj", scale=2.0, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)
mesh_static_2 = Mesh("../models/OBJ/square_big.obj", scale=0.15, trans=ti.math.vec3(0.0, -0.5, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)
mesh_static_3 = Mesh("../models/OBJ/square_huge.obj", scale=0.25, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(0.0, 15.0, 0.0), is_static=True)
mesh_static_4 = Mesh("../models/OBJ/triangle.obj", scale=1.0, trans=ti.math.vec3(0.0, 0.0, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)

meshes_static.append(mesh_static_1)
# meshes_static.append(mesh_static_3)
# meshes_static.append(mesh_static_4)
# meshes_static.append(mesh_static_2)

particles = []
particle_1 = Particle('../models/VTK/cube87K.vtk', trans=ti.math.vec3(0.0, 1.0, 0.0), scale=1.2, radius=0.01)
particle_2 = Particle('../models/VTK/cube1K.vtk', trans=ti.math.vec3(-0.8, 2.0, 0.8), scale=1.2, radius=0.01)
# particle_2 = Particle('../models/VTK/bunny.vtk', trans=ti.math.vec3(1.0, 0.0, 0.0), radius=0.01)


# particles.append(particle_1)
# particles.append(particle_2)

sim = xpbd.Solver(meshes_dynamic, meshes_static, tet_meshes_dynamic, particles, g=ti.math.vec3(0.0, -9.81, 0.0), dt=0.03, grid_size=ti.math.vec3(3.5, 3.5, 3.5), particle_radius=0.02)
window = ti.ui.Window("PBD framework", (1024, 768), fps_limit=200)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(2., 4.0, 7.0)
camera.fov(40)
camera.up(0, 1, 0)


colors_tet_dynamic = []

for tid in range(len(tet_meshes_dynamic)):
    color = (random.randrange(0, 255) / 256, random.randrange(0, 255) / 256, random.randrange(0, 255) / 256)
    colors_tet_dynamic.append(color)


colors_tri_dynamic = []
for mid in range(len(meshes_dynamic)):
    color = (random.randrange(0, 255) / 256, random.randrange(0, 255) / 256, random.randrange(0, 255) / 256)
    colors_tri_dynamic.append(color)

colors_static = []
colors_static.append((0, 0.5, 0))
colors_static.append((0, 0.5, 0))

run_sim = True

while window.running:

    camera.lookat(0.0, 0.0, 0.0)
    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.3, 0.3, 0.3))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.3, 0.3, 0.3))

    if window.get_event(ti.ui.PRESS):
        if window.event.key == ' ':
            run_sim = not run_sim

        if window.event.key == 'r':
            sim.reset()
            run_sim = False

        if window.event.key == 'v':
            sim.enable_velocity_update = not sim.enable_velocity_update
            if sim.enable_velocity_update is True:
                print("enable velocity update")
            else:
                print("disable velocity update")

    if run_sim:
        sim.forward(n_substeps=20)

    for mid in range(len(meshes_dynamic)):
        # scene.mesh(sim.meshes_dynamic[mid].mesh.verts.x, indices=sim.meshes_dynamic[mid].face_indices, color=(0, 0, 0), show_wireframe=True)
        scene.mesh(sim.meshes_dynamic[mid].mesh.verts.x, indices=sim.meshes_dynamic[mid].face_indices, color=colors_tri_dynamic[mid])
        # scene.particles(sim.meshes[mid].mesh.verts.x, radius=sim.cell_size, color=colors[mid])

    for tid in range(len(tet_meshes_dynamic)):
        # scene.mesh(sim.tet_meshes_dynamic[tid].verts.x, indices=sim.tet_meshes_dynamic[tid].face_indices, color=(0, 0, 0), show_wireframe=True)
        scene.mesh(sim.tet_meshes_dynamic[tid].verts.x, indices=sim.tet_meshes_dynamic[tid].face_indices, color=colors_tet_dynamic[tid])

    for pid in range(len(particles)):
        scene.particles(sim.particles[pid].x, radius=sim.particle_radius, color=(1, 0, 0))

    for mid in range(len(meshes_static)):
        # scene.mesh(sim.meshes_static[mid].mesh.verts.x, indices=sim.meshes_static[mid].face_indices, color=(0, 0, 0), show_wireframe=True)
        scene.mesh(sim.meshes_static[mid].mesh.verts.x, indices=sim.meshes_static[mid].face_indices, color=colors_static[mid])

    # scene.particles(sim.x, radius=sim.cell_size, color=(0, 0, 0))
    # scene.particles(sim.x_static, radius=sim.cell_size, color=(0, 0, 0))
    scene.lines(sim.grid_vertices, indices=sim.grid_edge_indices, width=1.0, color=(0, 0, 0))
    # scene.mesh(sim.x, indices=sim, color=(0, 0, 0), show_wireframe=True)

    # scene.lines(sim.x, indices=sim.edge_indices_dynamic, color=(0, 0, 0), width=1.0)
    # scene.lines(sim.x_static, indices=sim.edge_indices_static, color=(0, 0, 0), width=1.0)
    scene.mesh(sim.x_static,  indices=sim.face_indices_static, color=(0, 0, 0), show_wireframe=True)
    scene.mesh(sim.x,  indices=sim.face_indices_dynamic, color=(0, 0, 0), show_wireframe=True)
    camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
    canvas.scene(scene)
    window.show()