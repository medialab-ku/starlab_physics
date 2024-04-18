import meshio
import taichi as ti
import meshtaichi_patcher as patcher
import numpy as np
from mesh import Mesh
from particle import Particle
from TetMesh import TetMesh
import XPBD as xpbd

ti.init(arch=ti.cuda, device_memory_GB=12)

meshes_dynamic = []
mesh_dynamic_1 = Mesh("../models/OBJ/square_big.obj", scale=0.1, trans=ti.math.vec3(0.0, 1.5, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0))
# mesh_static_1 = Mesh("../models/OBJ/square_big.obj", scale=0.15, trans=ti.math.vec3(0.0, -0.6, 0.0), rot=ti.math.vec3(0.0, 10.0, 0.0), is_static=True)
mesh_dynamic_2 = Mesh("../models/OBJ/square_big.obj", scale=0.1, trans=ti.math.vec3(0.0, 1.3, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0))
mesh_dynamic_3 = Mesh("../models/OBJ/square_big.obj", scale=0.1, trans=ti.math.vec3(0.0, 1.5, 0.0), rot=ti.math.vec3(0.0, 0.0, 0.0))

# meshes_dynamic.append(mesh_dynamic_1)
# meshes_dynamic.append(mesh_dynamic_2)
# meshes.append(mesh_3)
# meshes.append(mesh_4)

meshes_static = []
mesh_static_1 = Mesh("../models/OBJ/square_big.obj", scale=0.15, trans=ti.math.vec3(0.0, -1.0, 0.0), rot=ti.math.vec3(0.0, 10.0, 0.0), is_static=True)

# meshes_static.append(mesh_static_1)

particles = []
particle_1 = Particle('../models/VTK/bunny.vtk', trans=ti.math.vec3(0.0, 0.2, 0.0), radius=0.01)
# particle_2 = Particle('../models/VTK/bunny.vtk', trans=ti.math.vec3(1.0, 0.0, 0.0), radius=0.01)


print(len(particles))

particles.append(particle_1)
# particles.append(particle_2)

sim = xpbd.Solver(meshes_dynamic, meshes_static, particles, g=ti.math.vec3(0.0, -1.0, 0.0), dt=0.01, grid_size=ti.math.vec3(3.0, 3.0, 3.0), particle_radius=0.01)
window = ti.ui.Window("PBD framework", (1024, 768), fps_limit=200)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(2., 4.0, 7.0)
camera.fov(40)
camera.up(0, 1, 0)

colors_dynamic = []
colors_dynamic.append((1, 0.5, 0))

colors_static = []
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

    if run_sim:
        sim.forward(n_substeps=10)

    for mid in range(len(meshes_dynamic)):
        scene.mesh(sim.meshes_dynamic[mid].mesh.verts.x, indices=sim.meshes_dynamic[mid].face_indices, color=(0, 0, 0), show_wireframe=True)
        scene.mesh(sim.meshes_dynamic[mid].mesh.verts.x, indices=sim.meshes_dynamic[mid].face_indices, color=colors_dynamic[mid])
        # scene.particles(sim.meshes[mid].mesh.verts.x, radius=sim.cell_size, color=colors[mid])


    for mid in range(len(meshes_static)):
        scene.mesh(sim.meshes_static[mid].mesh.verts.x, indices=sim.meshes_static[mid].face_indices, color=(0, 0, 0), show_wireframe=True)
        scene.mesh(sim.meshes_static[mid].mesh.verts.x, indices=sim.meshes_static[mid].face_indices, color=colors_static[mid])

    for pid in range(len(particles)):
        scene.particles(sim.particles[pid].x, radius=sim.particles[pid].radius, color=(1, 0, 0))

    # scene.particles(sim.x, radius=sim.cell_size, color=(0, 0, 0))
    # scene.particles(sim.x_static, radius=sim.cell_size, color=(0, 0, 0))
    scene.lines(sim.grid_vertices, indices=sim.grid_edge_indices, width=1.0, color=(0, 0, 0))
    # scene.mesh(sim.x, indices=sim.face_indices, color=(0, 0, 0), show_wireframe=True)

    # scene.lines(sim.x, indices=sim.edge_indices, color=(0, 0, 0), width=1.0)

    camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
    canvas.scene(scene)
    window.show()