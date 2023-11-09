import taichi as ti
import numpy as np

from ccd import *
from my_mesh import Mesh
from my_solver import Solver

ti.init(arch=ti.gpu)

# particle = ti.Vector.field(3, dtype=ti.f32, shape=1)
# p_np = np.array([[0.5, 2.0, 0.5]])
# particle.from_numpy(p_np)

dt = 0.003
radius = 0.1
mesh = Mesh('obj_models/square.obj', scale=0.3, trans=(0.0, 1.0, 0.0))
static_mesh = Mesh('obj_models/square_16K.obj', scale=0.1)
sim = Solver(mesh, static_mesh, None, dt=dt, max_iter=10)

window = ti.ui.Window("simple window", (1024, 1024))
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(5, 4, 5)
camera.fov(50)
camera.up(0, 1, 0)

run_sim = False
frame = 0
while window.running:
    if window.get_event(ti.ui.PRESS):
        if window.event.key == ' ':
            run_sim = not run_sim

        if window.event.key == 'r':
            sim.reset()
            frame = 0
            run_sim = False

    if run_sim:
        sim.update(dt=dt, num_sub_steps=10)
        frame += 1
        print('frame:', frame)
    # if frame % 200 == 0 and frame > 0:
    #     sim.move_static_pos(rot=(0.0, 0.0, 0.0), trans=(0.0, 1.0, 0.0))

    camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
    camera.lookat(0.5, 0.5, 0.5)
    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.3, 0.3, 0.3))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.3, 0.3, 0.3))
    # scene.particles(sim.p, radius=0.05, color=(1.0, 0.0, 0.0))
    # scene.mesh(sim.verts.x, sim.face_indices, color=(1.0, 0.5, 0.0))
    # scene.mesh(sim.verts_static.x, sim.face_indices_static, color=(0.5, 0.5, 0.5))
    scene.particles(sim.verts.x, radius=radius, color=(1.0, 0.5, 0.0))
    scene.particles(sim.verts_static.x, radius=radius, color=(0.5, 0.5, 0.5))
    canvas.scene(scene)
    window.show()