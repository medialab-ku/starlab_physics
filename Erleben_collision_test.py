import taichi as ti
from my_mesh import Mesh
from my_solver import Solver

ti.init(arch=ti.cuda, device_memory_fraction=0.50)
vec = ti.math.vec3

SAVE_FRAMES = False

window_size = 1024  # Number of pixels of the window
dt = 0.001  # Larger dt might lead to unstable results.

######### Erleben's collision test cases #########
# https://dl.acm.org/doi/10.1145/3096239
# spike vs. spike
# mesh = Mesh("obj_models/erleben/spike.obj", scale=0.5, rot=ti.math.vec3(180.0, 0.0, 0.0), trans=ti.math.vec3(0.0, 2.5, 0.0))
# static_mesh = Mesh("obj_models/erleben/spike.obj", scale=0.5, rot=ti.math.vec3(0.0, 0.0, 0.0), trans=ti.math.vec3(0.0, 0.0, 0.0))

# spike vs. wedge
mesh = Mesh("obj_models/erleben/spike.obj", scale=0.5, rot=ti.math.vec3(180.0, 0.0, 0.0), trans=ti.math.vec3(0.0, 2.5, 0.0))
static_mesh = Mesh("obj_models/erleben/wedge.obj", scale=0.5, rot=ti.math.vec3(0.0, 0.0, 0.0), trans=ti.math.vec3(0.0, 0.0, 0.0))

# wedge vs. wedge
# mesh = Mesh("obj_models/erleben/wedge.obj", scale=0.5, rot=ti.math.vec3(180.0, 0.0, 0.0), trans=ti.math.vec3(0.0, 2.5, 0.0))
# static_mesh = Mesh("obj_models/erleben/wedge.obj", scale=0.5, rot=ti.math.vec3(0.0, 0.0, 0.0), trans=ti.math.vec3(0.0, 0.0, 0.0))

# spike in hole (initial velocity = (1.0, 0.0, 0.0)
# mesh = Mesh("obj_models/erleben/spike.obj", scale=0.5, rot=ti.math.vec3(180.0, 0.0, 0.0), trans=ti.math.vec3(0.0, 1.2, 0.0))
# static_mesh = Mesh("obj_models/erleben/hole.obj", scale=0.5, rot=ti.math.vec3(0.0, 0.0, 0.0), trans=ti.math.vec3(0.0, 0.0, 0.0))

# spike in crack (initial velocity = (1.0, 0.0, 0.0)
# mesh = Mesh("obj_models/erleben/spike.obj", scale=0.5, rot=ti.math.vec3(180.0, 0.0, 0.0), trans=ti.math.vec3(0.0, 1.5, 0.0))
# static_mesh = Mesh("obj_models/erleben/crack.obj", scale=0.5)

# wedge in crack (initial velocity = (1.0, 0.0, 0.0)
# mesh = Mesh("obj_models/erleben/wedge.obj", scale=0.5, rot=ti.math.vec3(180.0, 0.0, 0.0), trans=ti.math.vec3(0.0, 1.05, 0.0))
# static_mesh = Mesh("obj_models/erleben/crack.obj", scale=0.5)

# sliding spike (initial velocity = (1.0, 0.0, 0.0))
# mesh = Mesh("obj_models/erleben/spike.obj", scale=0.5, rot=ti.math.vec3(180.0, 0.0, 0.0), trans=ti.math.vec3(0.0, 1.2, 0.0))
# static_mesh = Mesh("obj_models/square.obj", scale=1.0)

# sliding wedge (initial velocity = (1.0, 0.0, 0.0))
# mesh = Mesh("obj_models/erleben/wedge.obj", scale=0.5, rot=ti.math.vec3(180.0, 0.0, 0.0), trans=ti.math.vec3(0.0, 1.01, 0.0))
# static_mesh = Mesh("obj_models/square.obj", scale=1.0)

# internal edges
# mesh = Mesh("obj_models/cube.obj", scale=0.5, rot=ti.math.vec3(180.0, 0.0, 0.0), trans=ti.math.vec3(0.0, 0.41, 0.0))
# static_mesh = Mesh("obj_models/erleben/internal_edges.obj", scale=0.5)

# cliff edges
# mesh = Mesh("obj_models/cube.obj", scale=0.5, rot=ti.math.vec3(180.0, 0.0, 0.0), trans=ti.math.vec3(0.0, 0.51, 0.0))
# static_mesh = Mesh("obj_models/erleben/cliff.obj", scale=0.5)

# mesh = Mesh("obj_models/square_big.obj", scale=0.2, rot=ti.math.vec3(0.0, 0.0, 20.0), trans=ti.math.vec3(0.5, 1.0, 0.5))
# static_mesh =Mesh("obj_models/square_big.obj", scale=0.4, rot=ti.math.vec3(0.0, 20.0, 0.0), trans=ti.math.vec3(0.5, 0.0, 0.5))

per_vertex_color = ti.Vector.field(3, ti.float32, shape=4)
debug_edge_indices = ti.field(dtype=ti.i32, shape=2)

debug_edge_indices[0] = 0
debug_edge_indices[1] = 1


@ti.kernel
def init_color():
    per_vertex_color[0] = ti.math.vec3(1, 0, 0)
    per_vertex_color[1] = ti.math.vec3(0, 1, 0)
    per_vertex_color[2] = ti.math.vec3(0, 0, 1)
    per_vertex_color[3] = ti.math.vec3(0, 1, 1)

init_color()

sim = Solver(mesh, bottom=0.0, static_mesh=static_mesh, dt=dt, max_iter=20)

window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 768), fps_limit=200)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(1., 2.0, 3.5)
camera.fov(30)
camera.up(0, 1, 0)
# x0 = sim.verts.x
# sim.update()
# x1 = sim.verts.y

run_sim = False

while window.running:
    if window.get_event(ti.ui.PRESS):
        if window.event.key == ' ':
            run_sim = not run_sim

        if window.event.key == 'r':
            sim.reset()
            run_sim = False

    if run_sim:
        sim.update()
    camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
    camera.lookat(0.5, 0.5, 0.5)
    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.3, 0.3, 0.3))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.3, 0.3, 0.3))
    # scene.particles(sim.verts.x, radius=0.01, color=(0, 1, 0), per_vertex_color=per_vertex_color)
    # scene.particles(sim.intersect, radius=0.01, color=(0, 1, 0), per_vertex_color=per_vertex_color)
    # scene.particles(static_mesh.mesh.verts.x, radius=sim.radius, color=(0, 1, 0), per_vertex_color=per_vertex_color)
    scene.mesh(sim.verts.x, mesh.face_indices, color=(1., 0.5, 0.0))
    scene.mesh(static_mesh.mesh.verts.x, static_mesh.face_indices, color=(0.5, 0.5, 0.5))
    scene.lines(sim.verts.x, width=0.5, indices=mesh.edge_indices, color=(0., 0., 0.))
    # scene.lines(x1, width=0.5, indices=mesh.edge_indices, color=(0., 0., 0.))
    scene.lines(sim.p, width=0.5, indices=debug_edge_indices, color=(1., 0., 0.))
    scene.lines(static_mesh.mesh.verts.x, width=0.5, indices=static_mesh.edge_indices, color=(0., 0., 0.))
    canvas.scene(scene)
    window.show()
