import taichi as ti
from my_mesh import Mesh
from my_solver import Solver

ti.init(arch=ti.cuda)
vec = ti.math.vec3

SAVE_FRAMES = False

window_size = 1024  # Number of pixels of the window
dt = 0.001  # Larger dt might lead to unstable results.

# mesh = Mesh("obj_models/cube.obj", scale=0.1, rot=ti.math.vec3(90.0, 0.0, 0.0),trans=ti.math.vec3(0.3, 0.5, 0.3))
# static_mesh =Mesh("obj_models/cube.obj", scale=0.1, rot=ti.math.vec3(180.0, 0.0, 0.0), trans=ti.math.vec3(0.3, 0.2, 0.3))

# mesh = Mesh("obj_models/clubbing_dress.obj", scale=0.571, rot=ti.math.vec3(90.0, 0.0, 0.0),trans=ti.math.vec3(0.307, 0.47, 0.31))
# static_mesh =Mesh("obj_models/kyra_model_reduced.obj", scale=0.40, rot=ti.math.vec3(90.0, 00.0, 0.0), trans=ti.math.vec3(0.305, 0.58, 0.325))

#mesh = Mesh("tet_models/bunny_small.mesh", scale=0.1, rot=ti.math.vec3(0.0, 0.0, 0.0))
# static_mesh = Mesh("obj_models/bunny.obj", scale=0.4, rot=ti.math.vec3(0.0, 0.0, 0.0), trans=ti.math.vec3(0.0, -0.5, 0.0))
# static_mesh =Mesh("obj_models/cube.obj", scale=0.1, rot=ti.math.vec3(90.0, 0.0, 0.0), trans=ti.math.vec3(0.3, 0.2, 0.3))

# case: face vs. vertex
# mesh = Mesh("obj_models/tetrahedron.obj", scale=0.1, rot=ti.math.vec3(0.0, 0.0, 0.0),trans=ti.math.vec3(0.3, 0.5, 0.3))
# static_mesh = Mesh("obj_models/square.obj", scale=0.4, rot=ti.math.vec3(180.0, 0.0, 0.0), trans=ti.math.vec3(0.2, 0.2, 0.3))


# case: face vs. face
# mesh = Mesh("obj_models/tetrahedron.obj", scale=0.4, rot=ti.math.vec3(0.0, 0.0, 0.0), trans=ti.math.vec3(0.4, 0.5, 0.5))
# static_mesh = Mesh("obj_models/square.obj", scale=0.4, rot=ti.math.vec3(0.0, 0.0, 0.0), trans=ti.math.vec3(0.2, 0.2, 0.3))

#case: edge vs. edge
# mesh = Mesh("obj_models/tetrahedron.obj", scale=0.3, rot=ti.math.vec3(60.0, 0.0, 0.0),trans=ti.math.vec3(0.5, 0.8, 0.5))
# static_mesh = Mesh("obj_models/tetrahedron.obj", scale=0.3, rot=ti.math.vec3(60.0, 0.0, 0.0), trans=ti.math.vec3(0.5, 0.21, 0.5))

#case: face vs. face
# mesh = Mesh("obj_models/tetrahedron.obj", scale=0.3, rot=ti.math.vec3(0.0, 0.0, 0.0),trans=ti.math.vec3(0.3, 0.6, 0.3))
# static_mesh = Mesh("obj_models/tetrahedron.obj", scale=0.3, rot=ti.math.vec3(180.0, 0.0, 45.0), trans=ti.math.vec3(0.3, 0.2, 0.3))

# mesh = Mesh("obj_models/tetrahedron.obj", scale=0.3, rot=ti.math.vec3(0.0, 0.0, 0.0),trans=ti.math.vec3(0.3, 0.6, 0.3))
# static_mesh = Mesh("obj_models/tetrahedron.obj", scale=0.3, rot=ti.math.vec3(180.0, 0.0, 0.0), trans=ti.math.vec3(0.3, 0.2, 0.3))

#case: vertex vs. vertex
# mesh = Mesh("obj_models/tetrahedron.obj", scale=0.3, rot=ti.math.vec3(180.0, 0.0, 0.0),trans=ti.math.vec3(0.3, 0.9, 0.3))
# static_mesh = Mesh("obj_models/tetrahedron.obj", scale=0.3, rot=ti.math.vec3(0.0, 0.0, 0.0), trans=ti.math.vec3(0.3, 0.2, 0.3))

#case: edge vs. edge, edge vertex
# mesh = Mesh("obj_models/tetrahedron.obj", scale=0.3, rot=ti.math.vec3(0.0, 0.0, 20.0),trans=ti.math.vec3(0.5, 0.9, 0.5))
# static_mesh = Mesh("obj_models/tetrahedron.obj", scale=0.3, rot=ti.math.vec3(0.0, 0.0, 0.0), trans=ti.math.vec3(0.3, 0.2, 0.3))

# case: vertex vs. face
# mesh = Mesh("obj_models/tetrahedron.obj", scale=0.3, rot=ti.math.vec3(0.0, 0.0, 0.0),trans=ti.math.vec3(0.3, 0.9, 0.3))
# static_mesh =Mesh("obj_models/tetrahedron.obj", scale=0.3, rot=ti.math.vec3(0.0, 0.0, 0.0), trans=ti.math.vec3(0.3, 0.2, 0.3))

# mesh = Mesh("obj_models/square_big.obj", scale=0.05, rot=ti.math.vec3(0.0, 0.0, 0.0),trans=ti.math.vec3(0.5, 0.8, 0.5))
# static_mesh =Mesh("obj_models/cube.obj", scale=0.5, rot=ti.math.vec3(90.0, 0.0, 0.0), trans=ti.math.vec3(0.5, 0.21, 0.5))


#line r g, dtype = 3
# mesh = Mesh("obj_models/triangle.obj", scale=0.5, rot=ti.math.vec3(90, 0.0, 0.0),trans=ti.math.vec3(1.0, 0.7, 1.01))
# static_mesh = Mesh("obj_models/triangle.obj", scale=0.5, rot=ti.math.vec3(0.0, 0.0, 0.0),trans=ti.math.vec3(0.5, 0.1, 0.5))

#line c g, dtype = 4
# mesh = Mesh("obj_models/triangle.obj", scale=0.5, rot=ti.math.vec3(90, 0.0, 0.0),trans=ti.math.vec3(1.6, 0.7, 0.5))
# static_mesh = Mesh("obj_models/triangle.obj", scale=0.5, rot=ti.math.vec3(0.0, 0.0, 0.0),trans=ti.math.vec3(0.5, 0.1, 0.5))

#line r c, dtype = 5
# mesh = Mesh("obj_models/triangle.obj", scale=0.5, rot=ti.math.vec3(90, 0.0, 0.0),trans=ti.math.vec3(0.7, 0.7,0.7))
# static_mesh = Mesh("obj_models/triangle.obj", scale=0.5, rot=ti.math.vec3(0.0, 0.0, 0.0),trans=ti.math.vec3(0.5, 0.1, 0.5))

#point r, dtype = 0
# mesh = Mesh("obj_models/triangle.obj", scale=0.5, rot=ti.math.vec3(90, 0.0, 0.0),trans=ti.math.vec3(0.3, 0.7, 1.1))
# static_mesh = Mesh("obj_models/triangle.obj", scale=0.5, rot=ti.math.vec3(0.0, 0.0, 0.0),trans=ti.math.vec3(0.5, 0.1, 0.5))

#point g, dtype = 1
# mesh = Mesh("obj_models/triangle.obj", scale=0.5, rot=ti.math.vec3(90, 0.0, 0.0),trans=ti.math.vec3(1.7, 0.7, 1.7))
# static_mesh = Mesh("obj_models/triangle.obj", scale=0.5, rot=ti.math.vec3(0.0, 0.0, 0.0),trans=ti.math.vec3(0.5, 0.1, 0.5))

#triangle r g c, dtype = 6
# mesh = Mesh("obj_models/triangle.obj", scale=0.5, rot=ti.math.vec3(90, 0.0, 0.0),trans=ti.math.vec3(1.1, 0.7, 0.7))
# static_mesh = Mesh("obj_models/triangle.obj", scale=0.5, rot=ti.math.vec3(0.0, 0.0, 0.0),trans=ti.math.vec3(0.5, 0.1, 0.5))

# x0:r, x1: b, x2:g, x3:c
# PP(x1, x3) dtype = 4
# mesh = Mesh("obj_models/triangle.obj", scale=0.5, rot=ti.math.vec3(180, 0.0, 0.0),trans=ti.math.vec3(1.6, 0.1, -0.6))
# static_mesh = Mesh("obj_models/triangle.obj", scale=0.5, rot=ti.math.vec3(0.0, 0.0, 0.0),trans=ti.math.vec3(0.5, 0.1, 0.5))

# PP(x0, x3) dtype = 1
# mesh = Mesh("obj_models/triangle.obj", scale=0.5, rot=ti.math.vec3(0, 0.0, 0.0),trans=ti.math.vec3(1.5, 0.1, -0.6))
# static_mesh = Mesh("obj_models/triangle.obj", scale=0.5, rot=ti.math.vec3(0.0, 0.0, 0.0),trans=ti.math.vec3(0.5, 0.1, 0.5))

# PP(x1, x2) dtype = 3
# mesh = Mesh("obj_models/triangle.obj", scale=0.5, rot=ti.math.vec3(0, 0.0, 0.0),trans=ti.math.vec3(1.5, 0.1, 1.6))
# static_mesh = Mesh("obj_models/triangle.obj", scale=0.5, rot=ti.math.vec3(0.0, 0.0, 0.0),trans=ti.math.vec3(0.5, 0.1, 0.5))

# PE(x0, x2, x3) dtype = 2
# mesh = Mesh("obj_models/triangle.obj", scale=0.5, rot=ti.math.vec3(-90, 0.0, 0.0),trans=ti.math.vec3(1.5, 0.7, 0.6))
# static_mesh = Mesh("obj_models/triangle.obj", scale=0.5, rot=ti.math.vec3(0.0, 0.0, 0.0),trans=ti.math.vec3(0.5, 0.1, 0.5))

# PE(x1, x2, x3) dtype = 5

# mesh = Mesh("obj_models/triangle.obj", scale=0.5, rot=ti.math.vec3(90, 0.0, 0.0),trans=ti.math.vec3(1.5, 0.7, 0.6))
# static_mesh = Mesh("obj_models/triangle.obj", scale=0.5, rot=ti.math.vec3(0.0, 0.0, 0.0),trans=ti.math.vec3(0.5, 0.1, 0.5))

mesh = Mesh("obj_models/square_huge.obj", scale=0.05, rot=ti.math.vec3(0.0, 0.0, 0.0), trans=ti.math.vec3(0.5, 1.3, 0.5))
static_mesh =Mesh("obj_models/square_big.obj", scale=0.4, rot=ti.math.vec3(0.0, 10.0, 0.0), trans=ti.math.vec3(0.5, 0.0, 0.5))

per_vertex_color = ti.Vector.field(3, ti.float32, shape=4)
debug_edge_indices = ti.field(dtype=ti.i32, shape=2)

center = ti.Vector.field(n=3, dtype=ti.float32, shape=1)
center[0] = ti.math.vec3(0.5, 0.5, 0.5)

debug_edge_indices[0] = 0
debug_edge_indices[1] = 1
#
# mesh = Mesh("obj_models/square_huge.obj", scale=0.4, trans=ti.math.vec3(0, 1, 0), rot=ti.math.vec3(0.0, 0.0, 0.0))
# static_mesh = Mesh("obj_models/sphere1K.obj", scale=3.0, trans=ti.math.vec3(0, -2, 0), rot=ti.math.vec3(0.0, 0.0, 0.0))

@ti.kernel
def init_color():
    per_vertex_color[0] = ti.math.vec3(1, 0, 0)
    per_vertex_color[1] = ti.math.vec3(0, 1, 0)
    per_vertex_color[2] = ti.math.vec3(0, 0, 1)
    per_vertex_color[3] = ti.math.vec3(0, 1, 1)

init_color()

sim = Solver(mesh, bottom=0.0, static_mesh=static_mesh, dt=dt, max_iter=1)

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
    # scene.particles(centers=center, radius=0.3, color=(1, 0, 0))
    # scene.particles(sim.verts.x, radius=0.01, color=(0, 1, 0))
    # scene.particles(sim.intersect, radius=0.01, color=(0, 1, 0), per_vertex_color=per_vertex_color)
    # scene.particles(static_mesh.mesh.verts.x, radius=sim.radius, color=(0, 1, 0), per_vertex_color=per_vertex_color)
    scene.mesh(sim.verts.x, mesh.face_indices, color=(1., 0.5, 0.0))
    # scene.mesh(static_mesh.mesh.verts.x, static_mesh.face_indices, color=(0.5, 0.5, 0.5))
    scene.lines(sim.verts.x, width=0.5, indices=mesh.edge_indices, color=(0., 0., 0.))
    # scene.lines(x1, width=0.5, indices=mesh.edge_indices, color=(0., 0., 0.))
    # scene.lines(sim.p, width=0.5, indices=debug_edge_indices, color=(1., 0., 0.))
    # scene.lines(static_mesh.mesh.verts.x, width=0.5, indices=static_mesh.edge_indices, color=(0., 0., 0.))
    canvas.scene(scene)
    window.show()
    # window.save_image(f'results/test/{step:06d}.jpg')
