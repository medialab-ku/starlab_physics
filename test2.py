import taichi as ti
from my_mesh import Mesh
from my_solver import Solver

ti.init(arch=ti.cuda, device_memory_fraction=0.95)
vec = ti.math.vec3

SAVE_FRAMES = False

window_size = 1024  # Number of pixels of the window
dt = 0.01  # Larger dt might lead to unstable results.

# mesh = Mesh("obj_models/triangle.obj", scale=0.4, rot=ti.math.vec3(90.0, 0.0, 0.0),trans=ti.math.vec3(0.3, 0.2, 0.3))
#mesh = Mesh("obj_models/cube.obj", scale=0.1, rot=ti.math.vec3(90.0, 0.0, 0.0),trans=ti.math.vec3(0.3, 0.5, 0.3))

#mesh = Mesh("tet_models/bunny_small.mesh", scale=0.1, rot=ti.math.vec3(0.0, 0.0, 0.0))
# static_mesh = Mesh("obj_models/bunny.obj", scale=0.4, rot=ti.math.vec3(0.0, 0.0, 0.0), trans=ti.math.vec3(0.0, -0.5, 0.0))
# static_mesh =Mesh("obj_models/cube.obj", scale=0.1, rot=ti.math.vec3(90.0, 0.0, 0.0), trans=ti.math.vec3(0.3, 0.2, 0.3))

# case: face vs. vertex
# mesh = Mesh("obj_models/tetrahedron.obj", scale=0.1, rot=ti.math.vec3(180.0, 0.0, 0.0),trans=ti.math.vec3(0.3, 0.5, 0.3))
# static_mesh =Mesh("obj_models/tetrahedron.obj", scale=0.1, rot=ti.math.vec3(180.0, 0.0, 0.0), trans=ti.math.vec3(0.3, 0.2, 0.3))

#case: edge vs. edge
mesh = Mesh("obj_models/tetrahedron.obj", scale=0.1, rot=ti.math.vec3(45.0, 0.0, 0.0),trans=ti.math.vec3(0.3, 0.5, 0.3))
static_mesh =Mesh("obj_models/tetrahedron.obj", scale=0.1, rot=ti.math.vec3(45.0, 0.0, 0.0), trans=ti.math.vec3(0.3, 0.2, 0.3))

#case: face vs. face
# mesh = Mesh("obj_models/tetrahedron.obj", scale=0.1, rot=ti.math.vec3(0.0, 0.0, 0.0),trans=ti.math.vec3(0.3, 0.5, 0.3))
# static_mesh =Mesh("obj_models/tetrahedron.obj", scale=0.1, rot=ti.math.vec3(180.0, 0.0, 45.0), trans=ti.math.vec3(0.3, 0.2, 0.3))



sim = Solver(mesh, bottom=0.0, static_mesh=static_mesh, dt=dt, max_iter=40)

window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 768), vsync=True, fps_limit=200)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()

while window.running:
    sim.update()
    camera.position(1.6, 1, 1.5)
    camera.lookat(0.5, 0.5, 0.5)
    camera.fov(30)
    camera.up(0, 1, 0)
    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.3, 0.3, 0.3))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.3, 0.3, 0.3))
    #scene.particles(sim.nodes.x, radius=sim.radius, color=(0, 1, 0))
    #scene.particles(sim.static_nodes.x, radius=sim.radius, color=(0, 1, 0))
    scene.mesh(sim.nodes.x, mesh.face_indices, color=(0.5, 0.5, 0.5))
    scene.mesh(static_mesh.mesh.verts.x, static_mesh.face_indices, color=(0.5, 0.5, 0.5))
    scene.lines(sim.nodes.x, width=0.5, indices=mesh.edge_indices, color=(0., 0., 0.))
    scene.lines(sim.static_nodes.x, width=0.5, indices=static_mesh.edge_indices, color=(0., 0., 0.))
    canvas.scene(scene)
    window.show()
    # window.save_image(f'results/test/{step:06d}.jpg')
