import taichi as ti
import meshtaichi_patcher as patcher
import math
import os

ti.init(arch=ti.cuda)
vec = ti.math.vec3

SAVE_FRAMES = False

window_size = 1024  # Number of pixels of the window
n = 9000  # Number of grains

density = 100.0
stiffness = 8e2
restitution_coef = 0.001
gravity = -5.81
dt = 0.001  # Larger dt might lead to unstable results.
substeps = 60


mesh_obstacle = patcher.load_mesh("obj_models/bunny.obj", relations=["EV", "FV", "EF"])
mesh_obstacle.verts.place({'p': ti.math.vec3,
                           'x': ti.math.vec3,
                           'g': ti.math.vec3,
                           'h': ti.f32,
                           'm': ti.f32,
                           'r': ti.f32,
                           'y': ti.math.vec3,
                           'v': ti.math.vec3,
                           'a': ti.math.vec3,
                           'f': ti.math.vec3}, reorder=False)

mesh_obstacle.edges.place({'l0': ti.f32
                                        }, reorder=False)

mesh_obstacle.verts.p.from_numpy(mesh_obstacle.get_position_as_numpy())
num_verts = len(mesh_obstacle.verts)
@ti.dataclass
class Grain:
    p: vec  # Position
    m: ti.f32  # Mass
    r: ti.f32  # Radius
    v: vec  # Velocity
    a: vec  # Acceleration
    f: vec  # Force

gf = Grain.field(shape=(n, ))

grid_n = 64
grid_size = 5.0 / grid_n  # Simulation domain of size [0, 1]
print(f"Grid size: {grid_n}x{grid_n}x{grid_n}")

grain_r = 0.01

assert grain_r * 2 < grid_size

region_height = n / 10
padding = 0.2
region_width = 1.0 - padding * 2

dt = 0.001

@ti.kernel
def translate(trans: ti.math.vec3):
    for v in mesh_obstacle.verts:
        v.p += trans

@ti.kernel
def set_to_center():
    center = mesh_obstacle.verts.p[0] - mesh_obstacle.verts.p[0]
    for v in mesh_obstacle.verts:
        center += v.p

    center /= num_verts
    for v in mesh_obstacle.verts:
        v.p = v.p - center

@ti.kernel
def scale(scale_factor: ti.float32):
    for v in mesh_obstacle.verts:
        v.p = scale_factor * v.p


@ti.kernel
def init():
    for v in mesh_obstacle.verts:
        # Spread grains in a restricted area.
        # h = i // region_height
        # sq = i % region_height
        # l = sq * grid_size

        #  all random
        # pos = vec(0 + ti.random() * 1, ti.random() * 0.3, ti.random() * 1)

        # v.p = pos
        v.r = grain_r
        v.m = density * math.pi * v.r ** 2

    for e in mesh_obstacle.edges:
        e.l0 = (e.verts[0].p - e.verts[1].p).norm()

@ti.kernel
def computeExternalForce():
    for v in mesh_obstacle.verts:
        v.f = vec(0., gravity * v.m, 0)


@ti.kernel
def compute_y(dt: ti.f32):
    for v in mesh_obstacle.verts:
        v.y = v.p + v.v * dt + (v.f / v.m) * dt * dt
        v.x = v.y

damping_factor = 0.01

@ti.kernel
def applyDamping(damping_factor: ti.f32):
    for v in mesh_obstacle.verts:
        v.v = (1 - damping_factor) * v.v

k = 1e4
@ti.kernel
def solveStretch():
    # ti.loop_config(block_dim=self.block_size)
    ti.mesh_local(mesh_obstacle.verts.g, mesh_obstacle.verts.x)
    for e in mesh_obstacle.edges:
        v0, v1 = e.verts[0], e.verts[1]
        n = v0.x - v1.x
        d = n.norm()
        coeff = dt * dt * k
        f = coeff * (d - e.l0) * n.normalized(1e-12)
        v0.g += f
        v1.g -= f

        v0.h += coeff
        v1.h += coeff

        # w1, w2 = v0.invM, v1.invM
        # if w1 + w2 > 0.:
        #     n = v0.new_x - v1.new_x
        #     d = n.norm()
        #     dp = ti.zero(n)
        #     constraint = (d - e.rest_len)
        #     # if ti.static(self.XPBD):  # https://matthias-research.github.io/pages/publications/XPBD.pdf
        #     #     compliance = e.stretch_compliance / (dt ** 2)
        #     #     d_lambda = -(constraint + compliance * e.la_s) / (
        #     #                 w1 + w2 + compliance) * self.stretch_relaxation  # eq. (18)
        #     #     dp = d_lambda * n.normalized(1e-12)  # eq. (17)
        #     #     e.la_s += d_lambda
        #     else:  # https://matthias-research.github.io/pages/publications/posBasedDyn.pdf
        #         dp = -constraint / (w1 + w2) * n.normalized(1e-12) * self.stretch_relaxation  # eq. (1)
        # v0.dp += dp * w1
        # v1.dp -= dp * w2

@ti.kernel
def compute_gradient_and_hessian():

    for v in mesh_obstacle.verts:
        v.g = v.m * (v.x - v.y)
        v.h = v.m



    grid_particles_count.fill(0)
    for v in mesh_obstacle.verts:
        grid_idx = ti.floor(v.p * grid_n, int)
        # print(grid_idx, grid_particles_count[grid_idx])
        ti.append(grid_particles_list.parent(), grid_idx, int(v.id))
        ti.atomic_add(grid_particles_count[grid_idx], 1)

    # Fast collision detection
    for v in mesh_obstacle.verts:
        grid_idx = ti.floor(v.p * grid_n, int)
        x_begin = max(grid_idx[0] - 1, 0)
        x_end = min(grid_idx[0] + 2, grid_n)

        y_begin = max(grid_idx[1] - 1, 0)
        y_end = min(grid_idx[1] + 2, grid_n)

        z_begin = max(grid_idx[2] - 1, 0)

        # only need one side
        z_end = min(grid_idx[2] + 1, grid_n)

        # todo still serialize
        for neigh_i, neigh_j, neigh_k in ti.ndrange((x_begin, x_end), (y_begin, y_end), (z_begin, z_end)):

            # on split plane
            if neigh_k == grid_idx[2] and (neigh_i + neigh_j) > (grid_idx[0] + grid_idx[1]) and neigh_i <= grid_idx[0]:
                continue
            # same grid
            iscur = neigh_i == grid_idx[0] and neigh_j == grid_idx[1] and neigh_k == grid_idx[2]
            for l in range(grid_particles_count[neigh_i, neigh_j, neigh_k]):
                j = grid_particles_list[neigh_i, neigh_j, neigh_k, l]

                if iscur and v.id >= j:
                    continue
                resolve(v.id, j)


@ti.kernel
def update_state():
    for v in mesh_obstacle.verts:
        v.x -= (v.g / v.h)

@ti.kernel
def computeNextState(dt: ti.f32):
    for v in mesh_obstacle.verts:
        v.v = (v.x - v.p) / dt
        v.p = v.x




@ti.kernel
def update():
    for i in range(num_verts):
        mesh_obstacle.verts.f[i] = vec(0., gravity * mesh_obstacle.verts.m[i], 0)  # Apply gravity.
        a = mesh_obstacle.verts.f[i] / mesh_obstacle.verts.m[i]
        mesh_obstacle.verts.v[i] += (mesh_obstacle.verts.a[i] + a) * dt / 2.0
        mesh_obstacle.verts.p[i] += mesh_obstacle.verts.v[i] * dt + 0.5 * a * dt**2
        mesh_obstacle.verts.a[i] = a


@ti.kernel
def apply_bc():
    bounce_coef = 0.2  # Velocity damping
    for v in mesh_obstacle.verts:
        x = v.x[0]
        y = v.x[1]
        z = v.x[2]

        if z - v.r < 0:
            v.x[2] = v.r
            # v.v[2] *= -bounce_coef

        elif z + v.r > 1.0:
            v.x[2] = 1.0 - v.r
            # v.v[2] *= -bounce_coef

        if y - v.r < 0:
            v.x[1] = v.r
            # v.v[1] *= -bounce_coef

        elif y + v.r > 1.0:
            v.x[1] = 1.0 - v.r
            # v.v[1] *= -bounce_coef

        if x - v.r < 0:
            v.x[0] = v.r
            # v.v[0] *= -bounce_coef

        elif x + v.r > 1.0:
            v.x[0] = 1.0 - v.r
            # v.v[0] *= -bounce_coef


@ti.func
def resolve(i, j):
    rel_pos = mesh_obstacle.verts.x[i] - mesh_obstacle.verts.x[j]
    dist = ti.sqrt(rel_pos[0]**2 + rel_pos[1]**2 + rel_pos[2]**2)
    delta = -dist + mesh_obstacle.verts.r[i] + mesh_obstacle.verts.r[j]  # delta = d - 2 * r
    if delta > 0:  # in contact
        normal = rel_pos / dist
        f1 = normal * delta * stiffness
        # # Damping force
        # M = (mesh_obstacle.verts.m[i] * mesh_obstacle.verts.m[j]) / (mesh_obstacle.verts.m[i] + mesh_obstacle.verts.m[j])
        # K = stiffness
        # C = 2. * (1. / ti.sqrt(1. + (math.pi / ti.log(restitution_coef))**2)) * ti.sqrt(K * M)
        # V = (mesh_obstacle.verts.v[j] - mesh_obstacle.verts.v[i]) * normal
        # f2 = C * V * normal
        mesh_obstacle.verts.g[i] -= dt * dt * f1
        mesh_obstacle.verts.g[j] += dt * dt * f1
        mesh_obstacle.verts.h[i] += dt * dt * stiffness
        mesh_obstacle.verts.h[j] += dt * dt * stiffness


grid_particles_list = ti.field(ti.i32)
grid_block = ti.root.dense(ti.ijk, (grid_n, grid_n, grid_n))
partical_array = grid_block.dynamic(ti.l, n)
partical_array.place(grid_particles_list)

grid_particles_count = ti.field(ti.i32)
ti.root.dense(ti.ijk, (grid_n, grid_n, grid_n)).place(grid_particles_count)


set_to_center()
scale(4)
translate(ti.math.vec3(0.5, 0.5, 0.5))
init()
window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024), vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
step = 0

if SAVE_FRAMES:
    os.makedirs('output', exist_ok=True)

while window.running:
    for s in range(substeps):
        computeExternalForce()
        compute_y(dt)

        ti.deactivate_all_snodes()
        compute_gradient_and_hessian()
        solveStretch()
        update_state()
        apply_bc()
        computeNextState(dt)
        applyDamping(damping_factor)
    camera.position(3, 2, 3)
    camera.lookat(0.5, 0.5, 0.5)
    camera.fov(30)
    camera.up(0, 1, 0)
    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.3, 0.3, 0.3))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.3, 0.3, 0.3))
    scene.particles(mesh_obstacle.verts.p, radius= grain_r, color=(0.5, 0.5, 0.5))
    canvas.scene(scene)
    window.show()