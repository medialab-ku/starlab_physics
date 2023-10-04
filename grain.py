import taichi as ti
import math
import os

ti.init(arch=ti.gpu)
vec = ti.math.vec3

SAVE_FRAMES = False

window_size = 1024  # Number of pixels of the window
n = 10000  # Number of grains

density = 100.0
stiffness = 8e3
restitution_coef = 0.001
gravity = -9.81
dt = 0.0001  # Larger dt might lead to unstable results.
substeps = 60


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


@ti.kernel
def init():
    for i in gf:
        # Spread grains in a restricted area.
        h = i // region_height
        sq = i % region_height
        l = sq * grid_size

        #  all random
        pos = vec(0 + ti.random() * 1, ti.random() * 0.3, ti.random() * 1)

        gf[i].p = pos
        gf[i].r = grain_r
        gf[i].m = density * math.pi * gf[i].r ** 2


@ti.kernel
def update():
    for i in gf:
        a = gf[i].f / gf[i].m
        gf[i].v += (gf[i].a + a) * dt / 2.0
        gf[i].p += gf[i].v * dt + 0.5 * a * dt**2
        gf[i].a = a


@ti.kernel
def apply_bc():
    bounce_coef = 0.3  # Velocity damping
    for i in gf:
        x = gf[i].p[0]
        y = gf[i].p[1]
        z = gf[i].p[2]

        if z - gf[i].r < 0:
            gf[i].p[2] = gf[i].r
            gf[i].v[2] *= -bounce_coef

        elif z + gf[i].r > 1.0:
            gf[i].p[2] = 1.0 - gf[i].r
            gf[i].v[2] *= -bounce_coef

        if y - gf[i].r < 0:
            gf[i].p[1] = gf[i].r
            gf[i].v[1] *= -bounce_coef

        elif y + gf[i].r > 1.0:
            gf[i].p[1] = 1.0 - gf[i].r
            gf[i].v[1] *= -bounce_coef

        if x - gf[i].r < 0:
            gf[i].p[0] = gf[i].r
            gf[i].v[0] *= -bounce_coef

        elif x + gf[i].r > 1.0:
            gf[i].p[0] = 1.0 - gf[i].r
            gf[i].v[0] *= -bounce_coef


@ti.func
def resolve(i, j):
    rel_pos = gf[j].p - gf[i].p
    dist = ti.sqrt(rel_pos[0]**2 + rel_pos[1]**2 + rel_pos[2]**2)
    delta = -dist + gf[i].r + gf[j].r  # delta = d - 2 * r
    if delta > 0:  # in contact
        normal = rel_pos / dist
        f1 = normal * delta * stiffness
        # Damping force
        M = (gf[i].m * gf[j].m) / (gf[i].m + gf[j].m)
        K = stiffness
        C = 2. * (1. / ti.sqrt(1. + (math.pi / ti.log(restitution_coef))**2)
                  ) * ti.sqrt(K * M)
        V = (gf[j].v - gf[i].v) * normal
        f2 = C * V * normal
        gf[i].f += f2 - f1
        gf[j].f -= f2 - f1



grid_particles_list = ti.field(ti.i32)
grid_block = ti.root.dense(ti.ijk, (grid_n, grid_n, grid_n))
partical_array = grid_block.dynamic(ti.l, n)
partical_array.place(grid_particles_list)
grid_particles_count = ti.field(ti.i32)
ti.root.dense(ti.ijk, (grid_n, grid_n, grid_n)).place(grid_particles_count)

@ti.kernel
def contact(gf: ti.template(), step: int):
    '''
    Handle the collision between grains.
    '''
    for i in gf:
        gf[i].f = vec(0., gravity * gf[i].m, 0)  # Apply gravity.
        # """
        # tougong
        # _toCenter = gf[i].p - vec(0.5, 0.5, 0.5)
        # _norm = _toCenter.norm()
        #
        # if step > 500:
        #     if _norm < 0.1:
        #         gf[i].f += _toCenter.normalized() * (1 - _norm) * gf[i].m * 1000
        #
        # else:
        #     _rotateforce = vec(_toCenter[2], 0, -_toCenter[0]).normalized()
        #     gf[i].f += -_toCenter.normalized() * (1 - _norm) * gf[i].m * 50
        #     gf[i].f += _rotateforce * (0.5 - abs(0.5 - gf[i].p[1])) * gf[i].m * 5
        # # """

    grid_particles_count.fill(0)
    for i in range(n):
        grid_idx = ti.floor(gf[i].p * grid_n, int)
        # print(grid_idx, grid_particles_count[grid_idx])
        ti.append(grid_particles_list.parent(), grid_idx, int(i))
        ti.atomic_add(grid_particles_count[grid_idx], 1)

    # Fast collision detection
    for i in range(n):
        grid_idx = ti.floor(gf[i].p * grid_n, int)
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

                if iscur and i >= j:
                    continue
                resolve(i, j)


init()
window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 1024),
                      vsync=True)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
step = 0

if SAVE_FRAMES:
    os.makedirs('output', exist_ok=True)

while window.running:
    for s in range(substeps):
        update()
        apply_bc()
        ti.deactivate_all_snodes()
        contact(gf, step)
    camera.position(3, 2, 3)
    camera.lookat(0.5, 0.5, 0.5)
    camera.fov(30)
    camera.up(0, 1, 0)
    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.3, 0.3, 0.3))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.3, 0.3, 0.3))
    scene.particles(gf.p, radius=0.01, color=(0.5, 0.5, 0.5))
    canvas.scene(scene)
    window.show()