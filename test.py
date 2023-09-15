import taichi as ti
import meshtaichi_patcher as patcher
from meshtaichi_patcher_core import read_tetgen
import numpy as np
import math
import os

from mesh import TotalMesh

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

mesh_path_list = ["obj_models/cube.obj", "obj_models/cube.obj"]
mesh_scale_list = [1.0, 1.0]
mesh_pos_list = [vec(0.5, 3.0, 0.5), vec(0.5, 0.5, 0.5)]
mesh_num = len(mesh_path_list)

num_verts = ti.i32
mesh_list = []
mesh_num_vert_list = []
total_indices_list = []

total_mesh = TotalMesh()

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
def translate(mesh: ti.template(), trans: ti.math.vec3):
    for v in mesh.verts:
        v.p += trans

'''
@ti.kernel
def set_to_center():
    center = mesh_obstacle.verts.p[0] - mesh_obstacle.verts.p[0]
    for v in mesh_obstacle.verts:
        center += v.p

    center /= mesh_num_verts
    for v in mesh_obstacle.verts:
        v.p = v.p - center
'''

@ti.kernel
def scale(scale_factor: ti.float32, mesh: ti.template()):
    for v in mesh.verts:
        v.p = scale_factor * v.p

@ti.kernel
def rotate(mesh: ti.template(), rot_rad: ti.math.vec3):
    for v in mesh.verts:
        v_4d = ti.Vector([v.p[0], v.p[1], v.p[2], 1])
        rv = ti.math.rotation3d(rot_rad[0], rot_rad[1], rot_rad[2]) @ v_4d
        v.p = ti.Vector([rv[0], rv[1], rv[2]])

def initMesh(mesh_path):
    mesh_obstacle = patcher.load_mesh(mesh_path, relations=["EV", "FV", "EF"])
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

    mesh_obstacle.edges.place({'l0': ti.f32}, reorder=False)

    mesh_obstacle.verts.p.from_numpy(mesh_obstacle.get_position_as_numpy())
    mesh_num_vert_list.append(len(mesh_obstacle.verts))
    mesh_list.append(mesh_obstacle)


def setMeshes():
    for i in range(mesh_num):
        initMesh(mesh_path_list[i])
        translate(mesh_list[i], mesh_pos_list[i])

setMeshes()

total_indices_len = 0
for i in range(mesh_num):
    total_indices_len += len(mesh_list[i].faces) * 3
mesh_indices = ti.field(dtype=ti.u32, shape=(total_indices_len))

@ti.kernel
def initIndices(offset: int, mesh: ti.template()):
    for f in mesh.faces:
        mesh_indices[offset + f.id * 3 + 0] = f.verts[0].id
        mesh_indices[offset + f.id * 3 + 1] = f.verts[1].id
        mesh_indices[offset + f.id * 3 + 2] = f.verts[2].id

offset = 0
for i in range(mesh_num):
    initIndices(offset, mesh_list[i])
    offset += len(mesh_list[i].faces) * 3


num_verts = sum(mesh_num_vert_list)
total_verts = ti.field(dtype=ti.u32, shape=(num_verts, 3))

total_verts_np = np.zeros((num_verts, 3), dtype=np.float32)
for i in range(mesh_num):
    total_verts_np[i * mesh_num_vert_list[i]:(i + 1) * mesh_num_vert_list[i]] = mesh_list[i].verts.p.to_numpy()

# No duplicated vertices
assert len(np.unique(total_verts_np, axis=0)) == len(total_verts_np), "duplicated vertices"

total_verts.from_numpy(total_verts_np)


@ti.kernel
def init():
    '''
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
    '''
    for t_v in total_verts:
        t_v.r = grain_r
        t_v.m = density * math.pi * t_v.r ** 2


@ti.kernel
def computeExternalForce():
    for v in total_verts.verts:
        v.f = vec(0., gravity * v.m, 0)


@ti.kernel
def compute_y(dt: ti.f32):
    for v in total_verts.verts:
        v.y = v.p + v.v * dt + (v.f / v.m) * dt * dt
        v.x = v.y

damping_factor = 0.01

@ti.kernel
def applyDamping(damping_factor: ti.f32):
    for v in total_verts.verts:
        v.v = (1 - damping_factor) * v.v

k = 1e4
@ti.kernel
def solveStretch(mesh: ti.template()):
    # ti.loop_config(block_dim=self.block_size)
    ti.mesh_local(mesh.verts.g, mesh.verts.x)
    for e in mesh.edges:
        v0, v1 = e.verts[0], e.verts[1]
        n = v0.x - v1.x
        d = n.norm()
        coeff = dt * dt * k
        f = coeff * (d - e.l0) * n.normalized(1e-12)
        v0.g += f
        v1.g -= f

        v0.h += coeff
        v1.h += coeff

        '''
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
        '''


@ti.kernel
def compute_gradient_and_hessian(mesh: ti.template()):
    for v in mesh.verts:
        v.g = v.m * (v.x - v.y)
        v.h = v.m

    grid_particles_count.fill(0)
    for v in mesh.verts:
        grid_idx = ti.floor(v.p * grid_n, int)
        # print(grid_idx, grid_particles_count[grid_idx])
        ti.append(grid_particles_list.parent(), grid_idx, int(v.id))
        ti.atomic_add(grid_particles_count[grid_idx], 1)

    # Fast collision detection
    for v in mesh.verts:
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
def update_state(mesh: ti.template()):
    for v in mesh.verts:
        v.x -= (v.g / v.h)

@ti.kernel
def computeNextState(mesh: ti.template(), dt: ti.f32):
    for v in mesh.verts:
        v.v = (v.x - v.p) / dt
        v.p = v.x




@ti.kernel
def update(mesh: ti.template()):
    for i in range(len(mesh.verts)):
        mesh.verts.f[i] = vec(0., gravity * mesh.verts.m[i], 0)  # Apply gravity.
        a = mesh.verts.f[i] / mesh.verts.m[i]
        mesh.verts.v[i] += (mesh.verts.a[i] + a) * dt / 2.0
        mesh.verts.p[i] += mesh.verts.v[i] * dt + 0.5 * a * dt**2
        mesh.verts.a[i] = a


@ti.kernel
def apply_bc():
    bounce_coef = 0.2  # Velocity damping
    for v in total_verts.verts:
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


# TODO: combine mesh & primitive
@ti.func
def resolve(i, j):
    rel_pos = total_verts.verts.x[i] - total_verts.verts.x[j]
    dist = ti.sqrt(rel_pos[0]**2 + rel_pos[1]**2 + rel_pos[2]**2)
    delta = -dist + total_verts.verts.r[i] + total_verts.verts.r[j]  # delta = d - 2 * r
    if delta > 0:  # in contact
        normal = rel_pos / dist
        f1 = normal * delta * stiffness
        # # Damping force
        # M = (mesh_obstacle.verts.m[i] * mesh_obstacle.verts.m[j]) / (mesh_obstacle.verts.m[i] + mesh_obstacle.verts.m[j])
        # K = stiffness
        # C = 2. * (1. / ti.sqrt(1. + (math.pi / ti.log(restitution_coef))**2)) * ti.sqrt(K * M)
        # V = (mesh_obstacle.verts.v[j] - mesh_obstacle.verts.v[i]) * normal
        # f2 = C * V * normal
        total_verts.verts.g[i] -= dt * dt * f1
        total_verts.verts.g[j] += dt * dt * f1
        total_verts.verts.h[i] += dt * dt * stiffness
        total_verts.verts.h[j] += dt * dt * stiffness


grid_particles_list = ti.field(ti.i32)
grid_block = ti.root.dense(ti.ijk, (grid_n, grid_n, grid_n))
partical_array = grid_block.dynamic(ti.l, n)
partical_array.place(grid_particles_list)

grid_particles_count = ti.field(ti.i32)
ti.root.dense(ti.ijk, (grid_n, grid_n, grid_n)).place(grid_particles_count)


# set_to_center()
# scale(4, mesh_obstacle)
# translate(ti.math.vec3(0.5, 0.6, 0.5), mesh_obstacle)

# scale(0.05, primitive_mesh)
# translate(ti.math.vec3(0.5, 0.2, 0.5), primitive_mesh)
init()
window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 768), vsync=True)
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
    step += 1
    camera.position(3, 2, 3)
    camera.lookat(0.5, 0.5, 0.5)
    camera.fov(30)
    camera.up(0, 1, 0)
    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.3, 0.3, 0.3))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.3, 0.3, 0.3))
    # scene.particles(mesh_obstacle.verts.p, radius= grain_r, color=(0.5, 0.5, 0.5))
    # scene.particles(primitive_mesh.verts.p, radius= grain_r, color=(0.5, 0.5, 0.5))
    for i, mesh in enumerate(mesh_list):
        scene.mesh(mesh.verts.p, total_indices_list[i], color=(0.2, 0.3, 0.8))
    # scene.mesh(primitive_mesh.verts.p, p_indices, color=(0.5, 0.5, 0.5))

    canvas.scene(scene)
    window.show()
    # window.save_image(f'results/test/{step:06d}.jpg')
