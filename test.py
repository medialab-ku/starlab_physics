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
gravity = -5
dt = 0.001  # Larger dt might lead to unstable results.
substeps = 60

mesh_path_list = ["obj_models/cube.obj", "obj_models/cube.obj"]
mesh_scale_list = [0.05, 0.05]
mesh_pos_list = [vec(0.5, 1.0, 0.5), vec(0.5, 0.7, 0.5)]
mesh_num = len(mesh_path_list)

num_verts = ti.i32
mesh_list = []
mesh_num_vert_list = []
total_indices_list = []

@ti.dataclass
class Grain:
    p: vec  # Prev Position
    x: vec  # Curr Position
    g: vec
    h: ti.f32
    m: ti.f32  # Mass
    r: ti.f32  # Radius
    y: vec
    v: vec  # Velocity
    a: vec  # Acceleration
    f: vec  # Force

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

    for e in mesh.edges:
        e.l0 = (e.verts[0].p - e.verts[1].p).norm()

@ti.kernel
def rotate(mesh: ti.template(), rot_rad: ti.math.vec3):
    for v in mesh.verts:
        v_4d = ti.Vector([v.p[0], v.p[1], v.p[2], 1])
        rv = ti.math.rotation3d(rot_rad[0], rot_rad[1], rot_rad[2]) @ v_4d
        v.p = ti.Vector([rv[0], rv[1], rv[2]])

@ti.kernel
def initEdges(mesh: ti.template()):
    for e in mesh.edges:
        e.l0 = (e.verts[0].p - e.verts[1].p).norm()

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

    initEdges(mesh_obstacle)


def setMeshes():
    for i in range(mesh_num):
        initMesh(mesh_path_list[i])
        scale(mesh_scale_list[i], mesh_list[i])
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

total_verts_np = np.zeros((num_verts, 3), dtype=np.float32)
for i in range(mesh_num):
    total_verts_np[i * mesh_num_vert_list[i]:(i + 1) * mesh_num_vert_list[i]] = mesh_list[i].verts.p.to_numpy()

# No duplicated vertices
assert len(np.unique(total_verts_np, axis=0)) == len(total_verts_np), "duplicated vertices"

gf = Grain.field(shape=(num_verts, ))
total_verts = ti.field(dtype=ti.f32, shape=(num_verts * 3,))
total_verts.from_numpy(total_verts_np.reshape(-1))

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
    for i in gf:
        gf[i].p = vec(total_verts[i*3+0], total_verts[i*3+1], total_verts[i*3+2])
        gf[i].r = grain_r
        gf[i].m = density * math.pi * gf[i].r ** 2
        gf[i].x = gf[i].p
        gf[i].y = gf[i].p
        gf[i].v = vec(0., 0., 0.)
        gf[i].a = vec(0., 0., 0.)
        gf[i].f = vec(0., 0., 0.)


@ti.kernel
def computeExternalForce():
    for i in gf:
        gf[i].f = vec(0., gravity * gf[i].m, 0)


@ti.kernel
def compute_y(dt: ti.f32):
    for i in gf:
        gf[i].y = gf[i].p + gf[i].v * dt + (gf[i].f / gf[i].m) * dt * dt
        gf[i].x = gf[i].y

damping_factor = 0.01

@ti.kernel
def applyDamping(damping_factor: ti.f32):
    for i in gf:
        gf[i].v = (1 - damping_factor) * gf[i].v



@ti.kernel
def set_gf_info_to_mesh(mesh: ti.template(), mesh_start: ti.i32, mesh_end: ti.i32):
    for i in gf:
        if i >= mesh_start and i < mesh_end:
            mesh.verts.p[i - mesh_start] = gf[i].p
            mesh.verts.x[i - mesh_start] = gf[i].x
            mesh.verts.g[i - mesh_start] = gf[i].g
            mesh.verts.h[i - mesh_start] = gf[i].h
            mesh.verts.v[i - mesh_start] = gf[i].v
            mesh.verts.a[i - mesh_start] = gf[i].a
            mesh.verts.f[i - mesh_start] = gf[i].f

k = 1e3
@ti.kernel
def solveStretch(mesh: ti.template()):
    # ti.loop_config(block_dim=self.block_size)
    for e in mesh.edges:
        v0, v1 = e.verts[0], e.verts[1]
        v0.x = mesh.verts.x[v0.id]
        v1.x = mesh.verts.x[v1.id]
        n = v0.x - v1.x
        d = n.norm()
        coeff = dt * dt * k
        f = coeff * (d - e.l0) * n.normalized(1e-12)
        # print('v0: ', v0.id, 'v0 x: ', v0.x, 'v1: ', v1.id, 'v1 x: ', v1.x, 'f: ', f)

        mesh.verts.g[v0.id] += f
        mesh.verts.g[v1.id] -= f

        mesh.verts.h[v0.id] += coeff
        mesh.verts.h[v1.id] += coeff

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
def set_mesh_info_to_gf(mesh: ti.template(), mesh_offset: ti.i32):
    for v in mesh.verts:
        gf[v.id + mesh_offset].p = v.p
        gf[v.id + mesh_offset].x = v.x
        gf[v.id + mesh_offset].g = v.g
        gf[v.id + mesh_offset].h = v.h
        gf[v.id + mesh_offset].v = v.v
        gf[v.id + mesh_offset].a = v.a
        gf[v.id + mesh_offset].f = v.f


@ti.kernel
def compute_gradient_and_hessian():
    for i in gf:
        gf[i].g = gf[i].m * (gf[i].x - gf[i].y)
        gf[i].h = gf[i].m

    grid_particles_count.fill(0)
    for i in gf:
        grid_idx = ti.floor(gf[i].p * grid_n, int)
        # print(grid_idx, grid_particles_count[grid_idx])
        ti.append(grid_particles_list.parent(), grid_idx, i)
        ti.atomic_add(grid_particles_count[grid_idx], 1)

    # Fast collision detection
    for i in gf:
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



@ti.kernel
def update_state():
    for i in gf:
        gf[i].x -= (gf[i].g / gf[i].h)

@ti.kernel
def computeNextState(dt: ti.f32):
    for i in gf:
        gf[i].v = (gf[i].x - gf[i].p) / dt
        gf[i].p = gf[i].x




@ti.kernel
def update():
    for i in gf:
        gf[i].f = vec(0., gravity * gf[i].m, 0)  # Apply gravity.
        a = gf[i].f / gf[i].m
        gf[i].v += (gf[i].a + a) * dt / 2.0
        gf[i].p += gf[i].v * dt + 0.5 * a * dt**2
        gf[i].a = a


@ti.kernel
def apply_bc():
    bounce_coef = 0.2  # Velocity damping
    for i in gf:
        x = gf[i].x[0]
        y = gf[i].x[1]
        z = gf[i].x[2]

        if z - gf[i].r < 0:
            gf[i].x[2] = gf[i].r
            # v.v[2] *= -bounce_coef

        elif z + gf[i].r > 1.0:
            gf[i].x[2] = 1.0 - gf[i].r
            # v.v[2] *= -bounce_coef

        if y - gf[i].r < 0:
            gf[i].x[1] = gf[i].r
            # v.v[1] *= -bounce_coef

        elif y + gf[i].r > 1.0:
            gf[i].x[1] = 1.0 - gf[i].r
            # v.v[1] *= -bounce_coef

        if x - gf[i].r < 0:
            gf[i].x[0] = gf[i].r
            # v.v[0] *= -bounce_coef

        elif x + gf[i].r > 1.0:
            gf[i].x[0] = 1.0 - gf[i].r
            # v.v[0] *= -bounce_coef


# TODO: combine mesh & primitive
@ti.func
def resolve(i, j):
    rel_pos = gf[i].x - gf[j].x
    dist = ti.sqrt(rel_pos[0]**2 + rel_pos[1]**2 + rel_pos[2]**2)
    delta = -dist + gf[i].r + gf[j].r  # delta = d - 2 * r
    if delta > 0:  # in contact
        normal = rel_pos / dist
        f1 = normal * delta * stiffness
        # # Damping force
        # M = (mesh_obstacle.verts.m[i] * mesh_obstacle.verts.m[j]) / (mesh_obstacle.verts.m[i] + mesh_obstacle.verts.m[j])
        # K = stiffness
        # C = 2. * (1. / ti.sqrt(1. + (math.pi / ti.log(restitution_coef))**2)) * ti.sqrt(K * M)
        # V = (mesh_obstacle.verts.v[j] - mesh_obstacle.verts.v[i]) * normal
        # f2 = C * V * normal
        gf[i].g -= dt * dt * f1
        gf[j].g += dt * dt * f1
        gf[i].h += dt * dt * stiffness
        gf[j].h += dt * dt * stiffness


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
        # print(gf.g)
        for i in range(mesh_num):
            set_gf_info_to_mesh(mesh_list[i], sum(mesh_num_vert_list[:i]), sum(mesh_num_vert_list[:i + 1]))
            solveStretch(mesh_list[i])
            set_mesh_info_to_gf(mesh_list[i], sum(mesh_num_vert_list[:i]))
        # print(gf.h)
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
    scene.particles(gf.p, radius= grain_r, color=(0.5, 0.5, 0.5))
    # scene.particles(primitive_mesh.verts.p, radius= grain_r, color=(0.5, 0.5, 0.5))
    # for i, mesh in enumerate(mesh_list):
    #     scene.mesh(mesh.verts.p, total_indices_list[i], color=(0.2, 0.3, 0.8))
    # scene.mesh(primitive_mesh.verts.p, p_indices, color=(0.5, 0.5, 0.5))

    canvas.scene(scene)
    window.show()
    # window.save_image(f'results/test/{step:06d}.jpg')
