import taichi as ti
import meshtaichi_patcher as patcher
from meshtaichi_patcher_core import read_tetgen
import numpy as np
import math
import os

from mesh import TotalMesh

ti.init(arch=ti.cuda, device_memory_fraction=0.95)
vec = ti.math.vec3

SAVE_FRAMES = False

window_size = 1024  # Number of pixels of the window

density = 100.0
stiffness = 1e4
restitution_coef = 0.001
gravity = -10
dt = 0.003  # Larger dt might lead to unstable results.
substeps = 3

# mesh_path_list = ["obj_models/cube.obj", "obj_models/cube.obj"]
# mesh_scale_list = [0.1, 0.1]
# mesh_pos_list = [vec(0.5, 0.8, 0.5), vec(0.5, 0.5, 0.5)]
# mesh_rotate_list = [vec(0.0, 0.0, 0.0), vec(0.0, 0.0, 0.0)]

mesh_path_list = ["obj_models/clubbing_dress.obj"]
mesh_scale_list = [0.4]
mesh_pos_list = [vec(0.4, 0.2, 0.3)]
mesh_rotate_list = [vec(90.0, 90.0, 90.0)]

# mesh_path_list = ["obj_models/square_big.obj"]
# mesh_scale_list = [0.04]
# mesh_pos_list = [vec(0.5, 0.5, 0.5)]
# mesh_rotate_list = [vec(0.0, 0.0, 0.0)]

# mesh_path_list = ["seq_models/Kyra_DVStandClubbing/Kyra_DVStandClubbing_0100.obj"]
# mesh_scale_list = [0.4]
# mesh_pos_list = [vec(0.5, 0.2, 0.5)]
# mesh_rotate_list = [vec(90.0, 0.0, 0.0)]
# updating_mesh = [False]

# mesh_path_list = ["seq_models/Kyra_DVStandClubbing/Kyra_DVStandClubbing_0000.obj", "obj_models/cube.obj"]
# mesh_scale_list = [0.4, 0.5]
# mesh_pos_list = [vec(0.5, 0.2, 0.5), vec(0.8, 0.8, 0.8)]
# mesh_rotate_list = [vec(90.0, 0.0, 0.0), vec(0.0, 0.0, 0.0)]
mesh_num = len(mesh_path_list)

num_verts = ti.i32
mesh_list = []
mesh_num_vert_list = []
mesh_num_edges_list = []
mesh_num_faces_list = []
# mesh_sum_vert_list = ti.ndarray(shape=(mesh_num), dtype=ti.i32)
# mesh_sum_edges_list = ti.ndarray(shape=(mesh_num), dtype=ti.i32)
# mesh_sum_faces_list = ti.ndarray(shape=(mesh_num), dtype=ti.i32)
# mesh_sum_vert_list_np = np.empty()
# mesh_sum_edges_list_np = np.empty()
# mesh_sum_faces_list_np = np.empty()

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

grid_n = 128
grid_size = 1.0 / grid_n  # Simulation domain of size [0, 1]
print(f"Grid size: {grid_n}x{grid_n}x{grid_n}")

grain_r = 0.001

# assert grain_r * 2 < grid_size

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
def scale(mesh: ti.template(), scale_factor: ti.float32):
    for v in mesh.verts:
        v.p = scale_factor * v.p

    for e in mesh.edges:
        e.l0 = (e.verts[0].p - e.verts[1].p).norm()

@ti.kernel
def rotate(mesh: ti.template(), rot_deg: ti.math.vec3):
    for v in mesh.verts:
        v_4d = ti.Vector([v.p[0], v.p[1], v.p[2], 1])
        rot_rad = ti.math.radians(rot_deg)
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
    mesh_num_edges_list.append(len(mesh_obstacle.edges))
    mesh_num_faces_list.append(len(mesh_obstacle.faces))

    # mesh_sum_vert_list_np.append(sum(mesh_num_vert_list))
    # mesh_sum_edges_list_np.append(sum(mesh_num_edges_list))
    # mesh_sum_faces_list_np.append(sum(mesh_num_faces_list))
    mesh_list.append(mesh_obstacle)

    initEdges(mesh_obstacle)

def setMeshes():
    for i in range(mesh_num):
        initMesh(mesh_path_list[i])
        scale(mesh_list[i], mesh_scale_list[i])
        rotate(mesh_list[i], mesh_rotate_list[i])
        translate(mesh_list[i], mesh_pos_list[i])

setMeshes()
# mesh_sum_vert_list.from_numpy(mesh_sum_vert_list_np)
# mesh_sum_edges_list.from_numpy(mesh_sum_edges_list_np)
# mesh_sum_faces_list.from_numpy(mesh_sum_faces_list_np)

total_indices_len = 0
total_faces_len = 0
for i in range(mesh_num):
    total_indices_len += len(mesh_list[i].faces) * 3
    total_faces_len += len(mesh_list[i].faces)
mesh_indices = ti.field(dtype=ti.u32, shape=(total_indices_len))

@ti.kernel
def initIndices(offset: int, mesh: ti.template(), vert_offset: ti.i32):
    for f in mesh.faces:
        mesh_indices[offset + f.id * 3 + 0] = f.verts[0].id + vert_offset
        mesh_indices[offset + f.id * 3 + 1] = f.verts[1].id + vert_offset
        mesh_indices[offset + f.id * 3 + 2] = f.verts[2].id + vert_offset

offset = 0
for i in range(mesh_num):
    vert_offset = sum(mesh_num_vert_list[:i])
    initIndices(offset, mesh_list[i], vert_offset)
    offset += len(mesh_list[i].faces) * 3

num_verts = sum(mesh_num_vert_list)
num_edges = sum(mesh_num_edges_list)

print("num_verts:", num_verts, ", num_edges:", num_edges)

total_verts_np = np.zeros((num_verts, 3), dtype=np.float32)
for i in range(mesh_num):
    total_verts_np[i * mesh_num_vert_list[i]:(i + 1) * mesh_num_vert_list[i]] = mesh_list[i].verts.p.to_numpy()

print("vertex range: ", np.min(total_verts_np, axis=0), np.max(total_verts_np, axis=0))
print(total_verts_np)
print(mesh_indices)

# No duplicated vertices
# assert len(np.unique(total_verts_np, axis=0)) == len(total_verts_np), "duplicated vertices"

gf = Grain.field(shape=(num_verts, ))
total_verts = ti.field(dtype=ti.f32, shape=(num_verts * 3,))
total_verts.from_numpy(total_verts_np.reshape(-1))

total_edges = ti.field(dtype=ti.u32, shape=(num_edges * 2, ))
total_edge_l0s = ti.field(dtype=ti.f32, shape=(num_edges, ))
@ti.kernel
def setEdges(mesh: ti.template(), offset: ti.i32, vert_offset: ti.i32):
    for e in mesh.edges:
        total_edges[(offset + e.id) * 2 + 0] = e.verts[0].id + vert_offset
        total_edges[(offset + e.id) * 2 + 1] = e.verts[1].id + vert_offset
        total_edge_l0s[offset + e.id] = e.l0

for i in range(mesh_num):
    offset = sum(mesh_num_edges_list[:i])
    vert_offset = sum(mesh_num_vert_list[:i])
    setEdges(mesh_list[i], offset, vert_offset)

@ti.kernel
def init():
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
def solveStretch():
    # ti.loop_config(block_dim=self.block_size)
    for e in range(num_edges):
        e1, e2 = ti.i32(total_edges[e*2+0]), ti.i32(total_edges[e*2+1])
        v0, v1 = gf[e1], gf[e2]
        n = v0.x - v1.x
        d = n.norm()
        coeff = dt * dt * stiffness
        f = coeff * (d - total_edge_l0s[e]) * n.normalized(1e-12)
        # print('v0 x: ', v0.x, 'v1 x: ', v1.x, 'd: ', d, 'l0: ', total_edge_l0s[e], 'f: ', f)

        gf[e1].g += f
        gf[e2].g -= f

        gf[e1].h += coeff
        gf[e2].h += coeff

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

@ti.func
def compute_triangle_cells():
    for f in range(total_faces_len):
        p1I = mesh_indices[f * 3 + 0]
        p2I = mesh_indices[f * 3 + 1]
        p3I = mesh_indices[f * 3 + 2]

        p1 = gf[p1I].p
        p2 = gf[p2I].p
        p3 = gf[p3I].p

        normal = (p2 - p1).cross(p3 - p1).normalized(1e-12)

        p1g = ti.floor(p1 * grid_n, int)
        p2g = ti.floor(p2 * grid_n, int)
        p3g = ti.floor(p3 * grid_n, int)

        minI = ti.min(p1g, p2g, p3g)
        maxI = ti.max(p1g, p2g, p3g)

        for i, j, k in ti.ndrange((minI[0], maxI[0]+1), (minI[1], maxI[1]+1), (minI[2], maxI[2]+1)):
            p = ti.Vector([(i+0.5)/grid_n, (j+0.5)/grid_n, (k+0.5)/grid_n])
            dist = (p - p1).dot(normal)
            if abs(dist) < (1/grid_n) * 2:
                # print(i, j, k, f, dist)
                ti.append(grid_triangle_list.parent(), ti.Vector([i, j, k]), f)

@ti.kernel
def compute_gradient_and_hessian():
    for i in gf:
        gf[i].g = gf[i].m * (gf[i].x - gf[i].y)
        gf[i].h = gf[i].m

    # grid_particles_count.fill(0)
    # for i in gf:
    #     grid_idx = ti.floor(gf[i].p * grid_n, int)
    #     # print(grid_idx, grid_particles_count[grid_idx])
    #     ti.append(grid_particles_list.parent(), grid_idx, i)
    #     ti.atomic_add(grid_particles_count[grid_idx], 1)

    # compute_triangle_cells()
    #
    # # Fast collision detection
    # for i in gf:
    #     grid_idx = ti.floor(gf[i].p * grid_n, int)
    #     x_begin = max(grid_idx[0] - 1, 0)
    #     x_end = min(grid_idx[0] + 2, grid_n)
    #
    #     y_begin = max(grid_idx[1] - 1, 0)
    #     y_end = min(grid_idx[1] + 2, grid_n)
    #
    #     z_begin = max(grid_idx[2] - 1, 0)
    #
    #     # only need one side
    #     z_end = min(grid_idx[2] + 1, grid_n)
    #
    #     # todo still serialize
    #     for neigh_i, neigh_j, neigh_k in ti.ndrange((x_begin, x_end), (y_begin, y_end), (z_begin, z_end)):
    #
    #         # on split plane
    #         if neigh_k == grid_idx[2] and (neigh_i + neigh_j) > (grid_idx[0] + grid_idx[1]) and neigh_i <= grid_idx[0]:
    #             continue
    #         # same grid
    #         # iscur = neigh_i == grid_idx[0] and neigh_j == grid_idx[1] and neigh_k == grid_idx[2]
    #         # for l in range(grid_particles_count[neigh_i, neigh_j, neigh_k]):
    #         #     j = grid_particles_list[neigh_i, neigh_j, neigh_k, l]
    #         #
    #         #     if iscur and i >= j:
    #         #         continue
    #         #     resolve(i, j)
    #
    #         # collision with triangle
    #         if grid_triangle_list[neigh_i, neigh_j, neigh_k].length() != 0:
    #             for fIdx in range(grid_triangle_list[neigh_i, neigh_j, neigh_k].length()):
    #                 f = grid_triangle_list[neigh_i, neigh_j, neigh_k, fIdx]
    #                 # self_collsion = check_self_collision(i, f)
    #                 if i != mesh_indices[f*3+0] and i != mesh_indices[f*3+1] and i != mesh_indices[f*3+2]:
    #                     # if i is not in the triangle indices
    #                     # print(i, f)
    #                     resolve_triangle(i, f)



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

# @ti.func
# def check_self_collision(i, f):
#     vert_idx, face_idx = -1, -1
#     for idx in range(mesh_num):
#         if mesh_sum_vert_list[idx] <= i < mesh_sum_vert_list[idx+1]:
#             vert_idx = idx
#         if mesh_sum_faces_list[idx] <= f < mesh_sum_faces_list[idx+1]:
#             face_idx = idx
#     if vert_idx == face_idx:
#         return True
#     else:
#         return False

cor = 2e1
@ti.func
def resolve_triangle(i, f):
    f1 = gf[mesh_indices[f*3+0]].x
    f2 = gf[mesh_indices[f*3+1]].x
    f3 = gf[mesh_indices[f*3+2]].x
    point = gf[i].x

    e1 = f2 - f1
    e2 = f3 - f1
    normal = e1.cross(e2).normalized(1e-12)
    d = point - f1
    dist = d.dot(normal)
    point_on_triangle = point - dist * normal
    d1 = point_on_triangle - f1
    d2 = point_on_triangle - f2
    d3 = point_on_triangle - f3

    area_triangle = e1.cross(e2).norm() / 2
    area1 = d1.cross(d2).norm() / (2 * area_triangle)
    area2 = d2.cross(d3).norm() / (2 * area_triangle)
    area3 = d3.cross(d1).norm() / (2 * area_triangle)

    is_on_triangle = 0 <= area1 <= 1 and 0 <= area2 <= 1 and 0 <= area3 <= 1 and area1 + area2 + area3 == 1

    # weights (if the point_on_triangle is close to the vertex, the weight is large)
    w1 = 1 / (d1.norm() + 1e-8)
    w2 = 1 / (d2.norm() + 1e-8)
    w3 = 1 / (d3.norm() + 1e-8)
    total_w = w1 + w2 + w3
    w1 /= total_w
    w2 /= total_w
    w3 /= total_w

    if dist <= 1e-3 and is_on_triangle:
        # print('i:', i, 'f:', f, 'point:', point, 'f1:', f1, 'f2:', f2, 'f3:', f3)
        gf[i].g -= dt * dt * (1e-3 - dist) * normal * cor
        gf[mesh_indices[f*3+0]].g += dt * dt * (1e-3 - dist) * normal * cor * w1
        gf[mesh_indices[f*3+1]].g += dt * dt * (1e-3 - dist) * normal * cor * w2
        gf[mesh_indices[f*3+2]].g += dt * dt * (1e-3 - dist) * normal * cor * w3

        gf[i].h -= dt * dt * cor
        gf[mesh_indices[f*3+0]].h += dt * dt * cor * w1
        gf[mesh_indices[f*3+1]].h += dt * dt * cor * w2
        gf[mesh_indices[f*3+2]].h += dt * dt * cor * w3


grid_particles_list = ti.field(ti.i32)
grid_block = ti.root.dense(ti.ijk, (grid_n, grid_n, grid_n))
particle_array = grid_block.dynamic(ti.l, num_verts)
particle_array.place(grid_particles_list)

grid_particles_count = ti.field(ti.i32)
ti.root.dense(ti.ijk, (grid_n, grid_n, grid_n)).place(grid_particles_count)

triangle_block = ti.root.dense(ti.ijk, (grid_n, grid_n, grid_n))
triangle_idx = triangle_block.dynamic(ti.l, total_faces_len, chunk_size=64)
grid_triangle_list = ti.field(int)
triangle_idx.place(grid_triangle_list)


init()
window = ti.ui.Window("Taichi Cloth Simulation on GGUI", (1024, 768), vsync=True, fps_limit=200)
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
step = 0

if SAVE_FRAMES:
    os.makedirs('output', exist_ok=True)

while window.running:

    computeExternalForce()
    compute_y(dt)
    for s in range(substeps):
        ti.deactivate_all_snodes()
        compute_gradient_and_hessian()
        solveStretch()
        update_state()
        apply_bc()

    computeNextState(dt)
    applyDamping(damping_factor)

    step += 1
    print('step:', step)
    camera.position(2, 2, 2)
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
    # scene.mesh(gf.p, mesh_indices, color=(0.5, 0.5, 0.5))


    canvas.scene(scene)
    window.show()
    # window.save_image(f'results/test/{step:06d}.jpg')
