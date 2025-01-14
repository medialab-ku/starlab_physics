import random
import meshio
import networkx as nx
from pyquaternion.quaternion import Quaternion
import taichi as ti
from timeit import default_timer as timer
import numpy as np
import matplotlib as mpl

ti.init(arch=ti.gpu)

window = ti.ui.Window("PBD framework", (1024, 768), fps_limit=200)
gui = window.get_gui()
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = window.get_scene()
camera = ti.ui.Camera()
camera.position(2., 4.0, 7.0)
camera.fov(40)
camera.up(0, 1, 0)

run_sim = False

num_iters = 500
frame_cnt = 0
threshold = 2
has_end = False
end_frame = 200
PR = 6
k_at = 5
solver_type = 0
enable_pncg = False
enable_attachment = True
print_stat = False
enable_detection = False
enable_lines = False
enable_lines2 = False


def import_mesh(path, scale, translate, rotate):

    mesh = meshio.read(path)
    scale_lf = lambda x, sc: sc * x
    trans_lf = lambda x, trans: x + trans

    # rotate, scale, and translate all vertices
    x_np_temp = np.array(mesh.points, dtype=float)
    center = x_np_temp.sum(axis=0) / x_np_temp.shape[0]  # center position of the mesh
    x_np_temp = np.apply_along_axis(lambda row: trans_lf(row, -center), 1, x_np_temp)  # translate to origin
    x_np_temp = scale_lf(x_np_temp, scale)  # scale mesh to the particular ratio
    rot_quaternion = Quaternion(axis=[rotate[0], rotate[1], rotate[2]], angle=rotate[3])
    rot_matrix = rot_quaternion.rotation_matrix
    for j in range(x_np_temp.shape[0]):
        x_np_temp[j] = rot_matrix @ x_np_temp[j]

    x_np_temp = np.apply_along_axis(lambda row: trans_lf(row, center), 1,x_np_temp)  # translate back to the original position
    x_np_temp = np.apply_along_axis(lambda row: trans_lf(row, translate), 1, x_np_temp)  # translate again to the designated position!

    edges_tmp = np.empty((0, 2), dtype=int)
    faces = np.empty((0, 3), dtype=int)
    for face in mesh.cells_dict["triangle"]:
        edges_tmp = np.append(edges_tmp, sorted([face[0], face[1]]))
        edges_tmp = np.append(edges_tmp, sorted([face[1], face[2]]))
        edges_tmp = np.append(edges_tmp, sorted([face[2], face[0]]))
        faces = np.append(faces, sorted([face[0], face[1], face[2]]))

    faces = faces.reshape(-1, 3)
    edges_tmp = np.reshape(edges_tmp, (-1, 2))
    _, indxs, count = np.unique(edges_tmp, axis=0, return_index=True, return_counts=True)
    # surface_edges = edges_tmp[indxs[count==1]]
    edges = edges_tmp[indxs]
    # surface_edges.astype(int)
    # # print(surface_edges[0, 1])
    # boundary_vertices = set()
    # for i in range(surface_edges.shape[0]):
    #     boundary_vertices.add(surface_edges[i, 0])
    #     boundary_vertices.add(surface_edges[i, 1])
    #
    # boundary_vertices = np.array(list(boundary_vertices))

    return x_np_temp, edges, faces

x_np_temp, edges_np, faces_np = import_mesh("models/OBJ/plane.obj",  scale = 3.0, translate = [0.0, 0.0, 0.0], rotate = [1., 0., 0., 0.0])

num_particles = x_np_temp.shape[0]

num_edges = edges_np.shape[0]

adj_list = [[] for i in range(num_particles)]

edges_list = edges_np.tolist()
for i in range(num_edges):
    adj_list[edges_list[i][0]].append(edges_list[i][1])
    adj_list[edges_list[i][1]].append(edges_list[i][0])

partitioned_eid_set = []
partitioned_vid_set = []
num_edges_per_partition = 100
# num_partition = 2S

a = 0
for i in range(num_particles):
    a += len(adj_list[i])

while a > 0:
# for t in range(num_partition):
    st = -1
    while True:
        st = random.randint(0, num_particles - 1)

        if len(adj_list[st]) > 0:
            break

    pick = [st]
    partitioned_edges = []
    while len(pick) < num_edges_per_partition + 1:
        head = pick[-1]
        cnt = len(adj_list[head])
        if cnt > 0:
            test = 0
            for i in range(cnt):
                a = adj_list[head][i]
                test += 1
                if not a in pick:
                    pick.append(a)
                    adj_list[head].remove(a)
                    adj_list[a].remove(head)
                    break

            if test == cnt:
                break

        else:
            break

    if len(pick) > 1:
        partitioned_vid_set.append(pick)

    a = 0
    for i in range(num_particles):
        a += len(adj_list[i])

num_partition = len(partitioned_vid_set)


# print([len(i) for i in partitioned_vid_set])

# print(num_partition

edges_tmp = []

cnt = 0
for i in range(num_partition):
    eid = []
    for j in range(len(partitioned_vid_set[i]) - 1):
        a = sorted([partitioned_vid_set[i][j], partitioned_vid_set[i][j + 1]])
        edges_tmp.append(a)
        eid.append(cnt)
        cnt += 1

    partitioned_eid_set.append(eid)
edges_np = np.array(edges_tmp)
# print(edges_np)

num_edges = cnt
v_e = [[] for i in range(num_particles)]
for e in edges_tmp:
    v_e[e[0]].append(e[1])
    v_e[e[1]].append(e[0])

rel = []
cnt = 0

pid_set = [i for i in range(num_partition)]

non_overlapping_pairs = []

adj_list_part = [[] for i in range(num_partition)]

for i in range(num_partition - 1):
    for j in range(i + 1, num_partition):
        if len(set(partitioned_vid_set[i]) & set(partitioned_vid_set[j])) > 0:
            adj_list_part[i].append(j)
            adj_list_part[j].append(i)

# print(adj_list_part)

def greedy_graph_coloring(graph):
    # Number of vertices
    n = len(graph)

    # Store colors of vertices (-1 means no color assigned)
    colors = [-1] * n

    # Assign colors to vertices
    for vertex in range(n):
        # Find colors used by neighbors
        neighbor_colors = {colors[neighbor] for neighbor in graph[vertex] if colors[neighbor] != -1}

        # Find the smallest color not used by neighbors
        color = 0
        while color in neighbor_colors:
            color += 1

        # Assign this color to the current vertex
        colors[vertex] = color

    return colors

colors = greedy_graph_coloring(adj_list_part)
print(colors)


print("# partition: ", num_partition)
# print(rel)

# G = nx.Graph(list(rel), dtype=int)
# colors = nx.greedy_color(G)
# a = list(colors.values())

# print(a)

unique_elements, counts = np.unique(np.array(colors), return_counts=True)
num_colors = len(unique_elements)

pid_per_color = [[] for i in range(num_colors)]

print(num_colors)

# eid_per_color = [set() for i in range(num_colors)]
for i in range(num_partition):
    ci = colors[i]
    pid_per_color[ci].append(i)
    # for j in range(len(partitioned_vid_set[i])):
    #     vi = partitioned_vid_set[i][j]
    #     for k in range(len(v_e[vi])):
    #         eid_per_color[ci].add(v_e[vi][k])

num_partition_per_color_tmp  = [len(i) for i in pid_per_color]
# t = 0
# for c in range(num_colors):
#     t += num_partition_per_color_tmp[c]
#     for i in range(num_partition_per_color_tmp[c] - 1):
#         pi = pid_per_color[c][i]
#         for j in range(i + 1, num_partition_per_color_tmp[c]):
#             pj = pid_per_color[c][j]
#             if len(set(partitioned_vid_set[pi]) & set(partitioned_vid_set[pj])) > 0:
#                 # print(partitioned_vid_set[pi])
#                 # print(partitioned_vid_set[pj])
#                 print("test")

# print(t)

max_num_partition_per_color = max(num_partition_per_color_tmp)
# print(max_num_partition_per_color)
num_partition_per_color_np = np.array(num_partition_per_color_tmp)
num_partition_per_color_ti = ti.field(shape=num_colors, dtype=int)
num_partition_per_color_ti.from_numpy(num_partition_per_color_np)

pid_per_color_np = np.zeros(shape=(num_colors, max_num_partition_per_color), dtype=int)
for i in range(num_colors):
    for j in range(num_partition_per_color_np[i]):
        pid_per_color_np[i][j] = pid_per_color[i][j]

pid_per_color_ti = ti.field(shape=pid_per_color_np.shape, dtype=int)
pid_per_color_ti.from_numpy(pid_per_color_np)

num_edges_per_partition = np.array([len(i) for i in partitioned_eid_set])
num_edges_per_partition_ti = ti.field(shape=num_edges_per_partition.shape, dtype=int)
num_edges_per_partition_ti.from_numpy(num_edges_per_partition)

max_num_edges_per_partition = max(num_edges_per_partition)
partitioned_eid_set_np = np.zeros(shape=(num_partition, max_num_edges_per_partition), dtype=int)
partitioned_vid_set_np = np.zeros(shape=(num_partition, max_num_edges_per_partition + 1), dtype=int)

for i in range(num_partition):
    for j in range(num_edges_per_partition[i]):
        partitioned_eid_set_np[i][j] = partitioned_eid_set[i][j]
        partitioned_vid_set_np[i][j] = partitioned_vid_set[i][j]

    partitioned_vid_set_np[i][num_edges_per_partition[i]] = partitioned_vid_set[i][num_edges_per_partition[i]]

partitioned_eid_set_ti = ti.field(dtype=int, shape=partitioned_eid_set_np.shape)
partitioned_eid_set_ti.from_numpy(partitioned_eid_set_np)

partitioned_vid_set_ti = ti.field(dtype=int, shape=partitioned_vid_set_np.shape)
partitioned_vid_set_ti.from_numpy(partitioned_vid_set_np)

# print(edges_np)
# print(partitioned_eid_set_ti)
# print(partitioned_eid_set_np)
# print(partitioned_vid_set_ti)
# print(partitioned_vid_set_np)

edges = ti.field(shape=2 * num_edges, dtype=int)
edges.from_numpy(edges_np.reshape(-1))

# print(edges)
# print(partitioned_vid_set_ti)

num_faces = faces_np.shape[0]
faces = ti.field(shape=3 * num_faces, dtype=int)
faces.from_numpy(faces_np.reshape(-1))

# print(num_partition)

# num_max_vertices_per_partition = num_max_edges_per_partition + 1

l0 = ti.field(float, shape=2 * num_edges)
dt = 0.005
x    = ti.Vector.field(n=3, dtype=float, shape=num_particles)
mass = ti.field(dtype=float, shape=num_particles)
x_k  = ti.Vector.field(n=3, dtype=float, shape=num_particles)
x_k_minus_1 = ti.Vector.field(n=3, dtype=float, shape=num_particles)
y  = ti.Vector.field(n=3, dtype=float, shape=num_particles)
x0 = ti.Vector.field(n=3, dtype=float, shape=num_particles)
v  = ti.Vector.field(n=3, dtype=float, shape=num_particles)

grad   = ti.Vector.field(n=3, dtype=float, shape=num_particles)
P_grad = ti.Vector.field(n=3, dtype=float, shape=num_particles)
dx     = ti.Vector.field(n=3, dtype=float, shape=num_particles)

grad_k   = ti.Vector.field(n=3, dtype=float, shape=num_particles)
P_grad_k = ti.Vector.field(n=3, dtype=float, shape=num_particles)

grad_delta   = ti.Vector.field(n=3, dtype=float, shape=num_particles)
P_grad_delta = ti.Vector.field(n=3, dtype=float, shape=num_particles)

dx_k     = ti.Vector.field(n=3, dtype=float, shape=num_particles)

hii = ti.Matrix.field(n=3, m=3, dtype=float, shape=num_particles)
hij = ti.Matrix.field(n=3, m=3, dtype=float, shape=num_edges)

a_part       = ti.Matrix.field(n=3, m=3, dtype=float, shape=(num_partition, max_num_edges_per_partition + 1))
b_part       = ti.Matrix.field(n=3, m=3, dtype=float, shape=(num_partition, max_num_edges_per_partition + 1))
c_part       = ti.Matrix.field(n=3, m=3, dtype=float, shape=(num_partition, max_num_edges_per_partition + 1))
c_tilde_part = ti.Matrix.field(n=3, m=3, dtype=float, shape=(num_partition, max_num_edges_per_partition + 1))

d_part       = ti.Vector.field(n=3, dtype=float, shape=(num_partition, max_num_edges_per_partition + 1))
d_tilde_part = ti.Vector.field(n=3, dtype=float, shape=(num_partition, max_num_edges_per_partition + 1))
dx_part      = ti.Vector.field(n=3, dtype=float, shape=(num_partition, max_num_edges_per_partition + 1))

num_dup = ti.field(float, shape=num_particles)

collision_set = ti.field(int, shape=num_particles)
num_colliding_vertices = ti.field(int, shape=1)
K = ti.linalg.SparseMatrixBuilder(3 * num_particles, 3 * num_particles, max_num_triplets=10000000)
ndarr = ti.ndarray(ti.f32, shape=3 * num_particles)

# @ti.func
# def flattend_index(i: int, j: int):
#
#     return i + num_particles_x * j
#
#
# @ti.kernel
# def generate_particles(num_particles: int, delta: float):
#
#     for i in range(num_particles):
#         xi = i % num_particles_x
#         yi = i // num_particles_x
#         x0[i] = -ti.Vector([xi, 0, yi]) * delta
#
#     mass.fill(1.0)


@ti.kernel
def centeralize(x: ti.template()):

    center = ti.math.vec3(0.0)
    for i in x:
        center += x[i]

    center /= num_particles

    for i in range(num_particles):
        x0[i] -= center


@ti.kernel
def generate_indices4():

    density = 1.0
    for i in range(num_edges):
        v0, v1 = edges[2 * i + 0], edges[2 * i + 1]
        l0[i] = (x0[v0] - x0[v1]).norm()
        mass[v0] += 0.5 * l0[i] * density
        mass[v1] += 0.5 * l0[i] * density

        num_dup[v0] += 1.0
        num_dup[v1] += 1.0

    nx = ti.math.vec3([1.0, 0.0, 0.0])
    nz = ti.math.vec3([0.0, 0.0, 1.0])
    scale = 5.0
    x0[0] += scale * (-nx + nz)
    x0[1] += scale * (nx + nz)
    x0[2] += scale * (-nx - nz)
    x0[3] += scale * (nx - nz)

    # ti.loop_config(serialize=True)
    # for pi in range(num_max_partition):
    #     size_pi = num_edges_per_partition[pi]
    #
    #     ti.loop_config(serialize=True)
    #     for i in range(size_pi):
    #         ei = partitioned_set[pi, i]
    #         vi = indices[2 * ei + 0]
    #         # print(vi)
    #         num_dup[vi] += 1.0
    #     ei = partitioned_set[pi, size_pi - 1]
    #     vi = indices[2 * ei + 1]
    #     num_dup[vi] += 1.0
    #
    # for i in range(num_particles):
    #     if num_dup[i] < 1:
    #         mass[i] = 1.0

# @ti.kernel
def generate_mesh():

    x0.from_numpy(x_np_temp)
    # edges_flattened = edges.reshape(-1)
    # indices.from_numpy(edges_flattened)
    # print(indices)
    # generate_particles(num_particles, delta=1.0)
    centeralize(x0)

    mass.fill(0.0)
    num_dup.fill(0.0)
    generate_indices4()

    # print(num_dup)

    # print(num_edges_per_partition)


@ti.kernel
def compute_y():

    # apply gravity within boundary
    # g = ti.Vector([0.0, -9.81, 0.0])
    g = ti.Vector([0.0, 0.0, 0.0])
    for i in x:
        xi, vi = x[i], v[i]
        x_k[i] = y[i] = xi + vi * dt + g * dt * dt

@ti.kernel
def compute_v(damping: float):
    for i in x:
        v[i] = (1.0 - damping) * (x_k[i] - x[i])/ dt
        x[i] = x_k[i]

    # cnt = 0.0
    # for i in range(num_edges):
    #     v0, v1 = edges[2 * i + 0], edges[2 * i + 1]
    #     # print(v0, v1)
    #     x01 = x[v0] - x[v1]
    #     n = x01.normalized()
    #     if n.dot(v[v0] - v[v1]) > 0.0:
    #         cnt += 1.0
    #
    # print(cnt)


@ti.kernel
def compute_grad_and_hessian_momentum(x: ti.template()):

    id3 = ti.Matrix.identity(dt=float, n=3)
    for i in range(num_particles):
        grad[i] = mass[i] * (x[i] - y[i])
        hii[i] = mass[i] * id3

@ti.kernel
def compute_e_momentum(x: ti.template()) -> ti.f32:

    e = 0.0
    for i in range(num_particles):
        e += 0.5 * mass[i] * (x[i] - y[i]).dot((x[i] - y[i]))

    return e


@ti.kernel
def compute_grad_and_hessian_momentum_part(x: ti.template(), pi: int):

    id3 = ti.Matrix.identity(dt=float, n=3)
    for j in range(num_edges_per_partition_ti[pi] + 1):
        i = partitioned_vid_set_ti[pi, j]
        grad[i] = mass[i] * (x[i] - y[i])
        hii[i] = mass[i] * id3

@ti.kernel
def compute_grad_and_hessian_attachment(x: ti.template(), k: float):

    id3 = ti.Matrix.identity(dt=float, n=3)
    # ids = ti.Vector([i for i in range(num_particles_x)], dt=int)
    ids = ti.Vector([0, 1, 2, 3], dt=int)
    # print(ids)
    for i in range(ids.n):
        grad[ids[i]] += k * (x[ids[i]] - x0[ids[i]])
        hii[ids[i]] += k * id3


@ti.kernel
def compute_e_attachment(x: ti.template(), k: float) -> ti.f32:
    e = 0.0
    ids = ti.Vector([0, 1, 2, 3], dt=int)
    # print(ids)
    for i in range(ids.n):
        e += 0.5 * k * (x[ids[i]] - x0[ids[i]]).dot((x[ids[i]] - x0[ids[i]]))

    return e

@ti.kernel
def detect_collision(x: ti.template(), radius: float, center: ti.math.vec3):

    num_colliding_vertices.fill(0)
    for i in range(num_particles):
        ri = (x[i] - center).norm()
        if ri < radius:
            collision_set[ti.atomic_add(num_colliding_vertices[0], 1)] = i

@ti.kernel
def collision_aware_line_search(x: ti.template(), dx:ti.template(), radius: float, center: ti.math.vec3) -> float:

    alpha = 1.0
    # num_colliding_vertices.fill(0)
    for i in range(num_colliding_vertices[0]):
        id = collision_set[i]
        xi_temp = x[id] - dx[id]
        ri = (xi_temp - center).norm()
        if ri < radius:
            dx_norm = dx[id].norm()
            test = dx_norm - (ri - radius)
            alpha_tmp = test / dx_norm
            ti.atomic_min(alpha, alpha_tmp)

    return alpha

@ti.kernel
def compute_grad_and_hessian_collision(x: ti.template(), radius: float, center: ti.math.vec3, k: float):

    id3 = ti.Matrix.identity(dt=float, n=3)
    for i in range(num_colliding_vertices[0]):

        id = collision_set[i]
        ri = (x[id] - center).norm()
        if ri < radius:
            ni = (x[id] - center).normalized()
            pi = center + ni * radius
            grad[id] += k * (x[id] - pi)
            hii[id] += k * id3

@ti.kernel
def compute_grad_and_hessian_spring(x: ti.template(), k: float):

    for i in range(num_edges):
        v0, v1 = edges[2 * i + 0], edges[2 * i + 1]
        # print(v0, v1)
        x01 = x[v0] - x[v1]
        n = x01.normalized()
        l = x01.norm()
        dp01 = x01 - l0[i] * n

        grad[v0] += k * dp01
        grad[v1] -= k * dp01

        n = x01.normalized()
        alpha = abs(1.0 - l0[i] / l)
        tmp = ti.math.vec3(n[0] + ti.random(float), n[1] + ti.random(float), n[2] + ti.random(float))
        t1 = ti.math.normalize(n.cross(tmp))
        t2 = n.cross(t1)
        D = ti.math.mat3([k, 0.0, 0.0, 0.0, k * alpha, 0.0, 0.0, 0.0, k * alpha])
        P = ti.math.mat3([n[0], t1[0], t2[0], n[1], t1[1], t2[1], n[2], t1[2], t2[2]])
        B = (P @ D @ P.inverse())

        hii[v0] += B
        hii[v1] += B
        hij[i] = B

@ti.kernel
def compute_e_spring(x: ti.template(), k: float) -> ti.f32:
    e = 0.0
    for i in range(num_edges):
        v0, v1 = edges[2 * i + 0], edges[2 * i + 1]
        # print(v0, v1)
        x01 = x[v0] - x[v1]
        l = x01.norm()
        e += 0.5 * k * (l - l0[i]) ** 2

    return e


@ti.kernel
def compute_grad_and_hessian_spring_part(x: ti.template(), pi: int, k: float):

    for j in range(num_edges_per_partition_ti[pi]):
        i = partitioned_eid_set_ti[pi, j]
        v0, v1 = edges[2 * i + 0], edges[2 * i + 1]
        # print(v0, v1)
        x01 = x[v0] - x[v1]
        n = x01.normalized()
        l = x01.norm()
        dp01 = x01 - l0[i] * n

        grad[v0] += k * dp01
        grad[v1] -= k * dp01

        n = x01.normalized()
        alpha = abs(1.0 - l0[i] / l)
        tmp = ti.math.vec3(n[0] + ti.random(float), n[1] + ti.random(float), n[2] + ti.random(float))
        t1 = ti.math.normalize(n.cross(tmp))
        t2 = n.cross(t1)
        D = ti.math.mat3([k, 0.0, 0.0, 0.0, k * alpha, 0.0, 0.0, 0.0, k * alpha])
        P = ti.math.mat3([n[0], t1[0], t2[0], n[1], t1[1], t2[1], n[2], t1[2], t2[2]])
        B = (P @ D @ P.inverse())

        hii[v0] += B
        hii[v1] += B
        hij[i] = B

# @ti.func
# def BlockThomasAlgorithm(a: ti.template(), b: ti.template(), c: ti.template(), x: ti.template(), d: ti.template()):

@ti.kernel
def substep_Euler(Px: ti.template(), x: ti.template()):

    for i in range(num_edges):
        v0, v1 = edges[2 * i + 0], edges[2 * i + 1]
        tmp = hij[i] @ hii[v1].inverse()
        S = hii[v0] - tmp @ hij[i]
        Px0 = S.inverse() @ (x[v0] - tmp @ x[v1])
        Px1 = hii[v1].inverse() @ (x[v1] - hij[i] @ Px0)
        Px[v0] += Px0
        Px[v1] += Px1

        # Px[v0] += hii[v0].inverse() @ x[v0]
        # Px[v1] += hii[v1].inverse() @ x[v1]

    for i in range(num_particles):
        Px[i] /= num_dup[i]

@ti.kernel
def substep_test_attach(x: ti.template()):
    ids = ti.Vector([0, 1], dt=int)
    ti.loop_config(serialize=True)
    for a in range(ids.n):
        x[ids[a]] = x0[ids[a]]

@ti.kernel
def substep_GS(x: ti.template(), k: float):

    id3 = ti.Matrix.identity(dt=float, n=3)

    ti.loop_config(serialize=True)
    for i in range(num_edges):
        v0, v1 = edges[2 * i + 0], edges[2 * i + 1]
        x01 = x[v0] - x[v1]
        n = x01.normalized()
        l = x01.norm()
        dp01 = x01 - l0[i] * n

        grad0 = + k * dp01
        grad1 = - k * dp01

        n = x01.normalized()
        alpha = abs(1.0 - l0[i] / l)
        B = compute_Hessian(n, alpha)
        B *= k

        hii0 = mass[v0] * id3 - B
        hii1 = mass[v1] * id3 - B

        A = B @ hii1.inverse()
        S = hii0 - A @ B
        Px0 = S.inverse() @ (grad0 - A @ grad1)
        Px1 = hii1.inverse() @ (grad1 - B @ Px0)

        x[v0] -= Px0
        x[v1] -= Px1

@ti.func
def compute_Hessian(n, alpha):

    tmp = ti.math.vec3(n[0] + ti.random(float), n[1] + ti.random(float), n[2] + ti.random(float))
    t1 = ti.math.normalize(n.cross(tmp))
    t2 = n.cross(t1)
    D = ti.math.mat3([1.0, 0.0, 0.0, 0.0, alpha, 0.0, 0.0, 0.0, alpha])
    P = ti.math.mat3([n[0], t1[0], t2[0], n[1], t1[1], t2[1], n[2], t1[2], t2[2]])
    B = -(P @ D @ P.inverse())

    return B

@ti.kernel
def substep_Euler_GS(Px: ti.template(), ci: int, x: ti.template()):

    # ti.loop_config(serialize=True)
    for c in range(num_partition_per_color_ti[ci]):

        pi = pid_per_color_ti[ci, c]
        e_verts_pi = num_edges_per_partition_ti[pi]


        n_verts_pi = e_verts_pi + 1
        ti.loop_config(serialize=True)
        for i in range(e_verts_pi):
            ei = partitioned_eid_set_ti[pi, i]
            # print(ei)
            vi, vj = partitioned_vid_set_ti[pi, i], partitioned_vid_set_ti[pi, i + 1]
            # print(vi, vj)
            b_part[pi, i] = hii[vi]
            b_part[pi, i + 1] = hii[vj]
            a_part[pi, i + 1] = -hij[ei]
            c_part[pi, i] = -hij[ei]

            d_part[pi, i] = x[vi]
            d_part[pi, i + 1] = x[vj]

        c_tilde_part[pi, 0] = ti.math.inverse(b_part[pi, 0]) @ c_part[pi, 0]
        d_tilde_part[pi, 0] = ti.math.inverse(b_part[pi, 0]) @ d_part[pi, 0]

        ti.loop_config(serialize=True)
        for i in range(1, n_verts_pi - 1):
            tmp = ti.math.inverse(b_part[pi, i] - a_part[pi, i] @ c_tilde_part[pi, i - 1])
            c_tilde_part[pi, i] = tmp @ c_part[pi, i]
            d_tilde_part[pi, i] = tmp @ (d_part[pi, i] - a_part[pi, i] @ d_tilde_part[pi, i - 1])

        tmp = ti.math.inverse(b_part[pi, n_verts_pi - 1] - a_part[pi, n_verts_pi - 1] @ c_tilde_part[pi, n_verts_pi - 2])
        d_tilde_part[pi, n_verts_pi - 1] = tmp @ (d_part[pi, n_verts_pi - 1] - a_part[pi, n_verts_pi - 1] @ d_tilde_part[pi, n_verts_pi - 2])
        dx_part[pi, n_verts_pi - 1] = d_tilde_part[pi, n_verts_pi - 1]

        ti.loop_config(serialize=True)
        for i in range(n_verts_pi - 1):
            idx = n_verts_pi - 2 - i
            dx_part[pi, idx] = d_tilde_part[pi, idx] - c_tilde_part[pi, idx] @ dx_part[pi, idx + 1]
        #
        ti.loop_config(serialize=True)
        for i in range(n_verts_pi):
            vi = partitioned_vid_set_ti[pi, i]
            Px[vi] -= dx_part[pi, i]


@ti.kernel
def substep_Jacobi_GS(Px: ti.template(), ci: int, x: ti.template()):

    # ti.loop_config(serialize=True)
    for c in range(num_partition_per_color_ti[ci]):
        vi = partitioned_vid_set_ti[ci, i]
        Px[vi] -= x[vi]


@ti.kernel
def substep_Jacobi(Px: ti.template(), x: ti.template()):

    for i in range(num_particles):

        test = hii[i]
        U, sigma, V = ti.svd(test, dt=ti.f32)
        if sigma[0 ,0] * sigma[1 ,1] * sigma[2 ,2] < 0.0:
            print("test")

        Px[i] = hii[i].inverse() @ x[i]

@ti.kernel
def construct_Hessian(A: ti.types.sparse_matrix_builder()):

    for i in range(num_particles):
        B = hii[i]
        for m, n in ti.ndrange(3, 3):
            A[3 * i + m, 3 * i + n] += B[m, n]

    for i in range(num_edges):
        v0, v1 = edges[2 * i + 0], edges[2 * i + 1]
        B = hij[i]
        for m, n in ti.ndrange(3, 3):
            A[3 * v1 + m, 3 * v0 + n] += B[m, n]
            A[3 * v0 + m, 3 * v1 + n] += B[m, n]

@ti.kernel
def ndarr_to_vec_field(ndarr: ti.types.ndarray(), vec_field: ti.template()):

    for i in vec_field:
        vec_i = ti.math.vec3([ndarr[3 * i + 0], ndarr[3 * i + 1], ndarr[3 * i + 2]])
        vec_field[i] = vec_i

@ti.kernel
def vec_field_to_ndarr(vec_field: ti.template(), ndarr: ti.types.ndarray()):

    for i in vec_field:
        vec_i = vec_field[i]
        ndarr[3 * i + 0] = vec_i[0]
        ndarr[3 * i + 1] = vec_i[1]
        ndarr[3 * i + 2] = vec_i[2]

def substep_Newton(Px, x):
    construct_Hessian(K)
    H = K.build()
    solver = ti.linalg.SparseSolver(solver_type="LLT")
    solver.analyze_pattern(H)

    vec_field_to_ndarr(x, ndarr)
    Px_ndarr = solver.solve(ndarr)
    ndarr_to_vec_field(Px_ndarr, Px)

@ti.kernel
def add(ret: ti.template(), x: ti.template(), y: ti.template(), scale: float):
    for i in x:
        ret[i] = x[i] + y[i] * scale

@ti.kernel
def scale(ret: ti.template(), x: ti.template(), scale: float):
    for i in x:
        ret[i] = scale * x[i]

@ti.kernel
def dot(x: ti.template(), y: ti.template()) -> ti.f32:

    value = 0.0
    for i in x:
        value += x[i].dot(y[i])

    return value

@ti.kernel
def infinity_norm(x: ti.template()) -> ti.f32:

    value = 0.0
    for i in x:
        ti.atomic_max(value, x[i].norm())

    return value

@ti.kernel
def compute_beta(g:ti.template(), Py:ti.template(), y:ti.template(), p:ti.template()) -> ti.f32:

    g_Py = 0.0
    y_Py = 0.0
    p_g =0.0
    y_p = 0.0

    for i in  g:
        g_Py += g[i].dot(Py[i])
        y_Py += y[i].dot(Py[i])
        p_g  += p[i].dot(g[i])
        y_p  += y[i].dot(p[i])

    # print(g_Py, y_Py, p_g, y_p)

    return (g_Py - (y_Py / y_p) * p_g)/y_p

radius = 2.0
center = ti.Vector.field(n=3, shape=(1), dtype=ti.f32)
center[0] = ti.math.vec3(0.0, -3.5, 0.0)

def forward():

    compute_y()
    k = pow(10.0, PR) * dt ** 2
    k_col = pow(10.0, PR) * dt ** 2
    k_at = 5 * pow(10.0, PR) * dt ** 2
    termination_condition = pow(10.0, -threshold)
    itr_cnt = 0
    # print(PR)
    # radius = 2.0
    # center = ti.math.vec3(0.0, -3.5, 0.0)

    # if solver_type < 3:
    #     for _ in range(num_iters):
    #         # print(it)
    #         compute_grad_and_hessian_momentum(x_k)
    #         compute_grad_and_hessian_spring(x_k, k=k)
    #
    #         if enable_attachment:
    #             compute_grad_and_hessian_attachment(x_k, k=k_at)
    #
    #         if enable_detection:
    #             detect_collision(x_k, radius, center[0])
    #             compute_grad_and_hessian_collision(x_k, radius, center[0], k=k_col)
    #
    #         if solver_type == 0:
    #             substep_Euler(P_grad, grad)
    #
    #         if solver_type == 1:
    #             substep_Jacobi(P_grad, grad)
    #
    #         elif solver_type == 2:
    #             substep_Newton(P_grad, grad)
    #
    #         beta = 0.0
    #
    #         if itr_cnt > 0 and enable_pncg:
    #
    #             add(grad_delta, grad, grad_k, -1.0)
    #             add(P_grad_delta, P_grad, P_grad_k, -1.0)
    #             beta = compute_beta(grad, P_grad_delta, grad_delta, dx_k)
    #
    #         # print(beta)
    #         add(dx, P_grad, dx_k, -beta)
    #
    #         # alpha = 1.0
    #         # alpha_ccd = collision_aware_line_search(x_k, dx, radius, center)
    #         # print(alpha_ccd)
    #
    #         # scale(dx, dx, alpha_ccd)
    #         add(x_k, x_k, dx, -1.0)
    #         inf_norm = infinity_norm(dx)
    #
    #         dx_k.copy_from(dx)
    #         grad_k.copy_from(grad)
    #         P_grad_k.copy_from(P_grad)
    #
    #         itr_cnt += 1
    #         if print_stat:
    #             print(inf_norm)
    #
    #         if inf_norm < termination_condition:
    #
    #             if print_stat:
    #                 print("conv iter: ,", itr_cnt)
    #             break
    # else:

    e = 0.0
    e += compute_e_momentum(x_k)
    e += compute_e_spring(x_k, k=k)
    if enable_attachment:
        e += compute_e_attachment(x_k, k=k_at)

    print(e)

    for _ in range(num_iters):
        # for i in range(num_edges):
        x_k_minus_1.copy_from(x_k)

        for ci in range(num_colors):
            compute_grad_and_hessian_momentum(x_k)
            # compute_grad_and_hessian_momentum_part(x_k, i)
            compute_grad_and_hessian_spring(x_k, k=k)
            if enable_attachment:
                compute_grad_and_hessian_attachment(x_k, k=k_at)

            # for i in range(num_edges):
            # substep_GS(x_k,  k)
            substep_Euler_GS(x_k, ci, grad)

        add(dx, x_k, x_k_minus_1, -1.0)

        inf_norm = infinity_norm(dx)
        print(inf_norm)
        # e = 0.0
        # e += compute_e_momentum(x_k)
        # e += compute_e_spring(x_k, k=k)
        # if enable_attachment:
        #     e += compute_e_attachment(x_k, k=k_at)
        #
        # print(e)

        # add(dx, P_grad, dx_k, 0.0)

        # alpha = 1.0
        # alpha_ccd = collision_aware_line_search(x_k, dx, radius, center)
        # print(alpha_ccd)

        # scale(dx, dx, alpha_ccd)
        # add(x_k, x_k, P_grad, -1.0)

    # print(itr_cnt)
    compute_v(0.01)

    return itr_cnt

# debug = partition_total

def show_options():
    global num_iters
    global threshold
    global solver_type
    global PR
    global frame_cnt
    global print_stat
    global enable_pncg
    global enable_detection
    global enable_attachment
    global has_end
    global end_frame
    global enable_lines
    global enable_lines2
    global show_partition_id

    PR_old = PR

    with gui.sub_window("XPBD Settings", 0., 0., 0.3, 0.7) as w:

        solver_type = w.slider_int("solver type", solver_type, 0, 3)
        if solver_type == 0:
            w.text("Euler")
        elif solver_type == 1:
            w.text("Jacobi")
        elif solver_type == 2:
            w.text("Newton")
        elif solver_type == 3:
            w.text("Gauss-Seidel")

        num_iters = w.slider_int("max iter.", num_iters, 1, 10000)
        threshold = w.slider_int("threshold", threshold, 1, 6)
        PR_old = w.slider_int("stiffness", PR_old, 1, 8)
        # PR_old = w.slider_int("a", PR_old, 1, 8)
        enable_pncg = w.checkbox("enable PNCG", enable_pncg)
        enable_detection = w.checkbox("enable detection", enable_detection)
        enable_attachment = w.checkbox("enable attachment", enable_attachment)
        has_end = w.checkbox("has_end", has_end)
        print_stat = w.checkbox("print_stats", print_stat)
        enable_lines = w.checkbox("enable lines", enable_lines)
        enable_lines2 = w.checkbox("enable lines2", enable_lines2)

        # show_partition_id = w.slider_int("partition id", show_partition_id, 0, len(debug) - 1)

        if not PR_old == PR:
            PR = PR_old

        if has_end:
            end_frame = w.slider_int("end frame", end_frame, 0, 1000)

        w.text("")
        frame_str = "# frame " + str(frame_cnt)
        w.text(frame_str)

        verts_str = "# verts: " + str(num_particles)
        edges_str = "# edges: " + str(num_edges)
        w.text(verts_str)
        w.text(edges_str)

generate_mesh()
x.copy_from(x0)
conv_it_total = 0
elapsed_time_total = 0

while window.running:

    # if LOOKAt_ORIGIN:
    camera.lookat(0.0, 0.0, 0.0)

    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.3, 0.3, 0.3))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.3, 0.3, 0.3))

    if window.get_event(ti.ui.PRESS):

        if window.event.key == ' ':
            run_sim = not run_sim
            already_backward = False

        if window.event.key == 'r':
            x.copy_from(x0)
            v.fill(0.0)
            if frame_cnt > 0:
                if solver_type == 0:
                    print("Euler")
                elif solver_type == 1:
                    print("Jacobi")
                elif solver_type == 2:
                    print("Newton")
                print("Avg. iter    : ", int(conv_it_total / frame_cnt))
                print("Avg. time[ms]: ", round(100.0 * (elapsed_time_total / frame_cnt), 2))

            conv_it_total = 0
            frame_cnt = 0
            elapsed_time_total = 0
            run_sim = False

        if run_sim is False and window.event.key == 'o':
            start = timer()
            conv_itr = forward()
            end = timer()

            conv_it_total += conv_itr
            elapsed_time = (end - start)
            elapsed_time_total += elapsed_time
            frame_cnt += 1

    # color_partition(show_partition_id, debug)
    # colors.from_numpy(colors_np)

    if run_sim:

        start = timer()
        conv_itr = forward()
        end = timer()

        conv_it_total += conv_itr
        elapsed_time = (end - start)
        elapsed_time_total += elapsed_time
        frame_cnt += 1

    if has_end and frame_cnt >= end_frame:
        run_sim = False

    show_options()

    scene.particles(x, radius=0.05, color=(0.0, 0.0, 0.0))
    # scene.particles(center, radius=radius, color=(1.0, 0.5, 0.0))
    # scene.lines(x, indices=edges, color=(0.0, 0.0, 1.0), width=1.0)
    # scene.mesh(x, indices=faces, color=(0.0, 0.0, 0.0), show_wireframe=True)

    if enable_lines:
        # scene.mesh(x, indices=faces, color=(0.0, 0.0, 0.0), show_wireframe=True)
        scene.lines(x, indices=edges, color=(0.0, 0.0, 1.0), width=1.0)
    # if enable_lines2:
    #     scene.lines(x, indices=indices, color=(1.0, 0.0, 0.0), width=1.0)

    camera.track_user_inputs(window, movement_speed=0.4, hold_key=ti.ui.RMB)
    canvas.scene(scene)
    window.show()