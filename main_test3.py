import meshio
import networkx as nx
from PIL.ImagePalette import random
from pyquaternion.quaternion import Quaternion
import taichi as ti
from timeit import default_timer as timer
import numpy as np
import matplotlib as mpl

from tqdm import tqdm
import time

from test2 import partitioned_set
from test7 import solve_Jacobi

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

num_iters = 5
frame_cnt = 0
threshold = 2
has_end = False
end_frame = 200
PR = 6
k_at = 5
solver_type = 0
enable_pncg = False
enable_attachment = False
print_stat = False
enable_detection = False
enable_lines = False
enable_lines2 = False

num_particles_x = 151
num_particles_y = 151

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

x_np_temp, edges, faces = import_mesh("models/OBJ/even_plane.obj",  scale = 3.0, translate = [0.0, 0.0, 0.0], rotate = [1., 0., 0., 0.0])

num_particles = x_np_temp.shape[0]
num_max_partition = (num_particles_x - 1) * (num_particles_y - 1)
num_partition = ti.field(int, shape=1)
num_edges = edges.shape[0]

num_partitioned_edges = 0
faces_tmp = faces

boundaries = []
surface_edges = []

while faces_tmp.shape[0] > 0:
    # print(faces_tmp.shape[0])
    edges_tmp = np.empty((0, 2), dtype=int)
    # print(faces_tmp)
    for face in faces_tmp:
        edges_tmp = np.append(edges_tmp, sorted([face[0], face[1]]))
        edges_tmp = np.append(edges_tmp, sorted([face[1], face[2]]))
        edges_tmp = np.append(edges_tmp, sorted([face[2], face[0]]))

    edges_tmp = np.reshape(edges_tmp, (-1, 2))
    _, indxs, count = np.unique(edges_tmp, axis=0, return_index=True, return_counts=True)

    s_edges = edges_tmp[indxs[count==1]]
    s_edges.astype(int)
    # print(surface_edges.shape[0])
    # print(surface_edges[0, 1])
    boundary_vertices = set()
    for i in range(s_edges.shape[0]):
        boundary_vertices.add(s_edges[i, 0])
        boundary_vertices.add(s_edges[i, 1])

    # print(len(boundary_vertices))
    surface_edges.append(s_edges)
    face_test = np.empty((0, 3), dtype=int)

    for face in faces_tmp:
        # print(face)
        fset = set(face)
        # print(fset)
        if len(fset & boundary_vertices) < 1:
            face_test = np.append(face_test, sorted([face[0], face[1], face[2]]))

    # print(faces_tmp.shape[0])
    faces_tmp = np.reshape(face_test, (-1, 3))
    boundaries.append(list(boundary_vertices))


graph = nx.Graph()
graph.add_edges_from(edges)
# print(boundaries)
partition_cycle = []
partition_total = []

# for i in range(len(boundaries)):
for i in range(2):
    graph_tmp = nx.MultiGraph()
    graph_tmp.add_edges_from(surface_edges[i])
    # euler_graph = nx.eulerize(graph_tmp)  # after adding edges, we eulerize
    if nx.is_eulerian(graph_tmp):
        path_tmp = list(nx.eulerian_path(graph_tmp))
        path = []
        for edge in path_tmp:
            path.append(int(edge[0]))
        # path.append(path_tmp[-1][1])

        # a = []

        partition_cycle.append(path)
        partition_total.append(path)
        partition_total.append(path_tmp[-1])

    graph.remove_edges_from(surface_edges[i])

partition_rest = []
#
# for l in range(1):
#     path_test = []
#     for i in range(len(partition_cycle[l]) - 1):
#         src, dest = partition_cycle[l][i], partition_cycle[l][i + 1]
#         if nx.has_path(graph, src, dest):
#             path = nx.shortest_path(graph, src, dest)
#             # print(path)
#             # path_test.append(path)
#             partition_total.append(path)
#             for j in range(len(path) - 1):
#                 graph.remove_edge(path[j], path[j + 1])

    # aa = []
    # while True:
    #     if len(path_test) > 0:
    #         a = path_test.pop(0)
    #         if len(path_test) > 0:
    #             if a[-1] == path_test[0][0]:
    #                 b = path_test.pop(0)
    #                 # print(path_test)
    #                 path_test.append(a[:-1] + b)
    #             else:
    #                 partition_total.append(a)
    #         else:
    #             partition_total.append(a)
    #     else:
    #         break
#
# print(partition_total)
#
# for a in range(1):
#     longest_path_size = 0
#     longest_path = []
#     for i in range(len(partition_cycle[0]) - 1):
#         for j in range(i + 1, len(partition_cycle[0])):
#             src, dest = partition_cycle[0][i], partition_cycle[0][j]
#             if nx.has_path(graph, src, dest):
#                 path = nx.shortest_path(graph, src, dest)
#                 if longest_path_size < len(path):
#                     longest_path_size = len(path)
#                     longest_path = path
#                     # partition_total.append(path)
#                     # for k in range(len(path) - 1):
#                     #     graph.remove_edge(path[k], path[k + 1])
#
#     if longest_path_size > 0:
#         partition_total.append(longest_path)
#         for k in range(len(longest_path) - 1):
#             graph.remove_edge(longest_path[k], longest_path[k + 1])
# #
#     else: break


# for k in range(len(partition_cycle)):
#     set = len(partition_cycle[k]) // 4
#     offset = len(partition_cycle[k]) // 2
#     for i in range(set):
#         src, dest = partition_cycle[k][i], partition_cycle[k][i + offset]
#         if nx.has_path(graph, src, dest):
#             path = nx.shortest_path(graph, src, dest)
#             partition_rest.append(path)
#             partition_total.append(path)
#
#             for i in range(len(path) - 1):
#                 graph.remove_edge(path[i], path[i + 1])


#                     path_tmp = len(nx.shortest_path(graph, vi, vj)) - 1

# for b in range(len(boundaries)):
#     print("lev: ", b)
#     while True:
#         path_max = 0
#         src, dest = -1, -1
#         for i in range(len(boundaries[b]) - 1):
#             for j in range(i + 1, len(boundaries[b]) - 1):
#                 vi, vj = boundaries[b][i], boundaries[b][j]
#                 if nx.has_path(graph, vi, vj):
#                     path_tmp = len(nx.shortest_path(graph, vi, vj)) - 1
#                     if path_max < path_tmp:
#                         path_max = path_tmp
#                         src, dest = vi, vj
#
#
#
#         if path_max == 0:
#             break
#         path = nx.shortest_path(graph, src, dest)
#         partition_cycle.append(path)
#
#         for i in range(len(path) - 1):
#             graph.remove_edge(path[i], path[i + 1])

# partition_cycle = partition_rest

edges_test = np.array(graph.edges())
edges_test = np.reshape(edges_test, (-1))

indices_test = ti.field(int, shape= edges_test.shape[0])
indices_test.from_numpy(edges_test)

short = 1e3
start = -1
end = -1

partition_level = []

offset = 0
# for t in range(1):
#
#     if len(boundaries[0]) <1:
#         print("fuck")
#         offset += 1
#
#     for bi in boundaries[offset]:
#         start = bi
#         end = -1
#         short = 1e3
#         path = [start]
#
#         for i in range(offset + 1, len(boundaries)):
#
#             if len(boundaries[i]) < 1:
#                 print("fuck")
#
#             for bj in boundaries[i]:
#                 # print(bj)
#                 if nx.has_path(graph, path[-1], bj):
#                     path_tmp = len(nx.shortest_path(graph, path[-1], bj)) - 1
#                     # print(path_tmp)
#                     if path_tmp < short:
#                         short = path_tmp
#                         start = bj
#                         end = bj
#
#             if end == -1:
#                 boundaries[i - 1].remove(path[-1])
#                 break
#
#             if short > 1:
#                 break
#             else:
#                 graph.remove_edge(path[-1], end)
#                 path.append(end)
#                 end = -1
#                 short = 1e3
#
#         # print(path)
#
#         if len(path) > 1:
#             partition_cycle.append(path)


# print(partition_level)
# print(nx.shortest_path(graph, 11, 57))
# print(partition_cycle)
# print(np.array(graph.edges))


# partition_cycle = partition_level
# for e in graph.edges:
#     path = [e[0], e[1]]
#     # print(path)
#     partition_total.append(path)

num_max_partition = len(partition_total)
num_edges_per_partition_np = np.array([len(partition_total[i]) - 1 for i in range(num_max_partition)])
num_max_edges_per_partition = max(num_edges_per_partition_np)

num_edges_per_partition = ti.field(dtype=int, shape=num_max_partition)
num_edges_per_partition.from_numpy(num_edges_per_partition_np)

# print(num_edges_per_partition)
partitioned_set_np = np.zeros([num_max_partition, num_max_edges_per_partition], dtype=int)
cnt = 0
a = []

for i in range(num_max_partition):
    for j in range(len(partition_total[i]) - 1):
        partitioned_set_np[i, j] = cnt
        a.append(partition_total[i][j])
        a.append(partition_total[i][j + 1])
        cnt += 1

a = np.array(a)
b = np.array(graph.edges).reshape(-1)

# a = np.append(a, b)
# num_edges = cnt + len(graph.edges)

num_edges = cnt

indices = ti.field(int, shape= 2 * num_edges)
indices.from_numpy(np.array(a))

partitioned_set = ti.field(dtype=int, shape=(num_max_partition, num_max_edges_per_partition))
partitioned_set.from_numpy(partitioned_set_np)
# print(partitioned_set)

colors_np = np.empty((num_particles, 3), dtype=float)

viridis = mpl.colormaps['viridis'].resampled(len(boundaries))

show_partition_id = 0

def color_partition(id, set):
    for i in range(num_particles):
        colors_np[i, 0] = 0.0
        colors_np[i, 1] = 0.0
        colors_np[i, 2] = 0.0
    #
    # for i in range(len(boundaries)):
    #     r, g, b, _ = viridis(i)
    #     for vi in boundaries[i]:
    #         colors_np[vi, 0] = r
    #         colors_np[vi, 1] = g
    #         colors_np[vi, 2] = b

    # for i in range(id):
    for vi in set[id]:
        colors_np[vi, 0] = 1.0
        colors_np[vi, 1] = 0.0
        colors_np[vi, 2] = 0.0


colors = ti.Vector.field(n=3, shape=num_particles, dtype=float)
colors.from_numpy(colors_np)

num_max_vertices_per_partition = num_max_edges_per_partition + 1

l0 = ti.field(float, shape=2 * num_edges)
dt = 0.03
x    = ti.Vector.field(n=3, dtype=float, shape=num_particles)
mass = ti.field(dtype=float, shape=num_particles)
x_k  = ti.Vector.field(n=3, dtype=float, shape=num_particles)
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


a_part       = ti.Matrix.field(n=3, m=3, dtype=float, shape=(num_max_partition, num_max_vertices_per_partition))
b_part       = ti.Matrix.field(n=3, m=3, dtype=float, shape=(num_max_partition, num_max_vertices_per_partition))
c_part       = ti.Matrix.field(n=3, m=3, dtype=float, shape=(num_max_partition, num_max_vertices_per_partition))
c_tilde_part = ti.Matrix.field(n=3, m=3, dtype=float, shape=(num_max_partition, num_max_vertices_per_partition))

d_part       = ti.Vector.field(n=3, dtype=float, shape=(num_max_partition, num_max_vertices_per_partition))
d_tilde_part = ti.Vector.field(n=3, dtype=float, shape=(num_max_partition, num_max_vertices_per_partition))
x_part       = ti.Vector.field(n=3, dtype=float, shape=(num_max_partition, num_max_vertices_per_partition))

num_dup = ti.field(float, shape=num_particles)

collision_set = ti.field(int, shape=num_particles)
num_colliding_vertices = ti.field(int, shape=1)
K = ti.linalg.SparseMatrixBuilder(3 * num_particles, 3 * num_particles, max_num_triplets=10000000)
ndarr = ti.ndarray(ti.f32, shape=3 * num_particles)


@ti.func
def flattend_index(i: int, j: int):

    return i + num_particles_x * j


@ti.kernel
def generate_particles(num_particles: int, delta: float):

    for i in range(num_particles):
        xi = i % num_particles_x
        yi = i // num_particles_x
        x0[i] = -ti.Vector([xi, 0, yi]) * delta

    mass.fill(1.0)


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
        v0, v1 = indices[2 * i + 0], indices[2 * i + 1]
        l0[i] = (x0[v0] - x0[v1]).norm()
        mass[v0] += 0.5 * l0[i] * density
        mass[v1] += 0.5 * l0[i] * density


    ti.loop_config(serialize=True)
    for pi in range(num_max_partition):
        size_pi = num_edges_per_partition[pi]

        ti.loop_config(serialize=True)
        for i in range(size_pi):
            ei = partitioned_set[pi, i]
            vi = indices[2 * ei + 0]
            # print(vi)
            num_dup[vi] += 1.0
        ei = partitioned_set[pi, size_pi - 1]
        vi = indices[2 * ei + 1]
        num_dup[vi] += 1.0

    for i in range(num_particles):
        if num_dup[i] < 1:
            mass[i] = 1.0

# @ti.kernel
def generate_mesh():

    x0.from_numpy(x_np_temp)
    edges_flattened = edges.reshape(-1)
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
    g = ti.Vector([0.0, -9.81, 0.0])
    for i in x:
        xi, vi = x[i], v[i]
        x_k[i] = y[i] = xi + vi * dt + g * dt * dt

@ti.kernel
def compute_v():
    for i in x:
        v[i] = (x_k[i] - x[i])/ dt
        x[i] = x_k[i]

@ti.kernel
def compute_grad_and_hessian_momentum(x: ti.template()):

    id3 = ti.Matrix.identity(dt=float, n=3)
    for i in range(num_particles):
        grad[i] = mass[i] * (x[i] - y[i])
        hii[i] = mass[i] * id3

@ti.kernel
def compute_grad_and_hessian_attachment(x: ti.template(), k: float):

    id3 = ti.Matrix.identity(dt=float, n=3)
    # ids = ti.Vector([i for i in range(num_particles_x)], dt=int)
    ids = ti.Vector([0, 1, 2 , 3], dt=int)
    # print(ids)
    for i in range(ids.n):
        grad[ids[i]] += k * (x[ids[i]] - x0[ids[i]])
        hii[ids[i]] += k * id3

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
        v0, v1 = indices[2 * i + 0], indices[2 * i + 1]
        # print(v0, v1)
        x01 = x[v0] - x[v1]
        n = x01.normalized()
        l = x01.norm()
        dp01 = x01 - l0[i] * n

        grad[v0] += k * dp01
        grad[v1] -= k * dp01

        n = x01.normalized()
        alpha = l0[i] / l
        tmp = ti.math.vec3(n[0] + ti.random(float), n[1] + ti.random(float), n[2] + ti.random(float))
        t1 = ti.math.normalize(n.cross(tmp))
        t2 = n.cross(t1)
        D = ti.math.mat3([k, 0.0, 0.0, 0.0, k * abs(1.0 - alpha), 0.0, 0.0, 0.0, k * abs(1.0 - alpha)])
        P = ti.math.mat3([n[0], t1[0], t2[0], n[1], t1[1], t2[1], n[2], t1[2], t2[2]])
        B = (P @ D @ P.inverse())

        hii[v0] += B
        hii[v1] += B
        hij[i] = B

@ti.kernel
def solve_Jacobi(x: ti.template()):
    for vi in range(num_particles):
        x[vi] += hess[vi].inverse() @ grad[vi]

# @ti.func
# def BlockThomasAlgorithm(a: ti.template(), b: ti.template(), c: ti.template(), x: ti.template(), d: ti.template()):

@ti.kernel
def substep_Euler(Px: ti.template(), x: ti.template()):

    Px.fill(0.0)
    b_part.fill(0.0)
    a_part.fill(0.0)
    c_part.fill(0.0)

    # ti.loop_config(serialize=True)
    for pi in range(num_max_partition):
        size_pi = num_edges_per_partition[pi]

        # print(size_pi)
        n_verts_pi = size_pi + 1
        ti.loop_config(serialize=True)
        for i in range(size_pi):
            ei = partitioned_set[pi, i]
            # print(ei)
            vi, vj = indices[2 * ei + 0], indices[2 * ei + 1]
            # print(vi, vj)
            b_part[pi, i]     = hii[vi]
            b_part[pi, i + 1] = hii[vj]
            a_part[pi, i + 1] = -hij[ei]
            c_part[pi, i]     = -hij[ei]

            d_part[pi, i]     = x[vi]
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

        x_part[pi, n_verts_pi - 1] = d_tilde_part[pi, n_verts_pi - 1]
        ti.loop_config(serialize=True)
        for i in range(n_verts_pi - 1):
            idx = n_verts_pi - 2 - i
            x_part[pi, idx] = d_tilde_part[pi, idx] - c_tilde_part[pi, idx] @ x_part[pi, idx + 1]


    # ti.loop_config(serialize=True)
    for pi in range(num_max_partition):
        size_pi = num_edges_per_partition[pi]

        ti.loop_config(serialize=True)
        for i in range(size_pi):
            ei = partitioned_set[pi, i]
            vi = indices[2 * ei + 0]
            P_grad[vi] += x_part[pi, i]

        ei = partitioned_set[pi, size_pi - 1]
        vi = indices[2 * ei + 1]
        Px[vi] += x_part[pi, size_pi]

    for i in range(num_particles):
        if num_dup[i] > 0:
            Px[i] /= num_dup[i]
        else:
            Px[i] = ti.math.inverse(hii[i]) @ x[i]

@ti.kernel
def substep_Euler_GS(pi: int, x: ti.template(), k: float):

    id3 = ti.Matrix.identity(dt=float, n=3)
    ids = ti.Vector([0, 1, 2, 3], dt=int)

    b_part.fill(0.0)
    a_part.fill(0.0)
    c_part.fill(0.0)
    d_part.fill(0.0)

    size_pi = num_edges_per_partition[pi]
    # print(size_pi)
    n_verts_pi = size_pi + 1
    # ti.loop_config(serialize=True)
    for i in range(size_pi):
        ei = partitioned_set[pi, i]
        v0, v1 = indices[2 * ei + 0], indices[2 * ei + 1]

        d_part[pi, i] += mass[v0] * (x[v0] - y[v0])
        b_part[pi, i] += mass[v0] * id3

        if v0 == 0 or v0 == 1:
            d_part[pi, i] += k * (x[v0] - x0[v0])
            b_part[pi, i] += k * id3

        x01 = x[v0] - x[v1]
        n = x01.normalized()
        l = x01.norm()

        dp01 = x01 - l0[ei] * n
        d_part[pi, i]     += k * dp01
        d_part[pi, i + 1] -= k * dp01

        alpha = l0[ei] / l
        tmp = ti.math.vec3(n[0] + ti.random(float), n[1] + ti.random(float), n[2] + ti.random(float))
        t1 = ti.math.normalize(n.cross(tmp))
        t2 = n.cross(t1)
        D = ti.math.mat3([k, 0.0, 0.0, 0.0, k * abs(1.0 - alpha), 0.0, 0.0, 0.0, k * abs(1.0 - alpha)])
        P = ti.math.mat3([n[0], t1[0], t2[0], n[1], t1[1], t2[1], n[2], t1[2], t2[2]])
        B = (P @ D @ P.inverse())

        b_part[pi, i]     += B
        b_part[pi, i + 1] += B
        a_part[pi, i + 1] = -B
        c_part[pi, i]     = -B


    ei = partitioned_set[pi, size_pi - 1]
    vi = indices[2 * ei + 1]
    if vi == 0 or vi == 1:
        d_part[pi, size_pi] += k * (x[vi] - x0[vi])
        b_part[pi, size_pi] += k * id3

    d_part[pi, size_pi] += mass[vi] * (x[vi] - y[vi])
    b_part[pi, size_pi] += mass[vi] * id3

    c_tilde_part[pi, 0] = ti.math.inverse(b_part[pi, 0]) @ c_part[pi, 0]
    d_tilde_part[pi, 0] = ti.math.inverse(b_part[pi, 0]) @ d_part[pi, 0]

    ti.loop_config(serialize=True)
    for i in range(1, n_verts_pi - 1):
        tmp = ti.math.inverse(b_part[pi, i] - a_part[pi, i] @ c_tilde_part[pi, i - 1])
        c_tilde_part[pi, i] = tmp @ c_part[pi, i]
        d_tilde_part[pi, i] = tmp @ (d_part[pi, i] - a_part[pi, i] @ d_tilde_part[pi, i - 1])

    tmp = ti.math.inverse(b_part[pi, n_verts_pi - 1] - a_part[pi, n_verts_pi - 1] @ c_tilde_part[pi, n_verts_pi - 2])
    d_tilde_part[pi, n_verts_pi - 1] = tmp @ (d_part[pi, n_verts_pi - 1] - a_part[pi, n_verts_pi - 1] @ d_tilde_part[pi, n_verts_pi - 2])

    x_part[pi, n_verts_pi - 1] = d_tilde_part[pi, n_verts_pi - 1]
    ti.loop_config(serialize=True)
    for i in range(n_verts_pi - 1):
        idx = n_verts_pi - 2 - i
        x_part[pi, idx] = d_tilde_part[pi, idx] - c_tilde_part[pi, idx] @ x_part[pi, idx + 1]


    # ti.loop_config(serialize=True)
    for i in range(size_pi):
        ei = partitioned_set[pi, i]
        vi = indices[2 * ei + 0]
        x[vi] -= x_part[pi, i]

    ei = partitioned_set[pi, size_pi - 1]
    vi = indices[2 * ei + 1]
    x[vi] -= x_part[pi, size_pi]

@ti.kernel
def substep_Jacobi(Px: ti.template(), x: ti.template()):

    for i in range(num_particles):
        Px[i] = hii[i].inverse() @ x[i]

@ti.kernel
def construct_Hessian(A: ti.types.sparse_matrix_builder()):

    for i in range(num_particles):
        B = hii[i]
        for m, n in ti.ndrange(3, 3):
            A[3 * i + m, 3 * i + n] += B[m, n]

    for i in range(num_edges):
        v0, v1 = indices[2 * i + 0], indices[2 * i + 1]
        B = hij[i]
        for m, n in ti.ndrange(3, 3):
            A[3 * v1 + m, 3 * v0 + n] -= B[m, n]
            A[3 * v0 + m, 3 * v1 + n] -= B[m, n]

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

    for _ in range(num_iters):

        compute_grad_and_hessian_momentum(x)
        compute_grad_and_hessian_attachment(x)

        solve_Jacobi()

    compute_v()

    return itr_cnt


debug = partition_total


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

        solver_type = w.slider_int("solver type", solver_type, 0, 2)
        if solver_type == 0:
            w.text("Euler")
        elif solver_type == 1:
            w.text("Jacobi")
        elif solver_type == 2:
            w.text("Newton")

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

        show_partition_id = w.slider_int("partition id", show_partition_id, 0, len(debug) - 1)

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

    color_partition(show_partition_id, debug)
    colors.from_numpy(colors_np)

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

    scene.particles(x, radius=0.05, per_vertex_color=colors)
    # scene.particles(center, radius=radius, color=(1.0, 0.5, 0.0))

    if enable_lines:
        scene.lines(x, indices=indices_test, color=(0.0, 0.0, 1.0), width=1.0)
    if enable_lines2:
        scene.lines(x, indices=indices, color=(1.0, 0.0, 0.0), width=1.0)

    camera.track_user_inputs(window, movement_speed=0.4, hold_key=ti.ui.RMB)
    canvas.scene(scene)
    window.show()