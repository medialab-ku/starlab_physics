import taichi as ti
import numpy as np
import meshio
import networkx as nx
from timeit import default_timer as timer

from math_utils import elastic_util as eu
ti.init(arch=ti.cpu, kernel_profiler=True)

window = ti.ui.Window("Particle FEM", (1024, 768), fps_limit=200)
gui = window.get_gui()
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = window.get_scene()
camera = ti.ui.Camera()
camera.position(2., 4.0, 7.0)
camera.fov(40)
camera.up(0, 1, 0)

run_sim = False

num_iters = 2
frame_cnt = 0
threshold = 4
has_end = False
end_frame = 10
PR = 6
PR_b = 6
k_at = 5
solver_type = 0

enable_definite_fix = True
enable_pncg = False
enable_attachment = False
print_stat = False
enable_detection = False
enable_lines = False
enable_lines2 = False
enable_lines3 = False

model_path ="models/MSH/bar-14114.msh"
mesh = meshio.read(model_path)
scale_lf = lambda x, sc: sc * x
trans_lf = lambda x, trans: x + trans


x_np_temp = np.array(mesh.points, dtype=float)
num_particles = x_np_temp.shape[0]
center = x_np_temp.sum(axis=0) / x_np_temp.shape[0]

mass = ti.field(shape=num_particles, dtype=float)

tet_indices_np = np.array(mesh.cells[0].data, dtype=int)
num_tetras = tet_indices_np.shape[0]

tet_indices = ti.field(shape=tet_indices_np.shape, dtype=int)
tet_indices.from_numpy(tet_indices_np)

Dm_inv = ti.Matrix.field(n=3, m=3, shape=num_particles, dtype=float)
rest_volume = ti.field(shape=num_particles, dtype=float)

grad = ti.Vector.field(n=3, shape=num_particles, dtype=float)
hess = ti.Matrix.field(n=3, m=3, shape=num_particles, dtype=float)

edges = set()
triangles = set()
for i in range(num_tetras):
    edges.add(tuple(sorted([tet_indices_np[i][0], tet_indices_np[i][3]])))
    edges.add(tuple(sorted([tet_indices_np[i][1], tet_indices_np[i][3]])))
    edges.add(tuple(sorted([tet_indices_np[i][2], tet_indices_np[i][3]])))
    edges.add(tuple(sorted([tet_indices_np[i][0], tet_indices_np[i][1]])))
    edges.add(tuple(sorted([tet_indices_np[i][1], tet_indices_np[i][2]])))
    edges.add(tuple(sorted([tet_indices_np[i][2], tet_indices_np[i][0]])))

    triangles.add(tuple(sorted([tet_indices_np[i][0], tet_indices_np[i][1], tet_indices_np[i][2]])))
    triangles.add(tuple(sorted([tet_indices_np[i][1], tet_indices_np[i][2], tet_indices_np[i][3]])))
    triangles.add(tuple(sorted([tet_indices_np[i][2], tet_indices_np[i][3], tet_indices_np[i][0]])))
    triangles.add(tuple(sorted([tet_indices_np[i][3], tet_indices_np[i][0], tet_indices_np[i][1]])))


edges = list(edges)
edges_np = np.array(edges, dtype=int)

num_edges = edges_np.shape[0]

triangles = list(triangles)
triangles_np = np.array(triangles, dtype=int)
num_triangles = triangles_np.shape[0]
triangles = ti.field(shape=3 * num_triangles, dtype=int)
triangles.from_numpy(triangles_np.reshape(-1))

G = nx.Graph()
G.add_edges_from(edges_np)

vertex_neighbour_ids_tmp = [[] for i in range(num_particles)]
vertex_neighbour_edge_ids_tmp = [[] for i in range(num_particles)]

for i in range(num_edges):
    vertex_neighbour_ids_tmp[edges[i][0]].append(edges[i][1])
    vertex_neighbour_ids_tmp[edges[i][1]].append(edges[i][0])

    vertex_neighbour_edge_ids_tmp[edges[i][0]].append(i)
    vertex_neighbour_edge_ids_tmp[edges[i][1]].append(i)


tet_edge_indices_np = np.zeros(shape=(num_tetras, 6), dtype=int)

# print(vertex_neighbour_edge_ids_tmp)

# print(edges)

for i in range(num_tetras):
    a = sorted((tet_indices_np[i][0], tet_indices_np[i][3]))
    tet_edge_indices_np[i, 0] = edges.index((a[0], a[1]))

    a = sorted((tet_indices_np[i][1], tet_indices_np[i][3]))
    tet_edge_indices_np[i, 1] = edges.index((a[0], a[1]))

    a = sorted((tet_indices_np[i][2], tet_indices_np[i][3]))
    tet_edge_indices_np[i, 2] = edges.index((a[0], a[1]))

    a = sorted((tet_indices_np[i][0], tet_indices_np[i][1]))
    tet_edge_indices_np[i, 3] = edges.index((a[0], a[1]))

    a = sorted((tet_indices_np[i][1], tet_indices_np[i][2]))
    tet_edge_indices_np[i, 4] = edges.index((a[0], a[1]))

    a = sorted((tet_indices_np[i][2], tet_indices_np[i][0]))
    tet_edge_indices_np[i, 5] = edges.index((a[0], a[1]))

edges = ti.field(shape=2 * num_edges, dtype=int)
edges.from_numpy(edges_np.reshape(-1))

tet_edge_indices = ti.field(shape=tet_edge_indices_np.shape, dtype=int)
tet_edge_indices.from_numpy(tet_edge_indices_np)

edge_weight = ti.field(shape=num_edges, dtype=float)

num_vertex_neighbour_ids_np = np.array([len(i) for i in vertex_neighbour_ids_tmp], dtype=int)

num_vertex_neighbour_ids = ti.field(shape=num_vertex_neighbour_ids_np.shape, dtype=int)
num_vertex_neighbour_ids.from_numpy(num_vertex_neighbour_ids_np)

max_num_vertex_neighbour_ids = max(num_vertex_neighbour_ids_np)

print(np.where(num_vertex_neighbour_ids_np == max_num_vertex_neighbour_ids))

vertex_neighbour_ids_np = np.zeros(shape=(num_particles, max_num_vertex_neighbour_ids), dtype=int)
vertex_neighbour_edge_ids_np = np.zeros(shape=(num_particles, max_num_vertex_neighbour_ids), dtype=int)
for i in range(num_particles):
    for j in range(len(vertex_neighbour_ids_tmp[i])):
        vertex_neighbour_ids_np[i][j] = vertex_neighbour_ids_tmp[i][j]
        vertex_neighbour_edge_ids_np[i][j] = vertex_neighbour_edge_ids_tmp[i][j]

vertex_neighbour_ids = ti.field(shape=vertex_neighbour_ids_np.shape, dtype=int)
vertex_neighbour_ids.from_numpy(vertex_neighbour_ids_np)

vertex_neighbour_edge_ids = ti.field(shape=vertex_neighbour_edge_ids_np.shape, dtype=int)
vertex_neighbour_edge_ids.from_numpy(vertex_neighbour_edge_ids_np)

vertex_edge_ids = ti.field(shape=vertex_neighbour_ids_np.shape, dtype=int)


print(min(num_vertex_neighbour_ids_np), max(num_vertex_neighbour_ids_np))

# colors = nx.greedy_color(G)
# unique_elements, counts = np.unique(np.array(list(colors.values())), return_counts=True)

# vertex_element_ids_tmp = [[] for i in range(num_particles)]

# for i in range(num_tetras):
#     for j in range(4):
#         vertex_element_ids_tmp[tet_indices_np[i][j]].append(i)
#
# num_vertex_element_ids_np = np.array([len(vei) for vei in vertex_element_ids_tmp], dtype=int)
# max_elements = max(num_vertex_element_ids_np)
# vertex_element_ids_np = np.zeros(shape=(num_particles, max_elements), dtype=int)
# for i in range(num_particles):
#     for j in range(len(vertex_element_ids_tmp[i])):
#         vertex_element_ids_np[i][j] = vertex_element_ids_tmp[i][j]

# num_vertex_element_ids = ti.field(shape=num_vertex_element_ids_np.shape, dtype=int)
# num_vertex_element_ids.from_numpy(num_vertex_element_ids_np)
#
# vertex_element_ids = ti.field(shape=vertex_element_ids_np.shape, dtype=int)
# vertex_element_ids.from_numpy(vertex_element_ids_np)
#
#
# num_colors = len(unique_elements)
# partition = [[] for i in unique_elements]
# vertex_color_ids_np = np.zeros(shape=num_particles, dtype=int)
#
# for key in colors.keys():
#     vertex_color_ids_np[key] = colors[key]
#     partition[colors[key]].append(key)

# num_vertex_color_np = np.array([len(p) for p in partition], dtype=int)
# num_vertex_color = ti.field(shape=num_vertex_color_np.shape, dtype=int)
# num_vertex_color.from_numpy(num_vertex_color_np)
#
# # print(num_vertex_color)
#
# max_num_vertex_color = max(num_vertex_color_np)
# vertex_ids_partition_np =  np.zeros(shape=(num_colors, max_num_vertex_color), dtype=int)
# for i in range(num_colors):
#     for j in range(len(partition[i])):
#         vertex_ids_partition_np[i][j] = partition[i][j]
#
# vertex_ids_color = ti.field(shape=vertex_ids_partition_np.shape, dtype=int)
# vertex_ids_color.from_numpy(vertex_ids_partition_np)
#
# test = 0
# for i in range(len(partition)):
#     test += len(partition[i])

color_test = True
for i in range(edges_np.shape[0]):
    if edges_np[i][0] == edges_np[i][1]:
        color_test = False
        break

if color_test:
    print("graph color success...")


# vertex_colors_np = np.zeros(shape=(num_particles, 3), dtype=float)
#
# for j in range(num_particles):
#
#     if vertex_color_ids_np[j] == 0:
#         vertex_colors_np[j] = np.array([1, 0, 0])
#     if vertex_color_ids_np[j] == 1:
#         vertex_colors_np[j] = np.array([0, 1, 0])
#     if vertex_color_ids_np[j] == 2:
#         vertex_colors_np[j] = np.array([0, 0, 1])
#     if vertex_color_ids_np[j] == 3:
#         vertex_colors_np[j] = np.array([0, 1, 1])
#     if vertex_color_ids_np[j] == 4:
#         vertex_colors_np[j] = np.array([1, 0, 1])
#     if vertex_color_ids_np[j] == 4:
#         vertex_colors_np[j] = np.array([1, 1, 0])
#
# vertex_colors = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
# vertex_colors.from_numpy(vertex_colors_np)


dt = 0.03
x0  = ti.Vector.field(n=3, dtype=float, shape=num_particles)
x   = ti.Vector.field(n=3, dtype=float, shape=num_particles)
x_k = ti.Vector.field(n=3, dtype=float, shape=num_particles)
y = ti.Vector.field(n=3, dtype=float, shape=num_particles)
x0.from_numpy(x_np_temp)

v = ti.Vector.field(n=3, dtype=float, shape=num_particles)
v.fill(0.0)

@ti.kernel
def compute_F(x: ti.template()):

    # ti.loop_config(serialize=True)
    for vi in range(num_particles):
        Ds = ti.Matrix.zero(n=3, m=3, dt=float)
        B = Dm_inv[vi]

        for j in range(3):
            # vj = vertex_neighbour_ids[vi, j]
            # ej = vertex_neighbour_edge_ids[vi, j]
            xji0 = x0[j] - x0[vi]
            # wji = edge_weight[ej]
            xji = x[j] - x[vi]
            Ds += xji.outer_product(xji0)

    # print(F)

        F = Ds @ B
        U, sigma, V = eu.ssvd(F)
        R = U @ V.transpose()
        P = (R - F)


@ti.kernel
def particle_fem(x: ti.template(), k: float):

    I_3 = ti.Matrix.identity(dt=float, n=3)
    k_at = 1e5
    for vi in range(num_particles):

        grad[vi] = mass[vi] * (y[vi] - x[vi])
        hess[vi] = mass[vi] * I_3

        if vi == 0:
            grad[vi] += k_at * (x0[vi] - x[vi])
            hess[vi] += k_at * I_3

        if vi == 15:
            grad[vi] += k_at * (x0[vi] - x[vi])
            hess[vi] += k_at * I_3

    # for vi in range(num_particles):
        Ds = ti.Matrix.zero(n=3, m=3, dt=float)
        B = Dm_inv[vi]

        for j in range(num_vertex_neighbour_ids[vi]):
            vj = vertex_neighbour_ids[vi, j]
            ej = vertex_neighbour_edge_ids[vi, j]
            xji0 = x0[vj] - x0[vi]
            wji = edge_weight[ej]
            xji = x[vj] - x[vi]
            Ds += wji * xji.outer_product(xji0)

        # print(F)

        F = Ds @ B

        U, sigma, V = eu.ssvd(F)
        R = U @ V.transpose()
        P = (R - F)

        dFdxi = ti.math.vec3(0.0)
        coeff = rest_volume[vi] * k

        for j in range(num_vertex_neighbour_ids[vi]):
            vj = vertex_neighbour_ids[vi, j]
            ej = vertex_neighbour_edge_ids[vi, j]
            xji0 = x0[vj] - x0[vi]
            wji = edge_weight[ej]
            dFdxj = wji * B @ xji0
            dFdxi += dFdxj

            grad[vj] += coeff * P @ dFdxj
            hess[vj] += coeff * (dFdxj[0] ** 2 + dFdxj[1] ** 2 + dFdxj[2] ** 2) * I_3
           
        #
        grad[vi] -= coeff * P @ dFdxi
        hess[vi] += coeff * (dFdxi[0] ** 2 + dFdxi[1] ** 2 + dFdxi[2] ** 2) * I_3
        

    for vi in range(num_particles):
        x[vi] += hess[vi].inverse() @ grad[vi]






# @ti.kernel
# def color_kernel(pid: int, x: ti.template(), k: float):
#
#     id3 = ti.Matrix.identity(dt=float, n=3)
#     for i in range(num_vertex_color[pid]):
#
#         # inertia term
#         vi = vertex_ids_color[pid, i]
#         # print(vi)
#         grad_i = mass[vi] * (y[vi] - x[vi])
#         hess_i = mass[vi] * id3
#         k_at = 1e5
#         if vi == 0:
#             grad_i += k_at * (x0[vi] - x[vi])
#             hess_i += k_at * id3
#
#         # elastic energy term
#
#         Ds = ti.Matrix.zero(n=3, m=3, dt=float)
#         B = Dm_inv[vi]
#         xji0_sum = ti.math.vec3(0.0)
#         for j in range(num_vertex_neighbour_ids[vi]):
#             xji0 = x0[vertex_neighbour_ids[vi, j]] - x0[vi]
#             wji = edge_weight[vertex_neighbour_edge_ids[vi, j]]
#             bji = wji * B @ xji0
#             xji0_sum += bji
#             xji = x[vertex_neighbour_ids[vi, j]] - x[vi]
#             Ds += wji * xji.outer_product(xji0)
#
#         dFdxi = xji0_sum
#         F = Ds @ B
#
#         # F = id3
#         U, sigma, V = eu.ssvd(F)
#         R = U @ V.transpose()
#         P = (R - F)
#         # P.fill(0.0)
#
#         coeff = rest_volume[vi] * k
#
#         test = dFdxi[0] ** 2 + dFdxi[1] ** 2 + dFdxi[2] ** 2
#         grad_i -= coeff * P @ dFdxi
#         hess_i += coeff * test * id3
#         x[vi] += hess_i.inverse() @ grad_i


@ti.kernel
def compute_y():

    # apply gravity within boundary
    g = ti.Vector([0.0, -9.81, 0.0])
    # g = ti.Vector([0.0, 0.0, 0.0])
    for i in x:
        xi, vi = x[i], v[i]
        x_k[i] = y[i] = xi + vi * dt + g * dt * dt

@ti.kernel
def compute_v():
    for i in x:
        v[i] = (x_k[i] - x[i])/ dt
        x[i] = x_k[i]

def forward():

    compute_y()

    for _ in range(num_iters):
        k = dt * dt * pow(10, PR)
        particle_fem(x_k, k)



    compute_v()



@ti.kernel
def initialize():

    rest_volume.fill(0.0)
    edge_weight.fill(0.0)

    mass.fill(0.0)

    rest_volume_total = 0.0
    for i in range(num_tetras):

        Dm_i = ti.Matrix.cols([x[tet_indices[i, j]] - x[tet_indices[i, 3]] for j in ti.static(range(3))])
        rest_volume_i = ti.abs(Dm_i.determinant()) / 6.0
        for j in ti.static(range(4)):
            rest_volume[tet_indices[i, j]] += rest_volume_i

        for j in ti.static(range(6)):
            edge_weight[tet_edge_indices[i, j]] += rest_volume_i

    for i in range(num_particles):

        rest_volume_total += rest_volume[i]
        # print(vertex_neighbour_ids[i, 0], vertex_neighbour_ids[i, 1], vertex_neighbour_ids[i, 2])
        Dm_i = ti.Matrix.zero(n=3, m=3, dt=float)
        # Dm_i = ti.Matrix.cols([x[vertex_neighbour_ids[i, j]] - x[i] for j in range(3)])
        for j in range(num_vertex_neighbour_ids[i]):
            vj = vertex_neighbour_ids[i, j]
            ej = vertex_neighbour_edge_ids[i, j]
            xji0 = x[vj] - x[i]
            wji = edge_weight[ej]
            Dm_i += wji * xji0.outer_product(xji0)

        Dm_inv[i] = Dm_i.inverse()

    # for vi in range(num_particles):
    #
    #     F = ti.Matrix.zero(n=3, m=3, dt=float)
    #     B = Dm_inv[vi]
    #
    #     for j in range(num_vertex_neighbour_ids[vi]):
    #         vj = vertex_neighbour_ids[vi, j]
    #         ej = vertex_neighbour_edge_ids[vi, j]
    #         xji0 = x0[vj] - x0[vi]
    #         wji = edge_weight[ej]
    #         xji = x[vj] - x[vi]
    #         F += xji.outer_product(xji0) @ B
    #
    #     print(F)

    # print(edge_weight)


def reset():
    x.copy_from(x0)
    v.fill(0.0)

    initialize()

    ti.profiler.clear_kernel_profiler_info()
    for i in range(1000):
        compute_F(x)
    query_result = ti.profiler.query_kernel_profiler_info(compute_F.__name__)
    print("kernel executed times =", query_result.counter)
    # print("kernel elapsed time(min_in_ms) =", query_result.min)
    # print("kernel elapsed time(max_in_ms) =", query_result.max)
    print("kernel elapsed time(avg_in_ms) =", query_result.avg)

    # print(vertex_neighbour_ids)
    # print(edge_weight)
    #

    mass.copy_from(rest_volume)

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


        if not PR_old == PR:
            PR = PR_old

        if has_end:
            end_frame = w.slider_int("end frame", end_frame, 0, 1000)

        w.text("")
        frame_str = "# frame " + str(frame_cnt)
        w.text(frame_str)

        # color_str = "# colors: " + str(num_colors)
        # w.text(color_str)

        verts_str = "# verts: " + str(num_particles)
        tetra_str = "# tetras: " + str(num_tetras)
        edges_str = "# edges: " + str(num_edges)
        w.text(verts_str)
        w.text(tetra_str)
        w.text(edges_str)

reset()

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
            reset()
            # x.copy_from(x0)
            # v.fill(0.0)
            # if frame_cnt > 0:
            #     if solver_type == 0:
            #         print("Euler")
            #     elif solver_type == 1:
            #         print("Jacobi")
            #     elif solver_type == 2:
            #         print("Newton")
            #     print("Avg. iter    : ", int(conv_it_total / frame_cnt))
            #     print("Avg. time[ms]: ", round(100.0 * (elapsed_time_total / frame_cnt), 2))

            conv_it_total = 0
            frame_cnt = 0
            elapsed_time_total = 0
            run_sim = False

        # if run_sim is False and window.event.key == 'o':
        #     start = timer()
        #     conv_itr = forward()
        #     end = timer()
        #
        #     conv_it_total += conv_itr
        #     elapsed_time = (end - start)
        #     elapsed_time_total += elapsed_time
        #     frame_cnt += 1

    # color_partition(show_partition_id, debug)
    # colors.from_numpy(colors_np)

    if run_sim:

        # start = timer()
        forward()
        # end = timer()

        # conv_it_total += conv_itr
        # elapsed_time = (end - start)
        # elapsed_time_total += elapsed_time
        frame_cnt += 1

    if has_end and frame_cnt >= end_frame:
        run_sim = False
    #
    #
    show_options()

    # scene.particles(x, radius=0.01, per_vertex_color=vertex_colors)
    scene.mesh(x, indices=triangles, color=(0.0, 0.0, 0.0), show_wireframe=True)
    scene.mesh(x, indices=triangles, color=(1.0, 0.5, 0.0))
    # scene.lines(x, indices=edges, color=(0.0, 0.0, 0.0), width=1.0)
    # scene.particles(center, radius=radius, color=(1.0, 0.5, 0.0))

    # if enable_lines:
    #     scene.lines(x, indices=bending_indices, color=(1.0, 0.0, 0.0), width=1.0)
    # if enable_lines2:
    #     scene.lines(x, indices=indices, color=(0.0, 0.0, 1.0), width=1.0)
    # # if enable_lines3:
    # #     scene.lines(x, indices=indices_test2, color=(0.0, 1.0, 0.0), width=1.0)
    #
    #
    # scene.mesh(x, indices=face_indices, color=(0.0, 0.0, 0.0), show_wireframe=True)
    # scene.mesh(x, indices=face_indices, color=(1.0, 0.5, 0.0))

    camera.track_user_inputs(window, movement_speed=0.4, hold_key=ti.ui.RMB)
    canvas.scene(scene)
    window.show()