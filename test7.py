import taichi as ti
import numpy as np
import meshio
import networkx as nx
from timeit import default_timer as timer
from math_utils import elastic_util as eu
ti.init(arch=ti.cpu, kernel_profiler=True)

window = ti.ui.Window("Tet FEM", (1024, 768), fps_limit=200)
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

model_path ="models/MSH/tet.msh"
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

Dm_inv = ti.Matrix.field(n=3, m=3, shape=num_tetras, dtype=float)
rest_volume = ti.field(shape=num_tetras, dtype=float)

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

edges_np = np.array(list(edges), dtype=int)
num_edges = edges_np.shape[0]
G = nx.Graph()
G.add_edges_from(edges_np)

triangles = list(triangles)
triangles_np = np.array(triangles, dtype=int)
num_triangles = triangles_np.shape[0]
triangles = ti.field(shape=3 * num_triangles, dtype=int)
triangles.from_numpy(triangles_np.reshape(-1))


colors = nx.greedy_color(G)
unique_elements, counts = np.unique(np.array(list(colors.values())), return_counts=True)

vertex_element_ids_tmp = [[] for i in range(num_particles)]

for i in range(num_tetras):
    for j in range(4):
        vertex_element_ids_tmp[tet_indices_np[i][j]].append(i)

num_vertex_element_ids_np = np.array([len(vei) for vei in vertex_element_ids_tmp], dtype=int)
max_elements = max(num_vertex_element_ids_np)
vertex_element_ids_np = np.zeros(shape=(num_particles, max_elements), dtype=int)
for i in range(num_particles):
    for j in range(len(vertex_element_ids_tmp[i])):
        vertex_element_ids_np[i][j] = vertex_element_ids_tmp[i][j]

num_vertex_element_ids = ti.field(shape=num_vertex_element_ids_np.shape, dtype=int)
num_vertex_element_ids.from_numpy(num_vertex_element_ids_np)

vertex_element_ids = ti.field(shape=vertex_element_ids_np.shape, dtype=int)
vertex_element_ids.from_numpy(vertex_element_ids_np)

num_colors = len(unique_elements)
partition = [[] for i in unique_elements]
vertex_color_ids_np = np.zeros(shape=num_particles, dtype=int)

for key in colors.keys():
    vertex_color_ids_np[key] = colors[key]
    partition[colors[key]].append(key)

num_vertex_color_np = np.array([len(p) for p in partition], dtype=int)
num_vertex_color = ti.field(shape=num_vertex_color_np.shape, dtype=int)
num_vertex_color.from_numpy(num_vertex_color_np)

max_num_vertex_color = max(num_vertex_color_np)
vertex_ids_partition_np =  np.zeros(shape=(num_colors, max_num_vertex_color), dtype=int)
for i in range(num_colors):
    for j in range(len(partition[i])):
        vertex_ids_partition_np[i][j] = partition[i][j]

vertex_ids_color = ti.field(shape=vertex_ids_partition_np.shape, dtype=int)
vertex_ids_color.from_numpy(vertex_ids_partition_np)

test = 0
for i in range(len(partition)):
    test += len(partition[i])

color_test = True
for i in range(edges_np.shape[0]):
    if edges_np[i][0] == edges_np[i][1]:
        color_test = False
        break

if color_test:
    print("graph color success...")


vertex_colors_np = np.zeros(shape=(num_particles, 3), dtype=float)

for j in range(num_particles):

    if vertex_color_ids_np[j] == 0:
        vertex_colors_np[j] = np.array([1, 0, 0])
    if vertex_color_ids_np[j] == 1:
        vertex_colors_np[j] = np.array([0, 1, 0])
    if vertex_color_ids_np[j] == 2:
        vertex_colors_np[j] = np.array([0, 0, 1])
    if vertex_color_ids_np[j] == 3:
        vertex_colors_np[j] = np.array([0, 1, 1])
    if vertex_color_ids_np[j] == 4:
        vertex_colors_np[j] = np.array([1, 0, 1])
    if vertex_color_ids_np[j] == 4:
        vertex_colors_np[j] = np.array([1, 1, 0])

vertex_colors = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
vertex_colors.from_numpy(vertex_colors_np)

edges = ti.field(shape=2 * edges_np.shape[0], dtype=int)

edges.from_numpy(edges_np.reshape(-1))
dt = 0.03
x0  = ti.Vector.field(n=3, dtype=float, shape=num_particles)
x   = ti.Vector.field(n=3, dtype=float, shape=num_particles)
x_k = ti.Vector.field(n=3, dtype=float, shape=num_particles)
y = ti.Vector.field(n=3, dtype=float, shape=num_particles)
x0.from_numpy(x_np_temp)

v = ti.Vector.field(n=3, dtype=float, shape=num_particles)
v.fill(0.0)

grad = ti.Vector.field(n=3, shape=num_particles, dtype=float)
hess = ti.Matrix.field(n=3, m=3, shape=num_particles, dtype=float)

@ti.kernel
def compute_F(x: ti.template()):

    ti.loop_config(serialize=True)
    for tid in range(num_tetras):
        indices_tmp = ti.Vector([tet_indices[tid, i] for i in range(4)])
        # print(indices_tmp)
        Ds = ti.Matrix.cols([x[indices_tmp[j]] - x[indices_tmp[3]] for j in range(3)])
        B = Dm_inv[tid]
        F = Ds @ B
        U, sigma, V = eu.ssvd(F)
        R = U @ V.transpose()
        P = (R - F)

@ti.kernel
def solve_Jacobi(x: ti.template()):
    for vi in range(num_particles):
        x[vi] += hess[vi].inverse() @ grad[vi]

@ti.kernel
def compute_grad_and_hess_momentum(x: ti.template()):

    id3 = ti.Matrix.identity(dt=float, n=3)
    for vi in range(num_particles):

        grad[vi] = mass[vi] * (y[vi] - x[vi])
        hess[vi] = mass[vi] * id3
        k_at = 1e5
        if vi == 0:
            grad[vi] += k_at * (x0[vi] - x[vi])
            hess[vi] += k_at * id3

@ti.kernel
def compute_grad_and_hess_elasticity(x: ti.template(), k: float):

    id3 = ti.Matrix.identity(dt=float, n=3)

    for vi in range(num_particles):

        grad[vi] = mass[vi] * (y[vi] - x[vi])
        hess[vi] = mass[vi] * id3
        k_at = 1e5
        if vi == 0:
            grad[vi] += k_at * (x0[vi] - x[vi])
            hess[vi] += k_at * id3

        if vi == 15:
            grad[vi] += k_at * (x0[vi] - x[vi])
            hess[vi] += k_at * id3

    for tid in range(num_tetras):
        indices_tmp = ti.Vector([tet_indices[tid, i] for i in range(4)])
        # print(indices_tmp)
        Ds = ti.Matrix.cols([x[indices_tmp[j]] - x[indices_tmp[3]] for j in range(3)])
        B = Dm_inv[tid]
        F = Ds @ B
        U, sigma, V = eu.ssvd(F)
        R = U @ V.transpose()
        P = (R - F)

        dFdx0 = B.transpose()[:, 0]
        dFdx1 = B.transpose()[:, 1]
        dFdx2 = B.transpose()[:, 2]
        dFdx3 = -(dFdx0 + dFdx1 + dFdx2)

        p03 = P @ dFdx0
        p13 = P @ dFdx1
        p23 = P @ dFdx2

        coeff = rest_volume[tid] * k
        grad[indices_tmp[0]] += coeff * p03
        test = dFdx0[0] ** 2 + dFdx0[1] ** 2 + dFdx0[2] ** 2
        hess[indices_tmp[0]] += coeff * test * id3

        grad[indices_tmp[1]] += coeff * p13
        test = dFdx1[0] ** 2 + dFdx1[1] ** 2 + dFdx1[2] ** 2
        hess[indices_tmp[1]] += coeff * test * id3

        grad[indices_tmp[2]] += coeff * p23
        test = dFdx2[0] ** 2 + dFdx2[1] ** 2 + dFdx2[2] ** 2
        hess[indices_tmp[2]] += coeff * test * id3

        grad[indices_tmp[3]] -= coeff * (p03 + p13 + p23)
        test = dFdx3[0] ** 2 + dFdx3[1] ** 2 + dFdx3[2] ** 2
        hess[indices_tmp[3]] += coeff * test * id3

    for vi in range(num_particles):
        x[vi] += hess[vi].inverse() @ grad[vi]


@ti.kernel
def color_kernel(pid: int, x: ti.template(), k: float):

    id3 = ti.Matrix.identity(dt=float, n=3)
    for i in range(num_vertex_color[pid]):

        # inertia term
        vi = vertex_ids_color[pid, i]
        # print(vi)
        grad_i = mass[vi] * (y[vi] - x[vi])
        hess_i = mass[vi] * id3
        k_at = 1e5
        if vi == 0:
            grad_i += k_at * (x0[vi] - x[vi])
            hess_i += k_at * id3

        # elastic energy term
        for j in range(num_vertex_element_ids[vi]):
            tid = vertex_element_ids[vi, j]
            indices_tmp = ti.Vector([tet_indices[tid, i] for i in range(4)])
            # print(indices_tmp)
            Ds = ti.Matrix.cols([x[indices_tmp[j]] - x[indices_tmp[3]] for j in range(3)])
            B = Dm_inv[tid]
            F = Ds @ B
            U, sigma, V = eu.ssvd(F)
            R = U @ V.transpose()
            P = (R - F)

            dFdx0 = B.transpose()[:, 0]
            dFdx1 = B.transpose()[:, 1]
            dFdx2 = B.transpose()[:, 2]
            dFdx3 = -(dFdx0 + dFdx1 + dFdx2)
            # test = P @ B.transpose()[:, 0]

            p03 = P @ dFdx0
            p13 = P @ dFdx1
            p23 = P @ dFdx2

            coeff = rest_volume[tid] * k

            if vi == indices_tmp[0]:
                test = dFdx0[0] ** 2 + dFdx0[1] ** 2 + dFdx0[2] ** 2
                grad_i += coeff * p03
                hess_i += coeff * test * id3

            elif vi == indices_tmp[1]:
                test = dFdx1[0] ** 2 + dFdx1[1] ** 2 + dFdx1[2] ** 2
                grad_i += coeff * p13
                hess_i += coeff * test * id3

            elif vi == indices_tmp[2]:
                test = dFdx2[0] ** 2 + dFdx2[1] ** 2 + dFdx2[2] ** 2
                grad_i += coeff * p23
                hess_i += coeff * test * id3
            else:
                test = dFdx3[0] ** 2 + dFdx3[1] ** 2 + dFdx3[2] ** 2
                grad_i -= coeff * (p03 + p13 + p23)
                hess_i += coeff * test * id3

        x[vi] += ti.math.inverse(hess_i) @ grad_i


@ti.kernel
def test(k: float):

    for ti in range(num_tetras):
        # ti = vertex_element_ids[vi, j]
        indices_tmp = ti.Vector([tet_indices[ti, i] for i in range(4)])
        Ds = ti.Matrix.cols([x[indices_tmp[j]] - x[indices_tmp[3]] for j in range(3)])
        B = Dm_inv[ti]
        F = Ds @ B
        U, sigma, V = eu.ssvd(F)
        R = U @ V.transpose()
        P = (R - F)
        coeff = rest_volume[ti] * k

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
        # compute_grad_and_hess_momentum(x_k)
        compute_grad_and_hess_elasticity(x_k, k)
        # solve_Jacobi(x_k)
        # for i in range(num_colors):
        #     # print(i, "color")
        #     k = dt * dt * pow(10, PR)
        #     color_kernel(i, x_k, k)

    compute_v()



@ti.kernel
def initialize():

    rest_volume.fill(0.0)
    mass.fill(0.0)

    rest_volume_total = 0.0
    for i in range(num_tetras):

        Dm_i = ti.Matrix.cols([x[tet_indices[i, j]] - x[tet_indices[i, 3]] for j in ti.static(range(3))])
        Dm_inv[i] = Dm_i.inverse()

        rest_volume_i = ti.abs(Dm_i.determinant()) / 6.0
        rest_volume[i] = rest_volume_i
        rest_volume_total += rest_volume_i
        for j in ti.static(range(4)):
            mass[tet_indices[i, j]] += 0.25 * rest_volume_i

    # print(rest_volume_total)

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

    # print(elapsed_time)

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

        color_str = "# colors: " + str(len(partition))
        w.text(color_str)

        verts_str = "# verts: " + str(num_particles)
        tetra_str = "# tetrs: " + str(num_tetras)
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