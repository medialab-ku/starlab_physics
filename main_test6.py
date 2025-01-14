from turtledemo.sorting_animate import partition

import meshio
import networkx as nx
from PIL.ImagePalette import random
from igl import volume
from pyquaternion.quaternion import Quaternion
import taichi as ti
from timeit import default_timer as timer
import numpy as np

from math_utils import elastic_util as e_util
import matplotlib as mpl

from tqdm import tqdm
import time


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

num_iters = 1
frame_cnt = 0
threshold = 2
has_end = False
end_frame = 200
PR = 1
k_at = 5
solver_type = 0
enable_pncg = False
enable_attachment = False
print_stat = False
enable_detection = False
enable_lines = False
enable_lines2 = False

dt = 0.03
size = 0.4
move = size * np.array([[1.0, 0.0, 0.0], [.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=float)
x_np = size * np.array([[0.0, 0.0, 0.0], [0., 0.0, 1.0], [0., 1.0, 1.0]])

n_tet = 50
for i in range(n_tet):
    test = x_np[-1] + move[i % 3]
    x_np = np.append(x_np, test).reshape(-1, 3)


edge_indices_np = np.array([[0, 1], [1, 2]]).reshape(-1)
for i in range(n_tet):
    e0 = edge_indices_np[-1]
    e1 = e0 + 1
    edge_indices_np = np.append(edge_indices_np, np.array([e0, e1]))

print(edge_indices_np)

edge_indices = ti.field(dtype=int, shape=edge_indices_np.shape[0])
edge_indices.from_numpy(edge_indices_np)
n_edge = edge_indices.shape[0] // 2

# tet_edge_indices_np = np.array([]).reshape(-1)
# for i in range(n_tet):
#     tet_edge_indices_np = np.append(tet_edge_indices_np, np.array([i, i + 1, i + 2]))
#
# tet_edge_indices = ti.field(dtype=int, shape=tet_edge_indices_np.shape[0])
# tet_edge_indices.from_numpy(tet_edge_indices_np)

# n_tet = tet_edge_indices_np.shape[0] //3

partition_edge = [[i for i in range(n_tet + 2)]]
num_max_partition = len(partition_edge)
num_edges_per_partition_np = np.array([len(partition_edge[i]) for i in range(num_max_partition)])
num_max_edges_per_partition = max(num_edges_per_partition_np)

num_edges_per_partition = ti.field(dtype=int, shape=num_max_partition)
num_edges_per_partition.from_numpy(num_edges_per_partition_np)
partitioned_set_np = np.zeros([num_max_partition, num_max_edges_per_partition], dtype=int)
partitioned_set = ti.field(dtype=int, shape=(num_max_partition, num_max_edges_per_partition))

for i in range(num_max_partition):
    for j in range(len(partition_edge[i])):
        partitioned_set_np[i, j] = partition_edge[i][j]

partitioned_set.from_numpy(partitioned_set_np)

num_particles = x_np.shape[0]
x = ti.Vector.field(n=3, dtype=float,  shape=num_particles)
x0 = ti.Vector.field(n=3, dtype=float, shape=num_particles)
x.from_numpy(x_np)
x0.copy_from(x)
v = ti.Vector.field(n=3, dtype=float, shape=num_particles)
y = ti.Vector.field(n=3, dtype=float, shape=num_particles)
x_k = ti.Vector.field(n=3, dtype=float, shape=num_particles)

grad   = ti.Vector.field(n=3, dtype=float, shape=num_particles)
P_grad = ti.Vector.field(n=3, dtype=float, shape=num_particles)
dx     = ti.Vector.field(n=3, dtype=float, shape=num_particles)

grad_k   = ti.Vector.field(n=3, dtype=float, shape=num_particles)
P_grad_k = ti.Vector.field(n=3, dtype=float, shape=num_particles)

grad_delta   = ti.Vector.field(n=3, dtype=float, shape=num_particles)
P_grad_delta = ti.Vector.field(n=3, dtype=float, shape=num_particles)

dx_k     = ti.Vector.field(n=3, dtype=float, shape=num_particles)

hii = ti.Matrix.field(n=3, m=3, dtype=float, shape=num_particles)
hij = ti.Matrix.field(n=3, m=3, dtype=float, shape=n_edge)


num_max_partition = 1
num_max_vertices_per_partition = len(partition_edge[0]) + 1

a_part       = ti.Matrix.field(n=3, m=3, dtype=float, shape=(num_max_partition, num_max_vertices_per_partition))
b_part       = ti.Matrix.field(n=3, m=3, dtype=float, shape=(num_max_partition, num_max_vertices_per_partition))
c_part       = ti.Matrix.field(n=3, m=3, dtype=float, shape=(num_max_partition, num_max_vertices_per_partition))
c_tilde_part = ti.Matrix.field(n=3, m=3, dtype=float, shape=(num_max_partition, num_max_vertices_per_partition))

d_part       = ti.Vector.field(n=3, dtype=float, shape=(num_max_partition, num_max_vertices_per_partition))
d_tilde_part = ti.Vector.field(n=3, dtype=float, shape=(num_max_partition, num_max_vertices_per_partition))
x_part       = ti.Vector.field(n=3, dtype=float, shape=(num_max_partition, num_max_vertices_per_partition))

# indices_np = []
# vis_path_np = []
# for i in range(n_tri):
#     indices_np.append([i, i + 1, i + 2])
#     vis_path_np.append([i, i + 1])
#     vis_path_np.append([i + 1, i + 2])
#
# vis_path_np = np.array(vis_path_np, dtype=int).reshape(-1)
# vis_path = ti.field(dtype=int, shape=vis_path_np.shape[0])
# vis_path.from_numpy(vis_path_np)
#
# num_tri = len(indices_np)
# indices_np = np.array(indices_np, dtype=int)
# indices_np = np.reshape(indices_np, -1)
# indices = ti.field(dtype=int, shape=indices_np.shape[0])
# indices.from_numpy(indices_np)

Dm_inv      = ti.Matrix.field(n=3, m=3, shape=n_tet, dtype=float)
rest_volume = ti.field(shape=n_tet, dtype=float)
mass        = ti.field(shape=num_particles, dtype=float)

@ti.kernel
def reset():

    mass.fill(0.0)

    for i in range(n_tet):
        e0, e1, e2 = tet_edge_indices[3 * i + 0], tet_edge_indices[3 * i + 1], tet_edge_indices[3 * i + 2]

        v00, v01 = edge_indices[2 * e0 + 0], edge_indices[2 * e0 + 1]
        dx0 =  x0[v00] - x0[v01]

        v10, v11 = edge_indices[2 * e1 + 0], edge_indices[2 * e1 + 1]
        dx1 = x0[v10] - x0[v11]

        v20, v21 = edge_indices[2 * e2 + 0], edge_indices[2 * e2 + 1]
        dx2 = x0[v20] - x0[v21]

        Dm_inv[i] = ti.Matrix.cols([dx0, dx1, dx2]).inverse()

        rest_volume[i] = abs(dx0.cross(dx1).dot(dx2)) / 6.0

        volume_part = 0.25 * rest_volume[i]

        mass[v00] += volume_part
        mass[v01] += volume_part
        mass[v20] += volume_part
        mass[v21] += volume_part


@ti.kernel
def compute_y():

    # apply gravity within boundary
    g = ti.Vector([0.0, -9.81, 0.0])
    # g = ti.Vector([0.0, 0.0, 0.0])
    for i in x:
        xi, vi = x[i], v[i]
        x_k[i] = y[i] = xi + vi * dt + g * dt * dt

@ti.kernel
def compute_v(damping: float):
    for i in x:
        v[i] = (1.0 - damping) * (x_k[i] - x[i])/ dt
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
    ids = ti.Vector([0, 1, 2], dt=int)
    # print(ids)
    for i in range(ids.n):
        grad[ids[i]] += k * (x[ids[i]] - x0[ids[i]])
        hii[ids[i]] += k * id3


@ti.func
def ssvd(F):
    U, sig, V = ti.svd(F)
    if U.determinant() < 0:
        for i in ti.static(range(3)):
            U[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    if V.determinant() < 0:
        for i in ti.static(range(3)):
            V[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    return U, sig, V

@ti.func
def volume_projection(sigma: ti.math.mat3) -> ti.math.mat3:

    singular_values = ti.math.vec3(sigma[0, 0], sigma[1, 1], sigma[2, 2])
    nabla_c = ti.math.vec3(0.0)
    # print(singular_values)
    count = 0
    count_max = 20
    threshold = 1e-4
    c = singular_values[0] * singular_values[1] * singular_values[2] - 1.0
    while abs(c) > threshold:

        if count > count_max: break
        c = singular_values[0] * singular_values[1] * singular_values[2] - 1.0
        nabla_c[0] = singular_values[1] * singular_values[2]
        nabla_c[1] = singular_values[0] * singular_values[2]
        nabla_c[2] = singular_values[0] * singular_values[1]
        ld = c / (nabla_c.dot(nabla_c) + 1e-3)
        singular_values -= ld * nabla_c

        singular_values = ti.math.clamp(singular_values, 0.0, 3.0)

        count += 1

    for i in ti.static(range(3)):
        sigma[i, i] = singular_values[i]

    return sigma

@ti.kernel
def compute_grad_and_hessian_stretch(x: ti.template(), k: float):

    id3 = ti.Matrix.identity(dt=float, n=3)
    for i in range(n_tet):
        e0, e1, e2 = tet_edge_indices[3 * i + 0], tet_edge_indices[3 * i + 1], tet_edge_indices[3 * i + 2]

        v00, v01 = edge_indices[2 * e0 + 0], edge_indices[2 * e0 + 1]
        dx0 = x[v00] - x[v01]

        v10, v11 = edge_indices[2 * e1 + 0], edge_indices[2 * e1 + 1]
        dx1 = x[v10] - x[v11]

        v20, v21 = edge_indices[2 * e2 + 0], edge_indices[2 * e2 + 1]
        dx2 = x[v20] - x[v21]

        Ds = ti.Matrix.cols([dx0, dx1, dx2])
        B = Dm_inv[i]
        F = Ds @ B

        U, sigma, V = ssvd(F)

        R = U @ V.transpose()
        P = F - R
        test = (P.transpose() @ P).trace()
        C = ti.sqrt(test)

        eps = 0.0
        if C < 1e-3:
            eps = 1e-3
        dCdx = (F - R) @ B.transpose() / (C + eps)

        grad0 = ti.math.vec3(dCdx[:, 0])
        grad1 = ti.math.vec3(dCdx[:, 1])
        grad2 = ti.math.vec3(dCdx[:, 2])

        schur = grad0.dot(grad0) + grad1.dot(grad1) + grad2.dot(grad2)
        ld = -C / (schur + 1e-3)

        p01 = ld * grad0
        p12 = ld * grad1
        p23 = ld * grad2


        weight = k * rest_volume[i]

        grad[v00] -= weight * p01
        grad[v01] += weight * p01

        hii[v00] += weight * id3
        hii[v01] += weight * id3

        grad[v10] -= weight * p12
        grad[v11] += weight * p12

        hii[v10] += weight * id3
        hii[v11] += weight * id3

        grad[v20] -= weight * p23
        grad[v21] += weight * p23

        hii[v20] += weight * id3
        hii[v21] += weight * id3

        hij[e0] = weight * id3
        hij[e1] = weight * id3
        hij[e2] = weight * id3


@ti.kernel
def compute_grad_and_hessian_volume(x: ti.template(), k: float):
    id3 = ti.Matrix.identity(dt=float, n=3)
    for i in range(n_tet):
        e0, e1, e2 = tet_edge_indices[3 * i + 0], tet_edge_indices[3 * i + 1], tet_edge_indices[3 * i + 2]

        v00, v01 = edge_indices[2 * e0 + 0], edge_indices[2 * e0 + 1]
        dx0 = x[v00] - x[v01]

        v10, v11 = edge_indices[2 * e1 + 0], edge_indices[2 * e1 + 1]
        dx1 = x[v10] - x[v11]

        v20, v21 = edge_indices[2 * e2 + 0], edge_indices[2 * e2 + 1]
        dx2 = x[v20] - x[v21]

        Ds = ti.Matrix.cols([dx0, dx1, dx2])
        B = Dm_inv[i]
        F = Ds @ B

        U, sigma, V = ssvd(F)

        sigma_vol = volume_projection(sigma)

        F_proj = U @ sigma_vol @ V.transpose()
        P = F - F_proj
        test = (P.transpose() @ P).trace()
        C = ti.sqrt(test)

        eps = 0.0
        if C < 1e-3:
            eps = 1e-3
        dCdx = (F - F_proj) @ B.transpose() / (C + eps)

        grad0 = ti.math.vec3(dCdx[:, 0])
        grad1 = ti.math.vec3(dCdx[:, 1])
        grad2 = ti.math.vec3(dCdx[:, 2])

        schur = grad0.dot(grad0) + grad1.dot(grad1) + grad2.dot(grad2)
        ld = -C / (schur + 1e-3)

        p01 = ld * grad0
        p12 = ld * grad1
        p23 = ld * grad2

        weight = k * rest_volume[i]

        grad[v00] -= weight * p01
        grad[v01] += weight * p01

        hii[v00] += weight * id3
        hii[v01] += weight * id3

        grad[v10] -= weight * p12
        grad[v11] += weight * p12

        hii[v10] += weight * id3
        hii[v11] += weight * id3

        grad[v20] -= weight * p23
        grad[v21] += weight * p23

        hii[v20] += weight * id3
        hii[v21] += weight * id3

        hij[e0] = weight * id3
        hij[e1] = weight * id3
        hij[e2] = weight * id3


@ti.kernel
def substep_Jacobi(Px: ti.template(), x: ti.template()):

    for i in range(num_particles):
        Px[i] = hii[i].inverse() @ x[i]

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
        # print(size_pi)
        n_verts_pi = size_pi + 1
        ti.loop_config(serialize=True)
        for i in range(size_pi):
            ei = partitioned_set[pi, i]
            # print(ei)
            vi, vj = edge_indices[2 * ei + 0], edge_indices[2 * ei + 1]
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
            vi = edge_indices[2 * ei + 0]
            P_grad[vi] += x_part[pi, i]

        ei = partitioned_set[pi, size_pi - 1]
        vi = edge_indices[2 * ei + 1]
        Px[vi] += x_part[pi, size_pi]

    # for i in range(num_particles):
    #     if num_dup[i] > 0:
    #         Px[i] /= num_dup[i]
    #     else:
    #         Px[i] = ti.math.inverse(hii[i]) @ x[i]

@ti.kernel
def add(ret: ti.template(), x: ti.template(), y: ti.template(), scale: float):
    for i in x:
        ret[i] = x[i] + y[i] * scale

@ti.kernel
def infinity_norm(x: ti.template()) -> ti.f32:

    value = 0.0
    for i in x:
        ti.atomic_max(value, x[i].norm())

    return value

def forward():

    # print(x_dup.shape)
    compute_y()
    k = pow(10.0, PR) * dt ** 2
    k_col = pow(10.0, PR) * dt ** 2
    k_at = pow(10.0, PR + 1) * dt ** 2
    termination_condition = pow(10.0, -threshold)
    itr_cnt = 0

    for _ in range(num_iters):

        compute_grad_and_hessian_momentum(x_k)
        # compute_grad_and_hessian_aggregate(x_k_dup, k=k_at)
        # # if enable_attachment:
        compute_grad_and_hessian_attachment(x_k, k=k_at)
        #
        compute_grad_and_hessian_stretch(x_k, k=k)
        compute_grad_and_hessian_volume(x_k, k=k)
        #
        if solver_type == 0:
            substep_Euler(P_grad, grad)
        #
        elif solver_type == 1:
            substep_Jacobi(P_grad, grad)

        beta = 0.0

        # print(beta)
        # add(dx, P_grad, dx_k, -beta)
    #
    #     # alpha = 1.0
    #     # alpha_ccd = collision_aware_line_search(x_k, dx, radius, center)
    #     # print(alpha_ccd)
    #
    #     # scale(dx, dx, alpha_ccd)
    #     P_grad_dup.fill(0.0)
        add(x_k, x_k, P_grad, -1.0)
        inf_norm = infinity_norm(P_grad)
    #
    #     dx_k.copy_from(dx)
    #     grad_k.copy_from(grad)
    #     P_grad_k.copy_from(P_grad)
    #
        itr_cnt += 1
    #     if print_stat:
    #         print(inf_norm)
    #
        if inf_norm < termination_condition:
    #
    #         if print_stat:
    #             print("conv iter: ,", itr_cnt)
            break

    # print(itr_cnt)
    compute_v(0.01)
    # compute_velocity_aggregate(v)

    return itr_cnt



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

conv_it_total = 0
elapsed_time_total = 0

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
            x.copy_from(x0)
            # copy_to_dup(x_dup, x0)
            # v_dup.fill(0.0)
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
    # scene.mesh(vertices=x, indices=indices, show_wireframe=True)

    if enable_lines:
        scene.lines(vertices=x, indices=edge_indices, width=2.0, color=(1.0, 0.0, 0.0))

    camera.track_user_inputs(window, movement_speed=0.4, hold_key=ti.ui.RMB)
    canvas.scene(scene)
    window.show()