from turtledemo.sorting_animate import partition

import meshio
import networkx as nx
from PIL.ImagePalette import random
from pyquaternion.quaternion import Quaternion
import taichi as ti
from timeit import default_timer as timer
import numpy as np

from math_utils import elastic_util as e_util
import matplotlib as mpl

from tqdm import tqdm
import time

from test2 import partitioned_set

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

dt = 0.03
n_tri = 1
size = 0.5
move = np.array([[1.0, 0.0, -1.0], [0., 0.0, 1.0]], dtype=float)
x_np = np.array([[0.0, 0.0, 0.0], [0., 0.0, 1.0]])
for i in range(n_tri):
    tmp = x_np[-1] + move[i % 2]
    x_np = np.append(x_np, tmp)
    x_np = x_np.reshape(-1, 3)


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
# hij = ti.Matrix.field(n=3, m=3, dtype=float, shape=num_edges)


indices_np = []
vis_path_np = []
for i in range(n_tri):
    indices_np.append([i, i + 1, i + 2])
    vis_path_np.append([i, i + 1])
    vis_path_np.append([i + 1, i + 2])

vis_path_np = np.array(vis_path_np, dtype=int).reshape(-1)
vis_path = ti.field(dtype=int, shape=vis_path_np.shape[0])
vis_path.from_numpy(vis_path_np)

num_tri = len(indices_np)
indices_np = np.array(indices_np, dtype=int)
indices_np = np.reshape(indices_np, -1)
indices = ti.field(dtype=int, shape=indices_np.shape[0])
indices.from_numpy(indices_np)

Dm_inv      = ti.Matrix.field(n=3, m=3, shape=num_tri, dtype=float)
rest_volume = ti.field(shape=num_tri, dtype=float)
mass        = ti.field(shape=num_particles, dtype=float)

@ti.kernel
def reset():

    mass.fill(0.0)

    for i in range(num_tri):
        vi, vj, vk = indices[3 * i + 0], indices[3 * i + 1], indices[3 * i + 2]
        xji = x0[vj] - x0[vi]
        xkj = x0[vk] - x0[vj]

        cross = xji.cross(xkj)
        area = 0.5 * cross.norm()
        n = cross / area

        rest_volume[i] = abs(cross.dot(n)) / 6.0
        Dm_inv[i] = ti.math.inverse(ti.Matrix.cols([xji, xkj, n]))

        area_part = area / 3.0

        mass[vi] += area_part
        mass[vj] += area_part
        mass[vk] += area_part


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
    ids = ti.Vector([0, 1], dt=int)
    # print(ids)
    for i in range(ids.n):
        grad[ids[i]] += k * (x[ids[i]] - x0[ids[i]])
        hii[ids[i]] += k * id3


@ti.func
def ssvd(F):
    U, sig, V = ti.svd(F)
    if U.determinant() < 0:
        for i in ti.static(range(3)): U[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    if V.determinant() < 0:
        for i in ti.static(range(3)): V[i, 2] *= -1
        sig[2, 2] = -sig[2, 2]
    return U, sig, V




@ti.kernel
def compute_grad_and_hessian_stretch(x: ti.template(), k: float):

    for i in range(num_tri):
        vi, vj, vk = indices[3 * i + 0], indices[3 * i + 1], indices[3 * i + 2]
        xji = x[vj] - x[vi]
        xkj = x[vk] - x[vj]

        cross = xji.cross(xkj)
        area = 0.5 * cross.norm()
        n = cross / area

        Ds = ti.Matrix.cols([xji, xkj, n])
        F = Ds @ Dm_inv[i]

        P = e_util.compute_dPsidF_FCR(F, k, 0.0)
        P_vec = e_util.flatten_matrix(P)
        dFdx = e_util.compute_dFdx_tmp(Dm_inv[i])
        grad_vec = dFdx @ P_vec

        d2PsidF2 = e_util.compute_d2PsidF2_FCR_filter(F, k, 0.0)
        # H = (dFdx.transpose()) @ d2PsidF2 @ dFdx

        ids = ti.Vector([vi, vj, vk], dt=int)
        for l in ti.static(range(3)):
            for j in ti.static(range(3)):
                grad[ids[l]] += grad_vec[3 * l + j]

            # for j, k in ti.static(range(3)):
            #     hii[ids[l]][j, k] += H[3 * l + j, 3 * l + k]



@ti.kernel
def substep_Jacobi(Px: ti.template(), x: ti.template()):

    for i in range(num_particles):
        Px[i] = hii[i].inverse() @ x[i]

@ti.kernel
def add(ret: ti.template(), x: ti.template(), y: ti.template(), scale: float):
    for i in x:
        ret[i] = x[i] + y[i] * scale

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
        #
        # if solver_type == 0:
        #     substep_Euler(P_grad_dup, grad_dup)
        #
        # elif solver_type == 1:
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
    #     inf_norm = infinity_norm(P_grad_dup)
    #
    #     dx_k.copy_from(dx)
    #     grad_k.copy_from(grad)
    #     P_grad_k.copy_from(P_grad)
    #
        itr_cnt += 1
    #     if print_stat:
    #         print(inf_norm)
    #
        # if inf_norm < termination_condition:
    #
    #         if print_stat:
    #             print("conv iter: ,", itr_cnt)
    #         break

    # print(itr_cnt)
    compute_v()
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
    scene.mesh(vertices=x, indices=indices, show_wireframe=True)

    if enable_lines:
        scene.lines(vertices=x, indices=vis_path, width=2.0, color=(1.0, 0.0, 0.0))

    camera.track_user_inputs(window, movement_speed=0.4, hold_key=ti.ui.RMB)
    canvas.scene(scene)
    window.show()