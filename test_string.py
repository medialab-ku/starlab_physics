# Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.
# Taichi implementation by Ye Kuang (k-ye)

import math

import numpy as np
import matplotlib.pyplot as plt
import taichi as ti
from timeit import default_timer as timer

from test2 import print_stats

ti.init(arch=ti.gpu)

screen_res = (1500, 1500)
screen_to_world_ratio = 10.0
boundary = (
    screen_res[0] / screen_to_world_ratio,
    screen_res[1] / screen_to_world_ratio,
)
cell_size = 2.51
cell_recpr = 1.0 / cell_size


def round_up(f, s):
    return (math.floor(f * cell_recpr / s) + 1) * s

grid_size = (round_up(boundary[0], 1), round_up(boundary[1], 1))

dim = 2
bg_color = 0x112F41
particle_color = ti.math.vec3(1.0, 1.0, 1.0)
# print(particle_color)
boundary_particle_color = 0xFFC433
# boundary_color = 0x068587
num_particles_x = 3
num_particles_y = 40
num_particles = num_particles_x * num_particles_y

num_max_partition = num_particles_x + num_particles_y + 2 * (num_particles_y - 1)
num_partition = ti.field(int, shape=1)


conv_iter_tot = 0
frame_cnt = 0
# if num_particles_x > 1:
#     num_partition += num_particles_x
#
# if num_particles_y > 1:
#     num_partition += num_particles_y

max_num_particles_per_cell = 80
max_num_neighbors = 50
time_delta = 1.0 / 30.0
epsilon = 1e-5
particle_radius = 10.0
particle_radius_in_world = particle_radius / screen_to_world_ratio

# PBF params
h_ = 2.1
# mass = 1.0
# rho0 = 1.0
lambda_epsilon = 100.0
pbf_num_iters = 500
threshold = 2
mass_ratio = 10.0
PR = 5
solver_type = 0
print_stat = False
enable_pncg = False

use_heatmap = False
definiteness_fix = False
corr_deltaQ_coeff = 0.3
corrK = 0.001
# Need ti.pow()
# corrN = 4.0
neighbor_radius = h_ * 1.05
poly6_factor = 315.0 / 64.0 / math.pi
spiky_grad_factor = -45.0 / math.pi

# old_positions = ti.Vector.field(dim, float)
# positions = ti.Vector.field(dim, float)
# velocities = ti.Vector.field(dim, float)
grid_num_particles = ti.field(int)
grid2particles = ti.field(int)
particle_num_neighbors = ti.field(int)
particle_neighbors = ti.field(int)

particle_num_neighbors_rest = ti.field(int)
particle_neighbors_rest = ti.field(int)

num_indices = ti.field(dtype=int, shape=1)
num_max_indices = num_particles_x * (num_particles_y - 1) + num_particles_y * (num_particles_x - 1) + 2 * num_particles_x * (num_particles_y - 1)

indices     = ti.field(int, shape=2 * num_max_indices)
indices_dup = ti.field(int, shape=2 * num_max_indices)


floor_indices     = ti.field(int, shape=4)
floor_positions   = ti.Vector.field(n=2, dtype=ti.f32, shape=4)
floor_pos_window  = ti.Vector.field(n=2, dtype=ti.f32, shape=4)

floor_position_colors = ti.Vector.field(n=3, dtype=ti.f32, shape=4)

l0 = ti.field(float, shape=2 * num_max_indices)
hij = ti.Matrix.field(n=2, m=2, dtype=float, shape=num_max_indices)
# lambdas = ti.field(float)
# position_deltas = ti.Vector.field(dim, float)
# # 0: x-pos, 1: timestep in sin()

board_states = ti.Vector.field(2, float, shape=())
#
# ti.root.dense(ti.i, num_particles).place(old_positions, positions, velocities)

grid_snode = ti.root.dense(ti.ij, grid_size)
grid_snode.place(grid_num_particles)
grid_snode.dense(ti.k, max_num_particles_per_cell).place(grid2particles)
# print(grid2particles.shape)

nb_node = ti.root.dense(ti.i, num_particles)
nb_node.place(particle_num_neighbors, particle_num_neighbors_rest)
nb_node.dense(ti.j, max_num_neighbors).place(particle_neighbors, particle_neighbors_rest)
# ti.root.dense(ti.i, num_particles).place(lambdas, position_deltas)

# print(particle_neighbors.shape)
# ti.root.place(board_states)
# print(board_states.shape)
old_positions = ti.Vector.field(dim, float, shape=num_particles)
positions = ti.Vector.field(dim, float, shape=num_particles)
x_k        = ti.Vector.field(dim, float, shape=num_particles)
x_k_plus_1 = ti.Vector.field(dim, float, shape=num_particles)
y = ti.Vector.field(dim, float, shape=num_particles)
hii = ti.Matrix.field(n=2, m=2, dtype=float, shape=num_particles)


a = ti.field(float, shape=num_particles)
b = ti.field(float, shape=num_particles)
c = ti.field(float, shape=num_particles)
c_tilde = ti.field(float, shape=num_particles)
gii = ti.Vector.field(dim, float, shape=num_particles)
d = ti.Vector.field(dim, float, shape=num_particles)
d_tilde = ti.Vector.field(dim, float, shape=num_particles)

a3 = ti.Matrix.field(m=2, n=2, dtype=float, shape=num_particles)
b3 = ti.Matrix.field(m=2, n=2,dtype=float, shape=num_particles)
c3 = ti.Matrix.field(m=2, n=2,dtype=float, shape=num_particles)
c3_tilde = ti.Matrix.field(m=2, n=2,dtype=float, shape=num_particles)

x0 = ti.Vector.field(dim, float, shape=num_particles)
positions_window = ti.Vector.field(dim, float, shape=num_particles)
colors = ti.Vector.field(3, float, shape=num_particles)
heat_map = ti.Vector.field(3, float, shape=num_particles)

velocities = ti.Vector.field(dim, float, shape=num_particles)
old_velocities = ti.Vector.field(dim, float, shape=num_particles)



# grid_num_particles = ti.field(int, shape=grid_size)
# grid2particles = ti.field(int, shape=(grid_size[0], grid_size[1], max_num_particles_per_cell))
# particle_num_neighbors = ti.field(int, shape=num_particles)
# particle_neighbors = ti.field(int)
lambdas = ti.field(float, shape=num_particles)
rho0 = ti.field(float, shape=num_particles)
mass = ti.field(float, shape=num_particles)
V0 = ti.field(float, shape=num_particles)
invDm = ti.Matrix.field(n=2, m=2, dtype=float, shape=num_particles)
material_type = ti.field(float, shape=num_particles)
P_grad = ti.Vector.field(dim, float, shape=num_particles)
P_grad_delta = ti.Vector.field(dim, float, shape=num_particles)
P_grad_prev = ti.Vector.field(dim, float, shape=num_particles)
dx = ti.Vector.field(dim, float, shape=num_particles)
dx_prev = ti.Vector.field(dim, float, shape=num_particles)

num_dup = ti.field(float, shape=num_particles)
grad = ti.Vector.field(dim, float, shape=num_particles)
grad_prev = ti.Vector.field(dim, float, shape=num_particles)
grad_delta = ti.Vector.field(dim, float, shape=num_particles)


Hp = ti.Vector.field(dim, float, shape=num_particles)

# Dm_inv = ti.Matrix.field(dim, float, shape=num_particles)
num_boundary_particles_x = 40
num_boundary_particles = 10 * num_boundary_particles_x
boundary_positions = ti.Vector.field(dim, float, shape=num_boundary_particles)
boundary_positions_window = ti.Vector.field(dim, float, shape=num_boundary_particles)

# num_boundary_particles_x = 40
# num_boundary_particles = 10 * num_boundary_particles_x
# ti.root.dense(ti.i, num_boundary_particles).place(boundary_positions, boundary_positions_window)
# board_states = ti.Vector.field(2, float)

K = ti.linalg.SparseMatrixBuilder(2 * num_particles, 2 * num_particles, max_num_triplets=10000000)
ndarr = ti.ndarray(ti.f32, shape=2 * num_particles)

num_max_edges_per_partition = num_particles_y - 1 if num_particles_y > num_particles_x else num_particles_x - 1
partitioned_set = ti.field(dtype=int, shape=(num_max_partition, num_max_edges_per_partition))
num_edges_per_partition = ti.field(dtype=int, shape=num_max_partition)
num_max_vertices_per_partition = num_max_edges_per_partition + 1
print(num_max_vertices_per_partition)

a_part       = ti.Matrix.field(n=2, m=2, dtype=float, shape=(num_max_partition, num_max_vertices_per_partition))
b_part       = ti.Matrix.field(n=2, m=2, dtype=float, shape=(num_max_partition, num_max_vertices_per_partition))
c_part       = ti.Matrix.field(n=2, m=2, dtype=float, shape=(num_max_partition, num_max_vertices_per_partition))
c_tilde_part = ti.Matrix.field(n=2, m=2, dtype=float, shape=(num_max_partition, num_max_vertices_per_partition))

d_part       = ti.Vector.field(n=2, dtype=float, shape=(num_max_partition, num_max_vertices_per_partition))
d_tilde_part = ti.Vector.field(n=2, dtype=float, shape=(num_max_partition, num_max_vertices_per_partition))
x_part       = ti.Vector.field(n=2, dtype=float, shape=(num_max_partition, num_max_vertices_per_partition))

@ti.kernel
def ndarr_to_vec_field(ndarr: ti.types.ndarray(), vec_field: ti.template()):

    for i in vec_field:
        vec_i = ti.math.vec2([ndarr[2 * i + 0], ndarr[2 * i + 1]])
        vec_field[i] = vec_i

@ti.kernel
def vec_field_to_ndarr(vec_field: ti.template(), ndarr: ti.types.ndarray()):

    for i in vec_field:
        vec_i = vec_field[i]
        ndarr[2 * i + 0] = vec_i[0]
        ndarr[2 * i + 1] = vec_i[1]

def poly6_value(s, h):
    result = 0.0
    if 0 <= s and s < h:
        x = (h * h - s * s) / (h * h * h)
        result = poly6_factor * x * x * x
    return result


@ti.func
def spiky_gradient(r, h):
    result = ti.Vector([0.0, 0.0])
    r_len = r.norm()
    if 0 < r_len and r_len < h:
        x = (h - r_len) / (h * h * h)
        g_factor = spiky_grad_factor * x * x
        result = r * g_factor / r_len
    return result


@ti.func
def compute_scorr(pos_ji):
    # Eq (13)
    x = poly6_value(pos_ji.norm(), h_) / poly6_value(corr_deltaQ_coeff * h_, h_)
    # pow(x, 4)
    x = x * x
    x = x * x
    return (-corrK) * x

@ti.func
def get_cell(pos) -> ti.math.ivec3:
    return int(pos * cell_recpr)


@ti.func
def is_in_grid(c):
    # @c: Vector(i32)
    return 0 <= c[0] and c[0] < grid_size[0] and 0 <= c[1] and c[1] < grid_size[1]


@ti.func
def confine_position_to_boundary(p):
    bmin = h_
    bmax = ti.Vector([board_states[None][0], boundary[1]]) - h_

    # print(bmax)
    for i in ti.static(range(dim)):
        # Use randomness to prevent particles from sticking into each other after clamping
        if p[i] <= bmin:
            p[i] = bmin + epsilon * ti.random()
        elif bmax[i] <= p[i]:
            p[i] = bmax[i] - epsilon * ti.random()
    return p


@ti.kernel
def move_board():
    # probably more accurate to exert force on particles according to hooke's law.
    b = board_states[None]
    b[1] += 1.0
    period = 90
    vel_strength = 8.0
    if b[1] >= 2 * period:
        b[1] = 0
    b[0] += -ti.sin(b[1] * np.pi / period) * vel_strength * time_delta
    board_states[None] = b


@ti.kernel
def compute_y():

    # save old positions
    for i in positions:
        old_positions[i] = positions[i]
        old_velocities[i] = velocities[i]

    # apply gravity within boundary
    for i in positions:
        g = ti.Vector([0.0, -9.81])
        pos, vel = positions[i], velocities[i]
        # vel += mass[i] * g * time_delta
        # pos += vel * time_delta
        positions[i] = y[i] = pos + vel * time_delta + g * time_delta * time_delta


@ti.kernel
def substep():
    # compute lambdas
    # Eq (8) ~ (11)
    for p_i in positions:
        pos_i = positions[p_i]
        grad_i = ti.math.vec2(0.0)
        sum_gradient_sqr = 0.0
        density_constraint = 0.0

        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            pos_ji = pos_i - positions[p_j]
            grad_j = spiky_gradient(pos_ji, h_)
            grad_i += grad_j
            sum_gradient_sqr += grad_j.dot(grad_j)
            # Eq(2)
            density_constraint += mass[p_j] * poly6_value(pos_ji.norm(), h_)
        # Eq(1)
        density_constraint = (density_constraint / rho0[p_i])
        sum_gradient_sqr += grad_i.dot(grad_i)
        lambdas[p_i] = (-density_constraint) / (sum_gradient_sqr + lambda_epsilon)
    # compute position deltas
    # Eq(12), (14)
    for p_i in positions:
        pos_i = positions[p_i]
        lambda_i = lambdas[p_i]

        pos_delta_i = ti.math.vec2(0.0)
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            # if p_j < 0:
            #     break
            lambda_j = lambdas[p_j]
            pos_ji = pos_i - positions[p_j]
            scorr_ij = compute_scorr(pos_ji)
            pos_delta_i += (lambda_i + lambda_j) * spiky_gradient(pos_ji, h_)

        pos_delta_i /= rho0[p_i]
        # position_deltas[p_i] = pos_delta_i
    # apply position deltas
    # for i in positions:
        positions[p_i] += pos_delta_i


@ti.func
def ssvd(F):
    U, sig, V = ti.svd(F)
    if U.determinant() < 0:
        for i in ti.static(range(2)): U[i, 1] *= -1
        sig[1, 1] = -sig[1, 1]
    if V.determinant() < 0:
        for i in ti.static(range(2)): V[i, 1] *= -1
        sig[1, 1] = -sig[1, 1]
    return U, sig, V

@ti.kernel
def substep_pbd_mass_spring(k: float):

    hii.fill(0.0)
    gii.fill(0.0)

    for i in range(num_particles_x - 1):
        v0, v1 = indices[2 * i + 0], indices[2 * i + 1]
        x10 = positions[v0] - positions[v1]
        lij = x10.norm()

        C = (lij - l0[i])
        nabla_C = x10.normalized()
        schur = mass[v0] + mass[v1]

        ld = k * C / (k * schur + 1.0)

        gii[v0] -= mass[v0] * ld * nabla_C
        gii[v1] += mass[v1] * ld * nabla_C

        hii[v0] += 1.0
        hii[v1] += 1.0

    for i in range(num_particles_x):
        positions[i] += gii[i] / hii[i]

@ti.kernel
def substep_Euler(Px: ti.template(), x: ti.template()):

    Px.fill(0.0)
    b_part.fill(0.0)
    a_part.fill(0.0)
    c_part.fill(0.0)

    # ti.loop_config(serialize=True)
    for pi in range(num_partition[0]):
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
    for pi in range(num_partition[0]):
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
        Px[i] /= num_dup[i]

@ti.func
def barrier_g(d, dHat, kappa):
    t2 = d - dHat
    g = kappa * (t2 * ti.log(d / dHat) * (-2.0) - (t2 ** 2) / d)
    return g

@ti.func
def barrier_H(d, dHat, kappa):
    H = kappa * (-2) * ti.log(d / dHat) - 4 + 4 * dHat / d + (d - dHat) ** 2 / d ** 2
    return H


@ti.kernel
def compute_grad_and_hessian_momentum():

    id3 = ti.Matrix.identity(dt=float, n=2)
    for i in range(num_particles):
        grad[i] = (positions[i] - y[i])
        hii[i] = id3

@ti.kernel
def compute_grad_and_hessian_attachment(k: float):

    id3 = ti.Matrix.identity(dt=float, n=2)
    ids = ti.Vector([0, num_particles_x - 1], dt=int)
    # print(ids)
    for i in range(2):
        grad[ids[i]] += k * (positions[ids[i]] - x0[ids[i]])
        hii[ids[i]] += k * id3

@ti.kernel
def compute_grad_and_hessian_spring(k: float):

    for i in range(num_indices[0]):
        v0, v1 = indices[2 * i + 0], indices[2 * i + 1]
        # print(v0, v1)
        x01 = positions[v0] - positions[v1]
        n = x01.normalized()
        l = x01.norm()
        dp01 = x01 - l0[i] * n

        grad[v0] += k * dp01
        grad[v1] -= k * dp01
        alpha = l0[i] / l
        D = ti.math.mat2([1.0, 0.0, 0.0, abs(1.0 - alpha)])
        t = ti.math.vec2(n[1], -n[0])
        P = ti.math.mat2([n[0], t[0], n[1], t[1]])
        B = k * (P @ D @ P.inverse())

        hii[v0] += B
        hii[v1] += B
        hij[i] = B

@ti.kernel
def compute_grad_and_hessian_collision(k: float, dhat: float):


    id3 = ti.Matrix.identity(dt=float, n=2)
    floor_y = 0.3 * boundary[1] + dhat

    for i in range(num_particles):
        xi = positions[i]
        if xi[1] < floor_y:
            p = ti.math.vec2([xi[0], floor_y])
            grad[i] += k * (positions[i] - p)
            hii[i]  += k * id3


@ti.kernel
def compute_energy(x: ti.template(), k: float) -> float:

    E = 0.0

    id3 = ti.Matrix.identity(dt=float, n=2)
    for i in range(num_particles):
        grad[i] = (x[i] - y[i])
        hii[i] = id3

    stiffness = 1e5
    ids = ti.Vector([0, num_particles_x - 1], dt=int)
    # print(ids)
    for i in range(2):

        grad[ids[i]] += stiffness * (x[ids[i]] - x0[ids[i]])
        hii[ids[i]] += stiffness * id3

    for i in range(num_indices[0]):
        v0, v1 = indices[2 * i + 0], indices[2 * i + 1]
        # print(v0, v1)
        x01 = x[v0] - x[v1]
        n = x01.normalized()
        l = x01.norm()

    return E

@ti.kernel
def substep_Jacobi(Px: ti.template(), x: ti.template()):

    for i in range(num_particles):
        Px[i] = hii[i].inverse() @ x[i]

def substep_Newton(Px, x):

    construct_Hessian(K)
    H = K.build()
    solver = ti.linalg.SparseSolver(solver_type="LLT")
    solver.analyze_pattern(H)

    vec_field_to_ndarr(x, ndarr)
    Px_ndarr = solver.solve(ndarr)
    ndarr_to_vec_field(Px_ndarr, Px)


@ti.kernel
def construct_Hessian(A: ti.types.sparse_matrix_builder()):

    for i in range(num_particles):
        B = hii[i]
        for m, n in ti.ndrange(2, 2):
            A[2 * i + m, 2 * i + n] += B[m, n]

    for i in range(num_indices[0]):
        v0, v1 = indices[2 * i + 0], indices[2 * i + 1]
        B = hij[i]
        for m, n in ti.ndrange(2, 2):
            A[2 * v1 + m, 2 * v0 + n] -= B[m, n]
            A[2 * v0 + m, 2 * v1 + n] -= B[m, n]


@ti.kernel
def matrix_free_Hx(Hx: ti.template(), x: ti.template()):

    for i in range(num_particles):
        Hx[i] = x[i]

    stiffness = 1e5
    ids = ti.Vector([0, num_particles_x - 1], dt=int)
    # print(ids)

    for i in range(2):
        Hx[ids[i]] += stiffness * x[ids[i]]

    for i in range(num_indices[0]):
        v0, v1 = indices[2 * i + 0], indices[2 * i + 1]
        B = hij[i]
        Bxij = B @ (x[v0] - x[v1])
        Hx[v0] += Bxij
        Hx[v1] -= Bxij


@ti.kernel
def epilogue():
    # confine to boundary
    for i in positions:
        pos = positions[i]
        positions[i] = pos
    # update velocities
    for i in positions:
        velocities[i] = (positions[i] - old_positions[i]) / time_delta

    # velocities[0] = 0

    # for i in positions:
    #     positions[i] = old_positions[i] + velocities[i] * time_delta

    # no vorticity/xsph because we cannot do cross product in 2D...

@ti.kernel
def add(ret: ti.template(), x: ti.template(), y: ti.template(), scale: float):
    for i in x:
        ret[i] = x[i] + y[i] * scale

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

def run():
    compute_y()

    k = pow(10.0, PR) * time_delta ** 2
    iter_cnt = 0
    termination_condition = pow(10.0, -threshold)
    # solver_init = ti.linalg.SparseSolver(solver_type="LLT")
    if print_stat:
        print("inf. norm")

    for _ in range(pbf_num_iters):

        compute_grad_and_hessian_momentum()
        compute_grad_and_hessian_attachment(k=1e8)

        compute_grad_and_hessian_collision(k=1e5, dhat=dhat)
        compute_grad_and_hessian_spring(k=k)

        if solver_type == 0:
            substep_Euler(P_grad, grad)

        elif solver_type == 1:
            substep_Jacobi(P_grad, grad)

        elif solver_type == 2:
            substep_Newton(P_grad, grad)

        beta = 0.0

        if iter_cnt > 0 and enable_pncg:
            # print("test")
            add(grad_delta, grad, grad_prev, -1.0)
            add(P_grad_delta, P_grad, P_grad_prev, -1.0)
            beta = compute_beta(grad, P_grad_delta, grad_delta, dx_prev)


        add(dx, P_grad, dx_prev, -beta)

        alpha = 1.0
        add(positions, positions, dx, -alpha)

        iter_cnt += 1

        P_grad_prev.copy_from(P_grad)
        dx_prev.copy_from(dx)

        inf_norm = infinity_norm(dx)

        if print_stat:
            print(inf_norm)

        if  inf_norm < time_delta * termination_condition:
            break

    epilogue()

    return iter_cnt

@ti.kernel
def render_kernel(pos: ti.template(), pos_window: ti.template(), ratio: float, res_x: float, res_y: float):
    # print(ratio, res_x, res_y)
    # print(pos[0][0])
    for i in pos:
        pos_window[i][0] = (ratio / res_x) * pos[i][0]
        pos_window[i][1] = (ratio / res_y) * pos[i][1]




def render(gui):
    # gui.clear(bg_color)
    render_kernel(positions, positions_window, screen_to_world_ratio, screen_res[0], screen_res[1])
    # pos_np = positions.to_numpy()
    # boundary_pos_np = boundary_positions.to_numpy()
    # for j in range(dim):
    #     pos_np[:, j] *= screen_to_world_ratio / screen_res[j]
    #     boundary_pos_np[:, j] *= screen_to_world_ratio / screen_res[j]
    #
    # colors.fill(1.0)
    gui.circles(positions_window, radius=particle_radius,  per_vertex_color=colors)
    # gui.circles(boundary_pos_np, radius=particle_radius, color=boundary_particle_color)
    # gui.rect(
    #     (0, 0),
    #     (board_states[None][0] / boundary[0], 1),
    #     radius=1.5,
    #     color=boundary_color,
    # )
    # gui.show()


@ti.func
def outer_product(u: ti.math.vec2, v: ti.math.vec2)->ti.math.mat2:

    uvT = ti.math.mat2(0.0)
    for I in ti.grouped(ti.ndrange((0, 2), (0, 2))):
        uvT[I] += u[I[0]] * v[I[1]]

    return uvT


dhat = 0.01 * boundary[1]

@ti.kernel
def reset(mass_ratio: float):
    # print(boundary)

    # dhat = 0.01 * boundary[1]

    floor_indices[0] = 0
    floor_indices[1] = 1

    floor_positions[0] = ti.math.vec2([0,  0.3 * boundary[1]])
    floor_positions[1] = ti.math.vec2([boundary[0],  0.3 * boundary[1]])

    floor_indices[2] = 2
    floor_indices[3] = 3

    floor_positions[2] = ti.math.vec2([0, 0.3 * boundary[1] + dhat])
    floor_positions[3] = ti.math.vec2([boundary[0], 0.3 * boundary[1] + dhat])

    floor_position_colors[0] = ti.math.vec3([1.0, 0.0, 0.0])
    floor_position_colors[1] = ti.math.vec3([1.0, 0.0, 0.0])

    floor_position_colors[2] = ti.math.vec3([0.0, 1.0, 0.0])
    floor_position_colors[3] = ti.math.vec3([0.0, 1.0, 0.0])

    num_dup.fill(0.0)
    # off_test = ti.math.vec2([0.0, boundary[1] * 0.05])
    delta = h_ * 1.0
    for i in range(num_particles):
        # delta = h_ * 0.2
        offs = ti.Vector([boundary[0] * 0.9, boundary[1] * 0.9])
        x = i % num_particles_x
        y = i // num_particles_x
        positions[i] = -ti.Vector([x, y]) * delta + offs
        # mass[i] = 1.0
        # if 0 <= y < 5:
        colors[i] = ti.math.vec3(255.0, 128.0, 0.0) / 255.0
        material_type[i] = 0
        x0[i] = positions[i]

    cnt = 0
    partition_cnt = 0

    ti.loop_config(serialize=True)
    for j in range(num_particles_y):
        num_edges_per_partition[j] = num_particles_x - 1
        partition_cnt += 1
        for i in range(num_particles_x - 1):
            partitioned_set[j, i] = cnt
            # print(offset)
            # print(i + num_verts_y * j, i + num_verts_y * j + 1)
            indices[2 * cnt + 0] = i + num_particles_x * j
            indices[2 * cnt + 1] = i + num_particles_x * j + 1
            l0[cnt] = delta
            cnt += 1

    partition_offset = partition_cnt

    ti.loop_config(serialize=True)
    for i in range(num_particles_x):

        num_edges_per_partition[i + partition_offset] = num_particles_y - 1
        partition_cnt += 1
        for j in range(num_particles_y - 1):
            partitioned_set[i + partition_offset, j] = cnt
            indices[2 * cnt + 0] = i + num_particles_x * j
            indices[2 * cnt + 1] = i + num_particles_x * (j + 1)
            l0[cnt] = delta
            cnt += 1

    partition_offset = partition_cnt

    ti.loop_config(serialize=True)
    for j in range(num_particles_y - 1):
        num_edges_per_partition[j + partition_offset] = num_particles_x - 1
        partition_cnt += 1
        for i in range(num_particles_x - 1):
            partitioned_set[j + partition_offset, i] = cnt

            if i % 2 == 0:
                indices[2 * cnt + 0] = i + num_particles_x * j
                indices[2 * cnt + 1] = i + 1 + num_particles_x * (j + 1)
            else:
                indices[2 * cnt + 0] = i + num_particles_x * (j + 1)
                indices[2 * cnt + 1] = i + 1 + num_particles_x * j

            l0[cnt] = ti.sqrt(2.0) * delta
            cnt += 1


    # partition_offset = partition_cnt
    #
    # ti.loop_config(serialize=True)
    # for j in range(0, num_particles_y - 1):
    #     num_edges_per_partition[j + partition_offset] = num_particles_x - 1
    #     partition_cnt += 1
    #     for i in range(num_particles_x - 1):
    #         partitioned_set[j + partition_offset, i] = cnt
    #
    #         if i % 2 == 1:
    #             indices[2 * cnt + 0] = i + num_particles_x * j
    #             indices[2 * cnt + 1] = i + 1 + num_particles_x * (j + 1)
    #         else:
    #             indices[2 * cnt + 0] = i + num_particles_x * (j + 1)
    #             indices[2 * cnt + 1] = i + 1 + num_particles_x * j
    #
    #         l0[cnt] = ti.sqrt(2.0) * delta
    #         cnt += 1

    num_indices[0] = cnt
    num_partition[0] = partition_cnt
    # print("----")

    ti.loop_config(serialize=True)
    for pi in range(num_partition[0]):
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
        if material_type[i] == 0:
            # positions[i] += offs
            mass[i] = 1.0
            # rho0[i] = mass[i] * poly6_value(0.0, h_)

        elif material_type[i] == 1:
            # positions[i] -= offs
            mass[i] = mass_ratio
            # rho0[i] = mass[i] * poly6_value(0.0, h_)

        elif material_type[i] == 2:
            mass[i] = 10 * mass_ratio
            # rho0[i] = mass[i] * poly6_value(0.0, h_)
    # mass[0] = 0.0
    velocities.fill(0.0)
    board_states[None] = ti.Vector([boundary[0] - epsilon, -0.0])


@ti.kernel
def switch_material():
    # print(boundary)
    for i in range(num_particles):
        if material_type[i] == 0:
            material_type[i] = 2
        elif material_type[i] == 2:
            material_type[i] = 0

    # for i in range(num_particles):
    #     if material_type[i] == 0:
    #         mass[i] = 1.0
    #         rho0[i] = mass[i] * poly6_value(0.0, h_)
    #
    #     elif material_type[i] == 1:
    #         mass[i] = 10.0
    #         rho0[i] = mass[i] * poly6_value(0.0, h_)

        # elif material_type[i] == 2:
        #     mass[i] = 3.0
        #     rho0[i] = mass[i] * poly6_value(0.0, h_)


# def print_stats():
#     print("PBF stats:")
#     num = grid_num_particles.to_numpy()
#     avg, max_ = np.mean(num), np.max(num)
#     print(f"  #particles per cell: avg={avg:.2f} max={max_}")
#     num = particle_num_neighbors.to_numpy()
#     avg, max_ = np.mean(num), np.max(num)
#     print(f"  #neighbors per particle: avg={avg:.2f} max={max_}")

frame_cnt = 0

def main():

    global frame_cnt
    run_sim = False
    reset(mass_ratio)

    print(f"boundary={boundary} grid={grid_size} cell_size={cell_size}")

    window = ti.ui.Window(name='PBF2D', res = screen_res, fps_limit=200, pos = (150, 150))
    gui = window.get_gui()
    canvas = window.get_canvas()
    canvas.set_background_color((0.2, 0.2, 0.2))

    def show_options():

        global pbf_num_iters
        global threshold
        global solver_type
        global PR
        global frame_cnt
        global print_stat
        global enable_pncg

        PR_old = PR
        threshold_old = threshold

        with gui.sub_window("XPBD Settings", 0., 0., 0.3, 0.7) as w:


            solver_type = w.slider_int("solver type", solver_type, 0, 2)
            if solver_type == 0:
                w.text("Euler")
            elif solver_type == 1:
                w.text("Jacobi")
            elif solver_type == 2:
                w.text("Newton")

            pbf_num_iters = w.slider_int("max iter.", pbf_num_iters, 1, 10000)
            threshold = w.slider_int("threshold", threshold, 1, 6)
            PR_old = w.slider_int("stiffness", PR_old, 1, 8)
            enable_pncg = w.checkbox("enable PNCG", enable_pncg)
            print_stat = w.checkbox("print_stats", print_stat)


            if not PR_old == PR:
                PR = PR_old

            w.text("")
            frame_str = "# frame " + str(frame_cnt)
            w.text(frame_str)

    conv_it_total = 0
    elapsed_time_total = 0
    global solver_type
    global print_stat
    while window.running:

        if window.get_event(ti.ui.PRESS):
            if window.event.key == ' ':
                run_sim = not run_sim

            if window.event.key == 'b':

                positions.copy_from(old_positions)
                velocities.copy_from(old_velocities)

                frame_cnt -= 1

            if window.event.key == 'r':
                reset(mass_ratio)

                if frame_cnt > 0:
                    if solver_type == 0:
                      print("Euler")
                    elif solver_type == 1:
                        print("Jacobi")
                    elif solver_type == 2:
                        print("Newton")
                    print("Avg. iter    : ", int(conv_it_total / frame_cnt))
                    print("Avg. time[ms]: ", round(100.0 * (elapsed_time_total / frame_cnt), 2))

                frame_cnt = 0
                elapsed_time_total = 0
                conv_it_total = 0
                run_sim = False

            if window.event.key == 's':
                switch_material()

            if window.event.key == 'o' and run_sim is False:
                start = timer()
                conv_it_total += run()
                frame_cnt += 1
                end = timer()
                elapsed_time_total += (end - start)

        if run_sim:
            start = timer()
            conv_it_total += run()
            frame_cnt += 1
            end = timer()

            elapsed_time = (end - start)
            elapsed_time_total += elapsed_time

        show_options()
        render_kernel(positions, positions_window, screen_to_world_ratio, screen_res[0], screen_res[1])

        render_kernel(floor_positions, floor_pos_window, screen_to_world_ratio, screen_res[0], screen_res[1])
        radius = 0.2 * (screen_to_world_ratio / screen_res[0]) * h_

        canvas.lines(positions_window, indices=indices, width=0.001, color=(1.0, 1.0, 1.0))
        canvas.circles(positions_window, radius=radius, per_vertex_color=colors)

        canvas.lines(floor_pos_window, indices=floor_indices, width=0.001, per_vertex_color=floor_position_colors)

        window.show()

if __name__ == "__main__":
    main()