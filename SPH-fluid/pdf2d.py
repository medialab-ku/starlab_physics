# Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.
# Taichi implementation by Ye Kuang (k-ye)

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize

import taichi as ti



ti.init(arch=ti.gpu, default_fp=ti.f32)

screen_res = (1000, 1000)
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
particle_color = 0x068587
boundary_color = 0xEBACA2
num_particles_x = 1
num_fluids = num_particles_x * 1

num_solid = 5
num_particles = num_fluids + num_solid
num_indices = num_solid - 1

max_num_particles_per_cell = 100
max_num_neighbors = 100
dt = 0.01
epsilon = 1e-5
particle_radius = 3.0
particle_radius_in_world = particle_radius / screen_to_world_ratio

# PBF params
h_ = 1.1
rho0 = 1.0
lambda_epsilon = 100.0
pbf_num_iters = 50
corr_deltaQ_coeff = 0.3
corrK = 0.001
# Need ti.pow()
# corrN = 4.0
neighbor_radius = h_ * 1.05

poly6_factor = 315.0 / 64.0 / math.pi
spiky_grad_factor = -45.0 / math.pi

x_old = ti.Vector.field(dim, float)
mass = ti.field(float)
x  = ti.Vector.field(dim, float)
x_tmp  = ti.Vector.field(dim, float)
x0 = ti.Vector.field(dim, float)
xWorld = ti.Vector.field(dim, float)
xHat = ti.Vector.field(dim, float)
v = ti.Vector.field(dim, float)
vHat = ti.Vector.field(dim, float)
grid_num_particles = ti.field(int)
grid2particles = ti.field(int)
particle_num_neighbors = ti.field(int)
particle_neighbors = ti.field(int)

rho = ti.field(float)
hii  = ti.Matrix.field(dim, dim, float)
h_e  = ti.Matrix.field(dim, dim, float)
h_fij = ti.Matrix.field(dim, dim, float)
grad_fij = ti.Vector.field(dim, float)
dx   = ti.Vector.field(dim, float)
grad = ti.Vector.field(dim, float)
indices = ti.field(int)
l0 = ti.field(float)



colors = ti.Vector.field(3, float)
# 0: x-pos, 1: timestep in sin()
board_states = ti.Vector.field(2, float)

ti.root.dense(ti.i, num_particles).place(x_old, x, x_tmp, x0, xHat, v, vHat, xWorld, colors, mass)
ti.root.dense(ti.i, 2 * num_indices).place(indices)
ti.root.dense(ti.i, num_indices).place(l0, h_e)
grid_snode = ti.root.dense(ti.ij, grid_size)
grid_snode.place(grid_num_particles)
grid_snode.dense(ti.k, max_num_particles_per_cell).place(grid2particles)
nb_node = ti.root.dense(ti.i, num_particles)
nb_node.place(particle_num_neighbors)
nb_node.dense(ti.j, max_num_neighbors).place(particle_neighbors, h_fij, grad_fij)
ti.root.dense(ti.i, num_particles).place(rho, dx, hii, grad)
ti.root.place(board_states)

pair = ti.Vector.field(dim, int)
h_c  = ti.Matrix.field(dim, dim, float)
n_c  = ti.field(int)
a_c  = ti.field(float)
num_max_pairs = int(1e3)
ti.root.dense(ti.i, num_max_pairs).place(pair, h_c, a_c)
ti.root.dense(ti.i, 1).place(n_c)

Ax = ti.Vector.field(dim, float)
Ap = ti.Vector.field(dim, float)
s  = ti.Vector.field(dim, float)
p  = ti.Vector.field(dim, float)
z  = ti.Vector.field(dim, float)
r  = ti.Vector.field(dim, float)
b  = ti.Vector.field(dim, float)
ti.root.dense(ti.i, num_particles).place(Ax, Ap, s, p, z, r, b)

l_min = 1.0

@ti.func
def barrier(d: float, dHat: float):
    k = (-ti.log(d / dHat))
    return k * (d - dHat) ** 2


@ti.func
def barrier_grad(d: float, dHat: float):
    k = (-ti.log(d / dHat))
    dkdx = -1.0 / d
    return  (2 * k  + dkdx * (d - dHat)) * (d - dHat)

@ti.func
def barrier_grad_sl(d, dHat, sl):
    ret = 0.0
    if d > dHat:
        a = (sl - dHat)
        b = dHat - d
        ret = -(2.0 * (b / (a ** 2)) * ti.log(1.0 - (b / a)) - ((b / a) ** 2) * (1.0 / (a - b)))

    return ret

@ti.func
def barrier_hess_sl(d, dHat, sl):
    ret = 0.0
    if d >= dHat:
        a = (sl - dHat)
        b = dHat - d
        ret = -(2.0 * (1 / (a ** 2)) * ti.log(1.0 - (b / a)) - 4.0 * b / (a ** 2) * (a - b) - (b ** 2) / (a * (a - b)) ** 2)

    return ret



@ti.func
def barrier_hess(d: float, dHat: float):
    k = (-ti.log(d / dHat))
    dkdx = -1.0 / d
    d2kdx2 = 1.0 / (d ** 2)
    return  (d2kdx2 * (d - dHat) + 4 * dkdx) * (d - dHat) + 2 * k

@ti.func
def poly6_value(s, h):
    result = 0.0
    if 0 <= s and s < h:
        x = (h * h - s * s) / (h * h * h)
        result = poly6_factor * x * x * x
    return result


@ti.func
def spiky_value(r, h):
    result = 0.0
    if 0.0 < r and r < h:
        x = (h - r) / (h * h)
        result = (spiky_grad_factor / 3) * x * x * x
    return result


@ti.func
def spiky_gradient(r, h):
    result = 0.0
    # r_len = r.norm()
    if 0.0 <= r and r < h:
        x = (h - r) / (h * h * h)
        result = spiky_grad_factor * x * x
    return result

@ti.func
def spiky_hessian(r, h):
    result = 0.0
    # r_len = r.norm()
    if 0.0 <= r and r < h:
        x = (h - r) / (h ** 6)
        result = -2.0 * spiky_grad_factor * x
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
def get_cell(pos):
    return int(pos * cell_recpr)


@ti.func
def is_in_grid(c):
    # @c: Vector(i32)
    return 0 <= c[0] and c[0] < grid_size[0] and 0 <= c[1] and c[1] < grid_size[1]


@ti.func
def confine_position_to_boundary(p):
    bmin = particle_radius_in_world
    bmax = ti.Vector([board_states[None][0], boundary[1]]) - particle_radius_in_world
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
    vel_strength = 4.0
    if b[1] >= 2 * period:
        b[1] = 0
    b[0] += -ti.sin(b[1] * np.pi / period) * vel_strength * dt
    board_states[None] = b


@ti.kernel
def computeXTilta():
    # save old positions
    for i in range(num_particles):
        x_old[i] = x[i]
    # apply gravity within boundary
    for i in range(num_particles):
        xHat[i] = x[i] + v[i] * dt
        # xHat[i] = confine_position_to_boundary(xHat[i])

@ti.kernel
def computeVTilta():
    # save old positions
    # for i in range(num_particles):
    #     x_old[i] = x[i]
    # apply gravity within boundary
    for i in range(num_particles):
        g = ti.Vector([0.0, -10.0])
        vHat[i] = v[i] + g * dt
        # xHat[i] = confine_position_to_boundary(xHat[i])

@ti.kernel
def neighbour_search(x: ti.template()):
    # clear neighbor lookup table
    for I in ti.grouped(grid_num_particles):
        grid_num_particles[I] = 0
    for I in ti.grouped(particle_neighbors):
        particle_neighbors[I] = -1

    # update grid
    for p_i in range(num_fluids):
        cell = get_cell(x[p_i])
        # ti.Vector doesn't seem to support unpacking yet
        # but we can directly use int Vectors as indices
        offs = ti.atomic_add(grid_num_particles[cell], 1)
        grid2particles[cell, offs] = p_i
    # find particle neighbors
    for p_i in range(num_fluids):
        pos_i = x[p_i]
        cell = get_cell(pos_i)
        nb_i = 0
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
            cell_to_check = cell + offs
            if is_in_grid(cell_to_check):
                for j in range(grid_num_particles[cell_to_check]):
                    p_j = grid2particles[cell_to_check, j]
                    if nb_i < max_num_neighbors and p_j != p_i and (pos_i - x[p_j]).norm() < neighbor_radius:
                        particle_neighbors[p_i, nb_i] = p_j
                        nb_i += 1
        particle_num_neighbors[p_i] = nb_i

@ti.kernel
def collision_pairs(x: ti.template(), dHat: float):
    n_c[0] = 0
    for vi in range(num_fluids):
        # vi = (num_particles - num_solid) + si
        for ei in range(num_indices):
            v0, v1 = indices[2 * ei + 0], indices[2 * ei + 1]
            if vi != v0 and vi != v1:
                x10 = x[v1] - x[v0]
                xi0 = x[vi] - x[v0]
                alpha = x10.dot(xi0) / x10.dot(x10)
                # print(alpha)
                alpha = ti.math.clamp(alpha, 0.0, 1.0)

                # if alpha > 0.0 and alpha < 1.0:
                p = alpha * x[v1] + (1.0 - alpha) * x[v0]
                d = (x[vi] - p).norm()
                # print(d)
                if d <= dHat:
                    nc = ti.atomic_add(n_c[0], 1)
                    pair[nc] = ti.math.ivec2([vi, ei])

    # return numCollision


@ti.kernel
def solveLinearSystem2():
    for i in x:
        dx[i] = -(hii[i].inverse()) @ grad[i]

@ti.kernel
def solveInertia(q: ti.template(), qHat: ti.template()):
    id2 = ti.math.mat2([[1.0, 0.0], [0.0, 1.0]])
    for i in x:
        grad[i] = mass[i] * (q[i] - qHat[i])
        hii[i] = mass[i] * id2

@ti.kernel
def computeInertiaPotential(a: ti.template()) -> float:

    value = 0.0
    for i in a:
        value += 0.5 * mass[i] * (a[i] - xHat[i]).dot(a[i] - xHat[i])

    return value

@ti.kernel
def solvePressure(k: float, dHat: float, fix: bool):

    # print(poly6_value(0.0, dHat))


    id2 = ti.math.mat2([[1.0, 0.0], [0.0, 1.0]])
    coeff = k * (dt ** 2)
    for p_i in range(num_fluids):

        rho0 = mass[p_i] * poly6_value(0.0, dHat)
        pos_i = x[p_i]
        dEdx_i = ti.Vector([0.0, 0.0])
        d2Edx2_i = ti.math.mat2(0.0)
        #
        c = mass[p_i] * poly6_value(0.0, dHat)
        # for p_j in range(p_i + 1, num_fluids):
        # rho[p_i] = 0.0
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break

            pos_ji = pos_i - x[p_j]
            r = pos_ji.norm()
            c += mass[p_j] * poly6_value(r, dHat)

        # Eq(1)
        # c = rho - 1.0

        # if c > 1.01 * rho0:
        #     print("fuck")

        # dEdc = abs(barrier_grad_sl(c, rho0, 1.01 * rho0))
        # d2Edc2 = abs(barrier_hess_sl(c,  rho0, 1.01 *  rho0))
        # if c > 1.0:


        dEdc = 3 * (c - rho0) ** 2
        d2Edc2 = 6 *  (c - rho0)

        dEdc = max(c - rho0, 0)
        d2Edc2 = 1.0

        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break

            pos_ji = pos_i - x[p_j]
            r = pos_ji.norm()
            n = pos_ji / r
            dWdr = spiky_gradient(r, dHat)
            d2Wdr2 = spiky_hessian(r, dHat)
            dcdx_j = dWdr * n
            dEdx_j = coeff * dEdc * dcdx_j
            dEdx_i += dEdx_j

            nnT = n.outer_product(n)
            alpha = abs(dWdr / r)
            beta = abs(d2Wdr2 - dWdr / r)
            d2cdx2_j = alpha * id2 + beta * nnT
            d2Edx2_j = coeff * mass[p_j] * (d2Edc2 * dcdx_j.outer_product(dcdx_j) + abs(dEdc) * d2cdx2_j)
            d2Edx2_i += d2Edx2_j

            grad[p_j] -= dEdx_j
            hii[p_j] += d2Edx2_j
            h_fij[p_i, j] = coeff * mass[p_j]  * abs(dEdc) * d2cdx2_j
            grad_fij[p_i, j] = ti.sqrt(coeff * (d2Edc2 * mass[p_j])) * dcdx_j
        #
        grad[p_i] += dEdx_i
        hii[p_i] += d2Edx2_i

@ti.kernel
def solvePressure_UL(k: float, dHat: float, fix: bool):

    coeff = k * (dt ** 2)
    for p_i in range(num_fluids):
        pos_i = x_old[p_i]
        dEdx_i = ti.Vector([0.0, 0.0])
        d2Edx2_i = ti.math.mat2(0.0)

        c = rho[p_i]
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break

            xji_old = pos_i - x_old[p_j]
            xji = x[p_i] - x[p_j]
            r = xji_old.norm()
            n = xji_old / r
            dWdr = spiky_gradient(r, dHat)
            c += dWdr * n.dot(xji - xji_old)

        # Eq(1)
        # if c > 1.0:
        dEdc = max(c - 1.0, 0)
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break

            xji_old = pos_i - x_old[p_j]
            r = xji_old.norm()
            n = xji_old / r
            dWdr = spiky_gradient(r, dHat)

            dcdx_j = dWdr * n
            dEdx_j = coeff * dEdc * dcdx_j
            dEdx_i += dEdx_j

            d2Edx2_j = coeff * (dcdx_j.outer_product(dcdx_j))
            d2Edx2_i += d2Edx2_j

            grad[p_j] -= dEdx_j
            hii[p_j] += d2Edx2_j
            h_fij[p_i, j] = ti.math.mat2(0.0)
            grad_fij[p_i, j] = ti.sqrt(coeff) * dcdx_j
        #
        grad[p_i] += dEdx_i
        hii[p_i] += d2Edx2_i

@ti.kernel
def solveDivergence(k: float, dHat: float):

    coeff = k * (dt ** 2)
    for p_i in range(num_fluids):
        dEdx_i = ti.Vector([0.0, 0.0])
        d2Edx2_i = ti.math.mat2(0.0)

        div = 0.0
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break

            xji = x[p_j] - x[p_i]
            vji = v[p_j] - v[p_i]

            r = xji.norm()
            n = xji / r
            dWdr = spiky_gradient(r, dHat)
            div += dWdr * n.dot(vji)

        # Eq(1)
        # if c > 1.0:

        # div = max(div, 0.0)

        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break

            xji = x[p_j] - x[p_i]
            r = xji.norm()
            n = xji / r
            dWdr = spiky_gradient(r, dHat)

            dcdx_j = ti.math.vec2(0.0)

            if div > 0.0:
                dcdx_j = dWdr * n


            dEdx_j = coeff * div * dcdx_j
            dEdx_i += dEdx_j

            d2Edx2_j = coeff * (dcdx_j.outer_product(dcdx_j))
            d2Edx2_i += d2Edx2_j

            grad[p_j] += dEdx_j
            hii[p_j] -= d2Edx2_j
            h_fij[p_i, j] = ti.math.mat2(0.0)
            grad_fij[p_i, j] = ti.sqrt(coeff) * dcdx_j
        #
        grad[p_i] -= dEdx_i
        hii[p_i] -= d2Edx2_i

@ti.kernel
def solvePressure_UL_J(k: float, dHat: float, fix: bool):


    # rho.fill(1.0)
    coeff = k * (dt ** 2)
    for p_i in range(num_fluids):
        pos_i = x_old[p_i]
        dEdx_i = ti.Vector([0.0, 0.0])
        d2Edx2_i = ti.math.mat2(0.0)

        J = 1.0 / rho[p_i]
        ul = 0.0
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break

            xji_old = x_old[p_j] - pos_i
            xji =  x[p_j] - x[p_i]
            r = xji_old.norm()
            n = xji_old / r
            dWdr = spiky_gradient(r, dHat)
            ul += (dWdr / rho[p_j]) * n.dot(xji - xji_old)

        # Eq(1)
        # if c > 1.0:
        c = J * (ul + 1.0)
        dEdc = max(c - 1.0, 0)
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break

            xji_old = x_old[p_j] - pos_i
            r = xji_old.norm()
            n = xji_old / r
            dWdr = spiky_gradient(r, dHat)

            dcdx_j = dWdr * n
            dEdx_j = (J / rho[p_j]) * coeff * dEdc * dcdx_j
            dEdx_i += dEdx_j

            d2Edx2_j = (J / rho[p_j]) * coeff * (dcdx_j.outer_product(dcdx_j))
            d2Edx2_i += d2Edx2_j

            grad[p_j] += dEdx_j
            hii[p_j] += d2Edx2_j
            h_fij[p_i, j] = ti.math.mat2(0.0)
            grad_fij[p_i, j] = ti.sqrt((J / rho[p_j]) * coeff) * dcdx_j
        #
        grad[p_i] -= dEdx_i
        hii[p_i] += d2Edx2_i

@ti.kernel
def computePressurePotential(x: ti.template(), k: float, dHat: float, fix: bool) -> float:

    ret = 0.0
    coeff = k * (dt ** 2)
    for p_i in range(num_fluids - 1):
        pos_i = x[p_i]
        c = poly6_value(0.0, dHat)
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break

            pos_ji = pos_i - x[p_j]
            r = pos_ji.norm()
            c += poly6_value(r, dHat)

        ret += 0.5 * coeff * (c - 1.0) ** 2

    return ret

@ti.kernel
def computeDensity(x: ti.template()):

    for p_i in range(num_fluids):
        pos_i = x[p_i]
        rho[p_i] = poly6_value(0, h_)
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            pos_ji = pos_i - x[p_j]
            r = pos_ji.norm()
            rho[p_i] += poly6_value(r, h_)

        # rho[p_i] += poly6_value(0.0, h_)

@ti.kernel
def computeDistance():

    for i in range(n_c[0]):
        vi, ei = pair[i]
        v0, v1 = indices[2 * ei + 0], indices[2 * ei + 1]
        x10 = x[v1] - x[v0]
        xi0 = x[vi] - x[v0]
        alpha = x10.dot(xi0) / x10.dot(x10)
        # print(alpha)
        if alpha > 0.0 and alpha < 1.0:
            p = alpha * x[v1] + (1.0 - alpha) * x[v0]
            d = (x[vi] - p).norm()

            if d < 1e-2:
                print("test")

    # for i in range(num_particles - num_fluids):
    #     rho[i + num_fluids] = 1e3



@ti.kernel
def solveCollision(dHat: float, k: float):
    id2 = ti.math.mat2([[1.0, 0.0], [0.0, 1.0]])
    # numCollision = 0
    # n_c[0] = 0
    for i in range(n_c[0]):
        vi, ei = pair[i]
        v0, v1 = indices[2 * ei + 0], indices[2 * ei + 1]
        x10 = x[v1] - x[v0]
        xi0 = x[vi] - x[v0]
        alpha = x10.dot(xi0) / x10.dot(x10)
        alpha = ti.math.clamp(alpha, 0.0, 1.0)

        # if alpha > 0.0 and alpha < 1.0:
        p = alpha * x[v1] + (1.0 - alpha) * x[v0]
        d = (x[vi] - p).norm()
        # print(d)
        if d <= dHat:
            n = (x[vi] - p) / d

            dbdx = barrier_grad(d, dHat)
            d2bdx2 = barrier_hess(d, dHat)

            # dbdx = spiky_gradient(d, dHat)
            # d2bdx2 = spiky_hessian(d, dHat)

            # dbdx = d - dHat
            # d2bdx2 = barrier_hess(d, dHat)

            grad[vi] += k * dbdx * n
            grad[v0] -= k * alpha * dbdx * n
            grad[v1] -= k * (1.0 - alpha) * dbdx * n

            nnT = n.outer_product(n)
            d2d_dx2 = (id2 - nnT) / d
            test = k * (abs(d2bdx2) * nnT)

            test = k * ((abs(d2bdx2) - abs(dbdx / d)) * nnT + abs(dbdx / d) * id2)
            hii[vi] += test
            hii[v0] += alpha * test
            hii[v1] += (1.0 - alpha) * test

            a_c[i] = alpha
            h_c[i] = test
        else:
            h_c[i] = ti.math.mat2(0.0)

@ti.kernel
def solveNonPenetration(dHat: float, k: float):

    for i in range(n_c[0]):
        vi, ei = pair[i]
        v0, v1 = indices[2 * ei + 0], indices[2 * ei + 1]
        x10 = x[v1] - x[v0]
        xi0 = x[vi] - x[v0]
        alpha = x10.dot(xi0) / x10.dot(x10)
        alpha = ti.math.clamp(alpha, 0.0, 1.0)

        # if alpha > 0.0 and alpha < 1.0:
        p = alpha * x[v1] + (1.0 - alpha) * x[v0]
        d = (x[vi] - p).norm()
        # print(d)
        if d <= dHat:
            n = (x[vi] - p) / d
            vp = alpha * v[v1] + (1.0 - alpha) * v[v0]
            dv = v[vi] - vp

            if n.dot(dv) <= 0.0:
                grad[vi] += k * n.dot(dv) * n
                grad[v0] -= k * alpha * n.dot(dv) * n
                grad[v1] -= k * (1.0 - alpha) * n.dot(dv) * n

                nnT = n.outer_product(n)
                test = k * nnT
                # test = k * (abs(dbdx / d) * id2 + abs(d2bdx2 - dbdx / d) * nnT)
                hii[vi] += test
                hii[v0] += alpha * test
                hii[v1] += (1.0 - alpha) * test

                a_c[i] = alpha
                h_c[i] = test

            else:
                h_c[i] = ti.math.mat2(0.0)

        else:
            h_c[i] = ti.math.mat2(0.0)


@ti.kernel
def computeCollisionPotential(x: ti.template(), dHat: float, k: float) -> float:

    value = 0.0
    for i in range(n_c[0]):
        vi, ei = pair[i]
        # for vi in range(num_fluids):
        #     for ei in range(num_indices):

        v0, v1 = indices[2 * ei + 0], indices[2 * ei + 1]
        x10 = x[v1] - x[v0]
        xi0 = x[vi] - x[v0]
        alpha = x10.dot(xi0) / x10.dot(x10)
        # print(alpha)

        alpha = ti.math.clamp(alpha, 0.0, 1.0)
        # if alpha > 0.0 and alpha < 1.0:
        p = alpha * x[v1] + (1.0 - alpha) * x[v0]
        d = (x[vi] - p).norm()
        if d <= dHat:
            ti.atomic_add(value, k * barrier(d, dHat))

    return value

# @ti.kernel
# def filterStepSize(x: ti.template(), dx: ti.template(), dHat: float) -> float:




@ti.kernel
def solveElasticity(k: float):

    # compute lambdas
    # Eq (8) ~ (11)
    offset = num_particles - num_solid
    id2 = ti.math.mat2([[1.0, 0.0], [0.0, 1.0]])

    coeff = k * (dt ** 2)
    for i in range(num_indices):
        v0, v1 = indices[2 * i + 0], indices[2 * i + 1]

        l = (x[v0] - x[v1]).norm()
        n = (x[v0] - x[v1]) / l
        grad[v0] += coeff * (l - l0[i]) * n
        grad[v1] -= coeff * (l - l0[i]) * n
        nnT = n.outer_product(n)
        # dndx = (id2 - nnT) / l
        alpha = abs(l - l0[i]) / l
        value = coeff * (alpha * id2 + (1.0 - alpha) * nnT)
        # value = coeff * (nnT)

        hii[v0] += value
        hii[v1] += value
        h_e[i] = value


    fixed_ids = ti.Vector([offset, num_solid - 1 + offset], dt=int)
    k_fix = 1e7
    for i in range(2):
        vi = fixed_ids[i]
        grad[vi] += k_fix * (x[vi] - x0[vi])
        hii[vi] += k_fix * id2

@ti.kernel
def computeElasticPotential(a: ti.template(), k: float) -> float:

    value = 0.0
    coeff = k * (dt ** 2)
    for i in range(num_indices):
        v0, v1 = indices[2 * i + 0], indices[2 * i + 1]

        l = (a[v0] - a[v1]).norm()
        value += 0.5 * coeff * (l - l0[i]) ** 2

    offset = num_particles - num_solid
    fixed_ids = ti.Vector([offset, num_solid - 1 + offset], dt=int)
    k_fix = 1e7
    for i in range(2):
        vi = fixed_ids[i]
        value += 0.5 * k_fix * (a[vi] - x0[vi]).dot(a[vi] - x0[vi])

    return value

@ti.kernel
def dot(a: ti.template(), b: ti.template()) -> float:

    ret = 0.0
    for i in range(num_particles):
        ret += ti.math.dot(a[i], b[i])

    return ret


@ti.kernel
def add(ret: ti.template(), v0: ti.template(), v1: ti.template()):
    for i in x:
        ret[i] = v0[i] + v1[i]

@ti.kernel
def add_pcg(ret: ti.template(), v0: ti.template(), scale: float, v1: ti.template()):
    for i in x:
        ret[i] = v0[i] + scale * v1[i]

@ti.kernel
def scale(alpha: float, b: ti.template()):
    for i in x:
        b[i] = alpha * b[i]

@ti.kernel
def inf_norm(a: ti.template()) -> float:

    ret = 0.0
    # ti.loop_config(serialize=True)
    for i in x:

        tmp = 0.0
        for j in range(2):
            if tmp < abs(a[i][j]):
                tmp = abs(a[i][j])

        ti.atomic_max(ret, tmp)

    return ret

@ti.kernel
def computeVelocity():
    # confine to boundary
    for i in x:
        pos = x[i]
        x[i] = confine_position_to_boundary(pos)
    # update velocities
    for i in x:
        v[i] = (x[i] - x_old[i]) / dt
    # no vorticity/xsph because we cannot do cross product in 2D...



@ti.kernel
def shrink():

    for i in range(num_indices):
        l0[i] *= 0.99


@ti.kernel
def matFreeAx(Ax: ti.template(), a: ti.template()):

    for i in range(num_particles):
        Ax[i] = mass[i] * a[i]

    for i in range(num_indices):
        v0, v1 = indices[2 * i + 0], indices[2 * i + 1]
        x01 = a[v0] - a[v1]
        Ax[v0] += h_e[i] @ x01
        Ax[v1] -= h_e[i] @ x01
    #
    offset = num_particles - num_solid
    fixed_ids = ti.Vector([offset, num_solid - 1 + offset], dt=int)
    k_fix = 1e7
    for i in range(2):
        vi = fixed_ids[i]
        Ax[vi] += k_fix * a[vi]
    #
    #
    for i in range(n_c[0]):
        vi, ei = pair[i]
        v0, v1 = indices[2 * ei + 0], indices[2 * ei + 1]

        alpha = a_c[i]
        p = alpha * a[v0] + (1.0 - alpha) * a[v1]
        xip = a[vi] - p
        Ax[vi] += h_c[i] @ xip
        Ax[v0] -= alpha * h_c[i] @ xip
        Ax[v1] -= (1.0 - alpha) * h_c[i] @ xip
    # # #
    # # #
    # for p_i in range(num_fluids):
    #     ai = a[p_i]
    #     alpha = 0.0
    #     grad_i = ti.math.vec2(0.0)
    #     for j in range(particle_num_neighbors[p_i]):
    #         p_j = particle_neighbors[p_i, j]
    #         if p_j < 0:
    #             break
    #         aij = ai - a[p_j]
    #         Ax[p_i] += h_fij[p_i, j] @ aij
    #         Ax[p_j] -= h_fij[p_i, j] @ aij
    #         alpha += grad_fij[p_i, j].dot(aij)
    #         grad_i += grad_fij[p_i, j]
    #     #
    #     #
    #     for j in range(particle_num_neighbors[p_i]):
    #         p_j = particle_neighbors[p_i, j]
    #         if p_j < 0:
    #             break
    #         Ax[p_j] -= alpha * grad_fij[p_i, j]
    #
    #     Ax[p_i] += alpha * grad_i

@ti.kernel
def matFreeAx2(Ax: ti.template(), a: ti.template()):

    for i in range(num_particles):
        Ax[i] = mass[i] * a[i]

    for p_i in range(num_fluids):
        ai = a[p_i]
        alpha = 0.0
        grad_i = ti.math.vec2(0.0)
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            aij = a[p_j] - ai
            alpha += grad_fij[p_i, j].dot(aij)
            grad_i += grad_fij[p_i, j]
        #
        #
        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            if p_j < 0:
                break
            Ax[p_j] += alpha * grad_fij[p_i, j]

        Ax[p_i] -= alpha * grad_i

    # for i in range(n_c[0]):
    #     vi, ei = pair[i]
    #     v0, v1 = indices[2 * ei + 0], indices[2 * ei + 1]
    #
    #     alpha = a_c[i]
    #     p = alpha * a[v0] + (1.0 - alpha) * a[v1]
    #     xip = a[vi] - p
    #     Ax[vi] += h_c[i] @ xip
    #     Ax[v0] -= alpha * h_c[i] @ xip
    #     Ax[v1] -= (1.0 - alpha) * h_c[i] @ xip

@ti.kernel
def applyPrecondition(z: ti.template(), r: ti.template()):
    for i in z:
        z[i] = hii[i].inverse() @ r[i]


def solvePCG(x: ti.template(), b: ti.template(), tol, matFreeAx):

    #x = vec(0)
    s.fill(0.0)

    #b = -grad
    # b.copy_from(grad)
    # scale(-1.0, b)

    # r = b - A @ x
    r.copy_from(b)

    #z = diag(A)^-1 * r
    applyPrecondition(z, r)

    # # print(z)
    # # z = M_inv * r
    p.copy_from(z)
    rs_old = dot(r, z)
    itrCnt = 0

    #
    if rs_old < tol:
        x.fill(0.0)
        return itrCnt

    maxPCGIter = int(1e4)
    for i in range(maxPCGIter):

        matFreeAx(Ap, p)
        pAp = dot(p, Ap)
        alpha = rs_old / pAp
        add_pcg(s, s, alpha, p)
        add_pcg(r, r, -alpha, Ap)

        applyPrecondition(z, r)
        rs_new = dot(r, z)

        itrCnt += 1
        if rs_new < tol:
            break
        beta = rs_new / rs_old
        add_pcg(p, z, beta, p)
        rs_old = rs_new


    if itrCnt == maxPCGIter:
        print("failed to converge...")

    # print("PCG iter: ", itrCnt)
    x.copy_from(s)

    return itrCnt

def run_pbf(show_plot):

    k_col = 1e3
    k_el = 1e5
    k_p = 1e5
    optIter = 0
    maxIter = int(1e2)
    dHat = 1.0

    computeVTilta()
    v.copy_from(vHat)
    computeXTilta()
    logg = []
    # print(dHat)
    pcgIter = 0


    # collision_pairs(x, dHat)
    for _ in range(maxIter):
        E = 0
        E += computeInertiaPotential(x)
        E += computeElasticPotential(x, k_el)
        collision_pairs(x, dHat)
        E += computeCollisionPotential(x, dHat, k_col)

        # neighbour_search(x)
        solveInertia(x, xHat)
        # solvePressure(k_p, dHat, False)
        # # solvePressure_UL(k_p, h_, False)
        solveElasticity(k_el)
        solveCollision(dHat, k_col)

        b.copy_from(grad)
        scale(-1.0, b)

        pcgIter += solvePCG(dx, b, 1e-4, matFreeAx)
        optIter += 1
        dx_inf_new = inf_norm(dx)

        logg.append(E)
        # print(alpha)

        # matFreeAx(Ax, dx)
        # pAp = dot(Ax, dx)
        # alpha = min(0.5 * dHat / inf_norm(dx), -dot(dx, grad) / pAp)
        # alpha = min(0.5 * dHat / inf_norm(dx), 1.0)
        alpha = 1.0
        #
        lsIter = 0
        for _ in range(100):

            add_pcg(x_tmp, x, alpha, dx)
            E_tmp = 0
            E_tmp += computeInertiaPotential(x_tmp)
            E_tmp += computeElasticPotential(x_tmp, k_el)
            collision_pairs(x_tmp, dHat)
            E_tmp += computeCollisionPotential(x_tmp, dHat, k_col)
            if E_tmp <= E:
                break
        # # #
            alpha *= 0.5
            lsIter += 1
        # # # # # # # # # # #
        # if lsIter > 0:
        #     print("LS Iter: ", lsIter)
        #     print("alpha: ", alpha)
        add_pcg(x, x, alpha, dx)

        # x_tmp.copy_from(x)
        # E_tmp = 0
        # E_tmp += computeInertiaPotential(x_tmp)
        # E_tmp += computeElasticPotential(x_tmp, k_el)
        # collision_pairs(x_tmp, dHat)
        # E_tmp += computeCollisionPotential(x_tmp, dHat, k_col)

        # if E_tmp > E:
        #     print("warning")

        if dx_inf_new < 3e-2:
            break

        dx_inf_old = dx_inf_new

    if optIter == maxIter:
        print("failed to converge...")
        plt.plot(np.array(logg))
        plt.yscale('log')
        plt.show()
        exit()

    if show_plot:
        # print("test")
        plt.plot(np.array(logg))
        plt.yscale('log')
        plt.show()

    # if optIter > 30:
    #     print("test")
    #     plt.plot(np.array(logg))
    #     plt.yscale('log')
    #     plt.show()
    #     # exit()

    print("Opt Iter", optIter)
    # print("avg. PCG Iter", (pcgIter // optIter))

    avg_pcg_iter = pcgIter // optIter
    computeVelocity()
    # shrink()

    return optIter, avg_pcg_iter


@ti.kernel
def render_kernel(ratio: float, res_x: float, res_y: float):

    for i in x:
        xWorld[i][0] = (ratio / res_x) * x[i][0]
        xWorld[i][1] = (ratio / res_y) * x[i][1]


def render(gui):
    gui.clear(bg_color)
    pos_np = x.to_numpy()
    for j in range(dim):
        pos_np[:, j] *= screen_to_world_ratio / screen_res[j]
    gui.circles(pos_np, radius=particle_radius, color=particle_color)
    gui.rect(
        (0, 0),
        (board_states[None][0] / boundary[0], 1),
        radius=1.5,
        color=boundary_color,
    )
    gui.show()


@ti.kernel
def init_particles() -> float:

    mass.fill(0.0)
    delta = h_ * 1.0

    rho_f = 1e5
    center = ti.math.vec2(0.0)
    for i in range(num_fluids):
        x[i] = ti.Vector([i % num_particles_x, i // num_particles_x]) * delta
        mass[i] = rho_f

        if i // num_particles_x < 0.5 * num_particles_x:
            colors[i] = ti.math.vec3(135 / 255, 206/ 255, 235/ 255)
            mass[i] = rho_f
        else:
            mass[i] = rho_f
            colors[i] = ti.math.vec3(1.0, 0.0, 0.0)

        center += x[i]
        for c in ti.static(range(dim)):
            # v[i][c] = (ti.random() - 0.5) * 4
            v[i][c] = 0.0
    board_states[None] = ti.Vector([boundary[0] - epsilon, -0.0])
    center /= num_fluids

    for i in range(num_fluids):
        x[i] += (0.5 * ti.math.vec2([boundary[0], boundary[1]]) - center)
        x0[i] = x[i]

    center = 0.5 * ti.math.vec2([boundary[0], boundary[1]])
    r = 0.2 * ti.Vector([0.0, boundary[1]])
    delta_theta = (2 * ti.math.pi) / (num_solid)
    # delta_theta *= 0.8
    offset = ti.math.pi / (num_solid + 0.3)
    # offset = 0.0
    for si in range(num_solid):
        i = (num_particles - num_solid) + si
        theta = si * delta_theta + offset
        rot = ti.math.mat2([ti.cos(theta), -ti.sin(theta), ti.sin(theta), ti.cos(theta)])

        x[i] = rot @ r + center
        x0[i] = x[i]
        # center += x[i]
        colors[i] = ti.math.vec3([1.0, 1.0, 1.0])

        for c in ti.static(range(dim)):
            # v[i][c] = (ti.random() - 0.5) * 4
            v[i][c] = 0.0

    for si in range(num_solid):
        i = (num_particles - num_solid) + si
        colors[i] = ti.math.vec3([1.0, 1.0, 1.0])


    size = 1.0
    rho = 1e2

    l_min = 1e3
    for si in range(num_indices):
        i = (num_particles - num_solid) + si
        indices[2 * si + 0] = i
        indices[2 * si + 1] = i + 1
        l0[si] = size * (x[i + 1] - x[i]).norm()
        ti.atomic_min(l_min, l0[si])
        mass[i]     += rho * 0.5 * l0[si]
        mass[i + 1] += rho * 0.5 * l0[si]

    return l_min
    #
    # indices[2 * (num_indices - 1) + 0] = num_particles - num_solid
    # indices[2 * (num_indices - 1) + 1] = num_particles - 1
    # l0[num_solid - 1] = size * (x[num_particles - 1] - x[num_particles - num_solid]).norm()



def print_stats():
    print("PBF stats:")
    num = grid_num_particles.to_numpy()
    avg, max_ = np.mean(num), np.max(num)
    print(f"  #particles per cell: avg={avg:.2f} max={max_}")
    num = particle_num_neighbors.to_numpy()
    avg, max_ = np.mean(num), np.max(num)
    print(f"  #neighbors per particle: avg={avg:.2f} max={max_}")

@ti.kernel
def compute_color_code(tol: float):

    for c in range(num_fluids):

        # if rho[c] >= tol * poly6_value(0.0, h_):
        #     colors[c] = ti.math.vec3(1.0, 0.0, 0.0)
        # else:
        colors[c] = ti.math.vec3(135 / 255, 206/ 255, 235/ 255)



def main():
    global dHat
    frame_cnt = 0
    l_min = init_particles()
    print(f"boundary={boundary} grid={grid_size} cell_size={cell_size}")
    window = ti.ui.Window(name='PBF2D', res=screen_res, fps_limit=200, pos=(150, 150))
    canvas = window.get_canvas()
    canvas.set_background_color((0.2, 0.2, 0.2))
    run = False
    show_plot = False
    while window.running:

        if window.get_event(ti.ui.PRESS):
            if window.event.key == ' ':
                run = not run

            if window.event.key == 'r':
                l_min = init_particles()
                run = False
                frame_cnt = 0

            if window.event.key == 's':
                show_plot = True
            # if window.event.key == 's':
            #    shrink()
        # move_board()

        if run:
            # move_board()
            optIter, avg_pcg_iter = run_pbf(show_plot)
            show_plot = False
            frame_cnt += 1


        # neighbour_search(x)
        # computeDensity(x)
        #
        # compute_color_code(1.2)

        #
        # scalar = rho.to_numpy()
        # norm = Normalize(vmin=min(scalar), vmax=max(scalar))
        # colormap = colormaps['coolwarm']  # Access colormap from the registry
        # a = np.array([colormap(norm(val))[:3] for val in scalar])
        # a.reshape(num_particles, 3)
        #
        # for i in range(num_fluids, num_particles):
        #     a[i] = np.array([1.0, 1.0, 1.0])


        # print(a.shape)
        #
        # test = colors.to_numpy()
        # print(test.shape)
        # colors.from_numpy(a)

        render_kernel(screen_to_world_ratio, screen_res[0], screen_res[1])
        canvas.circles(xWorld, radius=0.003, per_vertex_color=colors)
        canvas.lines(xWorld, indices=indices, color=(1.0, 1.0, 1.0), width=0.003)
        window.show()
        #
        # if run:
        #     fileName = "opt" + str(frame_cnt) + ".png"
        #     window.save_image(fileName)


if __name__ == "__main__":
    main()