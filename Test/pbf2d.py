# Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.
# Taichi implementation by Ye Kuang (k-ye)

import math
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import taichi as ti

ti.init(arch=ti.gpu)


pixels = ti.field(ti.u8, shape=(512, 512, 3))

screen_res = (400, 400)
screen_to_world_ratio = 10.0
boundary = (screen_res[0] / screen_to_world_ratio, screen_res[1] / screen_to_world_ratio)

particle_radius = 2.0
h_ = 0.4 * particle_radius
cell_size = 2 * h_
cell_recpr = 1.0 / cell_size


def round_up(f, s):
    return (math.floor(f * cell_recpr / s) + 1) * s

grid_size = (round_up(boundary[0], 1), round_up(boundary[1], 1))
dim = 2
bg_color = 0x112F41
particle_color = 0x068587
boundary_color = 0xEBACA2
num_particles_x = 40
num_particles = num_particles_x * 40
max_num_particles_per_cell = 100
max_num_neighbors = 100
time_delta = 1.0 / 600.0
epsilon = 1e-5
particle_radius_in_world = particle_radius / screen_res[1]
per_vertex_color = ti.Vector.field(3, ti.float32, shape=num_particles)
indices = np.zeros(num_particles)
# PBF params
mass = 1.0
rho0 = 1.0
lambda_epsilon = 100.0
pbf_num_iters = 2
corr_deltaQ_coeff = 0.3
corrK = 0.001
# Need ti.pow()
# corrN = 4.0
neighbor_radius = h_ * 1.05

poly6_factor = 315.0 / 64.0 / math.pi
spiky_grad_factor = -45.0 / math.pi

b = ti.Vector.field(2, dtype=ti.f32, shape=1)
b[0] = ti.math.vec2(0.5, 0.5)

x_old = ti.Vector.field(dim, float, shape=num_particles)
x = ti.Vector.field(dim, float, shape=num_particles)
positions_render = ti.Vector.field(dim, float)
v = ti.Vector.field(dim, float, shape=num_particles)
grid_num_particles = ti.field(int)
grid2particles = ti.field(int)
particle_num_neighbors = ti.field(int)
particle_neighbors = ti.field(int)
ld = ti.field(float, shape=num_particles)
dx = ti.Vector.field(dim, float, shape=num_particles)
# velocities_deltas = ti.Vector.field(dim, float)
# 0: x-pos, 1: timestep in sin()
board_states = ti.Vector.field(2, float)

ti.root.dense(ti.i, num_particles).place(positions_render)
grid_snode = ti.root.dense(ti.ij, grid_size)
grid_snode.place(grid_num_particles)
grid_snode.dense(ti.k, max_num_particles_per_cell).place(grid2particles)
nb_node = ti.root.dense(ti.i, num_particles)
nb_node.place(particle_num_neighbors)
nb_node.dense(ti.j, max_num_neighbors).place(particle_neighbors)
# ti.root.dense(ti.i, num_particles).place(lambdas, velocities_deltas)
ti.root.place(board_states)

grid_particles_num = ti.field(int, shape=int(grid_size[0] * grid_size[1]))
grid_particles_num_temp = ti.field(int, shape=int(grid_size[0] * grid_size[1]))
prefix_sum_executor = ti.algorithms.PrefixSumExecutor(grid_particles_num.shape[0])

grid_ids = ti.field(int, shape=num_particles)
grid_ids_buffer = ti.field(int, shape=num_particles)
grid_ids_new = ti.field(int, shape=num_particles)
cur2org = ti.field(int, shape=num_particles)


@ti.func
def poly6_value(s, h):
    result = 0.0
    if 0 < s and s < h:
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
def get_cell(pos):
    return (pos / cell_size).cast(int)


@ti.func
def is_in_grid(c):
    # @c: Vector(i32)
    return 0 <= c[0] and c[0] < grid_size[0] and 0 <= c[1] and c[1] < grid_size[1]


@ti.func
def confine_position_to_boundary(p):
    bmin = particle_radius_in_world
    bmax = ti.Vector([board_states[None][0], boundary[1]]) - particle_radius_in_world
    padding = 2.0
    for i in ti.static(range(dim)):
        # Use randomness to prevent particles from sticking into each other after clamping
        if p[i] <= bmin + padding:
            p[i] = bmin + padding
        elif bmax[i] - padding <= p[i]:
            p[i] = bmax[i] - padding

    # if p[0] > grid_size[0]:
    #     p[0] = grid_size[0]
    #
    # if p[0] < 0:
    #     p[0] = 0
    #
    # if p[1] > grid_size[1]:
    #     p[1] = grid_size[1]
    #
    # if p[1] < 0:
    #     p[1] = 0

    return p




@ti.kernel
def move_board():
    # probably more accurate to exert force on particles according to hooke's law.
    b = board_states[None]
    b[1] += 1.0
    period = 45
    vel_strength = 2.0
    if b[1] >= 2 * period:
        b[1] = 0
    b[0] += -ti.sin(b[1] * np.pi / period) * vel_strength * time_delta
    board_states[None] = b


@ti.func
def flatten_grid_index(grid_index):
    return grid_index[0] * grid_size[1] + grid_index[1]

def broad_phase():
    update_grid_id()
    prefix_sum_executor.run(grid_particles_num)
    counting_sort()


@ti.kernel
def update_grid_id():
    for I in ti.grouped(grid_particles_num):
        grid_particles_num[I] = 0

    # TODO: update the following two for-loops into a single one
    for i in range(num_particles):
        grid_ids[i] = flatten_grid_index(get_cell(x[i]))
        ti.atomic_add(grid_particles_num[grid_ids[i]], 1)

    for I in ti.grouped(grid_particles_num):
        grid_particles_num_temp[I] = grid_particles_num[I]

@ti.kernel
def counting_sort():
    for i in range(num_particles):
        I = num_particles - 1 - i
        base_offset = 0
        if grid_ids[I] - 1 >= 0:
            base_offset = grid_particles_num[grid_ids[I] - 1]
        grid_ids_new[I] = ti.atomic_sub(grid_particles_num_temp[grid_ids[I]], 1) - 1 + base_offset

    for i in grid_ids:
        new_index = grid_ids_new[i]
        cur2org[new_index] = i

@ti.kernel
def prologue():
    # save old positions
    for i in x:
        x_old[i] = x[i]
    # apply gravity within boundary
    for i in x:
        g = ti.Vector([0.0, -9.81])
        xi, vi = x[i], v[i]
        vi += g * time_delta
        xi += vi * time_delta
        x[i] = confine_position_to_boundary(xi)

    # # clear neighbor lookup table
    # for I in ti.grouped(grid_num_particles):
    #     grid_num_particles[I] = 0
    # for I in ti.grouped(particle_neighbors):
    #     particle_neighbors[I] = -1
    #
    # # update grid
    # for p_i in x:
    #     cell = get_cell(x[p_i])
    #     # ti.Vector doesn't seem to support unpacking yet
    #     # but we can directly use int Vectors as indices
    #     offs = ti.atomic_add(grid_num_particles[cell], 1)
    #     grid2particles[cell, offs] = p_i
    # # find particle neighbors
    # for p_i in x:
    #     pos_i = x[p_i]
    #     cell = get_cell(pos_i)
    #     nb_i = 0
    #     for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
    #         cell_to_check = cell + offs
    #         if is_in_grid(cell_to_check):
    #             for j in range(grid_num_particles[cell_to_check]):
    #                 p_j = grid2particles[cell_to_check, j]
    #                 # if nb_i < max_num_neighbors:
    #                 particle_neighbors[p_i, nb_i] = p_j
    #                 nb_i += 1
    #     particle_num_neighbors[p_i] = nb_i


@ti.func
def get_flatten_grid_index(pos: ti.math.vec2):
    return flatten_grid_index(get_cell(pos))

@ti.func
def flatten_grid_index(grid_index):
    return grid_index[0] * grid_size[1] + grid_index[1]


@ti.kernel
def substep():
    # compute lambdas
    # Eq (8) ~ (11)

    dx.fill(0.0)
    for p_i in x:
        xi = x[p_i]

        grad_i = ti.Vector([0.0, 0.0])
        sum_gradient_sqr = 0.0
        density_constraint = 0.0

        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            pos_ji = xi - x[p_j]
            grad_j = spiky_gradient(pos_ji, h_)
            grad_i += grad_j
            sum_gradient_sqr += grad_j.dot(grad_j)
            # Eq(2)
            density_constraint += poly6_value(pos_ji.norm(), h_)

        # Eq(1)
        density_constraint = density_constraint - 1.0
        if density_constraint < 0:
            density_constraint = 0.

        sum_gradient_sqr += grad_i.dot(grad_i)
        ld = (-density_constraint) / (sum_gradient_sqr + lambda_epsilon)

        for j in range(particle_num_neighbors[p_i]):
            p_j = particle_neighbors[p_i, j]
            pos_ji = xi - x[p_j]
            dx[p_j] -= ld * spiky_gradient(pos_ji, h_)

    for i in x:
        x[i] += dx[i]

@ti.kernel
def solve_pressure_constraints_x():

    dx.fill(0.0)
    for i in x:
        grad_i = ti.math.vec2(0.0)
        sum_gradient_sqr = 0.0
        density_constraint = 0.0
        center_cell = get_cell(x[i])
        for offset in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
            grid_index = flatten_grid_index(center_cell + offset)

            # if grid_index < 1:
            #     grid_index = 1

            if grid_index > int(grid_size[0] * grid_size[1]) - 1:
                grid_index =int(grid_size[0] * grid_size[1]) - 1

            for p_j in range(grid_particles_num[ti.max(0, grid_index - 1)], grid_particles_num[grid_index]):
                j = cur2org[p_j]
                pos_ji = x[i] - x[j]
                grad_j = spiky_gradient(pos_ji, h_)
                grad_i += grad_j
                sum_gradient_sqr += grad_j.dot(grad_j)
                # Eq(2)
                density_constraint += poly6_value(pos_ji.norm(), h_)

                # Eq(1)
        density_constraint = density_constraint - 1.0
        if density_constraint < 0:
            density_constraint = 0.

        sum_gradient_sqr += grad_i.dot(grad_i)
        ld = (-density_constraint) / (sum_gradient_sqr + lambda_epsilon)
        for offset in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
            grid_index = flatten_grid_index(center_cell + offset)
            if grid_index < 1:
                grid_index = 1

            if grid_index > int(grid_size[0] * grid_size[1]) - 1:
                grid_index = int(grid_size[0] * grid_size[1]) - 1
            for p_j in range(grid_particles_num[grid_index - 1], grid_particles_num[grid_index]):
                j = cur2org[p_j]
                pos_ji = x[i] - x[j]
                dx[j] -= ld * spiky_gradient(pos_ji, h_)


    for i in x:
        x[i] += dx[i]


@ti.kernel
def epilogue():
    # confine to boundary
    for i in x:
        x[i] = confine_position_to_boundary(x[i])

    # update velocities
    for i in x:
        v[i] = (x[i] - x_old[i]) / time_delta


    for i in x:
        positions_render[i].x = x[i].x * (screen_to_world_ratio / screen_res[0])
        positions_render[i].y = x[i].y * (screen_to_world_ratio / screen_res[1])
    # no vorticity/xsph because we cannot do cross product in 2D...

    # for i in velocities:
    #     vel_norm_i = ti.math.length(velocities[i])
    #
    #     if vel_norm_i <=
    #     (0.098, 0.51, 0.77)
    #     (0.13, 0.65, 0.94)
    #     (0.38, 0.74, 0.94)
    #     (0.65, 0.83, 0.92)
    #     (0.88, 0.88, 0.88)
    #
    #
    #
    # # ["#1984c5", "#22a7f0", "#63bff0", "#a7d5ed", "#e2e2e2", "#e1a692", "#de6e56", "#e14b31", "#c23728"]
    #
    # per_vertex_color.fill(ti.math.vec3(6, 133, 135))

def run_pbf():
    prologue()
    broad_phase()
    for _ in range(pbf_num_iters):
        # substep()
        solve_pressure_constraints_x()
    epilogue()


@ti.kernel
def init_particles():
    for i in range(num_particles):
        delta = h_ * 0.8
        offs = ti.Vector([(boundary[0] - delta * num_particles_x) * 0.5, boundary[1] * 0.02])
        x[i] = ti.Vector([i % num_particles_x, i // num_particles_x]) * delta + offs
        for c in ti.static(range(dim)):
            v[i][c] = (ti.random() - 0.5) * 4
            v[i][c] = 0
    board_states[None] = ti.Vector([boundary[0] - epsilon, -0.0])
    for i in x:
        positions_render[i].x = x[i].x * (screen_to_world_ratio / screen_res[0])
        positions_render[i].y = x[i].y * (screen_to_world_ratio / screen_res[1])


def print_stats():
    print("PBF stats:")
    num = grid_num_particles.to_numpy()
    avg, max_ = np.mean(num), np.max(num)
    print(f"  #particles per cell: avg={avg:.2f} max={max_}")
    num = particle_num_neighbors.to_numpy()
    avg, max_ = np.mean(num), np.max(num)
    print(f"  #neighbors per particle: avg={avg:.2f} max={max_}")

#
# @ti.kernel
# def compute_heat_map_rgb():
#
#     arr = velocities.to_numpy()

def main():
    run_sim = True
    init_particles()
    print(f"boundary={boundary} grid={grid_size} cell_size={cell_size}")
    window = ti.ui.Window(name="PBF 2D", res=screen_res)
    canvas = window.get_canvas()
    canvas.set_background_color((0.066, 0.18, 0.25))

    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == 'r':
               init_particles()
            if window.event.key == ' ':
                run_sim = not run_sim

        if run_sim:
            run_pbf()
        # move_board()

        arr = v.to_numpy()
        magnitudes = np.linalg.norm(arr, axis=1)  # Compute magnitudes of vectors
        norm = Normalize(vmin=np.min(magnitudes), vmax=np.max(magnitudes))
        heatmap_rgb = plt.cm.coolwarm(norm(magnitudes))[:, :3]  # Use plasma colormap for heatmap
        per_vertex_color.from_numpy(heatmap_rgb)
        canvas.circles(centers=positions_render, radius=particle_radius_in_world, per_vertex_color=per_vertex_color)
        window.show()

if __name__ == "__main__":
    main()