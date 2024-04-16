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
num_particles = 40
num_constraints = ti.field(dtype=float)

l0 = 0.4

frame = 0
max_num_particles_per_cell = 100
max_num_neighbors = 100
time_delta = 1.0 / 60.0
epsilon = 1e-5
particle_radius = 2.0
particle_radius_in_world = particle_radius / screen_res[1]
per_vertex_color = ti.Vector.field(3, ti.float32, shape=num_particles)
indices = np.zeros(num_particles)
# PBF params
h_ = 1.1
mass = 1.0
rho0 = 1.0
lambda_epsilon = 100.0
pbf_num_iters = 1
corr_deltaQ_coeff = 0.3
corrK = 0.001
# Need ti.pow()
# corrN = 4.0
neighbor_radius = h_ * 1.05
old_positions = ti.Vector.field(dim, float)
positions = ti.Vector.field(dim, float)
positions_render = ti.Vector.field(dim, float)
velocities = ti.Vector.field(dim, float)
grid_num_particles = ti.field(int)
grid2particles = ti.field(int)
particle_num_neighbors = ti.field(int)
particle_neighbors = ti.field(int)
lambdas = ti.field(float)
position_deltas = ti.Vector.field(dim, float)
# velocities_deltas = ti.Vector.field(dim, float)
# 0: x-pos, 1: timestep in sin()
board_states = ti.Vector.field(2, float)

ti.root.dense(ti.i, num_particles).place(old_positions, positions, velocities, positions_render, num_constraints)
grid_snode = ti.root.dense(ti.ij, grid_size)
grid_snode.place(grid_num_particles)
grid_snode.dense(ti.k, max_num_particles_per_cell).place(grid2particles)
nb_node = ti.root.dense(ti.i, num_particles)
nb_node.place(particle_num_neighbors)
nb_node.dense(ti.j, max_num_neighbors).place(particle_neighbors)
ti.root.dense(ti.i, num_particles).place(lambdas, position_deltas)
# ti.root.dense(ti.i, num_particles).place(lambdas, velocities_deltas)
ti.root.place(board_states)

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
    vel_strength = 2.0
    if b[1] >= 2 * period:
        b[1] = 0
    b[0] += -ti.sin(b[1] * np.pi / period) * vel_strength * time_delta
    board_states[None] = b


@ti.kernel
def prologue():
    # save old positions
    for i in positions:
        old_positions[i] = positions[i]
    # apply gravity within boundary
    for i in positions:
        g = ti.Vector([0.0, -9.81])
        pos, vel = positions[i], velocities[i]
        vel += g * time_delta
        pos += vel * time_delta
        positions[i] = pos


@ti.kernel
def substep():
    # compute lambdas
    # Eq (8) ~ (11)
    position_deltas.fill(0.0)
    for i in range(num_particles - 1):
      xij = positions[i + 1] - positions[i]
      com = 0.5 * (positions[i + 1] + positions[i])
      dir = ti.math.normalize(xij)
      pi = com - 0.5 * l0 * dir
      pj = com + 0.5 * l0 * dir
      position_deltas[i] += (pi - positions[i])
      position_deltas[i + 1] += (pj - positions[i + 1])

    # position_deltas[0] = ti.math.vec2([0.0, 0.0])

    for i in range(num_particles):
        positions[i] += (position_deltas[i] / num_constraints[i])

    positions[0] = old_positions[0]

@ti.kernel
def epilogue():

    for i in range(num_particles):
        velocities[i] = (positions[i] - old_positions[i]) / time_delta

    position_deltas.fill(0.0)
    for i in range(num_particles - 1):
        vi = 0.5 * (velocities[i + 1] + velocities[i])
        vj = 0.5 * (velocities[i + 1] + velocities[i])
        position_deltas[i] += (vi - velocities[i])
        position_deltas[i + 1] += (vj - velocities[i + 1])

    position_deltas[0] = ti.math.vec2([0.0, 0.0])

    # for i in range(num_particles):
    #     velocities[i] += (position_deltas[i] / num_constraints[i])

    for i in positions:
        positions[i] = old_positions[i] + velocities[i] * time_delta
        positions_render[i].x = positions[i].x * (screen_to_world_ratio / screen_res[0])
        positions_render[i].y = positions[i].y * (screen_to_world_ratio / screen_res[1])
  

def run_pbf():
    prologue()
    for _ in range(pbf_num_iters):
        substep()
    epilogue()




@ti.kernel
def init_particles():
    init_pos = ti.math.vec2(0.5 * boundary[0], boundary[1])
    dx = ti.math.vec2([l0, 0.])
    num_constraints.fill(2.0)
    num_constraints[0] = 1
    num_constraints[num_particles - 1] = 1

    for i in range(num_particles):
        positions[i] = init_pos + i * dx
    board_states[None] = ti.Vector([boundary[0] - epsilon, -0.0])


def print_stats():
    print("PBF stats:")
    num = grid_num_particles.to_numpy()
    avg, max_ = np.mean(num), np.max(num)
    print(f"  #particles per cell: avg={avg:.2f} max={max_}")
    num = particle_num_neighbors.to_numpy()
    avg, max_ = np.mean(num), np.max(num)
    print(f"  #neighbors per particle: avg={avg:.2f} max={max_}")

def main():
    init_particles()
    print(f"boundary={boundary} grid={grid_size} cell_size={cell_size}")
    window = ti.ui.Window(name="PBF 2D", res=screen_res)
    canvas = window.get_canvas()
    canvas.set_background_color((0.066, 0.18, 0.25))

    while window.running:
        run_pbf()
        # canvas.circles()
        canvas.circles(centers=positions_render, radius=particle_radius_in_world)
        window.show()

if __name__ == "__main__":
    main()