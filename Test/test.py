import taichi as ti
import numpy as np
import math

ti.init(arch=ti.gpu)

# -----PARAMETERS-----

# -WORLD-
dt = 1.0 / 20.0
solve_iteration = 5
res = (500, 500)
world = (50, 50)
boundary = 50
dimension = 2

# -Visual-
background_color = 0xe9f5f3
visual_radius = 2.5
particle_color = 0x34ebc6

# -Fluid_Setting-
num_particles = 1200
mass = 1.0
density = 1.0
rest_density = 1.0
radius = 0.4


# -Neighbours_Setting-
h = 1.0
h_2 = h * h
h_6 = h * h * h * h * h * h
h_9 = h * h * h * h * h * h * h * h * h
max_neighbour = 800

# -Grid_Setting-
grid_size = 1
grid_rows = world[0] // grid_size
grid_cols = world[1] // grid_size
max_particle_in_grid = 500

# -POLY6_KERNEL-
poly6_Coe = 315.0 / (64 * math.pi)

# -SPIKY_KERNEL-
spiky_Coe = -45.0 / math.pi

# -LAMBDAS-
lambda_epsilon = 100.0

# -S_CORR-
S_Corr_delta_q = 0.3
S_Corr_k = 0.0001

# ---Confinement/ XSPH Viscosity---
xsph_c = 0.01

# -----FIELDS-----
position = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)
last_position = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)
velocity = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)
lambdas = ti.field(dtype=ti.f32, shape=num_particles)
delta_qs = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)

# Wacky check table for grid
num_particle_in_grid = ti.field(dtype=ti.i32, shape=(grid_rows, grid_cols))
table_grid = ti.field(dtype=ti.i32, shape=(grid_rows, grid_cols, max_particle_in_grid))

# Wacky check table for neighbour
num_nb = ti.field(dtype=ti.i32, shape=num_particles)
table_nb = ti.field(dtype=ti.i32, shape=(num_particles, max_neighbour))

# --------------------FUNCTIONS--------------------


@ti.func
def poly6(dist):
    # dist is a VECTOR
    result = 0.0
    d = dist.norm()
    if 0 < d < h:
        rhs = (h_2 - d * d) * (h_2 - d * d) * (h_2 - d * d)
        result = poly6_Coe * rhs / h_9
    return result


@ti.func
def poly6_scalar(dist):
    # dist is a SCALAR
    result = 0.0
    d = dist
    if 0 < d < h:
        rhs = (h_2 - d * d) * (h_2 - d * d) * (h_2 - d * d)
        result = poly6_Coe * rhs / h_9
    return result


@ti.func
def spiky(dist):
    # dist is a VECTOR
    result = ti.Vector([0.0, 0.0])
    d = dist.norm()
    if 0 < d < h:
        m = (h - d) * (h - d)
        result = (spiky_Coe * m / (h_6 * d)) * dist
    return result
    # -Switch to 3D Vector when running in 3D
    # return ti.Vector([0.0, 0.0, 0.0])


@ti.func
def S_Corr(dist):
    upper = poly6(dist)
    lower = poly6_scalar(S_Corr_delta_q)
    m = upper/lower
    return -1.0 * S_Corr_k * m * m * m * m


@ti.func
def boundary_condition(v):
    # position filter
    # v is the position in vector form
    lower = radius
    upper = world[0] - radius
    # ---True Boundary---
    if v[0] <= lower:
        v[0] = lower + ti.random() * 1e-8
    elif upper <= v[0]:
        v[0] = upper - ti.random() * 1e-8
    if v[1] <= lower:
        v[1] = lower + ti.random() * 1e-8
    elif upper <= v[1]:
        v[1] = upper - ti.random() * 1e-8
    return v


@ti.func
def get_grid(cord):
    new_cord = boundary_condition(cord)
    g_x = int(new_cord[0] // grid_size)
    g_y = int(new_cord[1] // grid_size)
    return g_x, g_y


# --------------------KERNELS--------------------
# avoid nested for loop of position O(n^2)!

@ti.kernel
def pbf_prep():
    # ---save position---
    for p in position:
        last_position[p] = position[p]


@ti.kernel
def pbf_apply_force(d: float):
    # ---apply gravity/forces. Update velocity---
    gravity = ti.Vector([0.0, -5.8])
    force = ti.Vector([5.0, 1.0])
    for i in velocity:
        velocity[i] += dt * (gravity + d * force)
        # ---predict position---
        position[i] += dt * velocity[i]
        # ---Just in case it flew out of the grid too much---
        # position[i] = boundary_condition(position[i])


@ti.kernel
def pbf_neighbour_search():
    # ---clean tables---
    for i, j in num_particle_in_grid:
        num_particle_in_grid[i, j] = 0
    for I in ti.grouped(table_grid):
        table_grid[I] = -1
    for i in num_nb:
        num_nb[i] = 0
    for i, j in table_nb:
        table_nb[i, j] = -1
    # ---update grid---
    for p in position:
        pos = position[p]
        # p_grid = (pos[0] // grid_size, pos[1] // grid_size)
        p_grid = get_grid(pos)
        if p_grid[0] > grid_cols or p_grid[1] > grid_rows:
            print("grid error")
        g_index = ti.atomic_add(num_particle_in_grid[p_grid[0], p_grid[1]], 1)
        # ---ERROR CHECK---
        if g_index >= max_particle_in_grid:
            print("Grid overflows.")
        table_grid[p_grid[0], p_grid[1], g_index] = p
    # ---update neighbour---
    for p in position:
        pos = position[p]
        # p_grid = (pos[0] // grid_size, pos[1] // grid_size)
        p_grid = get_grid(pos)
        if p_grid[0] > grid_cols or p_grid[1] > grid_rows:
            print("grid error")
        # nb_grid = neighbour_gird(p_grid)
        for off_x in ti.static(range(-1, 2)):
            for off_y in ti.static(range(-1, 2)):
                if 0 <= p_grid[0] + off_x <= grid_cols:
                    if 0 <= p_grid[1] + off_y <= grid_rows:
                        nb = (p_grid[0] + off_x, p_grid[1] + off_y)
                        for i in range(num_particle_in_grid[nb[0], nb[1]]):
                            new_nb = table_grid[nb[0], nb[1], i]
                            n_index = ti.atomic_add(num_nb[p], 1)
                            # ---ERROR CHECK---
                            # if n_index >= max_neighbour:
                            #     print("Neighbour overflows.")
                            table_nb[p, n_index] = new_nb


@ti.kernel
def pbf_solve():
    # ---Calculate lambdas---
    for p in position:
        pos = position[p]
        lower_sum = 0.0
        p_i = 0.0
        spiky_i = ti.Vector([0.0, 0.0])
        for i in range(num_nb[p]):
            # ---Poly6---
            nb_index = table_nb[p, i]
            nb_pos = position[nb_index]
            p_i += mass * poly6(pos - nb_pos)
            # ---Spiky---
            s = spiky(pos - nb_pos) / rest_density
            spiky_i += s
            lower_sum += s.dot(s)
        constraint = (p_i / rest_density) - 1.0
        lower_sum += spiky_i.dot(spiky_i)
        lambdas[p] = -1.0 * (constraint / (lower_sum + lambda_epsilon))
    # ---Calculate delta Q---
    for p in position:
        delta_q = ti.Vector([0.0, 0.0])
        pos = position[p]
        for i in range(num_nb[p]):
            nb_index = table_nb[p, i]
            nb_pos = position[nb_index]
            # ---S_Corr---
            scorr = S_Corr(pos - nb_pos)
            left = lambdas[p] + lambdas[nb_index] + scorr
            right = spiky(pos - nb_pos)
            delta_q += left * right / rest_density
        delta_qs[p] = delta_q
    # ---Update position with delta Q---
    for p in position:
        position[p] += delta_qs[p]


@ti.kernel
def pbf_update():
    # ---Update Position---
    for p in position:
        position[p] = boundary_condition(position[p])
    # ---Check Boundary---
    for v in velocity:
        velocity[v] = (position[v] - last_position[v]) / dt



def pbf(d):
    pbf_prep()
    pbf_apply_force(d)
    pbf_neighbour_search()
    for i in range(solve_iteration):
        pbf_solve()
    pbf_update()


@ti.kernel
def init():
    for i in position:
        pos_x = 10 + 0.75 * (i % 40)
        pos_y = 1 + 0.8 * (i // 40)
        position[i] = ti.Vector([pos_x, pos_y])


def render(gui):
    gui.clear(background_color)
    render_position = position.to_numpy()
    render_position /= boundary
    gui.circles(render_position, radius=visual_radius, color=particle_color)
    gui.show()


def main():
    init()
    gui = ti.GUI('Position Based Fluid', res)
    while gui.running:
        gui.get_event()
        if gui.is_pressed('a'):
            pbf(-1.0)
        elif gui.is_pressed('d'):
            pbf(1.0)
        else:
            pbf(0.0)
        render(gui)
    return 0


if __name__ == '__main__':
    main()