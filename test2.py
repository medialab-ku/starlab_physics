# Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.
# Taichi implementation by Ye Kuang (k-ye)

import math

import numpy as np
import matplotlib.pyplot as plt
import taichi as ti
import taichi.ui

ti.init(arch=ti.gpu)

screen_res = (800, 400)
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
num_particles_x = 105
num_particles = num_particles_x * 24
max_num_particles_per_cell = 50
max_num_neighbors = 50
time_delta = 1.0 / 20.0
epsilon = 1e-5
particle_radius = 10.0
particle_radius_in_world = particle_radius / screen_to_world_ratio

# PBF params
h_ = 1.1
# mass = 1.0
# rho0 = 1.0
lambda_epsilon = 100.0
pbf_num_iters = 20
mass_ratio = 10.0
PR = 1e2
solver_type = 1
use_heatmap = False
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
hii = ti.field( float, shape=num_particles)
gii = ti.Vector.field(dim, float, shape=num_particles)

x0 = ti.Vector.field(dim, float, shape=num_particles)
positions_window = ti.Vector.field(dim, float, shape=num_particles)
colors = ti.Vector.field(3, float, shape=num_particles)
heat_map = ti.Vector.field(3, float, shape=num_particles)
velocities = ti.Vector.field(dim, float, shape=num_particles)
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
position_deltas = ti.Vector.field(dim, float, shape=num_particles)

# Dm_inv = ti.Matrix.field(dim, float, shape=num_particles)
num_boundary_particles_x = 40
num_boundary_particles = 10 * num_boundary_particles_x
boundary_positions = ti.Vector.field(dim, float, shape=num_boundary_particles)
boundary_positions_window = ti.Vector.field(dim, float, shape=num_boundary_particles)
# num_boundary_particles_x = 40
# num_boundary_particles = 10 * num_boundary_particles_x
# ti.root.dense(ti.i, num_boundary_particles).place(boundary_positions, boundary_positions_window)
# board_states = ti.Vector.field(2, float)
@ti.func
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
def prologue(mass_ratio: float):

    for i in range(num_particles):
        if material_type[i] == 0:
            # positions[i] += offs
            mass[i] = 1.0
            rho0[i] = mass[i] * poly6_value(0.0, h_)

        elif material_type[i] == 1:
            # positions[i] -= offs
            mass[i] = mass_ratio
            rho0[i] = mass[i] * poly6_value(0.0, h_)

        elif material_type[i] == 2:
            mass[i] = (mass_ratio ** 2)
            # rho0[i] = mass[i] * poly6_value(0.0, h_)

    for p_i in positions:
        pos_i = x0[p_i]
        # Dm = ti.math.mat2(0.0)
        V0 = 0.0
        mass_i = 0.0
        for j in range(particle_num_neighbors_rest[p_i]):
            p_j = particle_neighbors_rest[p_i, j]
            if p_j < 0:
                break
            pos_ji = x0[p_j] - pos_i
            # Dm += poly6_value(pos_ji.norm(), h_) * outer_product(pos_ji, pos_ji)
            mass_i += poly6_value(pos_ji.norm(), h_) * mass[p_j]

        mass[p_i] = mass_i
        rho0[p_i] = poly6_value(0.0, h_) * mass_i
        # invDm[p_i] = Dm.inverse()

    # save old positions
    for i in positions:
        old_positions[i] = positions[i]

    # apply gravity within boundary
    for i in positions:
        g = ti.Vector([0.0, 0.0])
        pos, vel = positions[i], velocities[i]
        vel += g * time_delta
        pos += vel * time_delta
        positions[i] = confine_position_to_boundary(pos)

    # clear neighbor lookup table
    for I in ti.grouped(grid_num_particles):
        grid_num_particles[I] = 0
    # for I in ti.grouped(particle_neighbors):
    #     particle_neighbors[I] = -1

    # update grid
    for p_i in positions:
        cell = get_cell(positions[p_i])
        # ti.Vector doesn't seem to support unpacking yet
        # but we can directly use int Vectors as indices
        offs = ti.atomic_add(grid_num_particles[cell], 1)
        grid2particles[cell, offs] = p_i
    # find particle neighbors
    for p_i in positions:
        pos_i = positions[p_i]
        cell = get_cell(pos_i)
        nb_i = 0
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
            cell_to_check = cell + offs
            if is_in_grid(cell_to_check):
                for j in range(grid_num_particles[cell_to_check]):
                    p_j = grid2particles[cell_to_check, j]
                    if nb_i < max_num_neighbors and p_j != p_i and (pos_i - positions[p_j]).norm() < neighbor_radius:
                        particle_neighbors[p_i, nb_i] = p_j
                        nb_i += 1
        particle_num_neighbors[p_i] = nb_i
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
def substep_fem(k: float):

    hii.fill(1.0)
    gii.fill(0.0)

    for p_i in positions:
        pos_i = positions[p_i]
        xi0 = x0[p_i]
        Ds = ti.math.mat2(0.0)
        for j in range(particle_num_neighbors_rest[p_i]):
            p_j = particle_neighbors_rest[p_i, j]
            if p_j < 0:
                break
            pos_ji = positions[p_j] - pos_i
            xji0 = x0[p_j] - xi0
            Ds += poly6_value(xji0.norm(), h_) * outer_product(pos_ji, xji0)

        F = Ds @ invDm[p_i]
        U, sig, V = ssvd(F)
        R = U @ V.transpose()
        wi = k
        for j in range(particle_num_neighbors_rest[p_i]):
            p_j = particle_neighbors_rest[p_i, j]
            if p_j < 0:
                break
            x0ji = x0[p_j] - xi0
            xji = positions[p_j] - pos_i
            dxji = R @ x0ji - xji

            gii[p_j] += wi * dxji
            gii[p_i] -= wi * dxji

            hii[p_i] += wi
            hii[p_j] += wi

    for i in positions:
        positions[i] += gii[i] / hii[i]



@ti.kernel
def epilogue():
    # confine to boundary
    for i in positions:
        pos = positions[i]
        positions[i] = confine_position_to_boundary(pos)
    # update velocities
    for i in positions:
        velocities[i] = (positions[i] - old_positions[i]) / time_delta


    # no vorticity/xsph because we cannot do cross product in 2D...


def run_pbf():
    prologue(mass_ratio)
    if solver_type == 0:
        for _ in range(pbf_num_iters):
            substep()
            k = PR * time_delta ** 2
            substep_fem(k)

    elif solver_type == 1:
        for _ in range(pbf_num_iters):
            substep()

        for _ in range(pbf_num_iters):
            k = PR * time_delta ** 2
            substep_fem(k)

    epilogue()

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

@ti.kernel
def init_particles(mass_ratio: float):
    # print(boundary)

    off_test = ti.math.vec2([0.0, boundary[1] * 0.2])
    for i in range(num_particles):
        delta = h_ * 0.7
        offs = ti.Vector([(boundary[0] - delta * num_particles_x) * 0.5, boundary[1] * 0.2])
        x = i % num_particles_x
        y = i // num_particles_x
        # mass[i] = 1.0
        if 0 <= y < 12:
            colors[i] = ti.math.vec3(255.0, 128.0, 0.0) / 255.0
            material_type[i] = 1
            positions[i] = ti.Vector([x, y]) * delta + offs + off_test

        elif 12 <= y < 24:
            colors[i] = ti.math.vec3(0.0, 128.0, 255.0) / 255.0
            material_type[i] = 0
            positions[i] = ti.Vector([x, y]) * delta + offs
        # else:
        #     colors[i] = ti.math.vec3(255.0, 0.0, 128.0) / 255.0
        #     material_type[i] = 2
        #     positions[i] = ti.Vector([x, y]) * delta + offs - off_test
        x0[i] = positions[i]
        # positions[i] = ti.Vector([x, y]) * delta + offs
    # delta = h_ * 0.7
    # offs = ti.Vector([.0, boundary[1] * 0.1])
    for i in range(num_particles):
        if material_type[i] == 0:
            # positions[i] += offs
            mass[i] = 1.0
            rho0[i] = mass[i] * poly6_value(0.0, h_)

        elif material_type[i] == 1:
            # positions[i] -= offs
            mass[i] = mass_ratio
            rho0[i] = mass[i] * poly6_value(0.0, h_)

        elif material_type[i] == 2:
            mass[i] = 10 * mass_ratio
            # rho0[i] = mass[i] * poly6_value(0.0, h_)

    velocities.fill(0.0)
    board_states[None] = ti.Vector([boundary[0] - epsilon, -0.0])

    # clear neighbor lookup table
    for I in ti.grouped(grid_num_particles):
        grid_num_particles[I] = 0

    for I in ti.grouped(particle_neighbors):
        particle_neighbors_rest[I] = -1

    for p_i in positions:
        cell = get_cell(positions[p_i])
        # ti.Vector doesn't seem to support unpacking yet
        # but we can directly use int Vectors as indices
        offs = ti.atomic_add(grid_num_particles[cell], 1)
        grid2particles[cell, offs] = p_i

    # find particle neighbors
    for p_i in positions:
        pos_i = x0[p_i]
        cell = get_cell(pos_i)
        nb_i = 0
        for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
            cell_to_check = cell + offs
            if is_in_grid(cell_to_check):
                for j in range(grid_num_particles[cell_to_check]):
                    p_j = grid2particles[cell_to_check, j]
                    if (nb_i < max_num_neighbors and p_j != p_i
                            and (pos_i - positions[p_j]).norm() < h_) and material_type[p_i] == material_type[p_j]:
                        particle_neighbors_rest[p_i, nb_i] = p_j
                        nb_i += 1
        if nb_i < 3:
            print("fuck")
        particle_num_neighbors_rest[p_i] = nb_i

    for p_i in positions:
        pos_i = x0[p_i]
        Dm = ti.math.mat2(0.0)
        V0 = 0.0
        mass_i = 0.0
        for j in range(particle_num_neighbors_rest[p_i]):
            p_j = particle_neighbors_rest[p_i, j]
            if p_j < 0:
                break
            pos_ji = x0[p_j] - pos_i
            Dm += poly6_value(pos_ji.norm(), h_) * outer_product(pos_ji, pos_ji)
            mass_i += poly6_value(pos_ji.norm(), h_) * mass[p_j]

        # mass[p_i] = mass_i
        # rho0[p_i] = poly6_value(0.0, h_) * mass_i
        invDm[p_i] = Dm.inverse()

@ti.kernel
def switch_material():
    # print(boundary)
    for i in range(num_particles):
        if material_type[i] == 0:
            material_type[i] = 1
        elif material_type[i] == 1:
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


def print_stats():
    print("PBF stats:")
    num = grid_num_particles.to_numpy()
    avg, max_ = np.mean(num), np.max(num)
    print(f"  #particles per cell: avg={avg:.2f} max={max_}")
    num = particle_num_neighbors.to_numpy()
    avg, max_ = np.mean(num), np.max(num)
    print(f"  #neighbors per particle: avg={avg:.2f} max={max_}")


def main():

    run_sim = False
    init_particles(mass_ratio)
    print(f"boundary={boundary} grid={grid_size} cell_size={cell_size}")

    window = ti.ui.Window(name='PBF2D', res = screen_res, fps_limit=200, pos = (150, 150))
    gui = window.get_gui()
    canvas = window.get_canvas()
    canvas.set_background_color((0.2, 0.2, 0.2))

    def show_options():

        global pbf_num_iters
        global mass_ratio
        global use_heatmap
        global solver_type
        # global dt_ui
        # global g_ui
        # global damping_ui
        # global YM_ui
        global PR

        # old_dHat = dHat_ui
        # old_damping = damping_ui
        # YM_old = YM_ui
        PR_old = PR
        mass_ratio_old = mass_ratio

        with gui.sub_window("XPBD Settings", 0., 0., 0.3, 0.7) as w:

            # dt_ui = w.slider_float("dt", dt_ui, 0.001, 0.101)
            # g_ui = w.slider_float("g", g_ui, -20.0, 20.0)
            # pbf_num_iters = w.slider_int("# iter", pbf_num_iters, 1, 100)
            solver_type = w.slider_int("solver type", solver_type, 0, 1)
            if solver_type == 0:
                w.text("locking")
            elif solver_type == 1:
                w.text("locking-free")

            pbf_num_iters = w.slider_int("# sub", pbf_num_iters, 1, 100)
            mass_ratio_old = w.slider_float("mass ratio", mass_ratio_old, 1, 100)
            PR_old = w.slider_float("PR", PR_old, 0.0, 1e5)
            use_heatmap = w.checkbox("heat map", use_heatmap)
            # YM_ui = w.slider_int("Young's Modulus", YM_ui, -1, 5)
            # PR_ui = w.slider_float("Poisson's Ratio", PR_ui, 0.0, 0.49)

        if not mass_ratio_old == mass_ratio:
            mass_ratio = mass_ratio_old
        #
        # if not YM_old == YM_ui:
        #     sim.YM = YM_ui
        #
        if not PR_old == PR:
            PR = PR_old
    while window.running:

        if window.get_event(ti.ui.PRESS):
            if window.event.key == ' ':
                run_sim = not run_sim

            if window.event.key == 'r':
                init_particles(mass_ratio)
                run_sim = False

            if window.event.key == 's':
                switch_material()

        if run_sim:
            run_pbf()

        show_options()
        render_kernel(positions, positions_window, screen_to_world_ratio, screen_res[0], screen_res[1])
        radius = 0.5 * (screen_to_world_ratio / screen_res[0]) * h_

        if use_heatmap:
            rho0_np = rho0.to_numpy()
            colormap = plt.colormaps['plasma']
            norm = plt.Normalize(vmin=np.min(rho0_np), vmax=np.max(rho0_np))
            rgb_array = colormap(norm(rho0_np))[:, :3]
            # print(rgb_array.shape)
            heat_map.from_numpy(rgb_array)
            canvas.circles(positions_window, radius=radius, per_vertex_color=heat_map)
        else:
            canvas.circles(positions_window, radius=radius, per_vertex_color=colors)
        window.show()

if __name__ == "__main__":
    main()