import taichi as ti
import numpy as np
import math

ti.init(arch=ti.gpu)

# -----PARAMETERS-----

# -WORLD-
dt = 1.0 / 20.0
solve_iteration = 10
res = (500, 500)
world = (20, 20, 20)
boundary = 20
dimension = 3

# -Visual-
background_color = 0xe9f5f3
visual_radius = 2.5
particle_color = 0x34ebc6

# -Fluid_Setting-
num_particles = 12000
mass = 1.0
density = 1.0
rest_density = 1.0
radius = 0.4


# -Neighbours_Setting-
h = 1.0
h_2 = h * h
h_6 = h * h * h * h * h * h
h_9 = h * h * h * h * h * h * h * h * h
max_neighbour = 4000

# -Grid_Setting-
grid_size = 1
grid_rows = world[0] // grid_size
grid_cols = world[1] // grid_size
grid_layers = world[2] // grid_size
max_particle_in_grid = 4000

# -Boundary Epsilon-
b_epsilon = 0.01

# -POLY6_KERNEL-
poly6_Coe = 315.0 / (64 * math.pi)

# -SPIKY_KERNEL-
spiky_Coe = -45.0 / math.pi

# -LAMBDAS-
lambda_epsilon = 100.0

# -S_CORR-
S_Corr_delta_q = 0.3
S_Corr_k = 0.0001

# -Confinement/ XSPH Viscosity-
xsph_c = 0.01
vorti_epsilon = 0.01

# -Gradient Approx. delta difference-
g_del = 0.01

# -----FIELDS-----
position = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
last_position = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
velocity = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
lambdas = ti.field(dtype=ti.f32, shape=num_particles)
delta_qs = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
vorticity = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)

# Wacky check table for grid
num_particle_in_grid = ti.field(dtype=ti.i32, shape=(grid_rows, grid_cols, grid_layers))
table_grid = ti.field(dtype=ti.i32, shape=(grid_rows, grid_cols, grid_layers, max_particle_in_grid))

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
    result = ti.Vector([0.0, 0.0, 0.0])
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
        v[0] = lower + ti.random() * b_epsilon
    elif upper <= v[0]:
        v[0] = upper - ti.random() * b_epsilon
    if v[1] <= lower:
        v[1] = lower + ti.random() * b_epsilon
    elif upper <= v[1]:
        v[1] = upper - ti.random() * b_epsilon
    if v[2] <= lower:
        v[2] = lower + ti.random() * b_epsilon
    elif upper <= v[2]:
        v[2] = upper - ti.random() * b_epsilon
    return v


@ti.func
def get_grid(cord):
    new_cord = boundary_condition(cord)
    g_x = int(new_cord[0] // grid_size)
    g_y = int(new_cord[1] // grid_size)
    g_z = int(new_cord[2] // grid_size)
    return g_x, g_y, g_z

# --------------------KERNELS--------------------
# avoid nested for loop of position O(n^2)!

@ti.kernel
def pbf_prep():
    # ---save position---
    for p in position:
        last_position[p] = position[p]


@ti.kernel
def pbf_apply_force(ad: float, ws: float):
    # ---apply gravity/forces. Update velocity---
    gravity = ti.Vector([0.0, 0.0, -9.8])
    ad_force = ti.Vector([5.0, 0.0, 0.0])
    ws_force = ti.Vector([0.0, 5.0, 0.0])
    for i in velocity:
        velocity[i] += dt * (gravity + ad * ad_force + ws * ws_force + vorticity[i])
        # ---predict position---
        position[i] += dt * velocity[i]


@ti.kernel
def pbf_neighbour_search():
    # ---clean tables---
    for I in ti.grouped(num_particle_in_grid):
        num_particle_in_grid[I] = 0
    for I in ti.grouped(table_grid):
        table_grid[I] = -1
    for i in num_nb:
        num_nb[i] = 0
    for i, j in table_nb:
        table_nb[i, j] = -1
    for i in vorticity:
        vorticity[i] = ti.Vector([0.0, 0.0, 0.0])
    # ---update grid---
    for p in position:
        pos = position[p]
        p_grid = get_grid(pos)
        g_index = ti.atomic_add(num_particle_in_grid[p_grid[0], p_grid[1], p_grid[2]], 1)
        # ---ERROR CHECK---
        if g_index >= max_particle_in_grid:
            print("Grid overflows.")
        table_grid[p_grid[0], p_grid[1], p_grid[2], g_index] = p
    # ---update neighbour---
    for p in position:
        pos = position[p]
        p_grid = get_grid(pos)
        # nb_grid = neighbour_gird(p_grid)
        for off_x in ti.static(range(-1, 2)):
            for off_y in ti.static(range(-1, 2)):
                for off_z in ti.static(range(-1, 2)):
                    if 0 <= p_grid[0] + off_x <= grid_cols:
                        if 0 <= p_grid[1] + off_y <= grid_rows:
                            if 0 <= p_grid[2] + off_z <= grid_layers:
                                nb = (p_grid[0] + off_x, p_grid[1] + off_y, p_grid[2] + off_z)
                                for i in range(num_particle_in_grid[nb[0], nb[1], nb[2]]):
                                    new_nb = table_grid[nb[0], nb[1], nb[2], i]
                                    n_index = ti.atomic_add(num_nb[p], 1)
                                    # ---ERROR CHECK---
                                    if n_index >= max_neighbour:
                                        print("Neighbour overflows.")
                                    table_nb[p, n_index] = new_nb


@ti.kernel
def pbf_solve():
    # ---Calculate lambdas---
    for p in position:
        pos = position[p]
        lower_sum = 0.0
        p_i = 0.0
        spiky_i = ti.Vector([0.0, 0.0, 0.0])
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
        delta_q = ti.Vector([0.0, 0.0, 0.0])
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
    # ---Update Velocity---
    for v in velocity:
        velocity[v] = (position[v] - last_position[v]) / dt
    # ---Confinement/ XSPH Viscosity---
    # ---Using wacky gradient approximation for omega---
    for p in position:
        pos = position[p]
        xsph_sum = ti.Vector([0.0, 0.0, 0.0])
        omega_sum = ti.Vector([0.0, 0.0, 0.0])
        # -For Gradient Approx.-
        dx_sum = ti.Vector([0.0, 0.0, 0.0])
        dy_sum = ti.Vector([0.0, 0.0, 0.0])
        dz_sum = ti.Vector([0.0, 0.0, 0.0])
        n_dx_sum = ti.Vector([0.0, 0.0, 0.0])
        n_dy_sum = ti.Vector([0.0, 0.0, 0.0])
        n_dz_sum = ti.Vector([0.0, 0.0, 0.0])
        dx = ti.Vector([g_del, 0.0, 0.0])
        dy = ti.Vector([0.0, g_del, 0.0])
        dz = ti.Vector([0.0, 0.0, g_del])
        for i in range(num_nb[p]):
            nb_index = table_nb[p, i]
            nb_pos = position[nb_index]
            v_ij = velocity[nb_index] - velocity[p]
            dist = pos - nb_pos
            # ---Vorticity---
            omega_sum += v_ij.cross(spiky(dist))
            # -Gradient Approx.-
            dx_sum += v_ij.cross(spiky(dist + dx))
            dy_sum += v_ij.cross(spiky(dist + dy))
            dz_sum += v_ij.cross(spiky(dist + dz))
            n_dx_sum += v_ij.cross(spiky(dist - dx))
            n_dy_sum += v_ij.cross(spiky(dist - dy))
            n_dz_sum += v_ij.cross(spiky(dist - dz))
            # ---Viscosity---
            poly = poly6(dist)
            xsph_sum += poly * v_ij
        # ---Vorticity---
        n_x = (dx_sum.norm() - n_dx_sum.norm()) / (2 * g_del)
        n_y = (dy_sum.norm() - n_dy_sum.norm()) / (2 * g_del)
        n_z = (dz_sum.norm() - n_dz_sum.norm()) / (2 * g_del)
        n = ti.Vector([n_x, n_y, n_z])
        big_n = n.normalized()
        if not omega_sum.norm() == 0.0:
            vorticity[p] = vorti_epsilon * big_n.cross(omega_sum)
        # ---Viscosity---
        xsph_sum *= xsph_c
        velocity[p] += xsph_sum

def pbf(ad, ws):
    pbf_prep()
    pbf_apply_force(ad, ws)
    pbf_neighbour_search()
    for i in range(solve_iteration):
        pbf_solve()
    pbf_update()


@ti.kernel
def init():
    for i in position:
        pos_x = 2 + 0.8 * (i % 20)
        pos_y = 2 + 0.8 * ((i % 400) // 20)
        pos_z = 1 + 0.8 * (i // 400)
        position[i] = ti.Vector([pos_x, pos_y, pos_z])
        vorticity[i] = ti.Vector([0.0, 0.0, 0.0])


def render(gui):
    gui.clear(background_color)
    render_position = position.to_numpy()
    render_position /= boundary
    gui.circles(render_position, radius=visual_radius, color=particle_color)
    gui.show()


def main():
    init()
    prefix = "./3d_ply/a.ply"
    gui = ti.GUI('PBF3D', res)
    frame_count = 0
    while gui.running:
        # ---Control Waves---
        ad = 0.0
        ws = 0.0
        gui.get_event()
        if gui.is_pressed('a'):
            ad = -1.0
        elif gui.is_pressed('d'):
            ad = 1.0
        if gui.is_pressed('w'):
            ws = -1.0
        elif gui.is_pressed('s'):
            ws = 1.0
        pbf(ad, ws)
        # ---Record 3D result---
        if frame_count > -1:
            np_pos = np.reshape(position.to_numpy(), (num_particles, 3))
            writer = ti.tools.PLYWriter(num_vertices=num_particles)
            writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
            writer.export_frame(frame_count, prefix)
        gui.clear(background_color)
        gui.show()
        # render(gui)
        # ---Frame Control---
        if frame_count % 100 == 0:
            print("Frame:".format(frame_count))
        frame_count += 1
    return 0


if __name__ == '__main__':
    main()

    arr = v.to_numpy()
    magnitudes = np.linalg.norm(arr, axis=1)  # Compute magnitudes of vectors
    norm = Normalize(vmin=np.min(magnitudes), vmax=np.max(magnitudes))
    heatmap_rgb = plt.cm.coolwarm(norm(magnitudes))[:, :3]  # Use plasma colormap for heatmap
    per_vertex_color.from_numpy(heatmap_rgb)