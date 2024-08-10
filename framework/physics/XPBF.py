import csv
import taichi as ti
import numpy as np
from ..physics import collision_constraints_x, collision_constraints_v#, solve_pressure_constraints_x
from ..collision.lbvh_cell import LBVH_CELL

@ti.data_oriented
class Solver:
    def __init__(self,
                 particle_dy,
                 particle_st,
                 radius,
                 g,
                 dt):

        self.particle_dy = particle_dy
        self.particle_st = particle_st
        self.g = g
        self.dt = dt
        # self.radius = radius
        self.damping = 0.001
        self.padding = 0.05

        self.enable_velocity_update = False
        self.enable_collision_handling = False
        self.enable_move_obstacle = False
        self.export_mesh = False

        self.num_particles = self.particle_dy.num_particles

        self.particle_rad = radius
        # self.cell_size = 1.5 * self.particle_rad
        # self.cell_size_recpr = 1.0 / self.cell_size

        self.kernel_radius = 1.1
        self.spiky_grad_factor = -45.0 / ti.math.pi
        self.poly6_factor = 315.0 / 64.0 / ti.math.pi

        # self.bd = np.array([20.0, 20.0, 20.0])
        self.bd_max = ti.math.vec3(10.0)
        self.bd_min = -self.bd_max

        # self.boundary = (self.bd[0], self.bd[1], self.bd[2])
        self.grid_size = (64, 64, 64)

        # self.bd = np.floor(self.bd / self.cell_size).astype(int) + 1
        # self.grid_size = (self.bd[0], self.bd[1], self.bd[2])
        self.cell_size = (self.bd_max - self.bd_min)[0] / self.grid_size[0]
        # self.cell_size_recpr = 1.0 / self.cell_size
        print(self.cell_size)

        self.x_p = self.particle_dy.x
        self.y_p = self.particle_dy.y
        self.v_p = self.particle_dy.v
        self.m_inv_p = self.particle_dy.m_inv

        self.cell_cache_size = 100
        self.nb_cache_size = 50

        self.grid_num_particles = ti.field(int)
        self.particles2grid = ti.field(int)

        grid_snode = ti.root.dense(ti.ijk, (4, 4, 4)).dense(ti.ijk, (4, 4, 4)).dense(ti.ijk, (4, 4, 4))
        grid_snode.place(self.grid_num_particles)
        grid_snode.dense(ti.l,  self.cell_cache_size).place(self.particles2grid)
        # print(self.grid2particles.shape)
        self.particle_num_neighbors = ti.field(int)
        self.particle_neighbors = ti.field(int)
        self.ld = ti.field(float)
        self.dx = ti.Vector.field(3, float)

        nb_node = ti.root.dense(ti.i, self.num_particles)
        nb_node.place(self.particle_num_neighbors)
        nb_node.dense(ti.j, self.nb_cache_size).place(self.particle_neighbors)
        ti.root.dense(ti.i, self.num_particles).place(self.ld, self.dx)

        # print(ld.shape)

        self.aabb_x0 = ti.Vector.field(n=3, dtype=float, shape=8)
        self.aabb_index0 = ti.field(dtype=int, shape=24)
        self.init_grid(self.bd_min, self.bd_max)

        origin = - ti.math.vec3(10.0)
        # self.test(ti.math.vec3(0.), origin=origin, cell_size=float(self.cell_size))
    @ti.kernel
    def test(self, y: ti.math.vec3, origin: ti.math.vec3, cell_size: float):
        test = (y - origin) / cell_size
        print(self.particles2grid[ti.math.ivec3(0), 0])
        print(ti.cast(test, int))
        # print(ti.cast(test, int))


    @ti.kernel
    def init_grid(self, bd_min: ti.math.vec3, bd_max: ti.math.vec3):

        aabb_min = bd_min
        aabb_max = bd_max

        self.aabb_x0[0] = ti.math.vec3(aabb_max[0], aabb_max[1], aabb_max[2])
        self.aabb_x0[1] = ti.math.vec3(aabb_min[0], aabb_max[1], aabb_max[2])
        self.aabb_x0[2] = ti.math.vec3(aabb_min[0], aabb_max[1], aabb_min[2])
        self.aabb_x0[3] = ti.math.vec3(aabb_max[0], aabb_max[1], aabb_min[2])

        self.aabb_x0[4] = ti.math.vec3(aabb_max[0], aabb_min[1], aabb_max[2])
        self.aabb_x0[5] = ti.math.vec3(aabb_min[0], aabb_min[1], aabb_max[2])
        self.aabb_x0[6] = ti.math.vec3(aabb_min[0], aabb_min[1], aabb_min[2])
        self.aabb_x0[7] = ti.math.vec3(aabb_max[0], aabb_min[1], aabb_min[2])

        self.aabb_index0[0] = 0
        self.aabb_index0[1] = 1
        self.aabb_index0[2] = 1
        self.aabb_index0[3] = 2
        self.aabb_index0[4] = 2
        self.aabb_index0[5] = 3
        self.aabb_index0[6] = 3
        self.aabb_index0[7] = 0
        self.aabb_index0[8] = 4
        self.aabb_index0[9] = 5
        self.aabb_index0[10] = 5
        self.aabb_index0[11] = 6
        self.aabb_index0[12] = 6
        self.aabb_index0[13] = 7
        self.aabb_index0[14] = 7
        self.aabb_index0[15] = 4
        self.aabb_index0[16] = 0
        self.aabb_index0[17] = 4
        self.aabb_index0[18] = 1
        self.aabb_index0[19] = 5
        self.aabb_index0[20] = 2
        self.aabb_index0[21] = 6
        self.aabb_index0[22] = 3
        self.aabb_index0[23] = 7


    def reset(self):
        self.particle_dy.reset()
        if self.particle_st != None:
            self.particle_st.reset()

    @ti.func
    def confine_boundary(self, p):
        boundary_min = self.bd_min + self.particle_rad
        boundary_max = self.bd_max - self.particle_rad

        for i in ti.static(range(3)):
            if p[i] <= boundary_min[i]:
                p[i] = boundary_min[i] + 1e-4 * ti.random()
            elif boundary_max[i] <= p[i]:
                p[i] = boundary_max[i] - 1e-4 * ti.random()

        return p

    @ti.kernel
    def compute_y(self, dt: float):
        # for v in self.mesh_dy.verts:
        #     v.y = v.x + v.fixed * v.v * dt + self.g * dt * dt
        for i in self.v_p:
            self.v_p[i] = self.v_p[i] + self.g * dt
            self.y_p[i] = self.x_p[i] + self.v_p[i] * dt
            self.y_p[i] = self.confine_boundary(self.y_p[i])

    @ti.kernel
    def update_state(self, damping: float, dt: float):

        # for v in self.mesh_dy.verts:
        #     # if v.id != 0:
        #     v.x += dt * v.v
        for i in self.x_p:
            new_x = self.confine_boundary(self.y_p[i])
            self.v_p[i] = (1.0 - damping) * (new_x - self.x_p[i]) / dt
            self.x_p[i] = new_x


    @ti.kernel
    def compute_velocity(self, damping: ti.f32, dt: ti.f32):
        # for v in self.mesh_dy.verts:
        #     v.v = (1.0 - damping) * v.fixed * (v.y - v.x) / dt
        pass

    @ti.func
    def is_in_grid(self, c):
        # @c: Vector(i32)
        return 0 <= c[0] and c[0] < self.grid_size[0] and 0 <= c[1] and c[1] < self.grid_size[1] and 0 <= c[2] and c[2] < self.grid_size[2]
    @ti.kernel
    def search_neighbours(self):

        for I in ti.grouped(self.grid_num_particles):
            self.grid_num_particles[I] = 0

        for I in ti.grouped(self.particle_neighbors):
            self.particle_neighbors[I] = -1

        for pi in self.x_p:
            cell_id = self.pos_to_cell_id(self.x_p[pi])
            # for i in ti.static(range(3)):
            #     if cell_id[i] < 0 or cell_id[i] >= self.grid_size[0]: print("fuck")
            counter = ti.atomic_add(self.grid_num_particles[cell_id[0], cell_id[1], cell_id[2]], 1)
            if counter < self.cell_cache_size:
                self.particles2grid[cell_id, counter] = pi

        for pi in self.x_p:
            pos_i = self.x_p[pi]
            cell_id = self.pos_to_cell_id(pos_i)
            nb_i = 0
            for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                cell_to_check = cell_id + offs
                if self.is_in_grid(cell_to_check):
                    for j in range(self.grid_num_particles[cell_to_check]):
                        pj = self.particles2grid[cell_to_check, j]
                        if nb_i < self.nb_cache_size and pj != pi and (pos_i - self.x_p[pj]).norm() < 1.2 * self.particle_rad:
                            self.particle_neighbors[pi, nb_i] = pj
                            nb_i += 1
            self.particle_num_neighbors[pi] = nb_i


            # else:
            #     print("fuck")


    def init_variables(self):
        #
        # self.mesh_dy.verts.dx.fill(0.0)
        # self.mesh_dy.verts.nc.fill(0.0)

        self.dx_p.fill(0.0)
        self.nc_p.fill(0.0)


    # def solve_constraints_jacobi_x(self, dt):

        # self.init_variables()
        # ti.deactivate_all_snodes()
        # self.solve_pressure_constraints_x()
        # self.update_dx()

    def solve_constraints_v(self):
        self.mesh_dy.verts.dv.fill(0.0)
        self.mesh_dy.verts.nc.fill(0.0)

        if self.enable_collision_handling:
            self.solve_collision_constraints_v(self.mu)

        self.update_dv()
    @ti.kernel
    def update_dx(self):
        # for v in self.mesh_dy.verts:
        #     if v.nc > 0:
        #         v.y += v.fixed * (v.dx / v.nc)

        for pi in self.y_p:
            # if self.nc_p[pi] > 0:
            self.y_p[pi] = self.y_p[pi] + self.dx_p[pi]
                # self.y_p[pi] = self.y_p[pi] + self.dx_p[pi]


    @ti.kernel
    def set_fixed_vertices(self, fixed_vertices: ti.template()):
        for v in self.mesh_dy.verts:
            if fixed_vertices[v.id] >= 1:
                v.fixed = 0.0
            else:
                v.fixed = 1.0

    @ti.kernel
    def update_dv(self):
        for v in self.mesh_dy.verts:
            if v.nc > 0.0:
                v.dv = v.dv / v.nc

            if v.m_inv > 0.0:
                v.v += v.fixed * v.dv

    @ti.func
    def pos_to_cell_id(self, y: ti.math.vec3) -> ti.math.ivec3:
        test = (y - self.bd_min) / self.cell_size
        return ti.cast(test, int)
    @ti.func
    def flatten_cell_id(self,cid_3D):
        cx = cid_3D[0]
        cy = cid_3D[1]
        cz = cid_3D[2]
        cnx = self.grid_size[0]
        cny = self.grid_size[1]
        cnz = self.grid_size[2]

        return cnx * cny * cz + cnx * cy + cx

    @ti.func
    def spiky_gradient(self,r, h):
        result = ti.Vector([0.0, 0.0, 0.0])
        r_len = r.norm()
        if 0 < r_len and r_len < h:
            x = (h - r_len) / (h * h * h)
            g_factor = self.spiky_grad_factor * x * x
            result = r * g_factor / r_len
        return result

    @ti.func
    def poly6_value(self,s, h):
        result = 0.0
        if 0 < s and s < h:
            x = (h * h - s * s) / (h * h * h)
            result = self.poly6_factor * x * x * x

        return result
    @ti.func
    def check_grid_in(self, gid):

        check_min = (gid[0] <0 )or (gid[1]<0) or (gid[2]<0)
        check_max = (gid[0]>self.grid_size[0]-1) or (gid[1]>self.grid_size[1]-1) or (gid[2]>self.grid_size[2]-1)

        return check_max or check_min


    @ti.kernel
    def solve_pressure_constraints_x(self):

        # self.num_particle_neighbours.fill(0)

        for p_i in self.y_p:
            pos_i = self.y_p[p_i]
            grad_i = ti.math.vec3(0.0)
            sum_gradient_sqr = 0.0
            density_constraint = 0.0

            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                if p_j < 0:
                    break
                pos_ji = pos_i - self.y_p[p_j]
                grad_j = self.spiky_gradient(pos_ji, self.particle_rad)
                grad_i += grad_j
                sum_gradient_sqr += grad_j.dot(grad_j)
                # Eq(2)
                density_constraint += self.poly6_value(pos_ji.norm(), self.particle_rad)

            # Eq(1)
            density_constraint = density_constraint - 1.0

            sum_gradient_sqr += grad_i.dot(grad_i)
            self.ld[p_i] = (-density_constraint) / (sum_gradient_sqr + 1e2)

        for p_i in self.y_p:
            pos_i = self.y_p[p_i]
            lambda_i = self.ld[p_i]

            pos_delta_i = ti.math.vec3(0.0)
            for j in range(self.particle_num_neighbors[p_i]):
                p_j = self.particle_neighbors[p_i, j]
                if p_j < 0:
                    break
                lambda_j = self.ld[p_j]
                pos_ji = pos_i - self.y_p[p_j]
                # scorr_ij = compute_scorr(pos_ji)
                pos_delta_i += (lambda_i + lambda_j) * self.spiky_gradient(pos_ji, self.particle_rad)

            # pos_delta_i /= rho0
            self.dx[p_i] = pos_delta_i
        # apply position deltas
        for p_i in self.y_p:
            self.y_p[p_i] += self.dx[p_i]

        # for vi in ti.ndrange(self.num_particles):
        #     for j in range(self.num_particle_neighbours[vi]):
        #         vj = self.particle_neighbours[vi, j]
        #         nabla_C_ji = self.particle_neighbours_gradients[vi, j]
        #         # self.dx_p[vj] += self.lambda_dens[vi] * nabla_C_ji
        #         self.dx_p[vj] += (self.lambda_dens[vi] + self.lambda_dens[vj]) * nabla_C_ji
                # self.nc_p[vj] += 1

            # self.lambda_dens[vi] = -self.c_dens[vi] / self.schur_p[vi]

        # for vi in ti.ndrange(self.num_particles):
            # if self.c_dens[vi] > 0.0:
            #     for j in range(self.num_particle_neighbours[vi]):
            #         vj = self.particle_neighbours[vi, j]
            #
            #         nabla_C_ji = self.particle_neighbours_gradients[vi, j]
            #
            #         # self.dx_p[vj] += self.lambda_dens[vi] * nabla_C_ji
            #         self.dx_p[vj] += (self.lambda_dens[vi]) * nabla_C_ji
            #         self.nc_p[vj] += 1


    def forward(self, n_substeps):

        # self.load_sewing_pairs()
        dt_sub = self.dt / n_substeps
        self.search_neighbours()

        for _ in range(n_substeps):

            self.compute_y(dt_sub)
            self.solve_pressure_constraints_x()

            # if self.enable_velocity_update:
            #     self.solve_constraints_v()

            self.update_state(self.damping, dt_sub)
            # self.compute_velocity(damping=self.damping, dt=dt_sub)
