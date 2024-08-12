import csv
import taichi as ti
import numpy as np
from ..physics import collision_constraints_x, collision_constraints_v, solve_pressure_constraints_x
from ..collision.lbvh_cell import LBVH_CELL

@ti.data_oriented
class Solver:
    def __init__(self,
                 particle,
                 g,
                 dt):

        self.particle = particle
        self.g = g
        self.dt = dt

        self.damping = 0.001
        self.padding = 0.05

        self.enable_velocity_update = False
        self.export_mesh = False

        self.num_particles = self.particle.num_particles
        self.num_particles_dy = self.particle.num_dynamic
        self.kernel_radius = 1.1

        self.corr_deltaQ_coeff = 0.3
        self.corrK=0.1


        self.spiky_grad_factor = -45.0 / ti.math.pi
        self.poly6_factor = 315.0 / 64.0 / ti.math.pi

        self.bd_max = ti.math.vec3(40.0)
        self.bd_min = -self.bd_max

        self.grid_size = (64, 64, 64)


        self.cell_size = (self.bd_max - self.bd_min)[0] / self.grid_size[0]
        self.particle_rad = 0.2 * self.cell_size
        self.x_p = self.particle.x
        self.y_p = self.particle.y
        self.v_p = self.particle.v
        self.m_inv_p = self.particle.m_inv

        self.cell_cache_size = 500
        self.nb_cache_size = 1000


        self.grid_num_particles = ti.field(int)
        self.particles2grid = ti.field(int)

        grid_snode = ti.root.dense(ti.ijk, (4, 4, 4)).dense(ti.ijk, (4, 4, 4)).dense(ti.ijk, (4, 4, 4))
        grid_snode.place(self.grid_num_particles)
        grid_snode.dense(ti.l,  self.cell_cache_size).place(self.particles2grid)

        self.particle_num_neighbors = ti.field(int)
        self.particle_neighbors = ti.field(int)
        self.ld = ti.field(float)
        self.dx = ti.Vector.field(3, float)
        self.nc = ti.field(float)

        nb_node = ti.root.dense(ti.i, self.num_particles)
        nb_node.place(self.particle_num_neighbors)
        nb_node.dense(ti.j, self.nb_cache_size).place(self.particle_neighbors)
        ti.root.dense(ti.i, self.num_particles).place(self.ld, self.dx, self.nc)

        self.aabb_x0 = ti.Vector.field(n=3, dtype=float, shape=8)
        self.aabb_index0 = ti.field(dtype=int, shape=24)
        self.init_grid(self.bd_min, self.bd_max)


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
        self.particle.reset()

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

        ti.block_local(self.m_inv_p, self.v_p, self.x_p, self.y_p)
        for i in self.y_p:
            if self.m_inv_p[i] > 0.0:
                self.v_p[i] = self.v_p[i] + self.g * dt
                self.y_p[i] = self.x_p[i] + self.v_p[i] * dt
            else:
                self.y_p[i] = self.x_p[i]

            self.y_p[i] = self.confine_boundary(self.y_p[i])


    @ti.kernel
    def update_state(self, damping: float, dt: float):

        ti.block_local(self.m_inv_p, self.v_p, self.x_p, self.y_p)
        for i in range(self.num_particles_dy):
            new_x = self.confine_boundary(self.y_p[i])
            self.v_p[i] = (1.0 - damping) * (new_x - self.x_p[i]) / dt
            self.x_p[i] = new_x

    @ti.kernel
    def search_neighbours(self):

        self.grid_num_particles.fill(0)
        self.particles2grid.fill(-1)

        self.particle_num_neighbors.fill(0)
        self.particle_neighbors.fill(-1)

        ti.block_local(self.x_p, self.grid_num_particles, self.particles2grid)
        for pi in self.x_p:
            cell_id = self.pos_to_cell_id(self.x_p[pi])
            counter = ti.atomic_add(self.grid_num_particles[cell_id], 1)
            if counter < self.cell_cache_size:
                self.particles2grid[cell_id, counter] = pi
            else :
                print("cache over",cell_id)

    @ti.func
    def pos_to_cell_id(self, y: ti.math.vec3) -> ti.math.ivec3:
        test = (y - self.bd_min) / self.cell_size
        return ti.cast(test, int)

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
    def compute_scorr(self,pos_ji):
        # Eq (13)
        h = self.kernel_radius
        x = self.poly6_value(pos_ji.norm(), h) / self.poly6_value(self.corr_deltaQ_coeff * h, h)
        x = x * x
        x = x * x
        return -self.corrK * x


    @ti.func
    def is_in_grid(self, c):
        # @c: Vector(i32)

        is_in_grid = True
        for i in ti.static(range(3)):
            is_in_grid = is_in_grid and (0 <= c[i] < self.grid_size[i])

        return is_in_grid

    @ti.kernel
    def solve_pressure_constraints_x_col(self):

        self.dx.fill(0.0)
        self.nc.fill(0.0)
        k = 1e8

        ti.block_local(self.dx, self.nc, self.grid_num_particles, self.particles2grid)
        for pi in range(self.num_particles_dy):
            pos_i = self.y_p[pi]
            inv_mi = self.m_inv_p[pi]
            cell_id = self.pos_to_cell_id(pos_i)
            nb_i = 0
            for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                cell_to_check = cell_id + offs
                if self.is_in_grid(cell_to_check):
                    for j in range(self.grid_num_particles[cell_to_check]):
                        pj = self.particles2grid[cell_to_check, j]
                        if pi != pj:
                            pos_j = self.y_p[pj]
                            inv_mj = self.m_inv_p[pj]
                            xji = pos_j - pos_i
                            c = xji.norm() - 2 * self.particle_rad
                            if c < 0:
                                schur = (inv_mi + inv_mj)
                                ld = k * c / (k * schur + 1.0)
                                grad_c = ti.math.normalize(xji)

                                self.dx[pi] += inv_mi * ld * grad_c
                                self.dx[pj] -= inv_mj * ld * grad_c

                                self.nc[pi] += 1
                                self.nc[pj] += 1

            # for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
            #     cell_to_check = cell_id + offs
            #     if self.is_in_grid(cell_to_check):
            #         for j in range(self.grid_num_particles_st[cell_to_check]):
            #             pj = self.particles2grid_st[cell_to_check, j]
            #             # if pi != pj:
            #             pos_j = self.x_st[pj]
            #             xji = pos_j - pos_i
            #
            #             if xji.norm() < 2 * self.particle_rad:
            #                 dir = ti.math.normalize(xji)
            #                 # proj0 = pos_i - self.particle_rad * dir
            #                 self.dx[pi] += ((xji.norm() - 2 * self.particle_rad) * dir)
            #                 self.nc[pi] += 1

        for pi in self.y_p:
            if self.nc[pi] > 0:
                self.y_p[pi] += (self.dx[pi] / self.nc[pi])
        # self.num_particle_neighbours.fill(0)
        # h = 1.1 * self.particle_rad
        # for p_i in self.y_p:
        #     pos_i = self.y_p[p_i]
        #     grad_i = ti.math.vec3(0.0)
        #     sum_gradient_sqr = 0.0
        #     density_constraint = 0.0
        #
        #     for j in range(self.particle_num_neighbors[p_i]):
        #         p_j = self.particle_neighbors[p_i, j]
        #         # if p_j < 0:
        #         #     break
        #         pos_ji = pos_i - self.y_p[p_j]
        #         grad_j = self.spiky_gradient(pos_ji, h)
        #         grad_i += grad_j
        #         sum_gradient_sqr += grad_j.dot(grad_j)
        #         # Eq(2)
        #         density_constraint += self.poly6_value(pos_ji.norm(), h)
        #
        #     # Eq(1)
        #     density_constraint = density_constraint - 1.0
        #
        #     sum_gradient_sqr += grad_i.dot(grad_i)
        #     self.ld[p_i] = (-density_constraint) / (sum_gradient_sqr + 1e2)
        #
        # for p_i in self.y_p:
        #     pos_i = self.y_p[p_i]
        #     lambda_i = self.ld[p_i]
        #
        #     pos_delta_i = ti.math.vec3(0.0)
        #     for j in range(self.particle_num_neighbors[p_i]):
        #         p_j = self.particle_neighbors[p_i, j]
        #         if p_j < 0:
        #             break
        #         lambda_j = self.ld[p_j]
        #         pos_ji = pos_i - self.y_p[p_j]
        #         # scorr_ij = compute_scorr(pos_ji)
        #         pos_delta_i += (lambda_i + lambda_j) * self.spiky_gradient(pos_ji, h)
        #
        #     # pos_delta_i /= rho0
        #     self.dx[p_i] = pos_delta_i
        # # apply position deltas
        # for p_i in self.y_p:
        #     self.y_p[p_i] += self.dx[p_i]

    @ti.kernel
    def solve_pressure_constraints_x(self):

        self.dx.fill(0.0)
        self.nc.fill(0.0)

        ti.block_local(self.dx, self.nc, self.grid_num_particles, self.particles2grid)
        for pi in self.y_p:
            C_dens = -1.0
            schur = 1e2
            C_i_nabla_i = ti.Vector([0.0,0.0,0.0])
            pos_i = self.y_p[pi]
            cell_id = self.pos_to_cell_id(pos_i)

            for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                cell_to_check = cell_id + offs
                if self.is_in_grid(cell_to_check):
                    for j in range(self.grid_num_particles[cell_to_check]):
                        pj = self.particles2grid[cell_to_check, j]

                        if pi == pj :
                            continue
                        pos_j = self.y_p[pj]
                        xji = pos_j - pos_i

                        if xji.norm() < self.kernel_radius * 1.03 + 1e-4 :

                            count = ti.atomic_add(self.particle_num_neighbors[pi],1)
                            if(count >= self.nb_cache_size) :
                                # print("neighbor over!!",count)
                                self.particle_num_neighbors[pi] = self.nb_cache_size
                            else:
                                self.particle_neighbors[pi,count] = pj
                                C_dens += self.poly6_value(xji.norm(),self.kernel_radius)
                                C_i_nabla_j = -self.spiky_gradient(xji,self.kernel_radius)
                                C_i_nabla_i -=C_i_nabla_j
                                schur += C_i_nabla_j.dot(C_i_nabla_j)
            schur += C_i_nabla_i.dot(C_i_nabla_i)

            # self.ld[pi] = -C_dens / schur if(C_dens > 0.0) else 0.0
            self.ld[pi] = -C_dens / schur

        for pi in self.y_p:
            pos_i = self.y_p[pi]
            ld_i = self.ld[pi]
            delta_x_agg = ti.Vector([0.0,0.0,0.0])
            for j in range(self.particle_num_neighbors[pi]):
                pj = self.particle_neighbors[pi, j]
                if(pj < 0) :
                    break

                ld_j = self.ld[pj]
                pos_j = self.y_p[pj]
                xij = pos_i - pos_j
                scorr = self.compute_scorr(xij)

                # delta_x_agg += (ld_i + ld_j ) * self.spiky_gradient(xij,self.kernel_radius)
                delta_x_agg += (ld_i + ld_j + scorr) * self.spiky_gradient(xij,self.kernel_radius)

            self.dx[pi] = delta_x_agg / (self.particle_num_neighbors[pi] + 1e-4)
            self.y_p[pi]+=self.dx[pi]

    def forward(self, n_substeps):

        dt_sub = self.dt / n_substeps
        self.search_neighbours()
        for _ in range(n_substeps):

            self.compute_y(dt_sub)
            self.solve_pressure_constraints_x()
            self.update_state(self.damping, dt_sub)

