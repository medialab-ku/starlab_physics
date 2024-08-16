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
        self.YM = 1e5
        self.PR = 0.2
        self.damping = 0.001
        self.padding = 0.05

        self.enable_velocity_update = False
        self.export_mesh = False

        self.num_particles = self.particle.num_particles
        self.num_particles_dy = self.particle.num_dynamic

        self.solver_type = 0

        self.corr_deltaQ_coeff = 0.3
        self.corrK = 0.1
        self.spiky_grad_factor = -45.0 / ti.math.pi
        self.poly6_factor = 315.0 / 64.0 / ti.math.pi

        self.bd_max = ti.math.vec3(40.0)
        self.bd_min = -self.bd_max

        self.grid_size = (64, 64, 64)
        self.cell_size = (self.bd_max - self.bd_min)[0] / self.grid_size[0]
        print(self.cell_size)

        self.particle_rad = 0.2 * self.cell_size
        self.kernel_radius = 1.5
        # self.particle_rad = 0.2 * self.kernel_radius
        self.x = self.particle.x
        self.x0 = self.particle.x0
        self.y = self.particle.y
        self.v = self.particle.v
        self.V0 = self.particle.V0
        self.F = self.particle.F
        self.L = self.particle.L
        self.m_inv_p = self.particle.m_inv
        self.rho0 = self.particle.rho0
        self.rho0.fill(1.0)

        self.cell_cache_size = 500
        self.nb_cache_size = 1000

        self.grid_num_particles = ti.field(int)
        self.particles2grid = ti.field(int)

        grid_snode = ti.root.dense(ti.ijk, (4, 4, 4)).dense(ti.ijk, (4, 4, 4)).dense(ti.ijk, (4, 4, 4))
        grid_snode.place(self.grid_num_particles)
        grid_snode.dense(ti.l,  self.cell_cache_size).place(self.particles2grid)

        self.particle_num_neighbors = ti.field(int)
        self.particle_neighbors = ti.field(int)

        self.particle_num_neighbors_rest = ti.field(int)
        self.particle_neighbors_rest = ti.field(int)

        self.ld = ti.field(float)
        self.dx = ti.Vector.field(3, float)
        self.nc = ti.field(float)

        nb_node = ti.root.dense(ti.i, self.num_particles)
        nb_node.place(self.particle_num_neighbors, self.particle_num_neighbors_rest)
        nb_node.dense(ti.j, self.nb_cache_size).place(self.particle_neighbors, self.particle_neighbors_rest)
        ti.root.dense(ti.i, self.num_particles).place(self.ld, self.dx, self.nc)

        self.aabb_x0 = ti.Vector.field(n=3, dtype=float, shape=8)
        self.aabb_index0 = ti.field(dtype=int, shape=24)
        self.init_grid(self.bd_min, self.bd_max)
        self.reset()
        # self.test_kernel()

    # @ti.func
    # def outer_product(self, u: ti.math.vec3, v: ti.math.vec3, uv: ti.math.vec3):
    #
    #     uvT = ti.math.mat3(0.0)
    #     for i in ti.grouped(ti.ndrange((0, 3), (0, 3))):
    #         uvT[i] = u[i[0]] * v[i[1]]
    #
    #     return uvT
    # @ti.kernel
    # def test_kernel(self):
    #
    #     u = ti.math.vec3(1.0)
    #     v = ti.math.vec3(2.0)
    #     mat = self.outer_product(u, v)
    #     print(mat)


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
        self.search_neighbours_rest()
        self.init_V0_and_L()

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
    def search_neighbours(self):

        self.grid_num_particles.fill(0)
        self.particle_num_neighbors.fill(0)

        ti.block_local(self.x, self.grid_num_particles, self.particles2grid)
        for pi in self.x:
            cell_id = self.pos_to_cell_id(self.x[pi])
            counter = ti.atomic_add(self.grid_num_particles[cell_id], 1)

            if counter < self.cell_cache_size:
                self.particles2grid[cell_id, counter] = pi

    @ti.kernel
    def search_neighbours_rest(self):

        self.grid_num_particles.fill(0)
        self.particle_num_neighbors_rest.fill(0)

        ti.block_local(self.x, self.grid_num_particles, self.particles2grid)
        for pi in range(self.num_particles_dy):
            cell_id = self.pos_to_cell_id(self.x[pi])
            counter = ti.atomic_add(self.grid_num_particles[cell_id], 1)
            if counter < self.cell_cache_size:
                self.particles2grid[cell_id, counter] = pi

        ti.block_local(self.grid_num_particles, self.particles2grid)
        for pi in range(self.num_particles_dy):
            pos_i = self.x[pi]
            cell_id = self.pos_to_cell_id(pos_i)
            for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                cell_to_check = cell_id + offs
                if self.is_in_grid(cell_to_check):
                    for j in range(self.grid_num_particles[cell_to_check]):
                        pj = self.particles2grid[cell_to_check, j]
                        if pi == pj:
                            continue
                        pos_j = self.x[pj]
                        xji = pos_j - pos_i
                        if xji.norm() < self.kernel_radius * 1.03 + 1e-4:
                            count = ti.atomic_add(self.particle_num_neighbors_rest[pi], 1)
                            if count >= self.nb_cache_size:
                                # print("neighbor over!!",count)
                                self.particle_num_neighbors_rest[pi] = self.nb_cache_size
                            else:
                                self.particle_neighbors[pi, count] = pj

        for pi in range(self.num_particles_dy):
            if self.particle_num_neighbors_rest[pi] < 3:
                print("fuck")

    @ti.kernel
    def init_V0_and_L(self):

        #init rest volume V0
        ti.block_local(self.L, self.x, self.rho0)
        for i in range(self.num_particles_dy):
            pos_i = self.x[i]
            Vi0 = 0.0
            for nj in range(self.particle_num_neighbors_rest[i]):
                j = self.particle_neighbors[i, nj]
                xji0 = self.x[j] - pos_i
                Vi0 += self.poly6_value(xji0.norm(), self.kernel_radius)
            self.V0[i] = (1.0 / Vi0)

        # init correction, L
        ti.block_local(self.L, self.x)
        for i in range(self.num_particles_dy):
            x0i = self.x[i]
            Li = ti.math.mat3(0.0)
            for nj in range(self.particle_num_neighbors_rest[i]):
                j = self.particle_neighbors[i, nj]
                xji0 = self.x[j] - x0i
                wji0 = self.poly6_value(xji0.norm(), self.kernel_radius)
                Li += self.V0[j] * wji0 * self.outer_product(xji0, xji0)
                # for I in ti.grouped(ti.ndrange((0, 3), (0, 3))):
                #     Li[I] += self.V0[j] * wji0 * xji0[I[0]] * xji0[I[1]]

            self.L[i] = ti.math.inverse(Li)

    @ti.func
    def pos_to_cell_id(self, y: ti.math.vec3) -> ti.math.ivec3:
        test = (y - self.bd_min) / self.cell_size
        return ti.cast(test, int)

    @ti.func
    def outer_product(self, u: ti.math.vec3, v: ti.math.vec3) -> ti.math.mat3:

        uvT = ti.math.mat3(0.0)
        for I in ti.grouped(ti.ndrange((0, 3), (0, 3))):
            uvT[I] += u[I[0]] * v[I[1]]

        return uvT


    @ti.func
    def spiky_gradient(self, r, h):

        result = ti.math.vec3(0.0)

        r_len = r.norm()
        if 0 < r_len and r_len < h:
            x = (h - r_len) / (h * h * h)
            g_factor = self.spiky_grad_factor * x * x
            result = r * g_factor / r_len

        return result

    @ti.func
    def poly6_value(self, s, h):
        result = 0.0
        if 0 < s and s < h:
            x = (h * h - s * s) / (h * h * h)
            result = self.poly6_factor * x * x * x

        return result

    @ti.func
    def compute_scorr(self, pos_ji):
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
            pos_i = self.y[pi]
            inv_mi = self.m_inv_p[pi]
            cell_id = self.pos_to_cell_id(pos_i)
            nb_i = 0
            for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                cell_to_check = cell_id + offs
                if self.is_in_grid(cell_to_check):
                    for j in range(self.grid_num_particles[cell_to_check]):
                        pj = self.particles2grid[cell_to_check, j]
                        if pi != pj:
                            pos_j = self.y[pj]
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

        for pi in self.y:
            if self.nc[pi] > 0:
                self.y[pi] += (self.dx[pi] / self.nc[pi])

    @ti.func
    def ssvd(self, F):
        U, sig, V = ti.svd(F)
        if U.determinant() < 0:
            for i in ti.static(range(3)): U[i, 2] *= -1
            sig[2, 2] = -sig[2, 2]
        if V.determinant() < 0:
            for i in ti.static(range(3)): V[i, 2] *= -1
            sig[2, 2] = -sig[2, 2]
        return U, sig, V

    @ti.kernel
    def solve_xpbd_fem_stretch_constraints_x(self, compliance_str: float):

        self.dx.fill(0.0)
        self.nc.fill(0.0)

        ti.block_local(self.y, self.L)
        for i in range(self.num_particles_dy):
            x0i, yi = self.x0[i], self.y[i]
            Dsi = ti.math.mat3(0.0)
            for nj in range(self.particle_num_neighbors_rest[i]):
                j = self.particle_neighbors[i, nj]
                yji = self.y[j] - yi
                x0ji = self.x0[j] - x0i
                V0wji0 = self.V0[j] * self.poly6_value(x0ji.norm(), self.kernel_radius)
                Dsi += V0wji0 * self.outer_product(yji, x0ji)

            F = Dsi @ self.L[i]
            U, sig, V = self.ssvd(F)
            R = U @ V.transpose()

            for nj in range(self.particle_num_neighbors_rest[i]):
                j = self.particle_neighbors[i, nj]
                x0ji = self.x0[j] - x0i
                yji = self.y[j] - yi
                dxji = R @ x0ji - yji

                self.dx[j] += (compliance_str / (compliance_str + 1.)) * dxji
                self.dx[i] -= (compliance_str / (compliance_str + 1.)) * dxji

                self.nc[i] += 1.0
                self.nc[j] += 1.0

        for i in range(self.num_particles_dy):
            self.y[i] += (self.dx[i] / self.nc[i])

    @ti.kernel
    def solve_pd_fem_stretch_x(self, compliance_str: float):

        self.dx.fill(0.0)
        self.nc.fill(1.0)

        ti.block_local(self.y, self.L)
        for i in range(self.num_particles_dy):
            x0i, yi = self.x0[i], self.y[i]
            Dsi = ti.math.mat3(0.0)
            wi = compliance_str * self.V0[i]
            for nj in range(self.particle_num_neighbors_rest[i]):
                j = self.particle_neighbors[i, nj]
                yji = self.y[j] - yi
                x0ji = self.x0[j] - x0i
                V0wji0 = self.V0[j] * self.poly6_value(x0ji.norm(), self.kernel_radius)
                Dsi += V0wji0 * self.outer_product(yji, x0ji)

            F = Dsi @ self.L[i]
            U, sig, V = self.ssvd(F)
            R = U @ V.transpose()

            for nj in range(self.particle_num_neighbors_rest[i]):
                j = self.particle_neighbors[i, nj]
                x0ji = self.x0[j] - x0i
                yji = self.y[j] - yi
                dxji = R @ x0ji - yji

                self.dx[j] += wi * dxji
                self.dx[i] -= wi * dxji

                self.nc[i] += wi
                self.nc[j] += wi


        for i in range(self.num_particles_dy):
            self.y[i] += (self.dx[i] / self.nc[i])

    @ti.kernel
    def solve_constraints_pressure_x(self):

        ti.block_local(self.grid_num_particles, self.particles2grid, self.rho0)
        for pi in range(self.num_particles):
            rho_i = self.poly6_value(0.0, self.kernel_radius)
            schur = 1e2
            nabla_Cii = ti.math.vec3(0.0)
            pos_i = self.y[pi]
            cell_id = self.pos_to_cell_id(pos_i)
            for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                cell_to_check = cell_id + offs
                if self.is_in_grid(cell_to_check):
                    for j in range(self.grid_num_particles[cell_to_check]):
                        pj = self.particles2grid[cell_to_check, j]

                        if pi == pj:
                            continue
                        pos_j = self.y[pj]
                        xji = pos_j - pos_i
                        if xji.norm() < self.kernel_radius * 1.03 + 1e-4:
                            count = ti.atomic_add(self.particle_num_neighbors[pi],1)
                            if count >= self.nb_cache_size:
                                # print("neighbor over!!",count)
                                self.particle_num_neighbors[pi] = self.nb_cache_size
                            else:
                                self.particle_neighbors[pi, count] = pj

                                rho_i += self.poly6_value(xji.norm(), self.kernel_radius)
                                nabla_Cij = -self.spiky_gradient(xji, self.kernel_radius)
                                nabla_Cii -= nabla_Cij
                                schur += ti.math.dot(nabla_Cij, nabla_Cij)

            schur += nabla_Cii.dot(nabla_Cii)
            C_dens = ti.max(rho_i - 1.0, 0.0)
            # if C_dens < 0.0:
            #     C_dens = 0.0
            k = 1e8
            self.ld[pi] = -k * C_dens / (k * schur + 1.0)

        for pi in range(self.num_particles_dy):
            pos_i = self.y[pi]
            ld_i = self.ld[pi]
            dx_i = ti.math.vec3(0.0)

            for j in range(self.particle_num_neighbors[pi]):
                pj = self.particle_neighbors[pi, j]
                ld_j = self.ld[pj]
                pos_j = self.y[pj]
                xij = pos_i - pos_j
                scorr = self.compute_scorr(xij)
                dx_i += (ld_i + ld_j) * self.spiky_gradient(xij, self.kernel_radius)

            self.y[pi] += dx_i

    @ti.kernel
    def solve_xpbd_collision_constraints_x(self, distance_threshold: float):

        self.dx.fill(0.0)
        self.nc.fill(0.0)

        ti.block_local(self.grid_num_particles, self.particles2grid, self.dx, self.nc)
        for pi in range(self.num_particles):
            pos_i = self.y[pi]
            cell_id = self.pos_to_cell_id(pos_i)
            for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                cell_to_check = cell_id + offs
                if self.is_in_grid(cell_to_check):
                    for j in range(self.grid_num_particles[cell_to_check]):
                        pj = self.particles2grid[cell_to_check, j]
                        if pi == pj:
                            continue
                        pos_j = self.y[pj]
                        xji = pos_j - pos_i
                        if xji.norm() < distance_threshold:
                            pji = distance_threshold * ti.math.normalize(xji)
                            dxji = pji - xji
                            self.dx[pi] -= dxji
                            self.dx[pj] += dxji

                            self.nc[pi] += 1
                            self.nc[pj] += 1

        ti.block_local(self.y, self.dx, self.nc)
        for pi in range(self.num_particles):
            if self.nc[pi] > 0:
                self.y[pi] += self.dx[pi] / self.nc[pi]


    @ti.kernel
    def compute_y(self, dt: float):

        ti.block_local(self.m_inv_p, self.v, self.x, self.y)
        for i in self.y:
            if self.m_inv_p[i] > 0.0:
                self.v[i] = self.v[i] + self.g * dt
                self.y[i] = self.x[i] + self.v[i] * dt
            else:
                self.y[i] = self.x[i]

            self.y[i] = self.confine_boundary(self.y[i])

    @ti.kernel
    def update_state(self, damping: float, dt: float):

        ti.block_local(self.m_inv_p, self.v, self.x, self.y)
        for i in range(self.num_particles_dy):
            new_x = self.confine_boundary(self.y[i])
            self.v[i] = (1.0 - damping) * (new_x - self.x[i]) / dt
            self.x[i] = new_x

    def forward(self, n_substeps):

        dt_sub = self.dt / n_substeps

        self.search_neighbours()
        for _ in range(n_substeps):
            self.compute_y(dt_sub)
            if self.solver_type == 0:
                dtSq = dt_sub ** 2

                mu = self.YM / 2.0 * (1.0 + self.PR)
                ld = (self.YM * self.PR) / ((1.0 + self.PR) * (1.0 - 2.0 * self.PR))

                compliance_str = 2.0 * mu * dtSq
                self.solve_xpbd_fem_stretch_constraints_x(compliance_str)

                self.solve_xpbd_collision_constraints_x(2.5 * self.particle_rad)
                # self.solve_constraints_pressure_x()
            elif self.solver_type == 1:
                dtSq = dt_sub ** 2

                mu = self.YM / 2.0 * (1.0 + self.PR)
                ld = (self.YM * self.PR) / ((1.0 + self.PR) * (1.0 - 2.0 * self.PR))

                compliance_str = 2.0 * mu * dtSq
                self.solve_pd_fem_stretch_x(compliance_str)

                # compliance_vol = mu * dtSq
                # self.solve_constraints_pressure_x()

            self.update_state(self.damping, dt_sub)

