import csv
from telnetlib import SEND_URL

import taichi as ti
import numpy as np
from ..physics import collision_constraints_x, collision_constraints_v, solve_pressure_constraints_x
from ..collision.lbvh_cell import LBVH_CELL

@ti.data_oriented
class Solver:
    def __init__(self,
                 particle,
                 g,
                 dt,
                 sh):

        self.particle = particle
        self.g = g
        self.dt = dt
        self.sh = sh

        self.YM = 1e2
        self.PR = 0.2
        self.ZE = 1.0
        self.damping = 0.0
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
        self.kernel_radius = 1.2
        # self.particle_rad = 0.2 * self.kernel_radius
        self.x = self.particle.x
        self.dx = self.particle.dx
        self.nc = self.particle.nc
        self.x0 = self.particle.x0
        self.y = self.particle.y
        self.v = self.particle.v
        self.V0 = self.particle.V0
        self.F = self.particle.F
        self.L = self.particle.L
        self.m_inv = self.particle.m_inv
        self.m = self.particle.m
        self.ld = self.particle.ld
        self.is_fixed = self.particle.is_fixed
        self.rho0 = self.particle.rho0
        self.rho0.fill(1.0)
        self.m.fill(1.0)
        self.reset()


    def reset(self):
        self.particle.reset()
        self.sh.search_neighbours(self.particle.x0)
        self.init_V0_and_L(self.solver_type)

    @ti.func
    def confine_boundary(self, p):
        boundary_min = self.sh.bbox_min + self.particle_rad
        boundary_max = self.sh.bbox_max - self.particle_rad

        for i in ti.static(range(3)):
            if p[i] <= boundary_min[i]:
                p[i] = boundary_min[i] + 1e-4 * ti.random()
            elif boundary_max[i] <= p[i]:
                p[i] = boundary_max[i] - 1e-4 * ti.random()

        return p

    # @ti.kernel
    # def search_neighbours(self):
    #
    #     self.grid_num_particles.fill(0)
    #     self.particle_num_neighbors.fill(0)
    #
    #     ti.block_local(self.x, self.grid_num_particles, self.particles2grid)
    #     for pi in self.x:
    #         cell_id = self.pos_to_cell_id(self.x[pi])
    #         counter = ti.atomic_add(self.grid_num_particles[cell_id], 1)
    #
    #         if counter < self.cell_cache_size:
    #             self.particles2grid[cell_id, counter] = pi
    #
    # @ti.kernel
    # def search_neighbours_rest(self):
    #
    #     self.grid_num_particles.fill(0)
    #     self.particle_num_neighbors_rest.fill(0)
    #
    #     ti.block_local(self.x, self.grid_num_particles, self.particles2grid)
    #     for pi in range(self.num_particles_dy):
    #         cell_id = self.pos_to_cell_id(self.x[pi])
    #         counter = ti.atomic_add(self.grid_num_particles[cell_id], 1)
    #         if counter < self.cell_cache_size:
    #             self.particles2grid[cell_id, counter] = pi
    #
    #     ti.block_local(self.grid_num_particles, self.particles2grid)
    #     for pi in range(self.num_particles_dy):
    #         pos_i = self.x[pi]
    #         cell_id = self.pos_to_cell_id(pos_i)
    #         for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
    #             cell_to_check = cell_id + offs
    #             if self.is_in_grid(cell_to_check):
    #                 for j in range(self.grid_num_particles[cell_to_check]):
    #                     pj = self.particles2grid[cell_to_check, j]
    #                     if pi == pj:
    #                         continue
    #                     pos_j = self.x[pj]
    #                     xji = pos_j - pos_i
    #                     if xji.norm() < self.kernel_radius * 1.03 + 1e-4:
    #                         count = ti.atomic_add(self.particle_num_neighbors_rest[pi], 1)
    #                         if count < self.nb_cache_size:
    #                         #     # print("neighbor over!!",count)
    #                         #     self.particle_neighbors_rest[pi] = self.nb_cache_size
    #                         # else:
    #                             self.particle_neighbors_rest[pi, count] = pj
    #
    #     for pi in range(self.num_particles_dy):
    #         if self.particle_num_neighbors_rest[pi] < 3:
    #             print("you need more neighbours...")
    #
    @ti.kernel
    def init_V0_and_L(self, solver_type: int):

        self.particle.num_particle_neighbours_rest.fill(0)
        for pi in range(self.num_particles_dy):
            pos_i = self.x0[pi]
            cell_id = self.sh.pos_to_cell_id(pos_i)
            for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                cell_to_check = cell_id + offs
                if self.sh.is_in_grid(cell_to_check):
                    for j in range(self.sh.num_particles_in_cell[cell_to_check]):
                        pj = self.sh.particle_ids_in_cell[cell_to_check, j]
                        if pi == pj:
                            continue
                        xji0 = self.x0[pj] - pos_i
                        n = self.particle.num_particle_neighbours_rest[pi]
                        if n < self.particle.particle_cache_size and xji0.norm() < self.kernel_radius:
                            self.particle.particle_neighbours_ids_rest[pi, n] = pj
                            self.particle.num_particle_neighbours_rest[pi] += 1

            # if self.particle.num_particle_neighbours_rest[pi] < 3:
            #     print("fuck")

        for i in range(self.num_particles_dy):
            x0i = self.x0[i]
            Li = ti.math.mat3(0.0)
            for nj in range(self.particle.num_particle_neighbours_rest[i]):
                j = self.particle.particle_neighbours_ids_rest[i, nj]
                xji0 = self.x0[j] - x0i
                wji0 = self.poly6_value(xji0.norm(), self.kernel_radius)
                Li += wji0 * self.outer_product(xji0, xji0)
            self.L[i] = ti.math.inverse(Li)
        #
        # for i in range(self.num_particles):
        #     wii = self.poly6_value(0.0, self.kernel_radius)
        #     self.rho0[i] = wii
    #
    #
    # @ti.func
    # def pos_to_cell_id(self, y: ti.math.vec3) -> ti.math.ivec3:
    #     test = (y - self.bd_min) / self.cell_size
    #     return ti.cast(test, int)
    #
    @ti.func
    def outer_product(self, u: ti.math.vec3, v: ti.math.vec3) -> ti.math.mat3:

        uvT = ti.math.mat3(0.0)
        for I in ti.grouped(ti.ndrange((0, 3), (0, 3))):
            uvT[I] += u[I[0]] * v[I[1]]

        return uvT


    @ti.func
    def spiky_gradient(self, r, h) -> ti.math.vec3:

        result = ti.math.vec3(0.0)
        r_len = r.norm()
        if 0 < r_len and r_len <= h:
            x = (h - r_len) / (h * h * h)
            g_factor = self.spiky_grad_factor * x * x
            result = r * g_factor / r_len

        return result

    @ti.func
    def cubic_spline_kernel(self, r_norm, h):

        rh = r_norm / h
        w = 0.0
        alpha = ti.math.pi * (h ** 3)
        if 0.0 <= rh < 1.0:
            w = 1.0 - 1.5 * (rh ** 2) + 0.75 * (rh ** 3)
        elif 1.0 <= rh < 2.0:
            w = 0.25 * (2.0 - rh) ** 3
        elif rh >= 2.0:
            w = 0.0

        w /= alpha
        return w

    @ti.func
    def cubic_spline_kernel_gradient(self, r, h):

        rh = r.norm() / h
        w = ti.math.vec3(0.0)

        alpha = ti.math.pi * (h ** 3)
        dir = ti.math.normalize(r) / h
        if 0.0 <= rh < 1.0:
            w = (-3.0 * rh + 2.25 * (rh ** 2)) * dir
        elif 1.0 <= rh < 2.0:
            w = -0.75 * (2.0 - rh) ** 2 * dir
        elif rh >= 2.0:
            w = ti.math.vec3(0.0)

        w /= alpha

        return w

    @ti.func
    def poly6_value(self, s, h):
        result = 0.0
        if 0 <= s and s <= h:
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

    #
    # @ti.func
    # def is_in_grid(self, c):
    #     # @c: Vector(i32)
    #     is_in_grid = True
    #     for i in ti.static(range(3)):
    #         is_in_grid = is_in_grid and (0 <= c[i] < self.grid_size[i])
    #
    #     return is_in_grid

    @ti.kernel
    def solve_pressure_constraints_x_col(self):

        # self.dx.fill(0.0)
        # self.nc.fill(0.0)

        k = 1e8
        # ti.block_local(self.dx, self.nc, self.grid_num_particles, self.particles2grid)
        for pi in range(self.num_particles_dy):
            pos_i = self.y[pi]
            inv_mi = self.m_inv[pi]
            cell_id = self.sh.pos_to_cell_id(pos_i)
            nb_i = 0
            for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                cell_to_check = cell_id + offs
                if self.is_in_grid(cell_to_check):
                    for j in range(self.sh.num_particles_in_cell[cell_to_check]):
                        pj = self.sh.particle_ids_in_cell[cell_to_check, j]
                        if pi != pj:
                            pos_j = self.y[pj]
                            inv_mj = self.m_inv[pj]
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

        # for pi in self.y:
        #     if self.nc[pi] > 0:
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
        self.nc.fill(1.0)

        ti.block_local(self.y, self.L)
        for i in range(self.num_particles_dy):
            x0i, yi = self.x0[i], self.y[i]
            Dsi = ti.math.mat3(0.0)
            for nj in range(self.particle.num_particle_neighbours_rest[i]):
                j = self.particle.particle_neighbours_ids_rest[i, nj]
                yji = self.y[j] - yi
                x0ji = self.x0[j] - x0i
                wji0 = self.poly6_value(x0ji.norm(), self.kernel_radius)
                Dsi += wji0 * self.outer_product(yji, x0ji)

            F = Dsi @ self.L[i]
            U, sig, V = self.ssvd(F)
            R = U @ V.transpose()

            for nj in range(self.particle.num_particle_neighbours_rest[i]):
                j = self.particle.particle_neighbours_ids_rest[i, nj]
                x0ji = self.x0[j] - x0i
                yji = self.y[j] - yi
                dxji = R @ x0ji - yji

                self.dx[j] += compliance_str * dxji
                self.dx[i] -= compliance_str * dxji

                self.nc[i] += compliance_str
                self.nc[j] += compliance_str

        for i in range(self.num_particles_dy):
            self.y[i] += (self.dx[i] / self.nc[i])

    @ti.kernel
    def solve_pd_fem_stretch_x(self, compliance_str: float, solver_type: int, alpha: float):

        # print(solver_type)
        self.dx.fill(0.0)
        self.nc.fill(1.0)

        ti.block_local(self.y, self.L)
        for i in range(self.num_particles_dy):
            x0i, yi = self.x0[i], self.y[i]
            Dsi = ti.math.mat3(0.0)
            wi = compliance_str * self.V0[i]
            for nj in range(self.particle_num_neighbors_rest[i]):
                j = self.particle_neighbors_rest[i, nj]
                yji = self.y[j] - yi
                x0ji = self.x0[j] - x0i
                # if solver_type == 1:
                V0wji0 = self.V0[j] * self.cubic_spline_kernel(x0ji.norm(), self.kernel_radius)
                Dsi += V0wji0 * self.outer_product(yji, x0ji)

                # elif solver_type == 2:
                    # V0wji0 = self.V0[j] * self.cubic_spline_kernel(x0ji.norm(), self.kernel_radius)
                    # Dsi += self.V0[j] * self.outer_product(yji, self.spiky_gradient(x0ji, self.kernel_radius))

            F = Dsi @ self.L[i]
            U, sig, V = self.ssvd(F)
            R = U @ V.transpose()

            for nj in range(self.particle_num_neighbors_rest[i]):
                j = self.particle_neighbors_rest[i, nj]
                x0ji = self.x0[j] - x0i
                yji = self.y[j] - yi
                dxji = R @ x0ji - yji

                self.dx[j] += (wi / self.m[j]) * dxji
                self.dx[i] -= (wi / self.m[i]) * dxji

                self.nc[i] += (wi / self.m[i])
                self.nc[j] += (wi / self.m[j])

            # wi = alpha * self.V0[i]
            # for nj in range(self.particle_num_neighbors_rest[i]):
            #     j = self.particle_neighbors[i, nj]
            #     x0ji = self.x0[j] - x0i
            #     yji = self.y[j] - yi
            #     dxji = F @ x0ji - yji
            #
            #     self.dx[j] += wi * dxji
            #     self.dx[i] -= wi * dxji
            #
            #     self.nc[i] += wi
            #     self.nc[j] += wi


        for i in range(self.num_particles_dy):
            self.y[i] += (self.dx[i] / self.nc[i])

    @ti.kernel
    def solve_constraints_pressure_x(self):

        self.dx.fill(0.0)
        # self.nc.fill(0.0)
        self.particle.num_particle_neighbours.fill(0)
        kernel_radius = 2.5 * self.particle_rad
        # ti.block_local(self.grid_num_particles, self.particles2grid, self.dx, self.nc)
        for pi in range(self.num_particles):
            pos_i = self.y[pi]
            cell_id = self.sh.pos_to_cell_id(pos_i)
            C = 0.0
            schur = 0.0

            nabla_Cii = ti.math.vec3(0.0)
            for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                cell_to_check = cell_id + offs
                if self.sh.is_in_grid(cell_to_check):
                    for j in range(self.sh.num_particles_in_cell[cell_to_check]):
                        pj = self.sh.particle_ids_in_cell[cell_to_check, j]
                        if pi == pj:
                            continue
                        pos_j = self.y[pj]
                        xji = pos_j - pos_i
                        nb_i = self.particle.num_particle_neighbours[pi]
                        if xji.norm() < kernel_radius and nb_i < self.particle.nb_cache_size:
                            C += self.m[pj] * self.poly6_value(xji.norm(), kernel_radius)
                            nabla_Cji = self.spiky_gradient(xji, kernel_radius)
                            nabla_Cii -= nabla_Cji
                            schur += nabla_Cji.dot(nabla_Cji)
                            self.particle.particle_neighbours_ids[pi, nb_i] = pj
                            self.particle.num_particle_neighbours[pi] += 1

            schur += nabla_Cii.dot(nabla_Cii)
            k = 1e8
            self.ld[pi] = -(k * C) / (k * schur + 1.0)

        for pi in range(self.num_particles):
            pos_i = self.y[pi]
            for j in range(self.particle.num_particle_neighbours[pi]):
                pj = self.particle.particle_neighbours_ids[pi, j]
                pos_j = self.y[pj]
                xji = pos_j - pos_i
                self.dx[pi] -= (self.ld[pi] + self.ld[pj]) * self.spiky_gradient(xji, kernel_radius)

            # self.dx[pi] += self.ld[pi] * nabla_Cii
            self.y[pi] += self.dx[pi]



    @ti.kernel
    def solve_xpbd_collision_constraints_x(self, distance_threshold: float):

        self.dx.fill(0.0)
        self.nc.fill(0.0)
        # self.m_inv.fill(1.0)
        # ti.block_local(self.grid_num_particles, self.particles2grid, self.dx, self.nc)
        for pi in range(self.num_particles):
            pos_i = self.y[pi]
            cell_id = self.sh.pos_to_cell_id(pos_i)
            C = 0.0
            schur = 0.0
            nabla_Cii = ti.math.vec3(0.0)
            for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                cell_to_check = cell_id + offs
                if self.sh.is_in_grid(cell_to_check):
                    for j in range(self.sh.num_particles_in_cell[cell_to_check]):
                        pj = self.sh.particle_ids_in_cell[cell_to_check, j]
                        if pi == pj:
                            continue
                        pos_j = self.y[pj]
                        xji = pos_j - pos_i
                        if xji.norm() < distance_threshold:
                            C = (xji.norm() - distance_threshold)
                            nabla_C = ti.math.normalize(xji)
                            schur = (self.is_fixed[pi] * self.m_inv[pi] + self.is_fixed[pj] * self.m_inv[pj])
                            k = 1e8
                            ld = -(k * C) / (k * schur + 1.0)

                            self.dx[pi] -= self.is_fixed[pi] * self.m_inv[pi] * ld * nabla_C
                            self.dx[pj] += self.is_fixed[pj] * self.m_inv[pj] * ld * nabla_C

                            self.nc[pi] += 1.0
                            self.nc[pj] += 1.0

        ti.block_local(self.y, self.dx, self.nc)
        for pi in range(self.num_particles):
            if self.nc[pi] > 0:
                self.y[pi] += self.dx[pi] / self.nc[pi]


    @ti.kernel
    def compute_y(self, dt: float):

        ti.block_local(self.m_inv, self.v, self.x, self.y)
        for i in self.y:
            if self.m_inv[i] > 0.0:
                self.v[i] = self.is_fixed[i] * self.v[i] + self.g * dt
                self.y[i] = self.x[i] + self.v[i] * dt
            else:
                self.y[i] = self.x[i]

            self.y[i] = self.confine_boundary(self.y[i])

    @ti.kernel
    def update_state(self, damping: float, dt: float):

        ti.block_local(self.m_inv, self.v, self.x, self.y)
        for i in range(self.num_particles_dy):
            new_x = self.confine_boundary(self.y[i])
            self.v[i] = self.is_fixed[i] * (1.0 - damping) * (new_x - self.x[i]) / dt
            self.x[i] += self.v[i] * dt

    @ti.kernel
    def randomize(self):

        for pi in range(self.num_particles_dy):
            x0 = ti.math.clamp(40 * ti.random(), -20., 20.)
            x1 = ti.math.clamp(40 * ti.random(), -20., 20.)
            x2 = ti.math.clamp(40 * ti.random(), -20., 20.)
            self.x[pi] = ti.math.vec3(x0, x1, x2)

    @ti.kernel
    def set_fixed_vertices(self, fixed_vertices: ti.template()):
        for pi in range(self.num_particles_dy):
            if fixed_vertices[pi] >= 1:
                self.is_fixed[pi] = 0.0
            else:
                self.is_fixed[pi] = 1.0

    def forward(self, n_substeps, n_iter):
        dt_sub = self.dt / n_substeps
        self.sh.search_neighbours(self.x)
        for _ in range(n_substeps):
            # self.sh.search_neighbours(self.x)
            self.compute_y(dt_sub)
            for _ in range(n_iter):
                # self.solve_xpbd_collision_constraints_x(2 * self.particle_rad)
                self.solve_constraints_pressure_x()

            for _ in range(n_iter):
                dtSq = dt_sub ** 2
                mu = self.YM / 2.0 * (1.0 + self.PR)
                compliance_str = 2.0 * mu * dtSq
                self.solve_xpbd_fem_stretch_constraints_x(compliance_str)
            #     self.solve_pd_fem_stretch_x(compliance_str, 1, self.ZE)

            self.update_state(self.damping, dt_sub)

