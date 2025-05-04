
import taichi as ti
from math_utils import *
from LBVH import LBVH
from sph_base import SPHBase
from PCG import PCG
from distance import *
import matplotlib.pyplot as plt
import numpy as np


class XSPHSolver(SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        # Pressure state function parameters(WCSPH)
        # self.exponent = 7.0
        # self.exponent = self.ps.cfg.get_cfg("exponent")
        #
        # self.stiffness = 50000.0
        # self.stiffness = self.ps.cfg.get_cfg("stiffness")
        
        self.surface_tension = 0.001
        self.dt[None] = self.ps.cfg.get_cfg("timeStepSize")
        self.PCG = PCG(self.ps.fluid_particle_num)

        self.LBVH = LBVH(self.ps.faces_st.shape[0] // 3)

        self.num_max_collision = 2 ** 20
        self.num_collision = ti.field(int, shape=())
        self.collision_info = ti.Vector.field(n=4, dtype=int, shape=self.num_max_collision)
        self.collision_bary = ti.Vector.field(n=4, dtype=float, shape=self.num_max_collision)
        self.collision_type = ti.field(int, shape=self.num_max_collision)
        self.collision_H = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.num_max_collision)

        self.num_max_candidate = 2 ** 20
        self.num_candidate = ti.field(int, shape=())
        self.candidate_info = ti.Vector.field(n=3, dtype=int, shape=self.num_max_collision)

        self.cache_size = 200
        self.num_collision_p = ti.field(int, shape=self.ps.fluid_particle_num)
        self.collision_idx_p = ti.field(dtype=int, shape=(self.ps.fluid_particle_num, self.cache_size))
        self.collision_H_p = ti.Matrix.field(n=3, m=3, dtype=float, shape=(self.ps.fluid_particle_num, self.cache_size))
        self.collision_grad_p = ti.Vector.field(n=3, dtype=float, shape=(self.ps.fluid_particle_num, self.cache_size))

    @ti.func
    def barrier_grad(self, d: float, dHat: float):
        k = (-ti.log(d / dHat))
        dkdx = -1.0 / d
        return (2 * k + dkdx * (d - dHat)) * (d - dHat)

    @ti.func
    def barrier_hess(self, d: float, dHat: float):
        k = (-ti.log(d / dHat))
        dkdx = -1.0 / d
        d2kdx2 = 1.0 / (d ** 2)
        return (d2kdx2 * (d - dHat) + 4 * dkdx) * (d - dHat) + 2 * k

    @ti.func
    def compute_densities_task(self, p_i, p_j, ret: ti.template()):
        x_i = self.ps.x[p_i]
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            x_j = self.ps.x[p_j]
            ret += self.ps.m_V[p_j] * self.cubic_kernel((x_i - x_j).norm())
        elif self.ps.material[p_j] == self.ps.material_solid:
            # Boundary neighbors
            ## Akinci2012
            x_j = self.ps.x[p_j]
            ret += self.ps.m_V[p_j] * self.cubic_kernel((x_i - x_j).norm())


    @ti.kernel
    def compute_densities(self, x: ti.template()):
        # for p_i in range(self.ps.particle_num[None]):
        a = self.ps
        h = self.ps.support_radius

        max_num = 0
        for p_i in ti.grouped(x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            self.ps.density[p_i] = self.ps.m_V[p_i] * self.cubic_kernel(0.0)
            # for p_i in ti.grouped(self.ps.x):
            self.num_collision_p[p_i] = 0
            center_cell = a.pos_to_index(x[p_i])
            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * a.dim)):
                grid_index = a.flatten_grid_index(center_cell + offset)
                for p_j in range(a.grid_particles_num[ti.max(0, grid_index - 1)], a.grid_particles_num[grid_index]):
                    if p_i[0] != p_j and (x[p_i] - x[p_j]).norm() < h:
                        x_i = x[p_i]
                        h = self.ps.support_radius
                        if self.ps.material[p_j] == self.ps.material_fluid:
                            x_j = x[p_j]
                            r = (x_i - x_j).norm()
                            self.ps.density[p_i] += self.ps.m_V[p_j] * self.cubic_kernel(r)
                            ni = ti.atomic_add(self.num_collision_p[p_i], 1)
                            self.collision_idx_p[p_i, ni] = p_j

            # ti.atomic_max(max_num, self.num_collision_p[p_i])
            self.ps.density[p_i] *= self.density_0
            if self.num_collision_p[p_i] > self.cache_size:
                print("warning")


        # print(max_num)

    @ti.func
    def compute_non_pressure_forces_task(self, p_i, p_j, ret: ti.template()):
        x_i = self.ps.x[p_i]
        
        ############## Surface Tension ###############
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            diameter2 = self.ps.particle_diameter * self.ps.particle_diameter
            x_j = self.ps.x[p_j]
            r = x_i - x_j
            r2 = r.dot(r)
            if r2 > diameter2:
                ret -= self.surface_tension / self.ps.m[p_i] * self.ps.m[p_j] * r * self.cubic_kernel(r.norm())
            else:
                ret -= self.surface_tension / self.ps.m[p_i] * self.ps.m[p_j] * r * self.cubic_kernel(ti.Vector([self.ps.particle_diameter, 0.0, 0.0]).norm())
            
        
        ############### Viscosoty Force ###############
        d = 2 * (self.ps.dim + 2)
        x_j = self.ps.x[p_j]
        # Compute the viscosity force contribution
        r = x_i - x_j
        v_xy = (self.ps.v[p_i] - self.ps.v[p_j]).dot(r)
        
        if self.ps.material[p_j] == self.ps.material_fluid:
            f_v = d * self.viscosity * (self.ps.m[p_j] / (self.ps.density[p_j])) * v_xy / (r.norm()**2 + 0.01 * self.ps.support_radius**2) * self.cubic_kernel_derivative(r)
            ret += f_v
        elif self.ps.material[p_j] == self.ps.material_solid:
            boundary_viscosity = 0.0
            # Boundary neighbors
            ## Akinci2012
            f_v = d * boundary_viscosity * (self.density_0 * self.ps.m_V[p_j] / (self.ps.density[p_i])) * v_xy / (
                r.norm()**2 + 0.01 * self.ps.support_radius**2) * self.cubic_kernel_derivative(r)
            ret += f_v
            if self.ps.is_dynamic_rigid_body(p_j):
                self.ps.acceleration[p_j] += -f_v * self.density_0 / self.ps.density[p_j]


    @ti.kernel
    def compute_non_pressure_forces(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_static_rigid_body(p_i):
                self.ps.acceleration[p_i].fill(0.0)
                continue
            ############## Body force ###############
            # Add body force
            d_v = ti.Vector(self.g)
            # d_v = ti.Vector([0.0, 0.0, 0.0])
            self.ps.acceleration[p_i] = d_v
            if self.ps.material[p_i] == self.ps.material_fluid:
                self.ps.for_all_neighbors(p_i, self.compute_non_pressure_forces_task, d_v)
                self.ps.acceleration[p_i] = d_v


    @ti.kernel
    def advect(self):
        # Symplectic Euler
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic[p_i]:
                self.ps.v[p_i] += self.dt[None] * self.ps.acceleration[p_i]
                self.ps.xOld[p_i] = self.ps.x[p_i] + self.dt[None] * self.ps.v[p_i]

    @ti.kernel
    def compute_xHat(self):
        for p_i in ti.grouped(self.ps.x):
            self.ps.xOld[p_i] = self.ps.x[p_i]
            if self.ps.is_dynamic[p_i]:
                self.ps.xHat[p_i] = self.ps.xOld[p_i] + self.dt[None] * self.ps.v[p_i]


    @ti.kernel
    def compute_velocity(self):
        for p_i in ti.grouped(self.ps.x):
            self.ps.v[p_i] = (self.ps.x[p_i] - self.ps.xOld[p_i]) / self.dt[None]

    @ti.kernel
    def compute_inertia(self):

        I_3x3 = ti.math.mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        density = 1e1
        for p_i in ti.grouped(self.ps.x):
            self.ps.grad[p_i]  = self.density_0 * self.ps.m_V[p_i] * (self.ps.x[p_i] - self.ps.xHat[p_i])
            self.ps.diagH[p_i] = self.density_0 * self.ps.m_V[p_i] * I_3x3

    @ti.func
    def compute_pressure_task(self, p_i, p_j, ret: ti.template()):

        I_3x3 = ti.math.mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        c = self.ps.density[p_i] / self.density_0 - 1.0

        x_i = self.ps.x[p_i]
        # mass_i = self.ps.m_V[p_i] * self.density_0
        h = self.ps.support_radius
        coef = 1e8
        k = coef * self.dt[None] * self.dt[None] * ((h ** 6))
        if self.ps.material[p_j] == self.ps.material_fluid:
            x_j = self.ps.x[p_j]
            r = (x_i - x_j).norm()
            if c >= 0:
                # print("test")
                dEdc = c ** 2
                d2Edc2 = 2 * c

                # dEdc = c
                # d2Edc2 = 1.0

                mass_j = self.ps.m_V[p_j] * self.density_0
                # if r < 1e-5:
                #     r = 1e-5
                #     print("test")


                # r += 1e-4
                n = (x_i - x_j) / r

                dWdr = self.spiky_kernel_derivative(r, h)
                dcdx = mass_j * dWdr * n / self.density_0
                d2Wdr2 = self.spiky_kernel_hessian(r, h)

                nnT = n.outer_product(n)
                alpha = abs(dWdr / r)
                beta = (d2Wdr2 - dWdr / r)
                # if beta < 0.0:
                #     print("test")
                d2cdx2 = (mass_j / self.density_0) * (alpha * I_3x3 + beta * nnT)
                self.ps.grad[p_i] += k * dEdc * dcdx
                self.ps.grad[p_j] -= k * dEdc * dcdx

                # if c >=0:
                self.ps.diagH[p_i] += k * (dEdc * d2cdx2 + d2Edc2 * dcdx.outer_product(dcdx))
                self.ps.diagH[p_j] += k * (dEdc * d2cdx2 + d2Edc2 * dcdx.outer_product(dcdx))
            # else:
            # dHat = 0.5 * self.ps.particle_radius
            # if r <= dHat:
            #     factor = 45.0 / (ti.math.pi * (dHat ** 6))
            #     # print(factor)
            #     mass_j = self.ps.m_V[p_j] * self.density_0
            #     Kappa = coef * self.dt[None] * self.dt[None] * (dHat ** 6)
            #     # Kappa = 1e3 * self.dt[None] * self.dt[None]
            #     dbdx = self.spiky_kernel_derivative(r, dHat)
            #     d2bdx2 = self.spiky_kernel_hessian(r, dHat)
            #     # dbdx = -factor * (r - dHat) ** 2
            #     # d2bdx2 = 2.0 * factor *  (r - dHat)
            #     # n = (x_i - x_j) / d
            #     # if d < 0.1 * dHat:
            #     #     d = 0.1 * dHat
            #     n = (x_i - x_j) / r
            #     nnT = n.outer_product(n)
            #     d2d_dx2 = (I_3x3 - nnT) / r
            #     test = (abs(d2bdx2) * nnT + abs(dbdx) * d2d_dx2)
            #     self.ps.diagH[p_i] += Kappa * test
            #     self.ps.diagH[p_j] += Kappa * test
            #     self.ps.grad[p_i] += Kappa * dbdx * n
            #     self.ps.grad[p_j] -= Kappa * dbdx * n

    @ti.func
    def compute_collision_particle_task(self, p_i, p_j, ret: ti.template()):

        I_3x3 = ti.math.mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        Kappa = 1e4 * self.dt[None] * self.dt[None]
        x_i = self.ps.x[p_i]
        x_j = self.ps.x[p_j]
        d = (x_i - x_j).norm()
        dHat = self.ps.particle_diameter
        if d <= dHat:
            dbdx = self.barrier_grad(d, dHat)
            d2bdx2 = self.barrier_hess(d, dHat)
            # n = (x_i - x_j) / d
            # if d < 0.1 * dHat:
            #     d = 0.1 * dHat

            n = (x_i - x_j) / d
            nnT = n.outer_product(n)
            d2d_dx2 = (I_3x3 - nnT) / d
            test = (abs(d2bdx2) * nnT + abs(dbdx) * d2d_dx2)
            self.ps.diagH[p_i] += Kappa * test
            self.ps.diagH[p_j] += Kappa * test
            self.ps.grad[p_i] += Kappa * dbdx * n
            self.ps.grad[p_j] -= Kappa * dbdx * n



    @ti.kernel
    def compute_pressure(self, k: float):
        h = 1.0 * self.ps.support_radius
        I_3x3 = ti.math.mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        c_max = 0.0
        for p_i in ti.grouped(self.ps.x):
            c = ti.max(self.ps.density[p_i] / self.density_0 - 1.0, 1e-9)
            for ni in range(self.num_collision_p[p_i]):
                p_j = self.collision_idx_p[p_i, ni]
                x_i = self.ps.x[p_i]
                # h = self.ps.support_radius
                if self.ps.material[p_j] == self.ps.material_fluid:
                    x_j = self.ps.x[p_j]
                    r = (x_i - x_j).norm()

                    dEdc = c
                    d2Edc2 = 1.0 if c >= 0 else 0

                    # dEdc = (c ** 2)
                    # d2Edc2 = 2 * c

                    # dEdc = 1.0
                    # d2Edc2 = 0

                    mass_j = self.ps.m_V[p_j] * self.density_0
                    if r < 1e-12:
                        r = 1e-12
                    #     print("test")

                    # r += 1e-4
                    n = (x_i - x_j) / r

                    dWdr = self.spiky_kernel_derivative(r, h)
                    dcdx = mass_j * dWdr * n / self.density_0
                    d2Wdr2 = self.spiky_kernel_hessian(r, h)

                    nnT = n.outer_product(n)
                    alpha = abs(dWdr / r)
                    beta = (d2Wdr2 - dWdr / r)
                    # if beta < 0.0:
                    #     print("test")
                    d2cdx2 = (mass_j / self.density_0) * (alpha * I_3x3 + beta * nnT)
                    self.ps.grad[p_i] += k * dEdc * dcdx
                    self.ps.grad[p_j] -= k * dEdc * dcdx

                    self.ps.diagH[p_i] += k * (dEdc * d2cdx2 + d2Edc2 * dcdx.outer_product(dcdx))
                    self.ps.diagH[p_j] += k * (dEdc * d2cdx2 + d2Edc2 * dcdx.outer_product(dcdx))

                    self.collision_grad_p[p_i, ni] = ti.sqrt(k * d2Edc2) * dcdx
                    self.collision_H_p[p_i, ni] = k * dEdc * d2cdx2


            c_max = max(c_max, c)

        print(c_max)

    @ti.kernel
    def compute_pressure_gn(self, k: float):

        h = 1.0 * self.ps.support_radius
        for p_i in ti.grouped(self.ps.x):
            # before = self.ps.x[p_i]

            density_i = self.ps.density[p_i]
            for ni in range(self.num_collision_p[p_i]):
                p_j = self.collision_idx_p[p_i, ni]
                m_j = self.ps.m_V[p_j] * self.density_0
                xji_old = self.ps.xOld[p_i] - self.ps.x[p_j]
                r_old = xji_old.norm()

                if r_old < 1e-12:
                    r_old = 1e-12

                n_old = xji_old / r_old
                xji = self.ps.x[p_i] - self.ps.x[p_j]
                density_i += m_j * self.spiky_kernel_derivative(r_old, h) * n_old.dot(xji - xji_old)


            c = ti.max(density_i / self.density_0 - 1.0, 1e-9)
            for ni in range(self.num_collision_p[p_i]):
                p_j = self.collision_idx_p[p_i, ni]
                m_j = self.ps.m_V[p_j] * self.density_0
                xji_old = self.ps.xOld[p_i] - self.ps.x[p_j]
                r_old = xji_old.norm()

                if r_old < 1e-12:
                    r_old = 1e-12

                n_old = xji_old / r_old
                dEdc = c
                d2Edc2 = 1.0 if c > 0 else 0.0

                dcdx = (m_j / self.density_0) * self.spiky_kernel_derivative(r_old, h) * n_old
                #
                # if ti.math.isnan(dcdx[0]):
                #     print("test")

                d2cdx2 = ti.math.mat3(0.0)

                self.ps.grad[p_i] += k * dEdc * dcdx
                self.ps.grad[p_j] -= k * dEdc * dcdx

                # if c >=0:
                self.ps.diagH[p_i] += k * (d2Edc2 * dcdx.outer_product(dcdx))
                self.ps.diagH[p_j] += k * (d2Edc2 * dcdx.outer_product(dcdx))

                self.collision_grad_p[p_i, ni] = ti.sqrt(k * d2Edc2) * dcdx
                self.collision_H_p[p_i, ni] = k * ti.math.mat3(0.0)



    @ti.kernel
    def compute_collision_static(self, pad: float):

        self.num_candidate[None] = 0
        for P in self.ps.x:
            xP = self.ps.x[P]

            _min0 = xP - ti.math.vec3(pad)
            _max0 = xP + ti.math.vec3(pad)
            self.LBVH.traverse_bvh_single_test(_min0, _max0, 0, P, self.candidate_info,  self.num_candidate)

        # print(self.num_candidate[None] )
        Kappa = 1e5 * self.dt[None] ** 2
        I_3x3 = ti.math.mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        dHat = 4 * self.ps.particle_radius
        self.num_collision[None] = 0

        for i in range(self.num_candidate[None]):

            info = self.candidate_info[i]
            P = info[0]
            j = info[1]

            xP = self.ps.x[P]

            T0, T1, T2 = self.ps.faces_st[3 * j + 0], self.ps.faces_st[3 * j + 1], self.ps.faces_st[3 * j + 2]
            xT0, xT1, xT2 = self.ps.x_st[T0], self.ps.x_st[T1], self.ps.x_st[T2]
            type = d_type_PT(xP, xT0, xT1, xT2)
                # print(type)
            bary = ti.math.vec3(0.0)
            if type == 0:
                bary[0] = 1.0

            elif type == 1:
                bary[1] = 1.0

            elif type == 2:
                bary[2] = 1.0

            elif type == 3:
                a = d_PE(xP, xT0, xT1)
                bary[0] = a[0]
                bary[1] = a[1]

            elif type == 4:
                a = d_PE(xP, xT1, xT2)
                bary[1] = a[0]
                bary[2] = a[1]

            elif type == 5:
                a = d_PE(xP, xT0, xT2)
                bary[0] = a[0]
                bary[2] = a[1]

            elif type == 6:
                bary = d_PT(xP, xT0, xT1, xT2)

            proj = bary[0] * xT0 + bary[1] * xT1 + bary[2] * xT2
            d = (xP - proj).norm()



            n = (xP - proj) / d
            if d <= dHat:

                # if d == 0:
                #     print("test")
                dbdx = self.barrier_grad(d, dHat)
                d2bdx2 = self.barrier_hess(d, dHat)

                nnT = n.outer_product(n)
                d2d_dx2 = (I_3x3 - nnT) / d
                test = (abs(d2bdx2) * nnT + abs(dbdx) * d2d_dx2)
                self.ps.diagH[P] += Kappa * test
                self.ps.grad[P] += Kappa * dbdx * n

                idx = ti.atomic_add(self.num_collision[None], 1)
                self.collision_H[idx] = Kappa * test
                self.collision_info[idx] = ti.math.ivec4([P, -1, -1, -1])
                self.collision_type[idx] = 0
                self.collision_bary[idx] = ti.math.vec4([1.0, 0.0, 0.0, 0.0])

    @ti.kernel
    def compute_search_dir(self):
        for p_i in ti.grouped(self.ps.x):
            dx = -self.ps.diagH[p_i].inverse() @ self.ps.grad[p_i]
            self.ps.dx[p_i] = dx

    @ti.kernel
    def inf_norm(self, p: ti.template()) -> float:

        val = 0.0
        for p_i in ti.grouped(p):
            tmp = p[p_i].norm()
            ti.atomic_max(val, tmp)

        return val

    @ti.kernel
    def update_x(self, alpha: float):

        for p_i in ti.grouped(self.ps.x):
            self.ps.x[p_i] += alpha * self.ps.dx[p_i]


    @ti.kernel
    def mat_free_Ax(self, Ax: ti.template(), x: ti.template()):

        for p_i in ti.grouped(self.ps.x):
            Ax[p_i] = self.density_0 * self.ps.m_V[p_i] * x[p_i]

        for i in range(self.num_collision[None]):
            bary = self.collision_bary[i]
            info = self.collision_info[i]
            H = self.collision_H[i]
            if self.collision_type[i] == 0:
                Ax[info[0]] += bary[0] * H @ x[info[0]]

        for p_i in ti.grouped(self.ps.x):
            value = 0.0

            # if self.ps.density[p_i] >= self.density_0:
            for j in range(self.num_collision_p[p_i]):
                p_j = self.collision_idx_p[p_i, j]
                xji = x[p_i] - x[p_j]

                value += self.collision_grad_p[p_i, j].dot(xji)

                Hji = self.collision_H_p[p_i, j]
                Ax[p_i] += Hji @ xji
                Ax[p_j] -= Hji @ xji

            # if ti.math.isnan(value):
            #     print("test")

            for j in range(self.num_collision_p[p_i]):
                p_j = self.collision_idx_p[p_i, j]
                grad = self.collision_grad_p[p_i, j]
                Ax[p_i] += value * grad
                Ax[p_j] -= value * grad

    def build_static_LBVH(self):
        pad = 0.5 * self.ps.particle_diameter
        self.LBVH.build(self.ps.x_st, self.ps.faces_st, pad=pad)


    @ti.kernel
    def compute_pressure_potential(self, k: float) -> ti.f32:

        value = 0.0
        for p_i in ti.grouped(self.ps.x):

            if self.ps.density[p_i] > self.density_0:
                value += (self.ps.density[p_i] / self.density_0 - 1.0) ** 3

        value *= k

        return value


    def line_search(self):

        E_k = self.compute_pressure_potential()

    def substep(self):


        self.surface_tension = 0.001
        self.viscosity = 0.004
        self.compute_non_pressure_forces()
        self.advect()
        self.compute_xHat()

        # self.ps.x.copy_from
        optIter = 0
        maxIter = int(1e3)
        eps = 5e-2
        tol = eps * self.dt[None]
        pad = 1.0 * self.ps.particle_diameter

        log_debug = []
        dx_norm_old = 0.0

        self.LBVH.build(self.ps.x_st, self.ps.faces_st, pad=pad)
        coef = 1e8
        h = 1.0 * self.ps.support_radius
        k = coef * self.dt[None] * self.dt[None] * (h**6)
        self.compute_densities(self.ps.xOld)
        for _ in range(maxIter):

            self.compute_inertia()
            # self.num_collision_p.fill(0)
            self.compute_densities(self.ps.x)
            self.compute_pressure(k)
            # self.compute_pressure_gn(k)
            self.compute_collision_static(pad)
            self.PCG.solve(self.ps.dx, self.ps.grad, self.ps.diagH, 1e-8, self.mat_free_Ax)

            # self.compute_search_dir()
            dx_norm = self.inf_norm(self.ps.dx)
            alpha = 1.0
            #
            # add(self.ps.xTmp, self.ps.x, alpha, self.ps.dx)
            E_b = self.compute_pressure_potential(k)
            #
            # lsIter = 0
            # for i in range(100):
            #     add(self.ps.xTmp, self.ps.x, alpha, self.ps.dx)
            #     self.compute_densities(self.ps.xTmp)
            #
            #     E_a = self.compute_pressure_potential(k)
            #     if E_a < E_b:
            #        alpha *= 0.5
            #        lsIter += 1
            #     else:
            #
            #         break
            # print("LS Iter:", lsIter)
            # add(self.ps.xTmp, self.ps.x, alpha, self.ps.dx)
            # self.compute_densities(self.ps.xTmp)
            # E_a = self.compute_pressure_potential(k)

            # if E_a < E_b:
            #     print("a")

            #
            # self.compute_pressure(k)


            # if dx_norm > 0.2 * self.ps.particle_radius:
            #     alpha = (0.2 * self.ps.particle_radius) / dx_norm
            self.update_x(alpha)
            # self.enforce_boundary_3D(self.ps.material_fluid)
            if dx_norm < tol:
                break
            optIter += 1
            dx_norm_old = dx_norm
            log_debug.append(dx_norm)

        print("opt Iter:", optIter)
        self.compute_velocity()
        #
        if optIter == maxIter:
            print("Failed to converge...")
            plt.plot(np.array(log_debug))
            plt.yscale('log')
            plt.show()
            exit()

        # self.compute_densities()
        # self.compute_pressure_forces()