from sph_base import SPHBase
from distance import *

class XSPHSolver(SPHBase):
    def __init__(self, particle_system):
        super().__init__(particle_system)
        # Pressure state function parameters(WCSPH)
        self.exponent = 7.0
        self.exponent = self.ps.cfg.get_cfg("exponent")

        self.stiffness = 50000.0
        self.stiffness = self.ps.cfg.get_cfg("stiffness")

        self.surface_tension = 0.01
        self.dt[None] = self.ps.cfg.get_cfg("timeStepSize")

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
    def compute_densities(self):
        # for p_i in range(self.ps.particle_num[None]):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            self.ps.density[p_i] = self.ps.m_V[p_i] * self.cubic_kernel(0.0)
            den = 0.0
            self.ps.for_all_neighbors(p_i, self.compute_densities_task, den)
            self.ps.density[p_i] += den
            self.ps.density[p_i] *= self.density_0

    @ti.func
    def compute_pressure_forces_task(self, p_i, p_j, ret: ti.template()):

        I_3x3 = ti.math.mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        c = self.ps.density[p_i] / self.density_0 - 1.0
        k = self.stiffness * self.dt[None] * self.dt[None]
        x_i = self.ps.x[p_i]
        # mass_i = self.ps.m_V[p_i] * self.density_0
        if self.ps.material[p_j] == self.ps.material_fluid:

            # if c >= 0:

            dEdc = c ** 2
            d2Edc2 = 2 * c

            # dEdc = c
            # d2Edc2 = 1.0

            mass_j = self.ps.m_V[p_j] * self.density_0
            x_j = self.ps.x[p_j]
            r = (x_i - x_j).norm()

            r += 1e-4
            n = (x_i - x_j) / r

            dWdr = self.spiky_kernel_derivative(r)
            dcdx = mass_j * dWdr * n / self.density_0
            dWdr = self.spiky_kernel_derivative(r)
            d2Wdr2 = self.spiky_kernel_hessian(r)

            nnT = n.outer_product(n)
            alpha = abs(dWdr / r)
            beta = (d2Wdr2 - dWdr / r)

            d2cdx2 = (mass_j/ self.density_0) * (alpha * I_3x3 + beta * nnT)
            self.ps.grad[p_i] += k * dEdc * dcdx
            self.ps.grad[p_j] -= k * dEdc * dcdx

            # if c >=0:
            self.ps.diagH[p_i] += k * (dEdc * d2cdx2 + d2Edc2 * dcdx.outer_product(dcdx))
            self.ps.diagH[p_j] += k * (dEdc * d2cdx2 + d2Edc2 * dcdx.outer_product(dcdx))

            # ret += (self.ps.pressure[p_i] + self.ps.pressure[p_j]) * dcdx
        # elif self.ps.material[p_j] == self.ps.material_solid:
        #     # Boundary neighbors
        #     dpj = self.ps.pressure[p_i] / self.density_0 ** 2
        #     ## Akinci2012
        #     x_j = self.ps.x[p_j]
        #     # Compute the pressure force contribution, Symmetric Formula
        #     f_p = -self.density_0 * self.ps.m_V[p_j] * (dpi + dpj) \
        #           * self.cubic_kernel_derivative(x_i - x_j)
        #     ret += f_p
        #     if self.ps.is_dynamic_rigid_body(p_j):
        #         self.ps.acceleration[p_j] += -f_p * self.density_0 / self.ps.density[p_j]

    @ti.kernel
    def compute_pressure_forces(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            # self.ps.density[p_i] = ti.max(self.ps.density[p_i], self.density_0)
            c = ti.max(self.ps.density[p_i] / self.density_0 - 1.0, 0.0)
            # self.ps.pressure[p_i] = self.stiffness * ti.pow(c, self.exponent)
            dv = ti.Vector([0.0 for _ in range(self.ps.dim)])
            if self.ps.density[p_i] >= self.density_0:
                self.ps.for_all_neighbors(p_i, self.compute_pressure_forces_task, dv)
        # for p_i in ti.grouped(self.ps.x):
        #     if self.ps.is_static_rigid_body(p_i):
        #         self.ps.acceleration[p_i].fill(0)
        #         continue
        #     elif self.ps.is_dynamic_rigid_body(p_i):
        #         continue
        #     dv = ti.Vector([0.0 for _ in range(self.ps.dim)])
        #

            # self.ps.for_all_neighbors(p_i, self.compute_pressure_forces_task, dv)
            # self.ps.grad[p_i] += self.dt[None] * self.dt[None] * dv

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
                ret -= self.surface_tension / self.ps.m[p_i] * self.ps.m[p_j] * r * self.cubic_kernel(
                    ti.Vector([self.ps.particle_diameter, 0.0, 0.0]).norm())

        ############### Viscosoty Force ###############
        d = 2 * (self.ps.dim + 2)
        x_j = self.ps.x[p_j]
        # Compute the viscosity force contribution
        r = x_i - x_j
        v_xy = (self.ps.v[p_i] -
                self.ps.v[p_j]).dot(r)

        if self.ps.material[p_j] == self.ps.material_fluid:
            f_v = d * self.viscosity * (self.ps.m[p_j] / (self.ps.density[p_j])) * v_xy / (r.norm() ** 2 + 0.01 * self.ps.support_radius ** 2) * self.cubic_kernel_derivative(r)
            ret += f_v
        elif self.ps.material[p_j] == self.ps.material_solid:
            boundary_viscosity = 0.0
            # Boundary neighbors
            ## Akinci2012
            f_v = d * boundary_viscosity * (self.density_0 * self.ps.m_V[p_j] / (self.ps.density[p_i])) * v_xy / (
                    r.norm() ** 2 + 0.01 * self.ps.support_radius ** 2) * self.cubic_kernel_derivative(r)
            ret += f_v
            if self.ps.is_dynamic_rigid_body(p_j):
                self.ps.acceleration[p_j] += -f_v * self.density_0 / self.ps.density[p_j]

    @ti.kernel
    def compute_elasticity(self):

        k = 1e6 * self.dt[None] * self.dt[None]
        num_edges = self.ps.edges_dy.shape[0] // 2
        I_3x3 = ti.math.mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        for e in range(num_edges):

            v0, v1 = self.ps.edges_dy[2 * e + 0], self.ps.edges_dy[2 * e + 1]
            x01 = self.ps.x_dy[v0] - self.ps.x_dy[v1]
            l = x01.norm()
            l0 = self.ps.l0[e]
            n = x01 / l
            nnT = n.outer_product(n)
            alpha = abs(l - l0) / l

            self.ps.grad_dy[v0] += k * (l - l0) * n
            self.ps.grad_dy[v1] -= k * (l - l0) * n

            self.ps.diagH_dy[v0] += k * (alpha * I_3x3 + (1.0 - alpha) * nnT)
            self.ps.diagH_dy[v1] += k * (alpha * I_3x3 + (1.0 - alpha) * nnT)

        # ids = ti.Vector([0, 1, 2, 3], dtype=int)

        k = 1e6
        for i in range(4):
            self.ps.grad_dy[i] += k * (self.ps.x_dy[i] - self.ps.x_0_dy[i])
            self.ps.diagH_dy[i] += k * I_3x3

        # self.ps.grad_dy[1] += k * (self.ps.x_dy[1] - self.ps.x_0_dy[i])
        # self.ps.diagH_dy[1] += k * I_3x3



    @ti.kernel
    def compute_non_pressure_forces(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_static_rigid_body(p_i):
                self.ps.acceleration[p_i].fill(0.0)
                continue
            ############## Body force ###############
            # Add body force
            d_v = ti.Vector(self.g)
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
                # self.ps.x[p_i] += self.dt[None] * self.ps.v[p_i]


    def substep(self):

        self.compute_non_pressure_forces()
        self.predict_velocity()
        # self.advect()
        self.compute_xHat()
        self.ps.x.copy_from(self.ps.xHat)
        self.ps.x_dy.copy_from(self.ps.xHat_dy)
        self.enforce_boundary_3D(self.ps.material_fluid)

        opt_iter = 0
        max_iter = int(1e1)

        dx_old = 0.0
        dx_init = 0.0

        self.ps.LB
        for _ in range(max_iter):
            self.compute_densities()
            self.compute_inertia()

            self.compute_elasticity()
            self.compute_pressure_forces()
            self.compute_dynamic_collision()
            self.compute_static_collision()
            self.compute_non_penetration()
            self.compute_search_dir()

            dx_new = self.inf_norm(self.ps.dx)
            # if opt_iter == 0:
            #     print("test")
            #     dx_old = dx_new
            #
            # if opt_iter > 1 and abs((dx_new - dx_old)) < 1e-1 * self.dt[None]:
            #     break

            self.update_x(1.0)
            self.enforce_boundary_3D(self.ps.material_fluid)
            dx_old = dx_new
            opt_iter += 1

        # print(opt_iter)

        self.compute_velocity()