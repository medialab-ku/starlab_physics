
import taichi as ti
from math_utils import *
from LBVH import LBVH
from sph_base import SPHBase
from PCG import PCG
from distance import *
import matplotlib.pyplot as plt
import numpy as np
from util.ti_ccd_module.CCDModule import CCDModule


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

        self.Vij_num = ti.field(dtype=int, shape=self.ps.fluid_particle_num)
        self.Vij_idx = ti.field(dtype=int, shape=(self.ps.fluid_particle_num, self.cache_size))
        self.Vij_val = ti.Matrix.field(n=3, m=3, dtype=float, shape=(self.ps.fluid_particle_num, self.cache_size))
        self.gn_grad = ti.Vector.field(n=3, dtype=float, shape=(self.ps.fluid_particle_num, self.cache_size))

        self.viscosity  = self.ps.cfg.get_cfg("viscosity")
        self.maxOptIter = int(self.ps.cfg.get_cfg("MaxOptIter"))
        self.tol        = self.ps.cfg.get_cfg("tol")
        self.k_rho      = self.ps.cfg.get_cfg("k_rho")
        self.da_ratio   = self.ps.cfg.get_cfg("da_ratio")
        self.use_gn     = bool(self.ps.cfg.get_cfg("use_gn"))
        self.use_div    = bool(self.ps.cfg.get_cfg("use_div"))
        self.ccd = CCDModule()


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
    def poly6_value(self, s, h):
        result = 0.0
        poly6_factor = 4.0 / ti.math.pi
        if 0 <= s and s < h:
            x = (h * h - s * s)
            result = (poly6_factor / ti.pow(h, 8)) * x * x * x

        return result

    @ti.func
    def cubic_value(self, r_norm, h):
        res = ti.cast(0.0, ti.f32)
        # h = self.ps.support_radius
        # value of cubic spline smoothing kernel
        k = 1.0
        if self.ps.dim == 1:
            k = 4 / 3
        elif self.ps.dim == 2:
            k = 40 / 7 / ti.math.pi
        elif self.ps.dim == 3:
            k = 8 / ti.math.pi
        k /= h ** self.ps.dim
        q = r_norm / h
        if q <= 1.0:
            if q <= 0.5:
                q2 = q * q
                q3 = q2 * q
                res = k * (6.0 * q3 - 6.0 * q2 + 1)
            else:
                res = k * 2 * ti.pow(1 - q, 3.0)
        return res

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
    def compute_densities(self, x: ti.template(), h: float):
        # for p_i in range(self.ps.particle_num[None]):
        a = self.ps
        # h = self.ps.support_radius

        max_num = 0
        for p_i in ti.grouped(x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            self.ps.density[p_i] = self.ps.m_V[p_i] * self.density_0 * self.cubic_value(0.0, h)
            # for p_i in ti.grouped(self.ps.x):
            self.num_collision_p[p_i] = 0
            center_cell = a.pos_to_index(x[p_i])
            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * a.dim)):
                grid_index = a.flatten_grid_index(center_cell + offset)
                for p_j in range(a.grid_particles_num[ti.max(0, grid_index - 1)], a.grid_particles_num[grid_index]):
                    if p_i[0] != p_j and (x[p_i] - x[p_j]).norm() < h:
                        x_i = x[p_i]
                        # h = self.ps.support_radius
                        if self.ps.material[p_j] == self.ps.material_fluid:
                            x_j = x[p_j]
                            r = (x_i - x_j).norm()
                            self.ps.density[p_i] += self.ps.m_V[p_j] * self.density_0 * self.cubic_value(r, h)
                            ni = ti.atomic_add(self.num_collision_p[p_i], 1)
                            self.collision_idx_p[p_i, ni] = p_j

            ti.atomic_max(max_num, self.num_collision_p[p_i])
            if self.num_collision_p[p_i] > self.cache_size:
                print("warning")

    @ti.kernel
    def compute_neighbors(self, x: ti.template(), h: float):

        a = self.ps
        max_num = 0
        for p_i in ti.grouped(x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue

            # self.ps.density[p_i] = self.ps.m_V[p_i] * self.density_0 * self.cubic_value(0.0, h)
            self.num_collision_p[p_i] = 0
            center_cell = a.pos_to_index(x[p_i])

            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * a.dim)):

                grid_index = a.flatten_grid_index(center_cell + offset)
                for p_j in range(a.grid_particles_num[ti.max(0, grid_index - 1)], a.grid_particles_num[grid_index]):
                    if p_i[0] != p_j and (x[p_i] - x[p_j]).norm() < h:
                        x_i = x[p_i]

                        if self.ps.material[p_j] == self.ps.material_fluid:
                            x_j = x[p_j]
                            r = (x_i - x_j).norm()
                            self.ps.density[p_i] += self.ps.m_V[p_j] * self.density_0 * self.cubic_value(r, h)
                            ni = ti.atomic_add(self.num_collision_p[p_i], 1)
                            self.collision_idx_p[p_i, ni] = p_j

            ti.atomic_max(max_num, self.num_collision_p[p_i])
            if self.num_collision_p[p_i] > self.cache_size:
                print("warning")


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
        # d = 2 * (self.ps.dim + 2)
        # x_j = self.ps.x[p_j]
        # # Compute the viscosity force contribution
        # r = x_i - x_j
        # v_xy = (self.ps.v[p_i] - self.ps.v[p_j]).dot(r)
        #
        # if self.ps.material[p_j] == self.ps.material_fluid:
        #     f_v = d * self.viscosity * (self.ps.m[p_j] / (self.ps.density[p_j])) * v_xy / (r.norm()**2 + 0.01 * self.ps.support_radius**2) * self.cubic_kernel_derivative(r)
        #     ret += f_v
        # elif self.ps.material[p_j] == self.ps.material_solid:
        #     boundary_viscosity = 0.0
        #     # Boundary neighbors
        #     ## Akinci2012
        #     f_v = d * boundary_viscosity * (self.density_0 * self.ps.m_V[p_j] / (self.ps.density[p_i])) * v_xy / (
        #         r.norm()**2 + 0.01 * self.ps.support_radius**2) * self.cubic_kernel_derivative(r)
        #     ret += f_v
        #     if self.ps.is_dynamic_rigid_body(p_j):
        #         self.ps.acceleration[p_j] += -f_v * self.density_0 / self.ps.density[p_j]


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
            # if self.ps.material[p_i] == self.ps.material_fluid:
            #     self.ps.for_all_neighbors(p_i, self.compute_non_pressure_forces_task, d_v)
            #     self.ps.acceleration[p_i] = d_v


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
            # self.ps.grad[p_i]  = self.density_0 * self.ps.m_V[p_i] * ti.math.vec3(0.0)
            self.ps.grad[p_i]  = self.density_0 * self.ps.m_V[p_i] * (self.ps.x[p_i] - self.ps.xHat[p_i])
            self.ps.diagH[p_i] = self.density_0 * self.ps.m_V[p_i] * I_3x3



    @ti.kernel
    def compute_pressure(self, k: float, h:float, eta: float):

        mass_i = self.ps.m_V[0] * self.density_0
        rho_min = mass_i * self.cubic_value(0.0, h)
        a = rho_min + eta * (self.density_0 - rho_min)
        for p_i in ti.grouped(self.ps.x):

            dEdc = 1.0
            d2Edc2 = 0.0
            #
            t = (ti.max(self.ps.density[p_i] - a, 0.0)) / (self.density_0 - a)
            if self.ps.density[p_i] < self.density_0:

                dEdc = 3 * (t ** 2) - 2 * t ** 3
                d2Edc2 = 6 * t * (1.0 - t) / (self.density_0 - a)

            for ni in range(self.num_collision_p[p_i]):
                p_j = self.collision_idx_p[p_i, ni]
                x_i = self.ps.x[p_i]
                if self.ps.material[p_j] == self.ps.material_fluid:
                    x_j = self.ps.x[p_j]
                    r = (x_i - x_j).norm()

                    if r < 1e-6:
                        r = 1e-6

                    n = (x_i - x_j) / r
                    mass_j = self.ps.m_V[p_j] * self.density_0

                    dWdr = self.spiky_kernel_derivative(r, h)
                    # dWdr = self.barrier_grad(r, h)
                    dcdx = mass_j * dWdr * n

                    d2Wdr2 = self.spiky_kernel_hessian(r, h)
                    # d2Wdr2 = self.barrier_hess(r, h)

                    nnT = n.outer_product(n)
                    d2cdx2 = mass_j * d2Wdr2 * nnT
                    self.ps.grad[p_i] += k * dEdc * dcdx
                    self.ps.grad[p_j] -= k * dEdc * dcdx

                    self.ps.diagH[p_i] += k * (dEdc * d2cdx2 + d2Edc2 * dcdx.outer_product(dcdx))
                    self.ps.diagH[p_j] += k * (dEdc * d2cdx2 + d2Edc2 * dcdx.outer_product(dcdx))

                    self.collision_grad_p[p_i, ni] = ti.sqrt(k * d2Edc2) * dcdx
                    self.collision_H_p[p_i, ni] = k * dEdc * d2cdx2

    @ti.kernel
    def precompute_viscosity(self, x: ti.template(), viscosity: float, h: float):

        reg = 0.01 * h ** 2
        a =self.ps
        k = 20 * viscosity * h * self.dt[None] * self.dt[None]
        for p_i in ti.grouped(self.ps.x):
            x_i = x[p_i]
            m_i = self.ps.m_V[p_i] * self.density_0
            rho_i = self.ps.density[p_i]
            self.Vij_num[p_i] = 0
            center_cell = a.pos_to_index(x[p_i])
            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * a.dim)):
                grid_index = a.flatten_grid_index(center_cell + offset)
                for p_j in range(a.grid_particles_num[ti.max(0, grid_index - 1)], a.grid_particles_num[grid_index]):
                    if p_i[0] != p_j and (x[p_i] - x[p_j]).norm() < h:
                        m_j = self.ps.m_V[p_j] * self.density_0
                        x_j = x[p_j]
                        x_ij = x_i - x_j
                        r = x_ij.norm()

                        if r < 1e-6:
                            r = 1e-6

                        n = x_ij / r
                        coef_ij = (k / (r * r + reg)) * (m_i * m_j) / (rho_i + self.ps.density[p_j])
                        dWdr = self.spiky_kernel_derivative(r, h)

                        ni = ti.atomic_add(self.Vij_num[p_i], 1)
                        self.Vij_idx[p_i, ni] = p_j
                        self.Vij_val[p_i, ni] = coef_ij * (-dWdr) * n.outer_product(x_ij)
    @ti.kernel
    def precompute_pressure_gn(self, x: ti.template(), k:float , h: float):

        for p_i in ti.grouped(self.ps.x):

            # dEdc = 1.0
            # d2Edc2 = 0.0
            Jn = self.density_0 / ti.max(self.ps.density[p_i], self.density_0)
            x_i = x[p_i]

            for ni in range(self.num_collision_p[p_i]):
                p_j = self.collision_idx_p[p_i, ni]
                x_j = x[p_j]
                xij = x_i - x_j
                r = xij.norm()
                mass_j = self.ps.m_V[p_j] * self.density_0
                dWdr = (mass_j / ti.max(self.ps.density[p_j], self.density_0)) * self.spiky_kernel_derivative(r, h)

                if r < 1e-5:
                    r = 1e-5

                n = xij / r

                grad_i = Jn * dWdr * n
                self.collision_grad_p[p_i, ni] = ti.sqrt(k) * grad_i
                self.collision_H_p[p_i, ni] = ti.math.mat3(0.0)

    @ti.kernel
    def compute_viscosity(self):

        for p_i in ti.grouped(self.ps.x):
            # dEdc = 1.0
            # d2Edc2 = 0.0

            v_i = (self.ps.x[p_i] - self.ps.xOld[p_i]) / self.dt[None]
            for ni in range(self.Vij_num[p_i]):
                p_j = self.Vij_idx[p_i, ni]
                v_j = (self.ps.x[p_j] - self.ps.xOld[p_j]) / self.dt[None]
                Vij = self.Vij_val[p_i, ni]
                val = Vij @ (v_i - v_j)
                self.ps.grad[p_i] += val
                self.ps.grad[p_j] -= val

                self.ps.diagH[p_i] += Vij
                self.ps.diagH[p_j] += Vij


    @ti.kernel
    def compute_pressure_gn(self, x: ti.template(),  k: float, h: float):

        for p_i in ti.grouped(self.ps.x):

            # dEdc = 1.0
            # d2Edc2 = 0.0
            Jn = self.density_0 / ti.max(self.ps.density[p_i], self.density_0)
            div = 0.0
            x_i = self.ps.xOld[p_i]
            v_i = (x[p_i] - self.ps.xOld[p_i])

            for ni in range(self.num_collision_p[p_i]):
                p_j = self.collision_idx_p[p_i, ni]
                v_j = (x[p_j] - self.ps.xOld[p_j])
                x_j = self.ps.xOld[p_j]

                xij = x_i - x_j
                r = xij.norm()

                if r < 1e-5:
                    r = 1e-5

                mass_j = self.ps.m_V[p_j] * self.density_0
                dWdr = (mass_j / ti.max(self.ps.density[p_j], self.density_0)) * self.spiky_kernel_derivative(r, h)
                n = xij / r

                div += dWdr * n.dot(v_j - v_i)

            Jnew = Jn * (1.0 + div)


            for ni in range(self.num_collision_p[p_i]):
                p_j = self.collision_idx_p[p_i, ni]
                grad_i =  self.collision_grad_p[p_i, ni] / ti.sqrt(k)

                self.ps.grad[p_i] -= k * (Jnew - 1.0) * grad_i
                self.ps.grad[p_j] += k * (Jnew - 1.0) * grad_i

                self.ps.diagH[p_i] += k * (grad_i.outer_product(grad_i))
                self.ps.diagH[p_j] += k * (grad_i.outer_product(grad_i))

    @ti.kernel
    def compute_collision_static(self, pad: float):

        self.num_candidate[None] = 0
        for P in self.ps.x:
            xP = self.ps.x[P]

            _min0 = xP - ti.math.vec3(pad)
            _max0 = xP + ti.math.vec3(pad)
            self.LBVH.traverse_bvh_single_test(_min0, _max0, 0, P, self.candidate_info,  self.num_candidate)

        # print(self.num_candidate[None] )
        Kappa = 1e4 * self.dt[None] * self.dt[None]
        I_3x3 = ti.math.mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        dHat = 0.5 * pad
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
                #
                # if d == 1e-4:
                #     print("test")
                dbdx = self.barrier_grad(d, dHat)
                d2bdx2 = self.barrier_hess(d, dHat)

                nnT = n.outer_product(n)
                d2d_dx2 = (I_3x3 - nnT) / d
                test = (d2bdx2 * nnT)
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


        #static collision
        for i in range(self.num_collision[None]):
            bary = self.collision_bary[i]
            info = self.collision_info[i]
            H = self.collision_H[i]
            if self.collision_type[i] == 0:
                Ax[info[0]] += bary[0] * H @ x[info[0]]


        #pressure
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

            for j in range(self.num_collision_p[p_i]):
                p_j = self.collision_idx_p[p_i, j]
                grad = self.collision_grad_p[p_i, j]
                Ax[p_i] += value * grad
                Ax[p_j] -= value * grad

        #viscosity
        for p_i in ti.grouped(self.ps.x):

            for j in range(self.Vij_num[p_i]):
                p_j = self.Vij_idx[p_i, j]
                xji = x[p_i] - x[p_j]

                Hji = self.Vij_val[p_i, j]
                Ax[p_i] += Hji @ xji
                Ax[p_j] -= Hji @ xji

    @ti.kernel
    def mat_free_Ax2(self, Ax: ti.template(), x: ti.template()):

        for p_i in ti.grouped(self.ps.x):
            Ax[p_i] = self.density_0 * self.ps.m_V[p_i] * x[p_i]

        # static collision
        for i in range(self.num_collision[None]):
            bary = self.collision_bary[i]
            info = self.collision_info[i]
            H = self.collision_H[i]
            if self.collision_type[i] == 0:
                Ax[info[0]] += bary[0] * H @ x[info[0]]

        # pressure
        for p_i in ti.grouped(self.ps.x):
            value = 0.0

            # if self.ps.density[p_i] >= self.density_0:
            for j in range(self.num_collision_p[p_i]):
                p_j = self.collision_idx_p[p_i, j]
                xji = x[p_i] - x[p_j]

                value += self.collision_grad_p[p_i, j].dot(xji)

            for j in range(self.num_collision_p[p_i]):
                p_j = self.collision_idx_p[p_i, j]
                grad = self.collision_grad_p[p_i, j]
                Ax[p_i] += value * grad
                Ax[p_j] -= value * grad

        # viscosity
        for p_i in ti.grouped(self.ps.x):

            for j in range(self.Vij_num[p_i]):
                p_j = self.Vij_idx[p_i, j]
                xji = x[p_i] - x[p_j]

                Hji = self.Vij_val[p_i, j]
                Ax[p_i] += Hji @ xji
                Ax[p_j] -= Hji @ xji

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


    @ti.kernel
    def filter_step_size_div(self, x: ti.template(), dx: ti.template(), h: float, rat: float) -> float:

        # h = self.ps.support_radius
        rho_max = 0.0
        # rat = 1.01
        for p_i in ti.grouped(self.ps.x):
            ti.atomic_max(rho_max, self.ps.density[p_i])

        alpha_k = 1.0
        for p_i in ti.grouped(self.ps.x):
            xA, dxA = x[p_i], dx[p_i]
            div = 0.0
            for j in range(self.num_collision_p[p_i]):
                p_j = self.collision_idx_p[p_i, j]
                if p_j < 0:
                    break
                xB, dxB = x[p_j], dx[p_j]

                r = (xA - xB).norm()
                n = (xA - xB) / r
                mass_j = self.ps.m_V[p_j] * self.density_0
                div += mass_j * self.spiky_kernel_derivative(r, h) * n.dot(dxA - dxB)

                if div > 0:
                    rho_adv = self.ps.density[p_i] + div
                    if rho_adv > rat * rho_max:
                        # print("test")
                        alpha_tmp = (rat * rho_max - self.ps.density[p_i]) / div
                        # print(alpha_tmp)
                        #     # alpha_tmp = (1.1 * rho0 - rho[p_i]) / div
                        ti.atomic_min(alpha_k, alpha_tmp)

        return alpha_k

    @ti.kernel
    def filter_step_size_ccd(self, x: ti.template(), dx: ti.template()) -> float:

        alpha = 1.0

        eta = 0.01
        thickness = 0.001

        self.num_candidate[None] = 0
        for P in self.ps.x:
            xP = x[P]

            _min0 = ti.min(xP, xP + dx[P])
            _max0 = ti.min(xP, xP + dx[P])
            self.LBVH.traverse_bvh_single_test(_min0, _max0, 0, P, self.candidate_info, self.num_candidate)

        for i in range(self.num_candidate[None]):
            info = self.candidate_info[i]
            P = info[0]
            j = info[1]

            xP = x[P]
            dxP = dx[P]
            T0, T1, T2 = self.ps.faces_st[3 * j + 0], self.ps.faces_st[3 * j + 1], self.ps.faces_st[3 * j + 2]
            xT0, xT1, xT2 = self.ps.x_st[T0], self.ps.x_st[T1], self.ps.x_st[T2]

            toc = self.ccd. point_triangle_ccd(xP, xT0, xT1, xT2, dxP, ti.math.vec3(0.0), ti.math.vec3(0.0), ti.math.vec3(0.0), eta, thickness)
            ti.atomic_min(alpha, toc)

        return alpha


    def substep(self):

        v_norm = self.inf_norm(self.ps.v)
        if v_norm * self.dt[None] > self.ps.support_radius:
            print("test")

        self.compute_non_pressure_forces()
        self.advect()
        self.compute_xHat()

        optIter = 0
        numLS = 0
        pcgIter_total = 0
        pad = 1.5 * self.ps.particle_diameter

        log_debug = []
        h = 2.0 * self.ps.particle_diameter
        self.LBVH.build(self.ps.x_st, self.ps.faces_st, pad=pad)
        k = self.k_rho * self.dt[None] * self.dt[None] * (h ** 6)

        if self.use_gn:
            self.compute_densities(self.ps.xOld, h)
            self.precompute_pressure_gn(self.ps.xOld, self.k_rho * self.dt[None] * self.dt[None] * (h ** 3), h)

        self.precompute_viscosity(self.ps.xOld, self.viscosity, h)
        for _ in range(self.maxOptIter):

            self.compute_inertia()
            if self.use_gn:
                self.compute_pressure_gn(self.ps.x, self.k_rho * (self.dt[None] ** 2) * (h ** 3), h)
            else:
                self.compute_densities(self.ps.x, h)
                self.compute_pressure(k, h, eta=self.ps.eta)

            self.compute_viscosity()
            self.compute_collision_static(pad)

            # pcgIter = 0
            if self.use_gn:
                pcgIter_total += self.PCG.solve(self.ps.dx, self.ps.grad, self.ps.diagH, 1e-5, self.mat_free_Ax2)
            else:
                pcgIter_total += self.PCG.solve(self.ps.dx, self.ps.grad, self.ps.diagH, 1e-5, self.mat_free_Ax)

            dx_norm = self.inf_norm(self.ps.dx)

            alpha = 1.0
            if self.use_div:
                alpha_div = self.filter_step_size_div(self.ps.x, self.ps.dx, h, self.da_ratio)
                if alpha_div < 1.0:
                    numLS += 1
                alpha = ti.min(alpha_div, alpha)

            alpha_ccd = self.filter_step_size_ccd(self.ps.x, self.ps.dx)
            if alpha_ccd < 1.0:
                print("alpha_ccd", alpha_ccd)

            alpha = ti.min(alpha_ccd, alpha)

            self.update_x(alpha)
            if dx_norm < self.tol * self.dt[None]:
                break

            optIter += 1
            dx_norm_old = dx_norm
            log_debug.append(dx_norm)

        print("opt/pcg/LS iter:", optIter, pcgIter_total, numLS)
        self.compute_velocity()


        # if optIter == self.maxOptIter:
        #     print("Failed to converge...")
        #     plt.plot(np.array(log_debug))
        #     plt.yscale('log')
        #     plt.show()
        #     exit()

        return optIter, pcgIter_total, log_debug