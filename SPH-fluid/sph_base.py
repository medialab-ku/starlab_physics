from matplotlib.pyplot import axis
import taichi as ti
import numpy as np
from distance import *


@ti.data_oriented
class SPHBase:
    def __init__(self, particle_system):
        self.ps = particle_system
        self.g = ti.Vector([0.0, -9.81, 0.0])  # Gravity
        if self.ps.dim == 2:
            self.g = ti.Vector([0.0, -9.81])
        self.g = np.array(self.ps.cfg.get_cfg("gravitation"))

        self.viscosity = 0.005  # viscosity

        self.density_0 = 1000.0  # reference density
        self.density_0 = self.ps.cfg.get_cfg("density0")
        self.surface_tension = 0.00
        self.dt = ti.field(float, shape=())
        self.dt[None] = 1e-4

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
    def cubic_kernel(self, r_norm):
        res = ti.cast(0.0, float)
        h = self.ps.support_radius
        # value of cubic spline smoothing kernel
        k = 1.0
        if self.ps.dim == 1:
            k = 4 / 3
        elif self.ps.dim == 2:
            k = 40 / 7 / np.pi
        elif self.ps.dim == 3:
            k = 8 / np.pi
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
    def cubic_kernel_derivative(self, r):
        h = self.ps.support_radius
        # derivative of cubic spline smoothing kernel
        k = 1.0
        if self.ps.dim == 1:
            k = 4 / 3
        elif self.ps.dim == 2:
            k = 40 / 7 / np.pi
        elif self.ps.dim == 3:
            k = 8 / np.pi
        k = 6. * k / h ** self.ps.dim
        r_norm = r.norm()
        q = r_norm / h
        res = ti.Vector([0.0 for _ in range(self.ps.dim)])
        if r_norm > 1e-5 and q <= 1.0:
            grad_q = r / (r_norm * h)
            if q <= 0.5:
                res = k * q * (3.0 * q - 2.0) * grad_q
            else:
                factor = 1.0 - q
                res = k * (-factor * factor) * grad_q
        return res

    @ti.func
    def poly6_kernel(self, r):

        h = self.ps.support_radius
        val = 0.0
        if r <= h:
            factor = 315.0 / (64.0 * ti.math.pi * (h ** 9))
            val = factor * (h ** 2 - r ** 2) ** 3

        return val

    @ti.func
    def spiky_kernel_derivative(self, r):

        val = 0.0
        h = self.ps.support_radius
        q = r

        # if q < 1e-8:
        #     q = 1e-8

        # n = r / q

        if 0.0 <= q and q < h:
            factor = 45.0 / (ti.math.pi * (h ** 6))
            val = -factor * (h - q) ** 2

        return val

    @ti.func
    def spiky_kernel_hessian(self, r):

        val = 0.0
        h = self.ps.support_radius
        if 0 <= r and r < h:
            factor = 90.0 / (ti.math.pi * (h ** 6))
            val = factor * (h - r)

        return val

    @ti.func
    def viscosity_force(self, p_i, p_j, r):
        # Compute the viscosity force contribution
        v_xy = (self.ps.v[p_i] -
                self.ps.v[p_j]).dot(r)
        res = 2 * (self.ps.dim + 2) * self.viscosity * (self.ps.m[p_j] / (self.ps.density[p_j])) * v_xy / (
            r.norm()**2 + 0.01 * self.ps.support_radius**2) * self.cubic_kernel_derivative(
                r)
        return res

    def initialize(self):
        self.ps.initialize_particle_system()
        for r_obj_id in self.ps.object_id_rigid_body:
            self.compute_rigid_rest_cm(r_obj_id)
        self.compute_static_boundary_volume()
        self.compute_moving_boundary_volume()

    @ti.kernel
    def compute_rigid_rest_cm(self, object_id: int):
        self.ps.rigid_rest_cm[object_id] = self.compute_com(object_id)

    @ti.kernel
    def compute_static_boundary_volume(self):
        for p_i in ti.grouped(self.ps.x):
            if not self.ps.is_static_rigid_body(p_i):
                continue
            delta = self.cubic_kernel(0.0)
            self.ps.for_all_neighbors(p_i, self.compute_boundary_volume_task, delta)
            self.ps.m_V[p_i] = 1.0 / delta * 3.0  # TODO: the 3.0 here is a coefficient for missing particles by trail and error... need to figure out how to determine it sophisticatedly

    @ti.func
    def compute_boundary_volume_task(self, p_i, p_j, delta: ti.template()):
        if self.ps.material[p_j] == self.ps.material_solid:
            delta += self.cubic_kernel((self.ps.x[p_i] - self.ps.x[p_j]).norm())


    @ti.kernel
    def compute_moving_boundary_volume(self):
        for p_i in ti.grouped(self.ps.x):
            if not self.ps.is_dynamic_rigid_body(p_i):
                continue
            delta = self.cubic_kernel(0.0)
            self.ps.for_all_neighbors(p_i, self.compute_boundary_volume_task, delta)
            self.ps.m_V[p_i] = 1.0 / delta * 3.0  # TODO: the 3.0 here is a coefficient for missing particles by trail and error... need to figure out how to determine it sophisticatedly

    def substep(self):
        pass

    @ti.func
    def simulate_collisions(self, p_i, vec):
        # Collision factor, assume roughly (1-c_f)*velocity loss after collision
        c_f = 0.5
        self.ps.v[p_i] -= (1.0 + c_f) * self.ps.v[p_i].dot(vec) * vec

    @ti.kernel
    def enforce_boundary_2D(self, particle_type:int):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] == particle_type and self.ps.is_dynamic[p_i]: 
                pos = self.ps.x[p_i]
                collision_normal = ti.Vector([0.0, 0.0])
                if pos[0] > self.ps.domain_size[0] - self.ps.padding:
                    collision_normal[0] += 1.0
                    self.ps.x[p_i][0] = self.ps.domain_size[0] - self.ps.padding
                if pos[0] <= self.ps.padding:
                    collision_normal[0] += -1.0
                    self.ps.x[p_i][0] = self.ps.padding

                if pos[1] > self.ps.domain_size[1] - self.ps.padding:
                    collision_normal[1] += 1.0
                    self.ps.x[p_i][1] = self.ps.domain_size[1] - self.ps.padding
                if pos[1] <= self.ps.padding:
                    collision_normal[1] += -1.0
                    self.ps.x[p_i][1] = self.ps.padding
                collision_normal_length = collision_normal.norm()
                if collision_normal_length > 1e-6:
                    self.simulate_collisions(p_i, collision_normal / collision_normal_length)

    @ti.kernel
    def enforce_boundary_3D(self, particle_type:int):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] == particle_type and self.ps.is_dynamic[p_i]:
                pos = self.ps.x[p_i]

                proj = pos
                # if ti.math.isnan(pos[0]) or ti.math.isnan(pos[1]) or ti.math.isnan(pos[2]):
                #     print("test")
                collision_normal = ti.Vector([0.0, 0.0, 0.0])
                if pos[0] > self.ps.domain_size[0] - self.ps.padding:
                    collision_normal[0] += 1.0
                    self.ps.x[p_i][0] = self.ps.domain_size[0] - self.ps.padding
                if pos[0] <= self.ps.padding:
                    collision_normal[0] += -1.0
                    self.ps.x[p_i][0] = self.ps.padding

                if pos[1] > self.ps.domain_size[1] - self.ps.padding:
                    collision_normal[1] += 1.0
                    self.ps.x[p_i][1] = self.ps.domain_size[1] - self.ps.padding
                if pos[1] <= self.ps.padding:
                    collision_normal[1] += -1.0
                    self.ps.x[p_i][1] = self.ps.padding

                if pos[2] > self.ps.domain_size[2] - self.ps.padding:
                    collision_normal[2] += 1.0
                    self.ps.x[p_i][2] = self.ps.domain_size[2] - self.ps.padding
                if pos[2] <= self.ps.padding:
                    collision_normal[2] += -1.0
                    self.ps.x[p_i][2] = self.ps.padding

                collision_normal_length = collision_normal.norm()
                if collision_normal_length > 1e-6:
                    self.simulate_collisions(p_i, collision_normal / collision_normal_length)


    @ti.func
    def compute_com(self, object_id):
        sum_m = 0.0
        cm = ti.Vector([0.0, 0.0, 0.0])
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic_rigid_body(p_i) and self.ps.object_id[p_i] == object_id:
                mass = self.ps.m_V0 * self.ps.density[p_i]
                cm += mass * self.ps.x[p_i]
                sum_m += mass
        cm /= sum_m
        return cm
    

    @ti.kernel
    def compute_com_kernel(self, object_id: int)->ti.types.vector(3, float):
        return self.compute_com(object_id)


    @ti.kernel
    def solve_constraints(self, object_id: int) -> ti.types.matrix(3, 3, float):
        # compute center of mass
        cm = self.compute_com(object_id)
        # A
        A = ti.Matrix([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic_rigid_body(p_i) and self.ps.object_id[p_i] == object_id:
                q = self.ps.x_0[p_i] - self.ps.rigid_rest_cm[object_id]
                p = self.ps.x[p_i] - cm
                A += self.ps.m_V0 * self.ps.density[p_i] * p.outer_product(q)

        R, S = ti.polar_decompose(A)
        
        if all(abs(R) < 1e-6):
            R = ti.Matrix.identity(ti.f32, 3)
        
        for p_i in range(self.ps.particle_num[None]):
            if self.ps.is_dynamic_rigid_body(p_i) and self.ps.object_id[p_i] == object_id:
                goal = cm + R @ (self.ps.x_0[p_i] - self.ps.rigid_rest_cm[object_id])
                corr = (goal - self.ps.x[p_i]) * 1.0
                self.ps.x[p_i] += corr
        return R
        

    # @ti.kernel
    # def compute_rigid_collision(self):
    #     # FIXME: This is a workaround, rigid collision failure in some cases is expected
    #     for p_i in range(self.ps.particle_num[None]):
    #         if not self.ps.is_dynamic_rigid_body(p_i):
    #             continue
    #         cnt = 0
    #         x_delta = ti.Vector([0.0 for i in range(self.ps.dim)])
    #         for j in range(self.ps.solid_neighbors_num[p_i]):
    #             p_j = self.ps.solid_neighbors[p_i, j]

    #             if self.ps.is_static_rigid_body(p_i):
    #                 cnt += 1
    #                 x_j = self.ps.x[p_j]
    #                 r = self.ps.x[p_i] - x_j
    #                 if r.norm() < self.ps.particle_diameter:
    #                     x_delta += (r.norm() - self.ps.particle_diameter) * r.normalized()
    #         if cnt > 0:
    #             self.ps.x[p_i] += 2.0 * x_delta # / cnt
                        


    def solve_rigid_body(self):
        for i in range(1):
            for r_obj_id in self.ps.object_id_rigid_body:
                if self.ps.object_collection[r_obj_id]["isDynamic"]:
                    R = self.solve_constraints(r_obj_id)

                    if self.ps.cfg.get_cfg("exportObj"):
                        # For output obj only: update the mesh
                        cm = self.compute_com_kernel(r_obj_id)
                        ret = R.to_numpy() @ (self.ps.object_collection[r_obj_id]["restPosition"] - self.ps.object_collection[r_obj_id]["restCenterOfMass"]).T
                        self.ps.object_collection[r_obj_id]["mesh"].vertices = cm.to_numpy() + ret.T

                    # self.compute_rigid_collision()
                    self.enforce_boundary_3D(self.ps.material_solid)

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
    def predict_velocity(self):
        # compute new velocities only considering non-pressure forces
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic[p_i] and self.ps.material[p_i] == self.ps.material_fluid:
                self.ps.v[p_i] += self.dt[None] * self.ps.acceleration[p_i]

        for p_i in ti.grouped(self.ps.x_dy):
            # if self.ps.is_dynamic[p_i] and self.ps.material[p_i] == self.ps.material_fluid:
            self.ps.v_dy[p_i] += self.dt[None] * ti.Vector(self.g)


    @ti.kernel
    def compute_xHat(self):

        for p_i in ti.grouped(self.ps.x):
            self.ps.xOld[p_i] = self.ps.x[p_i]
            if self.ps.is_dynamic[p_i]:
                self.ps.xHat[p_i] = self.ps.xOld[p_i] + self.dt[None] * self.ps.v[p_i]

        for p_i in ti.grouped(self.ps.x_dy):
            self.ps.xOld_dy[p_i] = self.ps.x_dy[p_i]
            # if self.ps.is_dynamic[p_i]:
            self.ps.xHat_dy[p_i] = self.ps.xOld_dy[p_i] + self.dt[None] * self.ps.v_dy[p_i]

    @ti.kernel
    def compute_velocity(self):
        for p_i in ti.grouped(self.ps.x):
            self.ps.v[p_i] = (self.ps.x[p_i] - self.ps.xOld[p_i]) / self.dt[None]

        for p_i in ti.grouped(self.ps.x_dy):
            self.ps.v_dy[p_i] = (self.ps.x_dy[p_i] - self.ps.xOld_dy[p_i]) / self.dt[None]


    @ti.kernel
    def compute_DFSPH_factor(self):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            sum_grad_p_k = 0.0
            grad_p_i = ti.Vector([0.0 for _ in range(self.ps.dim)])

            # `ret` concatenates `grad_p_i` and `sum_grad_p_k`
            ret = ti.Vector([0.0 for _ in range(self.ps.dim + 1)])

            self.ps.for_all_neighbors(p_i, self.compute_DFSPH_factor_task, ret)

            sum_grad_p_k = ret[3]
            for i in ti.static(range(3)):
                grad_p_i[i] = ret[i]

            sum_grad_p_k += grad_p_i.norm_sqr()
            self.ps.dfsph_factor[p_i] = sum_grad_p_k

    @ti.func
    def compute_DFSPH_factor_task(self, p_i, p_j, ret: ti.template()):
        if self.ps.material[p_j] == self.ps.material_fluid:
            # Fluid neighbors
            grad_p_j = -self.ps.m_V[p_j] * self.spiky_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
            ret[3] += grad_p_j.norm_sqr()  # sum_grad_p_k
            for i in ti.static(range(3)):  # grad_p_i
                ret[i] -= grad_p_j[i]
        # elif self.ps.material[p_j] == self.ps.material_solid:
        #     # Boundary neighbors
        #     ## Akinci2012
        #     grad_p_j = -self.spiky_kernel_derivative(self.ps.x[p_i] - self.ps.x[p_j])
        #     for i in ti.static(range(3)):  # grad_p_i
        #         ret[i] -= grad_p_j[i]

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

        # print( self.density_0 / (self.poly6_kernel(0.0 * self.ps.support_radius) * self.ps.m_V[0]))

        # for p_i in range(self.ps.particle_num[None]):
        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            self.ps.density[p_i] = self.ps.m_V[p_i] * self.poly6_kernel(0.0)
            den = 0.0
            self.ps.for_all_neighbors(p_i, self.compute_densities_task, den)
            self.ps.density[p_i] += den
            self.ps.density[p_i] *= self.density_0

    @ti.func
    def compute_pressure_forces_task(self, p_i, p_j, ret: ti.template()):

        k = 1e3
        k_corr = 1e-4
        x_i = self.ps.x[p_i]
        # dpi = self.ps.pressure[p_i] / self.ps.density[p_i] ** 2
        lambda_i = -k * self.ps.pressure[p_i] / (k * self.ps.dfsph_factor[p_i] + 1.0)

        # Fluid neighbors
        if self.ps.material[p_j] == self.ps.material_fluid:
            x_j = self.ps.x[p_j]
            r = (x_i - x_j).norm()
            lambda_corr = -k_corr * (self.poly6_kernel(r)/self.poly6_kernel(0)) ** 4
            lambda_j = -k * self.ps.pressure[p_j] / (k * self.ps.dfsph_factor[p_j] + 1.0)
            self.ps.x[p_i] -= self.ps.pressure[p_i] * self.ps.m_V[p_j] * self.cubic_kernel_derivative(x_i - x_j)
            self.ps.x[p_j] += self.ps.pressure[p_i] * self.ps.m_V[p_j] * self.cubic_kernel_derivative(x_i - x_j)

        # elif self.ps.material[p_j] == self.ps.material_solid:
        #     # Boundary neighbors
        #     dpj = self.ps.pressure[p_i] / self.density_0 ** 2
        #     ## Akinci2012
        #     x_j = self.ps.x[p_j]
        #     # Compute the pressure force contribution, Symmetric Formula
        #     f_p = -self.density_0 * self.ps.m_V[p_j] * (dpi + dpj) * self.cubic_kernel_derivative(x_i - x_j)
        #     ret += f_p
        #     if self.ps.is_dynamic_rigid_body(p_j):
        #         self.ps.acceleration[p_j] += -f_p * self.density_0 / self.ps.density[p_j]

    @ti.kernel
    def compute_pressure_forces(self):

        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            self.ps.pressure[p_i] = 1e8 * ti.max(self.ps.density[p_i] - self.density_0, 0.0)

            dx = ti.Vector([0.0 for _ in range(self.ps.dim)])
            self.ps.for_all_neighbors(p_i, self.compute_pressure_forces_task, dx)
            # self.ps.x[p_i] += dx

    @ti.kernel
    def compute_inertia(self):

        I_3x3 = ti.math.mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        density = 1e1
        for p_i in ti.grouped(self.ps.x):
            self.ps.grad[p_i]  = self.ps.density[p_i] * (self.ps.x[p_i] - self.ps.xHat[p_i])
            self.ps.diagH[p_i] = self.ps.density[p_i] * I_3x3

        for p_i in ti.grouped(self.ps.x_dy):
            self.ps.grad_dy[p_i] = density * self.ps.mass_dy[p_i] * (self.ps.x_dy[p_i] - self.ps.xHat_dy[p_i])
            self.ps.diagH_dy[p_i] = density * self.ps.mass_dy[p_i] * I_3x3

        k = 1e3
        self.ps.grad_dy[0] = k * (self.ps.x_dy[0] - self.ps.x_0_dy[0])
        self.ps.diagH_dy[0] = k  * I_3x3

    @ti.func
    def compute_pressure_task(self, p_i, p_j, ret):

        k = 1 * self.dt[None] * self.dt[None]
        I_3x3 = ti.math.mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

        c = (self.ps.density[p_i] / self.density_0 - 1.0)

        dEdc = 3 * c ** 2
        d2Edc2 = 6 * c ** 1

        dEdc = 1
        # d2Edc2 = 0.0

        xij = self.ps.x[p_i] - self.ps.x[p_j]

        # dHat = 0.1 * self.ps.support_radius
        r = xij.norm()
        n = xij / r

        if r < 1e-5:
            # r = 1e-5
            n = ti.math.vec3([0.0, 0.0, 0.0])

        # n = xij / r
        # a = 1e3
        # if r <= self.ps.dHat:
        #     dbdx = self.barrier_grad(r, self.ps.dHat)
        #     d2bdx2 = self.barrier_hess(r, self.ps.dHat)
        #
        #     nnT = n.outer_product(n)
        #     d2d_dx2 = (I_3x3 - nnT) / r
        #     test = (abs(d2bdx2) * nnT + abs(dbdx) * d2d_dx2)
        #
        #     self.ps.grad[p_i] += a * dbdx * n
        #     self.ps.grad[p_j] -= a * dbdx * n
        #
        #     self.ps.diagH[p_i] += a * test
        #     self.ps.diagH[p_j] += a * test

        # if ti.math.isnan(r):
        #     print("test 0")

        # r = xij.norm()
        # n = xij / r
        # if r < 1e-8:
        #     r = 1e-8

        # n = xij / r
        # if r < 1e-8:
        #     r = 1e-8
        #     n = ti.math.vec3([0.0, 1.0, 0.0])
        # # if ti.math.isnan(n[0]) or ti.math.isnan(n[1]) or ti.math.isnan(n[1]):
        # #     print("test 1")
        #
        # #
        # # if r < 1e-8:
        # #     n = ti.math.vec3([0.0, 1.0, 0.0])
        #
        dWdr   = self.spiky_kernel_derivative(r)
        d2Wdr2 = self.spiky_kernel_hessian(r)
        dcdx_j = dWdr * n / self.density_0

        dEdx_j = dEdc * dcdx_j
        if ti.math.isnan(dEdx_j[0]) or ti.math.isnan(dEdx_j[1]) or ti.math.isnan(dEdx_j[1]):
            print("test 1")


        nnT = n.outer_product(n)
        alpha = abs(dWdr / r)
        beta = abs(d2Wdr2 - dWdr / r)
        d2cdx2_j = (alpha * I_3x3 + beta * nnT) / (self.density_0 * self.density_0)
        d2Edx2_j = (d2Edc2 * dcdx_j.outer_product(dcdx_j) + dEdc * d2cdx2_j)

        # if ti.math.isnan(dEdx_j[0]) or ti.math.isnan(dEdx_j[1]) or ti.math.isnan(dEdx_j[1]):
        #     print("test 2")


        self.ps.grad[p_i] += k * dEdx_j
        self.ps.grad[p_j] -= k * dEdx_j

        # self.ps.diagH[p_i] += k * d2Edx2_j
        # self.ps.diagH[p_j] += k * d2Edx2_j

        # self.h_ij[i, nj] = k_rho * abs(dEdc) * d2cdx2_j
        # self.grad_ij[i, nj] = ti.sqrt(k_rho * abs(d2Edc2)) * dcdx_j


    @ti.kernel
    def compute_pressure(self):

        for p_i in ti.grouped(self.ps.x):
            if self.ps.material[p_i] != self.ps.material_fluid:
                continue
            # self.ps.pressure[p_i] = ti.max(self.ps.density[p_i] / self.density_0 - 1.0, 0.0)
            ret = ti.Struct(grad_i = ti.math.vec3(0.0), diagH_i = ti.math.mat3(0.0))

            if self.ps.density[p_i] / self.density_0 - 1.0 >= 0.0:
                self.ps.for_all_neighbors(p_i, self.compute_pressure_task, ret)

        # for p_i in ti.grouped(self.ps.x):
        #     grad_i = self.ps.grad[p_i]
        #     if ti.math.isnan(grad_i[0]) or ti.math.isnan(grad_i[1]) or ti.math.isnan(grad_i[2]):
        #         print("test 1")
            # self.ps.grad[p_i]  += ret.grad_i
            # self.ps.diagH[p_i] += ret.diagH_i

    @ti.kernel
    def compute_static_collision(self):

        I_3x3 = ti.math.mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        num_static_faces = self.ps.faces_st.shape[0] // 3
        dHat = 2 * self.ps.particle_radius
        Kappa = 1e5
        # for P in ti.grouped(self.ps.x):
        #     xP = self.ps.x[P]
        #     for j in range(num_static_faces):
        #         T0, T1, T2 = self.ps.faces_st[3 * j + 0], self.ps.faces_st[3 * j + 1], self.ps.faces_st[3 * j + 2]
        #         xT0, xT1, xT2 = self.ps.x_st[T0], self.ps.x_st[T1], self.ps.x_st[T2]
        #         type = d_type_PT(xP, xT0, xT1, xT2)
        #         # print(type)
        #         bary = ti.math.vec3(0.0)
        #         if type == 0:
        #             bary[0] = 1.0
        #
        #         elif type == 1:
        #             bary[1] = 1.0
        #
        #         elif type == 2:
        #             bary[2] = 1.0
        #
        #         elif type == 3:
        #             a = d_PE(xP, xT0, xT1)
        #             bary[0] = a[0]
        #             bary[1] = a[1]
        #
        #         elif type == 4:
        #             a = d_PE(xP, xT1, xT2)
        #             bary[1] = a[0]
        #             bary[2] = a[1]
        #
        #         elif type == 5:
        #             a = d_PE(xP, xT0, xT2)
        #             bary[0] = a[0]
        #             bary[2] = a[1]
        #
        #             # print(bary[0], bary[2])
        #
        #         elif type == 6:
        #             bary = d_PT(xP, xT0, xT1, xT2)
        #
        #         proj = bary[0] * xT0 + bary[1] * xT1 + bary[2] * xT2
        #         d = (xP - proj).norm()
        #
        #         # if d < 1e-5:
        #         #
        #
        #         n = (xP - proj) / d
        #
        #         if d <= dHat:
        #             dbdx = self.barrier_grad(d, dHat)
        #             d2bdx2 = self.barrier_hess(d, dHat)
        #
        #             nnT = n.outer_product(n)
        #             d2d_dx2 = (I_3x3 - nnT) / d
        #             test = (abs(d2bdx2) * nnT + abs(dbdx) * d2d_dx2)
        #             self.ps.diagH[P] += Kappa * test
        #             self.ps.grad[P] += Kappa * dbdx * n

        for P in ti.grouped(self.ps.x_dy):
            xP = self.ps.x_dy[P]
            for j in range(num_static_faces):
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

                    # print(bary[0], bary[2])

                elif type == 6:
                    bary = d_PT(xP, xT0, xT1, xT2)

                proj = bary[0] * xT0 + bary[1] * xT1 + bary[2] * xT2
                d = (xP - proj).norm()

                n = (xP - proj) / d

                if d <= dHat:
                    dbdx = self.barrier_grad(d, dHat)
                    d2bdx2 = self.barrier_hess(d, dHat)

                    nnT = n.outer_product(n)
                    d2d_dx2 = (I_3x3 - nnT) / d
                    test = (abs(d2bdx2) * nnT + abs(dbdx) * d2d_dx2)
                    self.ps.diagH_dy[P] += Kappa * test
                    self.ps.grad_dy[P]  += Kappa * dbdx * n

    @ti.kernel
    def compute_dynamic_collision(self):

        I_3x3 = ti.math.mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        num_faces_dy = self.ps.faces_dy.shape[0] // 3
        dHat = 2 * self.ps.particle_radius
        Kappa = 1e2
        for P in ti.grouped(self.ps.x):
            xP = self.ps.x[P]
            for j in range(num_faces_dy):
                T0, T1, T2 = self.ps.faces_dy[3 * j + 0], self.ps.faces_dy[3 * j + 1], self.ps.faces_dy[3 * j + 2]
                xT0, xT1, xT2 = self.ps.x_dy[T0], self.ps.x_dy[T1], self.ps.x_dy[T2]
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

                    # print(bary[0], bary[2])

                elif type == 6:
                    bary = d_PT(xP, xT0, xT1, xT2)

                proj = bary[0] * xT0 + bary[1] * xT1 + bary[2] * xT2
                d = (xP - proj).norm()

                # if d < 1e-5:
                #

                n = (xP - proj) / d

                if d <= dHat:
                    dbdx = self.barrier_grad(d, dHat)
                    d2bdx2 = self.barrier_hess(d, dHat)

                    nnT = n.outer_product(n)
                    d2d_dx2 = (I_3x3 - nnT) / d
                    test = (abs(d2bdx2) * nnT + abs(dbdx) * d2d_dx2)
                    self.ps.diagH[P] += Kappa * test
                    self.ps.grad[P] += Kappa * dbdx * n

                    self.ps.grad_dy[T0] -= bary[0] * Kappa * dbdx * n
                    self.ps.grad_dy[T1] -= bary[1] * Kappa * dbdx * n
                    self.ps.grad_dy[T2] -= bary[2] * Kappa * dbdx * n

                    self.ps.diagH_dy[T0] += bary[0] * Kappa * test
                    self.ps.diagH_dy[T1] += bary[1] * Kappa * test
                    self.ps.diagH_dy[T2] += bary[2] * Kappa * test



    @ti.kernel
    def compute_non_penetration(self):

        k = 1e7 * self.dt[None] * self.dt[None]
        I_3x3 = ti.math.mat3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        for p_i in ti.grouped(self.ps.x):
            if self.ps.is_dynamic[p_i]:
                pos = self.ps.x[p_i]
                proj = pos

                flag = False
                # if ti.math.isnan(pos[0]) or ti.math.isnan(pos[1]) or ti.math.isnan(pos[2]):
                #     print("test")
                n = ti.Vector([0.0, 0.0, 0.0])
                d = 0.0
                if pos[0] > self.ps.domain_size[0] - self.ps.padding:
                    n[0] += 1.0
                    proj[0] = self.ps.domain_size[0] - self.ps.padding
                    flag = True
                if pos[0] <= self.ps.padding:
                    n[0] += -1.0
                    proj[0] = self.ps.padding
                    flag = True
                if pos[1] > self.ps.domain_size[1] - self.ps.padding:
                    n[1] += 1.0
                    proj[1] = self.ps.domain_size[1] - self.ps.padding
                if pos[1] <= self.ps.padding:
                    n[1] += -1.0
                    proj[1] = self.ps.padding
                    flag = True
                if pos[2] > self.ps.domain_size[2] - self.ps.padding:
                    n[2] += 1.0
                    proj[2] = self.ps.domain_size[2] - self.ps.padding
                    flag = True
                if pos[2] <= self.ps.padding:
                    n[2] += -1.0
                    proj[2] = self.ps.padding
                    flag = True

                if flag:
                    d = (pos - proj).norm()
                    nnT = n.outer_product(n)
                    d2d_dx2 = (I_3x3 - nnT) / d

                    self.ps.grad[p_i]  += (self.ps.density[p_i] + k) * d * n
                    self.ps.diagH[p_i] += (self.ps.density[p_i] + k) * I_3x3


    @ti.kernel
    def compute_search_dir(self):
        for p_i in ti.grouped(self.ps.x):
            dx = -self.ps.diagH[p_i].inverse() @ self.ps.grad[p_i]
            self.ps.dx[p_i] = dx

        for p_i in ti.grouped(self.ps.x_dy):
            dx = -self.ps.diagH_dy[p_i].inverse() @ self.ps.grad_dy[p_i]
            self.ps.dx_dy[p_i] = dx
    # #
    # @ti.func
    # def filter_step_size_task(self, p_i, p_j, ret: ti.template()):
    #
    #     x_rel = self.ps.x[p_i] - self.ps.x[p_j]
    #     dx_rel = self.ps.dx[p_i] - self.ps.dx[p_j]
    #
    #     ret = 1.0
    #
    #     if dx_rel.dot(dx_rel) > 0.0:




    @ti.kernel
    def filter_step_size(self) -> float:

        alpha = 1.0
        for p_i in ti.grouped(self.ps.x):
            # ret = 1.0
            center_cell = self.ps.pos_to_index(self.ps.x[p_i])
            for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.ps.dim)):
                grid_index = self.ps.flatten_grid_index(center_cell + offset)
                for p_j in range(self.ps.grid_particles_num[ti.max(0, grid_index - 1)], self.ps.grid_particles_num[grid_index]):
                    if p_i[0] != p_j and (self.ps.x[p_i] - self.ps.x[p_j]).norm() < self.ps.support_radius:
                        x_rel = self.ps.x[p_i] - self.ps.x[p_j]
                        dx_rel = self.ps.dx[p_i] - self.ps.dx[p_j]

                        a = dx_rel.dot(dx_rel)
                        b = dx_rel.dot(x_rel)
                        c = x_rel.dot(x_rel) - (self.ps.dHat) ** 2

                        det = b ** 2 - a * c
                        alpha_tmp = 1.0
                        if det > 0.0 and b < 0.0:
                            alpha_tmp = (-b + ti.sqrt(det)) / a
                            alpha_tmp = ti.min(alpha_tmp, 1.0)

                        ti.atomic_min(alpha, alpha_tmp)

        # for p_i in ti.grouped(self.ps.x):
        #     center_cell = self.ps.pos_to_index(self.ps.x[p_i])
        #     for offset in ti.grouped(ti.ndrange(*((-1, 2),) * self.ps.dim)):
        #         grid_index = self.ps.flatten_grid_index(center_cell + offset)
        #         for p_j in range(self.ps.grid_particles_num[ti.max(0, grid_index - 1)],
        #                          self.ps.grid_particles_num[grid_index]):
        #             if p_i[0] != p_j and (self.ps.x[p_i] - self.ps.x[p_j]).norm() < self.ps.support_radius:
        #                 x_rel = self.ps.x[p_i] - self.ps.x[p_j]
        #                 dx_rel = self.ps.dx[p_i] - self.ps.dx[p_j]
        #
        #                 test = x_rel + alpha * dx_rel
        #
        #                 if test.norm() < self.ps.dHat:
        #                     print("fuck")


        return alpha

    @ti.kernel
    def update_x(self, alpha: float):

        for p_i in ti.grouped(self.ps.x):
            self.ps.x[p_i] += alpha * self.ps.dx[p_i]

        for p_i in ti.grouped(self.ps.x_dy):
            self.ps.x_dy[p_i] += alpha * self.ps.dx_dy[p_i]

    @ti.kernel
    def warm_start(self):
        for p_i in ti.grouped(self.ps.x):
            self.ps.dx[p_i] = (self.ps.xHat[p_i] - self.ps.x[p_i])

    @ti.kernel
    def inf_norm(self, p: ti.template()) -> float:

        val = 0.0
        for p_i in ti.grouped(p):
            tmp = p[p_i].norm()
            ti.atomic_max(val, tmp)

        return val

    def step(self):

        # self.compute_non_pressure_forces()
        # self.predict_velocity()
        #
        # self.compute_xHat()
        # self.warm_start()
        # alpha = self.filter_step_size()
        # self.update_x(1)
        # self.enforce_boundary_3D(self.ps.material_fluid)
        # self.ps.initialize_particle_system()
        #
        #
        # # opt_itr = 0
        # # max_itr = int(2)
        # # for _ in range(max_itr):
        # #
        # #     self.compute_densities()
        # #     self.compute_inertia()
        # #     self.compute_pressure()
        # #     self.compute_non_penetration()
        # #
        # #     self.compute_search_dir()
        # #     p_inf = self.inf_norm(self.ps.dx)
        # #
        # #     opt_itr += 1
        # #     # if p_inf < 1e-4 * self.dt[None]:
        # #     #     break
        # #
        # #     alpha = self.filter_step_size()
        # #     # print(alpha)
        # #     self.update_x(1)
        # #     self.enforce_boundary_3D(self.ps.material_fluid)
        # #
        # # print(opt_itr)
        #
        # for _ in range(10):
        #     self.compute_densities()
        #     # self.compute_DFSPH_factor()
        #     self.compute_pressure_forces()
        #     self.enforce_boundary_3D(self.ps.material_fluid)
        #
        # self.compute_velocity()

        self.ps.initialize_particle_system()
        self.compute_moving_boundary_volume()
        self.substep()
        self.solve_rigid_body()
        if self.ps.dim == 2:
            self.enforce_boundary_2D(self.ps.material_fluid)
        elif self.ps.dim == 3:
            self.enforce_boundary_3D(self.ps.material_fluid)
