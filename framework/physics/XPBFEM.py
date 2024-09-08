import csv
import taichi as ti
import numpy as np
from fontTools.voltLib.ast import Range


@ti.data_oriented
class Solver:
    def __init__(self,
                 mesh_dy,
                 mesh_st=None,
                 g=ti.math.vec3(0.0, -9.81, 0.0),
                 YM=1e5,
                 PR=0.2,
                 dHat=1e-4,
                 dt=0.03):

        self.mesh_dy = mesh_dy
        self.mesh_st = mesh_st
        self.g = g
        self.YM = YM
        self.PR = PR
        self.dHat = dHat
        self.dt = dt

        self.damping = 0.00
        self.padding = 0.05
        self.solver_type = 0
        self.enable_velocity_update = False
        self.export_mesh = False

        self.y = self.mesh_dy.y
        self.x = self.mesh_dy.x
        self.dx = self.mesh_dy.dx
        # self.hii = self.tet_mesh.hii
        self.nc = self.mesh_dy.nc
        self.v = self.mesh_dy.v

        self.num_verts_dy = self.mesh_dy.x.shape[0]

        self.M = self.mesh_dy.M
        self.invM = self.mesh_dy.invM
        self.fixed = self.mesh_dy.fixed
        self.invDm = self.mesh_dy.invDm
        self.V0 = self.mesh_dy.V0

        self.faces = self.mesh_dy.surface_indices
        self.tetras = self.mesh_dy.tet_indices

        self.num_tets = self.mesh_dy.tet_indices.shape[0]

        self.bd_max = ti.math.vec3(40.0)
        self.bd_min = -self.bd_max

        self.aabb_x0 = ti.Vector.field(n=3, dtype=float, shape=8)
        self.aabb_index0 = ti.field(dtype=int, shape=24)
        self.init_grid(self.bd_min, self.bd_max)
        # self.reset()
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


    @ti.kernel
    def set_fixed_vertices(self, fixed_vertices: ti.template()):
        for i in range(self.num_verts_dy):
            if fixed_vertices[i] >= 1:
                self.fixed[i] = 0.0
            else:
                self.fixed[i] = 1.0

    def reset(self):
        self.mesh_dy.reset()
        # self.search_neighbours_rest()
        # self.init_V0_and_L()

    @ti.func
    def confine_boundary(self, p):
        boundary_min = self.bd_min + self.padding
        boundary_max = self.bd_max - self.padding

        for i in ti.static(range(3)):
            if p[i] <= boundary_min[i]:
                p[i] = boundary_min[i] + 1e-4 * ti.random()
            elif boundary_max[i] <= p[i]:
                p[i] = boundary_max[i] - 1e-4 * ti.random()

        return p

    @ti.kernel
    def compute_y(self, dt: float):

        # ti.block_local(self.m_inv_p, self.v, self.x, self.y)
        for i in self.y:
            # if self.m_inv_p[i] > 0.0:
            self.y[i] = self.mesh_dy.x[i] + (self.v[i] * dt + self.g * dt * dt)
            self.y[i] = self.confine_boundary(self.y[i])

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
    def solve_xpbd_fem_stretch_x(self, compliance_str: float):
        self.dx.fill(0.0)
        self.nc.fill(0.0)

        ti.block_local(self.invDm, self.dx, self.nc)
        for i in self.invDm:
            Ds = ti.Matrix.cols([self.y[self.tetras[i, j]] - self.y[self.tetras[i, 3]] for j in ti.static(range(3))])
            B = self.invDm[i]
            F = Ds @ B
            U, _, V = self.ssvd(F)
            R = U @ V.transpose()
            P = F - R
            test = (P.transpose() @ P).trace()
            C = ti.sqrt(test)
            # print(C)
            eps = 0.0
            if C < 1e-3:
                eps = 1e-3
            dCdx = (F - R) @ B.transpose() / (C + eps)

            grad0 = ti.math.vec3(dCdx[0, 0], dCdx[1, 0], dCdx[2, 0])
            grad1 = ti.math.vec3(dCdx[0, 1], dCdx[1, 1], dCdx[2, 1])
            grad2 = ti.math.vec3(dCdx[0, 2], dCdx[1, 2], dCdx[2, 2])
            grad3 = -(grad0 + grad1 + grad2)

            weight = self.V0[i] * compliance_str

            schur = (self.invM[self.tetras[i, 0]] * grad0.dot(grad0) +
                     self.invM[self.tetras[i, 1]] * grad1.dot(grad1) +
                     self.invM[self.tetras[i, 2]] * grad2.dot(grad2) +
                     self.invM[self.tetras[i, 3]] * grad3.dot(grad3))

            ld = -(weight * C) / (weight * schur + 1.0)
            # print(ld)

            self.dx[self.tetras[i, 0]] += self.invM[self.tetras[i, 0]] * ld * grad0
            self.dx[self.tetras[i, 1]] += self.invM[self.tetras[i, 1]] * ld * grad1
            self.dx[self.tetras[i, 2]] += self.invM[self.tetras[i, 2]] * ld * grad2
            self.dx[self.tetras[i, 3]] += self.invM[self.tetras[i, 3]] * ld * grad3

            self.nc[self.tetras[i, 0]] += 1.0
            self.nc[self.tetras[i, 1]] += 1.0
            self.nc[self.tetras[i, 2]] += 1.0
            self.nc[self.tetras[i, 3]] += 1.0

        ti.block_local(self.y, self.dx, self.nc)
        for i in self.dx:
            self.y[i] += (self.dx[i] / self.nc[i])

    @ti.kernel
    def solve_pd_fem_stretch_x(self, compliance_str: float):
        # print("fuck")
        self.dx.fill(0.0)
        self.nc.fill(0.0)

        ti.block_local(self.invDm, self.dx, self.nc)
        for i in self.invDm:
            Ds = ti.Matrix.cols([self.y[self.tetras[i, j]] - self.y[self.tetras[i, 3]] for j in ti.static(range(3))])
            B = self.invDm[i]
            F = Ds @ B
            U, _, V = self.ssvd(F)
            R = U @ V.transpose()
            P = F - R
            test = (P.transpose() @ P).trace()
            C = ti.sqrt(test)
            # print(C)
            eps = 0.0
            if C < 1e-3:
                eps = 1e-3
            dCdx = (F - R) @ B.transpose() / (C + eps)

            grad0 = ti.math.vec3(dCdx[0, 0], dCdx[1, 0], dCdx[2, 0])
            grad1 = ti.math.vec3(dCdx[0, 1], dCdx[1, 1], dCdx[2, 1])
            grad2 = ti.math.vec3(dCdx[0, 2], dCdx[1, 2], dCdx[2, 2])
            grad3 = -(grad0 + grad1 + grad2)

            weight = self.V0[i] * compliance_str

            schur = (self.invM[self.tetras[i, 0]] * grad0.dot(grad0) +
                     self.invM[self.tetras[i, 1]] * grad1.dot(grad1) +
                     self.invM[self.tetras[i, 2]] * grad2.dot(grad2) +
                     self.invM[self.tetras[i, 3]] * grad3.dot(grad3))

            ld = -(weight * C) / (weight * schur + 1.0)
            # print(ld)

            self.dx[self.tetras[i, 0]] += self.invM[self.tetras[i, 0]] * ld * grad0
            self.dx[self.tetras[i, 1]] += self.invM[self.tetras[i, 1]] * ld * grad1
            self.dx[self.tetras[i, 2]] += self.invM[self.tetras[i, 2]] * ld * grad2
            self.dx[self.tetras[i, 3]] += self.invM[self.tetras[i, 3]] * ld * grad3

            self.nc[self.tetras[i, 0]] += 1.0
            self.nc[self.tetras[i, 1]] += 1.0
            self.nc[self.tetras[i, 2]] += 1.0
            self.nc[self.tetras[i, 3]] += 1.0

        ti.block_local(self.y, self.dx, self.nc)
        for i in self.dx:
            self.y[i] += (self.dx[i] / self.nc[i])

    @ti.func
    def volume_projection(self, sigma: ti.math.mat3) -> ti.math.mat3:

        singular_values = ti.math.vec3(sigma[0, 0], sigma[1, 1], sigma[2, 2])
        nabla_c = ti.math.vec3(0.0)
        # print(singular_values)
        count = 0
        count_max = 20
        threshold = 1e-4
        c = singular_values[0] * singular_values[1] * singular_values[2] - 1.0
        while abs(c) > threshold:

            if count > count_max: break
            c = singular_values[0] * singular_values[1] * singular_values[2] - 1.0
            nabla_c[0] = singular_values[1] * singular_values[2]
            nabla_c[1] = singular_values[0] * singular_values[2]
            nabla_c[2] = singular_values[0] * singular_values[1]
            ld = c / (nabla_c.dot(nabla_c) + 1e-3)
            singular_values -= ld * nabla_c

            singular_values = ti.math.clamp(singular_values, 0.0, 3.0)

            count += 1

        for i in ti.static(range(3)):
            sigma[i, i] = singular_values[i]

        return sigma

    @ti.kernel
    def solve_pd_fem_volume_x(self, compliance_vol: float):

        self.dx.fill(0.0)
        self.nc.fill(0.0)

        ti.block_local(self.invDm, self.dx, self.nc)
        for i in self.invDm:
            Ds = ti.Matrix.cols([self.y[self.tetras[i, j]] - self.y[self.tetras[i, 3]] for j in ti.static(range(3))])
            B = self.invDm[i]
            F = Ds @ B

            f1 = ti.math.vec3(F[0, 0], F[1, 0], F[2, 0])
            f2 = ti.math.vec3(F[0, 1], F[1, 1], F[2, 1])
            f3 = ti.math.vec3(F[0, 2], F[1, 2], F[2, 2])

            J = (f1.cross(f2)).dot(f3)
            C = J - 1.0
            # print(C)
            dCdx = ti.Matrix.cols([f2.cross(f3), f3.cross(f1), f1.cross(f2)]) @ B.transpose()

            grad0 = ti.math.vec3(dCdx[0, 0], dCdx[1, 0], dCdx[2, 0])
            grad1 = ti.math.vec3(dCdx[0, 1], dCdx[1, 1], dCdx[2, 1])
            grad2 = ti.math.vec3(dCdx[0, 2], dCdx[1, 2], dCdx[2, 2])
            grad3 = -(grad0 + grad1 + grad2)

            weight = self.V0[i] * compliance_vol

            schur = (self.invM[self.tetras[i, 0]] * grad0.dot(grad0) +
                     self.invM[self.tetras[i, 1]] * grad1.dot(grad1) +
                     self.invM[self.tetras[i, 2]] * grad2.dot(grad2) +
                     self.invM[self.tetras[i, 3]] * grad3.dot(grad3))

            ld = -(weight * C) / (weight * schur + 1.0)
            # print(ld)

            self.dx[self.tetras[i, 0]] += self.invM[self.tetras[i, 0]] * ld * grad0
            self.dx[self.tetras[i, 1]] += self.invM[self.tetras[i, 1]] * ld * grad1
            self.dx[self.tetras[i, 2]] += self.invM[self.tetras[i, 2]] * ld * grad2
            self.dx[self.tetras[i, 3]] += self.invM[self.tetras[i, 3]] * ld * grad3

            self.nc[self.tetras[i, 0]] += 1.0
            self.nc[self.tetras[i, 1]] += 1.0
            self.nc[self.tetras[i, 2]] += 1.0
            self.nc[self.tetras[i, 3]] += 1.0

        ti.block_local(self.y, self.dx, self.nc)
        for i in self.dx:
            self.y[i] += (self.dx[i] / self.nc[i])

    @ti.kernel
    def solve_xpbd_fem_volume_x(self, compliance_vol: float):
        self.dx.fill(0.0)
        self.nc.fill(0.0)

        ti.block_local(self.invDm, self.dx, self.nc)
        for i in self.invDm:
            Ds = ti.Matrix.cols([self.y[self.tetras[i, j]] - self.y[self.tetras[i, 3]] for j in ti.static(range(3))])
            B = self.invDm[i]
            F = Ds @ B

            f1 = ti.math.vec3(F[0, 0], F[1, 0], F[2, 0])
            f2 = ti.math.vec3(F[0, 1], F[1, 1], F[2, 1])
            f3 = ti.math.vec3(F[0, 2], F[1, 2], F[2, 2])

            J = (f1.cross(f2)).dot(f3)
            C = J - 1.0
            # print(C)
            dCdx = ti.Matrix.cols([f2.cross(f3), f3.cross(f1), f1.cross(f2)]) @ B.transpose()

            grad0 = ti.math.vec3(dCdx[0, 0], dCdx[1, 0], dCdx[2, 0])
            grad1 = ti.math.vec3(dCdx[0, 1], dCdx[1, 1], dCdx[2, 1])
            grad2 = ti.math.vec3(dCdx[0, 2], dCdx[1, 2], dCdx[2, 2])
            grad3 = -(grad0 + grad1 + grad2)

            weight = self.V0[i] * compliance_vol

            schur = (self.invM[self.tetras[i, 0]] * grad0.dot(grad0) +
                     self.invM[self.tetras[i, 1]] * grad1.dot(grad1) +
                     self.invM[self.tetras[i, 2]] * grad2.dot(grad2) +
                     self.invM[self.tetras[i, 3]] * grad3.dot(grad3))

            ld = -(weight * C) / (weight * schur + 1.0)
            # print(ld)

            self.dx[self.tetras[i, 0]] += self.invM[self.tetras[i, 0]] * ld * grad0
            self.dx[self.tetras[i, 1]] += self.invM[self.tetras[i, 1]] * ld * grad1
            self.dx[self.tetras[i, 2]] += self.invM[self.tetras[i, 2]] * ld * grad2
            self.dx[self.tetras[i, 3]] += self.invM[self.tetras[i, 3]] * ld * grad3

            self.nc[self.tetras[i, 0]] += 1.0
            self.nc[self.tetras[i, 1]] += 1.0
            self.nc[self.tetras[i, 2]] += 1.0
            self.nc[self.tetras[i, 3]] += 1.0

        ti.block_local(self.y, self.dx, self.nc)
        for i in self.dx:
            self.y[i] += (self.dx[i] / self.nc[i])

    @ti.kernel
    def update_state(self, damping: float, dt: float):

        # ti.block_local(self.m_inv_p, self.v, self.x, self.y)
        for i in self.y:
            new_x = self.confine_boundary(self.y[i])
            self.v[i] = (1.0 - damping) * (new_x - self.x[i]) / dt
            self.x[i] += self.v[i] * dt

    def solve_constraints_jacobi(self, dt):

        dtSq = dt ** 2
        mu = self.YM / 2.0 / (1.0 + self.PR)
        ld = (self.YM * self.PR) / ((1.0 + self.PR) * (1.0 - 2.0 * self.PR))

        compliance_str = mu * dtSq
        self.solve_xpbd_fem_stretch_x(compliance_str)

        compliance_vol = ld * dtSq
        self.solve_xpbd_fem_volume_x(compliance_vol)

    def solve_PD_diag(self, dt):

        dtSq = dt ** 2
        mu = self.YM / 2.0 / (1.0 + self.PR)
        ld = (self.YM * self.PR) / ((1.0 + self.PR) * (1.0 - 2.0 * self.PR))

        compliance_str = mu * dtSq
        self.solve_pd_fem_stretch_x(compliance_str)

        compliance_vol = ld * dtSq
        self.solve_pd_fem_volume_x(compliance_vol)

    def forward(self, n_substeps, n_iter):

        dt_sub = self.dt / n_substeps
        for _ in range(n_substeps):
            self.compute_y(dt_sub)
            for _ in range(n_iter):
                if self.solver_type == 0:
                    self.solve_constraints_jacobi(dt_sub)
                elif self.solver_type == 1:
                    self.solve_PD_diag(dt_sub)

                self.update_state(self.damping, dt_sub)

