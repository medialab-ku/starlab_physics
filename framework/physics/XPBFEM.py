import csv
import taichi as ti
import numpy as np

@ti.data_oriented
class Solver:
    def __init__(self,
                 tet_mesh,
                 mesh_st=None,
                 g=ti.math.vec3(0.0, -9.81, 0.0),
                 YM=1e5,
                 PR=0.2,
                 dHat=1e-4,
                 dt=0.03):

        self.tet_mesh = tet_mesh
        self.mesh_st = mesh_st
        self.g = g
        self.YM = YM
        self.PR = PR
        self.dHat = dHat
        self.dt = dt

        self.damping = 0.001
        self.padding = 0.05
        self.solver_type = 0
        self.enable_velocity_update = False
        self.export_mesh = False

        self.y = self.tet_mesh.y
        self.x = self.tet_mesh.x
        self.dx = self.tet_mesh.dx
        # self.hii = self.tet_mesh.hii
        self.nc = self.tet_mesh.nc
        self.v = self.tet_mesh.v

        self.num_verts = self.tet_mesh.x.shape[0]

        self.M = self.tet_mesh.M
        self.invM = self.tet_mesh.invM
        self.invDm = self.tet_mesh.invDm
        self.V0 = self.tet_mesh.V0

        self.faces = self.tet_mesh.surface_indices
        self.tetras = self.tet_mesh.tet_indices

        self.num_tets = self.tet_mesh.tet_indices.shape[0]

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


    def reset(self):
        self.tet_mesh.reset()
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
            self.v[i] = self.v[i] + self.g * dt
            self.y[i] = self.x[i] + self.v[i] * dt
            # else:
            #     self.y[i] = self.x[i]

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

            Dm = ti.math.inverse(B)
            Ds_proj = R @ Dm

            proj03 = self.M[self.tetras[i, 0]] * ti.math.vec3(Ds_proj[0, 0], Ds_proj[1, 0], Ds_proj[2, 0])
            proj13 = self.M[self.tetras[i, 1]] * ti.math.vec3(Ds_proj[0, 1], Ds_proj[1, 1], Ds_proj[2, 1])
            proj23 = self.M[self.tetras[i, 2]] * ti.math.vec3(Ds_proj[0, 2], Ds_proj[1, 2], Ds_proj[2, 2])

            com = ti.math.vec3(0.0)
            m = 0.0

            for j in ti.static(range(4)):
                com += self.M[self.tetras[i, j]] * self.y[self.tetras[i, j]]
                m += self.M[self.tetras[i, j]]

            com /= m

            proj3 = com - (proj03 + proj13 + proj23) / m
            proj0 = proj03 / self.M[self.tetras[i, 0]] + proj3
            proj1 = proj13 / self.M[self.tetras[i, 1]] + proj3
            proj2 = proj23 / self.M[self.tetras[i, 2]] + proj3

            self.dx[self.tetras[i, 0]] += (proj0 - self.y[self.tetras[i, 0]])
            self.dx[self.tetras[i, 1]] += (proj1 - self.y[self.tetras[i, 1]])
            self.dx[self.tetras[i, 2]] += (proj2 - self.y[self.tetras[i, 2]])
            self.dx[self.tetras[i, 3]] += (proj3 - self.y[self.tetras[i, 3]])

            self.nc[self.tetras[i, 0]] += 1.0
            self.nc[self.tetras[i, 1]] += 1.0
            self.nc[self.tetras[i, 2]] += 1.0
            self.nc[self.tetras[i, 3]] += 1.0

        for i in self.dx:
            self.y[i] += (self.dx[i] / self.nc[i])

    @ti.kernel
    def solve_pd_fem_stretch_x(self, compliance_str: float):

        self.dx.fill(0.0)
        self.nc.fill(1.0)

        ti.block_local(self.invDm, self.dx, self.nc)
        for i in self.invDm:
            Ds = ti.Matrix.cols([self.y[self.tetras[i, j]] - self.y[self.tetras[i, 3]] for j in ti.static(range(3))])
            B = self.invDm[i]
            F = Ds @ B
            U, _, V = self.ssvd(F)
            R = U @ V.transpose()

            Dm = ti.math.inverse(B)
            Ds_proj = R @ Dm

            proj03 = ti.math.vec3(Ds_proj[0, 0], Ds_proj[1, 0], Ds_proj[2, 0])
            proj13 = ti.math.vec3(Ds_proj[0, 1], Ds_proj[1, 1], Ds_proj[2, 1])
            proj23 = ti.math.vec3(Ds_proj[0, 2], Ds_proj[1, 2], Ds_proj[2, 2])

            weight = self.V0[i] * compliance_str
            self.dx[self.tetras[i, 0]] += self.invM[self.tetras[i, 0]] * weight * (proj03 + self.y[self.tetras[i, 3]] - self.y[self.tetras[i, 0]])
            self.dx[self.tetras[i, 1]] += self.invM[self.tetras[i, 1]] * weight * (proj13 + self.y[self.tetras[i, 3]] - self.y[self.tetras[i, 1]])
            self.dx[self.tetras[i, 2]] += self.invM[self.tetras[i, 2]] * weight * (proj23 + self.y[self.tetras[i, 3]] - self.y[self.tetras[i, 2]])

            # self.dx[self.tetras[i, 3]] += self.invM[self.tetras[i, 3]] * weight * (proj3 - self.y[self.tetras[i, 3]])
            self.dx[self.tetras[i, 3]] -= self.invM[self.tetras[i, 3]] * weight * (proj03 + proj13 + proj23 + 3 * self.y[self.tetras[i, 3]] - self.y[self.tetras[i, 0]] - self.y[self.tetras[i, 1]] - self.y[self.tetras[i, 2]])

            self.nc[self.tetras[i, 0]] += self.invM[self.tetras[i, 0]] * weight
            self.nc[self.tetras[i, 1]] += self.invM[self.tetras[i, 1]] * weight
            self.nc[self.tetras[i, 2]] += self.invM[self.tetras[i, 2]] * weight
            self.nc[self.tetras[i, 3]] += 3 * self.invM[self.tetras[i, 3]] * weight

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
        self.nc.fill(1.0)

        ti.block_local(self.invDm, self.dx, self.nc)
        for i in self.invDm:
            Ds = ti.Matrix.cols([self.y[self.tetras[i, j]] - self.y[self.tetras[i, 3]] for j in ti.static(range(3))])
            B = self.invDm[i]
            F = Ds @ B
            U, sigma, V = self.ssvd(F)
            sigma_proj = self.volume_projection(sigma)
            R = U @ sigma_proj @ V.transpose()

            Dm = ti.math.inverse(B)
            Ds_proj = R @ Dm

            proj03 = ti.math.vec3(Ds_proj[0, 0], Ds_proj[1, 0], Ds_proj[2, 0])
            proj13 = ti.math.vec3(Ds_proj[0, 1], Ds_proj[1, 1], Ds_proj[2, 1])
            proj23 = ti.math.vec3(Ds_proj[0, 2], Ds_proj[1, 2], Ds_proj[2, 2])

            weight = self.V0[i] * compliance_vol
            self.dx[self.tetras[i, 0]] += self.invM[self.tetras[i, 0]] * weight * (proj03 + self.y[self.tetras[i, 3]] - self.y[self.tetras[i, 0]])
            self.dx[self.tetras[i, 1]] += self.invM[self.tetras[i, 1]] * weight * (proj13 + self.y[self.tetras[i, 3]] - self.y[self.tetras[i, 1]])
            self.dx[self.tetras[i, 2]] += self.invM[self.tetras[i, 2]] * weight * (proj23 + self.y[self.tetras[i, 3]] - self.y[self.tetras[i, 2]])

            # self.dx[self.tetras[i, 3]] += self.invM[self.tetras[i, 3]] * weight * (proj3 - self.y[self.tetras[i, 3]])
            self.dx[self.tetras[i, 3]] -= self.invM[self.tetras[i, 3]] * weight * (proj03 + proj13 + proj23 + 3 * self.y[self.tetras[i, 3]] - self.y[self.tetras[i, 0]] - self.y[self.tetras[i, 1]] - self.y[self.tetras[i, 2]])

            self.nc[self.tetras[i, 0]] += self.invM[self.tetras[i, 0]] * weight
            self.nc[self.tetras[i, 1]] += self.invM[self.tetras[i, 1]] * weight
            self.nc[self.tetras[i, 2]] += self.invM[self.tetras[i, 2]] * weight
            self.nc[self.tetras[i, 3]] += 3 * self.invM[self.tetras[i, 3]] * weight

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
            U, _, V = self.ssvd(F)
            R = U @ V.transpose()

            Dm = ti.math.inverse(B)
            Ds_proj = R @ Dm

            proj03 = self.M[self.tetras[i, 0]] * ti.math.vec3(Ds_proj[0, 0], Ds_proj[1, 0], Ds_proj[2, 0])
            proj13 = self.M[self.tetras[i, 1]] * ti.math.vec3(Ds_proj[0, 1], Ds_proj[1, 1], Ds_proj[2, 1])
            proj23 = self.M[self.tetras[i, 2]] * ti.math.vec3(Ds_proj[0, 2], Ds_proj[1, 2], Ds_proj[2, 2])

            com = ti.math.vec3(0.0)
            m = 0.0

            for j in ti.static(range(4)):
                com += self.M[self.tetras[i, j]] * self.y[self.tetras[i, j]]
                m += self.M[self.tetras[i, j]]

            com /= m

            proj3 = com - (proj03 + proj13 + proj23) / m
            proj0 = proj03 / self.M[self.tetras[i, 0]] + proj3
            proj1 = proj13 / self.M[self.tetras[i, 1]] + proj3
            proj2 = proj23 / self.M[self.tetras[i, 2]] + proj3

            self.dx[self.tetras[i, 0]] += (proj0 - self.y[self.tetras[i, 0]])
            self.dx[self.tetras[i, 1]] += (proj1 - self.y[self.tetras[i, 1]])
            self.dx[self.tetras[i, 2]] += (proj2 - self.y[self.tetras[i, 2]])
            self.dx[self.tetras[i, 3]] += (proj3 - self.y[self.tetras[i, 3]])

            self.nc[self.tetras[i, 0]] += 1.0
            self.nc[self.tetras[i, 1]] += 1.0
            self.nc[self.tetras[i, 2]] += 1.0
            self.nc[self.tetras[i, 3]] += 1.0

        for i in self.dx:
            self.y[i] += (self.dx[i] / self.nc[i])
            # nabla_Ci0 = ti.math.vec3(H[0, 0])

    @ti.kernel
    def update_state(self, damping: float, dt: float):

        # ti.block_local(self.m_inv_p, self.v, self.x, self.y)
        for i in self.y:
            new_x = self.confine_boundary(self.y[i])
            self.v[i] = (1.0 - damping) * (new_x - self.x[i]) / dt
            self.x[i] = new_x

    def solve_constraints_jacobi(self, dt):

        dtSq = dt ** 2
        mu = self.YM / 2.0 * (1.0 + self.PR)
        ld = (self.YM * self.PR) / ((1.0 + self.PR) * (1.0 - 2.0 * self.PR))

        compliance_str = 2.0 * mu * dtSq
        self.solve_xpbd_fem_stretch_x(compliance_str)
        #
        # compliance_vol = ld * dtSq
        # self.solve_constraints_fem_volume_x(compliance_vol)

    def solve_PD_diag(self, dt):

        dtSq = dt ** 2
        mu = self.YM / 2.0 * (1.0 + self.PR)
        ld = (self.YM * self.PR) / ((1.0 + self.PR) * (1.0 - 2.0 * self.PR))

        compliance_str = 2.0 * mu * dtSq
        self.solve_pd_fem_stretch_x(compliance_str)
        compliance_vol = ld * dtSq
        self.solve_pd_fem_volume_x(compliance_vol)

    def forward(self, n_substeps):

        dt_sub = self.dt / n_substeps
        for _ in range(n_substeps):
            self.compute_y(dt_sub)

            if self.solver_type == 0:
                self.solve_constraints_jacobi(dt_sub)
            elif self.solver_type == 1:
                self.solve_PD_diag(dt_sub)

            self.update_state(self.damping, dt_sub)

