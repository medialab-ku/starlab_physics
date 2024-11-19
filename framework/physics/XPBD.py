import taichi as ti
from pandas.core.ops.mask_ops import raise_for_nan

from framework.physics.conjugate_gradient import ConjugateGradient
@ti.data_oriented
class Solver:
    def __init__(
            self,
            mesh_dy,
            # mesh_st,
            particle_st,
            dHat,
            sh_st,
            sh_dy,
            # sh_st_e,
            stiffness_stretch,
            stiffness_bending,
            g,
            dt):

        self.mesh_dy = mesh_dy
        self.particle_st = particle_st
        # self.mesh_st = mesh_st
        self.dHat = dHat
        self.sh_dy = sh_dy
        self.sh_st = sh_st
        # self.sh_st_e = sh_st_e
        self.stiffness_stretch = stiffness_stretch
        self.stiffness_bending = stiffness_bending
        self.g = g
        self.dt = dt

        self.conv_iter = 0
        self.damping = 0.001
        self.threshold = 1e-4
        self.max_cg_iter = 100

        self.mu = 0.8
        self.PCG = ConjugateGradient()
        self.definiteness_fix = True
        self.print_stats = False
        self.enable_line_search = True
        self.enable_velocity_update = True
        self.enable_pncg = True

        self.num_verts_dy = self.mesh_dy.num_verts
        self.num_edges_dy = self.mesh_dy.num_edges
        self.num_faces_dy = self.mesh_dy.num_faces
        self.selected_solver_type = 2
        self.selected_precond_type = 2


        self.reset()


        self.a = self.mesh_dy.a_dup
        self.b = self.mesh_dy.b_dup
        self.c = self.mesh_dy.c_dup
        self.c_tilde = self.mesh_dy.c_dup_tilde

        self.a_1d = self.mesh_dy.a_dup_1d
        self.b_1d = self.mesh_dy.b_dup_1d
        self.c_1d = self.mesh_dy.c_dup_1d
        self.c_tilde_1d = self.mesh_dy.c_dup_tilde_1d

        self.d = self.mesh_dy.d_dup
        self.d_tilde = self.mesh_dy.d_dup_tilde


        self.compute_duplicates = True

        self.E_curr = 0.0
        self.E_max = 0.0
        self.E_min = 0.0

    ####################################################################################################################

    @ti.kernel
    def set_fixed_vertices(self, fixed_vertices: ti.template()):
        for i in range(self.num_verts_dy):
            if fixed_vertices[i] >= 1:
                self.mesh_dy.fixed[i] = 0.0
            else:
                self.mesh_dy.fixed[i] = 1.0




    def reset(self):
        print("reset...")
        self.mesh_dy.reset()
        # self.x_pbd_jacobi.copy_from(self.mesh_dy.x)
        self.compute_duplicates = True
        self.copy_to_dup()




    def init_variables(self):
        # initialize dx and number of constraints (used in the Jacobi solver)
        self.mesh_dy.p.fill(0.0)
        self.mesh_dy.nc.fill(0.0)
        self.mesh_dy.dx_euler.fill(0.0)

    @ti.func
    def confine_boundary(self, p):
        boundary_min = self.sh_st.bbox_min + self.dHat
        boundary_max = self.sh_st.bbox_max - self.dHat

        for i in ti.static(range(3)):
            if p[i] <= boundary_min[i]:
                p[i] = boundary_min[i] + 1e-4 * ti.random()
            elif boundary_max[i] <= p[i]:
                p[i] = boundary_max[i] - 1e-4 * ti.random()

        return p

    @ti.kernel
    def compute_y(self, g: ti.math.vec3, dt: ti.f32):
        # compute apporximate x_(t+1) (== y) by explicit way before projecting constraints to x_(t+1)...
        for i in range(self.num_verts_dy):
            self.mesh_dy.x_k[i] = (self.mesh_dy.x[i] + (self.mesh_dy.v[i] * dt + g * dt * dt))
            self.mesh_dy.y_tilde[i] = self.mesh_dy.x_k[i]



    @ti.kernel
    def compute_v(self, damping: ti.f32, dt: ti.f32):
        # compute v after all constraints are projected to x_(t+1)
        for i in range(self.num_verts_dy):
            self.mesh_dy.v[i] = (1.0 - damping) * (self.mesh_dy.x_k[i] - self.mesh_dy.x[i]) / dt


    @ti.kernel
    def copy_to_dup(self):

        for di in self.mesh_dy.x_dup:
            vi = self.mesh_dy.dup_to_ori[di]
            self.mesh_dy.x_dup[di] = self.mesh_dy.x[vi]

    @ti.kernel
    def update_x(self, dt: float):
        # eventually, update actual position x_(t+1) after velocities are computed...
        for i in range(self.num_verts_dy):
            self.mesh_dy.x[i] += dt * self.mesh_dy.v[i]


    def solve_constraints_pd_diag_x(self, dt):
        compliance_stretch = self.stiffness_stretch * dt * dt
        compliance_bending = self.stiffness_bending * dt * dt
        self.mesh_dy.nc.copy_from(self.mesh_dy.m)
        E_curr = self.solve_spring_constraints_pd_diag_x(compliance_stretch, compliance_bending)
        # self.solve_spring_constraints_hess_diag_x(compliance_stretch, compliance_bending)

        # return E_curr

    def solve_constraints_hess_diag_x(self, dt):
        compliance_stretch = self.stiffness_stretch * dt * dt
        compliance_bending = self.stiffness_bending * dt * dt

        self.solve_spring_constraints_hess_diag_x(compliance_stretch, compliance_bending)

    @ti.kernel
    def line_search(self, E_current: float) -> float:

        alpha = 1.0

        return alpha

    def solve_constraints_newton_pcg_x(self, dt, max_cg_iter, threshold):

        compliance_stretch = self.stiffness_stretch * dt * dt
        compliance_bending = self.stiffness_bending * dt * dt

        E_curr = self.compute_grad_and_hess_spring_x(compliance_stretch, compliance_bending, self.definiteness_fix)

        self.PCG.run(self.selected_precond_type, self.mesh_dy, max_cg_iter, threshold)
        alpha = 1.0
        # if self.use_line_search:
        #     self.PCG.compute_mat_free_Ax(self.mesh_dy, self.mesh_dy.Ax, self.mesh_dy.dx)
        #     test = self.PCG.dot_product(self.mesh_dy.dx, self.mesh_dy.Ax)
        #     if test > 1e-3:
        #         alpha = self.PCG.dot_product(self.mesh_dy.b, self.mesh_dy.dx) / test
        #     else:
        #         alpha = 1.0

        # print( self.PCG.cg_iter)

        # if self.print_stats:
        #     print("CG err: ", r_sq)
        #     print("CG iter: ", cg_iter)
        #
        # if self.print_stats and self.use_line_search:
        #     print("alpha: ", alpha)

        self.PCG.vector_add(self.mesh_dy.x_k, self.mesh_dy.x_k, self.mesh_dy.p, alpha)



    @ti.kernel
    def solve_spring_constraints_euler_path_tridiagonal_x(self,
                                                          compliance_stretch: ti.f32,
                                                          compliance_bending: ti.f32
                                                          ):

        id3 = ti.Matrix.identity(dt=float, n=3)

        compliance_attach =1e5
        for i in self.mesh_dy.p:
            self.mesh_dy.p[i] = self.mesh_dy.m[i] * (self.mesh_dy.y_tilde[i] - self.mesh_dy.x_k[i])
            self.mesh_dy.hii[i] = self.mesh_dy.m[i] * id3

            if self.mesh_dy.fixed[i] < 1.0:
                self.mesh_dy.p[i] += compliance_attach * (self.mesh_dy.x0[i] - self.mesh_dy.x_k[i])
                self.mesh_dy.hii[i] += compliance_attach * id3

        for i in range(self.num_edges_dy):
            l0 = self.mesh_dy.l0[i]
            v0_d, v1_d = self.mesh_dy.eid_dup[2 * i + 0], self.mesh_dy.eid_dup[2 * i + 1]
            v0, v1 = self.mesh_dy.dup_to_ori[v0_d], self.mesh_dy.dup_to_ori[v1_d]
            x01 = self.mesh_dy.x_k[v0] - self.mesh_dy.x_k[v1]
            l = x01.norm()
            n = x01.normalized()

            nnT = n.outer_product(n)
            alpha = (1.0 - l0 / l)
            alpha = 1.0
            B = compliance_stretch * (alpha * id3 + (1.0 - alpha) * nnT)
            # alpha = 0.5
            # self.mesh_dy.hij[i] = -compliance_stretch * (alpha * id3 + (1.0 - alpha) * n.outer_product(n))
            # self.mesh_dy.hij[i] = -compliance_stretch * id3

            self.c[v0_d] = -B
            self.a[v1_d] = -B

            dp01 = x01 - l0 * x01.normalized()

            self.mesh_dy.p[v0] -= compliance_stretch * dp01
            self.mesh_dy.p[v1] += compliance_stretch * dp01

            self.mesh_dy.hii[v0] += B
            self.mesh_dy.hii[v1] += B



        for di in self.mesh_dy.x_dup:
            vi = self.mesh_dy.dup_to_ori[di]

            self.b[di] = self.mesh_dy.hii[vi]
            self.d[di] = self.mesh_dy.p[vi]

        n_part = (self.mesh_dy.partition_offset.shape[0] - 1)
        for pi in range(n_part):

            size =  self.mesh_dy.vert_offset[pi + 1] - self.mesh_dy.vert_offset[pi]
            offset = self.mesh_dy.vert_offset[pi]

            # Thomas algorithm
            # https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm

            # for j in ti.static(range(3)):
            self.c_tilde[offset] = ti.math.inverse(self.b[offset]) @ self.c[offset]
            ti.loop_config(serialize=True)
            for id in range(size): # lb+1 ~ ub-1
                i = id + offset
                tmp = ti.math.inverse(self.b[i] - self.a[i] * self.c_tilde[i - 1])

                self.c_tilde[i] = tmp @ self.c[i]
            #
            self.d_tilde[offset] = ti.math.inverse(self.b[offset]) @ self.d[offset]
            ti.loop_config(serialize=True)
            for id in range(1, size): # lb+1 ~ ub
                i = id + offset
                tmp = ti.math.inverse(self.b[i] - self.a[i] * self.c_tilde[i - 1])
                self.d_tilde[i] = tmp @ (self.d[i] - self.a[i] @ self.d_tilde[i - 1])


            self.mesh_dy.dx_dup[offset + size - 1] = self.d_tilde[offset + size - 1]
            ti.loop_config(serialize=True)
            for i in range(0, size - 1):
                idx = size - 2 - i + offset # ub-1 ~ lb
                self.mesh_dy.dx_dup[idx] = self.d_tilde[idx] - self.c_tilde[idx] @ self.mesh_dy.dx_dup[idx + 1]

        self.mesh_dy.p.fill(0.0)
        for di in self.mesh_dy.x_dup:
            vi = self.mesh_dy.dup_to_ori[di]
            self.mesh_dy.p[vi] += self.mesh_dy.dx_dup[di]


        for i in self.mesh_dy.x_k:
            self.mesh_dy.x_k[i] += self.mesh_dy.p[i] / self.mesh_dy.num_dup[i]

    @ti.kernel
    def solve_spring_constraints_pd_diag_x(self, compliance_stretch: ti.f32, compliance_bending: ti.f32)->ti.f32:

        # self.mesh_dy.dx.fill(0.0)
        # self.mesh_dy.nc.fill(1.0)
        compliance_attach =1e5

        for i in range(self.num_verts_dy):
            self.mesh_dy.p[i] = self.mesh_dy.m[i] * (self.mesh_dy.y_tilde[i] - self.mesh_dy.x_k[i])

            if self.mesh_dy.fixed[i] < 1.0:
                self.mesh_dy.p[i] += compliance_attach * (self.mesh_dy.x0[i] - self.mesh_dy.x_k[i])
                self.mesh_dy.nc[i] += compliance_attach

        E_curr = 0.0
        # ti.loop_config(serialize=True)
        ti.block_local(self.mesh_dy.l0, self.mesh_dy.eid_field, self.mesh_dy.fixed, self.mesh_dy.m_inv)
        for i in range(self.num_edges_dy):
            l0 = self.mesh_dy.l0[i]
            v0_d, v1_d = self.mesh_dy.eid_dup[2 * i + 0], self.mesh_dy.eid_dup[2 * i + 1]
            v0, v1 = self.mesh_dy.dup_to_ori[v0_d], self.mesh_dy.dup_to_ori[v1_d]
            # v0, v1 = self.mesh_dy.eid_field[i, 0], self.mesh_dy.eid_field[i, 1]
            x01 = self.mesh_dy.x_k[v0] - self.mesh_dy.x_k[v1]
            l = x01.norm()
            dp01 = x01 - l0 * x01.normalized()

            E_curr += 0.5 * compliance_stretch * (l - l0) ** 2

            # alpha = (1.0 - l0 / x01.norm())

            self.mesh_dy.p[v0] -= compliance_stretch * dp01
            self.mesh_dy.p[v1] += compliance_stretch * dp01
            self.mesh_dy.nc[v0] += compliance_stretch
            self.mesh_dy.nc[v1] += compliance_stretch


        # for i in self.mesh_dy.bending_l0:
        #     l0 = self.mesh_dy.bending_l0[i]
        #     v0, v1 = self.mesh_dy.bending_indices[2 * i + 0], self.mesh_dy.bending_indices[2 * i + 1]
        #     x01 = self.mesh_dy.y[v0] - self.mesh_dy.y[v1]
        #     l = x01.norm()
        #     dp01 = x01 - l0 * x01.normalized()
        #     E_curr += 0.5 * compliance_bending * (l - l0) ** 2
        #     # alpha = (1.0 - l0 / x01.norm())
        #
        #     self.mesh_dy.dx[v0] -= self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] * compliance_bending * dp01
        #     self.mesh_dy.dx[v1] += self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1] * compliance_bending * dp01
        #
        #     self.mesh_dy.nc[v0] += self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] * compliance_bending
        #     self.mesh_dy.nc[v1] += self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1] * compliance_bending


        ti.block_local(self.mesh_dy.x_k, self.mesh_dy.p, self.mesh_dy.nc)
        for i in range(self.num_verts_dy):
            # if self.mesh_dy.nc[i] > 0:
            self.mesh_dy.x_k[i] += (self.mesh_dy.p[i] / self.mesh_dy.nc[i])

        return E_curr



    @ti.kernel
    def solve_spring_constraints_hess_diag_x(self, compliance_stretch: ti.f32, compliance_bending: ti.f32):

        self.mesh_dy.b.fill(0.0)
        self.mesh_dy.nc.fill(0.0)

        # id3 = ti.math.mat3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        # for i in range(self.num_verts_dy):
        #     self.mesh_dy.hii[i] = self.mesh_dy.m[i] * id3

        # ti.loop_config(serialize=True)
        ti.block_local(self.mesh_dy.l0, self.mesh_dy.eid_field, self.mesh_dy.fixed, self.mesh_dy.m_inv)
        for i in range(self.num_edges_dy):
            l0 = self.mesh_dy.l0[i]
            v0_d, v1_d = self.mesh_dy.eid_dup[2 * i + 0], self.mesh_dy.eid_dup[2 * i + 1]
            v0, v1 = self.mesh_dy.dup_to_ori[v0_d], self.mesh_dy.dup_to_ori[v1_d]
            x01 = self.mesh_dy.x_k[v0] - self.mesh_dy.x_k[v1]
            dp01 = x01 - l0 * x01.normalized()
            n = x01.normalized()
            alpha = (1.0 - l0 / x01.norm())
            # alpha = 0.0
            if alpha < 1e-2:
                alpha = 1e-2

            self.mesh_dy.b[v0] -= compliance_stretch * dp01
            self.mesh_dy.b[v1] += compliance_stretch * dp01

            self.mesh_dy.nc[v0] += compliance_stretch * alpha
            self.mesh_dy.nc[v1] += compliance_stretch * alpha

        ti.block_local(self.mesh_dy.x_k, self.mesh_dy.p, self.mesh_dy.nc)
        for i in range(self.num_verts_dy):
            # if self.mesh_dy.nc[i] > 0:
            self.mesh_dy.x_k[i] += self.mesh_dy.b[i] / (self.mesh_dy.m[i] + self.mesh_dy.nc[i])

    @ti.kernel
    def compute_grad_and_hess_spring_x(self, compliance_stretch: ti.f32, compliance_bending: ti.f32, definite_fix: bool)-> float:

        self.mesh_dy.b.fill(0.0)
        self.mesh_dy.hii_e.fill(0.0)

        E_cur = 0.0
        id3 = ti.math.mat3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        for i in range(self.num_verts_dy):
            # test = 1.0
            self.mesh_dy.hii[i] =self.mesh_dy.m[i] * id3
            if self.mesh_dy.fixed[i] < 1.0:
                test = 1e8
                self.mesh_dy.b[i]   += test * (self.mesh_dy.x0[i] - self.mesh_dy.x_k[i])
                self.mesh_dy.hii[i] += test * id3

        # ti.loop_config(serialize=True)
        for i in range(self.num_edges_dy):
            l0 = self.mesh_dy.l0[i]
            v0_d, v1_d = self.mesh_dy.eid_dup[2 * i + 0], self.mesh_dy.eid_dup[2 * i + 1]
            v0, v1 = self.mesh_dy.dup_to_ori[v0_d], self.mesh_dy.dup_to_ori[v1_d]
            x01 = self.mesh_dy.x_k[v0] - self.mesh_dy.x_k[v1]
            l = x01.norm()
            n = x01.normalized()
            dp01 = (l - l0) * n
            E_cur += 0.5 * compliance_stretch * (l - l0) ** 2
            alpha = 1.0 - l0 / x01.norm()
            if definite_fix and alpha < 1e-2:
                alpha = 1e-2

            # alpha = 1.0
            test = compliance_stretch * (alpha * id3 + (1.0 - alpha) * n.outer_product(n))
            self.mesh_dy.hij[i] = test

            self.c[v0_d] = -compliance_stretch * id3
            self.a[v1_d] = -compliance_stretch * id3


            self.mesh_dy.b[v0] -= compliance_stretch * dp01
            self.mesh_dy.b[v1] += compliance_stretch * dp01

            self.mesh_dy.hii_e[v0] += compliance_stretch * id3
            self.mesh_dy.hii_e[v1] += compliance_stretch * id3

        return E_cur

    @ti.kernel
    def compute_spring_energy(self, stretch_stiffness: ti.f32) -> float:
        E = 0.0
        # ti.loop_config(serialize=True)
        for i in range(self.num_edges_dy):
            l0 = self.mesh_dy.l0[i]
            v0_d, v1_d = self.mesh_dy.eid_dup[2 * i + 0], self.mesh_dy.eid_dup[2 * i + 1]
            v0, v1 = self.mesh_dy.dup_to_ori[v0_d], self.mesh_dy.dup_to_ori[v1_d]
            x01 = self.mesh_dy.x_k[v0] - self.mesh_dy.x_k[v1]
            l = x01.norm()
            E += 0.5 * stretch_stiffness * (l - l0) ** 2


        return E

    @ti.kernel
    def solve_collision_constraints_dy_pd_diag_test_x(self, kernel_radius: float, compliance_col: float):

        self.mesh_dy.p.fill(0.0)
        self.mesh_dy.nc.fill(1.0)
        self.mesh_dy.num_neighbours_dy.fill(0)

        for i in range(self.num_verts_dy):
            # if i < self.num_verts_dy:
            pi = i
            pos_i = self.mesh_dy.x_k[pi]
            cell_id = self.sh_dy.pos_to_cell_id(pos_i)
            for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                cell_to_check = cell_id + offs
                if self.sh_dy.is_in_grid(cell_to_check):
                    for j in range(self.sh_dy.num_particles_in_cell[cell_to_check]):
                        pj = self.sh_dy.particle_ids_in_cell[cell_to_check, j]
                        if pi == pj:
                            continue
                        #
                        if self.is_neighbour(pi, pj, self.mesh_dy.num_neighbours_rest, self.mesh_dy.neighbour_ids_rest) > 0:
                            continue

                        pos_j = self.mesh_dy.x_k[pj]
                        xji = pos_j - pos_i
                        # n = self.mesh_dy.num_neighbours[pi]
                        if xji.norm() < kernel_radius:
                            n = xji.normalized()
                            dp = xji - kernel_radius * n
                            self.mesh_dy.p[pi] += self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi] * compliance_col * dp
                            self.mesh_dy.p[pj] -= self.mesh_dy.fixed[pj] * self.mesh_dy.m_inv[pj] * compliance_col * dp
                            self.mesh_dy.nc[pi] += self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi] * compliance_col
                            self.mesh_dy.nc[pj] += self.mesh_dy.fixed[pj] * self.mesh_dy.m_inv[pj] * compliance_col
                            ni = self.mesh_dy.num_neighbours_dy[pi]
                            if ni < self.mesh_dy.cache_size_dy:
                                self.mesh_dy.neighbour_ids_dy[pi, ni] = pj
                                self.mesh_dy.num_neighbours_dy[pi] += 1

        for pi in range(self.num_verts_dy):
            # if self.mesh_dy.nc[pi] > 0:
            self.mesh_dy.x_k[pi] += self.mesh_dy.p[pi] / self.mesh_dy.nc[pi]


    def solve_constraints_euler_path_tridiagonal_x(self, dt):

        compliance_stretch = self.stiffness_stretch * dt * dt
        compliance_bending = self.stiffness_bending * dt * dt

        self.solve_spring_constraints_euler_path_tridiagonal_x(compliance_stretch, compliance_bending)

    @ti.kernel
    def compute_spring_E_grad_and_hess(self, compliance_stretch: ti.f32, compliance_bending: ti.f32) -> ti.f32:

        id3 = ti.Matrix.identity(dt=float, n=3)
        compliance_attach = 1e4
        self.mesh_dy.hii_e.fill(0.0)
        self.mesh_dy.hi_e.fill(0.0)
        E = 0.0

        for i in self.mesh_dy.p:
            dx_m = (self.mesh_dy.x_k[i] - self.mesh_dy.y_tilde[i])
            self.mesh_dy.grad[i] = self.mesh_dy.m[i] * dx_m
            E += 0.5 * self.mesh_dy.m[i] * dx_m.dot(dx_m)
            self.mesh_dy.hii[i] = self.mesh_dy.m[i] * id3
            self.mesh_dy.hi[i] = self.mesh_dy.m[i]

            if self.mesh_dy.fixed[i] < 1.0:
                dx = (self.mesh_dy.x_k[i] - self.mesh_dy.x0[i])
                self.mesh_dy.grad[i] += compliance_attach * dx
                self.mesh_dy.hii[i] += compliance_attach * id3
                self.mesh_dy.hi[i] += compliance_attach
                E += 0.5 * compliance_attach * dx.dot(dx)

        for i in range(self.num_edges_dy):
            l0 = self.mesh_dy.l0[i]
            v0_d, v1_d = self.mesh_dy.eid_dup[2 * i + 0], self.mesh_dy.eid_dup[2 * i + 1]
            v0, v1 = self.mesh_dy.dup_to_ori[v0_d], self.mesh_dy.dup_to_ori[v1_d]
            x01 = self.mesh_dy.x_k[v0] - self.mesh_dy.x_k[v1]
            l = x01.norm()
            n = x01.normalized()

            E += 0.5 * compliance_stretch * (l0 - l) ** 2
            nnT = n.outer_product(n)
            # alpha = (1.0 - l0 / l)

            alpha = 1.0
            B = compliance_stretch * (alpha * id3 + (1.0 - alpha) * nnT)
            # alpha = 0.5
            self.mesh_dy.hij[i] = B
            # self.mesh_dy.hij[i] = -compliance_stretch * id3

            self.c[v0_d] = -B
            self.c_1d[v0_d] = -compliance_stretch

            self.a[v1_d] = -B
            self.a_1d[v1_d] = -compliance_stretch

            dp01 = x01 - l0 * x01.normalized()

            self.mesh_dy.grad[v0] += compliance_stretch * dp01
            self.mesh_dy.grad[v1] -= compliance_stretch * dp01

            self.mesh_dy.hii_e[v0] += B
            self.mesh_dy.hii_e[v1] += B

            self.mesh_dy.hi_e[v0] += compliance_stretch
            self.mesh_dy.hi_e[v1] += compliance_stretch

        return E

    @ti.kernel
    def compute_spring_E(self, x: ti.template(), compliance_stretch: ti.f32, compliance_bending: ti.f32) -> ti.f32:

        compliance_attach = 1e5
        E = 0.0

        # Momentum term: 1/2 * || x - y ||^2_M
        for i in self.mesh_dy.p:
            dx_m = (x[i] - self.mesh_dy.y_tilde[i])
            E += 0.5 * self.mesh_dy.m[i] * dx_m.dot(dx_m)

            if self.mesh_dy.fixed[i] < 1.0:
                dx = x[i] - self.mesh_dy.x0[i]
                E += 0.5 * compliance_attach * dx.dot(dx)


        # Easticity term: spring energy
        for i in range(self.num_edges_dy):
            l0 = self.mesh_dy.l0[i]
            v0_d, v1_d = self.mesh_dy.eid_dup[2 * i + 0], self.mesh_dy.eid_dup[2 * i + 1]
            v0, v1 = self.mesh_dy.dup_to_ori[v0_d], self.mesh_dy.dup_to_ori[v1_d]
            x01 = x[v0] - x[v1]
            l = x01.norm()
            E += 0.5 * compliance_stretch * (l0 - l) ** 2

        return E


    @ti.kernel
    def apply_preconditioning_euler(self, ret: ti.template(), x: ti.template()):

        for di in self.mesh_dy.x_dup:
            vi = self.mesh_dy.dup_to_ori[di]

            self.b[di] = self.mesh_dy.hii[vi] + self.mesh_dy.hii_e[vi]
            self.d[di] = x[vi]

        n_part = (self.mesh_dy.partition_offset.shape[0] - 1)
        for pi in range(n_part):

            size = self.mesh_dy.vert_offset[pi + 1] - self.mesh_dy.vert_offset[pi]
            offset = self.mesh_dy.vert_offset[pi]

            # Thomas algorithm
            # https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm

            # for j in ti.static(range(3)):
            self.c_tilde[offset] = ti.math.inverse(self.b[offset]) @ self.c[offset]
            ti.loop_config(serialize=True)
            for id in range(size):  # lb+1 ~ ub-1
                i = id + offset
                tmp = ti.math.inverse(self.b[i] - self.a[i] @ self.c_tilde[i - 1])

                self.c_tilde[i] = tmp @ self.c[i]
            #
            self.d_tilde[offset] = ti.math.inverse(self.b[offset]) @ self.d[offset]
            ti.loop_config(serialize=True)
            for id in range(1, size):  # lb+1 ~ ub
                i = id + offset
                tmp = ti.math.inverse(self.b[i] - self.a[i] @ self.c_tilde[i - 1])
                self.d_tilde[i] = tmp @ (self.d[i] - self.a[i] @ self.d_tilde[i - 1])

            self.mesh_dy.dx_dup[offset + size - 1] = self.d_tilde[offset + size - 1]
            ti.loop_config(serialize=True)
            for i in range(0, size - 1):
                idx = size - 2 - i + offset  # ub-1 ~ lb
                self.mesh_dy.dx_dup[idx] = self.d_tilde[idx] - self.c_tilde[idx] @ self.mesh_dy.dx_dup[idx + 1]

        ret.fill(0.0)
        for di in self.mesh_dy.x_dup:
            vi = self.mesh_dy.dup_to_ori[di]
            ret[vi] += self.mesh_dy.dx_dup[di]

        for i in self.mesh_dy.x_k:
            ret[i] /= self.mesh_dy.num_dup[i]

    @ti.kernel
    def apply_preconditioning_euler_1d(self, ret: ti.template(), x: ti.template()):

        for di in self.mesh_dy.x_dup:
            vi = self.mesh_dy.dup_to_ori[di]

            self.b_1d[di] = self.mesh_dy.hi[vi] + self.mesh_dy.hi_e[vi]
            self.d[di] = x[vi]

        n_part = (self.mesh_dy.partition_offset.shape[0] - 1)
        for pi in range(n_part):

            size = self.mesh_dy.vert_offset[pi + 1] - self.mesh_dy.vert_offset[pi]
            offset = self.mesh_dy.vert_offset[pi]

            # Thomas algorithm
            # https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm

            # for j in ti.static(range(3)):
            # self.c_tilde[offset] = ti.math.inverse(self.b[offset]) @ self.c[offset]
            self.c_tilde_1d[offset] = self.c_1d[offset] / self.b_1d[offset]
            ti.loop_config(serialize=True)
            for id in range(size):  # lb+1 ~ ub-1
                i = id + offset
                self.c_tilde_1d[i] = self.c_1d[i] / (self.b_1d[i] - self.a_1d[i] * self.c_tilde_1d[i - 1])
            #
            # self.d_tilde[offset] = ti.math.inverse(self.b[offset]) @ self.d[offset]
            self.d_tilde[offset] = self.d[offset] / self.b_1d[offset]
            ti.loop_config(serialize=True)
            for id in range(1, size):  # lb+1 ~ ub
                i = id + offset
                # tmp = ti.math.inverse(self.b[i] - self.a[i] * self.c_tilde[i - 1])
                self.d_tilde[i] = (self.d[i] - self.a_1d[i] * self.d_tilde[i - 1]) / (self.b_1d[i] - self.a_1d[i] * self.c_tilde_1d[i - 1])

            self.mesh_dy.dx_dup[offset + size - 1] = self.d_tilde[offset + size - 1]
            ti.loop_config(serialize=True)
            for i in range(0, size - 1):
                idx = size - 2 - i + offset  # ub-1 ~ lb
                self.mesh_dy.dx_dup[idx] = self.d_tilde[idx] - self.mesh_dy.dx_dup[idx + 1] * self.c_tilde_1d[idx]

        ret.fill(0.0)
        for di in self.mesh_dy.x_dup:
            vi = self.mesh_dy.dup_to_ori[di]
            ret[vi] += self.mesh_dy.dx_dup[di]

        for i in self.mesh_dy.x_k:
            ret[i] /= self.mesh_dy.num_dup[i]

    @ti.kernel
    def apply_preconditioning_jacobi(self, ret: ti.template(), x: ti.template()):

        for i in self.mesh_dy.x_k:
            # ret[i] = ti.math.inverse(self.mesh_dy.hii[i] + self.mesh_dy.hii_e[i]) @ x[i]
            ret[i] =  x[i] / (self.mesh_dy.hi[i] + self.mesh_dy.hi_e[i])


    @ti.kernel
    def add(self, ret: ti.template(), x: ti.template(), y: ti.template(), scale: ti.f32):

        for i in ret:
            ret[i] = x[i] + y[i] * scale

    @ti.kernel
    def dot(self, x: ti.template(), y: ti.template()) -> ti.f32:
        ret = 0.0
        for i in x:
            ret += x[i].dot(y[i])

        return ret

    @ti.kernel
    def proceed(self, alpha: ti.f32):

        #g_k+1:  self.mesh_dy.grad
        #g_k:    self.mesh_dy.grad_prev
        #y_k:    g_k+1 - g_k

        #p_k+1:    self.mesh_dy.dx
        #p_k:    self.mesh_dy.dx_prev

        # beta = 0.0

        for i in self.mesh_dy.x_k:
            # self.mesh_dy.p[i] = -self.mesh_dy.P_grad[i] + beta * self.mesh_dy.p_k[i]
            self.mesh_dy.x_k[i] += alpha * self.mesh_dy.p[i]

            self.mesh_dy.p_k[i] = self.mesh_dy.p[i]
            self.mesh_dy.grad_k[i] = self.mesh_dy.grad[i]

    @ti.kernel
    def compute_beta(self, g:ti.template(), Py:ti.template(), y:ti.template(), p:ti.template()) -> ti.f32:

        g_Py = 0.0
        y_Py = 0.0
        p_g =0.0
        y_p = 0.0
        for i in  g:
            g_Py += g[i].dot(Py[i])
            y_Py += y[i].dot(Py[i])
            p_g  += p[i].dot(g[i])
            y_p  += y[i].dot(p[i])


        # print(g_Py, y_Py, p_g, y_p)

        return (g_Py - (y_Py / y_p) * p_g)/y_p

    @ti.kernel
    def compute_E_delta(self)-> (ti.f32, ti.f32):

        g_p = 0.0
        p_Hp = 0.0

        # max_value = ti.math.vec3(-1e3)
        # for i in self.mesh_dy.p:
        #     max_value = ti.atomic_max(abs(self.mesh_dy.p[i]), max_value)
        #


        # self.mesh_dy.Hp.fill(0.0)
        for i in self.mesh_dy.grad:
            g_p += self.mesh_dy.grad[i].dot(self.mesh_dy.p[i])
            p_Hp += self.mesh_dy.p[i].dot(self.mesh_dy.hii[i] @ self.mesh_dy.p[i])

        for i in range(self.num_edges_dy):
            v0_d, v1_d = self.mesh_dy.eid_dup[2 * i + 0], self.mesh_dy.eid_dup[2 * i + 1]
            v0, v1 = self.mesh_dy.dup_to_ori[v0_d], self.mesh_dy.dup_to_ori[v1_d]
            pji  = self.mesh_dy.p[v1] - self.mesh_dy.p[v0]
            p_Hp += pji.dot(self.mesh_dy.hij[i] @ pji)


        # alpha = g_p / p_Hp
        alpha = 1.0
        delta_E = -alpha * (g_p + 0.5 * alpha * p_Hp)

        return alpha, delta_E




    def forward(self, n_substeps, n_iter):

        dt_sub = self.dt / n_substeps
        delta_E0 = 0.0

        for _ in range(n_substeps):

            self.compute_y(self.g, dt_sub)
            self.conv_iter = 0
            for _ in range(n_iter):

                compliance_stretch = self.stiffness_stretch * dt_sub * dt_sub
                compliance_bending = self.stiffness_bending * dt_sub * dt_sub

                E_k = self.compute_spring_E_grad_and_hess(compliance_stretch, compliance_bending)

                if self.selected_precond_type == 0:

                    ti.profiler.clear_kernel_profiler_info()  # [1]
                    self.apply_preconditioning_euler_1d(self.mesh_dy.P_grad, self.mesh_dy.grad)
                    # self.apply_preconditioning_euler_1d(self.mesh_dy.P_grad, self.mesh_dy.grad)
                    query_result = ti.profiler.query_kernel_profiler_info(self.apply_preconditioning_euler_1d.__name__)  # [2]
                    print("Euler elapsed time(avg_in_ms) =", query_result.avg)

                    # self.apply_preconditioning_euler_1d(self.mesh_dy.P_grad, self.mesh_dy.grad)

                elif self.selected_precond_type == 1:
                    ti.profiler.clear_kernel_profiler_info()  # [1]
                    self.apply_preconditioning_jacobi(self.mesh_dy.P_grad, self.mesh_dy.grad)
                    query_result = ti.profiler.query_kernel_profiler_info(self.apply_preconditioning_jacobi.__name__)  # [2]
                    print("Jacobi elapsed time(avg_in_ms) =", query_result.avg)

                beta = 0.0

                if self.conv_iter > 0 and self.enable_pncg:
                    self.add(self.mesh_dy.grad_delta, self.mesh_dy.grad, self.mesh_dy.grad_k, -1.0)
                    if self.selected_precond_type == 0:
                        self.apply_preconditioning_euler_1d(self.mesh_dy.P_grad_delta, self.mesh_dy.grad_delta)
                        # self.apply_preconditioning_euler_1d(self.mesh_dy.P_grad_delta, self.mesh_dy.grad_delta)

                    elif self.selected_precond_type == 1:
                        self.apply_preconditioning_jacobi(self.mesh_dy.P_grad_delta, self.mesh_dy.grad_delta)

                    beta = self.compute_beta(self.mesh_dy.grad, self.mesh_dy.P_grad_delta, self.mesh_dy.grad_delta, self.mesh_dy.p_k)

                E = self.compute_spring_E(self.mesh_dy.x, compliance_stretch, compliance_bending)

                self.add(self.mesh_dy.p, self.mesh_dy.P_grad, self.mesh_dy.p_k, -beta)
                gP_g = self.dot(self.mesh_dy.grad, self.mesh_dy.P_grad)
                if gP_g < self.threshold:
                    break

                alpha = 1.0
                self.proceed(-alpha)
                self.conv_iter += 1


            # self.solve_constraints_newton_pcg_x(dt_sub, self.max_cg_iter, self.threshold)
            self.compute_v(damping=self.damping, dt=dt_sub)
            self.update_x(dt_sub)

            # if self.selected_precond_type == 0:
            #     query_result = ti.profiler.query_kernel_profiler_info(self.apply_preconditioning_euler.__name__)  # [2]
            #     # print("kernel executed times =", query_result.counter)
            #     # print("kernel elapsed time(min_in_ms) =", query_result.min)
            #     # print("kernel elapsed time(max_in_ms) =", query_result.max)
            #     print("kernel elapsed time(avg_in_ms) =", query_result.avg)
            #
            # elif self.selected_precond_type == 1:
            #     query_result = ti.profiler.query_kernel_profiler_info(self.apply_preconditioning_jacobi.__name__)  # [2]
            #     # print("kernel executed times =", query_result.counter)
            #     # print("kernel elapsed time(min_in_ms) =", query_result.min)
            #     # print("kernel elapsed time(max_in_ms) =", query_result.max)
            #     print("kernel elapsed time(avg_in_ms) =", query_result.avg)

            # self.copy_to_dup()
