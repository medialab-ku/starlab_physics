import taichi as ti
from framework.physics.conjugate_gradient import ConjugateGradient
@ti.data_oriented
class Solver:
    def __init__(
            self,
            mesh_dy,
            mesh_st,
            particle_st,
            dHat,
            sh_st,
            # sh_st_e,
            stiffness_stretch,
            stiffness_bending,
            g,
            dt):

        self.mesh_dy = mesh_dy
        self.mesh_st = mesh_st
        self.particle_st = particle_st
        self.dHat = dHat
        self.sh_st = sh_st
        # self.sh_st_e = sh_st_e
        self.stiffness_stretch = stiffness_stretch
        self.stiffness_bending = stiffness_bending
        self.g = g
        self.dt = dt
        self.damping = 0.001
        self.threshold = 1e-4
        self.max_cg_iter = 100

        self.mu = 0.8
        self.PCG = ConjugateGradient()
        self.selected_solver_type = 0
        self.definiteness_fix = True
        self.print_stats = False
        self.use_line_search = True
        self.enable_velocity_update = False

        self.num_verts_dy = self.mesh_dy.num_verts
        self.num_edges_dy = self.mesh_dy.num_edges
        self.num_faces_dy = self.mesh_dy.num_faces
        if mesh_st is not None:
            self.num_verts_st = self.mesh_st.num_verts
            self.num_edges_st = self.mesh_st.num_edges
            self.num_faces_st = self.mesh_st.num_faces
        else:
            self.num_verts_st = 0
            self.num_edges_st = 0
            self.num_faces_st = 0

        self.reset()

    ####################################################################################################################

    @ti.kernel
    def set_fixed_vertices(self, fixed_vertices: ti.template()):
        for i in range(self.num_verts_dy):
            if fixed_vertices[i] >= 1:
                self.mesh_dy.fixed[i] = 0.0
            else:
                self.mesh_dy.fixed[i] = 1.0

    def reset(self):
        self.mesh_dy.reset()
        # self.mesh_dy.particles.reset()
        if self.mesh_st is None:
            self.mesh_st.reset()

        self.set_edge_and_face_particles()

    def init_variables(self):
        # initialize dx and number of constraints (used in the Jacobi solver)
        self.mesh_dy.dx.fill(0.0)
        self.mesh_dy.nc.fill(0.0)

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
            self.mesh_dy.y[i] = self.mesh_dy.x[i] + self.mesh_dy.fixed[i] * (self.mesh_dy.v[i] * dt + g * dt * dt)
            self.mesh_dy.y[i] = self.confine_boundary(self.mesh_dy.y[i])
            self.mesh_dy.y_origin[i] = self.mesh_dy.y[i]

    @ti.kernel
    def update_dx(self):
        # update x_(t+1) by adding dx (used in the Jacobi solver)
        for i in range(self.num_verts_dy):
            if self.mesh_dy.nc[i] > 0:
                self.mesh_dy.y[i] += self.mesh_dy.fixed[i] * (self.mesh_dy.dx[i] / self.mesh_dy.nc[i])

    @ti.kernel
    def compute_v(self, damping: ti.f32, dt: ti.f32):
        # compute v after all constraints are projected to x_(t+1)
        for i in range(self.num_verts_dy):
            self.mesh_dy.v[i] = self.mesh_dy.fixed[i] * (1.0 - damping) * (self.mesh_dy.y[i] - self.mesh_dy.x[i]) / dt

    @ti.kernel
    def update_x(self, dt: ti.f32):
        # eventually, update actual position x_(t+1) after velocities are computed...
        for i in range(self.num_verts_dy):
            self.mesh_dy.x[i] += dt * self.mesh_dy.v[i]

        # update the particles' position!
        # num_faces = self.mesh_dy.particles.num_faces
        # num_particles_per_edge = self.mesh_dy.particles.num_particles_per_edge
        # num_particles_per_face = self.mesh_dy.particles.num_particles_per_face
        # ub, lb = self.mesh_dy.particles.ub, self.mesh_dy.particles.lb
        #
        # for i in range(num_faces):
        #     vid_0, vid_1, vid_2 = self.mesh_dy.fid_field[i,0], self.mesh_dy.fid_field[i,1], self.mesh_dy.fid_field[i,2]
        #     v0, v1, v2 = self.mesh_dy.x[vid_0], self.mesh_dy.x[vid_1], self.mesh_dy.x[vid_2]
        #
        #     count = 0
        #     for j in range(self.mesh_dy.particles.num_particles_per_edge):
        #         for k in range(j + 1):
        #             self.mesh_dy.particles.particles_per_face_field[num_particles_per_face * i + count] = (
        #                 (ub - ((ub - lb) * (j / (num_particles_per_edge - 1)))) * v0 +
        #                 ((lb / 2) + ((ub - lb) * ((j-k) / (num_particles_per_edge - 1)))) * v1 +
        #                 ((lb / 2) + ((ub - lb) * (k / (num_particles_per_edge - 1)))) * v2
        #             )
        #             count += 1

    ####################################################################################################################
    # Several constraint solvers to compute physics...
    def solve_constraints_jacobi_x(self, dt):
        # self.init_variables()

        compliance_stretch = self.stiffness_stretch * dt * dt
        compliance_bending = self.stiffness_bending * dt * dt

        self.solve_spring_constraints_jacobi_x(compliance_stretch, compliance_bending)
        # self.solve_xpbd_collision_constraints_st_x(2 * self.dHat)

    def solve_constraints_gauss_seidel_x(self, dt):
        compliance_stretch = self.stiffness_stretch * dt * dt
        compliance_bending = self.stiffness_bending * dt * dt

        self.solve_spring_constraints_gauss_seidel_x(compliance_stretch, compliance_bending)

    def solve_constraints_pd_diag_x(self, dt):
        compliance_stretch = self.stiffness_stretch * dt * dt
        compliance_bending = self.stiffness_bending * dt * dt

        self.solve_spring_constraints_pd_diag_x(compliance_stretch, compliance_bending)

        compliance_collision = 1e6 * dt * dt
        # self.solve_xpbd_collision_constraints_st_x(2 * self.dHat)
        if self.selected_solver_type == 1:
            self.solve_collision_constraints_st_pd_diag_x(2 * self.dHat, compliance_collision)
        elif self.selected_solver_type == 2:
            self.solve_collision_constraints_st_pd_diag_test_x(2 * self.dHat, compliance_collision)

    def solve_constraints_pd_diag_v(self):
        self.solve_collision_constraints_st_pd_diag_v(2 * self.dHat)

    def solve_constraints_hess_diag_x(self, dt):
        compliance_stretch = self.stiffness_stretch * dt * dt
        compliance_bending = self.stiffness_bending * dt * dt

        self.solve_spring_constraints_hess_diag_x(compliance_stretch, compliance_bending)


    @ti.func
    def outer_product(self, u: ti.math.vec3, v: ti.math.vec3) -> ti.math.mat3:

        uvT = ti.math.mat3(0.0)
        for I in ti.grouped(ti.ndrange((0, 3), (0, 3))):
            uvT[I] += u[I[0]] * v[I[1]]

        return uvT


    @ti.kernel
    def line_search(self, E_current: float) -> float:

        alpha = 1.0

        return alpha

    def solve_constraints_newton_pcg_x(self, dt, max_cg_iter, threshold):

        compliance_stretch = self.stiffness_stretch * dt * dt
        compliance_bending = self.stiffness_bending * dt * dt

        E_curr = self.compute_grad_and_hess_spring_x(compliance_stretch, compliance_bending, self.definiteness_fix)

        r_sq, cg_iter = self.PCG.run(self.mesh_dy, max_cg_iter, threshold)
        alpha = 1.0
        if self.use_line_search:
            self.PCG.compute_mat_free_Ax(self.mesh_dy, self.mesh_dy.Ax, self.mesh_dy.dx)
            test = self.PCG.dot_product(self.mesh_dy.dx, self.mesh_dy.Ax)
            if test > 1e-3:
                alpha = self.PCG.dot_product(self.mesh_dy.b, self.mesh_dy.dx) / test
            else:
                alpha = 1.0

        if self.print_stats:
            print("CG err: ", r_sq)
            print("CG iter: ", cg_iter)

        if self.print_stats and self.use_line_search:
            print("alpha: ", alpha)

        self.PCG.vector_add(self.mesh_dy.y, self.mesh_dy.y, self.mesh_dy.dx, alpha)

    def solve_constraints_pd_pcg_x(self, dt, max_cg_iter, threshold):

        compliance_stretch = self.stiffness_stretch * dt * dt
        compliance_bending = self.stiffness_bending * dt * dt

        self.compute_pd_grad_spring_x(compliance_stretch, compliance_bending)
        self.PCG.run(self.mesh_dy, max_cg_iter, threshold)
        alpha = 1.0
        if self.use_line_search:
            alpha = self.line_search(0.0)

        self.PCG.vector_add(self.mesh_dy.y, self.mesh_dy.y, self.mesh_dy.dx, alpha)


    ####################################################################################################################
    # Taichi kernel function which are called in solvers...

    @ti.kernel
    def solve_spring_constraints_jacobi_x(self, compliance_stretch: ti.f32, compliance_bending: ti.f32):
        # project x_(t+1) by solving constraints in parallel way

        self.mesh_dy.dx.fill(0.0)
        self.mesh_dy.nc.fill(0.0)

        ti.block_local(self.mesh_dy.l0, self.mesh_dy.eid_field, self.mesh_dy.fixed, self.mesh_dy.m_inv, self.mesh_dy.dx, self.mesh_dy.nc)
        for i in range(self.num_edges_dy):
            l0 = self.mesh_dy.l0[i]
            v0, v1 = self.mesh_dy.eid_field[i, 0], self.mesh_dy.eid_field[i, 1]
            x10 = self.mesh_dy.y[v0] - self.mesh_dy.y[v1]
            lij = x10.norm()

            C = (lij - l0)
            nabla_C = x10.normalized()
            schur = self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] + self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1]

            ld = compliance_stretch * C / (compliance_stretch * schur + 1.0)

            self.mesh_dy.dx[v0] -= self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] * ld * nabla_C
            self.mesh_dy.dx[v1] += self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1] * ld * nabla_C
            self.mesh_dy.nc[v0] += 1.0
            self.mesh_dy.nc[v1] += 1.0

        # after solving all constaints, we should coordinate the projected x_(t+1)!
        ti.block_local(self.mesh_dy.y, self.mesh_dy.dx, self.mesh_dy.nc)
        for i in range(self.num_verts_dy):
            # if self.mesh_dy.nc[i] > 0:
            self.mesh_dy.y[i] += self.mesh_dy.fixed[i] * (self.mesh_dy.dx[i] / self.mesh_dy.nc[i])

    @ti.kernel
    def solve_spring_constraints_gauss_seidel_x(self, compliance_stretch: ti.f32, compliance_bending: ti.f32):
        ti.loop_config(serialize=True)
        ti.block_local(self.mesh_dy.l0, self.mesh_dy.eid_field, self.mesh_dy.fixed, self.mesh_dy.m_inv)
        for i in range(self.num_edges_dy):
            l0 = self.mesh_dy.l0[i]
            v0, v1 = self.mesh_dy.eid_field[i,0], self.mesh_dy.eid_field[i,1]
            x10 = self.mesh_dy.y[v0] - self.mesh_dy.y[v1]
            lij = x10.norm()

            C = (lij - l0)
            nabla_C = x10.normalized()
            schur = self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] + self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1]

            ld = compliance_stretch * C / (compliance_stretch * schur + 1.0)

            self.mesh_dy.y[v0] -= self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] * ld * nabla_C
            self.mesh_dy.y[v1] += self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1] * ld * nabla_C

    @ti.kernel
    def solve_spring_constraints_pd_diag_x(self, compliance_stretch: ti.f32, compliance_bending: ti.f32):

        self.mesh_dy.dx.fill(0.0)
        self.mesh_dy.nc.fill(1.0)

        # ti.loop_config(serialize=True)
        ti.block_local(self.mesh_dy.l0, self.mesh_dy.eid_field, self.mesh_dy.fixed, self.mesh_dy.m_inv)
        for i in range(self.num_edges_dy):
            l0 = self.mesh_dy.l0[i]
            v0, v1 = self.mesh_dy.eid_field[i, 0], self.mesh_dy.eid_field[i, 1]
            x01 = self.mesh_dy.y[v0] - self.mesh_dy.y[v1]
            dp01 = x01 - l0 * x01.normalized()

            # alpha = (1.0 - l0 / x01.norm())

            self.mesh_dy.dx[v0] -= self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] * compliance_stretch * dp01
            self.mesh_dy.dx[v1] += self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1] * compliance_stretch * dp01

            self.mesh_dy.nc[v0] += self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] * compliance_stretch
            self.mesh_dy.nc[v1] += self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1] * compliance_stretch

        ti.block_local(self.mesh_dy.y, self.mesh_dy.dx, self.mesh_dy.nc)
        for i in range(self.num_verts_dy):
            # if self.mesh_dy.nc[i] > 0:
            self.mesh_dy.y[i] += (self.mesh_dy.dx[i] / self.mesh_dy.nc[i])

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
            v0, v1 = self.mesh_dy.eid_field[i, 0], self.mesh_dy.eid_field[i, 1]
            x01 = self.mesh_dy.y[v0] - self.mesh_dy.y[v1]
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
        ti.block_local(self.mesh_dy.y, self.mesh_dy.dx, self.mesh_dy.nc)
        for i in range(self.num_verts_dy):
            # if self.mesh_dy.nc[i] > 0:
            self.mesh_dy.y[i] += self.mesh_dy.b[i] / (self.mesh_dy.m[i] + self.mesh_dy.nc[i])

    @ti.kernel
    def compute_grad_and_hess_spring_x(self, compliance_stretch: ti.f32, compliance_bending: ti.f32, definite_fix: bool)-> float:

        self.mesh_dy.b.fill(0.0)
        E_cur = 0.0
        id3 = ti.math.mat3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        for i in range(self.num_verts_dy):
            test = 1.0
            if self.mesh_dy.fixed[i] < 1.0:
                test = 1e5
            self.mesh_dy.hii[i] = test * self.mesh_dy.m[i] * id3

        # ti.loop_config(serialize=True)
        ti.block_local(self.mesh_dy.l0, self.mesh_dy.eid_field, self.mesh_dy.fixed, self.mesh_dy.m_inv)
        for i in range(self.num_edges_dy):
            l0 = self.mesh_dy.l0[i]
            v0, v1 = self.mesh_dy.eid_field[i, 0], self.mesh_dy.eid_field[i, 1]
            x01 = self.mesh_dy.y[v0] - self.mesh_dy.y[v1]
            l = x01.norm()
            n = x01.normalized()
            dp01 = (l - l0) * n
            E_cur += 0.5 * compliance_stretch * (l - l0) ** 2
            alpha = 1.0 - l0 / x01.norm()
            if definite_fix and alpha < 1e-3:
                alpha = 1e-3

            # alpha = 0.5

            self.mesh_dy.hij[i] = compliance_stretch * (alpha * id3 + (1.0 - alpha) * self.outer_product(n, n))

            self.mesh_dy.b[v0] -= compliance_stretch * dp01
            self.mesh_dy.b[v1] += compliance_stretch * dp01

        return E_cur
    @ti.kernel
    def compute_pd_grad_spring_x(self, compliance_stretch: ti.f32, compliance_bending: ti.f32):

        self.mesh_dy.b.fill(0.0)

        id3 = ti.math.mat3([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        for i in range(self.num_verts_dy):
            test = 1.0
            if self.mesh_dy.fixed[i] < 1.0:
                test = 1e5
            self.mesh_dy.hii[i] = test * self.mesh_dy.m[i] * id3

        # ti.loop_config(serialize=True)
        ti.block_local(self.mesh_dy.l0, self.mesh_dy.eid_field, self.mesh_dy.fixed, self.mesh_dy.m_inv)
        for i in range(self.num_edges_dy):
            l0 = self.mesh_dy.l0[i]
            v0, v1 = self.mesh_dy.eid_field[i, 0], self.mesh_dy.eid_field[i, 1]
            x01 = self.mesh_dy.y[v0] - self.mesh_dy.y[v1]
            l = x01.norm()
            n = x01.normalized()
            dp01 = (l - l0) * n

            self.mesh_dy.hij[i] = compliance_stretch * id3
            self.mesh_dy.b[v0] -= compliance_stretch * dp01
            self.mesh_dy.b[v1] += compliance_stretch * dp01


    ####################################################################################################################


    @ti.kernel
    def solve_xpbd_collision_constraints_st_x(self, distance_threshold: float):

        self.mesh_dy.dx.fill(0.0)
        self.mesh_dy.nc.fill(0.0)

        for pi in range(self.num_verts_dy):
            pos_i = self.mesh_dy.y[pi]
            cell_id = self.sh_st.pos_to_cell_id(pos_i)
            for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                cell_to_check = cell_id + offs
                if self.sh_st.is_in_grid(cell_to_check):
                    for j in range(self.sh_st.num_particles_in_cell[cell_to_check]):
                        pj = self.sh_st.particle_ids_in_cell[cell_to_check, j]
                        if pi == pj:
                            continue
                        pos_j = self.mesh_st.x_test[pj]
                        xji = pos_j - pos_i
                        if xji.norm() < distance_threshold:
                            C = (xji.norm() - distance_threshold)
                            nabla_C = ti.math.normalize(xji)
                            schur = (self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi])
                            k = 1e8
                            ld = -(k * C) / (k * schur + 1.0)

                            self.mesh_dy.dx[pi] -= self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi] * ld * nabla_C
                            # self.dx[pj] += self.is_fixed[pj] * self.m_inv[pj] * ld * nabla_C
                            self.mesh_dy.nc[pi] += 1.0
                            # self.nc[pj] += 1.

        for pi in range(self.num_verts_dy):
            if self.mesh_dy.nc[pi] > 0:
                self.mesh_dy.y[pi] += self.mesh_dy.dx[pi] / self.mesh_dy.nc[pi]

    @ti.func
    def poly6_value(self, s, h):
        coef = 315.0 / 64.0 / ti.math.pi
        result = 0.0
        if 0 <= s and s <= h:
            x = (h * h - s * s) / (h * h * h)
            result = coef * x * x * x

        return result

    @ti.func
    def spiky_gradient(self, r, h) -> ti.math.vec3:
        coef = -45.0 / ti.math.pi
        result = ti.math.vec3(0.0)
        r_len = r.norm()
        if 0 < r_len and r_len <= h:
            x = (h - r_len) / (h * h * h)
            g_factor = coef * x * x
            result = r * g_factor / r_len

        return result

    @ti.kernel
    def solve_collision_constraints_st_pd_diag_x(self, kernel_radius: float, compliance_col: float):

        self.mesh_dy.dx.fill(0.0)
        self.mesh_dy.nc.fill(1.0)

        for i in range(self.num_verts_dy):
            # if i < self.num_verts_dy:
            pi = i
            pos_i = self.mesh_dy.y[pi]
            C = 0.0
            nabla_C = ti.math.vec3(0.0)
            # grad_rh0 = ti.math.vec3(0.0)
            ni = 0.
            cell_id = self.sh_st.pos_to_cell_id(pos_i)
            for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                cell_to_check = cell_id + offs
                if self.sh_st.is_in_grid(cell_to_check):
                    for j in range(self.sh_st.num_particles_in_cell[cell_to_check]):
                        pj = self.sh_st.particle_ids_in_cell[cell_to_check, j]
                        if pi == pj:
                            continue
                        pos_j = self.particle_st.x[pj]
                        xji = pos_j - pos_i
                        # n = self.mesh_dy.num_neighbours[pi]
                        if xji.norm() < kernel_radius:
                            ni += 1.
                            C += self.poly6_value(xji.norm(), kernel_radius)
                            nabla_C += self.spiky_gradient(xji, kernel_radius)

            dp = (C / (nabla_C.dot(nabla_C) + 1e-3)) * nabla_C


            self.mesh_dy.dx[pi] += ni * self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi] * compliance_col * dp
            self.mesh_dy.nc[pi] += ni * self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi] * compliance_col

        for i in range(self.num_edges_dy):
            v0, v1 = self.mesh_dy.eid_field[i, 0], self.mesh_dy.eid_field[i, 1]
            #     pos_i = 0.5 * (self.mesh_dy.y[v0] + self.mesh_dy.y[v1])
            pi = i
            pos_i = 0.5 * (self.mesh_dy.y[v0] + self.mesh_dy.y[v1])
            C = 0.0
            nabla_C = ti.math.vec3(0.0)
            # grad_rh0 = ti.math.vec3(0.0)
            ni = 0.0
            cell_id = self.sh_st.pos_to_cell_id(pos_i)
            for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                cell_to_check = cell_id + offs
                if self.sh_st.is_in_grid(cell_to_check):
                    for j in range(self.sh_st.num_particles_in_cell[cell_to_check]):
                        pj = self.sh_st.particle_ids_in_cell[cell_to_check, j]
                        if pi == pj:
                            continue
                        pos_j = self.particle_st.x[pj]
                        xji = pos_j - pos_i
                        # n = self.mesh_dy.num_neighbours[pi]
                        if xji.norm() < kernel_radius:
                            ni += 1.0
                            C += self.poly6_value(xji.norm(), kernel_radius)
                            nabla_C += self.spiky_gradient(xji, kernel_radius)

            dp = (C / (nabla_C.dot(nabla_C) + 1e-3)) * nabla_C

            self.mesh_dy.dx[v0] += ni * 0.5 * self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] * compliance_col * dp
            self.mesh_dy.dx[v1] += ni * 0.5 * self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1] * compliance_col * dp
            self.mesh_dy.nc[v0] += ni * 0.5 * self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] * compliance_col
            self.mesh_dy.nc[v1] += ni * 0.5 * self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1] * compliance_col

        for pi in range(self.num_faces_dy):
            # pi = i - self.num_verts_dy - self.num_edges_dy
            v0, v1, v2 = self.mesh_dy.face_indices_flatten[3 * pi + 0], self.mesh_dy.face_indices_flatten[3 * pi + 1], self.mesh_dy.face_indices_flatten[3 * pi + 2]
            inv3 = 1.0 / 3.0
            pos_i = (self.mesh_dy.y[v0] + self.mesh_dy.y[v1] + self.mesh_dy.y[v2]) * inv3
            C = 0.0
            nabla_C = ti.math.vec3(0.0)
            # grad_rh0 = ti.math.vec3(0.0)
            ni = 0.0
            cell_id = self.sh_st.pos_to_cell_id(pos_i)
            for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                cell_to_check = cell_id + offs
                if self.sh_st.is_in_grid(cell_to_check):
                    for j in range(self.sh_st.num_particles_in_cell[cell_to_check]):
                        pj = self.sh_st.particle_ids_in_cell[cell_to_check, j]
                        if pi == pj:
                            continue
                        pos_j = self.particle_st.x[pj]
                        xji = pos_j - pos_i
                        # n = self.mesh_dy.num_neighbours[pi]
                        if xji.norm() < kernel_radius:
                            ni += 1.0
                            C += self.poly6_value(xji.norm(), kernel_radius)
                            nabla_C += self.spiky_gradient(xji, kernel_radius)

            dp = (C / (nabla_C.dot(nabla_C) + 1e-3)) * nabla_C
                            # k = 1e8
                            # ld = -(k * C) / (k * schur + 1.0)

            self.mesh_dy.dx[v0] += inv3 * self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] * compliance_col * dp
            self.mesh_dy.dx[v1] += inv3 * self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1] * compliance_col * dp
            self.mesh_dy.dx[v2] += inv3 * self.mesh_dy.fixed[v2] * self.mesh_dy.m_inv[v2] * compliance_col * dp
            # self.dx[pj] += self.iinv3ixed[pj] * self.m_inv[pj] * ld * nabla_C
            self.mesh_dy.nc[v0] += inv3 * self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] * compliance_col
            self.mesh_dy.nc[v1] += inv3 * self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1] * compliance_col
            self.mesh_dy.nc[v2] += inv3 * self.mesh_dy.fixed[v2] * self.mesh_dy.m_inv[v2] * compliance_col

        for pi in range(self.num_verts_dy):
            # if self.mesh_dy.nc[pi] > 0:
            self.mesh_dy.y[pi] += self.mesh_dy.dx[pi] / self.mesh_dy.nc[pi]

    @ti.kernel
    def solve_collision_constraints_st_pd_diag_v(self, kernel_radius: float):

        self.mesh_dy.dx.fill(0.0)
        self.mesh_dy.nc.fill(0.0)

        for i in range(self.num_verts_dy):
            # if i < self.num_verts_dy:
            pi = i
            pos_i = self.mesh_dy.y[pi]
            for j in range(self.mesh_dy.num_neighbours[pi]):
                pj = self.mesh_dy.neighbour_ids[pi, j]
                pos_j = self.particle_st.x[pj]
                xji = pos_j - pos_i
                n = xji.normalized()
                cv = n.dot(self.mesh_dy.v[pi])
                if n.dot(self.mesh_dy.v[pi]) < 0.0:
                    v_tan = self.mesh_dy.v[pi] - cv * n
                    self.mesh_dy.dx[pi] += (v_tan - self.mesh_dy.v[pi])
                    self.mesh_dy.nc[pi] += 1.0

            # # Cv = nabla_C
            # schur = (self.mesh_dy.m_inv[pi] *nabla_C.dot(nabla_C) + 1e-3)
            # # dp = (C / (nabla_C.dot(nabla_C) + 1e-3)) * nabla_C
            # ld = 0.0
            # if C > 0.0 and Cv > 0.0:
            #     ld = Cv / schur
            # self.mesh_dy.dx[pi] += ni * self.mesh_dy.m_inv[pi] * ld * nabla_C
            # self.mesh_dy.nc[pi] += ni

        # for i in range(self.num_edges_dy):
        #     v0, v1 = self.mesh_dy.eid_field[i, 0], self.mesh_dy.eid_field[i, 1]
        #     #     pos_i = 0.5 * (self.mesh_dy.y[v0] + self.mesh_dy.y[v1])
        #     pi = i
        #     pos_i = 0.5 * (self.mesh_dy.y[v0] + self.mesh_dy.y[v1])
        #     C = 0.0
        #     nabla_C = ti.math.vec3(0.0)
        #     # grad_rh0 = ti.math.vec3(0.0)
        #     ni = 0.0
        #     cell_id = self.sh_st.pos_to_cell_id(pos_i)
        #     for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
        #         cell_to_check = cell_id + offs
        #         if self.sh_st.is_in_grid(cell_to_check):
        #             for j in range(self.sh_st.num_particles_in_cell[cell_to_check]):
        #                 pj = self.sh_st.particle_ids_in_cell[cell_to_check, j]
        #                 if pi == pj:
        #                     continue
        #                 pos_j = self.particle_st.x[pj]
        #                 xji = pos_j - pos_i
        #                 # n = self.mesh_dy.num_neighbours[pi]
        #                 if xji.norm() < kernel_radius:
        #                     ni += 1.0
        #                     C += self.poly6_value(xji.norm(), kernel_radius)
        #                     nabla_C += self.spiky_gradient(xji, kernel_radius)
        #
        #     dp = (C / (nabla_C.dot(nabla_C) + 1e-3)) * nabla_C
        #
        #     self.mesh_dy.dx[v0] += ni * 0.5 * self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] * dp
        #     self.mesh_dy.dx[v1] += ni * 0.5 * self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1] * dp
        #     self.mesh_dy.nc[v0] += ni * 0.5 * self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0]
        #     self.mesh_dy.nc[v1] += ni * 0.5 * self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1]
        #
        # for pi in range(self.num_faces_dy):
        #     # pi = i - self.num_verts_dy - self.num_edges_dy
        #     v0, v1, v2 = self.mesh_dy.face_indices_flatten[3 * pi + 0], self.mesh_dy.face_indices_flatten[3 * pi + 1], \
        #     self.mesh_dy.face_indices_flatten[3 * pi + 2]
        #     inv3 = 1.0 / 3.0
        #     pos_i = (self.mesh_dy.y[v0] + self.mesh_dy.y[v1] + self.mesh_dy.y[v2]) * inv3
        #     C = 0.0
        #     nabla_C = ti.math.vec3(0.0)
        #     # grad_rh0 = ti.math.vec3(0.0)
        #     ni = 0.0
        #     cell_id = self.sh_st.pos_to_cell_id(pos_i)
        #     for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
        #         cell_to_check = cell_id + offs
        #         if self.sh_st.is_in_grid(cell_to_check):
        #             for j in range(self.sh_st.num_particles_in_cell[cell_to_check]):
        #                 pj = self.sh_st.particle_ids_in_cell[cell_to_check, j]
        #                 if pi == pj:
        #                     continue
        #                 pos_j = self.particle_st.x[pj]
        #                 xji = pos_j - pos_i
        #                 # n = self.mesh_dy.num_neighbours[pi]
        #                 if xji.norm() < kernel_radius:
        #                     ni += 1.0
        #                     C += self.poly6_value(xji.norm(), kernel_radius)
        #                     nabla_C += self.spiky_gradient(xji, kernel_radius)
        #
        #     dp = (C / (nabla_C.dot(nabla_C) + 1e-3)) * nabla_C
        #     # k = 1e8
        #     # ld = -(k * C) / (k * schur + 1.0)
        #
        #     self.mesh_dy.dx[v0] += inv3 * self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] * dp
        #     self.mesh_dy.dx[v1] += inv3 * self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1] * dp
        #     self.mesh_dy.dx[v2] += inv3 * self.mesh_dy.fixed[v2] * self.mesh_dy.m_inv[v2] * dp
        #     # self.dx[pj] += self.iinv3ixed[pj] * self.m_inv[pj] * ld * nabla_C
        #     self.mesh_dy.nc[v0] += inv3 * self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0]
        #     self.mesh_dy.nc[v1] += inv3 * self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1]
        #     self.mesh_dy.nc[v2] += inv3 * self.mesh_dy.fixed[v2] * self.mesh_dy.m_inv[v2]
        #
        for pi in range(self.num_verts_dy):
            if self.mesh_dy.nc[pi] > 0:
                self.mesh_dy.v[pi] += self.mesh_dy.dx[pi] / self.mesh_dy.nc[pi]

    @ti.kernel
    def solve_collision_constraints_st_pd_diag_test_x(self, kernel_radius: float, compliance_col: float):

        self.mesh_dy.dx.fill(0.0)
        self.mesh_dy.nc.fill(1.0)

        for i in range(self.num_verts_dy):
            # if i < self.num_verts_dy:
            pi = i
            pos_i = self.mesh_dy.y[pi]
            # C = 0.0
            nabla_C = ti.math.vec3(0.0)
            # grad_rh0 = ti.math.vec3(0.0)
            # ni = 0
            cell_id = self.sh_st.pos_to_cell_id(pos_i)
            for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                cell_to_check = cell_id + offs
                if self.sh_st.is_in_grid(cell_to_check):
                    for j in range(self.sh_st.num_particles_in_cell[cell_to_check]):
                        pj = self.sh_st.particle_ids_in_cell[cell_to_check, j]
                        if pi == pj:
                            continue
                        pos_j = self.particle_st.x[pj]
                        xji = pos_j - pos_i
                        # n = self.mesh_dy.num_neighbours[pi]
                        if xji.norm() < kernel_radius:
                            n = xji.normalized()
                            dp = xji - kernel_radius * n
                            self.mesh_dy.dx[pi] += self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi] * compliance_col * dp
                            self.mesh_dy.nc[pi] += self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi] * compliance_col
                            ni = self.mesh_dy.num_neighbours[pi]
                            if ni < self.mesh_dy.cache_size:
                                self.mesh_dy.neighbour_ids[pi, ni] = pj
                                self.mesh_dy.num_neighbours[pi] += 1
                            # ni += 1.
                            # C += self.poly6_value(xji.norm(), kernel_radius)
                            # nabla_C += self.spiky_gradient(xji, kernel_radius)

            # dp = (C / (nabla_C.dot(nabla_C) + 1e-3)) * nabla_C
        # for i in range(self.num_edges_dy):
        #     v0, v1 = self.mesh_dy.eid_field[i, 0], self.mesh_dy.eid_field[i, 1]
        #     #     pos_i = 0.5 * (self.mesh_dy.y[v0] + self.mesh_dy.y[v1])
        #     pi = i
        #     pos_i = 0.5 * (self.mesh_dy.y[v0] + self.mesh_dy.y[v1])
        #     C = 0.0
        #     nabla_C = ti.math.vec3(0.0)
        #     # grad_rh0 = ti.math.vec3(0.0)
        #     ni = 0.0
        #     cell_id = self.sh_st.pos_to_cell_id(pos_i)
        #     for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
        #         cell_to_check = cell_id + offs
        #         if self.sh_st.is_in_grid(cell_to_check):
        #             for j in range(self.sh_st.num_particles_in_cell[cell_to_check]):
        #                 pj = self.sh_st.particle_ids_in_cell[cell_to_check, j]
        #                 if pi == pj:
        #                     continue
        #                 pos_j = self.particle_st.x[pj]
        #                 xji = pos_j - pos_i
        #                 # n = self.mesh_dy.num_neighbours[pi]
        #                 if xji.norm() < kernel_radius:
        #                     n = xji.normalized()
        #                     dp = xji - kernel_radius * n
        #                     # dp = (C / (nabla_C.dot(nabla_C) + 1e-3)) * nabla_C
        #                     self.mesh_dy.dx[v0] += 0.5 * self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] * compliance_col * dp
        #                     self.mesh_dy.dx[v1] += 0.5 * self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1] * compliance_col * dp
        #                     self.mesh_dy.nc[v0] += 0.5 * self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] * compliance_col
        #                     self.mesh_dy.nc[v1] += 0.5 * self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1] * compliance_col

            # elif i < self.num_verts_dy + self.num_edges_dy + self.num_faces_dy:
            #     pi = i - self.num_verts_dy - self.num_edges_dy
            #     v0, v1, v2 = self.mesh_dy.face_indices_flatten[3 * pi + 0], self.mesh_dy.face_indices_flatten[3 * pi + 1], self.mesh_dy.face_indices_flatten[3 * pi + 2]
            #     inv3 = 1.0 / 3.0
            #     pos_i = (self.mesh_dy.y[v0] + self.mesh_dy.y[v1] + self.mesh_dy.y[v2]) * inv3
            #     cell_id = self.sh_st.pos_to_cell_id(pos_i)
            #     for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
            #         cell_to_check = cell_id + offs
            #         if self.sh_st.is_in_grid(cell_to_check):
            #             for j in range(self.sh_st.num_particles_in_cell[cell_to_check]):
            #                 pj = self.sh_st.particle_ids_in_cell[cell_to_check, j]
            #                 if pi == pj:
            #                     continue
            #                 pos_j = self.particle_st.x[pj]
            #                 xji = pos_j - pos_i
            #                 if xji.norm() < distance_threshold:
            #                     # C = (xji.norm() - distance_threshold)
            #                     nabla_C = ti.math.normalize(xji)
            #                     # schur = (self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi])
            #                     pij = xji - distance_threshold * nabla_C
            #                     # k = 1e8
            #                     # ld = -(k * C) / (k * schur + 1.0)
            #
            #                     self.mesh_dy.dx[v0] += inv3 * self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] * compliance_col * pij
            #                     self.mesh_dy.dx[v1] += inv3 * self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1] * compliance_col * pij
            #                     self.mesh_dy.dx[v2] += inv3 * self.mesh_dy.fixed[v2] * self.mesh_dy.m_inv[v2] * compliance_col * pij
            #                     # self.dx[pj] += self.iinv3ixed[pj] * self.m_inv[pj] * ld * nabla_C
            #                     self.mesh_dy.nc[v0] += inv3 * self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] * compliance_col
            #                     self.mesh_dy.nc[v1] += inv3 * self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1] * compliance_col
            #                     self.mesh_dy.nc[v2] += inv3 * self.mesh_dy.fixed[v2] * self.mesh_dy.m_inv[v2] * compliance_col

        for pi in range(self.num_verts_dy):
            # if self.mesh_dy.nc[pi] > 0:
            self.mesh_dy.y[pi] += self.mesh_dy.dx[pi] / self.mesh_dy.nc[pi]

    @ti.kernel
    def set_edge_and_face_particles(self):

        for i in range(self.num_edges_dy):
            v0, v1 = self.mesh_dy.eid_field[i, 0], self.mesh_dy.eid_field[i, 1]
            self.mesh_dy.x_e[i] = 0.5 * (self.mesh_dy.x[v0] + self.mesh_dy.x[v1])

        for i in range(self.num_faces_dy):
            v0, v1, v2 = self.mesh_dy.face_indices_flatten[3 * i + 0], self.mesh_dy.face_indices_flatten[3 * i + 1], self.mesh_dy.face_indices_flatten[3 * i + 2]
            self.mesh_dy.x_f[i] = (self.mesh_dy.x[v0] + self.mesh_dy.x[v1] + self.mesh_dy.x[v2]) / 3.0

        for i in range(self.num_verts_st + self.num_edges_st + self.num_faces_st):
            if i < self.num_verts_st:
                pi = i
                self.mesh_st.x_test[i] = self.mesh_st.x[pi]
            elif i < self.num_verts_st + self.num_edges_st:
                pi = i - self.num_verts_st
                v0, v1 = self.mesh_st.eid_field[pi, 0], self.mesh_st.eid_field[pi, 1]
                self.mesh_st.x_test[i] = 0.5 * (self.mesh_st.x[v0] + self.mesh_st.x[v1])
            elif i < self.num_verts_st + self.num_edges_st + self.num_faces_st:
                pi = i - (self.num_verts_st + self.num_edges_st)
                v0, v1, v2 = self.mesh_st.face_indices_flatten[3 * pi + 0], self.mesh_st.face_indices_flatten[3 * pi + 1], self.mesh_st.face_indices_flatten[3 * pi + 2]
                self.mesh_st.x_test[i] = (self.mesh_st.x[v0] + self.mesh_st.x[v1] + self.mesh_st.x[v2]) / 3.0



    def forward(self, n_substeps, n_iter):

        dt_sub = self.dt / n_substeps

        # self.set_edge_and_face_particles()
        self.sh_st.insert_particles_in_grid(self.particle_st.x)
        # self.sh_st.search_neighbours(self.mesh_dy.y, self.mesh_dy.num_neighbours, self.mesh_dy.neighbour_ids, self.mesh_dy.cache_size)
        # self.sh_st_e.search_neighbours(self.mesh_st.x_e)

        for _ in range(n_substeps):
            self.compute_y(self.g, dt_sub)
            for _ in range(n_iter):
                if self.selected_solver_type == 0:
                    self.solve_constraints_jacobi_x(dt_sub)
                elif self.selected_solver_type >= 1:
                    self.solve_constraints_pd_diag_x(dt_sub)
                # elif self.selected_solver_type == 2:
                #     self.solve_constraints_hess_diag_x(dt_sub)
                # elif self.selected_solver_type == 3:
                #     self.solve_constraints_pd_diag_x(dt_sub)
                # elif self.selected_solver_type == 4:
                #     self.solve_constraints_newton_pcg_x(dt_sub, self.max_cg_iter, self.threshold)
                # elif self.selected_solver_type == 5:
                #     self.solve_constraints_pd_pcg_x(dt_sub, self.max_cg_iter, self.threshold)
            self.compute_v(damping=self.damping, dt=dt_sub)
            if self.enable_velocity_update:
                self.solve_constraints_pd_diag_v()

            if self.selected_solver_type == 0:
                self.solve_constraints_jacobi_x(dt_sub)
            elif self.selected_solver_type >= 1:
                self.solve_constraints_pd_diag_x(dt_sub)
            self.update_x(dt_sub)
        self.set_edge_and_face_particles()