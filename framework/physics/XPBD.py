import taichi as ti
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
        # self.mesh_st = mesh_st
        self.particle_st = particle_st
        self.dHat = dHat
        self.sh_dy = sh_dy
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
        self.selected_solver_type = 2
        self.definiteness_fix = True
        self.print_stats = False
        self.use_line_search = True
        self.enable_velocity_update = True

        self.num_verts_dy = self.mesh_dy.num_verts
        self.num_edges_dy = self.mesh_dy.num_edges
        self.num_faces_dy = self.mesh_dy.num_faces
        # if mesh_st is not None:
        #     self.num_verts_st = self.mesh_st.num_verts
        #     self.num_edges_st = self.mesh_st.num_edges
        #     self.num_faces_st = self.mesh_st.num_faces
        # else:
        #     self.num_verts_st = 0
        #     self.num_edges_st = 0
        #     self.num_faces_st = 0

        self.reset()

    ####################################################################################################################

    @ti.kernel
    def set_fixed_vertices(self, fixed_vertices: ti.template()):
        for i in range(self.num_verts_dy):
            if fixed_vertices[i] >= 1:
                self.mesh_dy.fixed[i] = 0.0
            else:
                self.mesh_dy.fixed[i] = 1.0


    @ti.kernel
    def reset_rho(self):

        center = ti.math.vec3(0, 0, 0)
        for i in range(self.particle_st.num_particles):
            center += self.particle_st.x[i]

        center /= self.particle_st.num_particles

        surface_radius = 0.0
        for i in range(self.particle_st.num_particles):
            l = (self.particle_st.x[i] - surface_radius).norm()
            ti.atomic_max(surface_radius, l)

        # print(surface_radius)
        for i in range(self.particle_st.num_particles):
            l = (self.particle_st.x[i] - center).norm()
            self.particle_st.rho0[i] = self.poly6_value(l, 1.5 * surface_radius)

        # self.particle_st.rho0.fill(10.0)

    def reset(self):
        print("reset...")
        self.mesh_dy.reset()
        self.update_sample_particle_pos()

        self.particle_st.reset()
        self.particle_st.v.fill(0.0)
        # self.particle_st.rho0.fill(1.0)
        # self.reset_rho()
        # if self.mesh_st is None:
        #     self.mesh_st.reset()
        self.update_sample_particle_pos()

        self.sh_dy.insert_particles_in_grid(self.mesh_dy.x_sample)
        self.init_rest_neighbours(2*self.dHat)


    @ti.kernel
    def init_rest_neighbours(self, kernel_radius: float):

        self.mesh_dy.num_neighbours_rest.fill(0)
        # self.mesh_dy.rho0.fill(0.0)

        # for i in range(self.num_verts_dy):
        #     pi = i
        #     pos_i = self.mesh_dy.x[pi]
        #     color_i = self.mesh_dy.colors[pi]
        #     cell_id = self.sh_dy.pos_to_cell_id(pos_i)
        #     for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
        #         cell_to_check = cell_id + offs
        #         if self.sh_dy.is_in_grid(cell_to_check):
        #             for j in range(self.sh_dy.num_particles_in_cell[cell_to_check]):
        #                 pj = self.sh_dy.particle_ids_in_cell[cell_to_check, j]
        #                 color_j = self.mesh_dy.colors[pj]
        #                 if pi == pj:
        #                     continue
        #                 pos_j = self.mesh_dy.x[pj]
        #                 xji = pos_j - pos_i
        #
        #                 if (color_i - color_j).norm() < 1e-3:
        #                     self.mesh_dy.rho0[i] += self.poly6_value(xji.norm(), kernel_radius)
        #
        #                 n = self.mesh_dy.num_neighbours_rest[pi]
        #                 if xji.norm() < kernel_radius and n < self.mesh_dy.cache_size:
        #                     # self.mesh_dy.rho0[pi] +=
        #                     self.mesh_dy.neighbour_ids_rest[pi, n] = pj
        #                     self.mesh_dy.num_neighbours_rest[pi] += 1

        self.mesh_dy.rho0_sample.fill(0.0)
        for i in self.mesh_dy.x_sample:
            pi = i
            pos_i = self.get_sample_pos(i, self.mesh_dy.x)
            # color_i = self.get_sample_pos(i, self.mesh_dy.colors)
            cell_id = self.sh_dy.pos_to_cell_id(pos_i)
            for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                cell_to_check = cell_id + offs
                if self.sh_dy.is_in_grid(cell_to_check):
                    for j in range(self.sh_dy.num_particles_in_cell[cell_to_check]):
                        pj = self.sh_dy.particle_ids_in_cell[cell_to_check, j]
                        # color_j = self.mesh_dy.colors[pj]
                        if pi == pj:
                            continue
                        pos_j = self.get_sample_pos(pj, self.mesh_dy.x)
                        xji = pos_j - pos_i

                        # if (color_i - color_j).norm() < 1e-3:
                        self.mesh_dy.rho0_sample[i] += self.poly6_value(xji.norm(), kernel_radius)

                        n = self.mesh_dy.num_neighbours_rest[pi]
                        if xji.norm() < kernel_radius and n < self.mesh_dy.cache_size:
                            # self.mesh_dy.rho0[pi] +=
                            self.mesh_dy.neighbour_ids_rest[pi, n] = pj
                            self.mesh_dy.num_neighbours_rest[pi] += 1


    @ti.func
    def is_neighbour(self, id_i, id_j, num_neighbours, neighbour_ids):

        n = 0
        for j in range(num_neighbours[id_i]):
            if id_j == neighbour_ids[id_i, j]:
                n += 1

        return n

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
    def compute_particle_v(self, dt: float):
        for i in self.particle_st.x_prev:
            # self.particle_st.x_prev[i] = self.particle_st.x[i]
            # self.particle_st.x_current[i] = self.particle_st.x_prev[i] + ti.math.vec3([0., 0.001, 0.0])
            self.particle_st.v[i] = (self.particle_st.x_current[i] - self.particle_st.x_prev[i]) / dt

    @ti.kernel
    def compute_y(self, g: ti.math.vec3, dt: ti.f32):
        # compute apporximate x_(t+1) (== y) by explicit way before projecting constraints to x_(t+1)...
        for i in range(self.num_verts_dy):
            self.mesh_dy.y[i] = self.mesh_dy.x[i] + self.mesh_dy.fixed[i] * (self.mesh_dy.v[i] * dt + g * dt * dt)
            self.mesh_dy.y[i] = self.confine_boundary(self.mesh_dy.y[i])
            self.mesh_dy.y_origin[i] = self.mesh_dy.y[i]

        for i in self.particle_st.x:
            self.particle_st.x[i] += self.particle_st.v[i] * dt



    @ti.kernel
    def update_dx(self):
        # update x_(t+1) by adding dx (used in the Jacobi solver)
        for i in range(self.num_verts_dy):
            if self.mesh_dy.nc[i] > 0:
                self.mesh_dy.y[i] += self.mesh_dy.fixed[i] * (self.mesh_dy.dx[i] / self.mesh_dy.nc[i])

    @ti.kernel
    def compute_v(self, dt: ti.f32):
        # compute v after all constraints are projected to x_(t+1)
        for i in range(self.num_verts_dy):
            self.mesh_dy.v[i] = self.mesh_dy.fixed[i] * (self.mesh_dy.y[i] - self.mesh_dy.x[i]) / dt


        # self.particle_st.x_prev.copy_from(self.particle_st.x)

    @ti.kernel
    def update_x(self, damping: float, dt: float):
        # eventually, update actual position x_(t+1) after velocities are computed...
        for i in range(self.num_verts_dy):
            self.mesh_dy.x[i] += self.mesh_dy.fixed[i] * dt * (1.0 - damping) * self.mesh_dy.v[i]


        # center = ti.math.vec3(0.0)
        # for i in range(self.particle_st.num_particles):
        #     center += self.particle_st.x[i]
        #
        # center /= self.particle_st.num_particles
        #
        # theta = 0.003
        # rot = ti.math.mat3([[ti.cos(theta), 0, ti.sin(theta)],
        #                     [0, 1, 0],
        #                     [-ti.sin(theta), 0, ti.cos(theta)]])
        for i in self.particle_st.x:
            # ri = self.particle_st.x[i] - center
            # x_new = rot @ ri + center
            # x_new = rot @ ri + center
            # self.particle_st.v[i] = ti.math.vec3([0.0, 0.5, 0.0])
            self.particle_st.x[i] += self.particle_st.v[i] * dt
        #


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
        # if self.selected_solver_type == 1:
        #     print("test")
        self.solve_collision_constraints_st_pd_diag_x(2 * self.dHat, compliance_collision)
        self.solve_collision_constraints_dy_pd_diag_x(2 * self.dHat, compliance_collision)

        # elif self.selected_solver_type == 2:
        #     print("test")
        # self.solve_collision_constraints_st_pd_diag_test_x(2 * self.dHat, compliance_collision)
        #     self.solve_collision_constraints_dy_pd_diag_test_x(2 * self.dHat, compliance_collision)

    def solve_constraints_pd_diag_v(self):

        # if self.selected_solver_type == 1:
        self.solve_collision_constraints_st_pd_diag_rho_v(2*self.dHat, self.mu)
        # elif self.selected_solver_type == 2:
        # self.solve_collision_constraints_st_pd_diag_v(self.mu)

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


        for i in self.mesh_dy.bending_l0:
            l0 = self.mesh_dy.bending_l0[i]
            v0, v1 = self.mesh_dy.bending_indices[2 * i + 0], self.mesh_dy.bending_indices[2 * i + 1]
            x01 = self.mesh_dy.y[v0] - self.mesh_dy.y[v1]
            dp01 = x01 - l0 * x01.normalized()

            # alpha = (1.0 - l0 / x01.norm())

            self.mesh_dy.dx[v0] -= self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] * compliance_bending * dp01
            self.mesh_dy.dx[v1] += self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1] * compliance_bending * dp01

            self.mesh_dy.nc[v0] += self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] * compliance_bending
            self.mesh_dy.nc[v1] += self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1] * compliance_bending

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
        self.mesh_dy.num_neighbours.fill(0)

        # for i in range(self.num_verts_dy):
        #     pi = i
        #     pos_i = self.mesh_dy.y[pi]
        #     C = 0.0
        #     schur = 0.0
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
        #                 if xji.norm() < kernel_radius:
        #                     C += self.particle_st.rho0[pj] * self.poly6_value(xji.norm(), kernel_radius)
        #                     nabla_Cji = self.particle_st.rho0[pj] * self.spiky_gradient(xji, kernel_radius)
        #                     schur += nabla_Cji.dot(nabla_Cji)
        #
        #                     ni = self.mesh_dy.num_neighbours[pi]
        #                     if ni < self.mesh_dy.cache_size:
        #                         self.mesh_dy.neighbour_ids[pi, ni] = pj
        #                         self.mesh_dy.num_neighbours[pi] += 1
        #
        #     ld = ti.max((C / (schur + 1e-3)), 0.0)
        #
        #     for j in range(self.mesh_dy.num_neighbours[pi]):
        #         pj = self.mesh_dy.neighbour_ids[pi, j]
        #         pos_j = self.particle_st.x[pj]
        #         xji = pos_j - pos_i
        #         nabla_Cji = self.particle_st.rho0[pj] * self.spiky_gradient(xji, kernel_radius)
        #         pji = -ld * nabla_Cji
        #
        #         self.mesh_dy.dx[pi] -= self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi] * compliance_col * pji
        #         self.mesh_dy.nc[pi] += self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi] * compliance_col

        for i in self.mesh_dy.x_sample:
            pi = i
            pos_i = self.get_sample_pos(i, self.mesh_dy.y)
            C = 0.0
            schur = 0.0
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
                        if xji.norm() < kernel_radius:
                            C += self.particle_st.rho0[pj] * self.poly6_value(xji.norm(), kernel_radius)
                            nabla_Cji = self.particle_st.rho0[pj] * self.spiky_gradient(xji, kernel_radius)
                            schur += nabla_Cji.dot(nabla_Cji)

                            ni = self.mesh_dy.num_neighbours[pi]
                            if ni < self.mesh_dy.cache_size:
                                self.mesh_dy.neighbour_ids[pi, ni] = pj
                                self.mesh_dy.num_neighbours[pi] += 1

            ld = ti.max((C / (schur + 1e-3)), 0.0)
            fid = ti.cast(self.mesh_dy.sample_indices[i, 0], int)
            u, v, w = self.mesh_dy.sample_indices[i, 1], self.mesh_dy.sample_indices[i, 2], self.mesh_dy.sample_indices[i, 3]
            v0, v1, v2 = self.mesh_dy.face_indices_flatten[3 * fid + 0], self.mesh_dy.face_indices_flatten[3 * fid + 1], self.mesh_dy.face_indices_flatten[3 * fid + 2]

            for j in range(self.mesh_dy.num_neighbours[pi]):
                pj = self.mesh_dy.neighbour_ids[pi, j]
                pos_j = self.particle_st.x[pj]
                xji = pos_j - pos_i
                nabla_Cji = self.particle_st.rho0[pj] * self.spiky_gradient(xji, kernel_radius)
                pji = -ld * nabla_Cji

                self.mesh_dy.dx[v0] -= self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] * u * compliance_col * pji
                self.mesh_dy.dx[v1] -= self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1] * v * compliance_col * pji
                self.mesh_dy.dx[v2] -= self.mesh_dy.fixed[v2] * self.mesh_dy.m_inv[v2] * w * compliance_col * pji

                self.mesh_dy.nc[v0] += self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] * u * compliance_col
                self.mesh_dy.nc[v1] += self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1] * v * compliance_col
                self.mesh_dy.nc[v2] += self.mesh_dy.fixed[v2] * self.mesh_dy.m_inv[v2] * w * compliance_col


        for pi in range(self.num_verts_dy):
            # if self.mesh_dy.nc[pi] > 0:
            self.mesh_dy.y[pi] += self.mesh_dy.dx[pi] / self.mesh_dy.nc[pi]

    @ti.kernel
    def solve_collision_constraints_dy_pd_diag_x(self, kernel_radius: float, compliance_col: float):

        self.mesh_dy.dx.fill(0.0)
        self.mesh_dy.nc.fill(1.0)
        self.mesh_dy.num_neighbours_dy.fill(0)

        # for i in range(self.num_verts_dy):
        #     # if i < self.num_verts_dy:
        #     pi = i
        #     pos_i = self.mesh_dy.y[pi]
        #
        #     C = -self.mesh_dy.rho0[i]
        #
        #     nabla_C = ti.math.vec3(0.0)
        #     schur = 0.0
        #     cell_id = self.sh_dy.pos_to_cell_id(pos_i)
        #     for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
        #         cell_to_check = cell_id + offs
        #         if self.sh_dy.is_in_grid(cell_to_check):
        #             for j in range(self.sh_dy.num_particles_in_cell[cell_to_check]):
        #                 pj = self.sh_dy.particle_ids_in_cell[cell_to_check, j]
        #                 if pi == pj:
        #                     continue
        #                 pos_j = self.mesh_dy.y[pj]
        #                 xji = pos_j - pos_i
        #                 if xji.norm() < kernel_radius:
        #                     C += self.poly6_value(xji.norm(), kernel_radius)
        #                     nabla_Cji = self.spiky_gradient(xji, kernel_radius)
        #                     schur += nabla_Cji.dot(nabla_Cji)
        #
        #                     nabla_C += nabla_Cji
        #                     ni = self.mesh_dy.num_neighbours_dy[pi]
        #                     if ni < self.mesh_dy.cache_size:
        #                         self.mesh_dy.neighbour_ids_dy[pi, ni] = pj
        #                         self.mesh_dy.num_neighbours_dy[pi] += 1
        #
        #
        #     ld = ti.max((C / (schur + 1e-3)), 0.0)
        #     for j in range(self.mesh_dy.num_neighbours_dy[pi]):
        #         pj = self.mesh_dy.neighbour_ids_dy[pi, j]
        #         pos_j = self.mesh_dy.y[pj]
        #         xji = pos_j - pos_i
        #         nabla_Cji = self.spiky_gradient(xji, kernel_radius)
        #         pji = -ld * nabla_Cji
        #
        #         self.mesh_dy.dx[pi] -= self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi] * compliance_col * pji
        #         self.mesh_dy.nc[pi] += self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi] * compliance_col
        #
        #         self.mesh_dy.dx[pj] += self.mesh_dy.fixed[pj] * self.mesh_dy.m_inv[pj] * compliance_col * pji
        #         self.mesh_dy.nc[pj] += self.mesh_dy.fixed[pj] * self.mesh_dy.m_inv[pj] * compliance_col

        for i in self.mesh_dy.x_sample:
            # if i < self.num_verts_dy:
            pi = i
            pos_i = self.get_sample_pos(i, self.mesh_dy.y)
            C = -self.mesh_dy.rho0_sample[i]
            nabla_C = ti.math.vec3(0.0)
            schur = 0.0
            cell_id = self.sh_dy.pos_to_cell_id(pos_i)
            for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                cell_to_check = cell_id + offs
                if self.sh_dy.is_in_grid(cell_to_check):
                    for j in range(self.sh_dy.num_particles_in_cell[cell_to_check]):
                        pj = self.sh_dy.particle_ids_in_cell[cell_to_check, j]
                        if pi == pj:
                            continue
                        pos_j = self.get_sample_pos(pj, self.mesh_dy.y)
                        xji = pos_j - pos_i
                        if xji.norm() < kernel_radius:
                            C += self.poly6_value(xji.norm(), kernel_radius)
                            nabla_Cji = self.spiky_gradient(xji, kernel_radius)
                            schur += nabla_Cji.dot(nabla_Cji)

                            nabla_C += nabla_Cji
                            ni = self.mesh_dy.num_neighbours_dy[pi]
                            if ni < self.mesh_dy.cache_size:
                                self.mesh_dy.neighbour_ids_dy[pi, ni] = pj
                                self.mesh_dy.num_neighbours_dy[pi] += 1

            fid = ti.cast(self.mesh_dy.sample_indices[i, 0], int)
            u, v, w = self.mesh_dy.sample_indices[i, 1], self.mesh_dy.sample_indices[i, 2], self.mesh_dy.sample_indices[i, 3]
            v0, v1, v2 = self.mesh_dy.face_indices_flatten[3 * fid + 0], self.mesh_dy.face_indices_flatten[3 * fid + 1], self.mesh_dy.face_indices_flatten[3 * fid + 2]

            ld = ti.max((C / (schur + 1e-3)), 0.0)
            # ld = 0.0
            for j in range(self.mesh_dy.num_neighbours_dy[pi]):
                pj = self.mesh_dy.neighbour_ids_dy[pi, j]
                fj = ti.cast(self.mesh_dy.sample_indices[pj, 0], int)
                uj, vj, wj = self.mesh_dy.sample_indices[pj, 1], self.mesh_dy.sample_indices[pj, 2], self.mesh_dy.sample_indices[pj, 3]
                vj0, vj1, vj2 = self.mesh_dy.face_indices_flatten[3 * fj + 0], self.mesh_dy.face_indices_flatten[3 * fj + 1], self.mesh_dy.face_indices_flatten[3 * fj + 2]
                pos_j = self.get_sample_pos(pj, self.mesh_dy.y)
                xji = pos_j - pos_i
                nabla_Cji = self.spiky_gradient(xji, kernel_radius)
                pji = -ld * nabla_Cji

                self.mesh_dy.dx[v0] -= self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] * u * compliance_col * pji
                self.mesh_dy.dx[v1] -= self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1] * v * compliance_col * pji
                self.mesh_dy.dx[v2] -= self.mesh_dy.fixed[v2] * self.mesh_dy.m_inv[v2] * w * compliance_col * pji

                self.mesh_dy.nc[v0] += self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] * u * compliance_col
                self.mesh_dy.nc[v1] += self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1] * v * compliance_col
                self.mesh_dy.nc[v2] += self.mesh_dy.fixed[v2] * self.mesh_dy.m_inv[v2] * w * compliance_col

                self.mesh_dy.dx[vj0] += self.mesh_dy.fixed[vj0] * self.mesh_dy.m_inv[vj0] * uj * compliance_col * pji
                self.mesh_dy.dx[vj1] += self.mesh_dy.fixed[vj1] * self.mesh_dy.m_inv[vj1] * vj * compliance_col * pji
                self.mesh_dy.dx[vj2] += self.mesh_dy.fixed[vj2] * self.mesh_dy.m_inv[vj2] * wj * compliance_col * pji

                self.mesh_dy.nc[vj0] += self.mesh_dy.fixed[vj0] * self.mesh_dy.m_inv[vj0] * uj * compliance_col
                self.mesh_dy.nc[vj1] += self.mesh_dy.fixed[vj1] * self.mesh_dy.m_inv[vj1] * vj * compliance_col
                self.mesh_dy.nc[vj2] += self.mesh_dy.fixed[vj2] * self.mesh_dy.m_inv[vj2] * wj * compliance_col


        for pi in range(self.num_verts_dy):
            # if self.mesh_dy.nc[pi] > 0:
            self.mesh_dy.y[pi] += self.mesh_dy.dx[pi] / self.mesh_dy.nc[pi]

    @ti.kernel
    def solve_collision_constraints_st_pd_diag_v(self, mu: float):

        # print(mu)
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
                vi = self.mesh_dy.v[pi]
                cv = n.dot(vi)
                if cv > 0.0:
                    v_tan = vi - cv * n
                    if v_tan.norm() <= mu * ti.abs(cv):
                        v_tan = ti.math.vec3(0.0)
                    else:
                        t = v_tan.normalized()
                        v_tan = v_tan - mu * cv * t

                    dv = v_tan - self.mesh_dy.v[pi]
                    self.mesh_dy.dx[pi] += dv
                    self.mesh_dy.nc[pi] += 1.0

        for pi in range(self.num_verts_dy):
            if self.mesh_dy.nc[pi] > 0:
                self.mesh_dy.v[pi] += self.mesh_dy.dx[pi] / self.mesh_dy.nc[pi]

        self.mesh_dy.dx.fill(0.0)
        self.mesh_dy.nc.fill(1.0)

        k = 1e8
        for i in range(self.num_verts_dy):
            pi = i
            pos_i = self.mesh_dy.y[pi]
            for j in range(self.mesh_dy.num_neighbours_dy[pi]):
                pj = self.mesh_dy.neighbour_ids_dy[pi, j]
                pos_j = self.mesh_dy.y[pj]
                xji = pos_j - pos_i
                n = xji.normalized()
                vji = self.mesh_dy.v[pj] - self.mesh_dy.v[pi]
                cv = n.dot(vji)
                if cv < 0.0:
                    dvji_tan = vji - cv * n
                    if dvji_tan.norm() <= mu * ti.abs(cv):
                        dvji_tan = ti.math.vec3(0.0)
                    else:
                        t = dvji_tan.normalized()
                        dvji_tan = dvji_tan + mu * cv * t

                    dvji = vji - dvji_tan
                    self.mesh_dy.dx[pi] += k * self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi] * dvji
                    self.mesh_dy.dx[pj] -= k * self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pj] * dvji
                    self.mesh_dy.nc[pi] += k * self.mesh_dy.fixed[pj] * self.mesh_dy.m_inv[pi]
                    self.mesh_dy.nc[pj] += k * self.mesh_dy.fixed[pj] * self.mesh_dy.m_inv[pj]
        #
        for pi in range(self.num_verts_dy):
            # if self.mesh_dy.nc[pi] > 0:
            self.mesh_dy.v[pi] += self.mesh_dy.dx[pi] / self.mesh_dy.nc[pi]

    @ti.kernel
    def solve_collision_constraints_st_pd_diag_rho_v(self, kernel_radius: float, mu: float):

        # print("fuck")
        self.mesh_dy.dx.fill(0.0)
        self.mesh_dy.nc.fill(1.0)

        k_n = 1e8
        k_f = 1e8
        # for i in range(self.num_verts_dy):
        #     # if i < self.num_verts_dy:
        #     pi = i
        #     pos_i = self.mesh_dy.y[pi]
        #     # cv = 0.0
        #     nabla_c = ti.math.vec3(0.0)
        #     schur = 0.0
        #     for j in range(self.mesh_dy.num_neighbours[pi]):
        #         pj = self.mesh_dy.neighbour_ids[pi, j]
        #         pos_j = self.particle_st.x[pj]
        #         xji = pos_j - pos_i
        #         nabla_cji = self.particle_st.rho0[pj] * self.spiky_gradient(xji, kernel_radius)
        #         nabla_c += nabla_cji
        #         schur += nabla_cji.dot(nabla_cji)
        #         # nabla_c += nabla_c_ji
        #         # vi = self.mesh_dy.v[pi]
        #         # cv += nabla_c_ji.dot(vi)
        #
        #     cv = nabla_c.dot(-self.mesh_dy.v[pi])
        #     if cv > 0.0:
        #         # schur = nabla_c.dot(nabla_c)
        #         for j in range(self.mesh_dy.num_neighbours[pi]):
        #             pj = self.mesh_dy.neighbour_ids[pi, j]
        #             pos_j = self.particle_st.x[pj]
        #             xji = pos_j - pos_i
        #             # nabla_cji = self.spiky_gradient(xji, kernel_radius)
        #             ld = cv / (schur + 1e-3)
        #             pji_nor = - ld * self.particle_st.rho0[pj] * self.spiky_gradient(xji, kernel_radius)
        #             # dvji = pji + self.mesh_dy.v[pi]
        #             f_n = k_n * pji_nor
        #             df_n_dx = k_n
        #             self.mesh_dy.dx[pi] -= self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi] * f_n
        #             self.mesh_dy.nc[pi] += self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi] * df_n_dx
        #
        #             u = self.mesh_dy.v[pi] + pji_nor
        #             if u.norm() <= mu * f_n.norm():
        #                 f_f = -k_f * u
        #                 df_f_dx = k_f
        #                 self.mesh_dy.dx[pi] += self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi] * f_f
        #                 self.mesh_dy.nc[pi] += self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi] * df_f_dx
        #             else:
        #                 df_f_dx = mu * (f_n.norm() / u.norm())
        #                 f_f = -df_f_dx * u
        #                 df_f_dx = df_f_dx
        #                 self.mesh_dy.dx[pi] += self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi] * f_f
        #                 self.mesh_dy.nc[pi] += self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi] * df_f_dx

        for i in self.mesh_dy.x_sample:
            # if i < self.num_verts_dy:
            pi = i
            # if i == 5:
            #     print(self.mesh_dy.num_neighbours[pi])
            fid = ti.cast(self.mesh_dy.sample_indices[i, 0], int)
            u, v, w = self.mesh_dy.sample_indices[i, 1], self.mesh_dy.sample_indices[i, 2], self.mesh_dy.sample_indices[i, 3]
            v0, v1, v2 = self.mesh_dy.face_indices_flatten[3 * fid + 0], self.mesh_dy.face_indices_flatten[3 * fid + 1], self.mesh_dy.face_indices_flatten[3 * fid + 2]
            pos_i = u * self.mesh_dy.y[v0] + v * self.mesh_dy.y[v1] + w * self.mesh_dy.y[v2]
            vel_i = u * self.mesh_dy.v[v0] + v * self.mesh_dy.v[v1] + w * self.mesh_dy.v[v2]

            # cv = 0.0
            nabla_c = ti.math.vec3(0.0)
            schur = 0.0
            cv = 0.0
            for j in range(self.mesh_dy.num_neighbours[pi]):
                pj = self.mesh_dy.neighbour_ids[pi, j]
                pos_j = self.particle_st.x[pj]
                xji = pos_j - pos_i
                nabla_cji = self.particle_st.rho0[pj] * self.spiky_gradient(xji, kernel_radius)
                nabla_c += nabla_cji
                schur += nabla_cji.dot(nabla_cji)
                # nabla_c += nabla_c_ji
                # vi = self.mesh_dy.v[pi]
                cv += nabla_cji.dot(self.particle_st.v[pj] - vel_i)

            # cv = nabla_c.dot(-vel_i)
            if cv > 0.0:
                # schur = nabla_c.dot(nabla_c)
                for j in range(self.mesh_dy.num_neighbours[pi]):
                    pj = self.mesh_dy.neighbour_ids[pi, j]
                    pos_j = self.particle_st.x[pj]
                    xji = pos_j - pos_i
                    # nabla_cji = self.spiky_gradient(xji, kernel_radius)
                    ld = cv / (schur + 1e-3)
                    pji_nor = -ld * self.particle_st.rho0[pj] * self.spiky_gradient(xji, kernel_radius)
                    # dvji = pji + self.mesh_dy.v[pi]
                    f_n = k_n * pji_nor
                    df_n_dx = k_n

                    self.mesh_dy.dx[v0] -= self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] * u * f_n
                    self.mesh_dy.dx[v1] -= self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1] * v * f_n
                    self.mesh_dy.dx[v2] -= self.mesh_dy.fixed[v2] * self.mesh_dy.m_inv[v2] * w * f_n

                    self.mesh_dy.nc[v0] += self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] * u * df_n_dx
                    self.mesh_dy.nc[v1] += self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1] * v * df_n_dx
                    self.mesh_dy.nc[v2] += self.mesh_dy.fixed[v2] * self.mesh_dy.m_inv[v2] * w * df_n_dx

                    t = vel_i - self.particle_st.v[pj] - pji_nor
                    u_norm = ti.sqrt(ti.math.dot(t,t))
                    f_n_norm = ti.sqrt(ti.math.dot(f_n, f_n))
                    if  u_norm <= mu * f_n_norm:
                        f_f = -k_f * t
                        df_f_dx = k_f
                        self.mesh_dy.dx[v0] += self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] * u * f_f
                        self.mesh_dy.dx[v1] += self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1] * v * f_f
                        self.mesh_dy.dx[v2] += self.mesh_dy.fixed[v2] * self.mesh_dy.m_inv[v2] * w * f_f

                        self.mesh_dy.nc[v0] += self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] * u * df_f_dx
                        self.mesh_dy.nc[v1] += self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1] * v * df_f_dx
                        self.mesh_dy.nc[v2] += self.mesh_dy.fixed[v2] * self.mesh_dy.m_inv[v2] * w * df_f_dx
                    else:
                        df_f_dx = mu * (f_n_norm / u_norm)
                        f_f = -df_f_dx * t
                        df_f_dx = df_f_dx
                        self.mesh_dy.dx[v0] += self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] * u * f_f
                        self.mesh_dy.dx[v1] += self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1] * v * f_f
                        self.mesh_dy.dx[v2] += self.mesh_dy.fixed[v2] * self.mesh_dy.m_inv[v2] * w * f_f

                        self.mesh_dy.nc[v0] += self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] * u * df_f_dx
                        self.mesh_dy.nc[v1] += self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1] * v * df_f_dx
                        self.mesh_dy.nc[v2] += self.mesh_dy.fixed[v2] * self.mesh_dy.m_inv[v2] * w * df_f_dx


        for pi in range(self.num_verts_dy):
            # if self.mesh_dy.nc[pi] > 0:
            self.mesh_dy.v[pi] += self.mesh_dy.dx[pi] / self.mesh_dy.nc[pi]

        # self.mesh_dy.dx.fill(0.0)
        # self.mesh_dy.nc.fill(1.0)
        #
        # for i in range(self.num_verts_dy):
        #     pi = i
        #     pos_i = self.mesh_dy.y[pi]
        #     c = 0.0
        #     schur = 0.0
        #     for j in range(self.mesh_dy.num_neighbours_dy[pi]):
        #         pj = self.mesh_dy.neighbour_ids_dy[pi, j]
        #         pos_j = self.mesh_dy.y[pj]
        #         xji = pos_j - pos_i
        #         vji =  self.mesh_dy.v[pj] -  self.mesh_dy.v[pi]
        #         nabla_cji = self.spiky_gradient(xji, kernel_radius)
        #         schur += nabla_cji.dot(nabla_cji)
        #         c += nabla_cji.dot(vji)
        #
        #     if c > 0.0:
        #         ld = c / (schur + 1e-3)
        #         for j in range(self.mesh_dy.num_neighbours_dy[pi]):
        #             pj = self.mesh_dy.neighbour_ids_dy[pi, j]
        #             pos_j = self.mesh_dy.y[pj]
        #             xji = pos_j - pos_i
        #             pji_nor = -ld * self.spiky_gradient(xji, kernel_radius)
        #             f_n = k_n * pji_nor
        #             df_n_dx = k_n
        #             self.mesh_dy.dx[pi] -= self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi] * f_n
        #             self.mesh_dy.dx[pj] += self.mesh_dy.fixed[pj] * self.mesh_dy.m_inv[pj] * f_n
        #             self.mesh_dy.nc[pi] += self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi] * df_n_dx
        #             self.mesh_dy.nc[pj] += self.mesh_dy.fixed[pj] * self.mesh_dy.m_inv[pj] * df_n_dx
        #
        #             uji = (self.mesh_dy.v[pj] - self.mesh_dy.v[pi]) + pji_nor
        #             if uji.norm() <= mu * f_n.norm():
        #                 f_f = -k_f * uji
        #                 df_f_dx = k_f
        #                 self.mesh_dy.dx[pi] -= self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi] * f_f
        #                 self.mesh_dy.dx[pj] += self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi] * f_f
        #                 self.mesh_dy.nc[pi] += self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi] * df_f_dx
        #                 self.mesh_dy.nc[pj] += self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi] * df_f_dx
        #             else:
        #                 df_f_dx = mu * (f_n.norm() / uji.norm())
        #                 f_f = -df_f_dx * uji
        #                 df_f_dx = df_f_dx
        #                 self.mesh_dy.dx[pi] -= self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi] * f_f
        #                 self.mesh_dy.dx[pj] += self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi] * f_f
        #                 self.mesh_dy.nc[pi] += self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi] * df_f_dx
        #                 self.mesh_dy.nc[pj] += self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi] * df_f_dx
        #
        # for pi in range(self.num_verts_dy):
        #     # if self.mesh_dy.nc[pi] > 0:
        #     self.mesh_dy.v[pi] += self.mesh_dy.dx[pi] / self.mesh_dy.nc[pi]

        # print(n)


    @ti.kernel
    def solve_collision_constraints_st_pd_diag_test_x(self, kernel_radius: float, compliance_col: float):

        self.mesh_dy.dx.fill(0.0)
        self.mesh_dy.nc.fill(1.0)
        self.mesh_dy.num_neighbours.fill(0)
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
    def solve_collision_constraints_dy_pd_diag_test_x(self, kernel_radius: float, compliance_col: float):

        self.mesh_dy.dx.fill(0.0)
        self.mesh_dy.nc.fill(1.0)
        self.mesh_dy.num_neighbours_dy.fill(0)

        for i in range(self.num_verts_dy):
            # if i < self.num_verts_dy:
            pi = i
            pos_i = self.mesh_dy.y[pi]
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

                        pos_j = self.mesh_dy.y[pj]
                        xji = pos_j - pos_i
                        # n = self.mesh_dy.num_neighbours[pi]
                        if xji.norm() < kernel_radius:
                            n = xji.normalized()
                            dp = xji - kernel_radius * n
                            self.mesh_dy.dx[pi] += self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi] * compliance_col * dp
                            self.mesh_dy.dx[pj] -= self.mesh_dy.fixed[pj] * self.mesh_dy.m_inv[pj] * compliance_col * dp
                            self.mesh_dy.nc[pi] += self.mesh_dy.fixed[pi] * self.mesh_dy.m_inv[pi] * compliance_col
                            self.mesh_dy.nc[pj] += self.mesh_dy.fixed[pj] * self.mesh_dy.m_inv[pj] * compliance_col
                            ni = self.mesh_dy.num_neighbours_dy[pi]
                            if ni < self.mesh_dy.cache_size_dy:
                                self.mesh_dy.neighbour_ids_dy[pi, ni] = pj
                                self.mesh_dy.num_neighbours_dy[pi] += 1

                            # ni = self.mesh_dy.num_neighbours[pi]
                            # if ni < self.mesh_dy.cache_size:
                            #     self.mesh_dy.neighbour_ids[pi, ni] = pj
                            #     self.mesh_dy.num_neighbours[pi] += 1
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


    @ti.func
    def get_sample_pos(self, i, x):
        fid = ti.cast(self.mesh_dy.sample_indices[i, 0], int)
        u, v, w = self.mesh_dy.sample_indices[i, 1], self.mesh_dy.sample_indices[i, 2], self.mesh_dy.sample_indices[i, 3]
        v0, v1, v2 = self.mesh_dy.face_indices_flatten[3 * fid + 0], self.mesh_dy.face_indices_flatten[3 * fid + 1], self.mesh_dy.face_indices_flatten[3 * fid + 2]
        return u * x[v0] + v * x[v1] + w * x[v2]


    @ti.kernel
    def update_sample_particle_pos(self):
        for i in self.mesh_dy.x_sample:
            self.mesh_dy.x_sample[i] = self.get_sample_pos(i, self.mesh_dy.x)
        #
        # for i in range(self.num_edges_dy):
        #     v0, v1 = self.mesh_dy.eid_field[i, 0], self.mesh_dy.eid_field[i, 1]
        #     self.mesh_dy.x_e[i] = 0.5 * (self.mesh_dy.x[v0] + self.mesh_dy.x[v1])
        #
        # for i in range(self.num_faces_dy):
        #     v0, v1, v2 = self.mesh_dy.face_indices_flatten[3 * i + 0], self.mesh_dy.face_indices_flatten[3 * i + 1], self.mesh_dy.face_indices_flatten[3 * i + 2]
        #     self.mesh_dy.x_f[i] = (self.mesh_dy.x[v0] + self.mesh_dy.x[v1] + self.mesh_dy.x[v2]) / 3.0
        #
        # for i in range(self.num_verts_st + self.num_edges_st + self.num_faces_st):
        #     if i < self.num_verts_st:
        #         pi = i
        #         self.mesh_st.x_test[i] = self.mesh_st.x[pi]
        #     elif i < self.num_verts_st + self.num_edges_st:
        #         pi = i - self.num_verts_st
        #         v0, v1 = self.mesh_st.eid_field[pi, 0], self.mesh_st.eid_field[pi, 1]
        #         self.mesh_st.x_test[i] = 0.5 * (self.mesh_st.x[v0] + self.mesh_st.x[v1])
        #     elif i < self.num_verts_st + self.num_edges_st + self.num_faces_st:
        #         pi = i - (self.num_verts_st + self.num_edges_st)
        #         v0, v1, v2 = self.mesh_st.face_indices_flatten[3 * pi + 0], self.mesh_st.face_indices_flatten[3 * pi + 1], self.mesh_st.face_indices_flatten[3 * pi + 2]
        #         self.mesh_st.x_test[i] = (self.mesh_st.x[v0] + self.mesh_st.x[v1] + self.mesh_st.x[v2]) / 3.0


    @ti.kernel
    def move_particle_x(self, theta: float):
        # sc = ti.math.vec3(0., theta, 0.)
        # for i in range(self.particle_st.num_particles):
        #     # self.particle_st.x_prev[i] = self.particle_st.x[i]
        #     self.particle_st.x_current[i] = sc + self.particle_st.x_prev[i]
        #     # self.particle_st.v[i] += sc + self.particle_st.x_prev[i]

        center = ti.math.vec3(0.0)
        for i in range(self.particle_st.num_particles):
            center += self.particle_st.x_prev[i]

        center /= self.particle_st.num_particles

        # theta = 0.003
        rot = ti.math.mat3([[ti.cos(theta), 0, ti.sin(theta)],
                            [0, 1, 0],
                            [-ti.sin(theta), 0, ti.cos(theta)]])

        for i in self.particle_st.x:
            ri = self.particle_st.x_prev[i] - center
            self.particle_st.x_current[i] = rot @ ri + center

    def forward(self, n_substeps, n_iter):

        dt_sub = self.dt / n_substeps

        self.particle_st.x.copy_from(self.particle_st.x_prev)
        self.sh_st.insert_particles_in_grid(self.particle_st.x)
        self.sh_dy.insert_particles_in_grid(self.mesh_dy.x_sample)

        self.compute_particle_v(n_substeps * dt_sub)

        for _ in range(n_substeps):
            self.compute_y(self.g, dt_sub)
            for _ in range(n_iter):
                # if self.selected_solver_type == 0:
                #     self.solve_constraints_jacobi_x(dt_sub)
                # elif self.selected_solver_type >= 1:
                self.solve_constraints_pd_diag_x(dt_sub)
                # elif self.selected_solver_type == 2:
                #     self.solve_constraints_hess_diag_x(dt_sub)r
                # elif self.selected_solver_type == 3:
                #     self.solve_constraints_pd_diag_x(dt_sub)
                # elif self.selected_solver_type == 4:
                #     self.solve_constraints_newton_pcg_x(dt_sub, self.max_cg_iter, self.threshold)
                # elif self.selected_solver_type == 5:
                #     self.solve_constraints_pd_pcg_x(dt_sub, self.max_cg_iter, self.threshold)
            self.compute_v(dt=dt_sub)
            if self.enable_velocity_update:
                self.solve_constraints_pd_diag_v()

            # if self.selected_solver_type == 0:
            #     self.solve_constraints_jacobi_x(dt_sub)
            # elif self.selected_solver_type >= 1:
            #     self.solve_constraints_pd_diag_x(dt_sub)
            self.update_x(self.damping, dt_sub)
            # self.compute_particle_v(dt_sub)
        self.update_sample_particle_pos()
        # self.particle_st.x_prev.copy_from(self.particle_st.x_current)