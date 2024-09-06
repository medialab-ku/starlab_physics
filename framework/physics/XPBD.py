import taichi as ti

@ti.data_oriented
class Solver:
    def __init__(
            self,
            mesh_dy,
            mesh_st,
            dHat,
            stiffness_stretch,
            stiffness_bending,
            g,
            dt):

        self.mesh_dy = mesh_dy
        self.mesh_st = mesh_st
        self.dHat = dHat
        self.stiffness_stretch = stiffness_stretch
        self.stiffness_bending = stiffness_bending
        self.g = g
        self.dt = dt
        self.damping = 0.001
        self.mu = 0.8

        self.selected_solver_type = 0

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

    ####################################################################################################################

    @ti.kernel
    def set_fixed_vertices(self, fixed_vertices: ti.template()):
        for i in range(self.num_verts_dy):
            if fixed_vertices[i] >= 1:
                self.mesh_dy.fixed[i] = 0.0
            else:
                self.mesh_dy.fixed[i] = 1.0

        for i in range(self.mesh_dy.euler_path_len):
            vid = self.mesh_dy.euler_path_field[i]
            if fixed_vertices[vid] >= 1:
                self.mesh_dy.fixed_euler[i] = 0.0
            else:
                self.mesh_dy.fixed_euler[i] = 1.0


    def reset(self):
        self.mesh_dy.reset()
        # self.mesh_dy.particles.reset()
        if self.mesh_st is None:
            self.mesh_st.reset()

    def init_variables(self):
        # initialize dx and number of constraints (used in the Jacobi solver)
        self.mesh_dy.dx.fill(0.0)
        self.mesh_dy.nc.fill(0.0)
        self.mesh_dy.dx_euler.fill(0.0)

    @ti.kernel
    def compute_y_tilde(self, g: ti.math.vec3, dt: ti.f32):
        # compute apporximate x_(t+1) (== y) by explicit way before projecting constraints to x_(t+1)...
        for i in range(self.num_verts_dy):
            self.mesh_dy.y[i] = (self.mesh_dy.x[i] + self.mesh_dy.fixed[i] * (self.mesh_dy.v[i] * dt + self.g * dt * dt))
            self.mesh_dy.y_tilde[i] = self.mesh_dy.y[i]

        # for Euler path...
        for i in range(self.mesh_dy.euler_path_len):
            self.mesh_dy.y_euler[i] = (self.mesh_dy.x_euler[i] +
                                       self.mesh_dy.fixed_euler[i] * (self.mesh_dy.v_euler[i] * dt + self.g * dt * dt))
            self.mesh_dy.y_tilde_euler[i] = self.mesh_dy.y_euler[i]

    @ti.kernel
    def update_y(self):
        # update x_(t+1) by adding dx (used in the Euler path solver)
        for i in range(self.num_verts_dy):
            self.mesh_dy.y[i] = (self.mesh_dy.y_tilde[i] +
                                 self.mesh_dy.fixed[i] * (self.mesh_dy.dx[i] / self.mesh_dy.nc[i]))

        for i in range(self.mesh_dy.euler_path_len):
            vid = self.mesh_dy.euler_path_field[i]
            self.mesh_dy.y_euler[i] = (self.mesh_dy.y_tilde_euler[i] +
                                       self.mesh_dy.fixed_euler[i] * (self.mesh_dy.dx_euler[i] / self.mesh_dy.duplicates_field[vid]))

    @ti.kernel
    def compute_v(self, damping: ti.f32, dt: ti.f32):
        # compute v after all constraints are projected to x_(t+1)
        for i in range(self.num_verts_dy):
            self.mesh_dy.v[i] = self.mesh_dy.fixed[i] * (1.0 - damping) * (self.mesh_dy.y[i] - self.mesh_dy.x[i]) / dt

        # for Euler path...
        for i in range(self.mesh_dy.euler_path_len):
            self.mesh_dy.v_euler[i] = self.mesh_dy.fixed_euler[i] * (1.0 - damping) * (self.mesh_dy.y_euler[i] - self.mesh_dy.x_euler[i]) / dt


    @ti.kernel
    def update_x(self, dt: ti.f32):
        # eventually, update actual position x_(t+1) after velocities are computed...
        for i in range(self.num_verts_dy):
            self.mesh_dy.x[i] += dt * self.mesh_dy.v[i]

        # for Euler path...
        for i in range(self.mesh_dy.euler_path_len):
            self.mesh_dy.x_euler[i] += dt * self.mesh_dy.v_euler[i]
        for i in range(self.mesh_dy.euler_edge_len):
            self.mesh_dy.colored_edge_pos_euler[i] = 0.5 * (self.mesh_dy.x_euler[i] + self.mesh_dy.x_euler[i+1])

    @ti.kernel
    def aggregate_duplicates(self):
        for i in range(self.mesh_dy.euler_path_len):
            vid = self.mesh_dy.euler_path_field[i]
            self.mesh_dy.y[vid] += self.mesh_dy.y_euler[i]

        for i in range(self.num_verts_dy):
            self.mesh_dy.y[i] /= self.mesh_dy.duplicates_field[i]

        for i in range(self.mesh_dy.euler_path_len):
            vid = self.mesh_dy.euler_path_field[i]
            self.mesh_dy.y_euler[i] = self.mesh_dy.y[vid]

    ####################################################################################################################
    # Several constraint solvers to compute physics...
    def solve_constraints_jacobi_x(self, dt):
        self.init_variables()

        compliance_stretch = self.stiffness_stretch * dt * dt
        compliance_bending = self.stiffness_bending * dt * dt

        self.solve_spring_constraints_jacobi_x(compliance_stretch, compliance_bending)

    def solve_constraints_gauss_seidel_x(self, dt):
        compliance_stretch = self.stiffness_stretch * dt * dt
        compliance_bending = self.stiffness_bending * dt * dt

        self.solve_spring_constraints_gauss_seidel_x(compliance_stretch, compliance_bending)

    def solve_constraints_parallel_gauss_seidel_x(self, dt):
        compliance_stretch = self.stiffness_stretch * dt * dt
        compliance_bending = self.stiffness_bending * dt * dt

        model_num = self.mesh_dy.num_model
        for i in range(model_num):
            prefix_sum_offset = self.mesh_dy.color_max
            for j in range(self.mesh_dy.color_max - 1):
                current_offset = self.mesh_dy.original_edge_color_prefix_sum_np[prefix_sum_offset * i + j]
                next_offset = self.mesh_dy.original_edge_color_prefix_sum_np[prefix_sum_offset * i + j + 1]
                if current_offset >= next_offset:
                    break

                self.solve_spring_constraints_original_parallel_gauss_seidel_x(compliance_stretch,
                                                                               compliance_bending,
                                                                               current_offset,
                                                                               next_offset)

    def solve_constraints_euler_path_gauss_seidel_x(self, dt):
        self.init_variables()

        compliance_stretch = self.stiffness_stretch * dt * dt
        compliance_bending = self.stiffness_bending * dt * dt

        model_num = self.mesh_dy.num_model

        for i in range(model_num):
            current_offset, next_offset = self.mesh_dy.euler_path_offsets_field[i], self.mesh_dy.euler_path_offsets_field[i+1]

            self.solve_spring_constraints_euler_path_gauss_seidel_x(compliance_stretch,
                                                                    compliance_bending,
                                                                    current_offset,
                                                                    next_offset,
                                                                    0)

            self.solve_spring_constraints_euler_path_gauss_seidel_x(compliance_stretch,
                                                                    compliance_bending,
                                                                    current_offset,
                                                                    next_offset,
                                                                    1)

        self.update_y()

    ####################################################################################################################
    # Taichi kernel function which are called in solvers...

    @ti.kernel
    def solve_spring_constraints_jacobi_x(self, compliance_stretch: ti.f32, compliance_bending: ti.f32):
        # project x_(t+1) by solving constraints in parallel way
        ti.block_local(self.mesh_dy.l0, self.mesh_dy.eid_field, self.mesh_dy.fixed, self.mesh_dy.m_inv,
                       self.mesh_dy.dx, self.mesh_dy.nc)
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
            if self.mesh_dy.nc[i] > 0:
                self.mesh_dy.y[i] += self.mesh_dy.fixed[i] * (self.mesh_dy.dx[i] / self.mesh_dy.nc[i])

    @ti.kernel
    def solve_spring_constraints_gauss_seidel_x(self, compliance_stretch: ti.f32, compliance_bending: ti.f32):
        ti.block_local(self.mesh_dy.l0, self.mesh_dy.eid_field, self.mesh_dy.fixed, self.mesh_dy.m_inv)
        ti.loop_config(serialize=True)
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
    def solve_spring_constraints_original_parallel_gauss_seidel_x(self,
                                                                  compliance_stretch: ti.f32,
                                                                  compliance_bending: ti.f32,
                                                                  current_offset: ti.i32,
                                                                  next_offset: ti.i32):
        ti.block_local(self.mesh_dy.l0, self.mesh_dy.eid_field, self.mesh_dy.fixed, self.mesh_dy.m_inv)
        # ti.loop_config(serialize=True)
        for i in range(current_offset, next_offset):
            v0, v1 = (self.mesh_dy.original_edge_color_field[i, 0],
                      self.mesh_dy.original_edge_color_field[i, 1])
            l0 = self.mesh_dy.l0_original_graph[i]
            x10 = self.mesh_dy.y[v0] - self.mesh_dy.y[v1]
            lij = x10.norm()

            C = (lij - l0)
            nabla_C = x10.normalized()
            schur = self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] + self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1]
            ld = compliance_stretch * C / (compliance_stretch * schur + 1.0)

            self.mesh_dy.y[v0] -= self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] * ld * nabla_C
            self.mesh_dy.y[v1] += self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1] * ld * nabla_C

    @ti.kernel
    def solve_spring_constraints_euler_path_gauss_seidel_x(self,
                                                           compliance_stretch: ti.f32,
                                                           compliance_bending: ti.f32,
                                                           current_offset: ti.i32,
                                                           next_offset: ti.i32,
                                                           edge_offset: ti.i32):
        # ti.loop_config(serialize=True)
        for i in range(0, (next_offset - current_offset - 1) // 2):
            idx = current_offset + (i * 2 + edge_offset) # current_offset + (0, 2, 4, ... or 1, 3, 5, ...)
            v0, v1 = self.mesh_dy.euler_path_field[idx], self.mesh_dy.euler_path_field[idx + 1]
            l0 = self.mesh_dy.l0_euler[idx]
            x10 = self.mesh_dy.y[v0] - self.mesh_dy.y[v1]
            lij = x10.norm()

            C = lij - l0
            nabla_C = x10.normalized()
            schur = self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] + self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1]
            ld = compliance_stretch * C / (compliance_stretch * schur + 1.0)

            self.mesh_dy.dx[v0] -= self.mesh_dy.fixed[v0] * self.mesh_dy.m_inv[v0] * ld * nabla_C
            self.mesh_dy.dx[v1] += self.mesh_dy.fixed[v1] * self.mesh_dy.m_inv[v1] * ld * nabla_C

            self.mesh_dy.nc[v0] += 1.0
            self.mesh_dy.nc[v1] += 1.0

        # ti.loop_config(serialize=True)
        for i in range(0, (next_offset - current_offset - 1) // 2):
            idx = current_offset + (i * 2 + edge_offset) # current_offset + (0, 2, 4, ... or 1, 3, 5, ...)
            l0 = self.mesh_dy.l0_euler[idx]
            x10 = self.mesh_dy.y_euler[idx] - self.mesh_dy.y_euler[idx + 1]
            lij = x10.norm()

            C = lij - l0
            nabla_C = x10.normalized()
            schur = (self.mesh_dy.fixed_euler[idx] * self.mesh_dy.m_inv_euler[idx] +
                     self.mesh_dy.fixed_euler[idx + 1] * self.mesh_dy.m_inv_euler[idx + 1])
            ld = compliance_stretch * C / (compliance_stretch * schur + 1.0)

            self.mesh_dy.dx_euler[idx] -= self.mesh_dy.fixed_euler[idx] * self.mesh_dy.m_inv_euler[idx] * ld * nabla_C
            self.mesh_dy.dx_euler[idx + 1] += self.mesh_dy.fixed_euler[idx + 1] * self.mesh_dy.m_inv_euler[idx + 1] * ld * nabla_C

    ####################################################################################################################

    def forward(self, n_substeps):
        dt_sub = self.dt / n_substeps

        for _ in range(n_substeps):
            self.compute_y_tilde(self.g, dt_sub)
            if self.selected_solver_type == 0:
                self.solve_constraints_jacobi_x(dt_sub)
            elif self.selected_solver_type == 1:
                self.solve_constraints_gauss_seidel_x(dt_sub)
            elif self.selected_solver_type == 2:
                self.solve_constraints_parallel_gauss_seidel_x(dt_sub)
            elif self.selected_solver_type == 3:
                self.solve_constraints_euler_path_gauss_seidel_x(dt_sub)

            self.compute_v(damping=self.damping, dt=dt_sub)
            self.update_x(dt_sub)