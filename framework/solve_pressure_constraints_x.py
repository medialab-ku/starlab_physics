import taichi as ti
@ti.kernel
def solve_pressure_constraints_x(self):

    for vi in range(self.offset_particle, self.max_num_verts_dynamic):
        self.c[vi] = - 1.0
        nabla_C_ii = ti.math.vec3(0.0)
        self.schur[vi] = 1e-4
        xi = self.y[vi]
        center_cell = self.pos_to_index(self.y[vi])
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * 3)):
            grid_index = self.flatten_cell_id(center_cell + offset)
            for p_j in range(self.grid_particles_num[ti.max(0, grid_index - 1)], self.grid_particles_num[grid_index]):
                vj = self.cur2org[p_j]
                xj = self.y[vj]
                xji = xj - xi

                if xji.norm() < self.kernel_radius and self.num_particle_neighbours[vi] < self.cache_size:
                    self.particle_neighbours[vi, self.num_particle_neighbours[vi]] = vj
                    nabla_C_ji = self.spiky_gradient(xji, self.kernel_radius)
                    self.particle_neighbours_gradients[vi, self.num_particle_neighbours[vi]] = nabla_C_ji
                    self.c[vi] += self.poly6_value(xji.norm(), self.kernel_radius)
                    nabla_C_ii -= nabla_C_ji
                    self.schur[vi] += nabla_C_ji.dot(nabla_C_ji)
                    ti.atomic_add(self.num_particle_neighbours[vi], 1)

        self.schur[vi] += nabla_C_ii.dot(nabla_C_ii)

        if self.c[vi] > 0.0:
            lambda_i = self.c[vi] / self.schur[vi]
            for j in range(self.num_particle_neighbours[vi]):
            # for offset in ti.grouped(ti.ndrange(*((-1, 2),) * 3)):
            #     grid_index = self.flatten_grid_index(center_cell + offset)
            #     for p_j in range(self.grid_particles_num[ti.max(0, grid_index - 1)], self.grid_particles_num[grid_index]):
            #         vj = self.cur2org[p_j]
                vj = self.particle_neighbours[vi, j]
                xj = self.y[vj]
                xji = xj - xi

                nabla_C_ji = self.particle_neighbours_gradients[vi, j]
                self.dx[vj] -= lambda_i * nabla_C_ji
                self.nc[vj] += 1

        # self.dx[vi] -= lambda_i * nabla_C_ii
        # self.nc[vi] += 1