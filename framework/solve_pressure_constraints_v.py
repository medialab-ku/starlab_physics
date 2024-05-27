import taichi as ti


@ti.kernel
def solve_pressure_constraints_v(self):
    for vi in range(self.offset_particle, self.max_num_verts_dynamic):
        Cv_i = 0.0
        nabla_Cv_ii = ti.math.vec3(0.0)
        for j in range(self.num_particle_neighbours[vi]):
            # for offset in ti.grouped(ti.ndrange(*((-1, 2),) * 3)):
            #     grid_index = self.flatten_grid_index(center_cell + offset)
            #     for p_j in range(self.grid_particles_num[ti.max(0, grid_index - 1)], self.grid_particles_num[grid_index]):
            #         vj = self.cur2org[p_j]
            vj = self.particle_neighbours[vi, j]

            # if xji.norm() < self.kernel_radius:
            nabla_Cv_ji = self.particle_neighbours_gradients[vi, j]
            Cv_i += nabla_Cv_ji.dot(self.v[vj])
            nabla_Cv_ii -= nabla_Cv_ji

        lambda_i = Cv_i / self.schur[vi]

        if self.c[vi] > 0.0 and Cv_i > 0:
            for j in range(self.num_particle_neighbours[vi]):
                # for offset in ti.grouped(ti.ndrange(*((-1, 2),) * 3)):
                #     grid_index = self.flatten_grid_index(center_cell + offset)
                #     for p_j in range(self.grid_particles_num[ti.max(0, grid_index - 1)], self.grid_particles_num[grid_index]):
                #         vj = self.cur2org[p_j]
                vj = self.particle_neighbours[vi, j]
                # if xji.norm() < self.kernel_radius:
                nabla_Cv_ji = self.particle_neighbours_gradients[vi, j]
                self.dv[vj] -= lambda_i * nabla_Cv_ji
                # self.nc[vj] += 1