import taichi as ti
import numpy as np
import distance as di

@ti.data_oriented
class Solver:
    def __init__(self,
                 meshes,
                 particles,
                 grid_size,
                 particle_radius,
                 g,
                 dt):
        self.meshes = meshes
        self.particles = particles
        self.g = g
        self.dt = dt
        self.dHat = 4 * particle_radius * particle_radius
        self.grid_size = grid_size
        self.particle_radius = particle_radius

        self.grid_vertices = ti.Vector.field(n=3, dtype=ti.f32, shape=8)
        self.grid_edge_indices = ti.field(dtype=ti.u32, shape=12 * 2)

        self.init_grid()

        self.cell_size = 4 * particle_radius
        self.grid_origin = -self.grid_size
        self.grid_num = np.ceil(2 * self.grid_size / self.cell_size).astype(int)

        self.enable_velocity_update = False
        print(self.grid_num)


        self.max_num_verts = 0
        self.max_num_edges = 0
        self.max_num_faces = 0

        self.offset_verts = ti.field(int, shape=len(self.meshes) + 1)
        self.offset_edges = ti.field(int, shape=len(self.meshes) + 1)
        self.offset_faces = ti.field(int, shape=len(self.meshes) + 1)

        for mid in range(len(self.meshes)):
            self.offset_verts[mid] = self.max_num_verts
            self.offset_edges[mid] = self.max_num_edges
            self.offset_faces[mid] = self.max_num_faces
            self.max_num_verts += len(self.meshes[mid].verts)
            self.max_num_edges += len(self.meshes[mid].edges)
            self.max_num_faces += len(self.meshes[mid].faces)


        self.y = ti.Vector.field(n=3, dtype=ti.f32, shape=self.max_num_verts)
        self.x = ti.Vector.field(n=3, dtype=ti.f32, shape=self.max_num_verts)
        self.dx = ti.Vector.field(n=3, dtype=ti.f32, shape=self.max_num_verts)
        self.dv = ti.Vector.field(n=3, dtype=ti.f32, shape=self.max_num_verts)
        self.v = ti.Vector.field(n=3, dtype=ti.f32, shape=self.max_num_verts)
        self.nc = ti.field(dtype=ti.int32, shape=self.max_num_verts)
        self.m_inv = ti.field(dtype=ti.f32, shape=self.max_num_verts)

        self.l0 = ti.field(dtype=ti.f32, shape=self.max_num_edges)

        self.face_indices = ti.field(dtype=ti.i32, shape=3 * self.max_num_faces)
        self.edge_indices = ti.field(dtype=ti.i32, shape=2 * self.max_num_edges)
        self.offset_verts[len(self.meshes)] = self.max_num_verts
        self.offset_faces[len(self.meshes)] = self.max_num_faces

        self.init_mesh_aggregation()

        self.grid_particles_num = ti.field(int, shape=int(self.grid_num[0] * self.grid_num[1] * self.grid_num[2]))
        self.grid_particles_num_temp = ti.field(int, shape=int(self.grid_num[0] * self.grid_num[1] * self.grid_num[2]))
        self.prefix_sum_executor = ti.algorithms.PrefixSumExecutor(self.grid_particles_num.shape[0])

        self.grid_ids = ti.field(int, shape=self.max_num_verts)
        self.grid_ids_buffer = ti.field(int, shape=self.max_num_verts)
        self.grid_ids_new = ti.field(int, shape=self.max_num_verts)
        self.cur2org = ti.field(int, shape=self.max_num_verts)


    def copy_to_meshes(self):
        for mid in range(len(self.meshes)):
            self.copy_to_meshes_device(self.offset_verts[mid], self.meshes[mid])

    @ti.kernel
    def copy_to_meshes_device(self, offset: ti.int32, mesh: ti.template()):
        for v in mesh.verts:
            v.x = self.x[offset + v.id]
            v.v = self.v[offset + v.id]


    @ti.kernel
    def init_quantities_device(self, offset: ti.int32, mesh: ti.template()):
        for v in mesh.verts:
            self.x[offset + v.id] = v.x
            self.v[offset + v.id] = v.v
            self.m_inv[offset + v.id] = v.m_inv

    @ti.kernel
    def init_edge_indices_device(self, offset_verts: ti.int32, offset_edges: ti.int32, mesh: ti.template()):
        for e in mesh.edges:
            self.edge_indices[2 * (offset_edges + e.id) + 0] = e.verts[0].id + offset_verts
            self.edge_indices[2 * (offset_edges + e.id) + 1] = e.verts[1].id + offset_verts

    @ti.kernel
    def init_rest_length(self):
        for ei in range(self.max_num_edges):
            v0, v1 = self.edge_indices[2 * ei + 0], self.edge_indices[2 * ei + 1]
            x0, x1 = self.x[v0], self.x[v1]
            x10 = x0 - x1
            self.l0[ei] = x10.norm()


    def init_mesh_aggregation(self):
        for mid in range(len(self.meshes)):
            self.init_quantities_device(self.offset_verts[mid], self.meshes[mid])
            self.init_edge_indices_device(self.offset_verts[mid], self.offset_edges[mid], self.meshes[mid])
            self.init_face_indices_device(self.offset_verts[mid], self.offset_faces[mid], self.meshes[mid])

        self.init_rest_length()

    @ti.kernel
    def init_face_indices_device(self, offset_verts: ti.int32, offset_faces: ti.int32, mesh: ti.template()):
        for f in mesh.faces:
            self.face_indices[3 * (offset_faces + f.id) + 0] = f.verts[0].id + offset_verts
            self.face_indices[3 * (offset_faces + f.id) + 1] = f.verts[1].id + offset_verts
            self.face_indices[3 * (offset_faces + f.id) + 2] = f.verts[2].id + offset_verts


    def init_grid(self):

        self.grid_vertices[0] = ti.math.vec3(self.grid_size[0], self.grid_size[1], self.grid_size[2])
        self.grid_vertices[1] = ti.math.vec3(-self.grid_size[0], self.grid_size[1], self.grid_size[2])
        self.grid_vertices[2] = ti.math.vec3(-self.grid_size[0], self.grid_size[1], -self.grid_size[2])
        self.grid_vertices[3] = ti.math.vec3(self.grid_size[0], self.grid_size[1], -self.grid_size[2])

        self.grid_vertices[4] = ti.math.vec3(self.grid_size[0], -self.grid_size[1], self.grid_size[2])
        self.grid_vertices[5] = ti.math.vec3(-self.grid_size[0], -self.grid_size[1], self.grid_size[2])
        self.grid_vertices[6] = ti.math.vec3(-self.grid_size[0], -self.grid_size[1], -self.grid_size[2])
        self.grid_vertices[7] = ti.math.vec3(self.grid_size[0], -self.grid_size[1], -self.grid_size[2])

        self.grid_edge_indices[0] = 0
        self.grid_edge_indices[1] = 1

        self.grid_edge_indices[2] = 1
        self.grid_edge_indices[3] = 2

        self.grid_edge_indices[4] = 2
        self.grid_edge_indices[5] = 3

        self.grid_edge_indices[6] = 3
        self.grid_edge_indices[7] = 0

        self.grid_edge_indices[8] = 4
        self.grid_edge_indices[9] = 5

        self.grid_edge_indices[10] = 5
        self.grid_edge_indices[11] = 6

        self.grid_edge_indices[12] = 6
        self.grid_edge_indices[13] = 7

        self.grid_edge_indices[14] = 7
        self.grid_edge_indices[15] = 4

        self.grid_edge_indices[16] = 0
        self.grid_edge_indices[17] = 4

        self.grid_edge_indices[18] = 1
        self.grid_edge_indices[19] = 5

        self.grid_edge_indices[20] = 2
        self.grid_edge_indices[21] = 6

        self.grid_edge_indices[22] = 3
        self.grid_edge_indices[23] = 7

        # print(self.grid_vertices)
        # print(self.grid_edge_indices)



    def reset(self):
        for mid in range(len(self.meshes)):
            self.meshes[mid].reset()

        self.init_mesh_aggregation()

    @ti.kernel
    def compute_y(self):

        # self.m_inv[0] = 0.0
        for i in range(self.max_num_verts):
            if self.m_inv[i] > 0.0:
                self.y[i] = self.x[i] + self.dt * self.v[i] + self.g * self.dt * self.dt
            else:
                self.y[i] = self.x[i]


    @ti.kernel
    def compute_y_particle(self, particle: ti.template()):
        for i in range(particle.num_particles):
            particle.y[i] = particle.x[i] + particle.v[i] * self.dt + self.g * self.dt * self.dt

    @ti.kernel
    def counting_sort(self):
        # FIXME: make it the actual particle num
        for i in range(self.max_num_verts):
            I = self.max_num_verts - 1 - i
            base_offset = 0
            if self.grid_ids[I] - 1 >= 0:
                base_offset = self.grid_particles_num[self.grid_ids[I] - 1]
            self.grid_ids_new[I] = ti.atomic_sub(self.grid_particles_num_temp[self.grid_ids[I]], 1) - 1 + base_offset

        for i in self.grid_ids:
            new_index = self.grid_ids_new[i]
            self.cur2org[new_index] = i

    @ti.func
    def flatten_grid_index(self, grid_index):
        return grid_index[0] * self.grid_num[1] * self.grid_num[2] + grid_index[1] * self.grid_num[2] + grid_index[2]

    @ti.func
    def pos_to_index(self, pos):
        return ((pos - self.grid_origin) / self.cell_size).cast(int)

    @ti.func
    def get_flatten_grid_index(self, pos: ti.math.vec3):
        return self.flatten_grid_index(self.pos_to_index(pos))




    def broad_phase(self):

        # self.grid_particles_num.fill(0)
        self.update_grid_id()
        self.prefix_sum_executor.run(self.grid_particles_num)
        self.counting_sort()

    @ti.kernel
    def update_grid_id(self):

        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num[I] = 0

        #TODO: update the following two for-loops into a single one
        for vi in range(self.max_num_verts):
            grid_index = self.get_flatten_grid_index(self.y[vi])
            self.grid_ids[vi] = grid_index
            ti.atomic_add(self.grid_particles_num[grid_index], 1)


    @ti.func
    def is_in_face(self, vid, fid):

        v1 = self.face_indices[3 * fid + 0]
        v2 = self.face_indices[3 * fid + 1]
        v3 = self.face_indices[3 * fid + 2]

        return (v1 == vid) or (v2 == vid) or (v3 == vid)

    @ti.func
    def solve_collision_vv(self, v0, v1):

        if v0 != v1:
            x0 = self.y[v0]
            x1 = self.y[v1]
            #
            # x01 = x1 - x0
            #
            # dist = ti.length(x01)
            # nor = x01 / dist
            #
            d = di.d_PP(x0, x1)

            g0, g1 = di.g_PP(x0, x1)
            if d < self.dHat:
                # if self.vt_dynamic_active_set_num[vi] < self.num_max_neighbors:
                #     self.vt_dynamic_active_set[vi, self.vt_dynamic_active_set_num[vi]] = fi
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v1] * g1.dot(g1) + 1e-4
                ld = (self.dHat - d) / schur
                # ti.atomic_add(self.vt_dynamic_active_set_num[vi], 1)
                self.dx[v0] += self.m_inv[v0] * ld * g0
                self.dx[v1] += self.m_inv[v1] * ld * g1

                self.nc[v0] += 1
                self.nc[v1] += 1

    @ti.func
    def solve_collision_vt_x(self, vid, fid):

        v0 = vid
        v1 = self.face_indices[3 * fid + 0]
        v2 = self.face_indices[3 * fid + 1]
        v3 = self.face_indices[3 * fid + 2]

        x0 = self.y[v0]
        x1 = self.y[v1]
        x2 = self.y[v2]
        x3 = self.y[v3]

        dtype = di.d_type_PT(x0, x1, x2, x3)

        if dtype == 0:
            d = di.d_PP(x0, x1)
            g0, g1 = di.g_PP(x0, x1)
            if d < self.dHat:
                # if self.vt_dynamic_active_set_num[vi] < self.num_max_neighbors:
                #     self.vt_dynamic_active_set[vi, self.vt_dynamic_active_set_num[vi]] = fi
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v1] * g1.dot(g1) + 1e-4
                ld = (self.dHat - d) / schur
                # ti.atomic_add(self.vt_dynamic_active_set_num[vi], 1)
                self.dx[v0] += self.m_inv[v0] * ld * g0
                self.dx[v1] += self.m_inv[v1] * ld * g1

                self.nc[v0] += 1
                self.nc[v1] += 1

        elif dtype == 1:
            d = di.d_PP(x0, x2)
            g0, g2 = di.g_PP(x0, x2)
            if d < self.dHat:
                # if self.vt_dynamic_active_set_num[vi] < self.num_max_neighbors:
                #     self.vt_dynamic_active_set[vi, self.vt_dynamic_active_set_num[vi]] = fi
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v2] * g2.dot(g2) + 1e-4
                ld = (self.dHat - d) / schur
                # ti.atomic_add(self.vt_dynamic_active_set_num[vi], 1)
                self.dx[v0] += self.m_inv[v0] * ld * g0
                self.dx[v2] += self.m_inv[v2] * ld * g2

                self.nc[v0] += 1
                self.nc[v2] += 1

        elif dtype == 2:
            d = di.d_PP(x0, x3)
            g0, g3 = di.g_PP(x0, x3)
            if d < self.dHat:
                # if self.vt_dynamic_active_set_num[vi] < self.num_max_neighbors:
                #     self.vt_dynamic_active_set[vi, self.vt_dynamic_active_set_num[vi]] = fi
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v3] * g3.dot(g3) + 1e-4
                ld = (self.dHat - d) / schur
                # ti.atomic_add(self.vt_dynamic_active_set_num[vi], 1)
                self.dx[v0] += self.m_inv[v0] * ld * g0
                self.dx[v3] += self.m_inv[v3] * ld * g3

                self.nc[v0] += 1
                self.nc[v3] += 1

        elif dtype == 3:
            d = di.d_PE(x0, x1, x2)
            g0, g1, g2 = di.g_PE(x0, x1, x2)
            if d < self.dHat:
                # if self.vt_dynamic_active_set_num[vi] < self.num_max_neighbors:
                #     self.vt_dynamic_active_set[vi, self.vt_dynamic_active_set_num[vi]] = fi
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v1] * g1.dot(g1) + self.m_inv[v2] * g2.dot(g2) + 1e-4
                ld = (self.dHat - d) / schur
                # ti.atomic_add(self.vt_dynamic_active_set_num[vi], 1)
                self.dx[v0] += self.m_inv[v0] * ld * g0
                self.dx[v1] += self.m_inv[v1] * ld * g1
                self.dx[v2] += self.m_inv[v2] * ld * g2

                self.nc[v0] += 1
                self.nc[v1] += 1
                self.nc[v2] += 1



        elif dtype == 4:
            d = di.d_PE(x0, x2, x3)
            g0, g2, g3 = di.g_PE(x0, x2, x3)
            if d < self.dHat:
                # if self.vt_dynamic_active_set_num[vi] < self.num_max_neighbors:
                #     self.vt_dynamic_active_set[vi, self.vt_dynamic_active_set_num[vi]] = fi
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v2] * g2.dot(g2) + self.m_inv[v3] * g3.dot(g3) + 1e-4
                ld = (self.dHat - d) / schur
                # ti.atomic_add(self.vt_dynamic_active_set_num[vi], 1)
                self.dx[v0] += self.m_inv[v0] * ld * g0
                self.dx[v2] += self.m_inv[v2] * ld * g2
                self.dx[v3] += self.m_inv[v3] * ld * g3
                self.nc[v0] += 1
                self.nc[v2] += 1
                self.nc[v3] += 1


        elif dtype == 5:
            d = di.d_PE(x0, x1, x3)
            g0, g1, g3 = di.g_PE(x0, x1, x3)
            if d < self.dHat:
                # if self.vt_dynamic_active_set_num[vi] < self.num_max_neighbors:
                #     self.vt_dynamic_active_set[vi, self.vt_dynamic_active_set_num[vi]] = fi
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v1] * g1.dot(g1) + self.m_inv[v3] * g3.dot(g3) + 1e-4
                ld = (self.dHat - d) / schur
                # ti.atomic_add(self.vt_dynamic_active_set_num[vi], 1)
                self.dx[v0] += self.m_inv[v0] * ld * g0
                self.dx[v1] += self.m_inv[v1] * ld * g1
                self.dx[v3] += self.m_inv[v3] * ld * g3
                self.nc[v0] += 1
                self.nc[v1] += 1
                self.nc[v3] += 1

        elif dtype == 6:
            d = di.d_PT(x0, x1, x2, x3)
            g0, g1, g2, g3 = di.g_PT(x0, x1, x2, x3)

            if d < self.dHat:
                # if self.vt_dynamic_active_set_num[vi] < self.num_max_neighbors:
                #     self.vt_dynamic_active_set[vi, self.vt_dynamic_active_set_num[vi]] = fi
                    schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v1] * g1.dot(g1) + self.m_inv[v2] * g2.dot(g2) + self.m_inv[v3] * g3.dot(g3) + 1e-4
                    ld = (self.dHat - d) / schur
                    # ti.atomic_add(self.vt_dynamic_active_set_num[vi], 1)
                    self.dx[v0] += self.m_inv[v0] * ld * g0
                    self.dx[v1] += self.m_inv[v1] * ld * g1
                    self.dx[v2] += self.m_inv[v2] * ld * g2
                    self.dx[v3] += self.m_inv[v3] * ld * g3
                    self.nc[v0] += 1
                    self.nc[v1] += 1
                    self.nc[v2] += 1
                    self.nc[v3] += 1

    @ti.func
    def solve_collision_vt_v(self, vid, fid):

        v0 = vid
        v1 = self.face_indices[3 * fid + 0]
        v2 = self.face_indices[3 * fid + 1]
        v3 = self.face_indices[3 * fid + 2]

        x0 = self.y[v0]
        x1 = self.y[v1]
        x2 = self.y[v2]
        x3 = self.y[v3]

        dtype = di.d_type_PT(x0, x1, x2, x3)

        if dtype == 0:
            d = di.d_PP(x0, x1)
            g0, g1 = di.g_PP(x0, x1)
            dvn = g0.dot(self.v[v0]) + g1.dot(self.v[v1])
            if d < self.dHat and dvn < 0.0:
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v1] * g1.dot(g1) + 1e-6
                ld = dvn / schur

                self.dv[v0] -= self.m_inv[v0] * ld * g0
                self.dv[v1] -= self.m_inv[v1] * ld * g1

                self.nc[v0] += 1
                self.nc[v1] += 1

        elif dtype == 1:
            d = di.d_PP(x0, x2)
            g0, g2 = di.g_PP(x0, x2)
            dvn = g0.dot(self.v[v0]) + g2.dot(self.v[v2])
            if d < self.dHat and dvn < 0.0:
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v2] * g2.dot(g2) + 1e-6
                ld = dvn / schur
                self.dv[v0] -= self.m_inv[v0] * ld * g0
                self.dv[v2] -= self.m_inv[v1] * ld * g2

                self.nc[v0] += 1
                self.nc[v2] += 1


        elif dtype == 2:
            d = di.d_PP(x0, x3)
            g0, g3 = di.g_PP(x0, x3)
            dvn = g0.dot(self.v[v0]) + g3.dot(self.v[v3])
            if d < self.dHat and dvn < 0.0:
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v3] * g3.dot(g3) + 1e-6
                ld = dvn / schur
                self.dv[v0] -= self.m_inv[v0] * ld * g0
                self.dv[v3] -= self.m_inv[v1] * ld * g3

                self.nc[v0] += 1
                self.nc[v3] += 1


        elif dtype == 3:
            d = di.d_PE(x0, x1, x2)
            g0, g1, g2 = di.g_PE(x0, x1, x2)
            dvn = g0.dot(self.v[v0]) + g1.dot(self.v[v1]) + g2.dot(self.v[v2])
            if d < self.dHat and dvn < 0.0:
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v1] * g1.dot(g1) + self.m_inv[v2] * g2.dot(g2) + 1e-6
                ld = dvn / schur
                self.dv[v0] -= self.m_inv[v0] * ld * g0
                self.dv[v1] -= self.m_inv[v0] * ld * g1
                self.dv[v2] -= self.m_inv[v1] * ld * g2

                self.nc[v0] += 1
                self.nc[v1] += 1
                self.nc[v2] += 1

        elif dtype == 4:
            d = di.d_PE(x0, x2, x3)
            g0, g2, g3 = di.g_PE(x0, x2, x3)
            dvn = g0.dot(self.v[v0]) + g2.dot(self.v[v2]) + g3.dot(self.v[v3])
            if d < self.dHat and dvn < 0.0:
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v2] * g2.dot(g2) + self.m_inv[v3] * g3.dot(g3) + 1e-6
                ld = dvn / schur
                self.dv[v0] -= self.m_inv[v0] * ld * g0
                self.dv[v2] -= self.m_inv[v2] * ld * g2
                self.dv[v3] -= self.m_inv[v3] * ld * g3

                self.nc[v0] += 1
                self.nc[v2] += 1
                self.nc[v3] += 1


        elif dtype == 5:
            d = di.d_PE(x0, x1, x3)
            g0, g1, g3 = di.g_PE(x0, x1, x3)
            dvn = g0.dot(self.v[v0]) + g1.dot(self.v[v1]) + g3.dot(self.v[v3])
            if d < self.dHat and dvn < 0.0:
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v1] * g1.dot(g1) + self.m_inv[v3] * g3.dot(g3) + 1e-6
                ld = dvn / schur
                self.dv[v0] -= self.m_inv[v0] * ld * g0
                self.dv[v1] -= self.m_inv[v1] * ld * g1
                self.dv[v3] -= self.m_inv[v3] * ld * g3

                self.nc[v0] += 1
                self.nc[v1] += 1
                self.nc[v3] += 1

        elif dtype == 6:
            d = di.d_PT(x0, x1, x2, x3)
            g0, g1, g2, g3 = di.g_PT(x0, x1, x2, x3)
            dvn = g0.dot(self.v[v0]) + g1.dot(self.v[v1]) + g2.dot(self.v[v2]) + g3.dot(self.v[v3])
            if d <= self.dHat and dvn < 0.0:
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v1] * g1.dot(g1) + self.m_inv[v2] * g2.dot(g2) + self.m_inv[v3] * g3.dot(g3) + 1e-6
                ld = dvn / schur
                self.dv[v0] -= self.m_inv[v0] * ld * g0
                self.dv[v1] -= self.m_inv[v1] * ld * g1
                self.dv[v2] -= self.m_inv[v2] * ld * g2
                self.dv[v3] -= self.m_inv[v3] * ld * g3

                self.nc[v0] += 1
                self.nc[v1] += 1
                self.nc[v2] += 1
                self.nc[v3] += 1


    @ti.kernel
    def solve_spring_constraints_x(self):

        for ei in range(self.max_num_edges):
            v0, v1 = self.edge_indices[2 * ei + 0], self.edge_indices[2 * ei + 1]
            x0, x1 = self.y[v0], self.y[v1]
            l0 = self.l0[ei]

            x10 = x0 - x1
            center = 0.5 * (x0 + x1)
            lij = x10.norm()
            normal = x10 / lij
            p0 = center + 0.5 * l0 * normal
            p1 = center - 0.5 * l0 * normal

            self.dx[v0] += (p0 - x0)
            self.dx[v1] += (p1 - x1)
            self.nc[v0] += 1
            self.nc[v1] += 1



    @ti.kernel
    def solve_collision_constraints_x(self):

        for vi in range(self.max_num_verts):
            # center_cell = self.pos_to_index(self.y[vi])
            # for offset in ti.grouped(ti.ndrange(*((-1, 2),) * 3)):
            #     grid_index = self.flatten_grid_index(center_cell)
            #     for p_j in range(self.grid_particles_num[ti.max(0, grid_index - 1)], self.grid_particles_num[grid_index]):
            #         vj = self.cur2org[p_j]r
            #     # for vj in range(self.max_num_verts):
            #         self.solve_collision_vv(vi, vj)
                    # if self.is_in_face(vi, fi) != True:
                    #     self.solve_collision_vt(vi, fi)
            for fi in range(self.max_num_faces):
                if self.is_in_face(vi, fi) != True:
                    self.solve_collision_vt_x(vi, fi)

    @ti.kernel
    def solve_collision_constraints_v(self):
        for vi in range(self.max_num_verts):
            # center_cell = self.pos_to_index(self.y[vi])
            # for offset in ti.grouped(ti.ndrange(*((-1, 2),) * 3)):
            #     grid_index = self.flatten_grid_index(center_cell)
            #     for p_j in range(self.grid_particles_num[ti.max(0, grid_index - 1)], self.grid_particles_num[grid_index]):
            #         vj = self.cur2org[p_j]r
            #     # for vj in range(self.max_num_verts):
            #         self.solve_collision_vv(vi, vj)
            # if self.is_in_face(vi, fi) != True:
            #     self.solve_collision_vt(vi, fi)
            for fi in range(self.max_num_faces):
                if self.is_in_face(vi, fi) != True:
                    self.solve_collision_vt_v(vi, fi)


    @ti.kernel
    def update_x(self):

        for i in range(self.max_num_verts):
            if self.m_inv[i] > 0.0:
                self.x[i] += self.v[i] * self.dt

            else:
                self.v[i] = ti.math.vec3(0.0)


    @ti.kernel
    def update_x_and_v_particle(self, particle: ti.template()):
        for i in range(particle.num_particles):
            particle.v[i] = (particle.y[i] - particle.x[i]) / self.dt
            particle.x[i] = particle.y[i]


    @ti.kernel
    def confine_to_boundary(self):

        for vi in range(self.max_num_verts):

            if self.y[vi][0] > self.grid_size[0]:
                self.y[vi][0] = self.grid_size[0]

            elif self.y[vi][0] < -self.grid_size[0]:
                self.y[vi][0] = -self.grid_size[0]

            elif self.y[vi][1] > self.grid_size[1]:
                self.y[vi][1] = self.grid_size[1]

            elif self.y[vi][1] < -self.grid_size[2]:
                self.y[vi][1] = -self.grid_size[2]

            elif self.y[vi][2] > self.grid_size[2]:
                self.y[vi][2] = self.grid_size[2]

            elif self.y[vi][2] < -self.grid_size[2]:
                self.y[vi][2] = -self.grid_size[2]

    @ti.kernel
    def compute_velocity(self):
        for i in range(self.max_num_verts):
            if self.m_inv[i] > 0.0:
                self.v[i] = (self.y[i] - self.x[i]) / self.dt
            else:
                self.v[i] = ti.math.vec3(0.)


    def solve_constraints_x(self):

        self.dx.fill(0.0)
        self.nc.fill(0)
        self.solve_spring_constraints_x()
        self.solve_collision_constraints_x()
        self.update_dx()

    def solve_constraints_v(self):
        self.dv.fill(0.0)
        self.nc.fill(0)
        self.solve_collision_constraints_v()
        self.update_dv()

    @ti.kernel
    def update_dx(self):
        for vi in range(self.max_num_verts):
            if self.nc[vi] > 0.0:
                self.dx[vi] = self.dx[vi] / self.nc[vi]

            if self.m_inv[vi] > 0.0:
                self.y[vi] += self.dx[vi]

    @ti.kernel
    def update_dv(self):
        for vi in range(self.max_num_verts):
            if self.nc[vi] > 0.0:
                self.dv[vi] = self.dv[vi] / self.nc[vi]

            if self.m_inv[vi] > 0.0:
                self.v[vi] += self.dv[vi]

    def forward(self, n_substeps):
        dt = self.dt
        self.dt = dt / n_substeps

        # self.broad_phase()
        for _ in range(n_substeps):
            self.compute_y()
            self.solve_constraints_x()
            self.confine_to_boundary()
            self.compute_velocity()

            if self.enable_velocity_update:
                self.solve_constraints_v()

            self.update_x()
        self.copy_to_meshes()

            # for mid in range(len(self.meshes)):
            #     self.compute_y(self.meshes[mid])
            #     # self.solve_collision_constraints_x(mid, self.meshes[mid])
            #     self.update_x_and_v(self.meshes[mid])
            #
            # for pid in range(len(self.particles)):
            #     self.compute_y_particle(self.particles[pid])
            #     self.update_x_and_v_particle(self.particles[pid])

        self.dt = dt

