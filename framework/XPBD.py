import taichi as ti
import numpy as np
import distance as di

@ti.data_oriented
class Solver:
    def __init__(self,
                 meshes_dynamic,
                 meshes_static,
                 tet_meshes_dynamic,
                 particles,
                 grid_size,
                 particle_radius,
                 dHat,
                 g,
                 dt):

        self.meshes_dynamic = meshes_dynamic
        self.tet_meshes_dynamic = tet_meshes_dynamic
        self.meshes_static = meshes_static
        self.particles = particles
        self.g = g
        self.dt = ti.field(dtype=ti.f32, shape=1)
        self.dt[0] = dt
        self.dHat = ti.field(dtype=ti.f32, shape=1)
        self.dHat[0] = dHat
        self.grid_size = grid_size
        self.particle_radius = particle_radius
        self.friction_coeff = ti.field(dtype=ti.f32, shape=1)
        self.friction_coeff[0] = 0.1
        self.grid_vertices = ti.Vector.field(n=3, dtype=ti.f32, shape=8)
        self.grid_edge_indices = ti.field(dtype=ti.u32, shape=12 * 2)

        self.padding = 0.9
        self.init_grid()

        self.kernel_radius = 4 * particle_radius
        self.cell_size = 3 * self.kernel_radius
        self.grid_origin = -self.grid_size
        self.grid_num = np.ceil(2 * self.grid_size / self.cell_size).astype(int)
        # print(self.grid_num)

        self.enable_velocity_update = False
        self.enable_collision_handling = False

        self.max_num_verts_dynamic = 0
        self.max_num_edges_dynamic = 0
        self.max_num_faces_dynamic = 0
        self.max_num_tetra_dynamic = 0

        num_meshes_dynamic = len(self.meshes_dynamic) + len(self.tet_meshes_dynamic)
        num_tet_meshes_dynamic = len(self.tet_meshes_dynamic)
        num_vert_offsets = num_meshes_dynamic + len(self.particles) + num_tet_meshes_dynamic

        is_verts_dynamic_empty = not bool(num_vert_offsets)
        is_mesh_dynamic_empty = not bool(len(self.meshes_dynamic))
        is_tet_mesh_dynamic_empty = not bool(len(self.tet_meshes_dynamic))

        # if is_verts_dynamic_empty is True:
        #     num_vert_offsets = 1
        #     self.max_num_verts_dynamic = 1
        #
        #
        # if is_mesh_dynamic_empty is True:
        #     num_meshes_dynamic = 1
        #     self.max_num_edges_dynamic = 1
        #     self.max_num_faces_dynamic = 1
        #
        #
        if is_tet_mesh_dynamic_empty is True:
            num_tet_meshes_dynamic = 1
            self.max_num_tetra_dynamic = 1


        self.offset_verts_dynamic = ti.field(int, shape=num_vert_offsets)
        self.offset_edges_dynamic = ti.field(int, shape=num_meshes_dynamic)
        self.offset_faces_dynamic = ti.field(int, shape=num_meshes_dynamic + num_tet_meshes_dynamic)
        self.offset_tetras_dynamic = ti.field(int, shape=num_tet_meshes_dynamic)

        for mid in range(len(self.meshes_dynamic)):
            self.offset_verts_dynamic[mid] = self.max_num_verts_dynamic
            self.offset_edges_dynamic[mid] = self.max_num_edges_dynamic
            self.offset_faces_dynamic[mid] = self.max_num_faces_dynamic

            self.max_num_verts_dynamic += len(self.meshes_dynamic[mid].verts)
            self.max_num_edges_dynamic += len(self.meshes_dynamic[mid].edges)
            self.max_num_faces_dynamic += len(self.meshes_dynamic[mid].faces)

        self.offset_tet_mesh = self.max_num_verts_dynamic
        self.offset_spring = self.max_num_edges_dynamic

        for tid in range(len(self.tet_meshes_dynamic)):
            self.offset_verts_dynamic[tid + len(self.meshes_dynamic)] = self.max_num_verts_dynamic
            self.offset_edges_dynamic[tid + len(self.meshes_dynamic)] = self.max_num_edges_dynamic
            self.offset_tetras_dynamic[tid] = self.max_num_tetra_dynamic
            self.offset_faces_dynamic[tid + len(self.meshes_dynamic)] = self.max_num_faces_dynamic
            self.max_num_verts_dynamic += len(self.tet_meshes_dynamic[tid].verts)
            self.max_num_tetra_dynamic += len(self.tet_meshes_dynamic[tid].cells)
            self.max_num_faces_dynamic += len(self.tet_meshes_dynamic[tid].faces)
            self.max_num_edges_dynamic += len(self.tet_meshes_dynamic[tid].edges)


        self.offset_particle = self.max_num_verts_dynamic

        for pid in range(len(self.particles)):
            self.offset_verts_dynamic[pid + len(self.meshes_dynamic) + len(self.tet_meshes_dynamic)] = self.max_num_verts_dynamic
            self.max_num_verts_dynamic += self.particles[pid].num_particles


        # print(self.offset_verts_dynamic)
        # print(self.offset_tetras_dynamic)
        # print(self.max_num_verts_dynamic)
        # print(self.max_num_tetra_dynamic)

        if self.max_num_edges_dynamic < 1:
            self.max_num_edges_dynamic = 1

        if self.max_num_verts_dynamic < 1:
            self.max_num_verts_dynamic = 1

        if self.max_num_tetra_dynamic < 1:
            self.max_num_tetra_dynamic = 1

        self.y = ti.Vector.field(n=3, dtype=ti.f32, shape=self.max_num_verts_dynamic)
        self.x = ti.Vector.field(n=3, dtype=ti.f32, shape=self.max_num_verts_dynamic)
        self.dx = ti.Vector.field(n=3, dtype=ti.f32, shape=self.max_num_verts_dynamic)
        self.dv = ti.Vector.field(n=3, dtype=ti.f32, shape=self.max_num_verts_dynamic)
        self.v = ti.Vector.field(n=3, dtype=ti.f32, shape=self.max_num_verts_dynamic)
        self.nc = ti.field(dtype=ti.int32, shape=self.max_num_verts_dynamic)
        self.schur = ti.field(dtype=ti.f32, shape=self.max_num_verts_dynamic)
        self.c = ti.field(dtype=ti.f32, shape=self.max_num_verts_dynamic)
        self.fixed = ti.field(dtype=ti.f32, shape=self.max_num_verts_dynamic)
        self.m_inv = ti.field(dtype=ti.f32, shape=self.max_num_verts_dynamic)
        self.Dm_inv = ti.Matrix.field(n=3, m=3, dtype=ti.f32, shape=self.max_num_tetra_dynamic)
        self.l0 = ti.field(dtype=ti.f32, shape=self.max_num_edges_dynamic)
        self.spring_ids = ti.field(dtype=ti.i32, shape=self.max_num_edges_dynamic)
        self.num_springs = ti.field(dtype=ti.i32, shape=1)
        self.schur_spring = ti.field(dtype=ti.f32, shape=self.max_num_edges_dynamic)
        self.gradient_spring = ti.Vector.field(n=3, dtype=ti.f32, shape=self.max_num_edges_dynamic)

        self.face_indices_dynamic = ti.field(dtype=ti.i32, shape=3 * self.max_num_faces_dynamic)
        self.edge_indices_dynamic = ti.field(dtype=ti.i32, shape=2 * self.max_num_edges_dynamic)
        self.tetra_indices_dynamic = ti.field(dtype=ti.i32, shape=4 * self.max_num_tetra_dynamic)

        self.fixed.fill(1)

        self.cache_size = 500


        self.vt_active_set = ti.Vector.field(n=2, dtype=ti.int32, shape=(self.max_num_verts_dynamic, self.cache_size))
        self.vt_active_set_g0 = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.max_num_verts_dynamic, self.cache_size))
        self.vt_active_set_schur = ti.field(dtype=ti.f32, shape=(self.max_num_verts_dynamic, self.cache_size))
        self.vt_active_set_num = ti.field(int, shape=self.max_num_verts_dynamic)

        self.vt_active_set_dynamic = ti.field(int, shape=(self.max_num_verts_dynamic, self.cache_size))
        self.vt_active_set_g_dynamic = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.max_num_verts_dynamic, self.cache_size, 4))
        self.vt_active_set_schur_dynamic = ti.field(dtype=ti.f32, shape=(self.max_num_verts_dynamic, self.cache_size))
        self.vt_active_set_num_dynamic = ti.field(int, shape=self.max_num_verts_dynamic)

        self.tv_active_set = ti.field(int, shape=(self.max_num_faces_dynamic, self.cache_size))
        self.tv_active_set_g = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.max_num_faces_dynamic, self.cache_size, 3))
        self.tv_active_set_schur = ti.field(dtype=ti.f32, shape=(self.max_num_faces_dynamic, self.cache_size))
        self.tv_active_set_num = ti.field(int, shape=(self.max_num_faces_dynamic))

        self.ee_active_set = ti.field(int, shape=(self.max_num_edges_dynamic, self.cache_size))
        self.ee_active_set_num = ti.field(int, shape=(self.max_num_edges_dynamic))
        self.ee_active_set_g = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.max_num_edges_dynamic, self.cache_size, 2))
        self.ee_active_set_schur = ti.field(dtype=ti.f32, shape=(self.max_num_edges_dynamic, self.cache_size))

        self.ee_active_set_dynamic = ti.Vector.field(n=2, dtype=int, shape=self.cache_size)
        self.ee_active_set_g_dynamic = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.cache_size, 4))
        self.ee_active_set_schur_dynamic = ti.field(dtype=ti.f32, shape=self.cache_size)
        self.ee_active_set_num_dynamic = ti.field(int, shape=1)


        self.num_cells = int(self.grid_num[0] * self.grid_num[1] * self.grid_num[2])
        self.particle_neighbours = ti.field(dtype=int, shape=(self.max_num_verts_dynamic, self.cache_size))
        self.particle_neighbours_gradients = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.max_num_verts_dynamic, self.cache_size))
        self.num_particle_neighbours = ti.field(int, shape=self.max_num_verts_dynamic)

        self.cell_particle_ids = ti.field(dtype=ti.uint32, shape=(self.num_cells, self.cache_size))
        self.cell_num_particles = ti.field(dtype=ti.uint32, shape=self.num_cells)

        self.max_num_verts_static = 0
        self.max_num_edges_static = 0
        self.max_num_faces_static = 0

        num_meshes_static = len(self.meshes_static)

        self.is_meshes_static_empty = not bool(num_meshes_static)
        if self.is_meshes_static_empty is True:
            num_meshes_static = 1
            self.max_num_verts_static = 1
            self.max_num_edges_static = 1
            self.max_num_faces_static = 1


        self.offset_verts_static = ti.field(int, shape=num_meshes_static)
        self.offset_edges_static = ti.field(int, shape=num_meshes_static)
        self.offset_faces_static = ti.field(int, shape=num_meshes_static)

        for mid in range(len(self.meshes_static)):
            self.offset_verts_static[mid] = self.max_num_verts_static
            self.offset_edges_static[mid] = self.max_num_edges_static
            self.offset_faces_static[mid] = self.max_num_faces_static
            self.max_num_verts_static += len(self.meshes_static[mid].verts)
            self.max_num_edges_static += len(self.meshes_static[mid].edges)
            self.max_num_faces_static += len(self.meshes_static[mid].faces)

        self.face_indices_static = ti.field(dtype=ti.i32, shape=3 * self.max_num_faces_static)
        self.edge_indices_static = ti.field(dtype=ti.i32, shape=2 * self.max_num_edges_static)

        print(self.max_num_edges_static)

        self.x_static = ti.Vector.field(n=3, dtype=ti.f32, shape=self.max_num_verts_static)

        if is_verts_dynamic_empty is True:
            self.max_num_verts_dynamic = 0

        if is_mesh_dynamic_empty and is_tet_mesh_dynamic_empty:
            self.max_num_edges_dynamic = 0
            self.max_num_faces_dynamic = 0

        if is_tet_mesh_dynamic_empty is True:
            self.max_num_tetra_dynamic = 0

        if self.is_meshes_static_empty is True:
            self.max_num_verts_static = 0
            self.max_num_edges_static = 0
            self.max_num_faces_static = 0

        self.init_mesh_aggregation()
        self.init_particle_aggregation()
        self.grid_particles_num = ti.field(int, shape=self.num_cells)
        self.grid_particles_num_temp = ti.field(int, shape= self.num_cells)
        self.prefix_sum_executor = ti.algorithms.PrefixSumExecutor(self.grid_particles_num.shape[0])

        self.grid_ids = ti.field(int, shape=self.max_num_verts_dynamic)
        self.grid_ids_buffer = ti.field(int, shape=self.max_num_verts_dynamic)
        self.grid_ids_new = ti.field(int, shape=self.max_num_verts_dynamic)
        self.cur2org = ti.field(int, shape=self.max_num_verts_dynamic)

        self.grid_particles_num_static = ti.field(int, shape=int(self.grid_num[0] * self.grid_num[1] * self.grid_num[2]))
        self.grid_particles_num_temp_static = ti.field(int, shape=int(self.grid_num[0] * self.grid_num[1] * self.grid_num[2]))
        self.prefix_sum_executor_static = ti.algorithms.PrefixSumExecutor(self.grid_particles_num_static.shape[0])

        num_grid_ids = self.max_num_verts_static + self.max_num_faces_static + self.max_num_edges_static

        if num_grid_ids < 1:
            num_grid_ids = 1

        self.grid_ids_static = ti.field(int, shape=num_grid_ids)
        self.grid_ids_buffer_static = ti.field(int, shape=num_grid_ids)
        self.grid_ids_new_static = ti.field(int, shape=num_grid_ids)
        self.cur2org_static = ti.field(int, shape=num_grid_ids)



        self.frame = ti.field(dtype=ti.i32, shape=1)
        self.frame[0] = 0

        self.max_num_anim = 40
        self.num_animation = ti.field(dtype=ti.i32, shape=4)   # maximum number of handle set
        self.cur_animation = ti.field(dtype=ti.i32, shape=4)   # maximum number of handle set
        self.anim_rotation_mat = ti.Matrix.field(4, 4, dtype=ti.f32, shape=4)

        self.anim_local_origin = ti.Vector.field(3, dtype=ti.f32, shape=4)
        self.active_anim_frame = ti.field(dtype=ti.i32, shape=(4, self.max_num_anim)) # maximum number of animation
        self.action_anim = ti.Vector.field(6, dtype=ti.f32, shape=(4, self.max_num_anim)) # maximum number of animation, a animation consist (vx,vy,vz,rx,ry,rz)
        self.anim_x = ti.Vector.field(n=3, dtype=ti.f32, shape=self.max_num_verts_dynamic)

        self.broad_phase_static()

    @ti.kernel
    def __init_animation_pos(self, is_selected: ti.template()):
        for i in is_selected:
            if is_selected != 0:
                self.anim_x[i] = self.x[i]

        for i in ti.ndrange(4):
            count = 0.0
            origin = ti.Vector([0.0, 0.0, 0.0])
            idx_set_count = i + 1

            for pidx in ti.ndrange(self.max_num_verts_dynamic):
                if is_selected[pidx] == idx_set_count:
                    count += 1
                    origin += self.x[pidx]

            if count > 0.001:
                self.anim_local_origin[i] = origin / count

    def _reset_animation(self):
        self.num_animation.fill(0)
        self.cur_animation.fill(0)
        self.active_anim_frame.fill(0)
        self.action_anim.fill(0.0)
        self.anim_local_origin.fill(0.0)
        self.anim_rotation_mat.fill(0.0)

    def _set_animation(self, animationDict, is_selected):
        self.num_animation.fill(0)
        self.cur_animation.fill(0)
        self.active_anim_frame.fill(0)
        self.action_anim.fill(0.0)
        self.anim_local_origin.fill(0.0)

        self.num_animation[0] = len(animationDict[1])
        self.num_animation[1] = len(animationDict[2])
        self.num_animation[2] = len(animationDict[3])
        self.num_animation[3] = len(animationDict[4])
        self.num_animation[3] = len(animationDict[4])

        if (self.num_animation[0] > self.max_num_anim) or (self.num_animation[1] > self.max_num_anim) \
                or (self.num_animation[2] > self.max_num_anim) or (self.num_animation[3] > self.max_num_anim):
            print("warning :: length of some animation is longer than ", self.max_num_anim,
                  ". Subsequent animations are ignored")
            self.num_animation[0] = self.num_animation[0] if self.num_animation[
                                                                 0] < self.max_num_anim else self.max_num_anim
            self.num_animation[1] = self.num_animation[1] if self.num_animation[
                                                                 1] < self.max_num_anim else self.max_num_anim
            self.num_animation[2] = self.num_animation[2] if self.num_animation[
                                                                 2] < self.max_num_anim else self.max_num_anim
            self.num_animation[3] = self.num_animation[3] if self.num_animation[
                                                                 3] < self.max_num_anim else self.max_num_anim

        self.__init_animation_pos(is_selected)

        for ic in range(4):
            animations_ic = animationDict[ic + 1]
            for a in range(self.num_animation[ic]):
                animation = animations_ic[a]

                self.active_anim_frame[ic, a] = animation[6]
                for j in range(6):
                    self.action_anim[ic, a][j] = animation[j]

        self.set_fixed_vertices(is_selected)

        # print(self.num_animation)
        # print(self.active_anim_frame)
        # print(self.action_anim)
        # print(self.anim_local_origin)

        # self.frame[0] = 0
        #
        # self.max_num_anim = 40
        # self.num_animation = ti.field(dtype = ti.i32,shape = 4)   # maximum number of handle set
        # self.anim_local_origin = ti.Vector.field(3, dtype=ti.f32, shape=4)
        # self.active_anim_frame = ti.field(dtype = ti.i32,shape = (4,self.max_num_anim)) # maximum number of animation
        # self.action_anim = ti.Vector.field(6,dtype = ti.f32,shape = (4,self.max_num_anim)) # maximum number of animation, a animation consist (vx,vy,vz,rx,ry,rz)
        #
        # self.anim_x = ti.Vector.field(n=3, dtype=ti.f32, shape=self.max_num_verts_dynamic)

    @ti.func
    def apply_rotation(self, mat_rot, vec):
        v = ti.Vector([vec[0], vec[1], vec[2], 1])
        vv = mat_rot @ v
        return ti.Vector([vv[0], vv[1], vv[2]])

    @ti.func
    def get_animation_rotation_mat(self, axis, degree):
        s = ti.sin(degree * 0.5)
        c = ti.cos(degree * 0.5)
        axis = s * axis
        q = ti.Vector([axis[0], axis[1], axis[2], c])
        M = ti.math.mat4(0.0)

        M[3, 3] = 1.0

        M[0, 0] = 1 - 2 * (q[1] ** 2 + q[2] ** 2)
        M[1, 1] = 1 - 2 * (q[0] ** 2 + q[2] ** 2)
        M[2, 2] = 1 - 2 * (q[0] ** 2 + q[1] ** 2)

        M[2, 1] = 2 * (q[1] * q[2] + q[3] * q[0])
        M[1, 2] = 2 * (q[1] * q[2] - q[3] * q[0])

        M[2, 0] = 2 * (q[0] * q[2] - q[3] * q[1])
        M[0, 2] = 2 * (q[0] * q[2] + q[3] * q[1])

        M[1, 0] = 2 * (q[0] * q[1] + q[3] * q[2])
        M[0, 1] = 2 * (q[0] * q[1] - q[3] * q[2])

        return M

    @ti.kernel
    def animate_handle(self, is_selected: ti.template()):

        for idx_set in ti.ndrange(4):
            cur_anim = self.cur_animation[idx_set]
            max_anim = self.num_animation[idx_set]
            is_animation_changed = False or (self.frame[0] == 0)  # when frame = 0 animation changed
            while self.active_anim_frame[idx_set, cur_anim] < self.frame[0] and cur_anim < max_anim:
                self.cur_animation[idx_set] = self.cur_animation[idx_set] + 1
                cur_anim = self.cur_animation[idx_set]
                is_animation_changed = True

            if cur_anim < max_anim:
                vel = self.action_anim[idx_set, cur_anim]
                lin_vel = ti.Vector([vel[0], vel[1], vel[2]])
                self.anim_local_origin[idx_set] += lin_vel * self.dt[0]

                ang_vel = ti.Vector([vel[3], vel[4], vel[5]])
                degree_rate = ang_vel.norm()

                if is_animation_changed and degree_rate > 1e-4:
                    axis = ti.math.normalize(ang_vel)
                    self.anim_rotation_mat[idx_set] = self.get_animation_rotation_mat(axis, degree_rate * self.dt[0])

        for i in self.anim_x:
            if is_selected[i] >= 1:

                idx_set = is_selected[i] - 1

                cur_anim = int(self.cur_animation[idx_set])
                max_anim = int(self.num_animation[idx_set])

                if cur_anim < max_anim:
                    vel = self.action_anim[idx_set, cur_anim]
                    lin_vel = ti.Vector([vel[0], vel[1], vel[2]])
                    self.anim_x[i] = self.anim_x[i] + lin_vel * self.dt[0]

                    ang_vel = ti.Vector([vel[3], vel[4], vel[5]])
                    degree_rate = ang_vel.norm()
                    if degree_rate > 1e-4:
                        mat_rot = self.anim_rotation_mat[idx_set]

                        self.anim_x[i] = self.apply_rotation(mat_rot,
                                                             self.anim_x[i] - self.anim_local_origin[idx_set]) + \
                                         self.anim_local_origin[idx_set]

                    self.x[i] = self.anim_x[i]

    def copy_to_meshes(self):
        for mid in range(len(self.meshes_dynamic)):
            self.copy_to_meshes_device(self.offset_verts_dynamic[mid], self.meshes_dynamic[mid])

        for tid in range(len(self.tet_meshes_dynamic)):
            self.copy_to_tet_meshes_device(self.offset_verts_dynamic[tid + len(self.meshes_dynamic)], self.tet_meshes_dynamic[tid])

    @ti.kernel
    def copy_to_meshes_device(self, offset: ti.int32, mesh: ti.template()):
        for v in mesh.verts:
            v.x = self.x[offset + v.id]
            v.v = self.v[offset + v.id]

    @ti.kernel
    def copy_to_tet_meshes_device(self, offset: ti.int32, mesh: ti.template()):
        for v in mesh.verts:
            v.x = self.x[offset + v.id]
            v.v = self.v[offset + v.id]

    def copy_to_particles(self):
        for pid in range(len(self.particles)):
            self.copy_to_particles_device(self.offset_verts_dynamic[pid + len(self.meshes_dynamic) + len(self.tet_meshes_dynamic)], self.particles[pid])

    @ti.kernel
    def copy_to_particles_device(self, offset: ti.int32, particle: ti.template()):
        for i in range(particle.num_particles):
            particle.x[i] = self.x[offset + i]
            particle.v[i] = self.v[offset + i]


    @ti.kernel
    def init_rest_length(self):

        # l0_min = 1e3
        for ei in range(self.max_num_edges_dynamic):
            v0, v1 = self.edge_indices_dynamic[2 * ei + 0], self.edge_indices_dynamic[2 * ei + 1]
            x0, x1 = self.x[v0], self.x[v1]
            x10 = x0 - x1
            self.l0[ei] = x10.norm()
            # if l0_min >  self.l0[ei]:
            #     l0_min = self.l0[ei]
            # l0_min = ti.atomic_min(l0_min,  self.l0[ei])

        # print(l0_min)
        # rif l0_min * l0_min > self.dHat[0]:
        #     self.dHat[0] = 0.9 * l0_min * l0_min



    @ti.kernel
    def init_Dm_inv(self):

        for ti in range(self.max_num_tetra_dynamic):
            v0 = self.tetra_indices_dynamic[4 * ti + 0]
            v1 = self.tetra_indices_dynamic[4 * ti + 1]
            v2 = self.tetra_indices_dynamic[4 * ti + 2]
            v3 = self.tetra_indices_dynamic[4 * ti + 3]

            x0, x1, x2, x3 = self.x[v0], self.x[v1], self.x[v2], self.x[v3]

            Dm = ti.Matrix.cols([x0 - x3, x1 - x3, x2 - x3])
            self.Dm_inv[ti] = Dm.inverse()



    def init_mesh_aggregation(self):
        for mid in range(len(self.meshes_dynamic)):
            self.init_mesh_quantities_dynamic_device(self.offset_verts_dynamic[mid], self.meshes_dynamic[mid])
            self.init_edge_indices_dynamic_device(self.offset_verts_dynamic[mid], self.offset_edges_dynamic[mid], self.meshes_dynamic[mid])
            self.init_face_indices_dynamic_device(self.offset_verts_dynamic[mid], self.offset_faces_dynamic[mid], self.meshes_dynamic[mid])

        for tid in range(len(self.tet_meshes_dynamic)):
            self.init_tet_mesh_quantities_dynamic_device(self.offset_verts_dynamic[tid + len(self.meshes_dynamic)], self.tet_meshes_dynamic[tid])
            self.init_edge_indices_dynamic_device(self.offset_verts_dynamic[tid + len(self.meshes_dynamic)], self.offset_edges_dynamic[tid + len(self.meshes_dynamic)], self.tet_meshes_dynamic[tid])
            self.init_face_indices_dynamic_device(self.offset_verts_dynamic[tid + len(self.meshes_dynamic)], self.offset_faces_dynamic[tid + len(self.meshes_dynamic)], self.tet_meshes_dynamic[tid])
            self.init_tet_indices_dynamic_device(self.offset_verts_dynamic[tid + len(self.meshes_dynamic)], self.offset_tetras_dynamic[tid], self.tet_meshes_dynamic[tid])

        for mid in range(len(self.meshes_static)):
            self.init_quantities_static_device(self.offset_verts_static[mid], self.meshes_static[mid])
            self.init_edge_indices_static_device(self.offset_verts_static[mid], self.offset_edges_static[mid], self.meshes_static[mid])
            self.init_face_indices_static_device(self.offset_verts_static[mid], self.offset_faces_static[mid], self.meshes_static[mid])

        self.init_rest_length()
        self.init_Dm_inv()


    @ti.kernel
    def init_mesh_quantities_dynamic_device(self, offset: ti.int32, mesh: ti.template()):
        for v in mesh.verts:
            self.x[offset + v.id] = v.x
            self.v[offset + v.id] = v.v
            self.m_inv[offset + v.id] = v.m_inv

    @ti.kernel
    def init_tet_mesh_quantities_dynamic_device(self, offset: ti.int32, mesh: ti.template()):
        for v in mesh.verts:
            self.x[offset + v.id] = v.x
            self.v[offset + v.id] = v.v
            self.m_inv[offset + v.id] = v.m_inv


    @ti.kernel
    def init_quantities_static_device(self, offset: ti.int32, mesh: ti.template()):
        for v in mesh.verts:
            self.x_static[offset + v.id] = v.x


    @ti.kernel
    def init_edge_indices_dynamic_device(self, offset_verts: ti.int32, offset_edges: ti.int32, mesh: ti.template()):
        for e in mesh.edges:
            self.edge_indices_dynamic[2 * (offset_edges + e.id) + 0] = e.verts[0].id + offset_verts
            self.edge_indices_dynamic[2 * (offset_edges + e.id) + 1] = e.verts[1].id + offset_verts

    @ti.kernel
    def init_face_indices_dynamic_device(self, offset_verts: ti.int32, offset_faces: ti.int32, mesh: ti.template()):
        for f in mesh.faces:
            self.face_indices_dynamic[3 * (offset_faces + f.id) + 0] = f.verts[0].id + offset_verts
            self.face_indices_dynamic[3 * (offset_faces + f.id) + 1] = f.verts[1].id + offset_verts
            self.face_indices_dynamic[3 * (offset_faces + f.id) + 2] = f.verts[2].id + offset_verts

    @ti.kernel
    def init_tet_indices_dynamic_device(self, offset_verts: ti.int32, offset_tets: ti.int32, mesh: ti.template()):
        for c in mesh.cells:
            self.tetra_indices_dynamic[4 * (offset_tets + c.id) + 0] = c.verts[0].id + offset_verts
            self.tetra_indices_dynamic[4 * (offset_tets + c.id) + 1] = c.verts[1].id + offset_verts
            self.tetra_indices_dynamic[4 * (offset_tets + c.id) + 2] = c.verts[2].id + offset_verts
            self.tetra_indices_dynamic[4 * (offset_tets + c.id) + 3] = c.verts[3].id + offset_verts


    @ti.kernel
    def init_edge_indices_static_device(self, offset_verts: ti.int32, offset_edges: ti.int32, mesh: ti.template()):
        for e in mesh.edges:
            self.edge_indices_static[2 * (offset_edges + e.id) + 0] = e.verts[0].id + offset_verts
            self.edge_indices_static[2 * (offset_edges + e.id) + 1] = e.verts[1].id + offset_verts

    @ti.kernel
    def init_face_indices_static_device(self, offset_verts: ti.int32, offset_faces: ti.int32, mesh: ti.template()):
        for f in mesh.faces:
            self.face_indices_static[3 * (offset_faces + f.id) + 0] = f.verts[0].id + offset_verts
            self.face_indices_static[3 * (offset_faces + f.id) + 1] = f.verts[1].id + offset_verts
            self.face_indices_static[3 * (offset_faces + f.id) + 2] = f.verts[2].id + offset_verts


    def init_particle_aggregation(self):
        for pid in range(len(self.particles)):
            self.init_particle_quantities_dynamic_device(self.offset_verts_dynamic[pid + len(self.meshes_dynamic) + len(self.tet_meshes_dynamic)], self.particles[pid])

    @ti.kernel
    def init_particle_quantities_dynamic_device(self, offset: ti.int32, particle: ti.template()):
        for i in range(particle.num_particles):
            self.x[offset + i] = particle.x[i]
            self.v[offset + i] = particle.v[i]
            self.m_inv[offset + i] = particle.m_inv[i]

    def init_grid(self):

        self.grid_vertices[0] = self.padding * ti.math.vec3(self.grid_size[0], self.grid_size[1], self.grid_size[2])
        self.grid_vertices[1] = self.padding * ti.math.vec3(-self.grid_size[0], self.grid_size[1], self.grid_size[2])
        self.grid_vertices[2] = self.padding * ti.math.vec3(-self.grid_size[0], self.grid_size[1], -self.grid_size[2])
        self.grid_vertices[3] = self.padding * ti.math.vec3(self.grid_size[0], self.grid_size[1], -self.grid_size[2])

        self.grid_vertices[4] = self.padding * ti.math.vec3(self.grid_size[0], -self.grid_size[1], self.grid_size[2])
        self.grid_vertices[5] = self.padding * ti.math.vec3(-self.grid_size[0], -self.grid_size[1], self.grid_size[2])
        self.grid_vertices[6] = self.padding * ti.math.vec3(-self.grid_size[0], -self.grid_size[1], -self.grid_size[2])
        self.grid_vertices[7] = self.padding * ti.math.vec3(self.grid_size[0], -self.grid_size[1], -self.grid_size[2])

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
        self.frame[0] = 0

        for mid in range(len(self.meshes_dynamic)):
            self.meshes_dynamic[mid].reset()

        for pid in range(len(self.particles)):
            self.particles[pid].reset()

        for tid in range(len(self.tet_meshes_dynamic)):
            self.tet_meshes_dynamic[tid].reset()

        self.init_mesh_aggregation()
        self.init_particle_aggregation()
        self._reset_animation()


    @ti.kernel
    def compute_y(self):

        for i in range(self.max_num_verts_dynamic):
            self.y[i] = self.x[i] + self.fixed[i] * (self.dt[0] * self.v[i] + self.g * self.dt[0] * self.dt[0])


    @ti.kernel
    def counting_sort(self):
        # FIXME: make it the actual particle num
        for i in range(self.max_num_verts_dynamic):
            I = self.max_num_verts_dynamic - 1 - i
            base_offset = 0
            if self.grid_ids[I] - 1 >= 0:
                base_offset = self.grid_particles_num[self.grid_ids[I] - 1]
            self.grid_ids_new[I] = ti.atomic_sub(self.grid_particles_num_temp[self.grid_ids[I]], 1) - 1 + base_offset

        for i in self.grid_ids:
            new_index = self.grid_ids_new[i]
            self.cur2org[new_index] = i

    @ti.kernel
    def counting_sort_static(self):
        # FIXME: make it the actual particle num
        for i in range(self.max_num_verts_static + self.max_num_faces_static + self.max_num_edges_static):
            I = self.max_num_verts_static + self.max_num_faces_static + self.max_num_edges_static - 1 - i
            base_offset_static = 0
            if self.grid_ids_static[I] - 1 >= 0:
                base_offset_static = self.grid_particles_num_static[self.grid_ids_static[I] - 1]
            self.grid_ids_new_static[I] = ti.atomic_sub(self.grid_particles_num_temp_static[self.grid_ids_static[I]], 1) - 1 + base_offset_static

        for i in self.grid_ids_static:
            new_index = self.grid_ids_new_static[i]
            self.cur2org_static[new_index] = i

    @ti.func
    def flatten_cell_id(self, cell_id):
        return cell_id[0] * self.grid_num[1] * self.grid_num[2] + cell_id[1] * self.grid_num[2] + cell_id[2]

    @ti.func
    def pos_to_index(self, pos):
        idx = ((pos - self.grid_origin) / self.cell_size).cast(int)

        for i in ti.static(range(3)):
            if idx[i] < 0:
                idx[i] = 0

            if idx[i] > self.grid_num[i] - 1:
                idx[i] = self.grid_num[i] - 1
        return idx

    @ti.func
    def get_flatten_grid_index(self, pos: ti.math.vec3):
        return self.flatten_cell_id(self.pos_to_index(pos))




    def broad_phase(self):
        # self.grid_particles_num.fill(0)
        self.update_grid_id()
        self.prefix_sum_executor.run(self.grid_particles_num)
        self.counting_sort()

    def broad_phase_static(self):

        # self.grid_particles_num.fill(0)
        self.update_grid_id_static()
        self.prefix_sum_executor_static.run(self.grid_particles_num_static)
        self.counting_sort_static()

    @ti.kernel
    def search_neighbours(self):

        for ci in range(self.num_cells):
            self.cell_num_particles[ci] = 0

        for vi in range(self.max_num_verts_dynamic):
            grid_index = self.get_flatten_grid_index(self.x[vi])
            if self.cell_num_particles[grid_index] < self.cache_size:
                self.cell_particle_ids[grid_index, self.cell_num_particles[grid_index]] = vi
                ti.atomic_add(self.cell_num_particles[grid_index], 1)


    @ti.kernel
    def update_grid_id(self):

        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num[I] = 0

        #TODO: update the following two for-loops into a single one
        for i in range(self.max_num_verts_dynamic):
            # if i < self.max_num_verts_dynamic:
            vi = i
            grid_index = self.get_flatten_grid_index(self.x[vi])
            self.grid_ids[vi] = grid_index
            ti.atomic_add(self.grid_particles_num[grid_index], 1)
            #
            # else:
            #     ei = i - self.max_num_verts_dynamic
            #
            #     v0, v1 = self.edge_indices_dynamic[2 * ei + 0], self.edge_indices_dynamic[2 * ei + 1]
            #     x0, x1 = self.x[v0], self.x[v1]
            #
            #     center = 0.5 * (x0 + x1)
            #
            #     grid_index = self.get_flatten_grid_index(center)
            #     self.grid_ids[ei + self.max_num_edges_dynamic] = grid_index
            #     ti.atomic_add(self.grid_particles_num[grid_index], 1)

        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num_temp[I] = self.grid_particles_num[I]

    @ti.kernel
    def update_grid_id_static(self):

        for I in ti.grouped(self.grid_particles_num_static):
            self.grid_particles_num_static[I] = 0

        # TODO: update the following two for-loops into a single one
        for vi in range(self.max_num_verts_static):
            grid_index = self.get_flatten_grid_index(self.x_static[vi])
            self.grid_ids_static[vi] = grid_index
            ti.atomic_add(self.grid_particles_num_static[grid_index], 1)

        for fi in range(self.max_num_faces_static):

            v0, v1, v2 = self.face_indices_static[3 * fi + 0], self.face_indices_static[3 * fi + 1], self.face_indices_static[3 * fi + 2]
            x0, x1, x2 = self.x_static[v0], self.x_static[v1], self.x_static[v2]

            center = (x0 + x1 + x2) / 3.0

            grid_index = self.get_flatten_grid_index(center)
            self.grid_ids_static[fi + self.max_num_verts_static] = grid_index
            ti.atomic_add(self.grid_particles_num_static[grid_index], 1)

        for ei in range(self.max_num_edges_static):
            v0, v1 = self.edge_indices_static[2 * ei + 0], self.edge_indices_static[2 * ei + 1]
            x0, x1 = self.x_static[v0], self.x_static[v1]

            center = 0.5 * (x0 + x1)

            grid_index = self.get_flatten_grid_index(center)
            self.grid_ids_static[ei + self.max_num_verts_static + self.max_num_edges_static] = grid_index
            ti.atomic_add(self.grid_particles_num_static[grid_index], 1)

        for I in ti.grouped(self.grid_particles_num_static):
            self.grid_particles_num_temp_static[I] = self.grid_particles_num_static[I]


    @ti.func
    def is_in_face(self, vid, fid):

        v1 = self.face_indices_dynamic[3 * fid + 0]
        v2 = self.face_indices_dynamic[3 * fid + 1]
        v3 = self.face_indices_dynamic[3 * fid + 2]

        return (v1 == vid) or (v2 == vid) or (v3 == vid)

    @ti.func
    def share_vertex(self, ei0, ei1):

        v0 = self.edge_indices_dynamic[2 * ei0 + 0]
        v1 = self.edge_indices_dynamic[2 * ei0 + 1]
        v2 = self.edge_indices_dynamic[2 * ei1 + 0]
        v3 = self.edge_indices_dynamic[2 * ei1 + 1]

        return (v0 == v2) or (v0 == v3) or (v1 == v2) or (v1 == v3)



    @ti.func
    def solve_collision_vt_dynamic_x(self, vid, fid, dHat):

        v0 = vid
        v1 = self.face_indices_dynamic[3 * fid + 0]
        v2 = self.face_indices_dynamic[3 * fid + 1]
        v3 = self.face_indices_dynamic[3 * fid + 2]

        x0 = self.y[v0]
        x1 = self.y[v1]
        x2 = self.y[v2]
        x3 = self.y[v3]

        dtype = di.d_type_PT(x0, x1, x2, x3)

        if dtype == 0:
            d = di.d_PP(x0, x1)
            g0, g1 = di.g_PP(x0, x1)
            if d < dHat:
                schur = self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) + self.fixed[v1] * self.m_inv[v1] * g1.dot(g1) + 1e-4
                ld = (dHat - d) / schur
                self.dx[v0] += self.fixed[v0] * self.m_inv[v0] * ld * g0
                self.dx[v1] += self.fixed[v1] * self.m_inv[v1] * ld * g1

                self.nc[v0] += 1
                self.nc[v1] += 1

                if self.vt_active_set_num_dynamic[vid] < self.cache_size:
                    self.vt_active_set_dynamic[vid, self.vt_active_set_num_dynamic[vid]] = fid
                    self.vt_active_set_schur_dynamic[vid, self.vt_active_set_num_dynamic[vid]] = schur
                    self.vt_active_set_g_dynamic[vid, self.vt_active_set_num_dynamic[vid], 0] = g0
                    self.vt_active_set_g_dynamic[vid, self.vt_active_set_num_dynamic[vid], 1] = g1
                    ti.atomic_add(self.vt_active_set_num_dynamic[vid], 1)

        elif dtype == 1:
            d = di.d_PP(x0, x2)
            g0, g2 = di.g_PP(x0, x2)
            if d < dHat:
                schur = self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) + self.fixed[v2] * self.m_inv[v2] * g2.dot(g2) + 1e-4
                ld = (dHat - d) / schur
                self.dx[v0] += self.fixed[v0] * self.m_inv[v0] * ld * g0
                self.dx[v2] += self.fixed[v1] * self.m_inv[v2] * ld * g2
                self.nc[v0] += 1
                self.nc[v2] += 1

                if self.vt_active_set_num_dynamic[vid] < self.cache_size:
                    self.vt_active_set_dynamic[vid, self.vt_active_set_num_dynamic[vid]] = fid
                    self.vt_active_set_schur_dynamic[vid, self.vt_active_set_num_dynamic[vid]] = schur
                    self.vt_active_set_g_dynamic[vid, self.vt_active_set_num_dynamic[vid], 0] = g0
                    self.vt_active_set_g_dynamic[vid, self.vt_active_set_num_dynamic[vid], 2] = g2
                    ti.atomic_add(self.vt_active_set_num_dynamic[vid], 1)

        elif dtype == 2:
            d = di.d_PP(x0, x3)
            g0, g3 = di.g_PP(x0, x3)
            if d < dHat:
                schur = self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) + self.fixed[v3] * self.m_inv[v3] * g3.dot(g3) + 1e-4
                ld = (dHat - d) / schur

                self.dx[v0] += self.fixed[v0] * self.m_inv[v0] * ld * g0
                self.dx[v3] += self.fixed[v3] * self.m_inv[v3] * ld * g3

                self.nc[v0] += 1
                self.nc[v3] += 1

                if self.vt_active_set_num_dynamic[vid] < self.cache_size:
                    self.vt_active_set_dynamic[vid, self.vt_active_set_num_dynamic[vid]] = fid
                    self.vt_active_set_schur_dynamic[vid, self.vt_active_set_num_dynamic[vid]] = schur
                    self.vt_active_set_g_dynamic[vid, self.vt_active_set_num_dynamic[vid], 0] = g0
                    self.vt_active_set_g_dynamic[vid, self.vt_active_set_num_dynamic[vid], 3] = g3
                    ti.atomic_add(self.vt_active_set_num_dynamic[vid], 1)

        elif dtype == 3:
            d = di.d_PE(x0, x1, x2)
            g0, g1, g2 = di.g_PE(x0, x1, x2)
            if d < dHat:

                schur = self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) + self.fixed[v1] * self.m_inv[v1] * g1.dot(g1) + self.fixed[v2] * self.m_inv[v2] * g2.dot(g2) + 1e-4
                ld = (dHat - d) / schur

                self.dx[v0] += self.fixed[v0] * self.m_inv[v0] * ld * g0
                self.dx[v1] += self.fixed[v1] * self.m_inv[v1] * ld * g1
                self.dx[v2] += self.fixed[v2] * self.m_inv[v2] * ld * g2

                self.nc[v0] += 1
                self.nc[v1] += 1
                self.nc[v2] += 1

                if self.vt_active_set_num_dynamic[vid] < self.cache_size:
                    self.vt_active_set_dynamic[vid, self.vt_active_set_num_dynamic[vid]] = fid
                    self.vt_active_set_schur_dynamic[vid, self.vt_active_set_num_dynamic[vid]] = schur
                    self.vt_active_set_g_dynamic[vid, self.vt_active_set_num_dynamic[vid], 0] = g0
                    self.vt_active_set_g_dynamic[vid, self.vt_active_set_num_dynamic[vid], 1] = g1
                    self.vt_active_set_g_dynamic[vid, self.vt_active_set_num_dynamic[vid], 2] = g2
                    ti.atomic_add(self.vt_active_set_num_dynamic[vid], 1)

        elif dtype == 4:
            d = di.d_PE(x0, x2, x3)
            g0, g2, g3 = di.g_PE(x0, x2, x3)
            if d < dHat:

                schur = self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) + self.fixed[v2] * self.m_inv[v2] * g2.dot(g2) + self.fixed[v3] * self.m_inv[v3] * g3.dot(g3) + 1e-4
                ld = (dHat - d) / schur

                self.dx[v0] += self.fixed[v0] * self.m_inv[v0] * ld * g0
                self.dx[v2] += self.fixed[v2] * self.m_inv[v2] * ld * g2
                self.dx[v3] += self.fixed[v3] * self.m_inv[v3] * ld * g3
                self.nc[v0] += 1
                self.nc[v2] += 1
                self.nc[v3] += 1

                if self.vt_active_set_num_dynamic[vid] < self.cache_size:
                    self.vt_active_set_dynamic[vid, self.vt_active_set_num_dynamic[vid]] = fid
                    self.vt_active_set_schur_dynamic[vid, self.vt_active_set_num_dynamic[vid]] = schur
                    self.vt_active_set_g_dynamic[vid, self.vt_active_set_num_dynamic[vid], 0] = g0
                    self.vt_active_set_g_dynamic[vid, self.vt_active_set_num_dynamic[vid], 2] = g2
                    self.vt_active_set_g_dynamic[vid, self.vt_active_set_num_dynamic[vid], 3] = g3
                    ti.atomic_add(self.vt_active_set_num_dynamic[vid], 1)

        elif dtype == 5:
            d = di.d_PE(x0, x1, x3)
            g0, g1, g3 = di.g_PE(x0, x1, x3)
            if d < dHat:

                schur = self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) + self.fixed[v1] * self.m_inv[v1] * g1.dot(g1) + self.fixed[v3] * self.m_inv[v3] * g3.dot(g3) + 1e-4
                ld = (dHat - d) / schur

                self.dx[v0] += self.fixed[v0] * self.m_inv[v0] * ld * g0
                self.dx[v1] += self.fixed[v1] * self.m_inv[v1] * ld * g1
                self.dx[v3] += self.fixed[v3] * self.m_inv[v3] * ld * g3
                self.nc[v0] += 1
                self.nc[v1] += 1
                self.nc[v3] += 1

                if self.vt_active_set_num_dynamic[vid] < self.cache_size:
                    self.vt_active_set_dynamic[vid, self.vt_active_set_num_dynamic[vid]] = fid
                    self.vt_active_set_schur_dynamic[vid, self.vt_active_set_num_dynamic[vid]] = schur
                    self.vt_active_set_g_dynamic[vid, self.vt_active_set_num_dynamic[vid], 0] = g0
                    self.vt_active_set_g_dynamic[vid, self.vt_active_set_num_dynamic[vid], 1] = g1
                    self.vt_active_set_g_dynamic[vid, self.vt_active_set_num_dynamic[vid], 3] = g3
                    ti.atomic_add(self.vt_active_set_num_dynamic[vid], 1)

        elif dtype == 6:
            d = di.d_PT(x0, x1, x2, x3)
            g0, g1, g2, g3 = di.g_PT(x0, x1, x2, x3)

            if d < dHat:
                schur = self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) + self.fixed[v1] * self.m_inv[v1] * g1.dot(g1) + self.fixed[v2] * self.m_inv[v2] * g2.dot(g2) + self.fixed[v3] * self.m_inv[v3] * g3.dot(g3) + 1e-4
                ld = (dHat - d) / schur
                self.dx[v0] += self.fixed[v0] * self.m_inv[v0] * ld * g0
                self.dx[v1] += self.fixed[v1] * self.m_inv[v1] * ld * g1
                self.dx[v2] += self.fixed[v2] * self.m_inv[v2] * ld * g2
                self.dx[v3] += self.fixed[v3] * self.m_inv[v3] * ld * g3
                self.nc[v0] += 1
                self.nc[v1] += 1
                self.nc[v2] += 1
                self.nc[v3] += 1

                if self.vt_active_set_num_dynamic[vid] < self.cache_size:
                    self.vt_active_set_dynamic[vid, self.vt_active_set_num_dynamic[vid]] = fid
                    self.vt_active_set_schur_dynamic[vid, self.vt_active_set_num_dynamic[vid]] = schur
                    self.vt_active_set_g_dynamic[vid, self.vt_active_set_num_dynamic[vid], 0] = g0
                    self.vt_active_set_g_dynamic[vid, self.vt_active_set_num_dynamic[vid], 1] = g1
                    self.vt_active_set_g_dynamic[vid, self.vt_active_set_num_dynamic[vid], 2] = g2
                    self.vt_active_set_g_dynamic[vid, self.vt_active_set_num_dynamic[vid], 3] = g3
                    ti.atomic_add(self.vt_active_set_num_dynamic[vid], 1)


    @ti.func
    def solve_collision_vt_static_x(self, vid_d, fid_s, dHat):

        v0 = vid_d
        v1 = self.face_indices_static[3 * fid_s + 0]
        v2 = self.face_indices_static[3 * fid_s + 1]
        v3 = self.face_indices_static[3 * fid_s + 2]

        x0 = self.y[v0]
        x1 = self.x_static[v1]
        x2 = self.x_static[v2]
        x3 = self.x_static[v3]

        g0 = ti.math.vec3(0.0)
        d = self.dHat[0]
        dtype = di.d_type_PT(x0, x1, x2, x3)

        if dtype == 0:
            d = di.d_PP(x0, x1)
            g0, g1 = di.g_PP(x0, x1)

        elif dtype == 1:
            d = di.d_PP(x0, x2)
            g0, g2 = di.g_PP(x0, x2)

        elif dtype == 2:
            d = di.d_PP(x0, x3)
            g0, g3 = di.g_PP(x0, x3)

        elif dtype == 3:
            d = di.d_PE(x0, x1, x2)
            g0, g1, g2 = di.g_PE(x0, x1, x2)

        elif dtype == 4:
            d = di.d_PE(x0, x2, x3)
            g0, g2, g3 = di.g_PE(x0, x2, x3)

        elif dtype == 5:
            d = di.d_PE(x0, x1, x3)
            g0, g1, g3 = di.g_PE(x0, x1, x3)

        elif dtype == 6:
            d = di.d_PT(x0, x1, x2, x3)
            g0, g1, g2, g3 = di.g_PT(x0, x1, x2, x3)

        if d < dHat:
            if self.vt_active_set_num[vid_d] < self.cache_size:
                self.vt_active_set[vid_d, self.vt_active_set_num[vid_d]] = ti.math.ivec2(fid_s, dtype)
                self.vt_active_set_g0[vid_d, self.vt_active_set_num[vid_d]] = g0
                schur = self.m_inv[v0] * g0.dot(g0) + 1e-4
                self.vt_active_set_schur[vid_d, self.vt_active_set_num[vid_d]] = schur
                ld = (dHat - d) / schur
                self.dx[v0] += self.m_inv[v0] * ld * g0
                self.nc[v0] += 1
                ti.atomic_add(self.vt_active_set_num[vid_d], 1)

    @ti.func
    def solve_collision_tv_static_x(self, fid_d, vid_s, dHat):

        v0 = vid_s
        v1 = self.face_indices_dynamic[3 * fid_d + 0]
        v2 = self.face_indices_dynamic[3 * fid_d + 1]
        v3 = self.face_indices_dynamic[3 * fid_d + 2]

        x0 = self.x_static[v0]
        x1 = self.y[v1]
        x2 = self.y[v2]
        x3 = self.y[v3]

        dtype = di.d_type_PT(x0, x1, x2, x3)

        if dtype == 0:
            d = di.d_PP(x0, x1)
            if d < dHat:
                g0, g1 = di.g_PP(x0, x1)
                schur = self.fixed[v1] * self.m_inv[v1] * g1.dot(g1) + 1e-4
                ld = (dHat - d) / schur
                self.dx[v1] += self.fixed[v1] * self.m_inv[v1] * ld * g1
                self.nc[v1] += 1
                if self.tv_active_set_num[fid_d] < self.cache_size:
                    self.tv_active_set_schur[fid_d, self.tv_active_set_num[fid_d]] = schur
                    self.tv_active_set_g[fid_d, self.tv_active_set_num[fid_d], 0] = g1
                    ti.atomic_add(self.tv_active_set_num[fid_d], 1)

        elif dtype == 1:
            d = di.d_PP(x0, x2)
            if d < dHat:
                g0, g2 = di.g_PP(x0, x2)
                schur = self.fixed[v2] * self.m_inv[v2] * g2.dot(g2) + 1e-4
                ld = (dHat - d) / schur
                self.dx[v2] += self.fixed[v2] * self.m_inv[v2] * ld * g2
                self.nc[v2] += 1
                if self.tv_active_set_num[fid_d] < self.cache_size:
                    self.tv_active_set_schur[fid_d, self.tv_active_set_num[fid_d]] = schur
                    self.tv_active_set_g[fid_d, self.tv_active_set_num[fid_d], 1] = g2
                    ti.atomic_add(self.tv_active_set_num[fid_d], 1)

        elif dtype == 2:
            d = di.d_PP(x0, x3)
            if d < dHat:
                g0, g3 = di.g_PP(x0, x3)
                schur = self.fixed[v3] * self.m_inv[v3] * g3.dot(g3) + 1e-4
                ld = (dHat - d) / schur
                self.dx[v3] += self.fixed[v3] * self.m_inv[v3] * ld * g3
                self.nc[v3] += 1
                if self.tv_active_set_num[fid_d] < self.cache_size:
                    self.tv_active_set_schur[fid_d, self.tv_active_set_num[fid_d]] = schur
                    self.tv_active_set_g[fid_d, self.tv_active_set_num[fid_d], 2] = g3
                    ti.atomic_add(self.tv_active_set_num[fid_d], 1)

        elif dtype == 3:
            d = di.d_PE(x0, x1, x2)
            if d < dHat:
                g0, g1, g2 = di.g_PE(x0, x1, x2)
                schur = self.fixed[v1] * self.m_inv[v1] * g1.dot(g1) + self.fixed[v2] * self.m_inv[v2] * g2.dot(g2) + 1e-4
                ld = (dHat - d) / schur
                self.dx[v1] += self.fixed[v1] * self.m_inv[v1] * ld * g1
                self.dx[v2] += self.fixed[v2] * self.m_inv[v2] * ld * g2
                self.nc[v1] += 1
                self.nc[v2] += 1
                if self.tv_active_set_num[fid_d] < self.cache_size:
                    self.tv_active_set_schur[fid_d, self.tv_active_set_num[fid_d]] = schur
                    self.tv_active_set_g[fid_d, self.tv_active_set_num[fid_d], 0] = g1
                    self.tv_active_set_g[fid_d, self.tv_active_set_num[fid_d], 1] = g2
                    ti.atomic_add(self.tv_active_set_num[fid_d], 1)


        elif dtype == 4:
            d = di.d_PE(x0, x2, x3)
            if d < dHat:
                g0, g2, g3 = di.g_PE(x0, x2, x3)
                schur = self.fixed[v2] * self.m_inv[v2] * g2.dot(g2) + self.fixed[v3] * self.m_inv[v3] * g3.dot(g3) + 1e-4
                ld = (dHat - d) / schur
                self.dx[v2] += self.fixed[v2] * self.m_inv[v2] * ld * g2
                self.dx[v3] += self.fixed[v3] * self.m_inv[v3] * ld * g3

                self.nc[v2] += 1
                self.nc[v3] += 1
                if self.tv_active_set_num[fid_d] < self.cache_size:
                    self.tv_active_set_schur[fid_d, self.tv_active_set_num[fid_d]] = schur
                    self.tv_active_set_g[fid_d, self.tv_active_set_num[fid_d], 1] = g2
                    self.tv_active_set_g[fid_d, self.tv_active_set_num[fid_d], 2] = g3
                    ti.atomic_add(self.tv_active_set_num[fid_d], 1)


        elif dtype == 5:
            d = di.d_PE(x0, x1, x3)
            if d < dHat:
                g0, g1, g3 = di.g_PE(x0, x1, x3)
                schur = self.fixed[v1] * self.m_inv[v1] * g1.dot(g1) + self.fixed[v3] * self.m_inv[v3] * g3.dot(g3) + 1e-4
                ld = (dHat - d) / schur
                self.dx[v1] += self.fixed[v1] * self.m_inv[v1] * ld * g1
                self.dx[v3] += self.fixed[v3] * self.m_inv[v3] * ld * g3
                self.nc[v1] += 1
                self.nc[v3] += 1
                if self.tv_active_set_num[fid_d] < self.cache_size:
                    self.tv_active_set_schur[fid_d, self.tv_active_set_num[fid_d]] = schur
                    self.tv_active_set_g[fid_d, self.tv_active_set_num[fid_d], 0] = g1
                    self.tv_active_set_g[fid_d, self.tv_active_set_num[fid_d], 2] = g3
                    ti.atomic_add(self.tv_active_set_num[fid_d], 1)

        elif dtype == 6:
            d = di.d_PT(x0, x1, x2, x3)
            if d < dHat:
                g0, g1, g2, g3 = di.g_PT(x0, x1, x2, x3)
                schur = self.fixed[v1] * self.m_inv[v1] * g1.dot(g1) + self.fixed[v2] * self.m_inv[v2] * g2.dot(g2) + self.fixed[v3] * self.m_inv[v3] * g3.dot(g3) + 1e-4
                ld = (dHat - d) / schur
                self.dx[v1] += self.fixed[v1] * self.m_inv[v1] * ld * g1
                self.dx[v2] += self.fixed[v2] * self.m_inv[v2] * ld * g2
                self.dx[v3] += self.fixed[v3] * self.m_inv[v3] * ld * g3
                self.nc[v1] += 1
                self.nc[v2] += 1
                self.nc[v3] += 1
                if self.tv_active_set_num[fid_d] < self.cache_size:
                    self.tv_active_set_schur[fid_d, self.tv_active_set_num[fid_d]] = schur
                    self.tv_active_set_g[fid_d, self.tv_active_set_num[fid_d], 0] = g1
                    self.tv_active_set_g[fid_d, self.tv_active_set_num[fid_d], 1] = g2
                    self.tv_active_set_g[fid_d, self.tv_active_set_num[fid_d], 2] = g3
                    ti.atomic_add(self.tv_active_set_num[fid_d], 1)

    @ti.func
    def solve_collision_vt_static_v(self, vid_d, g0, schur, friction_coeff):

        v0 = vid_d
        dvn = g0.dot(self.v[v0])
        if dvn < 0.0:
            ld = dvn / schur
            dv_nor = self.m_inv[v0] * ld * g0

            self.dv[v0] -= dv_nor
            self.nc[v0] += 1
            v_tan = self.v[v0] - ld * g0
            if v_tan.norm() < friction_coeff * abs(dvn):
                self.dv[v0] -= v_tan
            else:
                self.dv[v0] -= friction_coeff * v_tan



    @ti.func
    def solve_collision_vt_dynamic_v(self, vid_d, fid_d, g0, g1, g2, g3, schur, friction_coeff):

        v0 = vid_d
        v1 = self.face_indices_dynamic[3 * fid_d + 0]
        v2 = self.face_indices_dynamic[3 * fid_d + 1]
        v3 = self.face_indices_dynamic[3 * fid_d + 2]

        dvn = g0.dot(self.v[v0]) + g1.dot(self.v[v1]) + g2.dot(self.v[v2]) + g3.dot(self.v[v3])
        if dvn < 0.0:
            ld = dvn / schur
            if g0.norm() > 0.0:
                self.dv[v0] -= self.m_inv[v0] * ld * g0
                self.nc[v0] += 1

            if g1.norm() > 0.0:
                self.dv[v1] -= self.m_inv[v1] * ld * g1
                self.nc[v1] += 1

            if g2.norm() > 0.0:
                self.dv[v2] -= self.m_inv[v2] * ld * g2
                self.nc[v2] += 1

            if g3.norm() > 0.0:
                self.dv[v3] -= self.m_inv[v3] * ld * g3
                self.nc[v3] += 1

    @ti.func
    def solve_collision_tv_static_v(self, fid_d, g1, g2, g3, schur, friction_coeff):

        v1 = self.face_indices_dynamic[3 * fid_d + 0]
        v2 = self.face_indices_dynamic[3 * fid_d + 1]
        v3 = self.face_indices_dynamic[3 * fid_d + 2]

        dvn = g1.dot(self.v[v1]) + g2.dot(self.v[v2]) + g3.dot(self.v[v3])

        if dvn < 0.0:
            ld = dvn / schur

            if g1.norm() > 0.0:
                self.dv[v1] -= self.m_inv[v1] * ld * g1
                self.nc[v1] += 1
                v_tan = self.v[v1] - self.m_inv[v1] * ld * g1
                if v_tan.norm() < friction_coeff * abs(dvn):
                    self.dv[v1] -= v_tan
                else:
                    self.dv[v1] -= friction_coeff * v_tan

            if g2.norm() > 0.0:
                self.dv[v2] -= self.m_inv[v2] * ld * g2
                self.nc[v2] += 1
                v_tan = self.v[v1] - self.m_inv[v1] * ld * g1
                if v_tan.norm() < friction_coeff * abs(dvn):
                    self.dv[v2] -= v_tan
                else:
                    self.dv[v2] -= friction_coeff * v_tan

            if g3.norm() > 0.0:
                self.dv[v3] -= self.m_inv[v3] * ld * g3
                self.nc[v3] += 1
                v_tan = self.v[v1] - self.m_inv[v1] * ld * g1
                if v_tan.norm() < friction_coeff * abs(dvn):
                    self.dv[v3] -= v_tan
                else:
                    self.dv[v3] -= friction_coeff * v_tan

    @ti.func
    def solve_collision_ee_static_x(self, eid_d, eid_s, dHat):

        v0 = self.edge_indices_dynamic[2 * eid_d + 0]
        v1 = self.edge_indices_dynamic[2 * eid_d + 1]

        v2 = self.edge_indices_static[2 * eid_s + 0]
        v3 = self.edge_indices_static[2 * eid_s + 1]


        x0, x1 = self.y[v0], self.y[v1]
        x2, x3 = self.x_static[v2], self.x_static[v3]

        dtype = di.d_type_EE(x0, x1, x2, x3)

        if dtype == 0:
            d = di.d_PP(x0, x2)
            if d < dHat:
                g0, g2 = di.g_PP(x0, x2)
                schur = self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) + 1e-4
                ld = (dHat - d) / schur
                self.dx[v0] += self.fixed[v0] * self.m_inv[v0] * ld * g0
                self.nc[v0] += 1

                if self.ee_active_set_num[eid_d] < self.cache_size:
                    self.ee_active_set_schur[eid_d, self.ee_active_set_num[eid_d]] = schur
                    self.ee_active_set_g[eid_d, self.ee_active_set_num[eid_d], 0] = g0
                    ti.atomic_add(self.ee_active_set_num[eid_d], 1)


        elif dtype == 1:
            d = di.d_PP(x0, x3)
            if d < dHat:
                g0, g3 = di.g_PP(x0, x3)
                schur = self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) + 1e-4
                ld = (dHat - d) / schur
                self.dx[v0] += self.fixed[v0] * self.m_inv[v0] * ld * g0
                self.nc[v0] += 1

                if self.ee_active_set_num[eid_d] < self.cache_size:
                    self.ee_active_set_schur[eid_d, self.ee_active_set_num[eid_d]] = schur
                    self.ee_active_set_g[eid_d, self.ee_active_set_num[eid_d], 0] = g0
                    ti.atomic_add(self.ee_active_set_num[eid_d], 1)



        elif dtype == 2:
            d = di.d_PE(x0, x2, x3)
            if d < dHat:
                g0, g2, g3 = di.g_PE(x0, x2, x3)
                schur = self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) + 1e-4
                ld = (dHat - d) / schur
                self.dx[v0] += self.fixed[v0] * self.m_inv[v0] * ld * g0
                self.nc[v0] += 1

                if self.ee_active_set_num[eid_d] < self.cache_size:
                    self.ee_active_set_schur[eid_d, self.ee_active_set_num[eid_d]] = schur
                    self.ee_active_set_g[eid_d, self.ee_active_set_num[eid_d], 0] = g0
                    ti.atomic_add(self.ee_active_set_num[eid_d], 1)

        elif dtype == 3:
            d = di.d_PP(x1, x2)
            if d < dHat:
                g1, g2 = di.g_PP(x1, x2)
                schur = self.fixed[v1] * self.m_inv[v1] * g1.dot(g1) + 1e-4
                ld = (dHat - d) / schur

                self.dx[v1] += self.m_inv[v1] * ld * g1
                self.nc[v1] += 1

                if self.ee_active_set_num[eid_d] < self.cache_size:
                    self.ee_active_set_schur[eid_d, self.ee_active_set_num[eid_d]] = schur
                    self.ee_active_set_g[eid_d, self.ee_active_set_num[eid_d], 1] = g1
                    ti.atomic_add(self.ee_active_set_num[eid_d], 1)

        elif dtype == 4:
            d = di.d_PP(x1, x3)
            if d < dHat:
                g1, g3 = di.g_PP(x1, x3)
                schur = self.m_inv[v1] * g1.dot(g1) + 1e-4
                ld = (dHat - d) / schur

                self.dx[v1] += self.m_inv[v0] * ld * g1
                self.nc[v1] += 1

                if self.ee_active_set_num[eid_d] < self.cache_size:
                    self.ee_active_set_schur[eid_d, self.ee_active_set_num[eid_d]] = schur
                    self.ee_active_set_g[eid_d, self.ee_active_set_num[eid_d], 1] = g1
                    ti.atomic_add(self.ee_active_set_num[eid_d], 1)

        elif dtype == 5:
            d = di.d_PE(x1, x2, x3)
            if d < dHat:
                g1, g2, g3 = di.g_PE(x1, x2, x3)
                schur = self.m_inv[v1] * g1.dot(g1) + 1e-4
                ld = (dHat - d) / schur
                self.dx[v1] += self.m_inv[v0] * ld * g1
                self.nc[v1] += 1
                if self.ee_active_set_num[eid_d] < self.cache_size:
                    self.ee_active_set_schur[eid_d, self.ee_active_set_num[eid_d]] = schur
                    self.ee_active_set_g[eid_d, self.ee_active_set_num[eid_d], 1] = g1
                    ti.atomic_add(self.ee_active_set_num[eid_d], 1)

        elif dtype == 6:
            d = di.d_PE(x2, x0, x1)
            if d < dHat:
                g2, g0, g1 = di.g_PE(x2, x0, x1)
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v1] * g1.dot(g1) + 1e-4
                ld = (dHat - d) / schur

                self.dx[v0] += self.m_inv[v0] * ld * g0
                self.dx[v1] += self.m_inv[v1] * ld * g1
                self.nc[v0] += 1
                self.nc[v1] += 1
                if self.ee_active_set_num[eid_d] < self.cache_size:
                    self.ee_active_set_schur[eid_d, self.ee_active_set_num[eid_d]] = schur
                    self.ee_active_set_g[eid_d, self.ee_active_set_num[eid_d], 0] = g0
                    self.ee_active_set_g[eid_d, self.ee_active_set_num[eid_d], 1] = g1
                    ti.atomic_add(self.ee_active_set_num[eid_d], 1)

        elif dtype == 7:
            d = di.d_PE(x3, x0, x1)
            if d < dHat:
                g3, g0, g1 = di.g_PE(x3, x0, x1)
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v1] * g1.dot(g1) + 1e-4
                ld = (dHat - d) / schur

                self.dx[v0] += self.m_inv[v0] * ld * g0
                self.dx[v1] += self.m_inv[v1] * ld * g1
                self.nc[v0] += 1
                self.nc[v1] += 1
                if self.ee_active_set_num[eid_d] < self.cache_size:
                    self.ee_active_set_schur[eid_d, self.ee_active_set_num[eid_d]] = schur
                    self.ee_active_set_g[eid_d, self.ee_active_set_num[eid_d], 0] = g0
                    self.ee_active_set_g[eid_d, self.ee_active_set_num[eid_d], 1] = g1
                    ti.atomic_add(self.ee_active_set_num[eid_d], 1)

        elif dtype == 8:
            d = di.d_EE(x0, x1, x2, x3)
            if d < dHat:
                g0, g1, g2, g3 = di.g_EE(x0, x1, x2, x3)
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v1] * g1.dot(g1) + 1e-4
                ld = (dHat - d) / schur
                self.dx[v0] += self.m_inv[v0] * ld * g0
                self.dx[v1] += self.m_inv[v1] * ld * g1
                self.nc[v0] += 1
                self.nc[v1] += 1
                if self.ee_active_set_num[eid_d] < self.cache_size:
                    self.ee_active_set_schur[eid_d, self.ee_active_set_num[eid_d]] = schur
                    self.ee_active_set_g[eid_d, self.ee_active_set_num[eid_d], 0] = g0
                    self.ee_active_set_g[eid_d, self.ee_active_set_num[eid_d], 1] = g1
                    ti.atomic_add(self.ee_active_set_num[eid_d], 1)


    @ti.func
    def solve_collision_ee_static_v(self, eid_d, g0, g1, schur, friction_coeff):

        v0 = self.edge_indices_dynamic[2 * eid_d + 0]
        v1 = self.edge_indices_dynamic[2 * eid_d + 1]

        dvn = g0.dot(self.v[v0]) + g1.dot(self.v[v1])
        if dvn < 0.0:
            ld = dvn / schur
            if g0.norm() > 0.0:
                self.dv[v0] -= self.fixed[v0] * self.m_inv[v0] * ld * g0
                self.nc[v0] += 1
            if g1.norm() > 0.0:
                self.dv[v1] -= self.fixed[v1] * self.m_inv[v1] * ld * g1
                self.nc[v1] += 1


    @ti.func
    def solve_collision_ee_dynamic_x(self, ei0, ei1, dHat):

        v0 = self.edge_indices_dynamic[2 * ei0 + 0]
        v1 = self.edge_indices_dynamic[2 * ei0 + 1]

        v2 = self.edge_indices_dynamic[2 * ei1 + 0]
        v3 = self.edge_indices_dynamic[2 * ei1 + 1]

        x0 = self.y[v0]
        x1 = self.y[v1]

        x2 = self.y[v2]
        x3 = self.y[v3]

        dtype = di.d_type_EE(x0, x1, x2, x3)

        if dtype == 0:
            d = di.d_PP(x0, x2)
            if d < dHat:
                g0, g2 = di.g_PP(x0, x2)
                schur = self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) + self.fixed[v2] * self.m_inv[v2] * g2.dot(g2) + 1e-4
                ld = (dHat - d) / schur
                self.dx[v0] += self.fixed[v0] * self.m_inv[v0] * ld * g0
                self.dx[v2] += self.fixed[v2] * self.m_inv[v2] * ld * g2
                self.nc[v0] += 1
                self.nc[v2] += 1
                if self.ee_active_set_num_dynamic[0] < self.cache_size:
                    self.ee_active_set_dynamic[self.ee_active_set_num[0]] = ti.math.ivec2(ei0, ei1)
                    self.ee_active_set_schur_dynamic[self.ee_active_set_num[0]] = schur
                    self.ee_active_set_g_dynamic[self.ee_active_set_num[0], 0] = g0
                    self.ee_active_set_g_dynamic[self.ee_active_set_num[0], 2] = g2
                    ti.atomic_add(self.ee_active_set_num[0], 1)

        elif dtype == 1:
            d = di.d_PP(x0, x3)
            if d < dHat:
                g0, g3 = di.g_PP(x0, x3)
                schur = self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) + self.fixed[v3] * self.m_inv[v3] * g3.dot(g3) + 1e-4
                ld = (dHat - d) / schur

                self.dx[v0] += self.fixed[v0] * self.m_inv[v0] * ld * g0
                self.dx[v3] += self.fixed[v3] * self.m_inv[v3] * ld * g3
                self.nc[v0] += 1
                self.nc[v3] += 1
                if self.ee_active_set_num_dynamic[0] < self.cache_size:
                    self.ee_active_set_dynamic[self.ee_active_set_num[0]] = ti.math.ivec2(ei0, ei1)
                    self.ee_active_set_schur_dynamic[self.ee_active_set_num[0]] = schur
                    self.ee_active_set_g_dynamic[self.ee_active_set_num[0], 0] = g0
                    self.ee_active_set_g_dynamic[self.ee_active_set_num[0], 3] = g3
                    ti.atomic_add(self.ee_active_set_num[0], 1)



        elif dtype == 2:
            d = di.d_PE(x0, x2, x3)
            if d < dHat:
                g0, g2, g3 = di.g_PE(x0, x2, x3)
                schur = self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) + self.fixed[v2] * self.m_inv[v2] * g2.dot(g2) + self.fixed[v3] * self.m_inv[v3] * g3.dot(g3) + 1e-4
                ld = (dHat - d) / schur
                self.dx[v0] += self.fixed[v0] * self.m_inv[v0] * ld * g0
                self.dx[v2] += self.fixed[v2] * self.m_inv[v2] * ld * g2
                self.dx[v3] += self.fixed[v3] * self.m_inv[v3] * ld * g3
                self.nc[v0] += 1
                self.nc[v2] += 1
                self.nc[v3] += 1
                if self.ee_active_set_num_dynamic[0] < self.cache_size:
                    self.ee_active_set_dynamic[self.ee_active_set_num[0]] = ti.math.ivec2(ei0, ei1)
                    self.ee_active_set_schur_dynamic[self.ee_active_set_num[0]] = schur
                    self.ee_active_set_g_dynamic[self.ee_active_set_num[0], 0] = g0
                    self.ee_active_set_g_dynamic[self.ee_active_set_num[0], 2] = g2
                    self.ee_active_set_g_dynamic[self.ee_active_set_num[0], 3] = g3
                    ti.atomic_add(self.ee_active_set_num[0], 1)


        elif dtype == 3:
            d = di.d_PP(x1, x2)
            if d < dHat:
                g1, g2 = di.g_PP(x1, x2)
                schur = self.fixed[v1] * self.m_inv[v1] * g1.dot(g1) + self.fixed[v2] * self.m_inv[v2] * g2.dot(g2) + 1e-4
                ld = (dHat - d) / schur
                self.dx[v1] += self.fixed[v1] * self.m_inv[v1] * ld * g1
                self.dx[v2] += self.fixed[v2] * self.m_inv[v2] * ld * g2
                self.nc[v1] += 1
                self.nc[v2] += 1
                if self.ee_active_set_num_dynamic[0] < self.cache_size:
                    self.ee_active_set_dynamic[self.ee_active_set_num[0]] = ti.math.ivec2(ei0, ei1)
                    self.ee_active_set_schur_dynamic[self.ee_active_set_num[0]] = schur
                    self.ee_active_set_g_dynamic[self.ee_active_set_num[0], 1] = g1
                    self.ee_active_set_g_dynamic[self.ee_active_set_num[0], 2] = g2
                    ti.atomic_add(self.ee_active_set_num[0], 1)

        elif dtype == 4:
            d = di.d_PP(x1, x3)
            if d < dHat:
                g1, g3 = di.g_PP(x1, x3)
                schur = self.fixed[v1] * self.m_inv[v1] * g1.dot(g1) + self.fixed[v3] * self.m_inv[v3] * g3.dot(g3) + 1e-4
                ld = (dHat - d) / schur
                self.dx[v1] += self.fixed[v1] * self.m_inv[v1] * ld * g1
                self.dx[v3] += self.fixed[v3] * self.m_inv[v3] * ld * g3
                self.nc[v1] += 1
                self.nc[v3] += 1
                if self.ee_active_set_num_dynamic[0] < self.cache_size:
                    self.ee_active_set_dynamic[self.ee_active_set_num[0]] = ti.math.ivec2(ei0, ei1)
                    self.ee_active_set_schur_dynamic[self.ee_active_set_num[0]] = schur
                    self.ee_active_set_g_dynamic[self.ee_active_set_num[0], 1] = g1
                    self.ee_active_set_g_dynamic[self.ee_active_set_num[0], 3] = g3
                    ti.atomic_add(self.ee_active_set_num[0], 1)

        elif dtype == 5:
            d = di.d_PE(x1, x2, x3)
            if d < dHat:
                g1, g2, g3 = di.g_PE(x1, x2, x3)
                schur = self.fixed[v1] * self.m_inv[v1] * g1.dot(g1) + self.fixed[v2] * self.m_inv[v2] * g2.dot(g2) + self.fixed[v3] * self.m_inv[v3] * g3.dot(g3) + 1e-4
                ld = (dHat - d) / schur

                self.dx[v1] += self.fixed[v1] * self.m_inv[v1] * ld * g1
                self.dx[v2] += self.fixed[v2] * self.m_inv[v2] * ld * g2
                self.dx[v3] += self.fixed[v3] * self.m_inv[v3] * ld * g3
                self.nc[v1] += 1
                self.nc[v2] += 1
                self.nc[v3] += 1

                if self.ee_active_set_num_dynamic[0] < self.cache_size:
                    self.ee_active_set_dynamic[self.ee_active_set_num[0]] = ti.math.ivec2(ei0, ei1)
                    self.ee_active_set_schur_dynamic[self.ee_active_set_num[0]] = schur
                    self.ee_active_set_g_dynamic[self.ee_active_set_num[0], 1] = g1
                    self.ee_active_set_g_dynamic[self.ee_active_set_num[0], 2] = g2
                    self.ee_active_set_g_dynamic[self.ee_active_set_num[0], 3] = g3
                    ti.atomic_add(self.ee_active_set_num[0], 1)

        elif dtype == 6:
            d = di.d_PE(x2, x0, x1)
            if d < dHat:
                g2, g0, g1 = di.g_PE(x2, x0, x1)
                schur = self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) + self.fixed[v1] * self.m_inv[v1] * g1.dot(g1) + self.fixed[v2] * self.m_inv[v2] * g2.dot(g2) + 1e-4
                ld = (dHat - d) / schur

                self.dx[v0] += self.fixed[v0] * self.m_inv[v0] * ld * g0
                self.dx[v1] += self.fixed[v1] * self.m_inv[v1] * ld * g1
                self.dx[v2] += self.fixed[v2] * self.m_inv[v2] * ld * g2

                self.nc[v0] += 1
                self.nc[v1] += 1
                self.nc[v2] += 1
                if self.ee_active_set_num_dynamic[0] < self.cache_size:
                    self.ee_active_set_dynamic[self.ee_active_set_num[0]] = ti.math.ivec2(ei0, ei1)
                    self.ee_active_set_schur_dynamic[self.ee_active_set_num[0]] = schur
                    self.ee_active_set_g_dynamic[self.ee_active_set_num[0], 0] = g0
                    self.ee_active_set_g_dynamic[self.ee_active_set_num[0], 1] = g1
                    self.ee_active_set_g_dynamic[self.ee_active_set_num[0], 2] = g2

                    ti.atomic_add(self.ee_active_set_num[0], 1)

        elif dtype == 7:
            d = di.d_PE(x3, x0, x1)
            if d < dHat:
                g3, g0, g1 = di.g_PE(x3, x0, x1)
                schur = self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) + self.fixed[v1] * self.m_inv[v1] * g1.dot(g1) + self.fixed[v3] * self.m_inv[v3] * g3.dot(g3) + 1e-4
                ld = (dHat - d) / schur

                self.dx[v0] += self.fixed[v0] * self.m_inv[v0] * ld * g0
                self.dx[v1] += self.fixed[v1] * self.m_inv[v1] * ld * g1
                self.dx[v3] += self.fixed[v3] * self.m_inv[v3] * ld * g3
                self.nc[v0] += 1
                self.nc[v1] += 1
                self.nc[v3] += 1
                if self.ee_active_set_num_dynamic[0] < self.cache_size:
                    self.ee_active_set_dynamic[self.ee_active_set_num[0]] = ti.math.ivec2(ei0, ei1)
                    self.ee_active_set_schur_dynamic[self.ee_active_set_num[0]] = schur
                    self.ee_active_set_g_dynamic[self.ee_active_set_num[0], 0] = g0
                    self.ee_active_set_g_dynamic[self.ee_active_set_num[0], 1] = g1
                    self.ee_active_set_g_dynamic[self.ee_active_set_num[0], 3] = g3

                    ti.atomic_add(self.ee_active_set_num[0], 1)

        elif dtype == 8:
            d = di.d_EE(x0, x1, x2, x3)
            if d < dHat:
                g0, g1, g2, g3 = di.g_EE(x0, x1, x2, x3)
                schur = self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) + self.fixed[v1] * self.m_inv[v1] * g1.dot(g1) + self.fixed[v2] * self.m_inv[v2] * g2.dot(g2) + self.fixed[v3] * self.m_inv[v3] * g3.dot(g3) + 1e-4
                ld = (dHat - d) / schur

                self.dx[v0] += self.fixed[v0] * self.m_inv[v0] * ld * g0
                self.dx[v1] += self.fixed[v1] * self.m_inv[v1] * ld * g1
                self.dx[v2] += self.fixed[v2] * self.m_inv[v2] * ld * g2
                self.dx[v3] += self.fixed[v3] * self.m_inv[v3] * ld * g3
                self.nc[v0] += 1
                self.nc[v1] += 1
                self.nc[v2] += 1
                self.nc[v3] += 1

                if self.ee_active_set_num_dynamic[0] < self.cache_size:
                    self.ee_active_set_dynamic[self.ee_active_set_num[0]] = ti.math.ivec2(ei0, ei1)
                    self.ee_active_set_schur_dynamic[self.ee_active_set_num[0]] = schur
                    self.ee_active_set_g_dynamic[self.ee_active_set_num[0], 0] = g0
                    self.ee_active_set_g_dynamic[self.ee_active_set_num[0], 1] = g1
                    self.ee_active_set_g_dynamic[self.ee_active_set_num[0], 2] = g2
                    self.ee_active_set_g_dynamic[self.ee_active_set_num[0], 3] = g3
                    ti.atomic_add(self.ee_active_set_num[0], 1)

    @ti.func
    def solve_collision_ee_dynamic_v(self, ei0, ei1, g0, g1, g2, g3, schur, friction_coeff):

        v0 = self.edge_indices_dynamic[2 * ei0 + 0]
        v1 = self.edge_indices_dynamic[2 * ei0 + 1]

        v2 = self.edge_indices_dynamic[2 * ei1 + 0]
        v3 = self.edge_indices_dynamic[2 * ei1 + 1]

        dvn = g0.dot(self.v[v0]) + g1.dot(self.v[v1]) + g2.dot(self.v[v2]) + g3.dot(self.v[v3])
        if dvn < 0.0:
            ld = dvn / schur

            if g0.norm() > 0:
                self.dx[v0] += self.fixed[v0] * self.m_inv[v0] * ld * g0
                self.nc[v0] += 1

            if g1.norm() > 1:
                self.dx[v1] += self.fixed[v1] * self.m_inv[v1] * ld * g1
                self.nc[v1] += 1

            if g2.norm() > 2:
                self.dx[v2] += self.fixed[v2] * self.m_inv[v2] * ld * g2
                self.nc[v2] += 1

            if g3.norm() > 3:
                self.dx[v3] += self.fixed[v3] * self.m_inv[v3] * ld * g3
                self.nc[v3] += 1


    @ti.kernel
    def solve_spring_constraints_x(self):
        for ei in range(self.offset_spring):
            v0, v1 = self.edge_indices_dynamic[2 * ei + 0], self.edge_indices_dynamic[2 * ei + 1]
            x0, x1 = self.y[v0], self.y[v1]
            l0 = self.l0[ei]
            x10 = x0 - x1
            lij = x10.norm()

            C = 0.5 * (lij - l0) * (lij - l0)
            nabla_C = (lij - l0) * (x10 / lij)
            schur = (self.fixed[v0] * self.m_inv[v0] + self.fixed[v1] * self.m_inv[v1]) * nabla_C.dot(nabla_C) + 1e-4
            ld = C / schur

            self.dx[v0] -= self.fixed[v0] * self.m_inv[v0] * ld * nabla_C
            self.dx[v1] += self.fixed[v1] * self.m_inv[v1] * ld * nabla_C
            self.nc[v0] += 1
            self.nc[v1] += 1

            # if lij > 1.05 * l0:
            #     self.spring_ids[self.num_springs[0]] = ei
            #     self.schur_spring[self.num_springs[0]] = schur
            #     self.gradient_spring[self.num_springs[0]] = nabla_C
            #     ti.atomic_add(self.num_springs[0], 1)





    @ti.kernel
    def solve_spring_constraints_v(self):

        for i in range(self.num_springs[0]):
            ei = self.spring_ids[i]
            v0, v1 = self.edge_indices_dynamic[2 * ei + 0], self.edge_indices_dynamic[2 * ei + 1]

            Cv = self.gradient_spring[i].dot(self.v[v0] - self.v[v1])
            if Cv > 0:
                ld_v = Cv / self.schur_spring[i]
                self.dv[v0] -= self.fixed[v0] * self.m_inv[v0] * ld_v * self.gradient_spring[i]
                self.dv[v1] += self.fixed[v1] * self.m_inv[v1] * ld_v * self.gradient_spring[i]
                self.nc[v0] += 1
                self.nc[v1] += 1

    @ti.func
    def spiky_gradient(self, r, h):
        result = ti.math.vec3(0.0)

        spiky_grad_factor = -45.0 / ti.math.pi
        r_len = r.norm()
        if 0 < r_len and r_len < h:
            x = (h - r_len) / (h * h * h)
            g_factor = spiky_grad_factor * x * x
            result = r * g_factor / r_len
        return result

    @ti.func
    def poly6_value(self, s, h):
        result = 0.0
        poly6_factor = 315.0 / 64.0 / ti.math.pi
        if 0 < s and s < h:
            x = (h * h - s * s) / (h * h * h)
            result = poly6_factor * x * x * x
        return result

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


    @ti.func
    def get_tri_bbox_idx(self,v1, v2, v3):

        # clamp 0, max grid num XYZ
        xmin = ti.math.min(v1[0], v2[0], v3[0]) - 1
        xmax = ti.math.max(v1[0], v2[0], v3[0]) + 1

        ymin = ti.math.min(v1[1], v2[1], v3[1]) - 1
        ymax = ti.math.max(v1[1], v2[1], v3[1]) + 1

        zmin = ti.math.min(v1[2], v2[2], v3[2]) - 1
        zmax = ti.math.max(v1[2], v2[2], v3[2]) + 1

        return ti.Vector([xmin, xmax, ymin, ymax, zmin, zmax])

    @ti.kernel
    def solve_collision_constraints_x(self):
        d = self.dHat[0]
        for i in range(self.max_num_faces_static + self.max_num_faces_dynamic):
            if i < self.max_num_faces_static:
                fi_s = i
                v0 = self.face_indices_static[3 * fi_s + 0]
                v1 = self.face_indices_static[3 * fi_s + 1]
                v2 = self.face_indices_static[3 * fi_s + 2]
                x0, x1, x2 = self.x_static[v0], self.x_static[v1], self.x_static[v2]
                idx0 = self.pos_to_index(x0)
                idx1 = self.pos_to_index(x1)
                idx2 = self.pos_to_index(x2)
                bbox = self.get_tri_bbox_idx(idx0, idx1, idx2)

                for ii in ti.ndrange(bbox[1] - bbox[0] + 1):
                    for jj in ti.ndrange(bbox[3] - bbox[2] + 1):
                        for kk in ti.ndrange(bbox[5] - bbox[4] + 1):
                            xi = bbox[0] + ii
                            yi = bbox[2] + jj
                            zi = bbox[4] + kk
                            grid_index = self.flatten_cell_id(ti.Vector([xi, yi, zi]))
                            for p_j in range(self.grid_particles_num[ti.max(0, grid_index - 1)], self.grid_particles_num[grid_index]):
                                vj_d = self.cur2org[p_j]
                                if vj_d < self.max_num_verts_dynamic:
                                    self.solve_collision_vt_static_x(vj_d, fi_s, d)

            else:
                fi_d = i - self.max_num_faces_static
                v0 = self.face_indices_dynamic[3 * fi_d + 0]
                v1 = self.face_indices_dynamic[3 * fi_d + 1]
                v2 = self.face_indices_dynamic[3 * fi_d + 2]
                x0, x1, x2 = self.y[v0], self.y[v1], self.y[v2]
                idx0 = self.pos_to_index(x0)
                idx1 = self.pos_to_index(x1)
                idx2 = self.pos_to_index(x2)
                bbox = self.get_tri_bbox_idx(idx0, idx1, idx2)

                for ii in ti.ndrange(bbox[1] - bbox[0] + 1):
                    for jj in ti.ndrange(bbox[3] - bbox[2] + 1):
                        for kk in ti.ndrange(bbox[5] - bbox[4] + 1):
                            xi = bbox[0] + ii
                            yi = bbox[2] + jj
                            zi = bbox[4] + kk
                            grid_index = self.flatten_cell_id(ti.Vector([xi, yi, zi]))
                            for p_j in range(self.grid_particles_num_static[ti.max(0, grid_index - 1)], self.grid_particles_num_static[grid_index]):
                                vj_s = self.cur2org_static[p_j]
                                if vj_s < self.max_num_verts_static:
                                    self.solve_collision_tv_static_x(fi_d, vj_s, d)

                            for p_j in range(self.grid_particles_num[ti.max(0, grid_index - 1)], self.grid_particles_num[grid_index]):
                                vj_d = self.cur2org[p_j]
                                if vj_d < self.max_num_verts_dynamic:
                                    self.solve_collision_vt_dynamic_x(vj_d, fi_d, d)




        # #-----------brute-force-------------------
        # for vi in range(self.max_num_verts_dynamic):
        #
        #     for ti in range(self.max_num_faces_dynamic):
        #         if self.is_in_face(vi, ti) != True:
        #             self.solve_collision_vt_dynamic_x(vi, ti, d)
        #
        #     for ti_s in range(self.max_num_faces_static):s
        #         self.solve_collision_vt_static_x(vi, ti_s, d)
        #
        #
        # for ti_s in range(self.max_num_faces_static):
        #     for vi in range(self.max_num_verts_dynamic):
        #         self.solve_collision_tv_static_x(ti_s, vi, d)

        #
        for ei in range(self.max_num_edges_dynamic):
            for ei_s in range(self.max_num_edges_static):
                self.solve_collision_ee_static_x(ei, ei_s, d)
        # #
            for ei_d in range(self.max_num_edges_dynamic):
                if ei != ei_d and self.share_vertex(ei, ei_d) != True:
                    self.solve_collision_ee_dynamic_x(ei, ei_d, d)


    @ti.kernel
    def solve_collision_constraints_v(self):

        friction_coeff = self.friction_coeff[0]
        # print(friction_coeff)
        for vi_d in range(self.max_num_verts_dynamic):
            for j in range(self.vt_active_set_num[vi_d]):
                g0, schur = self.vt_active_set_g0[vi_d, j], self.vt_active_set_schur[vi_d, j]
                self.solve_collision_vt_static_v(vi_d, g0, schur, friction_coeff)

            for j in range(self.vt_active_set_num_dynamic[vi_d]):
                fi_d = self.vt_active_set_dynamic[vi_d, j]
                schur = self.vt_active_set_schur_dynamic[vi_d, j]
                g0 = self.vt_active_set_g_dynamic[vi_d, j, 0]
                g1 = self.vt_active_set_g_dynamic[vi_d, j, 1]
                g2 = self.vt_active_set_g_dynamic[vi_d, j, 2]
                g3 = self.vt_active_set_g_dynamic[vi_d, j, 3]

                self.solve_collision_vt_dynamic_v(vi_d, fi_d, g0, g1, g2, g3, schur, friction_coeff)

        for fi_d in range(self.max_num_faces_dynamic):
            for j in range(self.tv_active_set_num[fi_d]):
                g1 = self.tv_active_set_g[fi_d, j, 0]
                g2 = self.tv_active_set_g[fi_d, j, 1]
                g3 = self.tv_active_set_g[fi_d, j, 2]
                schur = self.tv_active_set_schur[fi_d, j]
                self.solve_collision_tv_static_v(fi_d, g1, g2, g3, schur, friction_coeff)
        #
        # for i in range(self.ee_active_set_num_dynamic[0]):
        #     pair = self.ee_active_set_dynamic[i]
        #     g0 = self.ee_active_set_g_dynamic[i, 0]
        #     g1 = self.ee_active_set_g_dynamic[i, 1]
        #     g2 = self.ee_active_set_g_dynamic[i, 2]
        #     g3 = self.ee_active_set_g_dynamic[i, 3]
        #     schur = self.ee_active_set_schur_dynamic[i]
        #     self.solve_collision_ee_dynamic_v(pair[0], pair[1], g0, g1, g2, g3, schur, friction_coeff)

        # for fid_s in range(self.max_num_faces_static):
        #     for i in range(self.tv_active_set_num[fid_s]):
        #         vid_d = self.vt_active_set[fid_s, i]
        #         self.solve_collision_tv_static_v(fid_s, vid_d, d)
        #
        for eid_d in range(self.max_num_edges_dynamic):
            for i in range(self.ee_active_set_num[eid_d]):
                g0 = self.ee_active_set_g[eid_d, i, 0]
                g1 = self.ee_active_set_g[eid_d, i, 1]
                schur = self.ee_active_set_schur[eid_d, i]
                self.solve_collision_ee_static_v(eid_d, g0, g1, schur, friction_coeff)

        for i in range(self.ee_active_set_num_dynamic[0]):
            pair_i = self.ee_active_set_dynamic[i]
            schur = self.ee_active_set_schur_dynamic[i]

            g0 = self.ee_active_set_g_dynamic[i, 0]
            g1 = self.ee_active_set_g_dynamic[i, 1]
            g2 = self.ee_active_set_g_dynamic[i, 2]
            g3 = self.ee_active_set_g_dynamic[i, 3]

            self.solve_collision_ee_dynamic_v(pair_i[0], pair_i[1], g0, g1, g2, g3, schur, friction_coeff)

        # -----------brute-force-------------------
        # for vi in range(self.max_num_verts_dynamic):
        #
        #     # for ti in range(self.max_num_faces_dynamic):
        #     #     if self.is_in_face(vi, ti) != True:
        #     #         self.solve_collision_vt_dynamic_v(vi, ti, d)
        #
        #     for ti_s in range(self.max_num_faces_static):
        #         self.solve_collision_vt_static_v(vi, ti_s, d)
        #
        #
        # for ti_s in range(self.max_num_faces_static):
        #     for vi in range(self.max_num_verts_dynamic):
        #         self.solve_collision_tv_static_v(ti_s, vi, d)

        # for ei in range(self.max_num_edges_dynamic):
        #     for ei_s in range(self.max_num_edges_static):
        #         self.solve_collision_ee_static_v(ei, ei_s, d)
        #
        #     for ei_d in range(self.max_num_edges_dynamic):
        #         if ei != ei_d and self.share_vertex(ei, ei_d) != True:
        #             self.solve_collision_ee_dynamic_v(ei, ei_d, d)


    @ti.kernel
    def solve_fem_constraints_x(self):

        for tid in range(self.max_num_tetra_dynamic):

            v0 = self.tetra_indices_dynamic[4 * tid + 0]
            v1 = self.tetra_indices_dynamic[4 * tid + 1]
            v2 = self.tetra_indices_dynamic[4 * tid + 2]
            v3 = self.tetra_indices_dynamic[4 * tid + 3]

            x0, x1, x2, x3 = self.y[v0], self.y[v1], self.y[v2], self.y[v3]
            Ds = ti.Matrix.cols([x0 - x3, x1 - x3, x2 - x3])

            F = Ds @ self.Dm_inv[tid]
            U, sig, V = ti.svd(F)
            R = U @ V.transpose()

            H = (F - R) @ self.Dm_inv[tid].transpose()

            C = 0.5 * (F - R).norm() * (F - R).norm()

            nabla_C0 = ti.Vector([H[j, 0] for j in ti.static(range(3))])
            nabla_C1 = ti.Vector([H[j, 1] for j in ti.static(range(3))])
            nabla_C2 = ti.Vector([H[j, 2] for j in ti.static(range(3))])
            nabla_C3 = -(nabla_C0 + nabla_C1 + nabla_C2)

            schur = (self.fixed[v0] * self.m_inv[v0] * nabla_C0.dot(nabla_C0) +
                     self.fixed[v1] * self.m_inv[v1] * nabla_C1.dot(nabla_C1) +
                     self.fixed[v2] * self.m_inv[v2] * nabla_C2.dot(nabla_C2) +
                     self.fixed[v3] * self.m_inv[v3] * nabla_C3.dot(nabla_C3) + 1e-4)

            ld = C / schur

            self.dx[v0] -= self.fixed[v0] * self.m_inv[v0] * nabla_C0 * ld
            self.dx[v1] -= self.fixed[v1] * self.m_inv[v1] * nabla_C1 * ld
            self.dx[v2] -= self.fixed[v2] * self.m_inv[v2] * nabla_C2 * ld
            self.dx[v3] -= self.fixed[v3] * self.m_inv[v3] * nabla_C3 * ld

            # self.nc[v0] += 1
            # self.nc[v1] += 1
            # self.nc[v2] += 1
            # self.nc[v3] += 1

            J = ti.math.determinant(F)
            F_trace = sig[0, 0] + sig[1, 1] + sig[2, 2]

            C_vol = 0.5 * (F_trace - 3) * (F_trace - 3)
            H_vol = (F_trace - 3) * self.Dm_inv[tid].transpose()

            nabla_C_vol_0 = ti.Vector([H_vol[j, 0] for j in ti.static(range(3))])
            nabla_C_vol_1 = ti.Vector([H_vol[j, 1] for j in ti.static(range(3))])
            nabla_C_vol_2 = ti.Vector([H_vol[j, 2] for j in ti.static(range(3))])
            nabla_C_vol_3 = -(nabla_C_vol_0 + nabla_C_vol_1 + nabla_C_vol_2)

            schur_vol = (self.fixed[v0] * self.m_inv[v0] * nabla_C_vol_0.dot(nabla_C_vol_0) +
                        self.fixed[v1] * self.m_inv[v1] * nabla_C_vol_1.dot(nabla_C_vol_1) +
                        self.fixed[v2] * self.m_inv[v2] * nabla_C_vol_2.dot(nabla_C_vol_2) +
                        self.fixed[v3] * self.m_inv[v3] * nabla_C_vol_3.dot(nabla_C_vol_3) + 1e-4)

            ld_vol = C_vol / schur_vol

            self.dx[v0] -= self.fixed[v0] * self.m_inv[v0] * nabla_C_vol_0 * ld_vol
            self.dx[v1] -= self.fixed[v1] * self.m_inv[v1] * nabla_C_vol_1 * ld_vol
            self.dx[v2] -= self.fixed[v2] * self.m_inv[v2] * nabla_C_vol_2 * ld_vol
            self.dx[v3] -= self.fixed[v3] * self.m_inv[v3] * nabla_C_vol_3 * ld_vol


            self.nc[v0] += 2
            self.nc[v1] += 2
            self.nc[v2] += 2
            self.nc[v3] += 2


    @ti.kernel
    def update_x(self):

        for i in range(self.max_num_verts_dynamic):
                self.x[i] += self.v[i] * self.dt[0]


    @ti.kernel
    def confine_to_boundary(self):

        for vi in range(self.max_num_verts_dynamic):

            if self.y[vi][0] > self.padding * self.grid_size[0]:
                self.y[vi][0] = self.padding * self.grid_size[0]

            if self.y[vi][0] < -self.padding * self.grid_size[0]:
                self.y[vi][0] = -self.padding * self.grid_size[0]

            if self.y[vi][1] > self.padding * self.grid_size[1]:
                self.y[vi][1] = self.padding * self.grid_size[1]

            if self.y[vi][1] < -self.padding * self.grid_size[1]:
                self.y[vi][1] = -self.padding * self.grid_size[1]

            if self.y[vi][2] > self.padding * self.grid_size[2]:
                self.y[vi][2] = self.padding * self.grid_size[2]

            if self.y[vi][2] < -self.padding * self.grid_size[2]:
                self.y[vi][2] = -self.padding * self.grid_size[2]

    @ti.kernel
    def confine_to_boundary_v(self):

        for vi in range(self.max_num_verts_dynamic):

            if self.y[vi][0] > self.grid_size[0] and self.v[vi][0] > 0:
                self.v[vi][0] = 0

            if self.y[vi][0] < -self.grid_size[0] and self.v[vi][0] < 0:
                self.v[vi][0] = 0

            if self.y[vi][1] > self.grid_size[1] and self.v[vi][1] > 0:
                self.v[vi][1] = 0

            if self.y[vi][1] < -self.grid_size[1] and self.v[vi][1] < 0:
                self.v[vi][1] = 0

            if self.y[vi][2] > self.grid_size[2] and self.v[vi][2] > 0:
                self.v[vi][2] = 0

            if self.y[vi][2] < -self.grid_size[2] and self.v[vi][2] < 0:
                self.v[vi][2] = 0


    @ti.kernel
    def compute_velocity(self):
        for i in range(self.max_num_verts_dynamic):
                self.v[i] = (self.y[i] - self.x[i]) / self.dt[0]



    def solve_constraints_x(self):

        self.dx.fill(0.0)
        self.nc.fill(0)
        self.vt_active_set_num.fill(0)
        # self.num_springs[0] = 0
        # self.vt_active_set.fill(0)
        self.tv_active_set_num.fill(0)
        self.tv_active_set.fill(0)
        self.tv_active_set_g.fill(0)
        #
        self.vt_active_set_num_dynamic.fill(0)
        self.vt_active_set_dynamic.fill(0)
        self.vt_active_set_g_dynamic.fill(0)
        #
        self.ee_active_set_num.fill(0)
        self.ee_active_set_g.fill(0.0)

        self.ee_active_set_num_dynamic.fill(0)
        self.ee_active_set_g_dynamic.fill(0.0)

        self.num_particle_neighbours.fill(0)
        self.solve_spring_constraints_x()
        if self.enable_collision_handling:
            self.solve_collision_constraints_x()
        self.solve_fem_constraints_x()
        self.solve_pressure_constraints_x()
        self.update_dx()

        # print(self.num_springs[0])

    def solve_constraints_v(self):
        self.dv.fill(0.0)
        self.nc.fill(0)

        self.solve_spring_constraints_v()

        if self.enable_collision_handling:
            self.solve_collision_constraints_v()

        # self.solve_pressure_constraints_v()
        self.update_dv()

    @ti.kernel
    def update_dx(self):
        for vi in range(self.max_num_verts_dynamic):
            if self.nc[vi] > 0.0:
                self.dx[vi] = self.dx[vi] / self.nc[vi]
                self.y[vi] += self.fixed[vi] * self.dx[vi]

    @ti.kernel
    def set_fixed_vertices(self, fixed_vertices: ti.template()):
        for vi in range(self.max_num_verts_dynamic):
            if fixed_vertices[vi] >= 1:
                self.fixed[vi] = 0.0
            else:
                self.fixed[vi] = 1.0
    @ti.kernel
    def update_dv(self):
        for vi in range(self.max_num_verts_dynamic):
            if self.nc[vi] > 0.0:
                self.dv[vi] = self.dv[vi] / self.nc[vi]

            if self.m_inv[vi] > 0.0:
                self.v[vi] += self.fixed[vi] * self.dv[vi]

    def forward(self, n_substeps):

        dt = self.dt[0]
        self.dt[0] = dt / n_substeps


        # ti.profiler.clear_kernel_profiler_info()
        self.broad_phase()
        # self.search_neighbours()
        for _ in range(n_substeps):
            self.compute_y()
            self.confine_to_boundary()
            self.solve_constraints_x()
            self.confine_to_boundary()
            self.compute_velocity()

            if self.enable_velocity_update:
                self.solve_constraints_v()
                # self.confine_to_boundary_v()
            self.update_x()

        # b1 = ti.profiler.query_kernel_profiler_info(self.update_grid_id.__name__)
        # b2 = ti.profiler.query_kernel_profiler_info(self.prefix_sum_executor.run.__name__)
        # b3 = ti.profiler.query_kernel_profiler_info(self.counting_sort.__name__)
        # b4 = ti.profiler.query_kernel_profiler_info(self.search_neighbours.__name__)
        #
        # avg_overhead_b = b1.avg + b2.avg + b3.avg
        #
        # print("broadphase(): ", round(avg_overhead_b, 2))
        # print("search_neighbours(): ", round(b4.avg, 2))

        # if self.enable_velocity_update:
        #
        #     profile_collision_x = ti.profiler.query_kernel_profiler_info(self.solve_collision_constraints_x.__name__)
        #     profile_pressure_x = ti.profiler.query_kernel_profiler_info(self.solve_pressure_constraints_x.__name__)
        #
        #     profile_collision_v = ti.profiler.query_kernel_profiler_info(self.solve_collision_constraints_v.__name__)
        #     profile_pressure_v = ti.profiler.query_kernel_profiler_info(self.solve_pressure_constraints_v.__name__)
        #
        #     avg_overhead_x = profile_collision_x.avg + profile_pressure_x.avg
        #     avg_overhead_v = profile_collision_v.avg + profile_pressure_v.avg
        #
        #     # profile_spring_x = ti.profiler.query_kernel_profiler_info(self.solve_spring_constraints_x.__name__)
        #     # profile_spring_v = ti.profiler.query_kernel_profiler_info(self.solve_spring_constraints_v.__name__)
        #
        #
        #     # avg_overhead_x = profile_spring_x.avg
        #     # avg_overhead_v = profile_spring_v.avg
        #
        #
        #     print("constraint_solve_x(): ", round(avg_overhead_x, 2))
        #     print("constraint_solve_v(): ", round(avg_overhead_v, 2))
        #     print("ratio: ", round(100.0 * (avg_overhead_v / avg_overhead_x), 2), "%")

        self.copy_to_meshes()
        self.copy_to_particles()

        self.dt[0] = dt
        self.frame[0] = self.frame[0] + 1
        # print(self.frame[0])



