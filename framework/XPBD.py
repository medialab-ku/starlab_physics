import taichi as ti
import numpy as np
import distance as di

@ti.data_oriented
class Solver:
    def __init__(self,
                 enable_profiler,
                 mesh_dy,
                 mesh_st,
                 grid_size,
                 particle_radius,
                 dHat,
                 YM,
                 PR,
                 g,
                 dt):

        self.mesh_dy = mesh_dy
        self.mesh_st = mesh_st
        self.g = g
        self.dt = ti.field(dtype=ti.f32, shape=1)
        self.dt[0] = dt
        self.dHat = ti.field(dtype=ti.f32, shape=1)
        self.dHat[0] = dHat
        self.YM = ti.field(dtype=ti.f32, shape=1) # Young's modulus
        self.PR = ti.field(dtype=ti.f32, shape=1) # Poisson's ratio
        self.YM[0] = YM
        self.PR[0] = PR
        self.unit_vector = ti.Vector.field(n=3, dtype=ti.f32, shape=3)
        self.unit_vector[0] = ti.math.vec3(1.0, 0.0, 0.0)
        self.unit_vector[1] = ti.math.vec3(0.0, 1.0, 0.0)
        self.unit_vector[2] = ti.math.vec3(0.0, 0.0, 1.0)
        self.strain_limit = ti.field(dtype=ti.f32, shape=1)
        self.strain_limit[0] = 0.01

        self.rest_volume = ti.field(dtype=ti.f32, shape=1)
        self.rest_volume[0] = 0

        self.current_volume = ti.field(dtype=ti.f32, shape=1)
        self.current_volume[0] = 0.0

        self.num_inverted_elements = ti.field(dtype=ti.i32, shape=1)

        self.grid_size = grid_size
        self.particle_radius = particle_radius
        self.friction_coeff = ti.field(dtype=ti.f32, shape=1)
        self.friction_coeff[0] = 0
        self.grid_vertices = ti.Vector.field(n=3, dtype=ti.f32, shape=8)
        self.aabb_vertices = ti.Vector.field(n=3, dtype=ti.f32, shape=8)
        self.grid_edge_indices = ti.field(dtype=ti.u32, shape=12 * 2)


        self.padding = 0.2
        self.init_grid()

        self.kernel_radius = 4 * particle_radius
        self.cell_size = 5 * self.kernel_radius
        self.grid_origin = -self.grid_size
        self.grid_num = np.ceil(2 * self.grid_size / self.cell_size).astype(int)
        self.grid_num_dynamic = ti.Vector.field(n=3, dtype=ti.i32, shape=1)
        print("grid dim:", self.grid_num)

        self.enable_velocity_update = False
        self.enable_collision_handling = False
        self.enable_move_obstacle = False
        self.enable_profiler = enable_profiler
        self.export_mesh = False

        self.obs_lin_vel = ti.Vector.field(n=3, dtype=ti.f32, shape=1)
        self.obs_lin_vel.fill(0.0)
        self.obs_ang_vel = ti.Vector.field(n=3, dtype=ti.f32, shape=1)
        self.obs_ang_vel.fill(0.0)

        self.manual_lin_vels = ti.Vector.field(n=3, dtype=ti.f32, shape=4)
        self.manual_lin_vels.fill(0.0)
        self.manual_ang_vels = ti.Vector.field(n=3, dtype=ti.f32, shape=4)
        self.manual_ang_vels.fill(0.0)

        self.manual_ang_vels[0] = ti.Vector([70.0, 0.0, 0.0])
        self.manual_ang_vels[1] = ti.Vector([-70.0, 0.0, 0.0])
        self.manual_ang_vels[2] = ti.Vector([0, -5.0, 0.0])
        self.manual_ang_vels[3] = ti.Vector([0.0, 1.0, 0.0])



        self.max_num_verts_dynamic = len(self.mesh_dy.verts)
        self.max_num_edges_dynamic = len(self.mesh_dy.edges)
        self.max_num_faces_dynamic = len(self.mesh_dy.faces)


        print(self.max_num_verts_dynamic)

        self.vt_static_pair_cache_size = 40
        self.vt_static_pair = ti.field(dtype=ti.int32, shape=(self.max_num_verts_dynamic, self.vt_static_pair_cache_size, 2))
        self.vt_static_pair_num = ti.field(dtype=ti.int32, shape=self.max_num_verts_dynamic)
        self.vt_static_pair_g = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.max_num_verts_dynamic, self.vt_static_pair_cache_size, 4))
        self.vt_static_pair_schur = ti.field(dtype=ti.f32, shape=(self.max_num_verts_dynamic, self.vt_static_pair_cache_size))

        self.tv_static_pair_cache_size = 40
        self.tv_static_pair = ti.field(dtype=ti.int32, shape=(self.max_num_faces_dynamic, self.tv_static_pair_cache_size, 2))
        self.tv_static_pair_num = ti.field(dtype=ti.int32, shape=self.max_num_faces_dynamic)
        self.tv_static_pair_g = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.max_num_faces_dynamic, self.tv_static_pair_cache_size, 4))
        self.tv_static_pair_schur = ti.field(dtype=ti.f32, shape=(self.max_num_faces_dynamic, self.tv_static_pair_cache_size))

        self.ee_static_pair_cache_size = 40
        self.ee_static_pair = ti.field(dtype=ti.int32, shape=(self.max_num_edges_dynamic, self.ee_static_pair_cache_size, 2))
        self.ee_static_pair_num = ti.field(dtype=ti.int32, shape=self.max_num_edges_dynamic)
        self.ee_static_pair_g = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.max_num_edges_dynamic, self.ee_static_pair_cache_size, 4))
        self.ee_static_pair_schur = ti.field(dtype=ti.f32, shape=(self.max_num_edges_dynamic, self.ee_static_pair_cache_size))

        self.tv_dynamic_pair_cache_size = 40
        self.tv_dynamic_pair = ti.field(dtype=ti.int32, shape=(self.max_num_faces_dynamic, self.tv_dynamic_pair_cache_size, 2))
        self.tv_dynamic_pair_num = ti.field(dtype=ti.int32, shape=self.max_num_faces_dynamic)
        self.tv_dynamic_pair_g = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.max_num_faces_dynamic, self.tv_dynamic_pair_cache_size, 4))
        self.tv_dynamic_pair_schur = ti.field(dtype=ti.f32, shape=(self.max_num_faces_dynamic, self.tv_dynamic_pair_cache_size))

        self.ee_dynamic_pair_cache_size = 40
        self.ee_dynamic_pair = ti.field(dtype=ti.int32, shape=(self.max_num_edges_dynamic, self.ee_dynamic_pair_cache_size, 2))
        self.ee_dynamic_pair_num = ti.field(dtype=ti.int32, shape=self.max_num_edges_dynamic)
        self.ee_dynamic_pair_g = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.max_num_edges_dynamic, self.ee_dynamic_pair_cache_size, 4))
        self.ee_dynamic_pair_schur = ti.field(dtype=ti.f32, shape=(self.max_num_edges_dynamic, self.ee_dynamic_pair_cache_size))

        self.num_cells = int(self.grid_num[0] * self.grid_num[1] * self.grid_num[2])
        # self.particle_neighbours = ti.field(dtype=int, shape=(self.max_num_verts_dynamic, self.cache_size))
        # self.particle_neighbours_gradients = ti.Vector.field(n=3, dtype=ti.f32, shape=(self.max_num_verts_dynamic, self.cache_size))
        # self.num_particle_neighbours = ti.field(int, shape=self.max_num_verts_dynamic)

        self.cache_size = 10
        self.max_num_verts_static = len(self.mesh_st.verts)
        self.max_num_edges_static = len(self.mesh_st.edges)
        self.max_num_faces_static = len(self.mesh_st.faces)

        print(self.max_num_faces_static)
        # self.max_num_verts_static = len(self.mesh_st.verts)
        # self.max_num_faces_static = len(self.mesh_st.faces)

        # self.face_indices_static = ti.field(dtype=ti.i32, shape=3 * self.max_num_faces_static)
        # self.edge_indices_static = ti.field(dtype=ti.i32, shape=2 * self.max_num_edges_static)

        # self.grid_particles_num = ti.field(int, shape=self.num_cells)
        # self.grid_particles_num_temp = ti.field(int, shape= self.num_cells)
        # self.prefix_sum_executor = ti.algorithms.PrefixSumExecutor(self.grid_particles_num.shape[0])
        #
        # self.grid_ids = ti.field(int, shape=self.max_num_verts_dynamic)
        # self.grid_ids_buffer = ti.field(int, shape=self.max_num_verts_dynamic)
        # self.grid_ids_new = ti.field(int, shape=self.max_num_verts_dynamic)
        # self.cur2org = ti.field(int, shape=self.max_num_verts_dynamic)
        #
        # self.grid_particles_num_static = ti.field(int, shape=int(self.grid_num[0] * self.grid_num[1] * self.grid_num[2]))
        # self.grid_particles_num_temp_static = ti.field(int, shape=int(self.grid_num[0] * self.grid_num[1] * self.grid_num[2]))
        # self.prefix_sum_executor_static = ti.algorithms.PrefixSumExecutor(self.grid_particles_num_static.shape[0])
        #
        # num_grid_ids = self.max_num_verts_static + 3 * self.max_num_faces_static
        #
        # if num_grid_ids < 1:
        #     num_grid_ids = 1
        #
        # self.grid_ids_static = ti.field(int, shape=num_grid_ids)
        # self.grid_ids_buffer_static = ti.field(int, shape=num_grid_ids)
        # self.grid_ids_new_static = ti.field(int, shape=num_grid_ids)
        # self.cur2org_static = ti.field(int, shape=num_grid_ids)

        # self.frame = ti.field(dtype=ti.i32, shape=1)
        # self.frame[0] = 0
        #
        # self.max_num_anim = 40
        # self.num_animation = ti.field(dtype=ti.i32, shape=4)   # maximum number of handle set
        # self.cur_animation = ti.field(dtype=ti.i32, shape=4)   # maximum number of handle set
        # self.anim_rotation_mat = ti.Matrix.field(4, 4, dtype=ti.f32, shape=4)
        #
        # self.anim_local_origin = ti.Vector.field(3, dtype=ti.f32, shape=4)
        # self.active_anim_frame = ti.field(dtype=ti.i32, shape=(4, self.max_num_anim)) # maximum number of animation
        # self.action_anim = ti.Vector.field(6, dtype=ti.f32, shape=(4, self.max_num_anim)) # maximum number of animation, a animation consist (vx,vy,vz,rx,ry,rz)
        # self.anim_x = ti.Vector.field(n=3, dtype=ti.f32, shape=self.max_num_verts_dynamic)
        #
        # self.broad_phase_static()
        # self.test_var = ti.Vector.field(n=3, dtype=ti.f32, shape=self.max_num_verts_dynamic)
        # self.test()

    # @ti.kernel
    # def test(self):
    #
    #     for i in range(self.max_num_verts_dynamic):
    #         self.test_var[i] = ti.math.vec3(i, self.max_num_verts_dynamic - i, 0)
    #
    #     max = ti.math.vec3(0.0)
    #     for i in range(self.max_num_verts_dynamic):
    #         ti.atomic_max(max, self.test_var[i])
    #
    #     print(max[0], max[1], max[2])
    #
    # @ti.kernel
    # def compute_aabb(self):
    #     aabb_min = ti.math.vec3(0.0)
    #     aabb_max = ti.math.vec3(0.0)
    #
    #     for i in range(self.max_num_verts_dynamic):
    #         temp = self.y[i]
    #         ti.atomic_max(aabb_max, temp)
    #         ti.atomic_min(aabb_min, temp)
    #     # print(aabb_min)
    #     # print(aabb_max)
    #
    #     self.grid_max[0] = aabb_max
    #     self.grid_min[0] = aabb_min
    #
    #     self.aabb_vertices[0] = (1.0 + self.padding) * ti.math.vec3(aabb_max[0], aabb_max[1], aabb_max[2])
    #     self.aabb_vertices[1] = (1.0 + self.padding) * ti.math.vec3(aabb_min[0], aabb_max[1], aabb_max[2])
    #     self.aabb_vertices[2] = (1.0 + self.padding) * ti.math.vec3(aabb_min[0], aabb_max[1], aabb_min[2])
    #     self.aabb_vertices[3] = (1.0 + self.padding) * ti.math.vec3(aabb_max[0], aabb_max[1], aabb_min[2])
    #
    #     self.aabb_vertices[4] = (1.0 + self.padding) * ti.math.vec3(aabb_max[0], aabb_min[1], aabb_max[2])
    #     self.aabb_vertices[5] = (1.0 + self.padding) * ti.math.vec3(aabb_min[0], aabb_min[1], aabb_max[2])
    #     self.aabb_vertices[6] = (1.0 + self.padding) * ti.math.vec3(aabb_min[0], aabb_min[1], aabb_min[2])
    #     self.aabb_vertices[7] = (1.0 + self.padding) * ti.math.vec3(aabb_max[0], aabb_min[1], aabb_min[2])
    #
    # @ti.kernel
    # def compute_aabb_static(self):
    #     aabb_min = ti.math.vec3(0.0)
    #     aabb_max = ti.math.vec3(0.0)
    #
    #     for i in range(self.max_num_verts_static):
    #         temp = self.x_static[i]
    #         ti.atomic_max(aabb_max, temp)
    #         ti.atomic_min(aabb_min, temp)
    #
    #     ones = ti.math.vec3(1.0)
    #     aabb_max += self.padding * ones
    #     aabb_min -= self.padding * ones
    #     self.grid_max_static[0] = aabb_max
    #     self.grid_min_static[0] = aabb_min
    #
    #     self.aabb_vertices[0] = ti.math.vec3(aabb_max[0], aabb_max[1], aabb_max[2])
    #     self.aabb_vertices[1] = ti.math.vec3(aabb_min[0], aabb_max[1], aabb_max[2])
    #     self.aabb_vertices[2] = ti.math.vec3(aabb_min[0], aabb_max[1], aabb_min[2])
    #     self.aabb_vertices[3] = ti.math.vec3(aabb_max[0], aabb_max[1], aabb_min[2])
    #
    #     self.aabb_vertices[4] = ti.math.vec3(aabb_max[0], aabb_min[1], aabb_max[2])
    #     self.aabb_vertices[5] = ti.math.vec3(aabb_min[0], aabb_min[1], aabb_max[2])
    #     self.aabb_vertices[6] = ti.math.vec3(aabb_min[0], aabb_min[1], aabb_min[2])
    #     self.aabb_vertices[7] = ti.math.vec3(aabb_max[0], aabb_min[1], aabb_min[2])
    #
    #     #determine cell size
    #     r_padding = self.padding * ti.math.vec3(1.0)
    #     for fi in range(self.max_num_verts_static):
    #         v0, v1, v2 = self.face_indices_static[3 * fi + 0], self.face_indices_static[3 * fi + 1], self.face_indices_static[3 * fi + 2]
    #         x0, x1, x2 = self.x_static[v0], self.x_static[v1], self.x_static[v2]
    #
    #         self.aabb_face[fi, 0] = ti.min(x0, x1, x2)
    #         self.aabb_face[fi, 1] = ti.max(x0, x1, x2)
    #         r = (self.aabb_face[fi, 1] - self.aabb_face[fi, 0]) + r_padding
    #         ti.atomic_max(self.dynamic_cell_size[0], r)
    #
    #
    #
    # @ti.kernel
    # def determine_cell_size_and_num(self):
    #
    #     #0: min
    #     #1: max
    #
    #     self.dynamic_cell_size[0] = ti.math.vec3(0.0)
    #     r_padding = self.padding * ti.math.vec3(1.0)
    #     for fi in range(self.max_num_faces_dynamic):
    #         v0, v1, v2 = self.face_indices_dynamic[3 * fi + 0], self.face_indices_dynamic[3 * fi + 1], self.face_indices_dynamic[3 * fi + 2]
    #         x0, x1, x2 = self.x[v0], self.x[v1], self.x[v2]
    #
    #         self.aabb_face[fi, 0] = ti.min(x0, x1, x2)
    #         self.aabb_face[fi, 1] = ti.max(x0, x1, x2)
    #         r = (self.aabb_face[fi, 1] - self.aabb_face[fi, 0]) + r_padding
    #         ti.atomic_max(self.dynamic_cell_size[0], r)
    #
    #
    # @ti.kernel
    # def __init_animation_pos(self, is_selected: ti.template()):
    #     for i in is_selected:
    #         if is_selected != 0:
    #             self.anim_x[i] = self.x[i]
    #
    #     for i in ti.ndrange(4):
    #         count = 0.0
    #         origin = ti.Vector([0.0, 0.0, 0.0])
    #         idx_set_count = i + 1
    #
    #         for pidx in ti.ndrange(self.max_num_verts_dynamic):
    #             if is_selected[pidx] == idx_set_count:
    #                 count += 1
    #                 origin += self.x[pidx]
    #
    #         if count > 0.001:
    #             self.anim_local_origin[i] = origin / count
    #
    # def _reset_animation(self):
    #     self.num_animation.fill(0)
    #     self.cur_animation.fill(0)
    #     self.active_anim_frame.fill(0)
    #     self.action_anim.fill(0.0)
    #     self.anim_local_origin.fill(0.0)
    #     self.anim_rotation_mat.fill(0.0)
    #
    # def _set_animation(self, animationDict, is_selected):
    #     self.num_animation.fill(0)
    #     self.cur_animation.fill(0)
    #     self.active_anim_frame.fill(0)
    #     self.action_anim.fill(0.0)
    #     self.anim_local_origin.fill(0.0)
    #
    #     self.num_animation[0] = len(animationDict[1])
    #     self.num_animation[1] = len(animationDict[2])
    #     self.num_animation[2] = len(animationDict[3])
    #     self.num_animation[3] = len(animationDict[4])
    #     self.num_animation[3] = len(animationDict[4])
    #
    #     if (self.num_animation[0] > self.max_num_anim) or (self.num_animation[1] > self.max_num_anim) \
    #             or (self.num_animation[2] > self.max_num_anim) or (self.num_animation[3] > self.max_num_anim):
    #         print("warning :: length of some animation is longer than ", self.max_num_anim,
    #               ". Subsequent animations are ignored")
    #         self.num_animation[0] = self.num_animation[0] if self.num_animation[
    #                                                              0] < self.max_num_anim else self.max_num_anim
    #         self.num_animation[1] = self.num_animation[1] if self.num_animation[
    #                                                              1] < self.max_num_anim else self.max_num_anim
    #         self.num_animation[2] = self.num_animation[2] if self.num_animation[
    #                                                              2] < self.max_num_anim else self.max_num_anim
    #         self.num_animation[3] = self.num_animation[3] if self.num_animation[
    #                                                              3] < self.max_num_anim else self.max_num_anim
    #
    #     self.__init_animation_pos(is_selected)
    #
    #     for ic in range(4):
    #         animations_ic = animationDict[ic + 1]
    #         for a in range(self.num_animation[ic]):
    #             animation = animations_ic[a]
    #
    #             self.active_anim_frame[ic, a] = animation[6]
    #             for j in range(6):
    #                 self.action_anim[ic, a][j] = animation[j]
    #
    #     self.set_fixed_vertices(is_selected)
    #
    #     # print(self.num_animation)
    #     # print(self.active_anim_frame)
    #     # print(self.action_anim)
    #     # print(self.anim_local_origin)
    #
    #     # self.frame[0] = 0
    #     #
    #     # self.max_num_anim = 40
    #     # self.num_animation = ti.field(dtype = ti.i32,shape = 4)   # maximum number of handle set
    #     # self.anim_local_origin = ti.Vector.field(3, dtype=ti.f32, shape=4)
    #     # self.active_anim_frame = ti.field(dtype = ti.i32,shape = (4,self.max_num_anim)) # maximum number of animation
    #     # self.action_anim = ti.Vector.field(6,dtype = ti.f32,shape = (4,self.max_num_anim)) # maximum number of animation, a animation consist (vx,vy,vz,rx,ry,rz)
    #     #
    #     # self.anim_x = ti.Vector.field(n=3, dtype=ti.f32, shape=self.max_num_verts_dynamic)
    #
    # @ti.func
    # def apply_rotation(self, mat_rot, vec):
    #     v = ti.Vector([vec[0], vec[1], vec[2], 1])
    #     vv = mat_rot @ v
    #     return ti.Vector([vv[0], vv[1], vv[2]])
    #
    # @ti.func
    # def get_animation_rotation_mat(self, axis, degree):
    #     s = ti.sin(degree * 0.5)
    #     c = ti.cos(degree * 0.5)
    #     axis = s * axis
    #     q = ti.Vector([axis[0], axis[1], axis[2], c])
    #     M = ti.math.mat4(0.0)
    #
    #     M[3, 3] = 1.0
    #
    #     M[0, 0] = 1 - 2 * (q[1] ** 2 + q[2] ** 2)
    #     M[1, 1] = 1 - 2 * (q[0] ** 2 + q[2] ** 2)
    #     M[2, 2] = 1 - 2 * (q[0] ** 2 + q[1] ** 2)
    #
    #     M[2, 1] = 2 * (q[1] * q[2] + q[3] * q[0])
    #     M[1, 2] = 2 * (q[1] * q[2] - q[3] * q[0])
    #
    #     M[2, 0] = 2 * (q[0] * q[2] - q[3] * q[1])
    #     M[0, 2] = 2 * (q[0] * q[2] + q[3] * q[1])
    #
    #     M[1, 0] = 2 * (q[0] * q[1] + q[3] * q[2])
    #     M[0, 1] = 2 * (q[0] * q[1] - q[3] * q[2])
    #
    #     return M
    #
    # @ti.kernel
    # def animate_handle(self, is_selected: ti.template()):
    #
    #     for idx_set in ti.ndrange(4):
    #         cur_anim = self.cur_animation[idx_set]
    #         max_anim = self.num_animation[idx_set]
    #         is_animation_changed = False or (self.frame[0] == 0)  # when frame = 0 animation changed
    #         while self.active_anim_frame[idx_set, cur_anim] < self.frame[0] and cur_anim < max_anim:
    #             self.cur_animation[idx_set] = self.cur_animation[idx_set] + 1
    #             cur_anim = self.cur_animation[idx_set]
    #             is_animation_changed = True
    #
    #         if cur_anim < max_anim:
    #             vel = self.action_anim[idx_set, cur_anim]
    #             lin_vel = ti.Vector([vel[0], vel[1], vel[2]])
    #             self.anim_local_origin[idx_set] += lin_vel * self.dt[0]
    #
    #             ang_vel = ti.Vector([vel[3], vel[4], vel[5]])
    #             degree_rate = ang_vel.norm()
    #
    #             if is_animation_changed and degree_rate > 1e-4:
    #                 axis = ti.math.normalize(ang_vel)
    #                 self.anim_rotation_mat[idx_set] = self.get_animation_rotation_mat(axis, degree_rate * self.dt[0])
    #
    #     for i in self.anim_x:
    #         if is_selected[i] >= 1:
    #
    #             idx_set = int(is_selected[i] - 1)
    #
    #             cur_anim = int(self.cur_animation[idx_set])
    #             max_anim = int(self.num_animation[idx_set])
    #
    #             if cur_anim < max_anim:
    #                 vel = self.action_anim[idx_set, cur_anim]
    #                 lin_vel = ti.Vector([vel[0], vel[1], vel[2]])
    #                 self.anim_x[i] = self.anim_x[i] + lin_vel * self.dt[0]
    #
    #                 ang_vel = ti.Vector([vel[3], vel[4], vel[5]])
    #                 degree_rate = ang_vel.norm()
    #                 if degree_rate > 1e-4:
    #                     mat_rot = self.anim_rotation_mat[idx_set]
    #
    #                     self.anim_x[i] = self.apply_rotation(mat_rot,
    #                                                          self.anim_x[i] - self.anim_local_origin[idx_set]) + \
    #                                      self.anim_local_origin[idx_set]
    #
    #                 self.x[i] = self.anim_x[i]


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
        # self.frame[0] = 0
        self.mesh_dy.reset()
        if self.mesh_st != None:
            self.mesh_st.reset()
        # for mid in range(len(self.meshes_dynamic)):
        #     self.meshes_dynamic[mid].reset()
        #
        # for mid in range(len(self.meshes_static)):
        #     self.meshes_static[mid].reset()
        #
        # for pid in range(len(self.particles)):
        #     self.particles[pid].reset()
        #
        # for tid in range(len(self.tet_meshes_dynamic)):
        #     self.tet_meshes_dynamic[tid].reset()

        # self.init_mesh_aggregation()
        # self.init_particle_aggregation()
        # self._reset_animation()

        # self.init_vel()

    @ti.kernel
    def compute_y_test(self, dt: ti.f32):
        for v in self.mesh_dy.verts:
            v.y = v.x + v.v * dt + self.g * dt * dt

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
        for i in range(self.max_num_verts_static + 3 * self.max_num_faces_static):
            I = self.max_num_verts_static + 3 * self.max_num_faces_static - 1 - i
            base_offset_static = 0
            if self.grid_ids_static[I] - 1 >= 0:
                base_offset_static = self.grid_particles_num_static[self.grid_ids_static[I] - 1]
            self.grid_ids_new_static[I] = ti.atomic_sub(self.grid_particles_num_temp_static[self.grid_ids_static[I]], 1) - 1 + base_offset_static

        for i in self.grid_ids_static:
            new_index = self.grid_ids_new_static[i]
            self.cur2org_static[new_index] = i

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

    def broad_phase_static(self):
        self.update_grid_id_static()
        self.prefix_sum_executor_static.run(self.grid_particles_num_static)
        self.counting_sort_static()


    @ti.kernel
    def update_grid_id(self):

        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num[I] = 0

            # TODO: update the following two for-loops into a single one
        for i in range(self.max_num_verts_dynamic):
            if i < self.max_num_verts_dynamic:
                vi = i
                grid_index = self.get_flatten_grid_index(self.x[vi])
                self.grid_ids[vi] = grid_index
                ti.atomic_add(self.grid_particles_num[grid_index], 1)

            else:
                ei = i - self.max_num_verts_dynamic

                v0, v1 = self.edge_indices_dynamic[2 * ei + 0], self.edge_indices_dynamic[2 * ei + 1]
                x0, x1 = self.x[v0], self.x[v1]

                center = 0.5 * (x0 + x1)

                grid_index = self.get_flatten_grid_index(center)
                self.grid_ids[ei + self.max_num_edges_dynamic] = grid_index
                ti.atomic_add(self.grid_particles_num[grid_index], 1)

        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num_temp[I] = self.grid_particles_num[I]

    @ti.kernel
    def update_grid_id_static(self):

        for I in ti.grouped(self.grid_particles_num_static):
            self.grid_particles_num_static[I] = 0

        # TODO: update the following two for-loops into a single one

        for vi in range(self.max_num_verts_static):
            xi = self.x_static[vi]
            gi0 = self.get_flatten_grid_index(xi)
            self.grid_ids_static[vi] = gi0
            ti.atomic_add(self.grid_particles_num_static[gi0], 1)

        for fi in range(self.max_num_faces_static):

            v0, v1, v2 = self.face_indices_static[3 * fi + 0], self.face_indices_static[3 * fi + 1], self.face_indices_static[3 * fi + 2]
            x0, x1, x2 = self.x_static[v0], self.x_static[v1], self.x_static[v2]

            gi0 = self.get_flatten_grid_index(x0)
            self.grid_ids_static[self.max_num_verts_static + 3 * fi + 0] = gi0

            gi1 = self.get_flatten_grid_index(x1)
            self.grid_ids_static[self.max_num_verts_static + 3 * fi + 1] = gi1

            gi2 = self.get_flatten_grid_index(x2)
            self.grid_ids_static[self.max_num_verts_static + 3 * fi + 2] = gi2

            ti.atomic_add(self.grid_particles_num_static[gi0], 1)
            ti.atomic_add(self.grid_particles_num_static[gi1], 1)
            ti.atomic_add(self.grid_particles_num_static[gi2], 1)

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
    def solve_collision_tv_dynamic_x(self, fid, vid, dHat):

        v0 = vid
        v1 = self.face_indices_dynamic[3 * fid + 0]
        v2 = self.face_indices_dynamic[3 * fid + 1]
        v3 = self.face_indices_dynamic[3 * fid + 2]

        x0 = self.y[v0]
        x1 = self.y[v1]
        x2 = self.y[v2]
        x3 = self.y[v3]

        dtype = di.d_type_PT(x0, x1, x2, x3)
        d = dHat
        g0, g1, g2, g3 = ti.math.vec3(0.0), ti.math.vec3(0.0), ti.math.vec3(0.0), ti.math.vec3(0.0)
        schur = 0.0

        if dtype == 0:
            d = di.d_PP(x0, x1)
            if d < dHat:
                g0, g1 = di.g_PP(x0, x1)
                schur = self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) + self.fixed[v1] * self.m_inv[v1] * g1.dot(g1) + 1e-4
                ld = (dHat - d) / schur
                self.dx[v0] += self.fixed[v0] * self.m_inv[v0] * ld * g0
                self.dx[v1] += self.fixed[v1] * self.m_inv[v1] * ld * g1

                self.nc[v0] += 1
                self.nc[v1] += 1

        elif dtype == 1:
            d = di.d_PP(x0, x2)
            if d < dHat:
                g0, g2 = di.g_PP(x0, x2)
                schur = self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) + self.fixed[v2] * self.m_inv[v2] * g2.dot(g2) + 1e-4
                ld = (dHat - d) / schur
                self.dx[v0] += self.fixed[v0] * self.m_inv[v0] * ld * g0
                self.dx[v2] += self.fixed[v1] * self.m_inv[v2] * ld * g2
                self.nc[v0] += 1
                self.nc[v2] += 1


        elif dtype == 2:
            d = di.d_PP(x0, x3)
            if d < dHat:
                g0, g3 = di.g_PP(x0, x3)
                schur = self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) + self.fixed[v3] * self.m_inv[v3] * g3.dot(g3) + 1e-4
                ld = (dHat - d) / schur

                self.dx[v0] += self.fixed[v0] * self.m_inv[v0] * ld * g0
                self.dx[v3] += self.fixed[v3] * self.m_inv[v3] * ld * g3

                self.nc[v0] += 1
                self.nc[v3] += 1


        elif dtype == 3:
            d = di.d_PE(x0, x1, x2)
            if d < dHat:
                g0, g1, g2 = di.g_PE(x0, x1, x2)
                schur = self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) + self.fixed[v1] * self.m_inv[v1] * g1.dot(g1) + self.fixed[v2] * self.m_inv[v2] * g2.dot(g2) + 1e-4
                ld = (dHat - d) / schur

                self.dx[v0] += self.fixed[v0] * self.m_inv[v0] * ld * g0
                self.dx[v1] += self.fixed[v1] * self.m_inv[v1] * ld * g1
                self.dx[v2] += self.fixed[v2] * self.m_inv[v2] * ld * g2

                self.nc[v0] += 1
                self.nc[v1] += 1
                self.nc[v2] += 1

        elif dtype == 4:
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

        elif dtype == 5:
            d = di.d_PE(x0, x1, x3)
            if d < dHat:
                g0, g1, g3 = di.g_PE(x0, x1, x3)
                schur = self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) + self.fixed[v1] * self.m_inv[v1] * g1.dot(g1) + self.fixed[v3] * self.m_inv[v3] * g3.dot(g3) + 1e-4
                ld = (dHat - d) / schur

                self.dx[v0] += self.fixed[v0] * self.m_inv[v0] * ld * g0
                self.dx[v1] += self.fixed[v1] * self.m_inv[v1] * ld * g1
                self.dx[v3] += self.fixed[v3] * self.m_inv[v3] * ld * g3
                self.nc[v0] += 1
                self.nc[v1] += 1
                self.nc[v3] += 1

        elif dtype == 6:
            d = di.d_PT(x0, x1, x2, x3)

            if d < dHat:
                g0, g1, g2, g3 = di.g_PT(x0, x1, x2, x3)
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

        if d < dHat and self.tv_dynamic_pair_num[fid] < self.tv_dynamic_pair_cache_size:
            self.tv_dynamic_pair[fid, self.tv_dynamic_pair_num[fid], 0] = vid
            self.tv_dynamic_pair[fid, self.tv_dynamic_pair_num[fid], 1] = dtype
            self.tv_dynamic_pair_g[fid, self.tv_dynamic_pair_num[fid], 0] = g0
            self.tv_dynamic_pair_g[fid, self.tv_dynamic_pair_num[fid], 1] = g1
            self.tv_dynamic_pair_g[fid, self.tv_dynamic_pair_num[fid], 2] = g2
            self.tv_dynamic_pair_g[fid, self.tv_dynamic_pair_num[fid], 3] = g3
            self.tv_dynamic_pair_schur[fid, self.tv_dynamic_pair_num[fid]] = schur
            self.tv_dynamic_pair_num[fid] += 1


    @ti.func
    def solve_collision_vt_static_x(self, vid_d, fid_s, dHat):

        v0 = vid_d
        v1 = self.mesh_st.face_indices[3 * fid_s + 0]
        v2 = self.mesh_st.face_indices[3 * fid_s + 1]
        v3 = self.mesh_st.face_indices[3 * fid_s + 2]

        x0 = self.mesh_dy.verts.y[v0]

        x1 = self.mesh_st.verts.x[v1]
        x2 = self.mesh_st.verts.x[v2]
        x3 = self.mesh_st.verts.x[v3]

        g0 = ti.math.vec3(0.0)
        g1 = ti.math.vec3(0.0)
        g2 = ti.math.vec3(0.0)
        g3 = ti.math.vec3(0.0)

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
            schur = self.mesh_st.verts.m_inv[v0] * g0.dot(g0) + 1e-4
            ld = (dHat - d) / schur
            self.mesh_st.verts.dx[v0] += self.mesh_st.verts.m_inv[v0] * ld * g0
            self.mesh_st.verts.nc[v0] += 1
        #
        #     if self.vt_static_pair_num[v0] < self.vt_static_pair_cache_size:
        #         self.vt_static_pair[v0, self.vt_static_pair_num[v0], 0] = fid_s
        #         self.vt_static_pair[v0, self.vt_static_pair_num[v0], 1] = dtype
        #
        #         self.vt_static_pair_g[v0, self.vt_static_pair_num[v0], 0] = g0
        #         self.vt_static_pair_g[v0, self.vt_static_pair_num[v0], 1] = g1
        #         self.vt_static_pair_g[v0, self.vt_static_pair_num[v0], 2] = g2
        #         self.vt_static_pair_g[v0, self.vt_static_pair_num[v0], 3] = g3
        #
        #         self.vt_static_pair_schur[v0, self.vt_static_pair_num[v0]] = schur
        #         self.vt_static_pair_num[v0] += 1


    @ti.func
    def solve_collision_vt_static_v(self, vid_d, fid_s, dtype, g0, g1, g2, g3, schur, mu):

        v0 = vid_d
        v1 = self.face_indices_static[3 * fid_s + 0]
        v2 = self.face_indices_static[3 * fid_s + 1]
        v3 = self.face_indices_static[3 * fid_s + 2]

        Cv = g0.dot(self.v[v0]) + g1.dot(self.v_static[v1]) + g2.dot(self.v_static[v2]) + g3.dot(self.v_static[v3])
        if dtype == 0:
            if Cv < 0.:
                ld_v = -Cv / schur
                self.dv[v0] += self.m_inv[v0] * ld_v * g0
                self.nc[v0] += 1
                vTan0 = self.v[v0] + self.m_inv[v0] * ld_v * g0
                g0Tan = self.v_static[v1] - vTan0
                cTan = 0.5 * g0Tan.dot(g0Tan)
                schur = self.m_inv[v0] * g0Tan.dot(g0Tan) + 1e-4
                ldTan = cTan / schur
                dvTan = self.m_inv[v0] * ldTan * g0Tan

                if mu * abs(Cv) > cTan:
                    mu = 1.0
                self.dv[v0] += mu * dvTan

        elif dtype == 1:
            if Cv < 0.:
                ld_v = -Cv / schur
                self.dv[v0] += self.m_inv[v0] * ld_v * g0
                self.nc[v0] += 1
                vTan0 = self.v[v0] + self.m_inv[v0] * ld_v * g0
                g0Tan = self.v_static[v2] - vTan0
                cTan = 0.5 * g0Tan.dot(g0Tan)
                schur = self.m_inv[v0] * g0Tan.dot(g0Tan) + 1e-4
                ldTan = cTan / schur
                dvTan = self.m_inv[v0] * ldTan * g0Tan
                if mu * abs(Cv) > cTan:
                    mu = 1.0
                self.dv[v0] += mu * dvTan

        elif dtype == 2:
            if Cv < 0.:
                ld_v = -Cv / schur
                self.dv[v0] += self.m_inv[v0] * ld_v * g0
                self.nc[v0] += 1
                vTan0 = self.v[v0] + self.m_inv[v0] * ld_v * g0
                g0Tan = self.v_static[v3] - vTan0
                cTan = 0.5 * g0Tan.dot(g0Tan)
                schur = self.m_inv[v0] * g0Tan.dot(g0Tan) + 1e-4
                ldTan = cTan / schur
                dvTan = self.m_inv[v0] * ldTan * g0Tan
                if mu * abs(Cv) > cTan:
                    mu = 1.0
                self.dv[v0] += mu * dvTan


        elif dtype == 3:
            if Cv < 0.:
                ld_v = -Cv / schur
                self.dv[v0] += self.m_inv[v0] * ld_v * g0
                self.nc[v0] += 1
                vTan0 = self.v[v0] + self.m_inv[v0] * ld_v * g0
                a, b = g1.norm(), g2.norm()
                p = (a * self.v_static[v1] + b * self.v_static[v2]) / (a + b)
                g0Tan = p - vTan0
                cTan = 0.5 * g0Tan.dot(g0Tan)
                schur = self.m_inv[v0] * g0Tan.dot(g0Tan) + 1e-4
                ldTan = cTan / schur
                dvTan = self.m_inv[v0] * ldTan * g0Tan
                if mu * abs(Cv) > cTan:
                    mu = 1.0
                self.dv[v0] += mu * dvTan

        elif dtype == 4:
            if Cv < 0.:
                ld_v = -Cv / schur
                self.dv[v0] += self.m_inv[v0] * ld_v * g0
                self.nc[v0] += 1
                vTan0 = self.v[v0] + self.m_inv[v0] * ld_v * g0
                a, b = g2.norm(), g3.norm()
                p = (a * self.v_static[v2] + b * self.v_static[v3]) / (a + b)
                g0Tan = p - vTan0
                cTan = 0.5 * g0Tan.dot(g0Tan)
                schur = self.m_inv[v0] * g0Tan.dot(g0Tan) + 1e-4
                ldTan = cTan / schur
                dvTan = self.m_inv[v0] * ldTan * g0Tan
                if mu * abs(Cv) > cTan:
                    mu = 1.0
                self.dv[v0] += mu * dvTan

        elif dtype == 5:
            if Cv < 0.:
                ld_v = -Cv / schur
                self.dv[v0] += self.m_inv[v0] * ld_v * g0
                self.nc[v0] += 1
                vTan0 = self.v[v0] + self.m_inv[v0] * ld_v * g0
                a, b = g1.norm(), g3.norm()
                p = (a * self.v_static[v1] + b * self.v_static[v3]) / (a + b)
                g0Tan = vTan0 - p
                cTan = 0.5 * g0Tan.dot(g0Tan)
                schur = self.m_inv[v0] * g0Tan.dot(g0Tan) + 1e-4
                ldTan = cTan / schur
                dvTan = self.m_inv[v0] * ldTan * g0Tan
                if mu * abs(Cv) > cTan:
                    mu = 1.0
                self.dv[v0] += mu * dvTan

        elif dtype == 6:
            if Cv < 0.:
                ld_v = -Cv / schur
                self.dv[v0] += self.m_inv[v0] * ld_v * g0
                self.nc[v0] += 1
                vTan0 = self.v[v0] + self.m_inv[v0] * ld_v * g0
                a, b, c = g1.norm(), g2.norm(), g3.norm()
                p = (a * self.v_static[v1] + b * self.v_static[v2] + c * self.v_static[v3]) / (a + b + c)
                g0Tan = p - vTan0
                cTan = 0.5 * g0Tan.dot(g0Tan)
                schur = self.m_inv[v0] * g0Tan.dot(g0Tan) + 1e-4
                ldTan = cTan / schur
                dvTan = self.m_inv[v0] * ldTan * g0Tan
                if mu * abs(Cv) > cTan:
                    mu = 1.0
                self.dv[v0] += mu * dvTan

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
        d = dHat
        g0, g1, g2, g3 = ti.math.vec3(0.0), ti.math.vec3(0.0), ti.math.vec3(0.0), ti.math.vec3(0.0)
        schur = 0.0
        if dtype == 0:
            d = di.d_PP(x0, x1)
            if d < dHat:
                g0, g1 = di.g_PP(x0, x1)
                schur = self.fixed[v1] * self.m_inv[v1] * g1.dot(g1) + 1e-4
                ld = (dHat - d) / schur
                self.dx[v1] += self.fixed[v1] * self.m_inv[v1] * ld * g1
                self.nc[v1] += 1

        elif dtype == 1:
            d = di.d_PP(x0, x2)
            if d < dHat:
                g0, g2 = di.g_PP(x0, x2)
                schur = self.fixed[v2] * self.m_inv[v2] * g2.dot(g2) + 1e-4
                ld = (dHat - d) / schur
                self.dx[v2] += self.fixed[v2] * self.m_inv[v2] * ld * g2
                self.nc[v2] += 1


        elif dtype == 2:
            d = di.d_PP(x0, x3)
            if d < dHat:
                g0, g3 = di.g_PP(x0, x3)
                schur = self.fixed[v3] * self.m_inv[v3] * g3.dot(g3) + 1e-4
                ld = (dHat - d) / schur
                self.dx[v3] += self.fixed[v3] * self.m_inv[v3] * ld * g3
                self.nc[v3] += 1

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

        if d < dHat and self.tv_static_pair_num[fid_d] < self.tv_static_pair_cache_size:
            self.tv_static_pair[fid_d, self.tv_static_pair_num[fid_d], 0] = vid_s
            self.tv_static_pair[fid_d, self.tv_static_pair_num[fid_d], 1] = dtype
            self.tv_static_pair_g[fid_d, self.tv_static_pair_num[fid_d], 0] = g0
            self.tv_static_pair_g[fid_d, self.tv_static_pair_num[fid_d], 1] = g1
            self.tv_static_pair_g[fid_d, self.tv_static_pair_num[fid_d], 2] = g2
            self.tv_static_pair_g[fid_d, self.tv_static_pair_num[fid_d], 3] = g3
            self.tv_static_pair_schur[fid_d, self.tv_static_pair_num[fid_d]] = schur
            self.tv_static_pair_num[fid_d] += 1

    @ti.func
    def solve_collision_tv_static_v(self, fid_d, vid_s, dtype, g0, g1, g2, g3, schur, mu):

        v0 = vid_s
        v1 = self.face_indices_dynamic[3 * fid_d + 0]
        v2 = self.face_indices_dynamic[3 * fid_d + 1]
        v3 = self.face_indices_dynamic[3 * fid_d + 2]

        Cv = g0.dot(self.v_static[v0]) + g1.dot(self.v[v1]) + g3.dot(self.v[v3])
        if dtype == 0:
            if Cv < 0.0:
                ld_v = - Cv / schur
                self.dx[v1] += self.fixed[v1] * self.m_inv[v1] * ld_v * g1
                self.nc[v1] += 1
                vTan1 = self.v[v1] + self.fixed[v1] * self.m_inv[v1] * g1 * ld_v
                g1Tan = self.v_static[v0] - vTan1
                cTan = 0.5 * g1Tan.dot(g1Tan)
                schur = self.m_inv[v1] * g1Tan.dot(g1Tan) + 1e-4
                ldTan = cTan / schur
                dvTan1 = self.m_inv[v1] * ldTan * g1Tan

                if mu * abs(Cv) > cTan:
                    mu = 1.0
                self.dv[v1] += mu * dvTan1

        elif dtype == 1:
            if Cv < 0.0:
                ld_v = - Cv / schur
                self.dx[v2] += self.fixed[v2] * self.m_inv[v2] * ld_v * g2
                self.nc[v2] += 1

                vTan2 = self.v[v2] + self.fixed[v2] * self.m_inv[v2] * g2 * ld_v
                g2Tan = self.v_static[v0] - vTan2
                cTan = 0.5 * g2Tan.dot(g2Tan)
                schur = self.m_inv[v2] * g2Tan.dot(g2Tan) + 1e-4
                ldTan = cTan / schur
                dvTan2 = self.m_inv[v2] * ldTan * g2Tan

                if mu * abs(Cv) > cTan:
                    mu = 1.0
                self.dv[v2] += mu * dvTan2

        elif dtype == 2:
            if Cv < 0.0:
                ld_v = - Cv / schur
                self.dx[v3] += self.fixed[v3] * self.m_inv[v3] * ld_v * g3
                self.nc[v3] += 1

                vTan3 = self.v[v3] + self.fixed[v3] * self.m_inv[v3] * g3 * ld_v
                g3Tan = self.v_static[v0] - vTan3
                cTan = 0.5 * g3Tan.dot(g3Tan)
                schur = self.m_inv[v3] * g3Tan.dot(g3Tan) + 1e-4
                ldTan = cTan / schur
                dvTan3 = self.m_inv[v3] * ldTan * g3Tan

                if mu * abs(Cv) > cTan:
                    mu = 1.0
                self.dv[v3] += mu * dvTan3

        elif dtype == 3:

            if Cv < 0.0:
                ld_v = - Cv / schur
                self.dx[v1] += self.fixed[v1] * self.m_inv[v1] * ld_v * g1
                self.dx[v2] += self.fixed[v2] * self.m_inv[v2] * ld_v * g2
                self.nc[v1] += 1
                self.nc[v2] += 1

                vTan1 = self.v[v1] + self.fixed[v1] * self.m_inv[v1] * g1 * ld_v
                vTan2 = self.v[v2] + self.fixed[v2] * self.m_inv[v2] * g2 * ld_v

                a, b = g1.norm(), g2.norm()
                ab = (a + b)
                p = (a * vTan1 + b * vTan2) / (a + b)
                g1Tan = (a / ab) * (self.v_static[v0] - p)
                g2Tan = (b / ab) * (self.v_static[v0] - p)
                cTan = 0.5 * (self.v_static[v0] - p).dot(self.v_static[v0] - p)
                schur = self.m_inv[v1] * g1Tan.dot(g1Tan) + self.m_inv[v2] * g2Tan.dot(g2Tan) + 1e-4
                ldTan = cTan / schur
                dvTan1 = self.m_inv[v1] * ldTan * g1Tan
                dvTan2 = self.m_inv[v2] * ldTan * g2Tan

                if mu * abs(Cv) > cTan:
                    mu = 1.0

                self.dv[v1] += mu * dvTan1
                self.dv[v2] += mu * dvTan2

        elif dtype == 4:
            if Cv < 0.0:
                ld_v = -Cv / schur
                self.dx[v2] += self.fixed[v2] * self.m_inv[v2] * ld_v * g2
                self.dx[v3] += self.fixed[v3] * self.m_inv[v3] * ld_v * g3

                self.nc[v2] += 1
                self.nc[v3] += 1

                vTan2 = self.v[v2] + self.fixed[v2] * self.m_inv[v2] * g2 * ld_v
                vTan3 = self.v[v3] + self.fixed[v3] * self.m_inv[v3] * g3 * ld_v

                a, b = g2.norm(), g3.norm()
                ab = (a + b)
                p = (a * vTan2 + b * vTan3) /ab
                g2Tan = (a/ab) * (self.v_static[v0] - p)
                g3Tan = (b/ab) * (self.v_static[v0] - p)
                cTan = 0.5 * (self.v_static[v0] - p).dot(self.v_static[v0] - p)
                schur = self.m_inv[v2] * g2Tan.dot(g2Tan) + self.m_inv[v3] * g3Tan.dot(g3Tan) + 1e-4
                ldTan = cTan / schur
                dvTan2 = self.m_inv[v2] * ldTan * g2Tan
                dvTan3 = self.m_inv[v3] * ldTan * g3Tan


                if mu * abs(Cv) > cTan:
                    mu = 1.0

                self.dv[v2] += mu * dvTan2
                self.dv[v3] += mu * dvTan3

        elif dtype == 5:
            if Cv < 0.0:
                ld_v = -Cv / schur
                self.dx[v1] += self.fixed[v1] * self.m_inv[v1] * ld_v * g1
                self.dx[v3] += self.fixed[v3] * self.m_inv[v3] * ld_v * g3
                self.nc[v1] += 1
                self.nc[v3] += 1

                vTan1 = self.v[v1] + self.fixed[v1] * self.m_inv[v1] * g1 * ld_v
                vTan3 = self.v[v3] + self.fixed[v3] * self.m_inv[v3] * g3 * ld_v
                a, b = g1.norm(), g3.norm()
                ab = (a + b)
                p = (a * vTan1 + b * vTan3) / ab
                g1Tan = (a/ab) * (self.v_static[v0] - p)
                g3Tan = (b/ab) * (self.v_static[v0] - p)
                cTan = 0.5 * (self.v_static[v0] - p).dot(self.v_static[v0] - p)
                schur = self.m_inv[v1] * g1Tan.dot(g1Tan) + self.m_inv[v3] * g3Tan.dot(g3Tan) + 1e-4
                ldTan = cTan / schur
                dvTan1 = self.m_inv[v1] * ldTan * g1Tan
                dvTan3 = self.m_inv[v3] * ldTan * g3Tan

                if mu * abs(Cv) > cTan:
                    mu = 1.0

                self.dv[v1] += mu * dvTan1
                self.dv[v3] += mu * dvTan3

        elif dtype == 6:
            if Cv < 0.0:
                ld_v = -Cv / schur

                self.dx[v1] += self.fixed[v1] * self.m_inv[v1] * ld_v * g1
                self.dx[v2] += self.fixed[v2] * self.m_inv[v2] * ld_v * g2
                self.dx[v3] += self.fixed[v3] * self.m_inv[v3] * ld_v * g3
                self.nc[v1] += 1
                self.nc[v2] += 1
                self.nc[v3] += 1

                vTan1 = self.v[v1] + self.fixed[v1] * self.m_inv[v1] * g1 * ld_v
                vTan2 = self.v[v2] + self.fixed[v2] * self.m_inv[v2] * g2 * ld_v
                vTan3 = self.v[v3] + self.fixed[v3] * self.m_inv[v3] * g3 * ld_v

                a, b, c = g1.norm(), g2.norm(), g3.norm()
                abc = (a + b + c)
                p = (a * vTan1 + b * vTan2 + c * vTan3) / abc
                g1Tan = (a / abc) * (self.v_static[v0] - p)
                g2Tan = (b / abc) * (self.v_static[v0] - p)
                g3Tan = (c / abc) * (self.v_static[v0] - p)
                cTan = 0.5 * (self.v_static[v0] - p).dot(self.v_static[v0] - p)
                schur = self.m_inv[v1] * g1Tan.dot(g1Tan) + self.m_inv[v2] * g2Tan.dot(g2Tan) + self.m_inv[v3] * g3Tan.dot(g3Tan) + 1e-4
                ldTan = cTan / schur
                dvTan1 = self.m_inv[v1] * ldTan * g1Tan
                dvTan2 = self.m_inv[v2] * ldTan * g2Tan
                dvTan3 = self.m_inv[v3] * ldTan * g3Tan

                if mu * abs(Cv) > cTan:
                    mu = 1.0

                self.dv[v1] += mu * dvTan1
                self.dv[v2] += mu * dvTan2
                self.dv[v3] += mu * dvTan3

    @ti.func
    def solve_collision_tv_dynamic_v(self, fid_d, vid_d, dtype, g0, g1, g2, g3, schur, mu):

        v0 = vid_d
        v1 = self.face_indices_dynamic[3 * fid_d + 0]
        v2 = self.face_indices_dynamic[3 * fid_d + 1]
        v3 = self.face_indices_dynamic[3 * fid_d + 2]

        Cv = g0.dot(self.v[v0]) + g1.dot(self.v[v1]) + g2.dot(self.v[v2]) + g3.dot(self.v[v3])

        if Cv < 0.0:
            ld_v = -Cv / schur
            if dtype == 0:

                self.dv[v0] += self.fixed[v0] * self.m_inv[v0] * g0 * ld_v
                self.dv[v1] += self.fixed[v1] * self.m_inv[v1] * g1 * ld_v

                self.nc[v0] += 1
                self.nc[v1] += 1

                vTan0 = self.v[v0] + self.fixed[v0] * self.m_inv[v0] * g0 * ld_v
                vTan1 = self.v[v1] + self.fixed[v1] * self.m_inv[v1] * g1 * ld_v
                g0Tan = -(vTan0 - vTan1)
                g1Tan = vTan0 - vTan1
                cTan = 0.5 * (g1Tan.dot(g1Tan) + g0Tan.dot(g0Tan))
                schur = self.m_inv[v0] * g0Tan.dot(g0Tan) + self.m_inv[v1] * g1Tan.dot(g1Tan) + 1e-4
                ldTan = cTan / schur
                dvTan0 = self.m_inv[v0] * ldTan * g0Tan
                dvTan1 = self.m_inv[v1] * ldTan * g1Tan

                if mu * abs(Cv) > cTan:
                    mu = 1.0
                self.dv[v0] += mu * dvTan0
                self.dv[v1] += mu * dvTan1

            elif dtype == 1:

                self.dv[v0] += self.fixed[v0] * self.m_inv[v0] * g0 * ld_v
                self.dv[v2] += self.fixed[v2] * self.m_inv[v2] * g2 * ld_v

                self.nc[v0] += 1
                self.nc[v2] += 1

                vTan0 = self.v[v0] + self.fixed[v0] * self.m_inv[v0] * g0 * ld_v
                vTan2 = self.v[v2] + self.fixed[v2] * self.m_inv[v2] * g2 * ld_v

                g0Tan = -(vTan0 - vTan2)
                g2Tan = vTan0 - vTan2
                cTan = 0.5 * (g2Tan.dot(g2Tan) + g0Tan.dot(g0Tan))
                schur = self.m_inv[v0] * g0Tan.dot(g0Tan) + self.m_inv[v2] * g2Tan.dot(g2Tan) + 1e-4
                ldTan = cTan / schur
                dvTan0 = self.m_inv[v0] * ldTan * g0Tan
                dvTan2 = self.m_inv[v1] * ldTan * g2Tan

                if mu * abs(Cv) > cTan:
                    mu = 1.0
                self.dv[v0] += mu * dvTan0
                self.dv[v2] += mu * dvTan2

            elif dtype == 2:
                self.dv[v0] += self.fixed[v0] * self.m_inv[v0] * g0 * ld_v
                self.dv[v3] += self.fixed[v3] * self.m_inv[v3] * g3 * ld_v

                self.nc[v0] += 1
                self.nc[v3] += 1

                vTan0 = self.v[v0] + self.fixed[v0] * self.m_inv[v0] * g0 * ld_v
                vTan3 = self.v[v3] + self.fixed[v3] * self.m_inv[v3] * g3 * ld_v

                g0Tan = -(vTan0 - vTan3)
                g3Tan = vTan0 - vTan3
                cTan = 0.5 * (g3Tan.dot(g3Tan) + g0Tan.dot(g0Tan))
                schur = self.m_inv[v0] * g0Tan.dot(g0Tan) + self.m_inv[v3] * g3Tan.dot(g3Tan) + 1e-4
                ldTan = cTan / schur
                dvTan0 = self.m_inv[v0] * ldTan * g0Tan
                dvTan3 = self.m_inv[v3] * ldTan * g3Tan

                if mu * abs(Cv) > cTan:
                    mu = 1.0
                self.dv[v0] += mu * dvTan0
                self.dv[v3] += mu * dvTan3

            elif dtype == 3:

                self.dv[v0] += self.fixed[v0] * self.m_inv[v0] * g0 * ld_v
                self.dv[v1] += self.fixed[v1] * self.m_inv[v1] * g1 * ld_v
                self.dv[v2] += self.fixed[v2] * self.m_inv[v2] * g2 * ld_v

                self.nc[v0] += 1
                self.nc[v1] += 1
                self.nc[v2] += 1

                vTan0 = self.v[v0] + self.fixed[v0] * self.m_inv[v0] * g0 * ld_v
                vTan1 = self.v[v1] + self.fixed[v1] * self.m_inv[v1] * g1 * ld_v
                vTan2 = self.v[v2] + self.fixed[v2] * self.m_inv[v2] * g2 * ld_v

                a, b = g1.norm(), g2.norm()
                ab = a + b
                p = (a * vTan1 + b * vTan2) / ab
                g0Tan = p - vTan0
                g1Tan = -( a / ab) * g0Tan
                g2Tan = -( b / ab) * g0Tan
                cTan = 0.5 * (g0Tan.dot(g0Tan) + g1Tan.dot(g1Tan) + g2Tan.dot(g2Tan))
                schur = self.m_inv[v0] * g0Tan.dot(g0Tan) + self.m_inv[v1] * g1Tan.dot(g1Tan) + self.m_inv[v2] * g2Tan.dot(g2Tan) + 1e-4
                ldTan = cTan / schur
                dvTan0 = self.m_inv[v0] * ldTan * g0Tan
                dvTan1 = self.m_inv[v1] * ldTan * g1Tan
                dvTan2 = self.m_inv[v2] * ldTan * g2Tan
                if mu * abs(Cv) > cTan:
                    mu = 1.0
                self.dv[v0] += mu * dvTan0
                self.dv[v1] += mu * dvTan1
                self.dv[v2] += mu * dvTan2

            elif dtype == 4:

                self.dv[v0] += self.fixed[v0] * self.m_inv[v0] * g0 * ld_v
                self.dv[v2] += self.fixed[v2] * self.m_inv[v2] * g2 * ld_v
                self.dv[v3] += self.fixed[v3] * self.m_inv[v3] * g3 * ld_v

                self.nc[v0] += 1
                self.nc[v2] += 1
                self.nc[v3] += 1

                vTan0 = self.v[v0] + self.fixed[v0] * self.m_inv[v0] * g0 * ld_v
                vTan2 = self.v[v2] + self.fixed[v2] * self.m_inv[v2] * g2 * ld_v
                vTan3 = self.v[v3] + self.fixed[v3] * self.m_inv[v3] * g3 * ld_v

                a, b = g2.norm(), g3.norm()
                ab = a + b
                p = (a * vTan2 + b * vTan3) /ab
                g0Tan = p - vTan0
                g2Tan = -(a/ab) * g0Tan
                g3Tan = -(b/ab) * g0Tan
                cTan = 0.5 * (g0Tan.dot(g0Tan) + g2Tan.dot(g2Tan) + g3Tan.dot(g3Tan))
                schur = self.m_inv[v0] * g0Tan.dot(g0Tan) + self.m_inv[v2] * g2Tan.dot(g2Tan) + self.m_inv[v3] * g3Tan.dot(g3Tan) + 1e-4
                ldTan = cTan / schur
                dvTan0 = self.m_inv[v0] * ldTan * g0Tan
                dvTan2 = self.m_inv[v2] * ldTan * g2Tan
                dvTan3 = self.m_inv[v3] * ldTan * g3Tan
                if mu * abs(Cv) > cTan:
                    mu = 1.0
                self.dv[v0] += mu * dvTan0
                self.dv[v2] += mu * dvTan2
                self.dv[v3] += mu * dvTan3


            elif dtype == 5:

                self.dv[v0] += self.fixed[v0] * self.m_inv[v0] * g0 * ld_v
                self.dv[v1] += self.fixed[v1] * self.m_inv[v1] * g1 * ld_v
                self.dv[v3] += self.fixed[v3] * self.m_inv[v3] * g3 * ld_v

                self.nc[v0] += 1
                self.nc[v1] += 1
                self.nc[v3] += 1

                vTan0 = self.v[v0] + self.fixed[v0] * self.m_inv[v0] * g0 * ld_v
                vTan1 = self.v[v1] + self.fixed[v1] * self.m_inv[v1] * g1 * ld_v
                vTan3 = self.v[v3] + self.fixed[v3] * self.m_inv[v3] * g3 * ld_v

                a, b = g1.norm(), g3.norm()
                ab = a + b
                p = (a * vTan1 + b * vTan3) / ab
                g0Tan = p - vTan0
                g1Tan = -( a/ab)* g0Tan
                g3Tan = -( b/ab)* g0Tan
                cTan = 0.5 * (g0Tan.dot(g0Tan) + g1Tan.dot(g1Tan) + g3Tan.dot(g3Tan))
                schur = self.m_inv[v0] * g0Tan.dot(g0Tan) + self.m_inv[v1] * g1Tan.dot(g1Tan) + self.m_inv[v3] * g3Tan.dot(g3Tan) + 1e-4
                ldTan = cTan / schur
                dvTan0 = self.m_inv[v0] * ldTan * g0Tan
                dvTan1 = self.m_inv[v1] * ldTan * g1Tan
                dvTan3 = self.m_inv[v3] * ldTan * g3Tan
                if mu * abs(Cv) > cTan:
                    mu = 1.0
                self.dv[v0] += mu * dvTan0
                self.dv[v1] += mu * dvTan1
                self.dv[v3] += mu * dvTan3

            elif dtype == 6:

                self.dv[v0] += self.fixed[v0] * self.m_inv[v0] * g0 * ld_v
                self.dv[v1] += self.fixed[v1] * self.m_inv[v1] * g1 * ld_v
                self.dv[v2] += self.fixed[v2] * self.m_inv[v2] * g2 * ld_v
                self.dv[v3] += self.fixed[v3] * self.m_inv[v3] * g3 * ld_v

                self.nc[v0] += 1
                self.nc[v1] += 1
                self.nc[v2] += 1
                self.nc[v3] += 1

                vTan0 = self.v[v0] + self.fixed[v0] * self.m_inv[v0] * g0 * ld_v
                vTan1 = self.v[v1] + self.fixed[v1] * self.m_inv[v1] * g1 * ld_v
                vTan2 = self.v[v2] + self.fixed[v2] * self.m_inv[v2] * g2 * ld_v
                vTan3 = self.v[v3] + self.fixed[v3] * self.m_inv[v3] * g3 * ld_v

                a, b, c = g1.norm(), g2.norm(), g3.norm()
                abc = a + b + c
                p = (a * vTan1 + b * vTan2 + c * vTan3) / abc
                g0Tan = p - vTan0
                g1Tan = - (a/abc) * g0Tan
                g2Tan = - (b/abc) * g0Tan
                g3Tan = - (c/abc) * g0Tan
                cTan = 0.5 * (g0Tan.dot(g0Tan) + g1Tan.dot(g1Tan) + g2Tan.dot(g2Tan) + g3Tan.dot(g3Tan))
                schur = self.m_inv[v0] * g0Tan.dot(g0Tan) + self.m_inv[v1] * g1Tan.dot(g1Tan) + self.m_inv[v2] * g2Tan.dot(g2Tan) + self.m_inv[v3] * g3Tan.dot(g3Tan) + 1e-4
                ldTan = cTan / schur
                dvTan0 = self.m_inv[v0] * ldTan * g0Tan
                dvTan1 = self.m_inv[v1] * ldTan * g1Tan
                dvTan2 = self.m_inv[v2] * ldTan * g2Tan
                dvTan3 = self.m_inv[v3] * ldTan * g3Tan
                if mu * abs(Cv) > cTan:
                    mu = 1.0
                self.dv[v0] += mu * dvTan0
                self.dv[v1] += mu * dvTan1
                self.dv[v2] += mu * dvTan2
                self.dv[v3] += mu * dvTan3

    @ti.func
    def solve_collision_ee_static_x(self, eid_d, eid_s, dHat):

        v0 = self.edge_indices_dynamic[2 * eid_d + 0]
        v1 = self.edge_indices_dynamic[2 * eid_d + 1]

        v2 = self.edge_indices_static[2 * eid_s + 0]
        v3 = self.edge_indices_static[2 * eid_s + 1]

        x0, x1 = self.y[v0], self.y[v1]
        x2, x3 = self.x_static[v2], self.x_static[v3]

        dtype = di.d_type_EE(x0, x1, x2, x3)
        g0, g1, g2, g3 = ti.math.vec3(0.0), ti.math.vec3(0.0), ti.math.vec3(0.0), ti.math.vec3(0.0)
        d = dHat
        schur = 0.0
        if dtype == 0:
            d = di.d_PP(x0, x2)
            if d < dHat:
                g0, g2 = di.g_PP(x0, x2)
                schur = self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) + 1e-4
                ld = (dHat - d) / schur
                self.dx[v0] += self.fixed[v0] * self.m_inv[v0] * ld * g0
                self.nc[v0] += 1

        elif dtype == 1:
            d = di.d_PP(x0, x3)
            if d < dHat:
                g0, g3 = di.g_PP(x0, x3)
                schur = self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) + 1e-4
                ld = (dHat - d) / schur
                self.dx[v0] += self.fixed[v0] * self.m_inv[v0] * ld * g0
                self.nc[v0] += 1

        elif dtype == 2:
            d = di.d_PE(x0, x2, x3)
            if d < dHat:
                g0, g2, g3 = di.g_PE(x0, x2, x3)
                schur = self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) + 1e-4
                ld = (dHat - d) / schur
                self.dx[v0] += self.fixed[v0] * self.m_inv[v0] * ld * g0
                self.nc[v0] += 1

        elif dtype == 3:
            d = di.d_PP(x1, x2)
            if d < dHat:
                g1, g2 = di.g_PP(x1, x2)
                schur = self.fixed[v1] * self.m_inv[v1] * g1.dot(g1) + 1e-4
                ld = (dHat - d) / schur

                self.dx[v1] += self.m_inv[v1] * ld * g1
                self.nc[v1] += 1

        elif dtype == 4:
            d = di.d_PP(x1, x3)
            if d < dHat:
                g1, g3 = di.g_PP(x1, x3)
                schur = self.m_inv[v1] * g1.dot(g1) + 1e-4
                ld = (dHat - d) / schur

                self.dx[v1] += self.m_inv[v0] * ld * g1
                self.nc[v1] += 1

        elif dtype == 5:
            d = di.d_PE(x1, x2, x3)
            if d < dHat:
                g1, g2, g3 = di.g_PE(x1, x2, x3)
                schur = self.m_inv[v1] * g1.dot(g1) + 1e-4
                ld = (dHat - d) / schur
                self.dx[v1] += self.m_inv[v0] * ld * g1
                self.nc[v1] += 1

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


        elif dtype == 8:
            x01 = x0 - x1
            x23 = x2 - x3
            # metric_para_EE = x01.cross(x23).norm()
            # if metric_para_EE > 1e-3:
            d = di.d_EE(x0, x1, x2, x3)
            if d < dHat:
                g0, g1, g2, g3 = di.g_EE(x0, x1, x2, x3)
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v1] * g1.dot(g1) + 1e-4
                ld = (dHat - d) / schur
                self.dx[v0] += self.m_inv[v0] * ld * g0
                self.dx[v1] += self.m_inv[v1] * ld * g1
                self.nc[v0] += 1
                self.nc[v1] += 1

        if d < dHat and self.ee_static_pair_num[eid_d] < self.ee_static_pair_cache_size:
            self.ee_static_pair[eid_d, self.ee_static_pair_num[eid_d], 0] = eid_s
            self.ee_static_pair[eid_d, self.ee_static_pair_num[eid_d], 1] = dtype
            self.ee_static_pair_g[eid_d, self.ee_static_pair_num[eid_d], 0] = g0
            self.ee_static_pair_g[eid_d, self.ee_static_pair_num[eid_d], 1] = g1
            self.ee_static_pair_g[eid_d, self.ee_static_pair_num[eid_d], 2] = g2
            self.ee_static_pair_g[eid_d, self.ee_static_pair_num[eid_d], 3] = g3
            self.ee_static_pair_schur[eid_d, self.ee_static_pair_num[eid_d]] = schur
            self.ee_static_pair_num[eid_d] += 1


    @ti.func
    def solve_collision_ee_static_v(self, eid_d, eid_s, dHat, mu):

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
                Cv = g0.dot(self.v[v0]) + g2.dot(self.v_static[v2])
                if Cv < 0.0:
                    ld_v = -Cv / schur

                    self.dv[v0] += self.fixed[v0] * self.m_inv[v0] * ld_v * g0
                    self.nc[v0] += 1

                    vTan0 = self.v[v0] + self.fixed[v0] * self.m_inv[v0] * ld_v * g0
                    g0Tan = self.v_static[v2] - vTan0
                    cTan = 0.5 * g0Tan.dot(g0Tan)
                    schur = self.m_inv[v0] * g0Tan.dot(g0Tan) + 1e-4
                    ldTan = cTan / schur
                    dvTan0 = self.m_inv[v0] * ldTan * g0Tan

                    if mu * abs(Cv) > cTan:
                        mu = 1.0

                    self.dv[v0] += mu * dvTan0

        elif dtype == 1:
            d = di.d_PP(x0, x3)
            if d < dHat:
                g0, g3 = di.g_PP(x0, x3)
                schur = self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) + 1e-4
                Cv = g0.dot(self.v[v0]) + g3.dot(self.v_static[v3])
                if Cv < 0.0:
                    ld_v = -Cv / schur
                    self.dv[v0] += self.fixed[v0] * self.m_inv[v0] * ld_v * g0
                    self.nc[v0] += 1

                    vTan0 = self.v[v0] + self.fixed[v0] * self.m_inv[v0] * ld_v * g0
                    g0Tan = self.v_static[v3] - vTan0
                    cTan = 0.5 * g0Tan.dot(g0Tan)
                    schur = self.m_inv[v0] * g0Tan.dot(g0Tan) + 1e-4

                    ldTan = cTan / schur
                    dvTan0 = self.m_inv[v0] * ldTan * g0Tan
                    if mu * abs(Cv) > cTan:
                        mu = 1.0

                    self.dv[v0] += mu * dvTan0

        elif dtype == 2:
            d = di.d_PE(x0, x2, x3)
            if d < dHat:
                g0, g2, g3 = di.g_PE(x0, x2, x3)
                schur = self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) + 1e-4
                Cv = g0.dot(self.v[v0]) + g3.dot(self.v_static[v3])
                if Cv < 0.0:
                    ld_v = -Cv / schur
                    self.dv[v0] += self.fixed[v0] * self.m_inv[v0] * ld_v * g0
                    self.nc[v0] += 1

                    vTan0 = self.v[v0] + self.fixed[v0] * self.m_inv[v0] * ld_v * g0
                    c, d = g2.norm(), g3.norm()
                    c1 = c / (c + d)
                    d1 = d / (c + d)

                    p1 = c1 * self.v_static[v2] + d1 * self.v_static[v3]

                    g0Tan = (p1 - vTan0)
                    cTan = 0.5 * (p1 - vTan0).dot(p1 - vTan0)
                    schur = self.m_inv[v0] * g0Tan.dot(g0Tan) + 1e-4
                    ldTan = cTan / schur
                    dvTan0 = self.m_inv[v0] * ldTan * g0Tan

                    if mu * abs(Cv) > cTan:
                        mu = 1.0

                    self.dv[v0] += mu * dvTan0

        elif dtype == 3:
            d = di.d_PP(x1, x2)
            if d < dHat:
                g1, g2 = di.g_PP(x1, x2)
                schur = self.fixed[v1] * self.m_inv[v1] * g1.dot(g1) + 1e-4
                Cv = g1.dot(self.v[v1]) + g2.dot(self.v_static[v2])
                if Cv < 0.0:
                    ld_v = -Cv / schur
                    self.dv[v1] += self.fixed[v1] * self.m_inv[v1] * ld_v * g1
                    self.nc[v1] += 1

                    vTan1 = self.v[v1] + self.fixed[v1] * self.m_inv[v1] * ld_v * g1
                    g1Tan = self.v_static[v2] - vTan1
                    cTan = 0.5 * g1Tan.dot(g1Tan)
                    schur = self.m_inv[v1] * g1Tan.dot(g1Tan) + 1e-4
                    ldTan = cTan / schur
                    dvTan1 = self.m_inv[v1] * ldTan * g1Tan
                    if mu * abs(Cv) > cTan:
                        mu = 1.0
                    self.dv[v1] += mu * dvTan1

        elif dtype == 4:
            d = di.d_PP(x1, x3)
            if d < dHat:
                g1, g3 = di.g_PP(x1, x3)
                schur = self.fixed[v1] * self.m_inv[v1] * g1.dot(g1) + 1e-4
                Cv = g1.dot(self.v[v1]) + g3.dot(self.v_static[v3])
                if Cv < 0.0:
                    ld_v = -Cv / schur
                    self.dv[v1] += self.fixed[v1] * self.m_inv[v1] * ld_v * g1
                    self.nc[v1] += 1

                    vTan1 = self.v[v1] + self.fixed[v1] * self.m_inv[v1] * ld_v * g1
                    g1Tan = self.v_static[v3] - vTan1
                    cTan = 0.5 * g1Tan.dot(g1Tan)
                    schur = self.m_inv[v1] * g1Tan.dot(g1Tan) + 1e-4
                    ldTan = cTan / schur
                    dvTan1 = self.m_inv[v1] * ldTan * g1Tan
                    if mu * abs(Cv) > cTan:
                        mu = 1.0
                    self.dv[v1] += mu * dvTan1

        elif dtype == 5:
            d = di.d_PE(x1, x2, x3)
            if d < dHat:
                g1, g2, g3 = di.g_PE(x1, x2, x3)
                schur = self.m_inv[v1] * g1.dot(g1) + 1e-4
                Cv = g1.dot(self.v[v1]) + g2.dot(self.v_static[v2]) + g3.dot(self.v_static[v3])
                if Cv < 0.0:
                    ld_v = -Cv / schur
                    self.dv[v1] += self.fixed[v1] * self.m_inv[v1] * ld_v * g1
                    self.nc[v1] += 1

                    c, d = g2.norm(), g3.norm()
                    c1 = c / (c + d)
                    d1 = d / (c + d)

                    p1 = c1 * self.v_static[v2] + d1 * self.v_static[v3]

                    vTan1 = self.v[v1] + self.fixed[v1] * self.m_inv[v1] * ld_v * g1
                    g1Tan = p1 - vTan1
                    cTan = 0.5 * (p1 - vTan1).dot(p1 - vTan1)
                    schur = self.m_inv[v1] * g1Tan.dot(g1Tan) + 1e-4
                    ldTan = cTan / schur
                    dvTan1 = self.m_inv[v1] * ldTan * g1Tan
                    if mu * abs(Cv) > cTan:
                        mu = 1.0
                    self.dv[v1] += mu * dvTan1

        elif dtype == 6:
            d = di.d_PE(x2, x0, x1)
            if d < dHat:
                g2, g0, g1 = di.g_PE(x2, x0, x1)
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v1] * g1.dot(g1) + 1e-4
                Cv = g0.dot(self.v_static[v0]) + g1.dot(self.v[v1]) + g2.dot(self.v_static[v2])
                if Cv < 0.0:
                    ld_v = -Cv / schur
                    self.dv[v0] += self.fixed[v0] * self.m_inv[v0] * ld_v * g0
                    self.dv[v1] += self.fixed[v1] * self.m_inv[v1] * ld_v * g1
                    self.nc[v0] += 1
                    self.nc[v1] += 1

                    vTan0 = self.v[v0] + self.fixed[v0] * self.m_inv[v0] * ld_v * g0
                    vTan1 = self.v[v1] + self.fixed[v1] * self.m_inv[v1] * ld_v * g1
                    a, b = g0.norm(), g1.norm()

                    a1 = a / (a + b)
                    b1 = b / (a + b)
                    p0 = a1 * vTan0 + b1 * vTan1

                    g0Tan = a1 * (self.v_static[v2] - p0)
                    g1Tan = b1 * (self.v_static[v2] - p0)
                    cTan = 0.5 * (self.v_static[v2] - p0).dot(self.v_static[v2] - p0)
                    schur = self.m_inv[v0] * g0Tan.dot(g0Tan) + self.m_inv[v1] * g1Tan.dot(g1Tan) + 1e-4
                    ldTan = cTan / schur
                    dvTan0 = self.m_inv[v0] * ldTan * g0Tan
                    dvTan1 = self.m_inv[v1] * ldTan * g1Tan
                    if mu * abs(Cv) > cTan:
                        mu = 1.0

                    self.dv[v0] += mu * dvTan0
                    self.dv[v1] += mu * dvTan1

        elif dtype == 7:
            d = di.d_PE(x3, x0, x1)
            if d < dHat:
                g3, g0, g1 = di.g_PE(x3, x0, x1)
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v1] * g1.dot(g1) + 1e-4
                Cv = g0.dot(self.v_static[v0]) + g1.dot(self.v[v1]) + g3.dot(self.v_static[v3])
                if Cv < 0.0:
                    ld_v = -Cv / schur

                    self.dv[v0] += self.fixed[v0] * self.m_inv[v0] * ld_v * g0
                    self.dv[v1] += self.fixed[v1] * self.m_inv[v1] * ld_v * g1

                    self.nc[v0] += 1
                    self.nc[v1] += 1

                    vTan0 = self.v[v0] + self.fixed[v0] * self.m_inv[v0] * ld_v * g0
                    vTan1 = self.v[v1] + self.fixed[v1] * self.m_inv[v1] * ld_v * g1
                    a, b = g0.norm(), g1.norm()

                    a1 = a / (a + b)
                    b1 = b / (a + b)
                    p0 = a1 * vTan0 + b1 * vTan1

                    g0Tan = a1 * (self.v_static[v3] - p0)
                    g1Tan = b1 * (self.v_static[v3] - p0)
                    cTan = 0.5 * (self.v_static[v3] - p0).dot(self.v_static[v3] - p0)
                    schur = self.m_inv[v0] * g0Tan.dot(g0Tan) + self.m_inv[v1] * g1Tan.dot(g1Tan) + 1e-4
                    ldTan = cTan / schur
                    dvTan0 = self.m_inv[v0] * ldTan * g0Tan
                    dvTan1 = self.m_inv[v1] * ldTan * g1Tan

                    if mu * abs(Cv) > cTan:
                        mu = 1.0

                    self.dv[v0] += mu * dvTan0
                    self.dv[v1] += mu * dvTan1

        if dtype == 8:
            d = di.d_EE(x0, x1, x2, x3)
            if d < dHat:
                g0, g1, g2, g3 = di.g_EE(x0, x1, x2, x3)
                schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v1] * g1.dot(g1) + 1e-4
                Cv = g0.dot(self.v_static[v0]) + g1.dot(self.v[v1]) + g3.dot(self.v_static[v3])
                if Cv < 0.0:
                    ld_v = -Cv / schur

                    self.dv[v0] += self.fixed[v0] * self.m_inv[v0] * ld_v * g0
                    self.dv[v1] += self.fixed[v1] * self.m_inv[v1] * ld_v * g1

                    self.nc[v0] += 1
                    self.nc[v1] += 1

                    vTan0 = self.v[v0] + self.fixed[v0] * self.m_inv[v0] * ld_v * g0
                    vTan1 = self.v[v1] + self.fixed[v1] * self.m_inv[v1] * ld_v * g1

                    a, b, c, d = g0.norm(), g1.norm(), g2.norm(), g3.norm()

                    a1 = a / (a + b)
                    b1 = b / (a + b)
                    c1 = c / (c + d)
                    d1 = d / (c + d)

                    p0 = a1 * vTan0 + b1 * vTan1
                    p1 = c1 * self.v_static[v2] + d1 * self.v_static[v3]

                    g0Tan = a1 * (p1 - p0)
                    g1Tan = b1 * (p1 - p0)
                    cTan = 0.5 * (p1 - p0).dot(p1 - p0)
                    schur = self.m_inv[v0] * g0Tan.dot(g0Tan) + self.m_inv[v1] * g1Tan.dot(g1Tan) + 1e-4
                    ldTan = cTan / schur
                    dvTan0 = self.m_inv[v0] * ldTan * g0Tan
                    dvTan1 = self.m_inv[v1] * ldTan * g1Tan

                    if mu * abs(Cv) > cTan:
                        mu = 1.0

                    self.dv[v0] += mu * dvTan0
                    self.dv[v1] += mu * dvTan1


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
        d = dHat
        g0, g1, g2, g3 = ti.math.vec3(0.0), ti.math.vec3(0.0), ti.math.vec3(0.0), ti.math.vec3(0.0)
        schur = 0.0

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

        elif dtype == 8:
            x01 = x0 - x1
            x23 = x2 - x3

            metric_para_EE = x01.cross(x23).norm()
            if metric_para_EE > 1e-6:
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

        if d < dHat and self.ee_dynamic_pair_num[ei0] < self.ee_dynamic_pair_cache_size:
            self.ee_dynamic_pair[ei0, self.ee_dynamic_pair_num[ei0], 0] = ei1
            self.ee_dynamic_pair[ei0, self.ee_dynamic_pair_num[ei0], 1] = dtype
            self.ee_dynamic_pair_g[ei0, self.ee_dynamic_pair_num[ei0], 0] = g0
            self.ee_dynamic_pair_g[ei0, self.ee_dynamic_pair_num[ei0], 1] = g1
            self.ee_dynamic_pair_g[ei0, self.ee_dynamic_pair_num[ei0], 2] = g2
            self.ee_dynamic_pair_g[ei0, self.ee_dynamic_pair_num[ei0], 3] = g3
            self.ee_dynamic_pair_schur[ei0, self.ee_dynamic_pair_num[ei0]] = schur
            self.ee_dynamic_pair_num[ei0] += 1


    @ti.func
    def solve_collision_ee_dynamic_v(self, ei0, ei1, dtype, g0, g1, g2, g3, schur, mu):

        v0 = self.edge_indices_dynamic[2 * ei0 + 0]
        v1 = self.edge_indices_dynamic[2 * ei0 + 1]

        v2 = self.edge_indices_dynamic[2 * ei1 + 0]
        v3 = self.edge_indices_dynamic[2 * ei1 + 1]

        dvn = g0.dot(self.v[v0]) + g1.dot(self.v[v1]) + g2.dot(self.v[v2]) + g3.dot(self.v[v3])
        if dvn < 0.0:
            ld = dvn / schur

            if dtype == 0:

                self.dv[v0] -= self.fixed[v0] * self.m_inv[v0] * g0 * ld
                self.dv[v2] -= self.fixed[v2] * self.m_inv[v2] * g2 * ld

                self.nc[v0] += 1
                self.nc[v2] += 1

            elif dtype == 1:

                self.dv[v0] -= self.fixed[v0] * self.m_inv[v0] * g0 * ld
                self.dv[v3] -= self.fixed[v3] * self.m_inv[v3] * g3 * ld

                self.nc[v0] += 1
                self.nc[v3] += 1

            elif dtype == 2:

                self.dv[v0] -= self.fixed[v0] * self.m_inv[v0] * g0 * ld
                self.dv[v2] -= self.fixed[v2] * self.m_inv[v2] * g2 * ld
                self.dv[v3] -= self.fixed[v3] * self.m_inv[v3] * g3 * ld

                self.nc[v0] += 1
                self.nc[v2] += 1
                self.nc[v3] += 1

            elif dtype == 3:

                self.dv[v1] -= self.fixed[v1] * self.m_inv[v1] * g1 * ld
                self.dv[v2] -= self.fixed[v2] * self.m_inv[v2] * g2 * ld

                self.nc[v1] += 1
                self.nc[v2] += 1

            elif dtype == 4:

                self.dv[v1] -= self.fixed[v1] * self.m_inv[v1] * g1 * ld
                self.dv[v3] -= self.fixed[v3] * self.m_inv[v3] * g3 * ld

                self.nc[v1] += 1
                self.nc[v3] += 1

            elif dtype == 5:

                self.dv[v1] -= self.fixed[v1] * self.m_inv[v1] * g1 * ld
                self.dv[v2] -= self.fixed[v2] * self.m_inv[v2] * g2 * ld
                self.dv[v3] -= self.fixed[v3] * self.m_inv[v3] * g3 * ld

                self.nc[v1] += 1
                self.nc[v2] += 1
                self.nc[v3] += 1

            elif dtype == 6:

                self.dv[v0] -= self.fixed[v0] * self.m_inv[v0] * g0 * ld
                self.dv[v1] -= self.fixed[v1] * self.m_inv[v1] * g1 * ld
                self.dv[v2] -= self.fixed[v2] * self.m_inv[v2] * g2 * ld

                self.nc[v0] += 1
                self.nc[v1] += 1
                self.nc[v2] += 1

            elif dtype == 7:

                self.dv[v0] -= self.fixed[v0] * self.m_inv[v0] * g0 * ld
                self.dv[v1] -= self.fixed[v1] * self.m_inv[v1] * g1 * ld
                self.dv[v3] -= self.fixed[v3] * self.m_inv[v3] * g3 * ld

                self.nc[v0] += 1
                self.nc[v1] += 1
                self.nc[v3] += 1

            elif dtype == 8:

                self.dv[v0] -= self.fixed[v0] * self.m_inv[v0] * g0 * ld
                self.dv[v1] -= self.fixed[v1] * self.m_inv[v1] * g1 * ld
                self.dv[v2] -= self.fixed[v2] * self.m_inv[v2] * g2 * ld
                self.dv[v3] -= self.fixed[v3] * self.m_inv[v3] * g3 * ld

                self.nc[v0] += 1
                self.nc[v1] += 1
                self.nc[v2] += 1
                self.nc[v3] += 1

    @ti.kernel
    def solve_spring_constraints_x(self, YM: ti.float32,  strain_limit: ti.float32):
        for ei in range(self.offset_spring):
            v0, v1 = self.edge_indices_dynamic[2 * ei + 0], self.edge_indices_dynamic[2 * ei + 1]
            x0, x1 = self.y[v0], self.y[v1]
            l0 = self.l0[ei]
            x10 = x0 - x1
            lij = x10.norm()

            C = (lij - l0)
            nabla_C = x10.normalized()
            alpha = 1.0 / (YM * self.dt[0] * self.dt[0])
            schur = (self.fixed[v0] * self.m_inv[v0] + self.fixed[v1] * self.m_inv[v1]) * nabla_C.dot(nabla_C) + alpha
            ld = C / schur

            self.dx[v0] -= self.fixed[v0] * self.m_inv[v0] * ld * nabla_C
            self.dx[v1] += self.fixed[v1] * self.m_inv[v1] * ld * nabla_C
            self.nc[v0] += 1
            self.nc[v1] += 1
            # strain = lij / l0 - 1.0
            # if strain > strain_limit:
            #     self.spring_cache[ei] = 1
            #     self.spring_cache_schur[ei] = schur
            #     self.spring_cache_g[ei] = nabla_C
            # else:
            #     self.spring_cache[ei] = 0

    @ti.kernel
    def solve_spring_constraints_x_test(self, YM: ti.float32, strain_limit: ti.float32):

        for e in self.mesh_dy.edges:
            l0 = e.l0
            x10 = e.verts[0].x - e.verts[1].x
            lij = x10.norm()

            C = (lij - l0)
            nabla_C = x10.normalized()
            schur = (e.verts[0].fixed * e.verts[0].m_inv + e.verts[1].fixed * e.verts[1].m_inv) * nabla_C.dot(nabla_C)
            ld = C / schur

            e.verts[0].dx -= e.verts[0].fixed * e.verts[0].m_inv * ld * nabla_C
            e.verts[1].dx += e.verts[1].fixed * e.verts[1].m_inv * ld * nabla_C
            e.verts[0].nc += 1.0
            e.verts[1].nc += 1.0

    @ti.kernel
    def solve_spring_constraints_v(self):
        s = 0
        for ei in range(self.offset_spring):
            if self.spring_cache[ei] > 0:
                ti.atomic_add(s, 1)
                v0, v1 = self.edge_indices_dynamic[2 * ei + 0], self.edge_indices_dynamic[2 * ei + 1]
                nablaC = self.spring_cache_g[ei]
                Cv = (self.v[v0] - self.v[v1]).dot(nablaC)
                if Cv > 0.0:
                    ld_v = -Cv / self.spring_cache_schur[ei]
                    self.dv[v0] += self.fixed[v0] * self.m_inv[v0] * nablaC * ld_v
                    self.dv[v1] -= self.fixed[v1] * self.m_inv[v1] * nablaC * ld_v
                    self.nc[v0] += 1
                    self.nc[v1] += 1

        # print(s / self.offset_spring)

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

        # for v in self.mesh_dy.verts:
        #     if v.y[1] < 0.0:
        #         v.y[1] = 0.0

        for vi_d in range(self.max_num_verts_dynamic):
            for fi_s in range(self.max_num_faces_static):
                self.solve_collision_vt_static_x(vi_d, fi_s, d)


        # for fi_d in range(self.max_num_faces_dynamic):
        #     for vi_s in range(self.max_num_verts_static):
        #         self.solve_collision_tv_static_x(fi_d, vi_s, d)
        #
        # for fi_d in range(self.max_num_faces_dynamic):
        #     for vi_d in range(self.max_num_verts_dynamic):
        #         if self.is_in_face(vi_d, fi_d) != True:
        #             self.solve_collision_tv_dynamic_x(fi_d, vi_d, d)
        # #
        # for ei_d in range(self.max_num_edges_dynamic):
        #     for ei_s in range(self.max_num_edges_static):
        #         self.solve_collision_ee_static_x(ei_d, ei_s, d)

        # for ei_d in range(self.max_num_edges_dynamic):
        #     for ej_d in range(self.max_num_edges_dynamic):
        #         if self.share_vertex(ei_d, ej_d) != True and ei_d != ej_d:
        #             self.solve_collision_ee_dynamic_x(ei_d, ej_d, d)


        # for ei_d in range(self.max_num_edges_dynamic):
        #     for ei_s in range(self.max_num_edges_static):
        #         self.solve_collision_ee_static_x(ei_d, ei_s, d)

        # # #
        # for i in range(self.max_num_edges_dynamic * self.max_num_edges_static):
        #     ei_d = i // self.max_num_edges_static
        #     ei_s = i % self.max_num_edges_static
        #     self.solve_collision_ee_static_x(ei_d, ei_s, d)

        # for ei in range(self.max_num_edges_dynamic):
        #     for ei_d in range(self.max_num_edges_dynamic):
        #         if ei != ei_d and self.share_vertex(ei, ei_d) != True:
        #             self.solve_collision_ee_dynamic_x(ei, ei_d, d)


    @ti.kernel
    def solve_collision_constraints_v(self):

        mu = self.friction_coeff[0]
        d = self.dHat[0]

        for vi_d in range(self.max_num_verts_dynamic):
            for j in range(self.vt_static_pair_num[vi_d]):
                fi_s, dtype = self.vt_static_pair[vi_d, j, 0], self.vt_static_pair[vi_d, j, 1]
                g0, g1, g2, g3 = self.vt_static_pair_g[vi_d, j, 0],  self.vt_static_pair_g[vi_d, j, 1],  self.vt_static_pair_g[vi_d, j, 2],  self.vt_static_pair_g[vi_d, j, 3]
                schur = self.vt_static_pair_schur[vi_d, j]
                self.solve_collision_vt_static_v(vi_d, fi_s, dtype, g0, g1, g2, g3, schur, mu)

        for fi_d in range(self.max_num_faces_dynamic):
            for j in range(self.tv_static_pair_num[fi_d]):
                vi_s, dtype = self.tv_static_pair[fi_d, j, 0], self.tv_static_pair[fi_d, j, 1]
                g0, g1, g2, g3 = self.tv_static_pair_g[fi_d, j, 0],  self.tv_static_pair_g[fi_d, j, 1],  self.tv_static_pair_g[fi_d, j, 2],  self.tv_static_pair_g[fi_d, j, 3]
                schur = self.tv_static_pair_schur[fi_d, j]
                self.solve_collision_tv_static_v(fi_d, vi_s, dtype, g0, g1, g2, g3, schur, mu)

            for j in range(self.tv_dynamic_pair_num[fi_d]):
                vi_s, dtype = self.tv_dynamic_pair[fi_d, j, 0], self.tv_dynamic_pair[fi_d, j, 1]
                g0, g1, g2, g3 = self.tv_dynamic_pair_g[fi_d, j, 0], self.tv_dynamic_pair_g[fi_d, j, 1], self.tv_dynamic_pair_g[fi_d, j, 2], self.tv_dynamic_pair_g[fi_d, j, 3]
                schur = self.tv_dynamic_pair_schur[fi_d, j]
                self.solve_collision_tv_dynamic_v(fi_d, vi_s, dtype, g0, g1, g2, g3, schur, mu)

        for ei_d in range(self.max_num_edges_dynamic):
            for j in range(self.ee_static_pair_num[ei_d]):
                ei_s, dtype = self.ee_static_pair[ei_d, j, 0], self.ee_static_pair[ei_d, j, 1]
                g0, g1, g2, g3 = self.ee_static_pair_g[ei_d, j, 0], self.ee_static_pair_g[ei_d, j, 1], self.ee_static_pair_g[ei_d, j, 2], self.ee_static_pair_g[ei_d, j, 3]
                schur = self.ee_static_pair_schur[ei_d, j]
                self.solve_collision_ee_static_v(ei_d, ei_s, d, mu)

            for j in range(self.ee_dynamic_pair_num[ei_d]):
                ej_d, dtype = self.ee_dynamic_pair[ei_d, j, 0], self.ee_dynamic_pair[ei_d, j, 1]
                g0, g1, g2, g3 = self.ee_dynamic_pair_g[ei_d, j, 0], self.ee_dynamic_pair_g[ei_d, j, 1], self.ee_dynamic_pair_g[ei_d, j, 2], self.ee_dynamic_pair_g[ei_d, j, 3]
                schur = self.ee_dynamic_pair_schur[ei_d, j]
                self.solve_collision_ee_dynamic_v(ei_d, ej_d, dtype, g0, g1, g2, g3, schur, mu)


    # @ti.kernel
    # def solve_fem_constraints_x(self, YM: ti.f32, PR: ti.f32):
    #
    #     mu = YM / (2.0 * (1.0 + PR))
    #     la = (YM * PR) / ((1.0 + PR) * (1.0 - 2.0 * PR))
    #
    #     for tid in range(self.max_num_tetra_dynamic):
    #
    #         v0 = self.tetra_indices_dynamic[4 * tid + 0]
    #         v1 = self.tetra_indices_dynamic[4 * tid + 1]
    #         v2 = self.tetra_indices_dynamic[4 * tid + 2]
    #         v3 = self.tetra_indices_dynamic[4 * tid + 3]
    #
    #         x0, x1, x2, x3 = self.y[v0], self.y[v1], self.y[v2], self.y[v3]
    #
    #         x30 = x0 - x3
    #         x31 = x1 - x3
    #         x32 = x2 - x3
    #
    #         Ds = ti.Matrix.cols([x30, x31, x32])
    #
    #         F = Ds @ self.Dm_inv[tid]
    #         U, sig, V = ti.svd(F)
    #         R = U @ V.transpose()
    #
    #         # if U.determinant() < 0:
    #         #     for i in ti.static(range(3)): U[i, 2] *= -1
    #         #     sig[2, 2] = -sig[2, 2]
    #         # if V.determinant() < 0:
    #         #     for i in ti.static(range(3)): V[i, 2] *= -1
    #         #     sig[2, 2] = -sig[2, 2]
    #
    #         H = 2.0 * mu * self.V0[tid] * (F - R) @ self.Dm_inv[tid].transpose()
    #
    #         C = mu * self.V0[tid] * (ti.pow(sig[0, 0] - 1.0, 2) + ti.pow(sig[1, 1] - 1.0, 2) + ti.pow(sig[2, 2] - 1.0, 2))
    #
    #         g0 = ti.Vector([H[j, 0] for j in ti.static(range(3))])
    #         g1 = ti.Vector([H[j, 1] for j in ti.static(range(3))])
    #         g2 = ti.Vector([H[j, 2] for j in ti.static(range(3))])
    #         g3 = -(g0 + g1 + g2)
    #
    #         alpha = 1.0 / (mu * self.V0[tid] * self.dt[0] * self.dt[0])
    #         schur = (self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) +
    #                  self.fixed[v1] * self.m_inv[v1] * g1.dot(g1) +
    #                  self.fixed[v2] * self.m_inv[v2] * g2.dot(g2) +
    #                  self.fixed[v3] * self.m_inv[v3] * g3.dot(g3) + 1e-3)
    #
    #
    #
    #         ld = -C / schur
    #
    #         self.dx[v0] += self.fixed[v0] * self.m_inv[v0] * g0 * ld
    #         self.dx[v1] += self.fixed[v1] * self.m_inv[v1] * g1 * ld
    #         self.dx[v2] += self.fixed[v2] * self.m_inv[v2] * g2 * ld
    #         self.dx[v3] += self.fixed[v3] * self.m_inv[v3] * g3 * ld
    #
    #         self.nc[v0] += 1
    #         self.nc[v1] += 1
    #         self.nc[v2] += 1
    #         self.nc[v3] += 1
    #
    #
    #         J = sig[0, 0] * sig[1, 1] * sig[2, 2]
    #         J = F.trace()
    #         if sig[0, 0] * sig[1, 1] * sig[2, 2] < 0.0:
    #             ti.atomic_add(self.num_inverted_elements[0], 1)
    #
    #         gamma = 3.
    #         C_vol = 0.5 * la * self.V0[tid] * (J - gamma) * (J - gamma)
    #         H_vol = la * self.V0[tid] * (J - gamma) * self.Dm_inv[tid].transpose()
    #
    #         nabla_C_vol_0 = ti.Vector([H_vol[j, 0] for j in ti.static(range(3))])
    #         nabla_C_vol_1 = ti.Vector([H_vol[j, 1] for j in ti.static(range(3))])
    #         nabla_C_vol_2 = ti.Vector([H_vol[j, 2] for j in ti.static(range(3))])
    #         nabla_C_vol_3 = -(nabla_C_vol_0 + nabla_C_vol_1 + nabla_C_vol_2)
    #         alpha = 1.0 / (la * self.V0[tid] * self.dt[0] * self.dt[0])
    #         schur_vol = (self.fixed[v0] * self.m_inv[v0] * nabla_C_vol_0.dot(nabla_C_vol_0) +
    #                     self.fixed[v1] * self.m_inv[v1] * nabla_C_vol_1.dot(nabla_C_vol_1) +
    #                     self.fixed[v2] * self.m_inv[v2] * nabla_C_vol_2.dot(nabla_C_vol_2) +
    #                     self.fixed[v3] * self.m_inv[v3] * nabla_C_vol_3.dot(nabla_C_vol_3) + alpha)
    #
    #         ld_vol = C_vol / schur_vol
    #
    #         self.dx[v0] -= self.fixed[v0] * self.m_inv[v0] * nabla_C_vol_0 * ld_vol
    #         self.dx[v1] -= self.fixed[v1] * self.m_inv[v1] * nabla_C_vol_1 * ld_vol
    #         self.dx[v2] -= self.fixed[v2] * self.m_inv[v2] * nabla_C_vol_2 * ld_vol
    #         self.dx[v3] -= self.fixed[v3] * self.m_inv[v3] * nabla_C_vol_3 * ld_vol
    #
    #         self.nc[v0] += 1
    #         self.nc[v1] += 1
    #         self.nc[v2] += 1
    #         self.nc[v3] += 1
    #         #
    #         # ti.atomic_add(self.current_volume[0], vol)
    #
    # @ti.kernel
    # def solve_fem_constraints_v(self, YM: ti.f32, PR: ti.f32):
    #     mu = YM / (2.0 * (1.0 + PR))
    #     la = (YM * PR) / ((1.0 + PR) * (1.0 - 2.0 * PR))
    #
    #     for tid in range(self.max_num_tetra_dynamic):
    #
    #         v0 = self.tetra_indices_dynamic[4 * tid + 0]
    #         v1 = self.tetra_indices_dynamic[4 * tid + 1]
    #         v2 = self.tetra_indices_dynamic[4 * tid + 2]
    #         v3 = self.tetra_indices_dynamic[4 * tid + 3]
    #
    #         x0, x1, x2, x3 = self.y[v0], self.y[v1], self.y[v2], self.y[v3]
    #
    #         x30 = x0 - x3
    #         x31 = x1 - x3
    #         x32 = x2 - x3
    #
    #         Ds = ti.Matrix.cols([x30, x31, x32])
    #
    #         F = Ds @ self.Dm_inv[tid]
    #         U, sig, V = ti.svd(F)
    #         R = U @ V.transpose()
    #
    #         if U.determinant() < 0:
    #             for i in ti.static(range(3)): U[i, 2] *= -1
    #             sig[2, 2] = -sig[2, 2]
    #         if V.determinant() < 0:
    #             for i in ti.static(range(3)): V[i, 2] *= -1
    #             sig[2, 2] = -sig[2, 2]
    #
    #         H = 2.0 * mu * self.V0[tid] * (F - R) @ self.Dm_inv[tid].transpose()
    #
    #         g0 = ti.Vector([H[j, 0] for j in ti.static(range(3))])
    #         g1 = ti.Vector([H[j, 1] for j in ti.static(range(3))])
    #         g2 = ti.Vector([H[j, 2] for j in ti.static(range(3))])
    #         g3 = -(g0 + g1 + g2)
    #
    #         alpha = 1.0 / (mu * self.V0[tid] * self.dt[0] * self.dt[0])
    #         schur = (self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) +
    #                  self.fixed[v1] * self.m_inv[v1] * g1.dot(g1) +
    #                  self.fixed[v2] * self.m_inv[v2] * g2.dot(g2) +
    #                  self.fixed[v3] * self.m_inv[v3] * g3.dot(g3) + alpha)
    #
    #         Cv = g0.dot(self.v[v0]) + g1.dot(self.v[v1]) + g2.dot(self.v[v2]) + g3.dot(self.v[v3])
    #
    #         if Cv > 0:
    #             ld = -Cv / schur
    #
    #             self.dv[v0] += self.fixed[v0] * self.m_inv[v0] * g0 * ld
    #             self.dv[v1] += self.fixed[v1] * self.m_inv[v1] * g1 * ld
    #             self.dv[v2] += self.fixed[v2] * self.m_inv[v2] * g2 * ld
    #             self.dv[v3] += self.fixed[v3] * self.m_inv[v3] * g3 * ld
    #
    #             self.nc[v0] += 1
    #             self.nc[v1] += 1
    #             self.nc[v2] += 1
    #             self.nc[v3] += 1
    #
    #         J = sig[0, 0] * sig[1, 1] * sig[2, 2]
    #         J = F.trace()
    #         # if sig[0, 0] * sig[1, 1] * sig[2, 2] < 0.0:
    #         #     ti.atomic_add(self.num_inverted_elements[0], 1)
    #
    #         gamma = 3.0
    #         H_vol = la * self.V0[tid] * (J - gamma) * self.Dm_inv[tid].transpose()
    #
    #         nabla_C_vol_0 = ti.Vector([H_vol[j, 0] for j in ti.static(range(3))])
    #         nabla_C_vol_1 = ti.Vector([H_vol[j, 1] for j in ti.static(range(3))])
    #         nabla_C_vol_2 = ti.Vector([H_vol[j, 2] for j in ti.static(range(3))])
    #         nabla_C_vol_3 = -(nabla_C_vol_0 + nabla_C_vol_1 + nabla_C_vol_2)
    #         alpha = 1.0 / (la * self.V0[tid] * self.dt[0] * self.dt[0])
    #         schur_vol = (self.fixed[v0] * self.m_inv[v0] * nabla_C_vol_0.dot(nabla_C_vol_0) +
    #                      self.fixed[v1] * self.m_inv[v1] * nabla_C_vol_1.dot(nabla_C_vol_1) +
    #                      self.fixed[v2] * self.m_inv[v2] * nabla_C_vol_2.dot(nabla_C_vol_2) +
    #                      self.fixed[v3] * self.m_inv[v3] * nabla_C_vol_3.dot(nabla_C_vol_3) + alpha)
    #
    #         Cv = g0.dot(self.v[v0]) + g1.dot(self.v[v1]) + g2.dot(self.v[v2]) + g3.dot(self.v[v3])
    #
    #         if Cv > 0:
    #             ld_vol = -Cv / schur_vol
    #             self.dx[v0] += self.fixed[v0] * self.m_inv[v0] * nabla_C_vol_0 * ld_vol
    #             self.dx[v1] += self.fixed[v1] * self.m_inv[v1] * nabla_C_vol_1 * ld_vol
    #             self.dx[v2] += self.fixed[v2] * self.m_inv[v2] * nabla_C_vol_2 * ld_vol
    #             self.dx[v3] += self.fixed[v3] * self.m_inv[v3] * nabla_C_vol_3 * ld_vol
    #
    #             self.nc[v0] += 1
    #             self.nc[v1] += 1
    #             self.nc[v2] += 1
    #             self.nc[v3] += 1


    @ti.kernel
    def update_x_test(self, dt: ti.f32):

        for v in self.mesh_dy.verts:
            if v.id != 0:
                v.x += dt * v.v
    # @ti.kernel
    # def confine_to_boundary(self):
    #
    #     for vi in range(self.max_num_verts_dynamic):
    #
    #         if self.y[vi][0] > self.padding * self.grid_size[0]:
    #             self.y[vi][0] = self.padding * self.grid_size[0]
    #
    #         if self.y[vi][0] < -self.padding * self.grid_size[0]:
    #             self.y[vi][0] = -self.padding * self.grid_size[0]
    #
    #         if self.y[vi][1] > self.padding * self.grid_size[1]:
    #             self.y[vi][1] = self.padding * self.grid_size[1]
    #
    #         if self.y[vi][1] < -self.padding * self.grid_size[1]:
    #             self.y[vi][1] = -self.padding * self.grid_size[1]
    #
    #         if self.y[vi][2] > self.padding * self.grid_size[2]:
    #             self.y[vi][2] = self.padding * self.grid_size[2]
    #
    #         if self.y[vi][2] < -self.padding * self.grid_size[2]:
    #             self.y[vi][2] = -self.padding * self.grid_size[2]


    @ti.kernel
    def compute_velocity_test(self, dt: ti.f32):
        for v in self.mesh_dy.verts:
            v.v = (v.y - v.x) / dt

    def init_variables(self):

        # self.vt_static_pair_num.fill(0)
        # self.vt_static_pair_g.fill(0.0)
        # self.vt_static_pair_schur.fill(0.0)
        # self.vt_static_pair.fill(0)
        #
        #
        # self.tv_static_pair_num.fill(0)
        # self.tv_static_pair_g.fill(0.0)
        # self.tv_static_pair_schur.fill(0.0)
        # self.tv_static_pair.fill(0)
        #
        # self.tv_dynamic_pair_num.fill(0)
        # self.tv_dynamic_pair_g.fill(0.0)
        # self.tv_dynamic_pair_schur.fill(0.0)
        # self.tv_dynamic_pair.fill(0)
        #
        # self.ee_static_pair_num.fill(0)
        # self.ee_static_pair_g.fill(0.0)
        # self.ee_static_pair_schur.fill(0.0)
        # self.ee_static_pair.fill(0)
        #
        # self.ee_dynamic_pair_num.fill(0)
        # self.ee_dynamic_pair_g.fill(0.0)
        # self.ee_dynamic_pair_schur.fill(0.0)
        # self.ee_dynamic_pair.fill(0)
        #
        # self.num_inverted_elements.fill(0)

        self.mesh_dy.verts.dx.fill(0.0)
        self.mesh_dy.verts.nc.fill(0.0)


    def solve_constraints_x(self):

        self.init_variables()
        self.solve_spring_constraints_x_test(self.YM[0], self.strain_limit[0])

        # if self.enable_collision_handling:
        self.solve_collision_constraints_x()
        #
        # self.solve_fem_constraints_x(self.YM[0], self.PR[0])

        # self.solve_pressure_constraints_x()
        self.update_dx_test()

        # print(self.num_springs[0])
    #
    # def solve_constraints_v(self):
    #     self.dv.fill(0.0)
    #     self.nc.fill(0)
    #
    #     # self.solve_spring_constraints_v()
    #
    #     if self.enable_collision_handling:
    #         self.solve_collision_constraints_v()
    #
    #     self.solve_fem_constraints_v(self.YM[0], self.PR[0])
    #     # self.solve_pressure_constraints_v()
    #     self.update_dv()

    @ti.kernel
    def update_dx_test(self):
        for v in self.mesh_dy.verts:
            if v.nc > 0.0:
                v.y += (v.dx / v.nc)

    # @ti.kernel
    # def set_fixed_vertices(self, fixed_vertices: ti.template()):
    #     for vi in range(self.max_num_verts_dynamic):
    #         if fixed_vertices[vi] >= 1:
    #             self.fixed[vi] = 0.0
    #         else:
    #             self.fixed[vi] = 1.0
    # @ti.kernel
    # def update_dv(self):
    #     for vi in range(self.max_num_verts_dynamic):
    #         if self.nc[vi] > 0.0:
    #             self.dv[vi] = self.dv[vi] / self.nc[vi]
    #
    #         if self.m_inv[vi] > 0.0:
    #             self.v[vi] += self.fixed[vi] * self.dv[vi]

    # @ti.kernel
    # def move_static_object(self):
    #
    #     center = ti.math.vec3(0.0)
    #     for i in self.x_static:
    #         center += self.x_static[i]
    #
    #     center /= self.max_num_verts_static
    #
    #     for i in self.x_static:
    #         old = self.x_static[i]
    #         ri = self.x_static[i] - center
    #         v_4d = ti.Vector([ri[0], ri[1], ri[2], 1])
    #         rot_rad = ti.math.radians(self.obs_ang_vel[0] * self.dt[0])
    #         rv = ti.math.rotation3d(rot_rad[0], rot_rad[1], rot_rad[2]) @ v_4d
    #
    #         center += self.obs_lin_vel[0] * self.dt[0]
    #
    #         self.x_static[i] = ti.math.vec3(rv[0], rv[1], rv[2]) + center
    #         self.v_static[i] = (self.x_static[i] - old) / self.dt[0]
    #
    #     # for i in self.x_static:
    #     #     x_cur = self.x_static[i]
    #     #     offset = 30
    #     #     if self.frame[0] >= 30:
    #     #         self.v_static[i] = ti.math.vec3(0., 20.0 * ti.math.sin(30. * (self.frame[0] - offset) * self.dt[0]), 0.)
    #     #         self.x_static[i] += self.v_static[i] * self.dt[0]

    # @ti.kernel
    # def move_each_static_object(self, start_idx: ti.template(), len: ti.template(), sid: ti.template()):
    #
    #     center = ti.math.vec3(0.0)
    #     for i in ti.ndrange(len):
    #         center += self.x_static[i + start_idx]
    #
    #     center /= len
    #
    #     for i in ti.ndrange(len):
    #         old = self.x_static[i + start_idx]
    #         ri = self.x_static[i + start_idx] - center
    #         v_4d = ti.Vector([ri[0], ri[1], ri[2], 1])
    #
    #         rot_rad = ti.math.radians(self.manual_ang_vels[sid] * self.dt[0])
    #         rv = ti.math.rotation3d(rot_rad[0], rot_rad[1], rot_rad[2]) @ v_4d
    #
    #         center += self.obs_lin_vel[sid] * self.dt[0]
    #
    #         self.x_static[i + start_idx] = ti.math.vec3(rv[0], rv[1], rv[2]) + center
    #         self.v_static[i + start_idx] = (self.x_static[i + start_idx] - old) / self.dt[0]


    def forward(self, n_substeps):

        dt = self.dt[0]
        self.dt[0] = dt / n_substeps

        # ti.profiler.clear_kernel_profiler_info()
        for _ in range(n_substeps):

            self.compute_y_test(dt)

            self.solve_constraints_x()

            self.compute_velocity_test(dt)
            # self.solve_constraints_v()

            self.update_x_test(dt)

        # compute_y_result = ti.profiler.query_kernel_profiler_info(self.compute_y.__name__)
        # solve_spring_constraints_x_result = ti.profiler.query_kernel_profiler_info(self.solve_spring_constraints_x.__name__)
        # compute_velocity_result = ti.profiler.query_kernel_profiler_info(self.compute_velocity.__name__)
        # update_x_result = ti.profiler.query_kernel_profiler_info(self.update_x.__name__)
        #
        # total_1 = compute_y_result.avg + solve_spring_constraints_x_result.avg + compute_velocity_result.avg + update_x_result.avg
        #
        # compute_y_result = ti.profiler.query_kernel_profiler_info(self.compute_y_test.__name__)
        # solve_spring_constraints_x_result = ti.profiler.query_kernel_profiler_info(self.solve_spring_constraints_x_test.__name__)
        # compute_velocity_result = ti.profiler.query_kernel_profiler_info(self.compute_velocity_test.__name__)
        # update_x_result = ti.profiler.query_kernel_profiler_info(self.update_x_test.__name__)
        #
        # total_2 = compute_y_result.avg + solve_spring_constraints_x_result.avg + compute_velocity_result.avg + update_x_result.avg
        #
        # if self.frame[0] < 100:
        #     print(round(total_1 / total_2, 4))

        # self.copy_to_meshes()
        # self.copy_to_particles()

        self.dt[0] = dt
        # self.frame[0] = self.frame[0] + 1
    #
    #
    #
    # @ti.kernel
    # def random_noise(self):
    #     scale = 2.0
    #     for vi in range(self.max_num_verts_dynamic):
    #         v = ti.math.vec3(ti.random(dtype=float), ti.random(dtype=float), ti.random(dtype=float))
    #         # v.normalized()
    #         self.x[vi] += scale * v * self.dt[0]
    #
