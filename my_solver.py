import taichi as ti
import numpy as np
import meshtaichi_patcher as Patcher

import ccd as ccd
import ipc_utils as cu
import barrier_functions as barrier


@ti.data_oriented
class Solver:
    def __init__(self,
                 my_mesh,
                 static_mesh,
                 bottom,
                 min_range,
                 max_range,
                 k=1e6,
                 dt=1e-3,
                 max_iter=1000):
        self.my_mesh = my_mesh
        self.static_mesh = static_mesh
        self.grid_origin = ti.math.vec3([-4, -4, -4])
        self.grid_size = ti.math.vec3([8, 8, 8])
        self.grid_min = ti.math.vec3(min_range[0], min_range[1], min_range[2])
        self.grid_max = ti.math.vec3(max_range[0], max_range[1], max_range[2])
        self.domain_size = self.grid_size - self.grid_origin


        self.radius = 0.008
        self.grid_size = 8 * self.radius
        self.grid_num = np.ceil(self.domain_size / self.grid_size).astype(int)
        print("grid size: ", self.grid_num)
        self.padding = self.grid_size


        self.k = k
        self.dt = dt
        self.dtSq = dt ** 2
        self.max_iter = max_iter
        self.gravity = -9.81
        self.bottom = bottom
        self.id3 = ti.math.mat3([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])

        self.id2 = ti.math.mat2([[1, 0],
                                 [0, 1]])

        self.verts = self.my_mesh.mesh.verts
        self.num_verts = len(self.my_mesh.mesh.verts)
        self.edges = self.my_mesh.mesh.edges
        self.num_edges = len(self.edges)
        self.faces = self.my_mesh.mesh.faces
         
        self.num_faces = len(self.my_mesh.mesh.faces)
        self.face_indices = self.my_mesh.face_indices

        self.verts_static = self.static_mesh.mesh.verts
        self.num_verts_static = len(self.static_mesh.mesh.verts)
        self.edges_static = self.static_mesh.mesh.edges
        self.num_edges_static = len(self.edges_static)
        self.faces_static = self.static_mesh.mesh.faces
        self.face_indices_static = self.static_mesh.face_indices
        self.num_faces_static = len(self.static_mesh.mesh.faces)

        self.dHat = 1e-3

        self.contact_stiffness = 1e3
        self.damping_factor = 1e-4
        self.grid_n = 256


        self.dynamic_head = ti.field(dtype=ti.i32, shape=self.grid_n * self.grid_n * self.grid_n)
        self.dynamic_cur = ti.field(dtype=ti.i32, shape=self.grid_n * self.grid_n * self.grid_n)
        self.dynamic_tail = ti.field(dtype=ti.i32, shape=self.grid_n * self.grid_n * self.grid_n)
        self.dynamic_count = ti.field(dtype=ti.i32,
                                      shape=(self.grid_n, self.grid_n, self.grid_n),
                                      name="dynamic_count")
        self.dynamic_column_sum = ti.field(dtype=ti.i32, shape=(self.grid_n, self.grid_n), name="dynamic_column_sum")
        self.dynamic_prefix_sum = ti.field(dtype=ti.i32, shape=(self.grid_n, self.grid_n), name="dynamic_prefix_sum")
        self.dynamic_particle_id = ti.field(dtype=ti.i32, shape=self.num_verts, name="dynamic_particle_id")

        self.static_head = ti.field(dtype=ti.i32, shape=self.grid_n * self.grid_n * self.grid_n)
        self.static_cur = ti.field(dtype=ti.i32, shape=self.grid_n * self.grid_n * self.grid_n)
        self.static_tail = ti.field(dtype=ti.i32, shape=self.grid_n * self.grid_n * self.grid_n)
        self.static_count = ti.field(dtype=ti.i32,
                                     shape=(self.grid_n, self.grid_n, self.grid_n),
                                     name="static_count")
        self.static_column_sum = ti.field(dtype=ti.i32, shape=(self.grid_n, self.grid_n), name="static_column_sum")
        self.static_prefix_sum = ti.field(dtype=ti.i32, shape=(self.grid_n, self.grid_n), name="static_prefix_sum")
        self.static_particle_id = ti.field(dtype=ti.i32, shape=self.num_verts_static, name="static_particle_id")

        self.static_collision_pair = ti.field(dtype=ti.i32, shape=(self.num_verts, self.num_verts_static))
        self.self_collision_pair = ti.field(dtype=ti.i32, shape=(self.num_verts, self.num_verts))

        print(f"verts #: {len(self.my_mesh.mesh.verts)}, edges #: {len(self.my_mesh.mesh.edges)} faces #: {len(self.my_mesh.mesh.faces)}")

        # for PCG
        self.b = ti.Vector.field(3, dtype=ti.f32, shape=self.num_verts)
        self.r = ti.Vector.field(3, dtype=ti.f32, shape=self.num_verts)
        self.p = ti.Vector.field(3, dtype=ti.f32, shape=self.num_verts)
        self.Ap = ti.Vector.field(3, dtype=ti.f32, shape=self.num_verts)
        self.z = ti.Vector.field(3, dtype=ti.f32, shape=self.num_verts)

        self.grid_particles_num = ti.field(int, shape=int(self.grid_num[0]*self.grid_num[1]*self.grid_num[2]))
        self.grid_particles_num_temp = ti.field(int, shape=int(self.grid_num[0]*self.grid_num[1]*self.grid_num[2]))
        self.prefix_sum_executor = ti.algorithms.PrefixSumExecutor(self.grid_particles_num.shape[0])

        self.max_num_verts = self.num_verts
        self.grid_ids = ti.field(int, shape=self.max_num_verts)
        self.grid_ids_buffer = ti.field(int, shape=self.max_num_verts)
        self.grid_ids_new = ti.field(int, shape=self.max_num_verts)
        self.cur2org = ti.field(int, shape=self.max_num_verts)

        # self.object_id_buffer = ti.field(dtype=int, shape=self.num_verts)
        # self.object_id = ti.field(dtype=int, shape=self.num_verts)

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
        #     self.grid_ids_buffer[new_index] = self.grid_ids[I]
        #     self.object_id_buffer[new_index] = self.object_id[I]
        #
        # for I in range(self.num_verts):
        #     self.grid_ids[I] = self.grid_ids_buffer[I]
        #     self.object_id[I] = self.object_id_buffer[I]



    @ti.func
    def flatten_grid_index(self, grid_index):
        return grid_index[0] * self.grid_num[1] * self.grid_num[2] + grid_index[1] * self.grid_num[2] + grid_index[2]

    @ti.func
    def get_flatten_grid_index(self, pos):
        return self.flatten_grid_index(self.pos_to_index(pos))

    @ti.kernel
    def update_grid_id(self):
        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num[I] = 0

        for v in self.verts:
            grid_index = self.get_flatten_grid_index(v.x)
            self.grid_ids[v.id] = grid_index
            ti.atomic_add(self.grid_particles_num[grid_index], 1)

        for I in ti.grouped(self.grid_particles_num):
            self.grid_particles_num_temp[I] = self.grid_particles_num[I]

    def initialize_particle_system(self):
        self.update_grid_id()
        self.prefix_sum_executor.run(self.grid_particles_num)
        self.counting_sort()

    @ti.func
    def pos_to_index(self, pos):
        return ( (pos-self.grid_origin) / self.grid_size ).cast(int)

    @ti.func
    def for_all_neighbors(self, p_i, task: ti.template()):
        center_cell = self.pos_to_index(self.verts.x_k[p_i])
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * 3)):
            grid_index = self.flatten_grid_index(center_cell + offset)
            for p_j in range(self.grid_particles_num[ti.max(0, grid_index-1)], self.grid_particles_num[grid_index]):
                p_j_cur = self.cur2org[p_j]
                if p_i != p_j_cur:
                    task(p_i, p_j_cur)



    @ti.kernel
    def reset_kernel(self):
        for e in self.edges:
            e.verts[0].deg += 1
            e.verts[1].deg += 1
            # h = ti.math.mat2([[e.verts[0].h, 0], [0, e.verts[1].h]])
            # e.hinv = h.inverse()

        # for v in self.verts:
        #     # a = v.edges[0].id
        #     print(f'{v.id}: ')
        #     for ni in range(v.deg):
        #         print(v.edges[ni].id)


    def reset(self):

        self.verts.x.copy_from(self.verts.x0)
        self.verts.v.fill(0.0)
        self.verts.deg.fill(0)

        self.reset_kernel()

    @ti.func
    def aabb_intersect(self, a_min: ti.math.vec3, a_max: ti.math.vec3,
                       b_min: ti.math.vec3, b_max: ti.math.vec3):

        return  a_min[0] <= b_max[0] and \
                a_max[0] >= b_min[0] and \
                a_min[1] <= b_max[1] and \
                a_max[1] >= b_min[1] and \
                a_min[2] <= b_max[2] and \
                a_max[2] >= b_min[2]


    @ti.kernel
    def computeVtemp(self):
        for v in self.verts:
            if v.id == 0 or v.id == 2:
                v.v = ti.math.vec3(0.0, 0.0, 0.0)
            else:
                v.v += (v.f_ext / v.m) * self.dt

    @ti.kernel
    def add(self, ans: ti.template(), a: ti.template(), k: ti.f32, b: ti.template()):
        for i in ans:
            ans[i] = a[i] + k * b[i]

    @ti.kernel
    def dot(self, a: ti.template(), b: ti.template()) -> ti.f32:
        ans = 0.0
        ti.loop_config(block_dim=32)
        for i in a: ans += a[i].dot(b[i])
        return ans

    @ti.func
    def abT(self, a: ti.math.vec3, b: ti.math.vec3) -> ti.math.mat3:

        abT = ti.math.mat3([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

        abT[0, 0] = a[0] * b[0]
        abT[0, 1] = a[0] * b[1]
        abT[0, 2] = a[0] * b[2]

        abT[1, 0] = a[1] * b[0]
        abT[1, 1] = a[1] * b[1]
        abT[1, 2] = a[1] * b[2]

        abT[2, 0] = a[2] * b[0]
        abT[2, 1] = a[2] * b[1]
        abT[2, 2] = a[2] * b[2]

        return abT

    @ti.kernel
    def computeY(self):
        for v in self.verts:
            v.y = v.x + v.v * self.dt

        # center = ti.math.vec3(0.5, 0.5, 0.5)
        # rad = 0.4
        # for v in self.verts:
        #     dist = (v.y - center).norm()
        #     nor = (v.y - center).normalized(1e-12)
        #     if dist < rad:
        #         v.y = center + rad * nor

    @ti.kernel
    def  computeNextState(self):

        for v in self.verts:
            v.v = (1.0 - self.damping_factor) * (v.x_k - v.x) / self.dt
            v.x = v.x_k

    @ti.kernel
    def evaluateMomentumConstraint(self):
        for v in self.verts:
            v.g = v.m * (v.x_k - v.y)
            v.h = v.m

    @ti.kernel
    def evaluate_gradient_and_hessian(self):
        # self.candidatesPC.deactivate()
        coef = self.dtSq * 1e7

        xij = self.verts.x_k[0] - self.verts.x0[0]
        grad = coef * xij
        self.verts.g[0] -= grad
        self.verts.h[0] += coef

        xij = self.verts.x_k[2] - self.verts.x0[2]
        grad = coef * xij
        self.verts.g[2] -= grad
        self.verts.h[2] += coef

        for e in self.edges:
            xij = e.verts[0].x_k - e.verts[1].x_k
            lij = xij.norm()
            grad = coef * (xij - (e.l0/lij) * xij)
            e.verts[0].g -= grad
            e.verts[1].g += grad
            e.verts[0].h += coef
            e.verts[1].h += coef

            hij = coef * (self.id3 - e.l0 / lij * (self.id3 - (self.abT(xij, xij)) / (lij ** 2)))
            # hij = coef * (self.id3)
            U, sig, V = ti.svd(hij)

            for i in range(3):
                if sig[i, i] < 1e-6:
                    sig[i, i] = 1e-6

            hij = U @ sig @ V.transpose()
            e.hij = hij

        for e in self.edges:
            h = ti.math.mat2([[e.verts[0].h, 0],
                              [0, e.verts[1].h]]) + \
                coef * ti.math.mat2([[0, -1],
                                     [-1, 0]])
            e.hinv = h.inverse()

    @ti.kernel
    def step_forward(self):
        for v in self.verts:
            if v.id == 0 or v.id == 2:
                v.x_k = v.x_k
            else:
                v.x_k += v.dx


    @ti.kernel
    def set_grid_particles(self):
        ##### dynamic particles indexing #####
        self.dynamic_count.fill(0)

        for i in range(self.num_verts):
            dynamic_vert_idx = self.compute_grid_index(self.verts.x_k[i])
            self.dynamic_count[dynamic_vert_idx] += 1

        self.dynamic_column_sum.fill(0)
        # kernel comunicate with global variable ???? this is a bit amazing
        for i, j, k in ti.ndrange(self.grid_n, self.grid_n, self.grid_n):
            ti.atomic_add(self.dynamic_column_sum[i, j], self.dynamic_count[i, j, k])

        # this is because memory mapping can be out of order
        _dynamic_prefix_sum_cur = 0

        for i, j in ti.ndrange(self.grid_n, self.grid_n):
            self.dynamic_prefix_sum[i, j] = ti.atomic_add(_dynamic_prefix_sum_cur, self.dynamic_column_sum[i, j])

        for i, j, k in ti.ndrange(self.grid_n, self.grid_n, self.grid_n):
            # we cannot visit prefix_sum[i,j] in this loop
            pre = ti.atomic_add(self.dynamic_prefix_sum[i, j], self.dynamic_count[i, j, k])
            linear_idx = i * self.grid_n * self.grid_n + j * self.grid_n + k
            self.dynamic_head[linear_idx] = pre
            self.dynamic_cur[linear_idx] = self.dynamic_head[linear_idx]
            # only pre pointer is useable
            self.dynamic_tail[linear_idx] = pre + self.dynamic_count[i, j, k]

        for i in range(self.num_verts):
            dynamic_vert_idx = self.compute_grid_index(self.verts.x_k[i])
            linear_idx = dynamic_vert_idx[0] * self.grid_n * self.grid_n + dynamic_vert_idx[1] * self.grid_n + dynamic_vert_idx[2]
            location = ti.atomic_add(self.dynamic_cur[linear_idx], 1)
            self.dynamic_particle_id[location] = i


        ##### static particles indexing #####
        self.static_count.fill(0)

        for i in range(self.num_verts_static):
            static_vert_idx = self.compute_grid_index(self.verts_static.x[i])
            self.static_count[static_vert_idx] += 1

        self.static_column_sum.fill(0)
        # kernel comunicate with global variable ???? this is a bit amazing
        for i, j, k in ti.ndrange(self.grid_n, self.grid_n, self.grid_n):
            ti.atomic_add(self.static_column_sum[i, j], self.static_count[i, j, k])

        # this is because memory mapping can be out of order
        _static_prefix_sum_cur = 0

        for i, j in ti.ndrange(self.grid_n, self.grid_n):
            self.static_prefix_sum[i, j] = ti.atomic_add(_static_prefix_sum_cur, self.static_column_sum[i, j])

        for i, j, k in ti.ndrange(self.grid_n, self.grid_n, self.grid_n):
            # we cannot visit prefix_sum[i,j] in this loop
            pre = ti.atomic_add(self.static_prefix_sum[i, j], self.static_count[i, j, k])
            linear_idx = i * self.grid_n * self.grid_n + j * self.grid_n + k
            self.static_head[linear_idx] = pre
            self.static_cur[linear_idx] = self.static_head[linear_idx]
            # only pre pointer is useable
            self.static_tail[linear_idx] = pre + self.static_count[i, j, k]

        for i in range(self.num_verts_static):
            static_vert_idx = self.compute_grid_index(self.verts_static.x[i])
            linear_idx = static_vert_idx[0] * self.grid_n * self.grid_n + static_vert_idx[1] * self.grid_n + static_vert_idx[2]
            location = ti.atomic_add(self.static_cur[linear_idx], 1)
            self.static_particle_id[location] = i


    @ti.kernel
    def handle_contacts(self):

        for v in self.verts:
            self.for_all_neighbors(v.id, self.resolve_self)

        static_collision_count = 0
        self_collision_count = 0
        loop_count = 0
        # for v in self.verts:
        #     for sv in range(self.num_verts_static):
        #         static_collision_count += 1
        #         loop_count += 1
        #         self.resolve(v.id, sv)
        #
        # for v in self.verts:
        #     for sv in range(v.id + 1, self.num_verts):
        #         self_collision_count += 1
        #         loop_count += 1
        #         self.resolve_self(v.id, sv)

        # self.static_collision_pair.fill(0)
        # self.self_collision_pair.fill(0)
        # loop_count = 0
        # for v in self.verts:
        #     vIdx = self.compute_grid_index(v.x_k)
        #     if vIdx[0] < 0 or vIdx[1] < 0 or vIdx[2] < 0:
        #         continue
        #
        #     x_begin = ti.max(vIdx[0] - 1, 0)
        #     x_end = ti.min(vIdx[0] + 2, self.grid_n)
        #
        #     y_begin = ti.max(vIdx[1] - 1, 0)
        #     y_end = ti.min(vIdx[1] + 2, self.grid_n)
        #
        #     z_begin = ti.max(vIdx[2] - 1, 0)
        #     z_end = ti.min(vIdx[2] + 2, self.grid_n)
        #
        #     for neigh_i, neigh_j, neigh_k in ti.ndrange((x_begin, x_end), (y_begin, y_end), (z_begin, z_end)):
        #         # on split plane
        #         if neigh_k == vIdx[2] and (neigh_i + neigh_j) > (vIdx[0] + vIdx[1]) and neigh_i <= vIdx[0]:
        #             continue
        #
        #         neigh_linear_idx = neigh_i * self.grid_n * self.grid_n + neigh_j * self.grid_n + neigh_k
        #         for p_idx in range(self.static_head[neigh_linear_idx],
        #                               self.static_tail[neigh_linear_idx]):
        #               sv = self.static_particle_id[p_idx]
        #               static_collision_count += 1
        #               self.resolve(v.id, sv)
        #
        #         for p_idx in range(self.dynamic_head[neigh_linear_idx],
        #                            self.dynamic_tail[neigh_linear_idx]):
        #                 dv = self.dynamic_particle_id[p_idx]
        #                 if v.id >= dv:
        #                     continue
        #                 self_collision_count += 1
        #                 self.resolve_self(v.id, dv)


        # for i, j, k in ti.ndrange(self.grid_n, self.grid_n, self.grid_n):
        #     # collision with static mesh
        #     if self.grid_static_verts[i, j, k].length() != 0:


        # print(f"static collision #: {static_collision_count}, self collision #: {self_collision_count}", f"loop count: {loop_count}")

        for v in self.verts:
            if v.nc > 0:
                v.x_k = v.p / v.nc


    @ti.func
    def resolve(self, i, j):
        dx = self.verts.x_k[i] - self.verts_static.x[j]
        d = dx.norm()

        if d < 2.0 * self.radius:  # in contact
            normal = dx / d
            p = self.verts_static.x[j] + 2.0 * self.radius * normal
            v = (p - self.verts.x[i]) / self.dt
            if v.dot(normal) < 0.:
                # print("test")
                v -= v.dot(normal) * normal
                p = self.verts.x[i] + v * self.dt
            self.verts.p[i] += p
            self.verts.nc[i] += 1

    @ti.func
    def resolve_self(self, i, j):
        dx = self.verts.x_k[i] - self.verts.x_k[j]
        d = dx.norm()

        if d < 2.0 * self.radius:  # in contact
            normal = dx / d
            center = 0.5 * (self.verts.x_k[i] + self.verts.x_k[j])
            p1 = center + self.radius * normal
            p2 = center - self.radius * normal
            self.verts.p[i] += p1
            self.verts.p[j] += p2
            self.verts.nc[i] += 1
            self.verts.nc[j] += 1


    @ti.kernel
    def set_init_guess_pcg(self) -> ti.f32:

        # for v in self.verts:
        #     v.dx = v.g / v.h

        ti.mesh_local(self.Ap)
        for v in self.verts:
            self.Ap[v.id] = v.dx * v.m

        for e in self.edges:
            u = e.verts[0].id
            v = e.verts[1].id

            dx_u = e.verts[0].dx
            dx_v = e.verts[2].dx

            d = e.hij @ (dx_u - dx_v)
            self.Ap[u] += d
            self.Ap[v] -= d

        ti.mesh_local(self.r, self.Ap)
        for v in self.verts:
            self.r[v.id] = v.g - self.Ap[v.id]

        # self.r.copy_from(self.verts.g)

        ti.mesh_local(self.z, self.r, self.p)
        for v in self.verts:
           self.p[v.id] = self.z[v.id] = self.r[v.id] / v.h

        # r_2 = self.dot(self.z, self.r)

        r_2 = ti.float32(0.0)
        ti.loop_config(block_dim=64)
        for i in range(self.num_verts):
            r_2 += self.z[i].dot(self.r[i])

        return r_2


    @ti.kernel
    def apply_precondition(self, z: ti.template(), r: ti.template()):

        # ti.mesh_local(z, r)
        for e in self.edges:
            i, j = e.verts[0].id, e.verts[1].id
            ri, rj = r[i], r[j]
            rx = ti.math.vec2(ri.x, rj.x)
            ry = ti.math.vec2(ri.y, rj.y)
            rz = ti.math.vec2(ri.z, rj.z)

            zx = e.hinv @ rx
            zy = e.hinv @ ry
            zz = e.hinv @ rz

            zi = ti.math.vec3([zx[0], zy[0], zz[0]])
            zj = ti.math.vec3([zx[1], zy[1], zz[1]])
            z[i] += zi
            z[j] += zj


        ti.mesh_local(z)
        for v in self.verts:
            z[v.id] = z[v.id] / v.deg

        for i in z:
            z[i] = r[i] / self.verts.h[i]

    @ti.kernel
    def cg_iterate(self, r_2: ti.f32) -> ti.f32:

        # Ap = A * x
        ti.mesh_local(self.Ap, self.p)
        for v in self.verts:
            self.Ap[v.id] = self.p[v.id] * v.m

        ti.mesh_local(self.Ap, self.p)
        for e in self.edges:
            u = e.verts[0].id
            v = e.verts[1].id
            d = e.hij @ (self.p[u] - self.p[v])
            self.Ap[u] += d
            self.Ap[v] -= d


        pAp = ti.float32(0.0)
        ti.loop_config(block_dim=64)
        for i in range(self.num_verts):
            pAp += self.p[i].dot(self.Ap[i])

        alpha = r_2 / pAp

        ti.mesh_local(self.Ap, self.r)
        for v in self.verts:
            v.dx += alpha * self.p[v.id]
            self.r[v.id] -= alpha * self.Ap[v.id]

        # ti.mesh_local(self.z, self.r)
        for e in self.edges:
            i, j = e.verts[0].id, e.verts[1].id
            ri, rj = self.r[i], self.r[j]
            rx = ti.math.vec2(ri.x, rj.x)
            ry = ti.math.vec2(ri.y, rj.y)
            rz = ti.math.vec2(ri.z, rj.z)

            zx = e.hinv @ rx
            zy = e.hinv @ ry
            zz = e.hinv @ rz

            zi = ti.math.vec3(zx[0], zy[0], zz[0])
            zj = ti.math.vec3(zx[1], zy[1], zz[1])

            self.z[i] += zi
            self.z[j] += zj


        ti.mesh_local(self.z)
        for v in self.verts:
            self.z[v.id] = self.z[v.id] / v.deg

        ti.mesh_local(self.z, self.r)
        for v in self.verts:
            self.z[v.id] = self.r[v.id] / v.h

        r_2_new = ti.float32(0.0)
        ti.loop_config(block_dim=64)
        for v in self.verts:
            r_2_new += self.z[v.id].dot(self.r[v.id])

        beta = r_2_new / r_2

        ti.loop_config(block_dim=64)
        for i in range(self.num_verts):
            self.p[i] = self.z[i] + beta * self.p[i]

        return r_2_new



    def newton_pcg(self, tol, max_iter):

        self.verts.dx.fill(0.0)
        r_2 = self.set_init_guess_pcg()
        r_2_new = r_2

        # ti.profiler.clear_kernel_profiler_info()
        for iter in range(max_iter):

            self.z.fill(0.0)
            r_2_new = self.cg_iterate(r_2_new)

            if r_2_new <= tol:
                break


        # self.add(self.verts.x_k, self.verts.x_k, -1.0, self.verts.dx)

    @ti.func
    def compute_grid_index(self, pos: ti.math.vec3)->ti.math.ivec3:
        idx3d = ti.floor((pos - self.grid_min) * self.grid_n / (self.grid_max - self.grid_min), int)
        # idx = idx3d[0] * self.grid_n * self.grid_n + idx3d[1] * self.grid_n + idx3d[2]
        # if idx3d[0] < 0 or idx3d[1] < 0 or idx3d[2] < 0:
            # print(f"Wrong indexing!!! Position out of range. pos: {pos}, idx: {idx3d}")
        return idx3d


    # @ti.kernel
    # def construct_grid(self):
    #
    #     start_idx = ti.math.vec3(-3.0, -3.0, -3.0)
    #     for v in self.verts:
    #         grid_idx = ti.floor((v.x - start_idx) * (self.grid_n / 6.0), int)
    #         self.grain_count[grid_idx] += 1
    #
    #     self.column_sum.fill(0)
    #     # kernel comunicate with global variable ???? this is a bit amazing
    #     for i, j, k in ti.ndrange(self.grid_n, self.grid_n, self.grid_n):
    #         ti.atomic_add(self.column_sum[i, j], self.grain_count[i, j, k])
    #
    #     # this is because memory mapping can be out of order
    #     _prefix_sum_cur = 0
    #
    #     for i, j in ti.ndrange(self.grid_n, self.grid_n):
    #         self.prefix_sum[i, j] = ti.atomic_add(_prefix_sum_cur, self.column_sum[i, j])
    #
    #     """
    #     # case 1 wrong
    #     for i, j, k in ti.ndrange(grid_n, grid_n, grid_n):
    #         #print(i, j ,k)
    #         ti.atomic_add(prefix_sum[i,j], grain_count[i, j, k])
    #         linear_idx = i * grid_n * grid_n + j * grid_n + k
    #         list_head[linear_idx] = prefix_sum[i,j]- grain_count[i, j, k]
    #         list_cur[linear_idx] = list_head[linear_idx]
    #         list_tail[linear_idx] = prefix_sum[i,j]
    #
    #     """
    #
    #     # """
    #     # case 2 test okay
    #     for i, j, k in ti.ndrange(self.grid_n, self.grid_n, self.grid_n):
    #         # we cannot visit prefix_sum[i,j] in this loop
    #         pre = ti.atomic_add(self.prefix_sum[i, j], self.grain_count[i, j, k])
    #         linear_idx = i * self.grid_n * self.grid_n + j * self.grid_n + k
    #         self.list_head[linear_idx] = pre
    #         self.list_cur[linear_idx] = self.list_head[linear_idx]
    #         # only pre pointer is useable
    #         self.list_tail[linear_idx] = pre + self.grain_count[i, j, k]
    #         # """
    #     # e
    #     for v in self.verts_static:
    #         grid_idx = ti.floor((v.x - start_idx) * (self.grid_n / 2.0), int)
    #         linear_idx = grid_idx[0] * self.grid_n * self.grid_n + grid_idx[1] * self.grid_n + grid_idx[2]
    #         grain_location = ti.atomic_add(self.list_cur[linear_idx], 1)
    #         self.particle_id[grain_location] = v.id

    def update(self, dt, num_sub_steps):

        self.dt = dt / num_sub_steps
        self.dtSq = self.dt ** 2

        # ti.profiler.clear_kernel_profiler_info()

        self.initialize_particle_system()
        for sub_step in range(num_sub_steps):
            self.verts.f_ext.fill([0.0, self.gravity, 0.0])
            self.computeVtemp()

            self.computeY()
            self.verts.x_k.copy_from(self.verts.y)
            # self.set_grid_particles()
            tol = 1e-3

            for i in range(1):
                # i += 1
                self.verts.g.fill(0.)
                self.verts.h.copy_from(self.verts.m)
                self.evaluate_gradient_and_hessian()

                self.newton_pcg(tol=1e-6, max_iter=100)

                # dx_NormSq = self.dot(self.verts.dx, self.verts.dx)

                # alpha = 1.0
                self.step_forward()
                # if dx_NormSq < tol:
                #     break


            # print(f'opt iter: {i}')
            # ti.profiler.clear_kernel_profiler_info()
            # for i in range(3):
            self.verts.p.fill(0.0)
            self.verts.nc.fill(0.0)
            self.handle_contacts()
            # self.set_grid_particles()
            # # self.set_grid_particles()
            # self.handle_contacts()
            # query_result1 = ti.profiler.query_kernel_profiler_info(self.set_grid_particles.__name__)
            # query_result2 = ti.profiler.query_kernel_profiler_info(self.handle_contacts.__name__)
            # # print("kernel exec. #: ", query_result1.counter, query_result2.counter)
            # # print(f"Min set_grid_particles: {query_result1.min}, handle_contacts: {query_result2.min}")
            # # print(f"Max set_grid_particles: {query_result1.max}, handle_contacts: {query_result2.max}")
            # # print("total[ms]      :", float(query_result1.counter * query_result1.avg), float(query_result2.counter * query_result2.avg))
            # # print("avg[ms]        :", float(query_result1.avg), float(query_result2.avg))
            #
            # query_result = ti.profiler.query_kernel_profiler_info(self.handle_contacts.__name__)
            # print("kernel exec. #: ", query_result.counter)
            # print(f"Min: {query_result.min}, Max: {query_result.max}")
            # print("total[ms]      :", float(query_result.counter * query_result.avg))
            # print("avg[ms]        :", float(query_result.avg))
            #
            # ti.deactivate_all_snodes()
            self.computeNextState()

            # self.verts.x_k.copy_from(self.verts.x)
            # self.verts.p.fill(0.0)
            # self.verts.nc.fill(0.0)
            # self.set_grid_particles()
            # self.handle_contacts()
            # self.computeNextState()

        # cg_iterate_profile = ti.profiler.query_kernel_profiler_info(self.cg_iterate.__name__)
        # print("total cg_iterate call per frame: ", cg_iterate_profile.counter)
        # print("total[ms]     : ", float(cg_iterate_profile.counter * cg_iterate_profile.avg))
        # print("avg[ms]       : ", float(cg_iterate_profile.avg))
        #
        # evaluate_gradient_and_hessian_profile = ti.profiler.query_kernel_profiler_info(self.cg_iterate.__name__)
        # print("gradient/hessian call per frame: ", evaluate_gradient_and_hessian_profile.counter)
        # print("total[ms]     : ", float(evaluate_gradient_and_hessian_profile.counter * evaluate_gradient_and_hessian_profile.avg))
        # print("avg[ms]       : ", float(evaluate_gradient_and_hessian_profile.avg))


        # neighbour_search_profile_1 = ti.profiler.query_kernel_profiler_info(self.update_grid_id.__name__)
        # neighbour_search_profile_2 = ti.profiler.query_kernel_profiler_info(self.prefix_sum_executor.run.__name__)
        # neighbour_search_profile_3 = ti.profiler.query_kernel_profiler_info(self.counting_sort.__name__)
        # neighbour_search_profile_4 = ti.profiler.query_kernel_profiler_info(self.set_grid_particles.__name__)
        # # print("neighbour search call per frame: ", neighbour_search_profile.counter)
        # total_1 = neighbour_search_profile_1.counter * neighbour_search_profile_1.avg
        # total_2 = neighbour_search_profile_2.counter * neighbour_search_profile_2.avg
        # total_3 = neighbour_search_profile_3.counter * neighbour_search_profile_3.avg
        # total_4 = neighbour_search_profile_4.counter * neighbour_search_profile_4.avg

        # total_1
        #
        # print("total[ms]: ", float(total_1 + total_2 + total_3), float(total_4))
        # print("avg[ms]       : ", float(neighbour_search_profile.avg))


