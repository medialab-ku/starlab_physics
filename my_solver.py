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


        self.radius = 0.005
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
        self.damping_factor = 6e-4

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

        self.max_num_verts = self.num_verts + self.num_verts_static
        self.grid_ids = ti.field(int, shape=self.max_num_verts)
        self.grid_ids_buffer = ti.field(int, shape=self.max_num_verts)
        self.grid_ids_new = ti.field(int, shape=self.max_num_verts)
        self.cur2org = ti.field(int, shape=self.max_num_verts)
        self.reset()

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

        for v in self.verts_static:
            grid_index = self.get_flatten_grid_index(v.x)
            self.grid_ids[v.id + self.num_verts] = grid_index
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
    def for_all_neighbors(self, p_i):
        center_cell = self.pos_to_index(self.verts.x_k[p_i])
        for offset in ti.grouped(ti.ndrange(*((-1, 2),) * 3)):
            grid_index = self.flatten_grid_index(center_cell + offset)
            for p_j in range(self.grid_particles_num[ti.max(0, grid_index-1)], self.grid_particles_num[grid_index]):
                p_j_cur = self.cur2org[p_j]

                if p_j_cur < self.num_verts:
                    if p_i != p_j_cur:
                        self.resolve_self(p_i, p_j_cur)
                if p_j_cur >= self.num_verts:
                    self.resolve(p_i, p_j_cur - self.num_verts)


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
        self.verts.f_ext.fill([0.0, self.gravity, 0.0])
        self.reset_kernel()

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
            # # hij = coef * (self.id3)
            U, sig, V = ti.svd(hij)

            for i in range(3):
                if sig[i, i] < 1e-6:
                    sig[i, i] = 1e-6

            hij = U @ sig @ V.transpose()
            e.hij = hij

        for v in self.verts:
            self.for_all_neighbors(v.id)

        # for e in self.edges:
        #     h = ti.math.mat2([[e.verts[0].h, 0],
        #                       [0, e.verts[1].h]]) + \
        #         coef * ti.math.mat2([[0, -1],
        #                              [-1, 0]])
        #     e.hinv = h.inverse()

    @ti.kernel
    def step_forward(self):
        for v in self.verts:
            if v.id == 0 or v.id == 2:
                v.x_k = v.x_k
            else:
                v.x_k += v.dx

    @ti.kernel
    def handle_contacts(self):

        for v in self.verts:
            self.for_all_neighbors(v.id)


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

            self.verts.g[i] += self.dtSq * 1e7 * (p - self.verts.x_k[i])
            self.verts.h[i] += self.dtSq * 1e7
            self.verts.hc[i] += self.dtSq * 1e7

            # if v.dot(normal) < 0.:
            #     # print("test")
            #     v -= v.dot(normal) * normal
            #     p = self.verts.x[i] + v * self.dt
            # self.verts.p[i] += p
            # self.verts.nc[i] += 1

    @ti.func
    def resolve_self(self, i, j):
        dx = self.verts.x_k[i] - self.verts.x_k[j]
        d = dx.norm()

        if d < 2.0 * self.radius:  # in contact
            normal = dx / d
            center = 0.5 * (self.verts.x_k[i] + self.verts.x_k[j])
            p1 = center + self.radius * normal
            p2 = center - self.radius * normal

            v1 = (p1 - self.verts.x[i]) / self.dt
            v2 = (p2 - self.verts.x[j]) / self.dt
            v21 = v1 - v2
            dvn = normal.dot(v21)
            if dvn < 0.0:
                v1 -= 0.5 * dvn * normal
                v2 += 0.5 * dvn * normal
                p1 = self.verts.x[i] + v1 * self.dt
                p2 = self.verts.x[j] + v2 * self.dt

            self.verts.g[i] += self.dtSq * 1e7 * (p1 - self.verts.x_k[i])
            self.verts.g[j] += self.dtSq * 1e7 * (p2 - self.verts.x_k[j])
            self.verts.h[i] += self.dtSq * 1e7
            self.verts.h[j] += self.dtSq * 1e7
            self.verts.hc[i] += self.dtSq * 1e7
            self.verts.hc[j] += self.dtSq * 1e7

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
            self.Ap[v.id] = self.p[v.id] * v.m + self.p[v.id] * v.hc

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
        # for e in self.edges:
        #     i, j = e.verts[0].id, e.verts[1].id
        #     ri, rj = self.r[i], self.r[j]
        #     rx = ti.math.vec2(ri.x, rj.x)
        #     ry = ti.math.vec2(ri.y, rj.y)
        #     rz = ti.math.vec2(ri.z, rj.z)
        #
        #     zx = e.hinv @ rx
        #     zy = e.hinv @ ry
        #     zz = e.hinv @ rz
        #
        #     zi = ti.math.vec3(zx[0], zy[0], zz[0])
        #     zj = ti.math.vec3(zx[1], zy[1], zz[1])
        #
        #     self.z[i] += zi
        #     self.z[j] += zj
        #
        #
        # ti.mesh_local(self.z)
        # for v in self.verts:
        #     self.z[v.id] = self.z[v.id] / v.deg

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

        # query_result1 = ti.profiler.query_kernel_profiler_info(self.cg_iterate.__name__)
        # print("kernel exec. #: ", query_result1.counter)

        # self.add(self.verts.x_k, self.verts.x_k, -1.0, self.verts.dx)


    def update(self, dt, num_sub_steps):

        self.dt = dt / num_sub_steps
        self.dtSq = self.dt ** 2

        ti.profiler.clear_kernel_profiler_info()

        self.initialize_particle_system()
        for sub_step in range(num_sub_steps):
            self.computeVtemp()

            self.computeY()
            self.verts.x_k.copy_from(self.verts.y)
            # self.set_grid_particles()
            tol = 1e-6

            for i in range(self.max_iter):
                # i += 1
                self.verts.g.fill(0.)
                self.verts.h.copy_from(self.verts.m)
                self.verts.hc.fill(0.)
                self.evaluate_gradient_and_hessian()

                self.newton_pcg(tol=1e-4, max_iter=100)

                dx_NormSq = self.dot(self.verts.dx, self.verts.dx)

                # alpha = 1.0
                self.step_forward()
                if dx_NormSq < tol:
                    break


            # print(f'opt iter: {i}')
            # ti.profiler.clear_kernel_profiler_info()
            # for i in range(3):
            # self.verts.p.fill(0.0)
            # self.verts.nc.fill(0.0)
            # self.handle_contacts()
            self.computeNextState()

        query_result1 = ti.profiler.query_kernel_profiler_info(self.cg_iterate.__name__)
        print("kernel exec. #: ", query_result1.counter)