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
                 static_meshes,
                 bottom,
                 min_range,
                 max_range,
                 k=1e4,
                 dt=1e-3,
                 max_iter=1000):
        self.my_mesh = my_mesh
        self.static_mesh = static_mesh
        self.grid_origin = ti.math.vec3([-3, -3, -3])
        self.grid_size = ti.math.vec3([6, 6, 6])
        self.grid_min = ti.math.vec3(min_range[0], min_range[1], min_range[2])
        self.grid_max = ti.math.vec3(max_range[0], max_range[1], max_range[2])
        self.domain_size = self.grid_size - self.grid_origin


        self.radius = 0.008
        self.grid_size = 5 * self.radius
        self.grid_num = np.ceil(self.domain_size / self.grid_size).astype(int)
        print("grid size: ", self.grid_num)
        self.padding = self.grid_size


        self.k = k
        self.dt = dt
        self.dtSq = dt ** 2
        self.max_iter = max_iter
        self.gravity = -7.81
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
        self.damping_factor = 1.e-3
        self.batch_size = 10
        self.num_bats = self.num_verts // self.batch_size
        print(f'batches #: {self.num_bats}')
        print(f"verts #: {len(self.my_mesh.mesh.verts)}, edges #: {len(self.my_mesh.mesh.edges)} faces #: {len(self.my_mesh.mesh.faces)}")

        # for PCG
        self.b = ti.Vector.field(3, dtype=ti.f32, shape=self.num_verts)
        self.r = ti.Vector.field(3, dtype=ti.f32, shape=self.num_verts)
        self.p = ti.Vector.field(3, dtype=ti.f32, shape=self.num_verts)
        self.Ap = ti.Vector.field(3, dtype=ti.f32, shape=self.num_verts)
        self.z = ti.Vector.field(3, dtype=ti.f32, shape=self.num_verts)

        self.grid_particles_num = ti.field(int, shape=int(self.grid_num[0] * self.grid_num[1] * self.grid_num[2]))
        self.grid_particles_num_temp = ti.field(int, shape=int(self.grid_num[0] * self.grid_num[1] * self.grid_num[2]))
        self.prefix_sum_executor = ti.algorithms.PrefixSumExecutor(self.grid_particles_num.shape[0])

        self.max_num_verts = self.num_verts + self.num_verts_static + self.num_edges_static
        self.grid_ids = ti.field(int, shape=self.max_num_verts)
        self.grid_ids_buffer = ti.field(int, shape=self.max_num_verts)
        self.grid_ids_new = ti.field(int, shape=self.max_num_verts)
        self.cur2org = ti.field(int, shape=self.max_num_verts)

        self.frame = 0
        self.frames = static_meshes

        self.num_max_neighbors = 32
        self.support_radius = 0.05
        self.neighbor_ids = ti.field(int, shape=(self.num_verts, self.num_max_neighbors))
        self.num_neighbor = ti.field(int, shape=(self.num_verts))
        self.weights = ti.field(ti.math.vec2, shape=(self.num_verts, self.num_verts))
        self.adj = ti.field(int, shape=(self.num_verts, self.num_verts))
        self.c = ti.field(ti.f32, shape=1)
        self.w = 0.1
        self.reset()

    @ti.func
    def cubic_kernel(self, r_norm):
        res = ti.cast(0.0, ti.f32)
        h = self.support_radius
        # value of cubic spline smoothing kernel
        k = 1.0
        k = 8 / ti.math.pi
        k /= h ** 3
        q = r_norm / h
        if q <= 1.0:
            if q <= 0.5:
                q2 = q * q
                q3 = q2 * q
                res = k * (6.0 * q3 - 6.0 * q2 + 1)
            else:
                res = k * 2 * ti.pow(1 - q, 3.0)
        return res
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

        for e in self.edges_static:
            grid_index = self.get_flatten_grid_index(e.x)
            self.grid_ids[e.id + self.num_verts + self.num_edges_static] = grid_index
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
    def for_all_neighbors(self, p_i, task1: ti.template(), task2: ti.template(), task3: ti.template()):
        center_cell = self.pos_to_index(self.verts.x_k[p_i])
        # for offset in ti.grouped(ti.ndrange(*((-1, 2),) * 3)):
        grid_index = self.flatten_grid_index(center_cell)
        for p_j in range(self.grid_particles_num[ti.max(0, grid_index-1)], self.grid_particles_num[grid_index]):
            p_j_cur = self.cur2org[p_j]
            if p_j_cur < self.num_verts:
                if p_i != p_j_cur:
                    task1(p_i, p_j_cur)
            elif p_j_cur >= self.num_verts and p_j_cur < self.num_verts + self.num_edges_static:
                task2(p_i, p_j_cur - self.num_verts)
            else:
                task3(p_i, p_j_cur - self.num_verts - self.num_edges_static)


    @ti.kernel
    def reset_kernel(self):

        for e in self.edges_static:
            e.x = 0.5 * (e.verts[0].x + e.verts[1].x)
            i, j = e.verts[0].id, e.verts[1].id
            # self.adj[i, j] = 0
            # self.adj[j, i] = 0

        for e in self.edges:
            i, j = e.verts[0].id, e.verts[1].id
            self.adj[i, j] = 0
            self.adj[j, i] = 0

        for vi in range(0, self.num_verts):
            i = 0
            for vj in range(0, self.num_verts):
                if vi != vj and i < self.num_max_neighbors:
                    lij0 = (self.verts.x0[vi] - self.verts.x0[vj]).norm()
                    if lij0 < self.support_radius:
                        self.neighbor_ids[vi, i] = vj
                        self.weights[vi, vj][0] = lij0
                        self.weights[vi, vj][1] = self.cubic_kernel(lij0)
                        i+=1

            self.num_neighbor[vi] = i

        # center = ti.math.vec3(0.5, -0.8, 0.5)
        # for v in self.verts:
        #     dxz = v.x[2] - center[2]
        #     v.x[2] = center[2] + 1.03 * dxz


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
        self.adj.fill(1)
        self.reset_kernel()
        # print(self.num_neighbor)

    @ti.kernel
    def pre_stabilization(self):
        for v in self.verts:
            self.for_all_neighbors(v.id, self.resolve_self_pre, self.resolve_pre, self.resolve_edge_pre)

        for v in self.verts:
            if v.nc > 0:
                v.x += (v.dx / v.nc)
                v.x_k += (v.dx / v.nc)

    @ti.kernel
    def computeVtemp(self):
        for v in self.verts:
            if v.id == 0 or v.id == 2:
                v.v = ti.math.vec3(0.0)
            else:
                v.v += (v.f_ext / v.m) * self.dt


        # v.nc += 1

        # for v in self.verts:
        #      self.for_all_neighbors(v.id, self.resolve_v_self, self.resolve_v, self.resolve_v_edge)
        # # #
        # for v in self.verts:
        #     if v.nc > 0:
        #         v.v += (v.dx / v.nc)
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

        # for v in self.verts:
        #      self.for_all_neighbors(v.id, self.resolve_v_self, self.resolve_v, self.resolve_v_edge)
        # # # #
        # for v in self.verts:
        #     if v.nc > 0:
        #         v.v += (v.dx / v.nc)

    @ti.kernel
    def evaluate_gradient_and_hessian(self):
        # self.candidatesPC.deactivate()
        coef = self.dtSq * self.k
        # xij = self.verts.x_k[0] - self.verts.x0[0]
        # grad = coef * xij
        # self.verts.g[0] -= grad
        # self.verts.h[0] += coef
        #
        # xij = self.verts.x_k[2] - self.verts.x0[2]
        # grad = coef * xij
        # self.verts.g[2] -= grad
        # self.verts.h[2] += coef

        for e in self.edges:
            xij = e.verts[0].x_k - e.verts[1].x_k
            center = 0.5 * (e.verts[0].x_k + e.verts[1].x_k)
            lij = xij.norm()
            # grad = coef * (xij - (e.l0/lij) * xij)
            normal = xij / lij
            p1 = center + 0.5 * e.l0 * normal
            p2 = center - 0.5 * e.l0 * normal


            e.verts[0].dx += (p1 - e.verts[0].x_k)
            e.verts[1].dx += (p2 - e.verts[1].x_k)
            e.verts[0].nc += 1
            e.verts[1].nc += 1

        # for v in self.verts:
        #     self.for_all_neighbors(v.id, self.resolve_self, self.resolve, self.resolve_edge)


    @ti.kernel
    def step_forward(self):
        w = 1.0
        for v in self.verts:
            if v.id == 0 or v.id == 2:
                v.x_k = v.x
            else:
                v.x_k += w * (v.dx / v.nc)

    @ti.kernel
    def handle_contacts(self):

        for v in self.verts:
            self.for_all_neighbors(v.id, self.resolve_self, self.resolve, self.resolve_edge)


        for v in self.verts:
            if v.nc > 0:
                v.x_k = v.dx / v.nc

        # for v in self.verts:
        #     v.x_k += (v.gc / v.hc)


    @ti.func
    def resolve(self, i, j):
        dx = self.verts.x_k[i] - self.verts_static.x[j]
        d = dx.norm()
        coef = self.dtSq * self.contact_stiffness
        if d < 2.0 * self.radius:  # in contact
            normal = dx / d
            p = self.verts_static.x[j] + 2.0 * self.radius * normal

            w = ti.math.exp((d / (2 * self.radius))) / ti.math.exp(1.0)
            self.verts.dx[i] += self.w * (p - self.verts.x_k[i])
            self.verts.nc[i] += 1

    @ti.func
    def resolve_edge(self, i, j):
        dx = self.verts.x_k[i] - self.edges_static.x[j]
        d = dx.norm()
        coef = self.dtSq * self.contact_stiffness
        if d < 2.0 * self.radius:  # in contact
            normal = dx / d
            p = self.edges_static.x[j] + 2.0 * self.radius * normal
            v = (p - self.verts.x[i]) / self.dt

            w = ti.math.exp((d / (2 * self.radius))) / ti.math.exp(1.0)
            self.verts.dx[i] += self.w * (p - self.verts.x_k[i])
            self.verts.nc[i] += 1
            # self.verts.p[i] += p
            # self.verts.nc[i] += 1


    @ti.func
    def resolve_self(self, i, j):
        dx = self.verts.x_k[i] - self.verts.x_k[j]
        d = dx.norm()
        coef = self.adj[i, j] * self.dtSq * self.contact_stiffness
        if d < 2.0 * self.radius:  # in contact
            normal = dx / d

            # g = coef * (d - 2.0 * self.radius) * normal
            # self.c[0] += (coef/2.0) * (d - 2.0 * self.radius) ** 2
            center = 0.5 * (self.verts.x_k[i] + self.verts.x_k[j])
            p1 = center + self.radius * normal
            p2 = center - self.radius * normal

            self.verts.dx[i] += self.w * self.adj[i, j] * (p1 - self.verts.x_k[i])
            self.verts.dx[j] += self.w * self.adj[i, j] * (p2 - self.verts.x_k[j])
            self.verts.nc[i] += self.adj[i, j]
            self.verts.nc[j] += self.adj[i, j]

    @ti.func
    def resolve_v_self(self, i, j):
        dx = self.verts.x_k[i] - self.verts.x_k[j]
        dv = self.verts.v[i] - self.verts.v[j]
        d = dx.norm()
        n = dx / d
        c = dv.dot(n)
        if d < 2.0 * self.radius and c < 0.0:  # in contact
            self.verts.dx[i] -= c * n
            self.verts.dx[j] += c * n
            self.verts.nc[i] += 1
            self.verts.nc[j] += 1

    @ti.func
    def resolve_v(self, i, j):
        dx = self.verts.x_k[i] - self.verts_static.x[j]
        dv = self.verts.v[i] - self.verts_static.v[j]
        d = dx.norm()
        n = dx / d
        c = dv.dot(n)

        if d <= 2.0 * self.radius and c < 0.0:  # in contact
            self.verts.dx[i] -= c * n

            dvt = dv - c * n
            if dvt.norm() < 0.8 * abs(c):
                self.verts.dx[i] -= dvt
            else:
                self.verts.dx[i] -= 0.8 * dvt

            self.verts.nc[i] += 1

    @ti.func
    def resolve_v_edge(self, i, j):
        dx = self.verts.x_k[i] - self.edges_static.x[j]
        dv = self.verts.v[i] - self.edges_static.v[j]
        d = dx.norm()
        n = dx / d
        c = dv.dot(n)

        if d < 2.0 * self.radius and c < 0.0:  # in contact
            self.verts.dx[i] -= c * n
            dvt = dv - c * n
            if dvt.norm() < 0.8 * abs(c):
                self.verts.dx[i] -= dvt
            else:
                self.verts.dx[i] -= 0.8 * dvt
            self.verts.nc[i] += 1

    @ti.func
    def resolve_pre(self, i, j):
        dx = self.verts.x[i] - self.verts_static.x[j]
        d = dx.norm()
        if d < 2.0 * self.radius:  # in contact
            normal = dx / d
            p = self.verts_static.x[j] + 2.0 * self.radius * normal

            w = ti.math.exp((d / (2 * self.radius))) / ti.math.exp(1.0)
            self.verts.dx[i] += self.w * (p - self.verts.x[i])
            self.verts.nc[i] += 1

    @ti.func
    def resolve_edge_pre(self, i, j):
        dx = self.verts.x[i] - self.edges_static.x[j]
        d = dx.norm()
        if d < 2.0 * self.radius:  # in contact
            normal = dx / d
            p = self.edges_static.x[j] + 2.0 * self.radius * normal

            w = ti.math.exp((d / (2 * self.radius))) / ti.math.exp(1.0)
            self.verts.dx[i] += self.w * (p - self.verts.x[i])
            self.verts.nc[i] += 1

    @ti.func
    def resolve_self_pre(self, i, j):
        dx = self.verts.x[i] - self.verts.x[j]
        d = dx.norm()
        if d < 2.0 * self.radius:  # in contact
            normal = dx / d
            center = 0.5 * (self.verts.x[i] + self.verts.x[j])
            p1 = center + self.radius * normal
            p2 = center - self.radius * normal
            # self.verts.dx[i] += self.adj[i, j] * (p1 - self.verts.x[i])
            # self.verts.dx[j] += self.adj[i, j] * (p2 - self.verts.x[j])
            # self.verts.nc[i] += self.adj[i, j]
            # self.verts.nc[j] += self.adj[i, j]


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

    @ti.func
    def precond_value_z(self, z: ti.template(), r: ti.template()):

        # 7 ~ 8 iter
        for i in z:
            z[i] = r[i] / self.verts.h[i]

        # 20 iter
        # for i in z:
        #     z[i] = r[i]


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

        self.precond_value_z(self.z, self.r)
        # ti.mesh_local(self.z, self.r)
        # for v in self.verts:
        #     self.z[v.id] = self.r[v.id] / v.h

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

        ti.profiler.clear_kernel_profiler_info()
        for iter in range(max_iter):

            self.z.fill(0.0)
            r_2_new = self.cg_iterate(r_2_new)

            if r_2_new <= tol:
                break

        query_result1 = ti.profiler.query_kernel_profiler_info(self.cg_iterate.__name__)
        print("kernel exec. #: ", query_result1.counter)

        # self.add(self.verts.x_k, self.verts.x_k, -1.0, self.verts.dx)


    @ti.kernel
    def diag_hessian(self):
        for v in self.verts:
            v.dx = (v.g + v.gc) / (v.h + v.hc)

    @ti.kernel
    def edge_wise_jacobi(self):
        coef = self.dtSq * self.k
        for e in self.edges:
            hii, hjj = e.verts[0].h, e.verts[1].h
            # hiic, hjjc = e.verts[0].hc, e.verts[1].hc
            # hii += hiic
            # hjj += hjjc
            gi, gj = e.verts[0].g, e.verts[1].g
            det = hii * hjj - (coef ** 2)
            hinv = ti.math.mat2([[hjj, coef], [coef, hii]]) / det
            gx = ti.math.vec2(gi.x, gj.x)
            gy = ti.math.vec2(gi.y, gj.y)
            gz = ti.math.vec2(gi.z, gj.z)

            dx = hinv @ gx
            dy = hinv @ gy
            dz = hinv @ gz

            dxi = ti.math.vec3(dx[0], dy[0], dz[0])
            dxj = ti.math.vec3(dx[1], dy[1], dz[1])

            e.verts[0].dx += dxi
            e.verts[1].dx += dxj

        for v in self.verts:
            v.dx /= v.deg
    def evaluate_search_dir(self, policy):

        if policy == 0:
            self.newton_pcg(tol=1e-4, max_iter=100)
        elif policy == 1:
            self.diag_hessian()
        elif policy == 2:
            self.edge_wise_jacobi()


    # def line_search(self):
    #
    #     alpha = 1.0
    #     e_cur = self.evaluate_current_energy()
    #     for i in range(10):
    #
    # @ti.kernel
    # def evaluate_current_energy(self) -> ti.f32:
    #
    #     for e in self.edges:

    @ti.kernel
    def update_static_mesh(self, frame: ti.i32, frame_rate:ti.i32, scale: ti.f32, trans: ti.math.vec3, rot: ti.math.vec3):
        rot_rad = ti.math.radians(rot)
        r3d = ti.math.rotation3d(rot_rad[0], rot_rad[1], rot_rad[2])

        for v in range(self.num_verts_static):
            # linear interpolation between frames
            # frame_rate = 10
            keyframe = frame//frame_rate
            alpha = (frame/frame_rate) - keyframe
            pos = self.frames[keyframe, v] * (1.0 - alpha) + self.frames[keyframe + 1, v] * alpha

            pos *= scale

            v_4d = ti.Vector([pos[0], pos[1], pos[2], 1])
            rv = r3d @ v_4d
            rotated_pos = ti.Vector([rv[0], rv[1], rv[2]])

            rotated_pos += trans

            self.static_mesh.mesh.verts.x[v] = rotated_pos
            self.verts_static.x[v] = rotated_pos


    @ti.kernel
    def update_edge_particle_pos(self):

        for e in self.edges_static:
            e.x = 0.5 * (e.verts[0].x + e.verts[1].x)
            e.v = 0.5 * (e.verts[0].v + e.verts[1].v)

    def update(self, dt, num_sub_steps):

        self.dt = dt / num_sub_steps
        self.dtSq = self.dt ** 2

        # ti.profiler.clear_kernel_profiler_info()
        self.update_edge_particle_pos()
        self.initialize_particle_system()
        for sub_step in range(num_sub_steps):
            self.computeVtemp()
            self.computeY()
            self.verts.x_k.copy_from(self.verts.y)

            # for i in range(2):
            #     self.verts.dx.fill(0.0)
            #     self.verts.nc.fill(0)
            #     self.pre_stabilization()

            tol = 1e-2

            for i in range(self.max_iter):
                self.verts.dx.fill(0.0)
                self.verts.nc.fill(0)
                self.evaluate_gradient_and_hessian()
                self.step_forward()

                # alpha = 1.0
            # self.verts.dx.fill(0.0)
            # self.verts.nc.fill(0)
            self.computeNextState()

        self.frame += 1

        # query_result1 = ti.profiler.query_kernel_profiler_info(self.cg_iterate.__name__)
        # print("kernel exec. #: ", query_result1.counter)