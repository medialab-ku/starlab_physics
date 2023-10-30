import taichi as ti
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
                 k=1e6,
                 dt=1e-3,
                 max_iter=1000):
        self.my_mesh = my_mesh
        self.static_mesh = static_mesh
        self.k = k
        self.dt = dt
        self.dtSq = dt ** 2
        self.max_iter = max_iter
        self.gravity = -9.81
        self.bottom = bottom
        self.id3 = ti.math.mat3([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])

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

        self.radius = 0.01
        self.contact_stiffness = 1e3
        self.damping_factor = 1e-4
        self.grid_n = 32
        # self.grid_particles_list = ti.field(ti.i32)
        # self.grid_block = ti.root.dense(ti.ijk, (self.grid_n, self.grid_n, self.grid_n))
        # self.partical_array = self.grid_block.dynamic(ti.l, self.num_verts_static)
        # self.partical_array.place(self.grid_particles_list)
        # self.grid_particles_count = ti.field(ti.i32)
        # ti.root.dense(ti.ijk, (self.grid_n, self.grid_n, self.grid_n)).place(self.grid_particles_count)

        self.list_head = ti.field(dtype=ti.i32, shape=self.grid_n ** 3)
        self.list_cur = ti.field(dtype=ti.i32, shape=self.grid_n ** 3)
        self.list_tail = ti.field(dtype=ti.i32, shape=self.grid_n ** 3)

        self.grain_count = ti.field(dtype=ti.i32,
                               shape=(self.grid_n, self.grid_n, self.grid_n),
                               name="grain_count")
        self.column_sum = ti.field(dtype=ti.i32, shape=(self.grid_n, self.grid_n), name="column_sum")
        self.prefix_sum = ti.field(dtype=ti.i32, shape=(self.grid_n, self.grid_n), name="prefix_sum")
        self.particle_id = ti.field(dtype=ti.i32, shape=self.num_verts, name="particle_id")
        print(f"verts #: {len(self.my_mesh.mesh.verts)}, edges #: {len(self.my_mesh.mesh.edges)} faces #: {len(self.my_mesh.mesh.faces)}")

        # for PCG
        self.b = ti.Vector.field(3, dtype=ti.f32, shape=self.num_verts)
        self.r = ti.Vector.field(3, dtype=ti.f32, shape=self.num_verts)
        self.p = ti.Vector.field(3, dtype=ti.f32, shape=self.num_verts)
        self.Ap = ti.Vector.field(3, dtype=ti.f32, shape=self.num_verts)
        self.z = ti.Vector.field(3, dtype=ti.f32, shape=self.num_verts)


    def reset(self):

        self.verts.x.copy_from(self.verts.x0)
        self.verts.v.fill(0.0)


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
    def compute_candidates_PT(self):

        self.candidatesPT.deactivate()
        for v in self.verts:
            for tid in range(self.num_faces_static):
                a_min_p, a_max_p = v.aabb_min, v.aabb_max
                a_min_t, a_max_t = self.faces_static.aabb_min[tid], self.faces_static.aabb_max[tid]

                if self.aabb_intersect(a_min_p, a_max_p, a_min_t, a_max_t):
                    self.candidatesPT.append(self.PT_type(pid=v.id, tid=tid))

        # print(self.candidatesPT.length())

    @ti.kernel
    def computeVtemp(self):
        for v in self.verts:
            if v.id == 0 or v.id == 2:
                v.v = ti.math.vec3(0.0, 0.0, 0.0)
            else:
                v.v += (v.f_ext / v.m) * self.dt

    @ti.kernel
    def globalSolveVelocity(self):
        for v in self.verts:
            v.v -= v.g / v.h

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
        coef = self.dtSq * self.k
        for e in self.edges:
            xij = e.verts[0].x_k - e.verts[1].x_k
            lij = xij.norm()
            grad = coef * (xij - (e.l0/lij) * xij)
            e.verts[0].g -= grad
            e.verts[1].g += grad
            e.verts[0].h += coef
            e.verts[1].h += coef

            hij = coef * (self.id3 - e.l0 / lij * (self.id3 - (self.abT(xij, xij)) / (lij ** 2)))
            U, sig, V = ti.svd(hij)

            for i in range(3):
                if sig[i, i] < 1e-6:
                    sig[i, i] = 1e-6

            hij = U @ sig @ V.transpose()
            e.hij = hij



    @ti.kernel
    def step_forward(self):
        for v in self.verts:
            if v.id == 0 or v.id == 2:
                v.x_k = v.x_k
            else:
                v.x_k += v.dx

    @ti.kernel
    def handle_contacts(self):
        # start_idx = ti.math.vec3(-3, -3, -3)
        for v in self.verts:
            for s in range(self.num_verts_static):
                self.resolve(v.id, s)

        for v in self.verts:
            for s in range(v.id + 1, self.num_verts):
                self.resolve_self(v.id, s)
            # grid_idx = ti.floor((v.x - start_idx) * (self.grid_n / 6.0), int)
            # x_begin = max(grid_idx[0] - 1, 0)
            # x_end = min(grid_idx[0] + 2, self.grid_n)
            #
            # y_begin = max(grid_idx[1] - 1, 0)
            # y_end = min(grid_idx[1] + 2, self.grid_n)
            #
            # z_begin = max(grid_idx[2] - 1, 0)
            #
            # # only need one side
            # z_end = min(grid_idx[2] + 1, self.grid_n)
            #
            # # todo still serialize
            # for neigh_i, neigh_j, neigh_k in ti.ndrange((x_begin, x_end), (y_begin, y_end), (z_begin, z_end)):
            #
            #     # on split plane
            #     if neigh_k == grid_idx[2] and (neigh_i + neigh_j) > (grid_idx[0] + grid_idx[1]) and neigh_i <= grid_idx[0]:
            #         continue
            #     # same grid
            #     iscur = neigh_i == grid_idx[0] and neigh_j == grid_idx[1] and neigh_k == grid_idx[2]
            #
            #     neigh_linear_idx = neigh_i * self.grid_n * self.grid_n + neigh_j * self.grid_n + neigh_k
            #     for p_idx in range(self.list_head[neigh_linear_idx], self.list_tail[neigh_linear_idx]):
            #         j = self.particle_id[p_idx]
            #         if iscur and v.id >= j:
            #             continue
            #         self.resolve(v.id, j)

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
    def apply_precondition(self, z: ti.template(), r: ti.template()):
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
        self.r.copy_from(self.verts.g)

        self.apply_precondition(self.z, self.r)
        self.p.copy_from(self.z)

        r_2 = self.dot(self.z, self.r)

        r_2_new = r_2

        ti.profiler.clear_kernel_profiler_info()
        for iter in range(max_iter):
            r_2_new = self.cg_iterate(r_2_new)

            if r_2_new <= tol:
                break
        query_result = ti.profiler.query_kernel_profiler_info(self.cg_iterate.__name__)
        print("kernel exec. #: ", query_result.counter)
        # print("kernel elapsed time(min_in_ms) =", query_result.min)
        # print("kernel elapsed time(max_in_ms) =", query_result.max)
        print("total[ms]     : ", float(query_result.counter * query_result.avg))
        print("avg[ms]       : ", float(query_result.avg))

        # self.add(self.verts.x_k, self.verts.x_k, -1.0, self.verts.dx)

    @ti.kernel
    def construct_grid(self):

        start_idx = ti.math.vec3(-3.0, -3.0, -3.0)
        for v in self.verts:
            grid_idx = ti.floor((v.x - start_idx) * (self.grid_n / 6.0), int)
            self.grain_count[grid_idx] += 1

        self.column_sum.fill(0)
        # kernel comunicate with global variable ???? this is a bit amazing
        for i, j, k in ti.ndrange(self.grid_n, self.grid_n, self.grid_n):
            ti.atomic_add(self.column_sum[i, j], self.grain_count[i, j, k])

        # this is because memory mapping can be out of order
        _prefix_sum_cur = 0

        for i, j in ti.ndrange(self.grid_n, self.grid_n):
            self.prefix_sum[i, j] = ti.atomic_add(_prefix_sum_cur, self.column_sum[i, j])

        """
        # case 1 wrong
        for i, j, k in ti.ndrange(grid_n, grid_n, grid_n):
            #print(i, j ,k)        
            ti.atomic_add(prefix_sum[i,j], grain_count[i, j, k])    
            linear_idx = i * grid_n * grid_n + j * grid_n + k
            list_head[linear_idx] = prefix_sum[i,j]- grain_count[i, j, k]
            list_cur[linear_idx] = list_head[linear_idx]
            list_tail[linear_idx] = prefix_sum[i,j]

        """

        # """
        # case 2 test okay
        for i, j, k in ti.ndrange(self.grid_n, self.grid_n, self.grid_n):
            # we cannot visit prefix_sum[i,j] in this loop
            pre = ti.atomic_add(self.prefix_sum[i, j], self.grain_count[i, j, k])
            linear_idx = i * self.grid_n * self.grid_n + j * self.grid_n + k
            self.list_head[linear_idx] = pre
            self.list_cur[linear_idx] = self.list_head[linear_idx]
            # only pre pointer is useable
            self.list_tail[linear_idx] = pre + self.grain_count[i, j, k]
            # """
        # e
        for v in self.verts_static:
            grid_idx = ti.floor((v.x - start_idx) * (self.grid_n / 2.0), int)
            linear_idx = grid_idx[0] * self.grid_n * self.grid_n + grid_idx[1] * self.grid_n + grid_idx[2]
            grain_location = ti.atomic_add(self.list_cur[linear_idx], 1)
            self.particle_id[grain_location] = v.id

    def update(self, dt, num_sub_steps):

        self.dt = dt / num_sub_steps
        self.dtSq = self.dt ** 2

        # self.construct_grid()
        for sub_step in range(num_sub_steps):
            self.verts.f_ext.fill([0.0, self.gravity, 0.0])
            self.computeVtemp()

            # self.compute_aabb()
            # self.compute_candidates_PT()

            self.computeY()
            self.verts.x_k.copy_from(self.verts.y)

            for i in range(1):
                self.verts.g.fill(0.)
                self.verts.h.copy_from(self.verts.m)
                self.evaluate_gradient_and_hessian()
                self.newton_pcg(tol=1e-12, max_iter=100)
                # alpha = 1.0
                self.step_forward()

            # for i in range(3):
            # self.verts.p.fill(0.0)
            # self.verts.nc.fill(0.0)
            # self.handle_contacts()

            ti.deactivate_all_snodes()
            self.computeNextState()






