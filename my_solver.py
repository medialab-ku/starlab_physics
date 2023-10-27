import taichi as ti
import meshtaichi_patcher as Patcher

import ccd as ccd
import ipc_utils as cu
import barrier_functions as barrier

# @ti.dataclass
# class TP:
#     tid: ti.uint32
#     fid: ti.uint32



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

        self.rd_TP = ti.root.dynamic(ti.i, 1024, chunk_size=32)
        # self.rd_ee = ti.root.dynamic(ti.i, 1024, chunk_size=32)
        self.candidatesPT = ti.field(ti.math.ivec3)
        self.rd_TP.place(self.candidatesPT)
        # self.mmcvid_ee = ti.field(ti.math.ivec2)
        # self.S.place(self.mmcvid)
        # self.S.place(self.mmcvid_ee)
        self.dHat = 1e-4
        # self.test()
        #
        # self.normals = ti.Vector.field(n=3, dtype = ti.f32, shape = 2 * self.num_faces)
        self.normals_static = ti.Vector.field(n=3, dtype=ti.f32, shape=2 * self.num_faces_static)

        self.radius = 0.01
        self.contact_stiffness = 1e3
        self.damping_factor = 0.0
        self.grid_n = 128
        self.grid_particles_list = ti.field(ti.i32)
        self.grid_block = ti.root.dense(ti.ijk, (self.grid_n, self.grid_n, self.grid_n))
        self.partical_array = self.grid_block.dynamic(ti.l, len(self.my_mesh.mesh.verts))
        self.partical_array.place(self.grid_particles_list)
        self.grid_particles_count = ti.field(ti.i32)
        ti.root.dense(ti.ijk, (self.grid_n, self.grid_n, self.grid_n)).place(self.grid_particles_count)
        self.x_t = ti.Vector.field(n=3, dtype=ti.f32, shape=len(self.verts))

        self.dist_tol = 1e-2

        self.p1 = ti.math.vec3([0., 0., 0.])
        self.p2 = ti.math.vec3([0., 0., 0.])
        self.alpha = ti.math.vec3([0., 0., 0.])

        self.p = ti.Vector.field(n=3, shape=2, dtype=ti.f32)

        self.intersect = ti.Vector.field(n=3, dtype=ti.f32, shape=len(self.verts))

        print(f"verts #: {len(self.my_mesh.mesh.verts)}, edges #: {len(self.my_mesh.mesh.edges)} faces #: {len(self.my_mesh.mesh.faces)}")
        # self.setRadius()
        # print(f"radius: {self.radius}")
        #
        # print(f'{self.edges.vid}')
        # print(f'{self.edges_static.vid[4]}')
        # self.reset()

        # for PCG
        self.b = ti.Vector.field(3, dtype=ti.f32, shape=self.num_verts)
        self.r = ti.Vector.field(3, dtype=ti.f32, shape=self.num_verts)
        self.p = ti.Vector.field(3, dtype=ti.f32, shape=self.num_verts)
        self.Ap = ti.Vector.field(3, dtype=ti.f32, shape=self.num_verts)
        self.z = ti.Vector.field(3, dtype=ti.f32, shape=self.num_verts)
        self.mul_ans = ti.Vector.field(3, dtype=ti.f32, shape=self.num_verts)
        test = 9 * self.num_verts ** 2
        # self.K = ti.linalg.SparseMatrixBuilder(3 * self.num_verts, 3 * self.num_verts)
        # self.A = ti.linalg.SparseMatrix(3 * self.num_verts, 3 * self.num_verts)
        # self.solver = ti.linalg.SparseSolver(solver_type="LLT")
        self.test_x = ti.ndarray(ti.f32, 3 * self.num_verts)
        self.grad = ti.ndarray(ti.f32, 3 * self.num_verts)
        # self.test()



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
                    self.candidatesPT.append(ti.math.ivec3(v.id, tid, 0))

        # print(self.candidatesPT.length())

    @ti.kernel
    def computeVtemp(self):
        for v in self.verts:
            if v.id == 61 or v.id == 78:
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
    def computeNextState(self):
        for v in self.verts:
            v.v = (1.0 - self.damping_factor) * (v.x_k - v.x) / self.dt
            v.x = v.x_k

        center = ti.math.vec3(0.5, 0.5, 0.5)
        rad = 0.4
        # # coef = self.dtSq * 1e6
        # for v in self.verts:
        #     dist = (v.x - center).norm()
        #     nor = (v.x - center).normalized(1e-12)
        #     if dist < rad:
        #         v.x = center + rad * nor

    @ti.kernel
    def evaluateMomentumConstraint(self):
        for v in self.verts:
            v.g = v.m * (v.x_k - v.y)
            v.h = v.m

    @ti.kernel
    def evaluate_gradient_and_hessian(self):

        for e in self.edges:
            ei, ej = e.verts[0].id, e.verts[1].id
            xij = e.verts[0].x_k - e.verts[1].x_k
            lij = xij.norm()
            coef = self.dtSq * self.k
            grad = coef * (xij - e.l0 * xij.normalized(1e-6))
            dir = (e.verts[0].x_k - e.verts[1].x_k).normalized(1e-4)
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

        # for cid in self.candidatesPT:
        #     pid, fid, d_type = self.candidatesPT[cid]
        #     ld = self.compute_gradient_and_hessian_PT(pid, fid)

    @ti.kernel
    def compute_search_dir(self):
        for v in self.verts:
            v.dx = v.g / v.h


    @ti.kernel
    def step_forward(self, step_size: ti.f32):
        for v in self.verts:
            if v.id == 61 or v.id == 78:
                v.x_k = v.x_k
            else:
                v.x_k += step_size * v.dx

        center = ti.math.vec3(0.5, 0.5, 0.5)
        rad = 0.4
        # coef = self.dtSq * 1e6
        # for v in self.verts:
        #     dist = (v.x_k - center).norm()
        #     nor = (v.x_k - center).normalized(1e-12)
        #     if dist < rad:
        #         v.x_k = center + rad * nor



    @ti.kernel
    def compute_aabb(self):

        padding_size = 1e-1
        padding = ti.math.vec3([padding_size, padding_size, padding_size])

        for v in self.verts:
            dx = v.v * self.dt
            v.aabb_max = ti.max(v.x + dx, v.x - dx)
            v.aabb_min = ti.max(v.x + dx, v.x - dx)

        for f in self.faces:
            x0, x1, x2 = f.verts[0].x, f.verts[1].x, f.verts[2].x
            v0, v1, v2 = f.verts[0].v, f.verts[1].v, f.verts[2].v

            dx0 = v0 * self.dt
            dx1 = v1 * self.dt
            dx2 = v2 * self.dt

            f.aabb_max = ti.max(x0 + dx0, x0 - dx0, x1 + dx1, x1 - dx1, x2 + dx2, x2 - dx2)
            f.aabb_min = ti.min(x0 + dx0, x0 - dx0, x1 + dx1, x1 - dx1, x2 + dx2, x2 - dx2)

        for f in self.faces_static:
            x0, x1, x2 = f.verts[0].x, f.verts[1].x, f.verts[2].x

            f.aabb_max = ti.max(x0, x1, x2) + padding
            f.aabb_min = ti.min(x0, x1, x2) - padding



    @ti.func
    def compute_gradient_and_hessian_PT(self, pid: ti.int32, tid: ti.int32):

        v0 = pid
        v1 = self.face_indices_static[3 * tid + 0]
        v2 = self.face_indices_static[3 * tid + 1]
        v3 = self.face_indices_static[3 * tid + 2]


        x0 = self.verts.x_k[v0]
        x1 = self.verts_static.x[v1]   #r
        x2 = self.verts_static.x[v2]   #g
        x3 = self.verts_static.x[v3]   #c

        ld = 0.0
        dtype = cu.d_type_PT(x0, x1, x2, x3)
        if dtype == 0:           #r
            d = cu.d_PP(x0, x1)
            if d < self.dHat:
                g0, g1 = cu.g_PP(x0, x1)

                ld = barrier.compute_g_b(d, self.dHat)
                sch = g0.dot(g0) / self.verts.h[v0]
                ld = (d - self.dHat) / sch
                self.verts.g[v0] += ld * g0
                self.verts.h[v0] += ld



        elif dtype == 1:
            d = cu.d_PP(x0, x2)  #g
            if d < self.dHat:
                g0, g2 = cu.g_PP(x0, x2)

                ld = barrier.compute_g_b(d, self.dHat)
                sch = g0.dot(g0) / self.verts.h[v0]
                ld = (d - self.dHat) / sch
                self.verts.g[v0] += ld * g0
                self.verts.h[v0] += ld

        elif dtype == 2:
            d = cu.d_PP(x0, x3) #c
            if d < self.dHat:
                g0, g3 = cu.g_PP(x0, x3)

                ld = barrier.compute_g_b(d, self.dHat)
                sch = g0.dot(g0) / self.verts.h[v0]
                ld = (d - self.dHat) / sch
                self.verts.g[v0] += ld * g0
                self.verts.h[v0] += ld


        elif dtype == 3:
            d = cu.d_PE(x0, x1, x2) # r-g
            if d < self.dHat:
                g0, g1, g2 = cu.g_PE(x0, x1, x2)
                ld = barrier.compute_g_b(d, self.dHat)
                sch = g0.dot(g0) / self.verts.h[v0]
                ld = (d - self.dHat) / sch
                self.verts.g[v0] += ld * g0
                self.verts.h[v0] += ld

        elif dtype == 4:
            d = cu.d_PE(x0, x2, x3) #g-c
            if d < self.dHat:
                g0, g2, g3 = cu.g_PE(x0, x1, x2)
                ld = barrier.compute_g_b(d, self.dHat)
                sch = g0.dot(g0) / self.verts.h[v0]
                ld = (d - self.dHat) / sch
                self.verts.g[v0] += ld * g0
                self.verts.h[v0] += ld


        elif dtype == 5:
            d = cu.d_PE(x0, x3, x1) #c-r
            if d < self.dHat:
                g0, g3, g1 = cu.g_PE(x0, x3, x1)

                ld = barrier.compute_g_b(d, self.dHat)
                sch = g0.dot(g0) / self.verts.h[v0]
                ld = (d - self.dHat) / sch
                self.verts.g[v0] += ld * g0
                self.verts.h[v0] += ld


        elif dtype == 6:            # inside triangle
            d = cu.d_PT(x0, x1, x2, x3)
            if d < self.dHat:
                g0, g1, g2, g3 = cu.g_PT(x0, x1, x2, x3)
                ld = barrier.compute_g_b(d, self.dHat)
                sch = g0.dot(g0) / self.verts.h[v0]
                ld = (d - self.dHat) / sch
                self.verts.g[v0] += ld * g0
                self.verts.h[v0] += ld


        return ld

    @ti.func
    def compute_constraint_energy_PT(self, x: ti.template(), pid: ti.int32, tid: ti.int32) -> ti.f32:

        energy = 0.0
        v0 = pid
        v1 = self.face_indices_static[3 * tid + 0]
        v2 = self.face_indices_static[3 * tid + 1]
        v3 = self.face_indices_static[3 * tid + 2]


        x0 = x[v0]
        x1 = self.verts_static.x[v1]   #r
        x2 = self.verts_static.x[v2]   #g
        x3 = self.verts_static.x[v3]   #c

        dtype = cu.d_type_PT(x0, x1, x2, x3)
        if dtype == 0:           #r
            d = cu.d_PP(x0, x1)
            if d < self.dHat:
                g0, g1 = cu.g_PP(x0, x1)

                ld = barrier.compute_g_b(d, self.dHat)
                sch = g0.dot(g0) / self.verts.h[v0]
                ld = (d - self.dHat) / sch
                energy = 0.5 * ld * (d - self.dHat)

        elif dtype == 1:
            d = cu.d_PP(x0, x2)  #g
            if d < self.dHat:
                g0, g2 = cu.g_PP(x0, x2)

                ld = barrier.compute_g_b(d, self.dHat)
                sch = g0.dot(g0) / self.verts.h[v0]
                ld = (d - self.dHat) / sch
                energy = 0.5 * ld * (d - self.dHat)

        elif dtype == 2:
            d = cu.d_PP(x0, x3) #c
            if d < self.dHat:
                g0, g3 = cu.g_PP(x0, x3)
                ld = barrier.compute_g_b(d, self.dHat)
                sch = g0.dot(g0) / self.verts.h[v0]
                ld = (d - self.dHat) / sch
                energy = 0.5 * ld * (d - self.dHat)

        elif dtype == 3:
            d = cu.d_PE(x0, x1, x2) # r-g
            if d < self.dHat:
                g0, g1, g2 = cu.g_PE(x0, x1, x2)
                ld = barrier.compute_g_b(d, self.dHat)
                sch = g0.dot(g0) / self.verts.h[v0]
                ld = (d - self.dHat) / sch
                energy = 0.5 * ld * (d - self.dHat)

        elif dtype == 4:
            d = cu.d_PE(x0, x2, x3) #g-c
            if d < self.dHat:
                g0, g2, g3 = cu.g_PE(x0, x1, x2)
                ld = barrier.compute_g_b(d, self.dHat)
                sch = g0.dot(g0) / self.verts.h[v0]
                ld = (d - self.dHat) / sch
                energy = 0.5 * ld * (d - self.dHat)

        elif dtype == 5:
            d = cu.d_PE(x0, x3, x1) #c-r
            if d < self.dHat:
                g0, g3, g1 = cu.g_PE(x0, x3, x1)

                ld = barrier.compute_g_b(d, self.dHat)
                sch = g0.dot(g0) / self.verts.h[v0]
                ld = (d - self.dHat) / sch
                energy = 0.5 * ld * (d - self.dHat)

        elif dtype == 6:            # inside triangle
            d = cu.d_PT(x0, x1, x2, x3)
            if d < self.dHat:
                g0, g1, g2, g3 = cu.g_PT(x0, x1, x2, x3)
                ld = barrier.compute_g_b(d, self.dHat)
                sch = g0.dot(g0) / self.verts.h[v0]
                ld = (d - self.dHat) / sch
                energy = 0.5 * ld * (d - self.dHat)

        return energy
    @ti.func
    def computeConstraintSet_TP(self, tid: ti.int32, pid: ti.int32):

        v0 = pid
        v1 = self.face_indices[3 * tid + 0]
        v2 = self.face_indices[3 * tid + 1]
        v3 = self.face_indices[3 * tid + 2]

        # print(f'{v0}, {v1}, {v2}, {v3}')

        x0 = self.verts_static.x[v0]
        x1 = self.verts.x_k[v1]   #r
        x2 = self.verts.x_k[v2]   #g
        x3 = self.verts.x_k[v3]   #c

        dtype = cu.d_type_PT(x0, x1, x2, x3)
        if dtype == 0:           #r
            d = cu.d_PP(x0, x1)
            if d < self.dHat:
                g0, g1 = cu.g_PP(x0, x1)
                ld = barrier.compute_g_b(d, self.dHat)
                # sch = g1.dot(g1) / self.verts.h[v1]
                # ld = (d - self.dHat) / sch

                self.verts.g[v1] += ld * g1
                self.verts.h[v1] += ld

                # self.mmcvid.append(ti.math.ivec4([-v1-1, v0, -1, -1]))

        elif dtype == 1:
            d = cu.d_PP(x0, x2)  #g
            if d < self.dHat:
                g0, g2 = cu.g_PP(x0, x2)
                ld = barrier.compute_g_b(d, self.dHat)
                # sch = g2.dot(g2) / self.verts.h[v2]
                # ld = (d - self.dHat) / sch

                self.verts.g[v2] += ld * g2
                self.verts.h[v2] += ld

        elif dtype == 2:
            d = cu.d_PP(x0, x3) #c
            if d < self.dHat:
                g0, g3 = cu.g_PP(x0, x3)
                ld = barrier.compute_g_b(d, self.dHat)
                # sch = g3.dot(g3) / self.verts.h[v3]
                # ld = (d - self.dHat) / sch

                self.verts.g[v3] += ld * g3
                self.verts.h[v3] += ld

        # self.mmcvid.append(ti.math.ivec4([-v3-1, v0, -1, -1]))

        elif dtype == 3:
            d = cu.d_PE(x0, x1, x2) # r-g
            if d < self.dHat:
                g0, g1, g2 = cu.g_PE(x0, x1, x2)
                ld = barrier.compute_g_b(d, self.dHat)
                # sch = g1.dot(g1) / self.verts.h[v1] + g2.dot(g2) / self.verts.h[v2]
                # ld = (d - self.dHat) / sch

                self.verts.g[v1] += ld * g1
                self.verts.g[v2] += ld * g2
                self.verts.h[v1] += ld
                self.verts.h[v2] += ld
                # self.mmcvid.append(ti.math.ivec4([-v1-1, -v2-1, v0, -1]))

        elif dtype == 4:
            d = cu.d_PE(x0, x2, x3) #g-c
            if d < self.dHat:
                g0, g2, g3 = cu.g_PE(x0, x2, x3)
                ld = barrier.compute_g_b(d, self.dHat)
                # sch = g2.dot(g2) / self.verts.h[v2] + g3.dot(g3) / self.verts.h[v3]
                # ld = (d - self.dHat) / sch

                self.verts.g[v2] += ld * g2
                self.verts.g[v3] += ld * g3

                self.verts.h[v2] += ld
                self.verts.h[v3] += ld
                # self.mmcvid.append(ti.math.ivec4([-v2-1, -v3-1, v0, -1]))

        elif dtype == 5:
            d = cu.d_PE(x0, x3, x1) #c-r
            if d < self.dHat:
                g0, g3, g1 = cu.g_PE(x0, x3, x1)
                ld = barrier.compute_g_b(d, self.dHat)
                # sch = g1.dot(g1) / self.verts.h[v2] + g3.dot(g3) / self.verts.h[v3]
                # ld = (d - self.dHat) / sch

                self.verts.g[v1] += ld * g1
                self.verts.g[v3] += ld * g3

                self.verts.h[v1] += ld
                self.verts.h[v3] += ld
                # self.mmcvid.append(ti.math.ivec4([-v3-1, -v1-1, v0, -1]))

        elif dtype == 6:            # inside triangle
            d = cu.d_PT(x0, x1, x2, x3)
            if d < self.dHat:
                g0, g1, g2, g3 = cu.g_PT(x0, x1, x2, x3)
                ld = barrier.compute_g_b(d, self.dHat)
                # sch = g1.dot(g1) / self.verts.h[v1] + g2.dot(g2) / self.verts.h[v2] + g3.dot(g3) / self.verts.h[v3]
                # ld = (d - self.dHat) / sch

                self.verts.g[v1] += ld * g1
                self.verts.g[v2] += ld * g2
                self.verts.g[v3] += ld * g3

                self.verts.h[v1] += ld
                self.verts.h[v2] += ld
                self.verts.h[v3] += ld
                # self.mmcvid.append(ti.math.ivec4([-v1-1, -v2-1, -v3-1, v0]))

    @ti.func
    def computeConstraintSet_EE(self, eid0: ti.int32, eid1: ti.int32):

        v0 = self.edges.vid[eid0][0]
        v1 = self.edges.vid[eid0][1]
        v2 = self.edges_static.vid[eid1][0]
        v3 = self.edges_static.vid[eid1][1]

        x0 = self.verts.x_k[v0]
        x1 = self.verts.x_k[v1]
        x2 = self.verts_static.x[v2]
        x3 = self.verts_static.x[v3]

        d_type = cu.d_type_EE(x0, x1, x2, x3)
        x01 = x1-x0
        x32 = x2-x3
        # print(d_type)
        is_para = False
        if x01.cross(x32).norm() < 1e-3:
            is_para = True

        if is_para:
            print("para")                                                 
        # print(f'{d_type}, {is_para}')

        if d_type == 0:
            d = cu.d_PP(x0, x2)
            if(d < self.dHat):
                # print(d_type)
                g0, g2 = cu.g_PP(x0, x2)
                sch = g0.dot(g0) / self.verts.h[v0]
                ld = (d - self.dHat) / sch
                # p0 = x0 - step_size * g0

                self.verts.g[v0] += ld * g0
                self.verts.h[v0] += ld

        elif d_type == 1:
            d = cu.d_PP(x0, x3)
            if (d < self.dHat):
                # print(d_type)
                g0, g3 = cu.g_PP(x0, x3)
                sch = g0.dot(g0) / self.verts.h[v0]
                ld = (d - self.dHat) / sch
                # p0 = x0 - step_size * g0

                self.verts.g[v0] += ld * g0
                self.verts.h[v0] += ld

        elif d_type == 2:
            d = cu.d_PE(x0, x2, x3)
            if (d < self.dHat):
                # print(d_type)
                g0, g1, g2 = cu.g_PE(x0, x2, x3)
                sch = g0.dot(g0) / self.verts.h[v0] + g1.dot(g1) / self.verts.h[v1]
                step_size = (d - self.dHat) / sch
                ld = (d - self.dHat) / sch
                # p0 = x0 - step_size * g0

                self.verts.g[v0] += ld * g0
                self.verts.g[v1] += ld * g0
                self.verts.h[v0] += ld
                self.verts.h[v1] += ld

        elif d_type == 3:
            d = cu.d_PP(x1, x2)
            if (d < self.dHat):
                # print(d_type)
                g1, g2 = cu.g_PP(x1, x2)
                sch = g1.dot(g1) / self.verts.h[v1]
                ld = (d - self.dHat) / sch
                # p0 = x0 - step_size * g0

                self.verts.g[v1] += ld * g1
                self.verts.h[v1] += ld

        elif d_type == 4:
            d = cu.d_PP(x1, x3)
            if (d < self.dHat):
                # print(d_type)
                g1, g3 = cu.g_PP(x1, x3)
                sch = g1.dot(g1) / self.verts.h[v1]
                ld = (d - self.dHat) / sch
                # p0 = x0 - step_size * g0

                self.verts.g[v1] += ld * g1
                self.verts.h[v1] += ld

        elif d_type == 5:
            d = cu.d_PE(x1, x2, x3)
            if (d < self.dHat):
                # print(d_type)
                g1, g2, g3 = cu.g_PE(x1, x2, x3)
                sch = g1.dot(g1) / self.verts.h[v1]
                ld = (d - self.dHat) / sch
                # p0 = x0 - step_size * g0

                self.verts.g[v1] += ld * g1
                self.verts.h[v1] += ld

        elif d_type == 6:
            d = cu.d_PE(x2, x0, x1)
            if (d < self.dHat):
                # print(d_type)
                g2, g0, g1 = cu.g_PE(x2, x0, x1)
                sch = g0.dot(g0) / self.verts.h[v0] + g1.dot(g1) / self.verts.h[v1]
                ld = (d - self.dHat) / sch
                # p0 = x0 - step_size * g0

                self.verts.g[v0] += ld * g0
                self.verts.g[v1] += ld * g0
                self.verts.h[v0] += ld
                self.verts.h[v1] += ld

        elif d_type == 7:
            d = cu.d_PE(x3, x0, x1)
            if (d < self.dHat):
                # print(d_type)
                g3, g0, g1 = cu.g_PE(x3, x0, x1)
                sch = g0.dot(g0) / self.verts.h[v0] + g1.dot(g1) / self.verts.h[v1]
                ld = (d - self.dHat) / sch
                # p0 = x0 - step_size * g0

                self.verts.g[v0] += ld * g0
                self.verts.g[v1] += ld * g0
                self.verts.h[v0] += ld
                self.verts.h[v1] += ld

        elif d_type == 8:
            d = cu.d_EE(x0, x1, x2, x3)
            # print(d)
            if (d < self.dHat):
                # print("test")
                # if is_para:
                #     eps_x = cu.compute_eps_x(x0, x1, x2, x3)
                #     e = cu.compute_e(x0, x1, x2, x3, eps_x)
                #     g0, g1, g2, g3 = cu.compute_e_g(x0, x1, x2, x3, eps_x)
                #     sch = g0.dot(g0) / self.verts.h[v0] + g1.dot(g1) / self.verts.h[v1]
                #
                #     # ld = 0
                #     # if abs(sch) > 1e-6:
                #     ld = (d - self.dHat) / sch
                #
                #     self.verts.g[v0] += ld * g0
                #     self.verts.g[v1] += ld * g1
                #
                #     self.verts.h[v0] += ld
                #     self.verts.h[v1] += ld
                # else:

                g0, g1, g2, g3 = cu.g_EE(x0, x1, x2, x3)
                sch = g0.dot(g0) / self.verts.h[v0] + g1.dot(g1) / self.verts.h[v1]

                # ld = 0
                # if abs(sch) > 1e-6:
                ld = (d - self.dHat) / sch

                self.verts.g[v0] += ld * g0
                self.verts.g[v1] += ld * g1

                self.verts.h[v0] += ld
                self.verts.h[v1] += ld


    @ti.kernel
    def computeConstraintSet(self):

        for cid in self.candidatesPT:
            pid, fid, d_type = self.candidatesPT[cid]
            d_type = self.compute_gradient_and_hessian_PT(pid, fid)
            self.candidatesPT[cid][2] = d_type


    @ti.kernel
    def ccd_alpha(self) -> ti.f32:
        alpha = 1.0
        for cid in self.candidatesPT:
            pid, fid, d_type = self.candidatesPT[cid]
            x0 = self.verts.x_k[pid]
            dx0 = self.verts.dx[fid]

            v1 = self.face_indices_static[3 * fid + 0]
            v2 = self.face_indices_static[3 * fid + 1]
            v3 = self.face_indices_static[3 * fid + 2]

            x1 = self.verts_static.x[v1]
            x2 = self.verts_static.x[v2]
            x3 = self.verts_static.x[v3]

            dx_zero = ti.math.vec3([0.0, 0.0, 0.0])

            alpha_ccd = ccd.point_triangle_ccd(x0, x1, x2, x3, dx0, dx_zero, dx_zero, dx_zero, 0.1, self.dHat, 1.0)
            # print(alpha_ccd)
            if alpha > alpha_ccd:
                alpha = alpha_ccd

        return alpha

    def line_search(self):

        alpha = self.ccd_alpha()
        # print(alpha)
        e_cur = self.compute_spring_energy(self.verts.x_k) + self.compute_collision_energy(self.verts.x_k)
        # print(e_cur)
        for i in range(5):
            self.add(self.x_t, self.verts.x_k, alpha, self.verts.dx)
            e = self.compute_spring_energy(self.x_t) + self.compute_collision_energy(self.x_t)
            # print(e)
            if(e_cur < e):
                alpha /= 2.0
            else:
                # print(i)
                break
        return alpha

    @ti.kernel
    def compute_collision_energy(self, x: ti.template()) -> ti.f32:

        collision_e_total = 0.0
        for cid in self.candidatesPT:
            pid, fid, d_type = self.candidatesPT[cid]
            collision_e_total += self.compute_constraint_energy_PT(x, pid, fid)
        return collision_e_total
    @ti.kernel
    def compute_spring_energy(self, x: ti.template()) -> ti.f32:

        spring_e_total = 0.0
        for e in self.edges:
            v0, v1 = e.verts[0].id, e.verts[1].id
            xij = x[v0] - x[v1]
            l = xij.norm()
            coeff = 0.5 * self.dtSq * self.k
            spring_e_total += coeff * (l - e.l0) ** 2

        return spring_e_total

    @ti.kernel
    def modify_velocity(self):

        alpha = 1.0
        for cid in self.candidatesPT:
            pid, fid, d_type = self.candidatesPT[cid]
            x0 = self.verts.x[pid]
            dx0 = self.verts.y[pid] - x0

            v1 = self.face_indices_static[3 * fid + 0]
            v2 = self.face_indices_static[3 * fid + 1]
            v3 = self.face_indices_static[3 * fid + 2]

            x1 = self.verts_static.x[v1]
            x2 = self.verts_static.x[v2]
            x3 = self.verts_static.x[v3]

            dx_zero = ti.math.vec3([0.0, 0.0, 0.0])

            alpha_ccd = ccd.point_triangle_ccd(x0, x1, x2, x3, dx0, dx_zero, dx_zero, dx_zero, 0.1, self.dHat, 1.0)
            # print(alpha_ccd)
            # if alpha > alpha_ccd:
            #     alpha = alpha_ccd

            if alpha_ccd < 1.0:
                dtype = cu.d_type_PT(x0, x1, x2, x3)
                if dtype == 0:  # r
                    d = cu.d_PP(x0, x1)
                    # if d < self.dHat:
                    g0, g1 = cu.g_PP(x0, x1)

                    ld = barrier.compute_g_b(d, self.dHat)
                    sch = g0.dot(g0) / self.verts.h[pid]
                    ld = (d) / sch

                    self.verts.g[pid] += ld * g0
                    self.verts.h[pid] += ld



                elif dtype == 1:
                    d = cu.d_PP(x0, x2)  # g
                    # if d < self.dHat:
                    g0, g2 = cu.g_PP(x0, x2)

                    ld = barrier.compute_g_b(d, self.dHat)
                    sch = g0.dot(g0) / self.verts.h[pid]
                    ld = (d) / sch

                    self.verts.g[pid] += ld * g0
                    self.verts.h[pid] += ld

                elif dtype == 2:
                    d = cu.d_PP(x0, x3)  # c
                    # if d < self.dHat:
                    g0, g3 = cu.g_PP(x0, x3)

                    ld = barrier.compute_g_b(d, self.dHat)
                    sch = g0.dot(g0) / self.verts.h[pid]
                    ld = (d) / sch

                    self.verts.g[pid] += ld * g0
                    self.verts.h[pid] += ld


                elif dtype == 3:
                    d = cu.d_PE(x0, x1, x2)  # r-g
                    # if d < self.dHat:
                    g0, g1, g2 = cu.g_PE(x0, x1, x2)
                    ld = barrier.compute_g_b(d, self.dHat)
                    sch = g0.dot(g0) / self.verts.h[pid]
                    ld = (d) / sch

                    self.verts.g[pid] += ld * g0
                    self.verts.h[pid] += ld


                elif dtype == 4:
                    d = cu.d_PE(x0, x2, x3)  # g-c
                    # if d < self.dHat:
                    g0, g2, g3 = cu.g_PE(x0, x1, x2)
                    ld = barrier.compute_g_b(d, self.dHat)
                    sch = g0.dot(g0) / self.verts.h[pid]
                    ld = (d) / sch

                    self.verts.g[pid] += ld * g0
                    self.verts.h[pid] += ld


                elif dtype == 5:
                    d = cu.d_PE(x0, x3, x1)  # c-r
                    # if d < self.dHat:
                    g0, g3, g1 = cu.g_PE(x0, x3, x1)

                    ld = barrier.compute_g_b(d, self.dHat)
                    sch = g0.dot(g0) / self.verts.h[pid]
                    ld = (d) / sch

                    self.verts.g[pid] += ld * g0
                    self.verts.h[pid] += ld

                elif dtype == 6:  # inside triangle
                    d = cu.d_PT(x0, x1, x2, x3)
                    # if d < self.dHat:
                    g0, g1, g2, g3 = cu.g_PT(x0, x1, x2, x3)
                    ld = barrier.compute_g_b(d, self.dHat)
                    sch = g0.dot(g0) / self.verts.h[pid]
                    ld = (d) / sch
                    self.verts.g[pid] += ld * g0
                    self.verts.h[pid] += ld



    @ti.kernel
    def apply_precondition(self, z: ti.template(), r: ti.template()):
        for i in z:
            z[i] = r[i] / self.verts.h[i]


    @ti.kernel
    def cg_iterate(self, r_2: ti.f32) -> ti.f32:
        # Ap = A * x
        for v in self.verts:
            self.Ap[v.id] = self.p[v.id] * v.m

        ti.mesh_local(self.Ap, self.p)
        for e in self.edges:
            u = e.verts[0].id
            v = e.verts[1].id
            d = e.hij @ (self.p[u] - self.p[v])
            self.Ap[u] += d
            self.Ap[v] -= d

        pAp = 0.0
        ti.loop_config(block_dim=32)
        for v in self.verts:
            pAp += self.p[v.id].dot(self.Ap[v.id])

        alpha = r_2 / pAp
        for v in self.verts:
            v.dx += alpha * self.p[v.id]
            self.r[v.id] -= alpha * self.Ap[v.id]

        for v in self.verts:
            self.z[v.id] = self.r[v.id] / v.h

        r_2_new = 0.0
        ti.loop_config(block_dim=32)
        for v in self.verts:
            r_2_new += self.z[v.id].dot(self.r[v.id])

        beta = r_2_new / r_2
        for v in self.verts:
            self.p[v.id] = self.z[v.id] + beta * self.p[v.id]

        return r_2_new


    @ti.kernel
    def matrix_free_Ax(self, x: ti.template()):
        for v in self.verts:
            self.Ap[v.id] = self.verts.m[v.id] * x[v.id]

        ti.mesh_local(self.Ap, x)
        for e in self.edges:
            u = e.verts[0].id
            v = e.verts[1].id
            d = e.hij @ (x[u] - x[v])
            self.Ap[u] += d
            self.Ap[v] -= d


    def NewtonPCG(self):


        self.verts.dx.fill(0.0)
        self.r.copy_from(self.verts.g)

        self.apply_precondition(self.z, self.r)
        self.p.copy_from(self.z)
        r_2 = self.dot(self.z, self.r)
        n_iter = 1000  # CG iterations
        epsilon = 1e-5

        r_2_new = r_2
        i = 0
        for iter in range(n_iter):
            i += 1
            r_2_new = self.cg_iterate(r_2_new)

            if r_2_new <= epsilon:
                break

        print(f'cg iter: {i}')
        # self.add(self.verts.x_k, self.verts.x_k, -1.0, self.verts.dx)


    def update(self):

        self.verts.f_ext.fill([0.0, self.gravity, 0.0])
        self.computeVtemp()

        self.compute_aabb()
        self.compute_candidates_PT()

        self.computeY()
        self.verts.x_k.copy_from(self.verts.y)

        for i in range(1):
            self.verts.g.fill(0.)
            self.verts.h.copy_from(self.verts.m)
            self.evaluate_gradient_and_hessian()
            # self.compute_search_dir()

            self.NewtonPCG()
            alpha = 1.0
            self.step_forward(alpha)

        self.computeNextState()






