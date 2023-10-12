import taichi as ti
import meshtaichi_patcher as Patcher
@ti.data_oriented
class Solver:
    def __init__(self,
                 my_mesh,
                 static_mesh,
                 bottom,
                 k=1e3,
                 dt=1e-3,
                 max_iter=1000):
        self.my_mesh = my_mesh
        self.static_mesh = static_mesh
        self.k = k
        self.dt = dt
        self.dtSq = dt ** 2
        self.max_iter = max_iter
        self.gravity = -4.0
        self.bottom = bottom
        self.id3 = ti.math.mat3([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])

        self.verts = self.my_mesh.mesh.verts
        self.edges = self.my_mesh.mesh.edges

        self.radius = 0.005
        self.contact_stiffness = 1e3
        self.damping_factor = 0.001
        self.grid_n = 128
        self.grid_particles_list = ti.field(ti.i32)
        self.grid_block = ti.root.dense(ti.ijk, (self.grid_n, self.grid_n, self.grid_n))
        self.partical_array = self.grid_block.dynamic(ti.l, len(self.my_mesh.mesh.verts))
        self.partical_array.place(self.grid_particles_list)
        self.grid_particles_count = ti.field(ti.i32)
        ti.root.dense(ti.ijk, (self.grid_n, self.grid_n, self.grid_n)).place(self.grid_particles_count)

        self.A = ti.linalg.SparseMatrix(n=3 * len(self.verts), m=3 * len(self.verts), dtype=ti.f32)
        # self.construct_collision_grid()

        print(f"verts #: {len(self.my_mesh.mesh.verts)}, elements #: {len(self.my_mesh.mesh.edges)}")
        # self.setRadius()
        print(f"radius: {self.radius}")

    @ti.kernel
    def computeY(self):
        for v in self.verts:
            v.y = v.x + v.v * self.dt + (v.f_ext / v.m) * self.dtSq

    @ti.kernel
    def computeNextState(self):
        for v in self.verts:
            v.v = (v.x_k - v.x) / self.dt
            v.x = v.x_k

    @ti.kernel
    def evaluateMomentumGradientAndHessian(self):
        for v in self.verts:
            v.g = v.m * (v.x_k - v.y)
            v.h = v.m

    @ti.kernel
    def evaluateElasticEnergyGradientAndHessian(self):
        for e in self.edges:
            x_ij = e.verts[0].x_k - e.verts[1].x_k
            l_ij = x_ij.norm()
            normal = x_ij.normalized(1e-12)
            coeff = self.dtSq * self.k
            e.verts[0].g += coeff * (l_ij - e.l0) * normal
            e.verts[1].g -= coeff * (l_ij - e.l0) * normal

            e.verts[0].h += coeff * self.id3
            e.verts[1].h += coeff * self.id3
            e.hij = -coeff * self.id3

    @ti.kernel
    def evaluateCollisionEnergyGradientAndHessian(self):
        for v in self.verts:
            if(v.x_k[1] < 0):
                depth = v.x_k[1] - self.bottom
                up = ti.math.vec3(0, 1, 0)
                v.g += self.dtSq * self.contact_stiffness * depth * up
                v.h += self.dtSq * self.contact_stiffness * self.id3


    @ti.kernel
    def matrix_free_Ax(self, ans: ti.template(), x: ti.template()):

        for v in self.verts:
            ans[v.id] = v.h @ x[v.id]

        for e in self.edges:
            v0, v1 = e.verts[0], e.verts[1]
            ans[v0.id] += e.hij @ x[v0.id]
            ans[v1.id] += e.hij @ x[v1.id]

    @ti.kernel
    def dot(self, a: ti.template(), b: ti.template()) -> ti.f32:
        ans = 0.0
        ti.loop_config(block_dim=32)
        for i in a:
            ans += a[i].dot(b[i])
        return ans

    @ti.kernel
    def add(self, ans: ti.template(), a: ti.template(), k: ti.f32, b: ti.template()):
        for i in ans:
            ans[i] = a[i] + k * b[i]


    @ti.kernel
    def CGoneIter(self, r_Sq: ti.f32) -> ti.f32:

        # matrix-free Ap
        # for v in self.verts:
        #     v.Ap = v.h @ v.p
        # for e in self.edges:
        #     v0, v1 = e.verts[0], e.verts[1]
        #     v0.Ap += e.hij @ v0.dx
        #     v1.Ap += e.hij @ v1.dx

        pTAp = 0.0
        # ti.loop_config(block_dim=32)
        # for v in self.verts:
        #     pTAp += ti.math.dot(v.p, v.Ap)

        alpha = r_Sq / pTAp

        for v in self.verts:
            v.dx += alpha * v.p
            v.r -= alpha * v.Ap

        r_Sq_next = 0.0
        # ti.loop_config(block_dim=32)
        # for v in self.verts:
        #     r_Sq_next += ti.math.dot(v.r, v.r)

        beta = r_Sq_next / r_Sq
        for v in self.verts:
            v.p = v.r + beta * v.p

        return r_Sq_next

    @ti.kernel
    def test(self):
        for v in self.verts:
            v.x_k -= v.h.inverse()@ v.g

    def NewtonCG(self):

        # max_cg_iter = 5
        # self.verts.b.copy_from(self.verts.g)
        # self.verts.r.copy_from(self.verts.g)
        # self.verts.p.copy_from(self.verts.g)
        #
        # self.verts.Ap.fill(0.)
        # self.verts.dx.fill(0.)
        #
        # r_Sq = self.dot(self.verts.r, self.verts.r)
        # threshold = 1e-4
        #
        # for i in range(1, max_cg_iter):
        #     r_Sq_next = self.CGoneIter(r_Sq)
        #     if r_Sq_next < threshold:
        #         break
        #     r_Sq = r_Sq_next

        self.test()

    def update(self):
        self.verts.f_ext.fill([0.0, self.gravity, 0.0])
        self.computeY()
        self.verts.x_k.copy_from(self.verts.y)
        for i in range(self.max_iter):
            self.evaluateMomentumGradientAndHessian()
            self.evaluateElasticEnergyGradientAndHessian()
            self.evaluateCollisionEnergyGradientAndHessian()
            self.NewtonCG()

        self.computeNextState()


