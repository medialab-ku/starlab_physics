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
        self.faces = self.my_mesh.mesh.faces

        self.verts_static = self.static_mesh.mesh.verts
        self.edges_static = self.static_mesh.mesh.edges
        self.num_edges_static = len(self.edges_static)
        self.faces_static = self.static_mesh.mesh.faces

        self.radius = 0.001
        self.contact_stiffness = 1e3
        self.damping_factor = 0.001
        self.grid_n = 128
        self.grid_particles_list = ti.field(ti.i32)
        self.grid_block = ti.root.dense(ti.ijk, (self.grid_n, self.grid_n, self.grid_n))
        self.partical_array = self.grid_block.dynamic(ti.l, len(self.my_mesh.mesh.verts))
        self.partical_array.place(self.grid_particles_list)
        self.grid_particles_count = ti.field(ti.i32)
        ti.root.dense(ti.ijk, (self.grid_n, self.grid_n, self.grid_n)).place(self.grid_particles_count)

        # self.A = ti.linalg.SparseMatrix(n=3 * len(self.verts), m=3 * len(self.verts), dtype=ti.f32)
        # self.construct_collision_grid()

        self.p1 = ti.math.vec3([0., 0., 0.])
        self.p2 = ti.math.vec3([0., 0., 0.])

        self.p = ti.Vector.field(n=3, shape=2, dtype=ti.f32)

        print(f"verts #: {len(self.my_mesh.mesh.verts)}, elements #: {len(self.my_mesh.mesh.edges)}")
        # self.setRadius()
        print(f"radius: {self.radius}")

        print(f'{self.edges.vid[0]}')
        print(f'{self.edges_static.vid[4]}')


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
    def evaluateMomentumConstraint(self):
        for v in self.verts:
            v.g = v.m * (v.x_k - v.y)
            v.h = v.m

    @ti.kernel
    def evaluateSpringConstraint(self):
        for e in self.edges:
            x_ij = e.verts[0].x_k - e.verts[1].x_k
            l_ij = x_ij.norm()
            C = 0.5 * (l_ij - e.l0) ** 2
            nablaC = (1 - e.l0 / l_ij) * x_ij
            Schur = (1./e.verts[0].m + 1./e.verts[1].m) * ti.math.dot(nablaC, nablaC)

            ld = 0.0
            if Schur > 1e-4:
                ld = C / Schur

            e.verts[0].g += ld * nablaC
            e.verts[1].g -= ld * nablaC
            e.verts[0].h += ld
            e.verts[1].h += ld


    @ti.kernel
    def evaluateCollisionConstraint(self):

        # for e in self.edges:
        #     for es in range(self.num_edges_static):
        a, b = self.verts.x_k[1], self.verts.x_k[2]
        # cid, did = self.edges_static.vid[es][0], self.edges_static.vid[es][1]
        c, d = self.verts_static.x[0], self.verts_static.x[3]

        ab = b - a
        cd = d - c
        ac = c - a

        a11 = ab.dot(ab)
        a12 = -cd.dot(ab)
        a21 = cd.dot(ab)
        a22 = -cd.dot(cd)
        det = a11 * a22 - a12 * a21
        mat = ti.math.mat2([[ab.dot(ab), -cd.dot(ab)], [cd.dot(ab), -cd.dot(cd)]])

        #
        gg = ti.math.vec2([ab.dot(ac), cd.dot(ac)])

        t = ti.math.vec2([0.0, 0.0])
        if abs(det) > 1e-4:
            t = mat.inverse() @ gg


        #
        # s = (a22 * gg[0] - a12 * gg[1]) / det
        # t = (-a21 * gg[0] + a11 * gg[1]) / det
        # t2 = ti.min(1, ti.max(t2, 0))

        t1 = t[0]
        t2 = t[1]

        if t1 < 0.0:
            t1 = 0.0

        if t1 > 1.0:
            t1 = 1.0

        if t2 < 0.0:
            t2 = 0.0

        if t2 > 1.0:
            t2 = 1.0

        p1 = a + t1 * ab
        p2 = c + t2 * cd


        # self.p[0] = p1
        # self.p[1] = p2
        # p01 = self.p[0] - self.p[1]

        # print(f'{p01.dot(ab)}, {p01.dot(cd)}')

        dist = (p1 - p2).norm()
        # print(dist)
        tol = 1e-2
        if dist < tol:
            # print("test")
            C = 0.5 * (dist - tol) ** 2
            n = (p1 - p2).normalized(1e-6)
            nablaC_a = (dist - tol) * t1 * n
            nablaC_b = (dist - tol) * (1 - t1) * n
            Schur = nablaC_a.dot(nablaC_a) / self.verts.m[1] + nablaC_b.dot(nablaC_b) / self.verts.m[2]

            kc = 1e3

            ld = C / Schur

            self.verts.g[1] += ld * nablaC_a
            self.verts.g[2] += ld * nablaC_b
            self.verts.h[1] += ld
            self.verts.h[2] += ld


        for v in self.verts:
            if(v.x_k[1] < 0):
                # v.x_k[1] = 0.0
                depth = v.x_k[1] - self.bottom
                C = 0.5 * depth ** 2
                nablaC = depth * ti.math.vec3(0, 1, 0)
                Schur = ti.math.dot(nablaC, nablaC) / v.h
                ld = C / Schur
                v.g += ld * nablaC
                v.h += ld



    @ti.kernel
    def NewtonCG(self):

        # for v in self.verts:
        #     v.h += v.ld

        # for e in self.edges:
        #     e.verts[0].h += e.ld
        #     e.verts[1].h += e.ld

        for v in self.verts:
            v.x_k -= v.g / v.h


    def update(self):
        self.verts.f_ext.fill([0.0, self.gravity, 0.0])
        self.computeY()

        self.verts.x_k.copy_from(self.verts.y)
        self.verts.h.copy_from(self.verts.m)

        for i in range(self.max_iter):
            self.evaluateMomentumConstraint()
            self.evaluateSpringConstraint()
            self.evaluateCollisionConstraint()
            self.NewtonCG()

        self.computeNextState()


