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
        self.num_verts_static = len(self.static_mesh.mesh.verts)
        self.edges_static = self.static_mesh.mesh.edges
        self.num_edges_static = len(self.edges_static)
        self.faces_static = self.static_mesh.mesh.faces
        self.face_indices_static = self.static_mesh.face_indices
        self.num_faces_static = len(self.static_mesh.mesh.faces)
        #
        # self.normals = ti.Vector.field(n=3, dtype=ti.f32, shape=2 * self.num_faces)
        self.normals_static = ti.Vector.field(n=3, dtype=ti.f32, shape=2 * self.num_faces_static)

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

        for e in self.edges:
            for es in range(self.num_edges_static):
                aid, bid = e.verts[0].id, e.verts[1].id
                a, b = self.verts.x_k[aid], self.verts.x_k[bid]
                cid, did = self.edges_static.vid[es][0], self.edges_static.vid[es][1]
                c, d = self.verts_static.x[cid], self.verts_static.x[did]

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
                dist = (p1 - p2).norm()

                tol = 1e-2
                if dist < tol:

                    C = 0.5 * (dist - tol) ** 2
                    n = (p1 - p2).normalized(1e-6)
                    nablaC_a = (dist - tol) * (1 - t1) * n
                    nablaC_b = (dist - tol) * t1 * n
                    Schur = nablaC_a.dot(nablaC_a) / self.verts.m[aid] + nablaC_b.dot(nablaC_b) / self.verts.m[bid]

                    ld = C / Schur

                    self.verts.g[aid] += ld * nablaC_a
                    self.verts.g[bid] += ld * nablaC_b
                    self.verts.h[aid] += ld
                    self.verts.h[bid] += ld

        for v in self.verts:
            for fi in range(self.num_faces_static):
                fid0, fid1, fid2 = self.face_indices_static[fi * 3 + 0], self.face_indices_static[fi * 3 + 1], self.face_indices_static[fi * 3 + 2]

                f1 = self.verts_static.x[fid0]
                f2 = self.verts_static.x[fid1]
                f3 = self.verts_static.x[fid2]

                point = v.x_k

                e1 = f2 - f1
                e2 = f3 - f1
                normal = e1.cross(e2).normalized(1e-12)
                d = point - f1
                dist = d.dot(normal)
                point_on_triangle = point - dist * normal
                d1 = point_on_triangle - f1
                d2 = point_on_triangle - f2
                d3 = point_on_triangle - f3

                area_triangle = e1.cross(e2).norm() / 2
                area1 = d1.cross(d2).norm() / (2 * area_triangle)
                area2 = d2.cross(d3).norm() / (2 * area_triangle)
                area3 = d3.cross(d1).norm() / (2 * area_triangle)

                is_on_triangle = 0 <= area1 <= 1 and 0 <= area2 <= 1 and 0 <= area3 <= 1 and area1 + area2 + area3 == 1

                # weights (if the point_on_triangle is close to the vertex, the weight is large)
                w1 = 1 / (d1.norm() + 1e-8)
                w2 = 1 / (d2.norm() + 1e-8)
                w3 = 1 / (d3.norm() + 1e-8)
                total_w = w1 + w2 + w3
                w1 /= total_w
                w2 /= total_w
                w3 /= total_w

                tol = 1e-2
                if abs(dist) <= tol and is_on_triangle:
                    # print("test")
                    C = 0.5 * (dist - tol) ** 2
                    nablaC = (dist - tol) * normal
                    Schur = nablaC.dot(nablaC) / v.m

                    kc = 1e3
                    ld = C / Schur
                    v.g += ld * nablaC

                    v.h += ld

        for f in self.faces:
            for svid in range(self.num_verts_static):
                f1 = f.verts[0].x_k
                f2 = f.verts[1].x_k
                f3 = f.verts[2].x_k

                point = self.verts_static.x[svid]

                e1 = f2 - f1
                e2 = f3 - f1
                normal = e1.cross(e2).normalized(1e-12)
                d = point - f1
                dist = d.dot(normal)
                point_on_triangle = point - dist * normal
                d1 = point_on_triangle - f1
                d2 = point_on_triangle - f2
                d3 = point_on_triangle - f3

                area_triangle = e1.cross(e2).norm() / 2
                area1 = d1.cross(d2).norm() / (2 * area_triangle)
                area2 = d2.cross(d3).norm() / (2 * area_triangle)
                area3 = d3.cross(d1).norm() / (2 * area_triangle)

                is_on_triangle = 0 <= area1 <= 1 and 0 <= area2 <= 1 and 0 <= area3 <= 1 and area1 + area2 + area3 == 1

                # weights (if the point_on_triangle is close to the vertex, the weight is large)
                w1 = 1 / (d1.norm() + 1e-8)
                w2 = 1 / (d2.norm() + 1e-8)
                w3 = 1 / (d3.norm() + 1e-8)
                total_w = w1 + w2 + w3
                w1 /= total_w
                w2 /= total_w
                w3 /= total_w

                tol = 1e-2
                if abs(dist) <= tol and is_on_triangle:
                    # print("test")
                    C = 0.5 * (dist - tol) ** 2
                    nablaC_a = (dist - tol) * normal * w1
                    nablaC_b = (dist - tol) * normal * w2
                    nablaC_c = (dist - tol) * normal * w3

                    Schur = nablaC_a.dot(nablaC_a) / f.verts[1].m + nablaC_a.dot(nablaC_a) / f.verts[2].m + nablaC_a.dot(nablaC_a) / f.verts[2].m

                    kc = 1e3
                    ld = C / Schur
                    f.verts[0].g += ld * nablaC_a
                    f.verts[1].g += ld * nablaC_b
                    f.verts[2].g += ld * nablaC_c

                    f.verts[0].h += ld
                    f.verts[1].h += ld
                    f.verts[2].h += ld

        # for v in self.verts:
        #     if(v.x_k[1] < 0):
        #         # v.x_k[1] = 0.0
        #         depth = v.x_k[1] - self.bottom
        #         C = 0.5 * depth ** 2
        #         nablaC = depth * ti.math.vec3(0, 1, 0)
        #         Schur = ti.math.dot(nablaC, nablaC) / v.h
        #         ld = C / Schur
        #         v.g += ld * nablaC
        #         v.h += ld



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


