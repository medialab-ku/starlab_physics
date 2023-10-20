import taichi as ti
import meshtaichi_patcher as Patcher

@ti.data_oriented
class Solver:
    def __init__(self,
                 my_mesh,
                 static_mesh,
                 bottom,
                 k=1e4,
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
        self.edges = self.my_mesh.mesh.edges
        self.faces = self.my_mesh.mesh.faces

        self.verts_static = self.static_mesh.mesh.verts
        self.num_verts_static = len(self.static_mesh.mesh.verts)
        self.edges_static = self.static_mesh.mesh.edges
        self.num_edges_static = len(self.edges_static)
        self.faces_static = self.static_mesh.mesh.faces
        self.face_indices_static = self.static_mesh.face_indices
        self.num_faces_static = len(self.static_mesh.mesh.faces)

        self.snode = ti.root.dynamic(ti.i, 1024, chunk_size=32)
        self.candidatesVT = ti.field(ti.math.uvec2)
        self.snode.place(self.candidatesVT)

        self.S = ti.root.dynamic(ti.i, 1024, chunk_size=32)
        self.x = ti.field(ti.math.uvec2)
        self.S.place(self.x)
        # self.test()
        #
        # self.normals = ti.Vector.field(n=3, dtype=ti.f32, shape=2 * self.num_faces)
        self.normals_static = ti.Vector.field(n=3, dtype=ti.f32, shape=2 * self.num_faces_static)

        self.radius = 0.01
        self.contact_stiffness = 1e3
        self.damping_factor = 0.001
        self.grid_n = 128
        self.grid_particles_list = ti.field(ti.i32)
        self.grid_block = ti.root.dense(ti.ijk, (self.grid_n, self.grid_n, self.grid_n))
        self.partical_array = self.grid_block.dynamic(ti.l, len(self.my_mesh.mesh.verts))
        self.partical_array.place(self.grid_particles_list)
        self.grid_particles_count = ti.field(ti.i32)
        ti.root.dense(ti.ijk, (self.grid_n, self.grid_n, self.grid_n)).place(self.grid_particles_count)
        self.x_before = ti.Vector.field(n=3, dtype=ti.f32, shape=len(self.verts))


        self.dist_tol = 1e-2

        self.p1 = ti.math.vec3([0., 0., 0.])
        self.p2 = ti.math.vec3([0., 0., 0.])

        self.p = ti.Vector.field(n=3, shape=2, dtype=ti.f32)

        self.intersect = ti.Vector.field(n=3, dtype=ti.f32, shape=len(self.verts))


        print(f"verts #: {len(self.my_mesh.mesh.verts)}, elements #: {len(self.my_mesh.mesh.edges)}")
        # self.setRadius()
        print(f"radius: {self.radius}")

        print(f'{self.edges.vid[0]}')
        print(f'{self.edges_static.vid[5]}')

        # self.reset()


    def reset(self):
        self.verts.x.copy_from(self.verts.x0)
        self.verts.v.fill(0.0)

    @ti.kernel
    def test(self):
        for i in range(10):
            self.x.append(ti.math.uvec2([i, 2 * i]))

        print(self.x.length())
        self.x.deactivate()
        print(self.x.length())

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
    def compute_candidates(self):
        self.candidatesVT.deactivate()
        for v in self.verts:
            for fi in range(self.num_faces_static):
                f_aabb_min, f_aabb_max = self.faces_static.aabb_min[fi], self.faces_static.aabb_max[fi]
                if self.aabb_intersect(v.aabb_min, v.aabb_max, f_aabb_min, f_aabb_max):
                    self.candidatesVT.append(ti.math.uvec2(v.id, fi))

        # print(self.candidatesVT.length())

    @ti.kernel
    def computeVtemp(self):
        for v in self.verts:
            v.v += (v.f_ext / v.m) * self.dt

    @ti.kernel
    def globalSolveVelocity(self):
        for v in self.verts:
            v.v -= v.g / v.h

    @ti.kernel
    def computeY(self):
        for v in self.verts:
            v.y = v.x + v.v * self.dt

    @ti.kernel
    def computeNextState(self):
        for v in self.verts:
            # v.v = (v.x_k - v.x) / self.dt
            v.x += v.v * self.dt

    @ti.kernel
    def evaluateMomentumConstraint(self):
        for v in self.verts:
            v.g = v.m * (v.x_k - v.y)
            v.h = v.m

    @ti.kernel
    def evaluateSpringConstraint(self):
        for e in self.edges:
            m0, m1 = e.verts[0].m, e.verts[1].m
            msum = m0 + m1
            center = (m0 * e.verts[0].x_k + m1 * e.verts[1].x_k) / msum
            dir = (e.verts[0].x_k - e.verts[1].x_k).normalized(1e-4)
            l0 = e.l0
            p0 = center + l0 * (m0 / msum) * dir
            p1 = center - l0 * (m1 / msum) * dir

            e.verts[0].p += p0
            e.verts[1].p += p1

            e.verts[0].nc += 1
            e.verts[1].nc += 1

    @ti.kernel
    def global_solve(self):

        for v in self.verts:
            v.x_k = v.p / v.nc
            v.v = (v.x_k - v.x) / self.dt

    @ti.kernel
    def evaluateCollisionConstraint(self):

        for v in self.verts:
            for fid in range(self.num_faces_static):
                self.vertex_face_dcd(v.id, fid)

        # for e in self.edges:
        #     for eid in range(self.num_edges_static):
        self.edge_edge_dcd(0, 4)


    @ti.kernel
    def modify_velocity(self):
        for v in self.verts:
            for fid in range(self.num_faces_static):
                self.vertex_face_ccd(v.id, fid)

        # for e in self.edges:
        #     for eid in range(self.num_edges_static):
        #         self.edge_edge_ccd(e.id, eid)

        # for f in self.faces:
        #     for vid in range(self.num_verts_static):
        #         fvid0, fvid1, fvid2 = f.verts[0].id, f.verts[1].id, f.verts[2].id
        #         self.face_vertex_ccd(fvid0, fvid1, fvid2, vid)

        for v in self.verts:
            if v.nc >= 1:
                v.v = v.p / v.nc
    @ti.func
    def check_point_on_triangle(self, p, f1, f2, f3):
        e1 = f2 - f1
        e2 = f3 - f1
        d1 = p - f1
        d2 = p - f2
        d3 = p - f3
        area_triangle = e1.cross(e2).norm()
        area1 = d1.cross(d2).norm() / area_triangle
        area2 = d2.cross(d3).norm() / area_triangle
        area3 = d3.cross(d1).norm() / area_triangle

        return 0 <= area1 <= 1 and 0 <= area2 <= 1 and 0 <= area3 <= 1 and abs(area1 + area2 + area3 - 1.0) <= 1e-4

    @ti.func
    def vertex_face_ccd(self, vid: ti.int32, fid: ti.int32):
        # segment points p->x
        p = self.verts.x[vid]
        x = p + self.verts.v[vid] * self.dt

        fid0 = self.face_indices_static[fid * 3 + 0]
        fid1 = self.face_indices_static[fid * 3 + 1]
        fid2 = self.face_indices_static[fid * 3 + 2]

        f1 = self.verts_static.x[fid0]
        f2 = self.verts_static.x[fid1]
        f3 = self.verts_static.x[fid2]

        # calculate distance from point to plane
        n = (f2 - f1).cross(f3 - f1).normalized()
        dist_p = (p - f1).dot(n)
        dist_x = (x - f1).dot(n)

        if dist_p * dist_x < 0:
            # calculate intersection point
            alpha = abs(dist_p) / (abs(dist_p) + abs(dist_x))
            intersection = alpha * x + (1 - alpha) * p
            # self.intersect[vid] = intersection
            # if vid == 3 and fid == 0:
            #     print("fuck 1")
            # check if intersection point is in triangle
            e1 = f2 - f1
            e2 = f3 - f1
            d1 = intersection - f1
            d2 = intersection - f2
            d3 = intersection - f3
            area_triangle = e1.cross(e2).norm()
            area1 = d1.cross(d2).norm() / area_triangle
            area2 = d2.cross(d3).norm() / area_triangle
            area3 = d3.cross(d1).norm() / area_triangle

            is_on_triangle = (0 <= area1 <= 1 and 0 <= area2 <= 1 and 0 <= area3 <= 1 and abs(area1 + area2 + area3 - 1.0) <= 1e-4)

            # if vid == 3 and fid == 0:
            #      print(f'{area1}, {area2}, {area3}, {area1 + area2 + area3}')

            if is_on_triangle:
            # calculate new position
            #     if vid == 3 and fid == 0:
            #         print("fuck 2")
                min_dist = 1e-3
                if abs(dist_p) < min_dist:
                    alpha = 0
                    v = self.verts.v[vid]
                    vp = self.verts.v[vid]
                    if v.dot(n) < 0.0:
                        vp = v - v.dot(n) * n
                    self.verts.p[vid] += vp
                    self.verts.nc[vid] += 1
                else:
                    # find point on segment that has min_dist distance from plane
                    alpha = (abs(dist_p) - min_dist) / (abs(dist_p) + abs(dist_x))
                    intersect = alpha * x + (1 - alpha) * p
                    self.verts.p[vid] += (intersect - p) / self.dt
                    self.verts.nc[vid] += 1
                # self.verts.x_k[vid] = intersection
        else:
            point_on_triangle = x - dist_x * n
            is_on_triangle = self.check_point_on_triangle(point_on_triangle, f1, f2, f3)
            tol = 1e-4
            if abs(dist_x) <= tol and is_on_triangle:
                self.verts.p[vid] += (x - p) / self.dt
                self.verts.nc[vid] += 1

    @ti.func
    def edge_edge_dcd(self, eid: ti.int32, eid_s: ti.int32):

        vid0 = self.edges.vid[eid][0]
        vid1 = self.edges.vid[eid][1]

        vid2 = self.edges.vid[eid_s][0]
        vid3 = self.edges.vid[eid_s][1]

        e1, e2 = self.verts.x_k[vid0], self.verts.x_k[vid1]
        s1, s2 = self.verts_static.x[vid2], self.verts_static.x[vid3]


        e12 = e2 - e1
        s12 = s2 - s1
        nE = e12.normalized()
        nS = s12.normalized()

        line = e12.cross(s12)
        crossLineE = line.cross(e12)
        crossLineS = line.cross(s12)

        w1 = crossLineS.dot(s1 - e1) / crossLineS.dot(nE) * e12.norm()
        w2 = crossLineE.dot(e1 - s1) / crossLineE.dot(nS) * s12.norm()

        c1 = e1 + nE * w1
        c2 = s1 + nS * w2
        n = c2 - c1


        self.p[0] = c1
        self.p[1] = c2

        # dist = (p1 - p2).norm()
        #
        # p12 = p1 - p2
        #
        # n = (p1 - p2).normalized(1e-12)
        # v1, v2 = self.verts.v[vid0], self.verts.v[vid1]
        #
        # if dist < self.dist_tol:
        #     # print("test")
        #     p_c = p1 - (dist - self.dist_tol) * n
        #     Schur = (1.0 - t1) ** 2 / self.verts.m[vid0] + (t1 ** 2) / self.verts.m[vid1]
        #     ld = (p_c - p1) / Schur
        #
        #     p0 = a - (1.0 - t1) * (ld / self.verts.m[vid0])
        #     p2 = b - t1 * (ld / self.verts.m[vid1])
        #
        #     self.verts.p[vid0] += p0
        #     self.verts.p[vid1] += p2
        #
        #     self.verts.nc[vid0] += 1
        #     self.verts.nc[vid1] += 1

    @ti.func
    def face_vertex_ccd(self, fvid0: ti.int32, fvid1: ti.int32, fvid2: ti.int32, vid: ti.int32):

            f1 = self.verts.x[fvid0]
            f2 = self.verts.x[fvid1]
            f3 = self.verts.x[fvid2]

            point = self.verts_static.x[vid]

            e1 = f2 - f1
            e2 = f3 - f1

            a = ti.math.vec3(1, 0, 0)
            b = ti.math.vec3(0, 1, 0)
            normal = e1.normalized()
            # print(normal)
            # d = point - f1
            # dist = d.dot(normal)
            # point_on_triangle = point - dist * normal
            # d1 = point_on_triangle - f1
            # d2 = point_on_triangle - f2
            # d3 = point_on_triangle - f3
            #
            # area_triangle = e1.cross(e2).norm()
            # area1 = d1.cross(d2).norm() / area_triangle
            # area2 = d2.cross(d3).norm() / area_triangle
            # area3 = d3.cross(d1).norm() / area_triangle

            # is_on_triangle = self.check_point_on_triangle(point_on_triangle, f1, f2, f3)
            # # weights (if the point_on_triangle is close to the vertex, the weight is large)
            # w1 = 1 / (d1.norm() + 1e-8)
            # w2 = 1 / (d2.norm() + 1e-8)
            # w3 = 1 / (d3.norm() + 1e-8)
            # total_w = w1 + w2 + w3
            # w1 /= total_w
            # w2 /= total_w
            # w3 /= total_w

            tol = 1e-2
            # if abs(dist) <= tol and is_on_triangle:
            #
            #     self.verts.nc[fvid0] += 1
            #     self.verts.nc[fvid1] += 1
            #     self.verts.nc[fvid2] += 1

            #     # print("test")
            #     C = 0.5 * (dist - tol) ** 2
            #     nablaC_a = (dist - tol) * normal * w1
            #     nablaC_b = (dist - tol) * normal * w2
            #     nablaC_c = (dist - tol) * normal * w3
            #
            #     Schur = nablaC_a.dot(nablaC_a) / f.verts[1].m + nablaC_a.dot(nablaC_a) / f.verts[2].m + nablaC_a.dot(nablaC_a) / f.verts[2].m
            #
            #     kc = 1e3
            #     ld = C / Schur
            #     f.verts[0].g += ld * nablaC_a
            #     f.verts[1].g += ld * nablaC_b
            #     f.verts[2].g += ld * nablaC_c
            #
            #     f.verts[0].h += ld
            #     f.verts[1].h += ld
            #     f.verts[2].h += ld

    @ti.func
    def vertex_face_dcd(self, vid: ti.int32, fid: ti.int32):
        fid0 = self.face_indices_static[fid * 3 + 0]
        fid1 = self.face_indices_static[fid * 3 + 1]
        fid2 = self.face_indices_static[fid * 3 + 2]

        f1 = self.verts_static.x[fid0]
        f2 = self.verts_static.x[fid1]
        f3 = self.verts_static.x[fid2]

        point = self.verts.x_k[vid]

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

        is_on_triangle = 0 <= area1 <= 1 and 0 <= area2 <= 1 and 0 <= area3 <= 1 and abs(area1 + area2 + area3 - 1) <= 1e-3
        if abs(dist) < self.dist_tol and is_on_triangle:
            self.verts.p[vid] += (point + (self.dist_tol - abs(dist)) * normal)
            self.verts.nc[vid] += 1
    @ti.kernel
    def filterStepSize(self) -> ti.f32:

        alpha = 1.0
        for v in self.verts:
            alpha = ti.min(alpha, v.alpha)

        return alpha

    @ti.kernel
    def NewtonCG(self):
        for v in self.verts:
            v.dx = -v.g / v.h

        self.verts.alpha.fill(1.0)
        alpha = 1.0
        for v in self.verts:
            v.x_k += alpha * v.dx

    @ti.kernel
    def computeAABB(self):

        padding_size = 1e-2
        padding = ti.math.vec3([padding_size, padding_size, padding_size])

        for v in self.verts:
            for i in range(3):
                v.aabb_min[i] = ti.min(v.x[i], v.y[i])
                v.aabb_max[i] = ti.max(v.x[i], v.y[i])

            v.aabb_max += padding
            v.aabb_min -= padding


        for f in self.faces_static:

            x0 = f.verts[0].x
            x1 = f.verts[1].x
            x2 = f.verts[2].x

            for i in range(3):
                f.aabb_min[i] = ti.min(x0[i], x1[i], x2[i])
                f.aabb_max[i] = ti.max(x0[i], x1[i], x2[i])

            f.aabb_min -= padding
            f.aabb_max += padding

    def update(self):

        self.verts.f_ext.fill([0.0, self.gravity, 0.0])
        self.computeVtemp()
        self.computeY()
        self.verts.x_k.copy_from(self.verts.y)

        for i in range(self.max_iter):
            self.verts.p.fill(0.)
            self.verts.nc.fill(0)
            self.evaluateSpringConstraint()
            self.evaluateCollisionConstraint()
            self.global_solve()

        for i in range(self.max_iter):
            self.verts.p.fill(0.)
            self.verts.nc.fill(0)
            self.modify_velocity()

        self.computeNextState()




        # self.x_before.copy_from(self.verts.x)
        # self.modify_velocity()
        #
        # self.verts.g.fill(0.0)
        # self.verts.h.copy_from(self.verts.m)
        # for i in range(self.max_iter):
        #     self.computeCollisionAwareVelocity()
        #     # self.globalSolveVelocity()


        # self.computeY()
        # self.verts.x_k.copy_from(self.verts.y)


        # self.computeAABB()
        # self.compute_candidates()

        # self.verts.x_k.copy_from(self.verts.y)
        # self.verts.h.copy_from(self.verts.m)
        #     # self.evaluateMomentumConstraint()
        #     # self.evaluateSpringConstraint()
        #     self.evaluateCollisionConstraint()
        #     # self.filterStepSize()
        #     # self.NewtonCG()
        #


