import taichi as ti
import meshtaichi_patcher as Patcher
import ipc_utils as cu

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
        self.gravity = -9.81
        self.bottom = bottom
        self.id3 = ti.math.mat3([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 1]])

        self.verts = self.my_mesh.mesh.verts
        self.edges = self.my_mesh.mesh.edges
        self.faces = self.my_mesh.mesh.faces
        self.face_indices = self.my_mesh.face_indices

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
        self.mmcvid = ti.field(ti.math.ivec4)
        self.S.place(self.mmcvid)
        self.dHat = 1e-3
        # self.test()
        #
        # self.normals = ti.Vector.field(n=3, dtype = ti.f32, shape = 2 * self.num_faces)
        self.normals_static = ti.Vector.field(n=3, dtype=ti.f32, shape=2 * self.num_faces_static)

        self.radius = 0.01
        self.contact_stiffness = 1e3
        self.damping_factor = 0.002
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
        # print(f"radius: {self.radius}")
        #
        # print(f'{self.edges.vid}')
        # print(f'{self.edges_static.vid[4]}')




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
            v.v = (1 - self.damping_factor) * (v.x_k - v.x) / self.dt
            v.x = v.x_k

    @ti.kernel
    def evaluateMomentumConstraint(self):
        for v in self.verts:
            v.g = v.m * (v.x_k - v.y)
            v.h = v.m

    @ti.kernel
    def evaluateSpringConstraint(self):
        for e in self.edges:

            xij = e.verts[0].x_k - e.verts[1].x_k
            coeff = self.dtSq * self.k
            grad = coeff * (xij - e.l0 * xij.normalized(1e-6))

            dir = (e.verts[0].x_k - e.verts[1].x_k).normalized(1e-4)

            # m0, m1 = e.verts[0].m, e.verts[1].m
            # msum = m0 + m1
            # center = (m0 * e.verts[0].x_k + m1 * e.verts[1].x_k) / msum
            # dir = (e.verts[0].x_k - e.verts[1].x_k).normalized(1e-4)
            # l0 = e.l0
            # p0 = center + l0 * (m0 / msum) * dir
            # p1 = center - l0 * (m1 / msum) * dir

            e.verts[0].g += grad
            e.verts[1].g -= grad
            e.verts[0].h += coeff
            e.verts[1].h += coeff

    @ti.kernel
    def global_solve(self):

        for v in self.verts:
            v.x_k -= (v.g / v.h)

    @ti.kernel
    def evaluateCollisionConstraint(self):
        for i in self.mmcvid:
            mi = self.mmcvid[i]
            if mi[0] >= 0:
                print("EE")
                # cu.g_EE()
            else:
                if mi[1] >= 0:
                    vi = -mi[0]-1
                    if mi[2] < 0:
                        cu.g_PP(vi, mi[1])
                        print("PP")
                    elif mi[3] < 0:
                        cu.g_PE()
                        print("PE")
                    else:
                        # cu.g_PT()
                        print("PT")
                else:
                    if mi[2] < 0:
                        # cu.g_PT()
                        print("PT")
                    else:
                        # cu.g_PE()
                        print("PE")

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
    @ti.func
    def computeConstraintSet_PT(self, pid: ti.int32, tid: ti.int32):

        v0 = pid
        v1 = self.face_indices_static[3 * tid + 0]
        v2 = self.face_indices_static[3 * tid + 1]
        v3 = self.face_indices_static[3 * tid + 2]


        x0 = self.verts.x_k[v0]
        x1 = self.verts_static.x[v1]   #r
        x2 = self.verts_static.x[v2]   #g
        x3 = self.verts_static.x[v3]   #c

        dtype = cu.d_type_PT(x0, x1, x2, x3)
        if dtype == 0:           #r
            d = cu.d_PP(x0, x1)
            if d < self.dHat:
                g0, g1 = cu.g_PP(x0, x1)
                sch = g0.dot(g0) / self.verts.h[v0]
                ld = (d - self.dHat) / sch

                self.verts.g[v0] += ld * g0
                self.verts.h[v0] += ld


        elif dtype == 1:
            d = cu.d_PP(x0, x2)  #g
            if d < self.dHat:
                g0, g2 = cu.g_PP(x0, x2)
                sch = g0.dot(g0) / self.verts.h[v0]
                ld = (d - self.dHat) / sch

                self.verts.g[v0] += ld * g0
                self.verts.h[v0] += ld

        elif dtype == 2:
            d = cu.d_PP(x0, x3) #c
            if d < self.dHat:
                g0, g3 = cu.g_PP(x0, x3)
                sch = g0.dot(g0) / self.verts.h[v0]
                ld = (d - self.dHat) / sch

                self.verts.g[v0] += ld * g0
                self.verts.h[v0] += ld
              # s

        elif dtype == 3:
            d = cu.d_PE(x0, x1, x2) # r-g
            if d < self.dHat:
                g0, g1, g2 = cu.g_PE(x0, x1, x2)
                sch = g0.dot(g0) / self.verts.h[v0]
                ld = (d - self.dHat) / sch

                self.verts.g[v0] += ld * g0
                self.verts.h[v0] += ld

        elif dtype == 4:
            d = cu.d_PE(x0, x2, x3) #g-c
            if d < self.dHat:
                g0, g2, g3 = cu.g_PE(x0, x1, x2)
                sch = g0.dot(g0) / self.verts.h[v0]
                ld = (d - self.dHat) / sch

                self.verts.g[v0] += ld * g0
                self.verts.h[v0] += ld

        elif dtype == 5:
            d = cu.d_PE(x0, x3, x1) #c-r
            if d < self.dHat:
                g0, g3, g1 = cu.g_PE(x0, x3, x1)
                sch = g0.dot(g0) / self.verts.h[v0]
                ld = (d - self.dHat) / sch

                self.verts.g[v0] += ld * g0
                self.verts.h[v0] += ld

        elif dtype == 6:            # inside triangle
            d = cu.d_PT(x0, x1, x2, x3)
            if d < self.dHat:
                g0, g1, g2, g3 = cu.g_PT(x0, x1, x2, x3)
                sch = g0.dot(g0) / self.verts.h[v0]
                ld = (d - self.dHat) / sch
                self.verts.g[v0] += ld * g0
                self.verts.h[v0] += ld

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
                sch = g1.dot(g1) / self.verts.h[v1]
                ld = (d - self.dHat) / sch

                self.verts.g[v1] += ld * g1
                self.verts.h[v1] += ld

                # self.mmcvid.append(ti.math.ivec4([-v1-1, v0, -1, -1]))

        elif dtype == 1:
            d = cu.d_PP(x0, x2)  #g
            if d < self.dHat:
                g0, g2 = cu.g_PP(x0, x2)
                sch = g2.dot(g2) / self.verts.h[v2]
                ld = (d - self.dHat) / sch

                self.verts.g[v2] += ld * g2
                self.verts.h[v2] += ld

        elif dtype == 2:
            d = cu.d_PP(x0, x3) #c
            if d < self.dHat:
                g0, g3 = cu.g_PP(x0, x3)
                sch = g3.dot(g3) / self.verts.h[v3]
                ld = (d - self.dHat) / sch

                self.verts.g[v3] += ld * g3
                self.verts.h[v3] += ld

        # self.mmcvid.append(ti.math.ivec4([-v3-1, v0, -1, -1]))

        elif dtype == 3:
            d = cu.d_PE(x0, x1, x2) # r-g
            if d < self.dHat:
                g0, g1, g2 = cu.g_PE(x0, x1, x2)
                sch = g1.dot(g1) / self.verts.h[v1] + g2.dot(g2) / self.verts.h[v2]
                ld = (d - self.dHat) / sch

                self.verts.g[v1] += ld * g1
                self.verts.g[v2] += ld * g2
                self.verts.h[v1] += ld
                self.verts.h[v2] += ld
                # self.mmcvid.append(ti.math.ivec4([-v1-1, -v2-1, v0, -1]))

        elif dtype == 4:
            d = cu.d_PE(x0, x2, x3) #g-c
            if d < self.dHat:
                g0, g2, g3 = cu.g_PE(x0, x2, x3)
                sch = g2.dot(g2) / self.verts.h[v2] + g3.dot(g3) / self.verts.h[v3]
                ld = (d - self.dHat) / sch

                self.verts.g[v2] += ld * g2
                self.verts.g[v3] += ld * g3

                self.verts.h[v2] += ld
                self.verts.h[v3] += ld
                # self.mmcvid.append(ti.math.ivec4([-v2-1, -v3-1, v0, -1]))

        elif dtype == 5:
            d = cu.d_PE(x0, x3, x1) #c-r
            if d < self.dHat:
                g0, g3, g1 = cu.g_PE(x0, x3, x1)
                sch = g1.dot(g1) / self.verts.h[v2] + g3.dot(g3) / self.verts.h[v3]
                ld = (d - self.dHat) / sch

                self.verts.g[v1] += ld * g1
                self.verts.g[v3] += ld * g3

                self.verts.h[v1] += ld
                self.verts.h[v3] += ld
                # self.mmcvid.append(ti.math.ivec4([-v3-1, -v1-1, v0, -1]))

        elif dtype == 6:            # inside triangle
            d = cu.d_PT(x0, x1, x2, x3)
            if d < self.dHat:
                g0, g1, g2, g3 = cu.g_PT(x0, x1, x2, x3)
                sch = g1.dot(g1) / self.verts.h[v1] + g2.dot(g2) / self.verts.h[v2] + g3.dot(g3) / self.verts.h[v3]
                ld = (d - self.dHat) / sch

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
        if (x01.cross(x32).norm() < 1e-3):
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

        # self.mmcvid.deactivate()

        # # point - triangle
        for v in self.verts:
            for fid in range(self.num_faces_static):
                self.computeConstraintSet_PT(v.id, fid)
        #
        # # print(self.mmcvid.length())
        # # triangle - point
        for f in self.faces:
            for vid in range(self.num_verts_static):
                self.computeConstraintSet_TP(f.id, vid)

        # for e in self.edges:
        #     for eid in range(self.num_edges_static):
        #         self.computeConstraintSet_EE(e.id, eid)





    def update(self):

        self.verts.f_ext.fill([0.0, self.gravity, 0.0])
        self.computeVtemp()
        self.computeY()
        self.verts.x_k.copy_from(self.verts.y)

        for i in range(self.max_iter):
            self.verts.g.fill(0.)
            self.verts.h.copy_from(self.verts.m)
            self.evaluateSpringConstraint()
            self.computeConstraintSet()
            # self.evaluateCollisionConstraint()
            self.global_solve()

        self.computeNextState()
        # for i in range(self.max_iter):
        #     self.verts.p.fill(0.)
        #     self.verts.nc.fill(0)
        #     self.modify_velocity()







