import taichi as ti
import meshtaichi_patcher as Patcher

@ti.dataclass
class contact_particle:
    vid: ti.u8
    w  : ti.math.vec3

@ti.dataclass
class contact_triangle:
    x: ti.math.vec3
    vids: ti.math.ivec3
    w: ti.math.vec3

@ti.dataclass
class contact_edge:
    x: ti.math.vec3
    w: ti.math.vec2

@ti.dataclass
class edge:
    vid: ti.math.uvec2
    l0: ti.float32
    hij: ti.math.mat3

@ti.dataclass
class face:
    vid: ti.math.uvec3

@ti.dataclass
class node:
    x   : ti.math.vec3
    v   : ti.math.vec3
    f   : ti.math.vec3
    y   : ti.math.vec3
    x_k : ti.math.vec3
    m   : ti.float32
    grad: ti.math.vec3
    hii : ti.math.mat3

@ti.dataclass
class static_node:
    x: ti.math.vec3

@ti.data_oriented
class Solver:
    def __init__(self,
                 my_mesh,
                 static_mesh,
                 bottom,
                 k=1e2,
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
        self.idenity3 = ti.math.mat3([[1, 0, 0],
                                      [0, 1, 0],
                                      [0, 0, 1]])

        self.num_contacts = ti.field(ti.int32, shape=(1))
        self.radius = 0.005
        self.contact_stiffness = 1e3
        self.edges = edge.field(shape=len(self.my_mesh.mesh.edges))
        self.nodes = node.field(shape=(len(self.my_mesh.mesh.verts)))
        self.num_nodes = len(self.my_mesh.mesh.verts)
        self.num_static_verts = len(self.static_mesh.mesh.verts)
        self.num_static_edges = len(self.static_mesh.mesh.edges)
        self.num_static_faces = len(self.static_mesh.mesh.faces)

        self.static_faces = face.field(shape=self.num_static_faces)


        self.num_verts = len(self.my_mesh.mesh.verts)
        self.num_edges = len(self.my_mesh.mesh.edges)
        self.num_faces = len(self.my_mesh.mesh.faces)

        self.faces = face.field(shape=self.num_faces)

        self.static_edges = edge.field(shape=(self.num_static_edges))
        self.contact_triangles = contact_triangle.field(shape=len(self.my_mesh.mesh.edges))
        self.num_static_nodes = self.num_static_verts + self.num_static_edges + self.num_static_faces
        self.static_nodes = static_node.field(shape=(self.num_static_nodes))

        self.damping_factor = 0.001
        self.init_nodes()
        self.init_edges()
        self.init_faces()
        self.grid_n = 128
        self.grid_particles_list = ti.field(ti.i32)
        self.grid_block = ti.root.dense(ti.ijk, (self.grid_n, self.grid_n, self.grid_n))
        self.partical_array = self.grid_block.dynamic(ti.l, len(self.my_mesh.mesh.verts))
        self.partical_array.place(self.grid_particles_list)
        self.grid_particles_count = ti.field(ti.i32)
        ti.root.dense(ti.ijk, (self.grid_n, self.grid_n, self.grid_n)).place(self.grid_particles_count)

        # self.construct_collision_grid()

        print(f"verts #: {len(self.my_mesh.mesh.verts)}, elements #: {len(self.my_mesh.mesh.edges)}")
        # self.setRadius()
        print(f"radius: {self.radius}")

        #for CG
        self.x = ti.Vector.field(3, dtype=ti.f32, shape=self.num_verts)
        self.b = ti.Vector.field(3, dtype=ti.f32, shape=self.num_verts)
        self.r = ti.Vector.field(3, dtype=ti.f32, shape=self.num_verts)
        self.p = ti.Vector.field(3, dtype=ti.f32, shape=self.num_verts)
        self.Ap = ti.Vector.field(3, dtype=ti.f32, shape=self.num_verts)


        # self.initContactParticleData()



    @ti.kernel
    def init_edges(self):
        for e in self.my_mesh.mesh.edges:
            self.edges[e.id].vid[0] = e.verts[0].id
            self.edges[e.id].vid[1] = e.verts[1].id
            self.edges[e.id].l0 = e.l0

        for e in self.static_mesh.mesh.edges:
            self.static_edges[e.id].vid[0] = e.verts[0].id
            self.static_edges[e.id].vid[1] = e.verts[1].id

    @ti.kernel
    def init_faces(self):
        for f in self.my_mesh.mesh.faces:
            self.faces[f.id].vid[0] = f.verts[0].id
            self.faces[f.id].vid[1] = f.verts[1].id
            self.faces[f.id].vid[2] = f.verts[2].id

        for f in self.static_mesh.mesh.faces:
            self.static_faces[f.id].vid[0] = f.verts[0].id
            self.static_faces[f.id].vid[1] = f.verts[1].id
            self.static_faces[f.id].vid[2] = f.verts[2].id

    @ti.kernel
    def init_nodes(self):
        for v in self.my_mesh.mesh.verts:
            self.nodes[v.id].x = v.x
            self.nodes[v.id].m = v.m
            self.nodes[v.id].v = v.v

        for v in self.static_mesh.mesh.verts:
            self.static_nodes[v.id].x = v.x

        for e in self.static_mesh.mesh.edges:
            self.static_nodes[e.id + self.num_static_verts].x = 0.5 * (e.verts[0].x + e.verts[1].x)

        for f in self.static_mesh.mesh.faces:
            self.static_nodes[f.id + self.num_static_verts + self.num_static_edges].x = (f.verts[0].x + f.verts[1].x + f.verts[2].x) / 3

    @ti.kernel
    def construct_collision_grid(self):
        for sn in self.static_nodes:
            grid_idx = ti.floor(self.static_nodes[sn].x * self.grid_n, int)
            # print(grid_idx, grid_particles_count[grid_idx])
            ti.append(self.grid_particles_list.parent(), grid_idx, sn)
            ti.atomic_add(self.grid_particles_count[grid_idx], 1)

    @ti.kernel
    def init_contact_triangles(self):

        for v in self.my_mesh.mesh.verts:
            self.contact_triangles[v.id].x = v.x
            self.contact_triangles[v.id].vids[0] = v.id
            self.contact_triangles[v.id].vids[1] = -1
            self.contact_triangles[v.id].vids[2] = -1

        for e in self.my_mesh.mesh.edges:
            self.contact_triangles[e.id + self.num_verts].x = 0.5 * (e.verts[0].x + e.verts[1].x)
            self.contact_triangles[e.id + self.num_verts].vids[0] = e.verts[0].id
            self.contact_triangles[e.id + self.num_verts].vids[1] = e.verts[1].id
            self.contact_triangles[e.id + self.num_verts].vids[2] = -1

        for f in self.my_mesh.mesh.faces:
            self.contact_triangles[f.id + self.num_static_verts + self.num_static_faces].x = 0.333 * (f.verts[0].x + f.verts[1].x + f.verts[2].x)
            self.contact_triangles[f.id + self.num_static_verts + self.num_static_faces].vids[0] = f.verts[0].id
            self.contact_triangles[f.id + self.num_static_verts + self.num_static_faces].vids[1] = f.verts[1].id
            self.contact_triangles[f.id + self.num_static_verts + self.num_static_faces].vids[2] = f.verts[2].id

    @ti.func
    def resolve_contact(self, i, j):
        # test = i + j
        rel_pos = self.nodes[j].x_k - self.nodes[i].x_k
        dist = rel_pos.norm()
        delta = dist - 2 * self.radius  # delta = d - 2 * r
        coeff = self.contact_stiffness * self.dtSq
        if delta < 0:  # in contact
            normal = rel_pos / dist
            f1 = normal * delta * coeff
            self.nodes[i].grad -= f1
            self.nodes[j].grad += f1
            self.nodes[i].hii += self.idenity3 * coeff
            self.nodes[j].hii += self.idenity3 * coeff

    @ti.func
    def resolve_contact_static(self, i, si):
        rel_pos = self.static_nodes[si].x - self.nodes[i].x_k
        dist = rel_pos.norm()
        delta = dist - 2 * self.radius  # delta = d - 2 * r
        coeff = self.contact_stiffness * self.dtSq
        if delta < 0:  # in contact
            normal = rel_pos / dist
            f1 = normal * delta * coeff
            self.nodes[i].grad -= f1
            self.nodes[i].hii += self.idenity3 * coeff

    @ti.func
    def resolve_contact(self, i, j):
        rel_pos = self.nodes[j].x_k - self.nodes[i].x_k
        dist = rel_pos.norm()
        delta = dist - 2 * self.radius  # delta = d - 2 * r
        coeff = self.contact_stiffness * self.dtSq
        if delta < 0:  # in contact
            normal = rel_pos / dist
            f1 = normal * delta * coeff
            self.nodes[i].grad -= f1
            self.nodes[j].grad += f1
            self.nodes[i].hii += self.idenity3 * coeff
            self.nodes[j].hii += self.idenity3 * coeff


    @ti.func
    def resolve_vertex_triangle_static(self, vi, sfi):
        v0, v1, v2 = self.static_faces[sfi].vid[0], self.static_faces[sfi].vid[1], self.static_faces[sfi].vid[2]
        f1, f2, f3 = self.static_nodes[v0].x, self.static_nodes[v1].x, self.static_nodes[v2].x

        point = self.nodes[vi].x_k
        e1 = f3 - f1
        e2 = f3 - f1
        normal = e1.cross(e2).normalized(1e-12)
        d = point - f1
        dist = d.dot(normal)
        point_on_triangle = point - dist * normal
        d1 = point_on_triangle - f1
        d2 = point_on_triangle - f2
        d3 = point_on_triangle - f3

        area_triangle = e1.cross(e2).norm()
        area1 = d1.cross(d2).norm() / area_triangle
        area2 = d2.cross(d3).norm() / area_triangle
        area3 = d3.cross(d1).norm() / area_triangle

        is_on_triangle = 0 <= area1 <= 1 and 0 <= area2 <= 1 and 0 <= area3 <= 1 and area1 + area2 + area3 == 1

        if dist < 0 and abs(dist) < 3 * self.radius and is_on_triangle:
            self.nodes[vi].grad -= self.dtSq * (dist) * normal * self.contact_stiffness
            self.nodes[vi].hii += self.dtSq * self.contact_stiffness * self.idenity3

    @ti.func
    def resolve_vertex_triangle(self, vi, fi):
        v0, v1, v2 = self.faces[fi].vid[0], self.faces[fi].vid[1], self.faces[fi].vid[2]
        f1, f2, f3 = self.nodes[v0].x_k, self.nodes[v1].x_k, self.nodes[v2].x_k

        point = self.nodes[vi].x_k
        e1 = f3 - f1
        e2 = f3 - f1
        normal = e1.cross(e2).normalized(1e-12)
        d = point - f1
        dist = d.dot(normal)
        point_on_triangle = point - dist * normal
        d1 = point_on_triangle - f1
        d2 = point_on_triangle - f2
        d3 = point_on_triangle - f3

        area_triangle = e1.cross(e2).norm()
        area1 = d1.cross(d2).norm() / area_triangle
        area2 = d2.cross(d3).norm() / area_triangle
        area3 = d3.cross(d1).norm() / area_triangle

        w0 = area1 / area_triangle
        w1 = area1 / area_triangle
        w2 = area1 / area_triangle

        is_on_triangle = 0 <= area1 <= 1 and 0 <= area2 <= 1 and 0 <= area3 <= 1 and area1 + area2 + area3 == 1

        if dist < 0 and abs(dist) < 3 * self.radius and is_on_triangle:
            self.nodes[vi].grad -= self.dtSq * (dist) * normal * self.contact_stiffness
            self.nodes[vi].hii += self.dtSq * self.contact_stiffness * self.idenity3

            self.nodes[v0].grad += self.dtSq * (dist) * normal * self.contact_stiffness * w0
            self.nodes[v1].grad += self.dtSq * (dist) * normal * self.contact_stiffness * w1
            self.nodes[v2].grad += self.dtSq * (dist) * normal * self.contact_stiffness * w2

            self.nodes[v0].hii += self.dtSq * self.contact_stiffness * self.idenity3 * w0
            self.nodes[v1].hii += self.dtSq * self.contact_stiffness * self.idenity3 * w1
            self.nodes[v2].hii += self.dtSq * self.contact_stiffness * self.idenity3 * w2

    @ti.func
    def resolve_triangle_vertex_static(self, fi, svi):
        v0, v1, v2 = self.faces[fi].vid[0], self.faces[fi].vid[1], self.faces[fi].vid[2]
        f1, f2, f3 = self.nodes[v0].x, self.nodes[v1].x, self.nodes[v2].x

        point = self.static_nodes[svi].x
        e1 = f3 - f1
        e2 = f2 - f1
        normal = e1.cross(e2).normalized(1e-12)
        d = point - f1
        dist = d.dot(normal)
        point_on_triangle = point - dist * normal
        d1 = point_on_triangle - f1
        d2 = point_on_triangle - f2
        d3 = point_on_triangle - f3

        area_triangle = e1.cross(e2).norm()
        area1 = d1.cross(d2).norm() / area_triangle
        area2 = d2.cross(d3).norm() / area_triangle
        area3 = d3.cross(d1).norm() / area_triangle

        w0 = area1 / area_triangle
        w1 = area1 / area_triangle
        w2 = area1 / area_triangle

        is_on_triangle = 0 <= area1 <= 1 and 0 <= area2 <= 1 and 0 <= area3 <= 1 and area1 + area2 + area3 == 1

        if dist < 0 and abs(dist) < 3 * self.radius and is_on_triangle:

            self.nodes[v0].grad += self.dtSq * (dist) * normal * self.contact_stiffness * w0
            self.nodes[v1].grad += self.dtSq * (dist) * normal * self.contact_stiffness * w1
            self.nodes[v2].grad += self.dtSq * (dist) * normal * self.contact_stiffness * w2

            self.nodes[v0].hii += self.dtSq * self.contact_stiffness * self.idenity3 * w0
            self.nodes[v1].hii += self.dtSq * self.contact_stiffness * self.idenity3 * w1
            self.nodes[v2].hii += self.dtSq * self.contact_stiffness * self.idenity3 * w2


    @ti.func
    def resolve_edge_edge_static(self, ei, sei):
        v0, v1 = self.edges[ei].vid[0], self.edges[ei].vid[1]
        sv0, sv1 = self.static_edges[sei].vid[0], self.static_edges[sei].vid[1]
        A = self.nodes[v0].x_k
        B = self.nodes[v1].x_k
        C = self.static_nodes[sv0].x
        D = self.static_nodes[sv1].x

        AB = B - A
        CD = D - C
        AC = C - A

        mat = ti.math.mat2([[-CD.dot(AB), AB.dot(AB)],
                            [-CD.dot(CD), CD.dot(AB)]])

        b = ti.math.vec2([AB.dot(AC), CD.dot(AC)])

        t = mat.inverse() @ b
        t1 = t[0]
        t2 = t[1]
        w1 = t1
        w2 = (1 - t1)

        p1 = A + t1 * AB
        p2 = C + t2 * CD

        dist = (p1 - p2).norm()
        n = (p2 - p1).normalized(1e-12)
        test = 0.01
        if dist < test:
            self.nodes[v0].grad += self.dtSq * (test - dist) * n.normalized() * self.contact_stiffness
            self.nodes[v1].grad += self.dtSq * (test - dist) * n.normalized() * self.contact_stiffness
            self.nodes[v0].hii += self.dtSq * self.contact_stiffness
            self.nodes[v1].hii += self.dtSq * self.contact_stiffness

    @ti.kernel
    def add(self, ans: ti.template(), a: ti.template(), k: ti.f32, b: ti.template()):
        for i in ans:
            ans[i] = a[i] + k * b[i]

    @ti.kernel
    def computeNextState(self):
        for n in self.nodes:
            self.nodes[n].v = (self.nodes[n].x_k - self.nodes[n].x) / self.dt
            self.nodes[n].v *= (1 - self.damping_factor)
            self.nodes[n].x = self.nodes[n].x_k

        # for v in self.my_mesh.mesh.verts:
        #     v.v = (v.x_k - v.x) / self.dt
        #     v.x = v.x_k

    @ti.kernel
    def computeGradientAndElementWiseHessian(self):

        # momentum gradient M * (x - y) and hessian M
        for n in self.nodes:
            self.nodes[n].grad = self.nodes[n].m * (self.nodes[n].x_k - self.nodes[n].y) - self.nodes[n].f * self.dtSq
            self.nodes[n].hii = self.nodes[n].m * self.idenity3

        # elastic energy gradient \nabla E (x)
        for e in self.edges:
            v0, v1 = self.edges[e].vid[0], self.edges[e].vid[1]
            l = (self.nodes[v0].x_k - self.nodes[v1].x_k).norm()
            normal = (self.nodes[v0].x_k - self.nodes[v1].x_k).normalized(1e-12)
            coeff = self.dtSq * self.k
            grad_e = coeff * (l - self.edges[e].l0) * normal
            self.nodes[v0].grad += grad_e
            self.nodes[v1].grad -= grad_e
            self.nodes[v0].hii += coeff * self.idenity3
            self.nodes[v1].hii += coeff * self.idenity3
            self.edges[e].hij = -coeff * self.idenity3

        # handling bottom contact
        for n in self.nodes:
            if (self.nodes[n].x_k[1] < 0):
                depth = self.nodes[n].x_k[1] - self.bottom
                up = ti.math.vec3(0, 1, 0)
                self.nodes[n].grad += self.dtSq * self.contact_stiffness * depth * up
                self.nodes[n].hii  += self.dtSq * self.contact_stiffness * self.idenity3

            if (self.nodes[n].x_k[1] > 1):
                depth = 1 - self.nodes[n].x_k[1]
                up = ti.math.vec3(0, -1, 0)
                self.nodes[n].grad += self.dtSq * self.contact_stiffness * depth * up
                self.nodes[n].hii  += self.dtSq * self.contact_stiffness * self.idenity3

            # if (self.nodes[n].x_k[0] < 0):
            #     depth = self.nodes[n].x_k[0] - self.bottom
            #     up = ti.math.vec3(1, 0, 0)
            #     self.nodes[n].grad += self.dtSq * self.contact_stiffness * depth * up
            #     self.nodes[n].hii += self.dtSq * self.contact_stiffness * self.idenity3

            if (self.nodes[n].x_k[0] > 1):
                depth = 1 - self.nodes[n].x_k[0]
                up = ti.math.vec3(-1, 0, 0)
                self.nodes[n].grad += self.dtSq * self.contact_stiffness * depth * up
                self.nodes[n].hii += self.dtSq * self.contact_stiffness * self.idenity3

            if (self.nodes[n].x_k[2] < 0):
                depth = self.nodes[n].x_k[2] - self.bottom
                up = ti.math.vec3(0, 0, 1)
                self.nodes[n].grad += self.dtSq * self.contact_stiffness * depth * up
                self.nodes[n].hii += self.dtSq * self.contact_stiffness * self.idenity3

            if (self.nodes[n].x_k[2] > 1):
                depth = 1 - self.nodes[n].x_k[2]
                up = ti.math.vec3(0, 0, -1)
                self.nodes[n].grad += self.dtSq * self.contact_stiffness * depth * up
                self.nodes[n].hii += self.dtSq * self.contact_stiffness * self.idenity3


        # for vi in range(self.num_verts):
        #     for fi in range(self.num_faces):
        #         self.re

        for ni in range(self.num_verts):
            for fi in range(self.num_static_faces):
                self.resolve_vertex_triangle_static(ni, fi)

        for fi in range(self.num_faces):
            for svi in range(self.num_static_verts):
                self.resolve_triangle_vertex_static(fi, svi)

        # for ni in range(self.num_verts):
        #     for sni in range(self.num_static_nodes):
        #         self.resolve_contact_static(ni, sni)

        # for ni in range(self.num_verts):
        #     for nj in range(ni+1, self.num_verts):
        #         self.resolve_contact_static(ni, nj)
        #
        # for n in self.nodes:
        #     grid_idx = ti.floor(self.nodes[n].x_k * self.grid_n, int)
        #     x_begin = max(grid_idx[0] - 1, 0)
        #     x_end = min(grid_idx[0] + 2, self.grid_n)
        #
        #     y_begin = max(grid_idx[1] - 1, 0)
        #     y_end = min(grid_idx[1] + 2, self.grid_n)
        #
        #     z_begin = max(grid_idx[2] - 1, 0)
        #
        #     # only need one side
        #     z_end = min(grid_idx[2] + 1, self.grid_n)
        #
        #     # todo still serialize
        #     for neigh_i, neigh_j, neigh_k in ti.ndrange((x_begin, x_end), (y_begin, y_end), (z_begin, z_end)):
        #
        #         # on split plane
        #         if neigh_k == grid_idx[2] and (neigh_i + neigh_j) > (grid_idx[0] + grid_idx[1]) and neigh_i <= grid_idx[0]:
        #             continue
        #         # same grid
        #         iscur = neigh_i == grid_idx[0] and neigh_j == grid_idx[1] and neigh_k == grid_idx[2]
        #         for l in range(self.grid_particles_count[neigh_i, neigh_j, neigh_k]):
        #             j = self.grid_particles_list[neigh_i, neigh_j, neigh_k, l]
        #
        #             if iscur and n >= j:
        #                 continue
        #             self.resolve_contact_static(n, j)


        # for n in self.nodes:
        #     self.nodes[n].x_k -= self.nodes[n].hii.inverse() @ self.nodes[n].grad

    @ti.kernel
    def dot(self, a: ti.template(), b: ti.template()) -> ti.f32:
        ans = 0.0
        # ti.loop_config(block_dim=32)
        for i in a:
            ans += a[i].dot(b[i])
        return ans

    @ti.kernel
    def matrix_free_Ax(self, ans: ti.template(), x: ti.template()):

        for vi in range(self.num_verts):
            ans[vi] = self.nodes[vi].hii @ x[vi]

        for ei in range(self.num_edges):
            v0, v1 = self.edges[ei].vid[0], self.edges[ei].vid[1]
            ans[v0] += self.edges[ei].hij @ x[v0]
            ans[v1] += self.edges[ei].hij @ x[v1]


    @ti.kernel
    def update_xk(self, dx: ti.template()):

        for vi in range(self.num_verts):
            self.nodes[vi].x_k += dx[vi]

    def solveNewtonIterationWithCG(self):

        max_cg_iter = 10

        self.b.copy_from(self.nodes.grad)
        self.x.copy_from(self.nodes.grad)
        self.matrix_free_Ax(self.Ap, self.x)
        self.add(self.r, self.x, -1, self.Ap)

        # self.matrix_free_Ax(Ap, x)
        # self.add(r, Ap, -1, b)
        # p.copy_from(r)

        r_2 = self.dot(self.r, self.r)

        threshold = 1e-8
        if(r_2 < threshold):
            return
        #
        for i in range(max_cg_iter):
            self.matrix_free_Ax(self.Ap, self.p)
            alpha = r_2 / self.dot(self.p, self.Ap)
            self.add(self.x, alpha, self.p)
            self.add(self.r, -alpha, self.Ap)

            if (r_2 < threshold):
                print(f'{i}')
                return

            r_2_next = self.dot(self.r, self.r)
            beta = r_2_next / r_2

            self.add(self.r, beta, self.p)
            r_2 = r_2_next

        self.update_xk(self.x)

    # @ti.kernel
    # def computeExternalForce(self):
    #     for v in self.my_mesh.mesh.verts:

    @ti.kernel
    def computeY(self):
        # for v in self.my_mesh.mesh.verts:
        #     v.y = v.x + v.v * self.dt + (v.f / v.m) * self.dtSq

        for n in self.nodes:
            self.nodes[n].y = self.nodes[n].x + self.nodes[n].v * self.dt + (self.nodes[n].f / self.nodes[n].m) * self.dtSq

    def update(self):

        # self.computeExternalForce()
        # self.my_mesh.mesh.verts.f.fill([0.0, self.gravity, 0.0])
        self.nodes.f.fill([0.0, self.gravity, 0.0])
        self.computeY()
        self.nodes.x_k.copy_from(self.nodes.y)
        for i in range(self.max_iter):
            self.computeGradientAndElementWiseHessian()
            self.solveNewtonIterationWithCG()


        self.computeNextState()


