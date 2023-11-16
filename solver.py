import taichi as ti
import numpy as np
import meshtaichi_patcher as Patcher
import mf

from aabbtree import AABB
from aabbtree import AABBTree

mf = mf.mathFunctions()

@ti.dataclass
class vertex:
    p: ti.math.vec3

@ti.data_oriented
class solver:
    def __init__(self,
                 mesh,
                 primitive,
                 dt=1e-3,
                 max_iter=1000):
        self.dt = dt
        self.max_iter = max_iter
        self.frame = 0
        self.gravity = ti.Vector.field(3, ti.f32, shape=())
        self.gravity[None] = ti.Vector([0.0, -10.0, 0.0])

        self.mesh = mesh
        self.indices = ti.field(ti.u32, shape=len(mesh.faces) * 3)

        self.primitive = primitive
        self.p_indices = ti.field(ti.u32, shape=len(primitive.faces) * 3)

        self.grid_n = 100

        self.column_sum = ti.field(dtype=ti.i32, shape=self.grid_n, name="column_sum")
        self.prefix_sum = ti.field(dtype=ti.i32, shape=(self.grid_n, self.grid_n), name="prefix_sum")
        self.grain_count = ti.field(dtype=ti.i32,
                               shape=(self.grid_n, self.grid_n),
                               name="grain_count")

        self.vf = vertex.field(shape=(len(mesh.verts)+len(primitive.verts), ))


    @ti.kernel
    def initialize(self):
        # for e in self.mesh.edges:
        #     e.rest_len = (e.verts[0].x - e.verts[1].x).norm()

        for p_v0 in self.primitive.verts:
            p_v0_4 = ti.Vector([p_v0.x[0], p_v0.x[1], p_v0.x[2], 1])
            rot_rad = ti.math.radians(ti.Vector([90.0, 0.0, 0.0]))
            p = ti.math.rotation3d(rot_rad[0], rot_rad[1], rot_rad[2]) @ p_v0_4
            p_v0.x = ti.Vector([p[0], p[1], p[2]])

    @ti.kernel
    def setMeshPos(self, scale : ti.f32, offset : ti.template()):
        for v in self.mesh.verts:
            v.x = v.x * scale + ti.Vector(offset)

    @ti.kernel
    def initIndices(self):
        for f in self.mesh.faces:
            self.indices[f.id * 3 + 0] = f.verts[0].id
            self.indices[f.id * 3 + 1] = f.verts[1].id
            self.indices[f.id * 3 + 2] = f.verts[2].id

        for p_f in self.primitive.faces:
            self.p_indices[p_f.id * 3 + 0] = p_f.verts[0].id
            self.p_indices[p_f.id * 3 + 1] = p_f.verts[1].id
            self.p_indices[p_f.id * 3 + 2] = p_f.verts[2].id

    @ti.kernel
    def computeNormal(self):
        ti.mesh_local(self.mesh.verts.x, self.mesh.verts.n)
        for f in self.mesh.faces:
            v0 = f.verts[0]
            v1 = f.verts[1]
            v2 = f.verts[2]
            self.mesh.verts.n[f.id] = (v1.x - v0.x).cross(v2.x - v0.x).normalized()

        for p_f in self.primitive.faces:
            v0 = p_f.verts[0]
            v1 = p_f.verts[1]
            v2 = p_f.verts[2]
            self.primitive.verts.n[p_f.id] = (v1.x - v0.x).cross(v2.x - v0.x).normalized()

    @ti.kernel
    def solveStretch(self, dt : ti.f32):
        ti.loop_config(block_dim=self.block_size)
        ti.mesh_local(self.mesh.verts.dp, self.mesh.verts.invM, self.mesh.verts.new_x)
        for e in self.mesh.edges:
            v0, v1 = e.verts[0], e.verts[1]
            w1, w2 = v0.invM, v1.invM
            if w1 + w2 > 0.:
                n = v0.new_x - v1.new_x
                d = n.norm()
                dp = ti.zero(n)
                constraint = (d - e.rest_len)
                if ti.static(self.XPBD): # https://matthias-research.github.io/pages/publications/XPBD.pdf
                    compliance = e.stretch_compliance / (dt**2)
                    d_lambda = -(constraint + compliance * e.la_s) / (w1 + w2 + compliance) * self.stretch_relaxation # eq. (18)
                    dp = d_lambda * n.normalized(1e-12) # eq. (17)
                    e.la_s += d_lambda
                else: # https://matthias-research.github.io/pages/publications/posBasedDyn.pdf
                    dp = -constraint / (w1 + w2) * n.normalized(1e-12) * self.stretch_relaxation # eq. (1)
                v0.dp += dp * w1
                v1.dp -= dp * w2

    @ti.kernel
    def applyExtForce(self):
        for v0 in self.mesh.verts:
            v0.v += self.gravity[None] * self.dt
            v0.new_x = v0.x + v0.v * self.dt

    @ti.kernel
    def update(self):
        for v0 in self.mesh.verts:
            v0.x = v0.new_x


    ### collision detection ###
    # def collision_detection(self):
    #     tree = AABBTree()
    #
    #     for i, v in enumerate(self.mesh.verts.x.to_numpy()):
    #         aabb_v = AABB([(v[0]-0.5, v[0]+0.5),
    #                        (v[1]-0.5, v[1]+0.5),
    #                         (v[2]-0.5, v[2]+0.5)])
    #         tree.add(aabb_v, 'v'+str(i))
    #
    #     p_x = self.primitive.verts.x.to_numpy()
    #     p_i = self.p_indices.to_numpy().resize(-1, 3)
    #     for i, p_v in enumerate(p_i):
    #         max_x = max(p_x[p_v[0]][0], p_x[p_v[1]][0], p_x[p_v[2]][0]) + 0.5
    #         min_x = min(p_x[p_v[0]][0], p_x[p_v[1]][0], p_x[p_v[2]][0]) - 0.5
    #         max_y = max(p_x[p_v[0]][1], p_x[p_v[1]][1], p_x[p_v[2]][1]) + 0.5
    #         min_y = min(p_x[p_v[0]][1], p_x[p_v[1]][1], p_x[p_v[2]][1]) - 0.5
    #         max_z = max(p_x[p_v[0]][2], p_x[p_v[1]][2], p_x[p_v[2]][2]) + 0.5
    #         min_z = min(p_x[p_v[0]][2], p_x[p_v[1]][2], p_x[p_v[2]][2]) - 0.5
    #         aabb_p_v = AABB([(min_x, max_x),
    #                             (min_y, max_y),
    #                             (min_z, max_z)])
    #         tree.does_overlap(aabb_p_v)
    #
    #     print(tree)