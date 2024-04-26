import numpy as np
import taichi as ti
import meshtaichi_patcher as patcher

@ti.data_oriented
class TetMesh:

    def __init__(self,
                 model_path,
                 trans=ti.math.vec3(0, 0, 0),
                 rot=ti.math.vec3(0, 0, 0),
                 scale=1.0,
                 density=1.0,
                 is_static=False):

        self.is_static = is_static
        self.tet_mesh = patcher.load_mesh(model_path, relations=["FV", "CV", "VV", "CE"])
        self.tet_mesh.verts.place({'m': ti.f32,
                               'm_inv': ti.f32,
                               'x0': ti.math.vec3,
                               'x': ti.math.vec3,
                               'v': ti.math.vec3,
                               'f_ext': ti.math.vec3,
                               'y': ti.math.vec3,
                               'ld': ti.f32,
                               'x_k': ti.math.vec3,
                               'p': ti.math.vec3,
                               'nc': ti.uint32,
                                   'deg': ti.uint32,
                                   'dx': ti.math.vec3,
                                   'g': ti.math.vec3,
                                   'gc': ti.math.vec3,
                                   'h': ti.f32,
                                   'hc': ti.f32})
        self.density = density
        self.tet_mesh.verts.m.fill(1.0)
        if self.is_static:
            self.tet_mesh.verts.m_inv.fill(0.0)

        else:
            self.tet_mesh.verts.m_inv.fill(1.0)

        self.tet_mesh.verts.v.fill([0.0, 0.0, 0.0])
        self.tet_mesh.verts.x.from_numpy(self.tet_mesh.get_position_as_numpy())
        self.num_verts = len(self.tet_mesh.verts)


        self.tet_mesh.edges.place({'Dm_inv': ti.math.mat3, 'V0': ti.f32}) # bounding sphere radius


        self.face_indices = ti.field(dtype=ti.i32, shape=len(self.tet_mesh.faces) * 3)
        self.edge_indices = ti.field(dtype=ti.i32, shape=len(self.tet_mesh.edges) * 2)
        self.tetra_indices = ti.field(dtype=ti.i32, shape=len(self.tet_mesh.cells) * 4)
        self.initFaceIndices()
        self.initTetraIndices()
        # self.initEdgeIndices()

        print(len(self.tet_mesh.faces))
        print(len(self.tet_mesh.cells))

        self.verts = self.tet_mesh.verts
        self.cells = self.tet_mesh.cells

        self.trans = trans
        self.rot = rot
        self.scale = scale

        self.applyTransform()
        # self.computeInitialLength()
        # self.compute_Dm_inv()
        self.tet_mesh.verts.x0.copy_from(self.tet_mesh.verts.x)


    def reset(self):
        self.tet_mesh.verts.x.copy_from(self.tet_mesh.verts.x0)
        self.tet_mesh.verts.v.fill(0.)

    @ti.kernel
    def compute_Dm_inv(self):
        for c in self.tet_mesh.cells:
            Ds = ti.Matrix.cols([c.verts[i].x - c.verts[3].x for i in ti.static(range(3))])
            c.B = Ds.inverse()
            c.W = ti.abs(Ds.determinant()) / 6
            ind = [[0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3]]

            for i in ti.static(range(4)):
                c.verts[i].m += self.density * c.W / 4


    @ti.kernel
    def initFaceIndices(self):
        for f in self.tet_mesh.faces:
            self.face_indices[f.id * 3 + 0] = f.verts[0].id
            self.face_indices[f.id * 3 + 1] = f.verts[1].id
            self.face_indices[f.id * 3 + 2] = f.verts[2].id

    @ti.kernel
    def initEdgeIndices(self):
        for e in self.tet_mesh.edges:
            self.edge_indices[e.id * 2 + 0] = e.verts[0].id
            self.edge_indices[e.id * 2 + 1] = e.verts[1].id

    @ti.kernel
    def initTetraIndices(self):
        for c in self.tet_mesh.cells:
            self.tetra_indices[c.id * 4 + 0] = c.verts[0].id
            self.tetra_indices[c.id * 4 + 1] = c.verts[1].id
            self.tetra_indices[c.id * 4 + 2] = c.verts[2].id
            self.tetra_indices[c.id * 4 + 3] = c.verts[3].id

    @ti.kernel
    def setCenterToOrigin(self):

        center = ti.math.vec3(0, 0, 0)
        for v in self.tet_mesh.verts:
            center += v.x

        center /= self.num_verts
        for v in self.tet_mesh.verts:
            v.x -= center

    @ti.kernel
    def applyTransform(self):

        # self.setCenterToOrigin()

        for v in self.tet_mesh.verts:
            v.x *= self.scale

        for v in self.tet_mesh.verts:
            v_4d = ti.Vector([v.x[0], v.x[1], v.x[2], 1])
            rot_rad = ti.math.radians(self.rot)
            rv = ti.math.rotation3d(rot_rad[0], rot_rad[1], rot_rad[2]) @ v_4d
            v.x = ti.Vector([rv[0], rv[1], rv[2]])

        for v in self.tet_mesh.verts:
            v.x += self.trans

