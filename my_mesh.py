import taichi as ti
import meshtaichi_patcher as patcher

@ti.data_oriented
class Mesh:

    def __init__(self,
                 model_path,
                 trans=ti.math.vec3(0, 0, 0),
                 rot=ti.math.vec3(0, 0, 0),
                 scale=1.0):

        self.mesh = patcher.load_mesh(model_path, relations=["FV", "EV"])
        self.mesh.verts.place({'m': ti.f32,
                               'x': ti.math.vec3,
                               'v': ti.math.vec3,
                               'f_ext': ti.math.vec3,
                               'y': ti.math.vec3,
                               'x_k': ti.math.vec3,
                               'g': ti.math.vec3,
                               'h': ti.math.mat3,
                               'dx': ti.math.vec3,
                               'b': ti.math.vec3,
                               'r': ti.math.vec3,
                               'p': ti.math.vec3,
                               'Ap': ti.math.vec3
                               })



        self.mesh.verts.m.fill(1.0)
        self.mesh.verts.v.fill([0.0, 0.0, 0.0])
        self.mesh.verts.x.from_numpy(self.mesh.get_position_as_numpy())
        self.num_verts = len(self.mesh.verts)

        self.mesh.edges.place({'l0': ti.f32,
                               'hij': ti.math.mat3})
        self.setCenterToOrigin()
        self.face_indices = ti.field(dtype=ti.i32, shape=len(self.mesh.faces) * 3)
        self.edge_indices = ti.field(dtype=ti.i32, shape=len(self.mesh.edges) * 2)
        self.initFaceIndices()
        self.initEdgeIndices()

        self.trans = trans
        self.rot = rot
        self.scale = scale

        self.applyTransform()
        self.computeInitialLength()

    @ti.kernel
    def computeInitialLength(self):
        for e in self.mesh.edges:
            e.l0 = (e.verts[0].x - e.verts[1].x).norm()

    @ti.kernel
    def initFaceIndices(self):
        for f in self.mesh.faces:
            self.face_indices[f.id * 3 + 0] = f.verts[0].id
            self.face_indices[f.id * 3 + 1] = f.verts[1].id
            self.face_indices[f.id * 3 + 2] = f.verts[2].id

    @ti.kernel
    def initEdgeIndices(self):
        for e in self.mesh.edges:
            self.edge_indices[e.id * 2 + 0] = e.verts[0].id
            self.edge_indices[e.id * 2 + 1] = e.verts[1].id

    @ti.kernel
    def setCenterToOrigin(self):

        center = ti.math.vec3(0, 0, 0)
        for v in self.mesh.verts:
            center += v.x

        center /= self.num_verts
        for v in self.mesh.verts:
            v.x -= center

    @ti.kernel
    def applyTransform(self):

        for v in self.mesh.verts:
            v.x *= self.scale

        for v in self.mesh.verts:
            v_4d = ti.Vector([v.x[0], v.x[1], v.x[2], 1])
            rot_rad = ti.math.radians(self.rot)
            rv = ti.math.rotation3d(rot_rad[0], rot_rad[1], rot_rad[2]) @ v_4d
            v.x = ti.Vector([rv[0], rv[1], rv[2]])

        for v in self.mesh.verts:
            v.x += self.trans