import numpy as np
import taichi as ti
import meshtaichi_patcher as patcher
import igl
import os

@ti.data_oriented
class Mesh:

    def __init__(self,
                 model_path,
                 trans=ti.math.vec3(0, 0, 0),
                 rot=ti.math.vec3(0, 0, 0),
                 scale=1.0,
                 is_static=False):

        self.is_static = is_static
        self.mesh = patcher.load_mesh(model_path, relations=["FV", "EV"])
        self.mesh.verts.place({'m': ti.f32,
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



        self.mesh.verts.m.fill(1.0)
        if self.is_static:
            self.mesh.verts.m_inv.fill(0.0)

        else:
            self.mesh.verts.m_inv.fill(1.0)

        self.mesh.verts.v.fill([0.0, 0.0, 0.0])
        # self.x_np = self.mesh.get_position_as_numpy()
        self.mesh.verts.x.from_numpy(self.mesh.get_position_as_numpy())
        self.num_verts = len(self.mesh.verts)


        self.mesh.edges.place({'l0': ti.f32,
                               'ld': ti.f32,
                               'vid': ti.math.ivec2,
                               'x': ti.math.vec3,
                               'v': ti.math.vec3,
                               'hij': ti.math.mat3, # bounding sphere radius
                               'hinv': ti.math.mat2}) # bounding sphere radius

        # self.mesh.faces.place({'aabb_min': ti.math.vec3,
        #                        'aabb_max': ti.math.vec3})  # bounding sphere radius

        # self.setCenterToOrigin()
        self.face_indices = ti.field(dtype=ti.i32, shape=len(self.mesh.faces) * 3)
        self.edge_indices = ti.field(dtype=ti.i32, shape=len(self.mesh.edges) * 2)
        self.initFaceIndices()
        self.initEdgeIndices()
        self.fid_np = self.face_indices.to_numpy()
        self.fid_np = np.reshape(self.fid_np, (len(self.mesh.faces), 3))
        self.verts = self.mesh.verts
        self.faces = self.mesh.faces
        self.edges = self.mesh.edges

        self.trans = trans
        self.rot = rot
        self.scale = scale

        self.applyTransform()
        self.computeInitialLength()
        self.mesh.verts.x0.copy_from(self.mesh.verts.x)


    def reset(self):
        self.mesh.verts.x.copy_from(self.mesh.verts.x0)
        self.mesh.verts.v.fill(0.)

    @ti.kernel
    def computeInitialLength(self):
        for e in self.mesh.edges:
            e.l0 = (e.verts[0].x - e.verts[1].x).norm()
            e.vid[0] = e.verts[0].id
            e.vid[1] = e.verts[1].id

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

        # self.setCenterToOrigin()

        for v in self.mesh.verts:
            v.x *= self.scale

        for v in self.mesh.verts:
            v_4d = ti.Vector([v.x[0], v.x[1], v.x[2], 1])
            rot_rad = ti.math.radians(self.rot)
            rv = ti.math.rotation3d(rot_rad[0], rot_rad[1], rot_rad[2]) @ v_4d
            v.x = ti.Vector([rv[0], rv[1], rv[2]])

        for v in self.mesh.verts:
            v.x += self.trans

    def export(self, scene_name, mesh_id, frame):
        directory = os.path.join("results/",scene_name,"Mesh_ID_"+str(mesh_id))

        try :
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print("Error: Failed to create folder" + directory)

        x_np = self.mesh.verts.x.to_numpy()
        file_name = "Mesh_obj_" + str(frame) + ".obj"
        file_path = os.path.join(directory, file_name)
        print("exporting ", file_path.__str__())
        igl.write_triangle_mesh(file_path, x_np, self.fid_np, force_ascii=True)
        print("done")
