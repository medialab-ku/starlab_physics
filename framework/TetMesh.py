import numpy as np
import taichi as ti
import meshtaichi_patcher as patcher

import os
import igl

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
        self.tet_mesh = patcher.load_mesh(model_path, relations=["FV", "CV", "EV"])
        self.tet_mesh.verts.place({'x0': ti.math.vec3, 'x': ti.math.vec3, 'v': ti.math.vec3})
        self.density = density
        self.tet_mesh.verts.v.fill([0.0, 0.0, 0.0])
        self.tet_mesh.verts.x.from_numpy(self.tet_mesh.get_position_as_numpy())
        self.num_verts = len(self.tet_mesh.verts)


        self.face_indices = ti.field(dtype=ti.i32, shape=len(self.tet_mesh.faces) * 3)
        self.edge_indices = ti.field(dtype=ti.i32, shape=len(self.tet_mesh.edges) * 2)
        self.tetra_indices = ti.field(dtype=ti.i32, shape=len(self.tet_mesh.cells) * 4)
        self.initTetraIndices()
        self.fid_np = self.get_surface_id()
        self.num_faces = self.fid_np.shape[0]
        # print(self.fid_np.shape)
        self.fid = ti.field(dtype=ti.i32, shape=self.fid_np.shape)
        self.fid.from_numpy(self.fid_np)
        # self.initEdgeIndices()

        # print(len(self.tet_mesh.verts))
        # print(len(self.tet_mesh.cells))

        self.verts = self.tet_mesh.verts
        self.cells = self.tet_mesh.cells
        self.faces = self.tet_mesh.faces
        self.edges = self.tet_mesh.edges

        self.trans = trans
        self.rot = rot
        self.scale = scale

        self.applyTransform()
        # self.computeInitialLength()
        # self.compute_Dm_inv()
        self.tet_mesh.verts.x0.copy_from(self.tet_mesh.verts.x)


        # self.fid_np = self.face_indices.to_numpy()
        # self.fid_np = np.reshape(self.fid_np, (len(self.tet_mesh.faces), 3))
        #
        # print("rawFID : ",self.fid_np.shape)
        self.initFaceIndices()

    def reset(self):
        self.tet_mesh.verts.x.copy_from(self.tet_mesh.verts.x0)
        self.tet_mesh.verts.v.fill(0.)


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


    def export(self, scene_name, mesh_id, frame):
        directory = os.path.join("results/",scene_name,"TetMesh_ID_"+str(mesh_id))

        try :
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print("Error: Failed to create folder" + directory)

        x_np = self.tet_mesh.verts.x.to_numpy()
        file_name = "TetMesh_obj_" + str(frame) + ".obj"
        file_path = os.path.join(directory, file_name)
        print("exporting ", file_path.__str__())

        igl.write_triangle_mesh(file_path, x_np, self.fid_np, force_ascii=True)

        print("done")

    def __list_faces(self,t):
        t.sort(axis=1)
        n_t, m_t = t.shape
        f = np.empty((4 * n_t, 3), dtype=int)
        i = 0
        for j in range(4):
            f[i:i + n_t, 0:j] = t[:, 0:j]
            f[i:i + n_t, j:3] = t[:, j + 1:4]
            i = i + n_t
        return f

    def __extract_unique_triangles(self,t):
        _, indxs, count = np.unique(t, axis=0, return_index=True, return_counts=True)
        return t[indxs[count == 1]]

    def get_surface_id(self):
        # https://stackoverflow.com/questions/66607716/how-to-extract-surface-triangles-from-a-tetrahedral-mesh
        tid_np = self.tetra_indices.to_numpy()
        tid_np = np.reshape(tid_np, (len(self.tet_mesh.cells), 4))

        f = self.__list_faces(tid_np)
        f = self.__extract_unique_triangles(f)

        return f


