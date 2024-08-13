import numpy as np
import taichi as ti
import meshio
import gmshparser

import os
import igl

@ti.data_oriented
class TetMeshWrapper:

    def __init__(self,
                 model_dir,
                 model_name,
                 trans=ti.math.vec3(0, 0, 0),
                 rot=ti.math.vec3(0, 0, 0),
                 scale=1.0,
                 density=1.0,
                 is_static=False):

        model_path = model_dir + "/" + model_name
        test = gmshparser.parse(model_path)

        # print(test)
        # print(model_dir + "/" + model_name)
        mesh = meshio.read(model_path)
        num_verts = mesh.points.shape[0]
        x_np = np.array(mesh.points, dtype=float)

        scale_lf = lambda x, sc: sc * x

        trans_lf = lambda x, trans: x + trans

        center = x_np.sum(axis=0) / x_np.shape[0]

        x_np = np.apply_along_axis(lambda row: trans_lf(row, -center), 1, x_np)
        x_np = scale_lf(x_np, scale)
        x_np = np.apply_along_axis(lambda row: trans_lf(row, trans), 1, x_np)

        self.y = ti.Vector.field(n=3, dtype=float)
        self.x = ti.Vector.field(n=3, dtype=float)
        self.x0 = ti.Vector.field(n=3, dtype=float)
        self.v = ti.Vector.field(n=3, dtype=float)
        dnode = ti.root.dense(ti.i, num_verts)
        dnode.place(self.y, self.x, self.v)
        dnode.place(self.x0)

        self.x.from_numpy(x_np)
        self.x0.copy_from(self.x)

        self.v.fill(0.0)

        tet_indices_np = np.array(mesh.cells[0].data, dtype=int)
        num_tetras = tet_indices_np.shape[0]
        self.Dm_inv = ti.Matrix.field(n=3, m=3, dtype=int)
        self.tet_indices = ti.field(dtype=int)
        node = ti.root.dense(ti.i, num_tetras)
        node.place(self.Dm_inv)
        ti.root.dense(ti.ij, (num_tetras, 4)).place(self.tet_indices)
        self.tet_indices.from_numpy(tet_indices_np)
        # print(mesh.cells)

        f = self.list_faces(tet_indices_np)
        _, indxs, count = np.unique(f, axis=0, return_index=True, return_counts=True)

        surface_indices_np = f[indxs[count == 1]]
        surface_indices_np.astype(int)
        num_faces = surface_indices_np.shape[0]
        # print(surface_indices_np)
        # print(surface_indices_np.reshape(3 * num_faces))

        self.surface_indices = ti.field(dtype=int)
        ti.root.dense(ti.i, 3 * num_faces).place(self.surface_indices)
        self.surface_indices.from_numpy(surface_indices_np.reshape(3 * num_faces))

        # print(mesh)

    def reset(self):
        self.x.copy_from(self.x0)

    def list_faces(self, t):
        t.sort(axis=1)
        n_t, m_t = t.shape
        f = np.empty((4 * n_t, 3), dtype=int)
        i = 0
        for j in range(4):
            f[i:i + n_t, 0:j] = t[:, 0:j]
            f[i:i + n_t, j:3] = t[:, j + 1:4]
            i = i + n_t
        return f

    def extract_unique_triangles(self, t):
        _, indxs, count = np.unique(t, axis=0, return_index=True, return_counts=True)
        return t[indxs[count == 1]]

    def get_surface_id(self):
        # https://stackoverflow.com/questions/66607716/how-to-extract-surface-triangles-from-a-tetrahedral-mesh
        tid_np = self.tetra_indices.to_numpy()
        tid_np = np.reshape(tid_np, (len(self.tet_mesh.cells), 4))

        # print("tid")
        # print(tid_np)
        # print(tid_np.shape)

        f = self.list_faces(tid_np)
        _, indxs, count = np.unique(f, axis=0, return_index=True, return_counts=True)

        return f[indxs[count == 1]]



