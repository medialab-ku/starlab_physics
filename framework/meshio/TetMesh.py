import numpy as np
import taichi as ti
import meshio
import random
import os
import igl

@ti.data_oriented
class TetMeshWrapper:

    def __init__(self,
                 model_dir,
                 model_name_list=[],
                 trans_list=[],
                 rot=ti.math.vec3(0, 0, 0),
                 scale_list=[],
                 density=1.0,
                 is_static=False):

        num_verts = 0
        num_tetras = 0
        x_np = np.empty((0, 3), dtype=float)
        tet_indices_np = np.empty((0, 4), dtype=int)
        offsets = [0]
        for i in range(len(model_name_list)):
            model_path = model_dir + "/" + model_name_list[i]
            mesh = meshio.read(model_path)
            scale_lf = lambda x, sc: sc * x
            trans_lf = lambda x, trans: x + trans

            x_np_temp = np.array(mesh.points, dtype=float)
            center = x_np_temp.sum(axis=0) / x_np_temp.shape[0]
            # print(center)
            x_np_temp = np.apply_along_axis(lambda row: trans_lf(row, -center), 1, x_np_temp)
            x_np_temp = scale_lf(x_np_temp, scale_list[i])
            x_np_temp = np.apply_along_axis(lambda row: trans_lf(row, trans_list[i]), 1, x_np_temp)
            x_np = np.append(x_np, x_np_temp, axis=0)
            tet_indices_np_temp = np.array(mesh.cells[0].data, dtype=int) + num_verts
            offset = num_verts * np.ones(shape=4, dtype=int)
            # tet_indices_np_temp = np.apply_along_axis(lambda row: trans_lf(row, offset), 1, tet_indices_np_temp)
            tet_indices_np = np.append(tet_indices_np, tet_indices_np_temp, axis=0)
            num_verts += mesh.points.shape[0]
            num_tetras += mesh.cells[0].data.shape[0]
            offsets.append(num_verts)

        # print(offsets)

        self.y = ti.Vector.field(n=3, dtype=float)
        self.x = ti.Vector.field(n=3, dtype=float)
        self.dx = ti.Vector.field(n=3, dtype=float)
        self.nc = ti.field(dtype=float)
        self.x0 = ti.Vector.field(n=3, dtype=float)
        self.v = ti.Vector.field(n=3, dtype=float)
        self.invM = ti.field(dtype=float)
        self.M = ti.field(dtype=float)
        self.color = ti.Vector.field(n=3, dtype=float)

        # print(x_np)

        dnode = ti.root.dense(ti.i, num_verts)
        dnode.place(self.y, self.dx, self.nc, self.x, self.v, self.invM, self.M)
        dnode.place(self.x0, self.color)

        self.init_color(offsets)

        self.x.from_numpy(x_np)
        self.x0.copy_from(self.x)
        self.v.fill(0.0)

        self.invDm = ti.Matrix.field(n=3, m=3, dtype=float)
        self.V0 = ti.field(dtype=float)
        self.tet_indices = ti.field(dtype=int)

        node = ti.root.dense(ti.i, num_tetras)
        node.place(self.invDm, self.V0)
        ti.root.dense(ti.ij, (num_tetras, 4)).place(self.tet_indices)
        self.tet_indices.from_numpy(tet_indices_np)

        f = self.list_faces(tet_indices_np)
        _, indxs, count = np.unique(f, axis=0, return_index=True, return_counts=True)

        surface_indices_np = f[indxs[count == 1]]
        surface_indices_np.astype(int)
        num_faces = surface_indices_np.shape[0]
        # print(surface_indices_np)

        self.surface_indices = ti.field(dtype=int)
        ti.root.dense(ti.i, 3 * num_faces).place(self.surface_indices)
        self.surface_indices.from_numpy(surface_indices_np.reshape(3 * num_faces))

        self.init()

    def reset(self):
        self.x.copy_from(self.x0)
        self.v.fill(0.0)

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

    def init_color(self, offsets):
        # print(self.offsets)
        for i in range(len(offsets) - 1):
            size = offsets[i + 1] - offsets[i]
            # if is_static[i] is True:
            #     self.init_colors(self.offsets[i], size, color=ti.math.vec3(0.5, 0.5, 0.5))
            # else:
            r = float(random.randrange(0, 255) / 256)
            g = float(random.randrange(0, 255) / 256)
            b = float(random.randrange(0, 255) / 256)

            self.init_colors(offsets[i], size, color=ti.math.vec3(r, g, b))

    @ti.kernel
    def init_colors(self, offset: ti.i32, size: ti.i32, color: ti.math.vec3):

        for i in range(size):
            self.color[i + offset] = color

    @ti.kernel
    def init(self):
        self.M.fill(0.0)
        for i in self.invDm:
            Dm_i = ti.Matrix.cols([self.y[self.tet_indices[i, j]] - self.y[self.tet_indices[i, 3]] for j in ti.static(range(3))])
            self.invDm[i] = Dm_i.inverse()
            V0_i = ti.abs(Dm_i.determinant()) / 6.0

            for j in ti.static(range(4)):
                self.M[self.tet_indices[i, j]] += 0.25 * V0_i

            self.V0[i] = ti.abs(Dm_i.determinant()) / 6.0

        for i in self.invM:
            self.invM[i] = 1.0 / self.M[i]

