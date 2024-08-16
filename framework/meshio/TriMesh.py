import taichi as ti
import numpy as np
import meshio
import igl
import random
import os
from pathlib import Path
from pyquaternion import Quaternion

model_path = Path(__file__).resolve().parent.parent.parent / "models"
OBJ = "OBJ"
dir = str(model_path) + "/OBJ"

@ti.data_oriented
class TriMesh:
    def __init__(
            self,
            model_dir,
            model_name_list=[],
            trans_list=[],
            rot_list=[],
            scale_list=[],
            is_static=False):

        self.is_static = is_static
        self.offsets = [] # offsets of each mesh

        self.num_verts = 0
        self.num_faces = 0
        self.num_edges = 0
        self.x_np = np.empty((0,3), dtype=float) # the vertices of mesh
        self.f_np = np.empty((0,3), dtype=int)   # the faces of mesh
        self.e_np = np.empty((0,2), dtype=int)   # the edges of mesh

        # concatenate all of meshes
        for i in range(len(model_name_list)):
            model_path = model_dir + "/" + model_name_list[i]
            mesh = meshio.read(model_path)
            if len(self.offsets) == 0:
                self.offsets.append(0)
            else:
                self.offsets.append(self.num_verts)

            scale_lf = lambda x, sc: sc * x
            trans_lf = lambda x, trans: x + trans

            # rotate, scale, and translate all vertices
            x_np_temp = np.array(mesh.points, dtype=float)
            center = x_np_temp.sum(axis=0) / x_np_temp.shape[0] # center position of the mesh
            x_np_temp = np.apply_along_axis(lambda row: trans_lf(row, -center), 1, x_np_temp) # translate to origin

            x_np_temp = scale_lf(x_np_temp, scale_list[i]) # scale mesh to the particular ratio
            if len(rot_list) > 0: # rotate mesh if it is demanded...
                rot_quaternion = Quaternion(axis=[rot_list[i][0], rot_list[i][1], rot_list[i][2]], angle=rot_list[i][3])
                rot_matrix = rot_quaternion.rotation_matrix
                for j in range(x_np_temp.shape[0]):
                    x_np_temp[j] = rot_matrix @ x_np_temp[j]

            x_np_temp = np.apply_along_axis(lambda row: trans_lf(row, trans_list[i]), 1, x_np_temp) # translate back to the particular position
            self.x_np = np.append(self.x_np, x_np_temp, axis=0)
            self.num_verts += mesh.points.shape[0]

            self.num_faces += len(mesh.cells_dict.get("triangle", []))
            self.f_np = np.append(self.f_np, np.array(mesh.cells_dict["triangle"]), axis=0)

            edges = set()
            for face in mesh.cells_dict["triangle"]:
                edges.add(tuple(sorted([face[0], face[1]])))
                edges.add(tuple(sorted([face[1], face[2]])))
                edges.add(tuple(sorted([face[2], face[0]])))
            self.num_edges += len(edges)
            self.e_np = np.append(self.e_np, np.array(list(edges)), axis=0)

        # fields about vertices
        self.y = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.y_origin = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.x = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.x0 = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.v = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.dx = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.dv = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.nc = ti.field(dtype=float, shape=self.num_verts)
        self.dup = ti.field(dtype=float, shape=self.num_verts)
        self.m_inv = ti.field(dtype=float, shape=self.num_verts)
        self.fixed = ti.field(dtype=float, shape=self.num_verts)
        self.colors = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)

        # initialize the vertex fields
        self.y.fill(0.0)
        self.y_origin.fill(0.0)
        self.x.from_numpy(self.x_np)
        self.x0.copy_from(self.x)
        self.v.fill(0.0)
        self.dx.fill(0.0)
        self.dv.fill(0.0)
        self.nc.fill(0.0)
        self.dup.fill(0.0)
        self.m_inv.fill(0.0)
        if self.is_static is False:
            self.fixed.fill(1.0)
        else:
            self.fixed.fill(0.0)

        # fields about edges
        self.l0 = ti.field(dtype=float, shape=self.num_edges)
        self.eid_field = ti.field(dtype=int, shape=(self.num_edges, 2))

        # initialize the edge fields
        self.l0.fill(0.0)
        self.eid_field.from_numpy(self.e_np)
        self.edge_indices_flatten = ti.field(dtype=ti.int32, shape=self.num_edges * 3)

        # fields about faces
        self.aabb_min = ti.field(dtype=float, shape=self.num_faces)
        self.aabb_max = ti.field(dtype=float, shape=self.num_faces)
        self.morton_code = ti.field(dtype=ti.uint32, shape=self.num_faces)
        self.fid_field = ti.field(dtype=int, shape=(self.num_faces, 3))
        self.face_indices_flatten = ti.field(dtype=ti.int32, shape=self.num_faces * 3)

        # initialize the face fields
        self.fid_field.from_numpy(self.f_np)

        self.init_edge_indices_flatten()
        self.init_face_indices_flatten()
        self.init_l0_m_inv()
        self.init_color()

    ####################################################################################################################
    def reset(self):
        self.y.fill(0.0)
        self.y_origin.fill(0.0)
        self.x.from_numpy(self.x_np)
        self.v.fill(0.0)
        self.dx.fill(0.0)
        self.dv.fill(0.0)
        self.nc.fill(0.0)

    @ti.kernel
    def init_edge_indices_flatten(self):
        for i in range(self.num_edges):
            self.edge_indices_flatten[2 * i + 0] = self.eid_field[i, 0]
            self.edge_indices_flatten[2 * i + 1] = self.eid_field[i, 1]

    @ti.kernel
    def init_face_indices_flatten(self):
        for i in range(self.num_faces):
            self.face_indices_flatten[3 * i + 0] = self.fid_field[i, 0]
            self.face_indices_flatten[3 * i + 1] = self.fid_field[i, 1]
            self.face_indices_flatten[3 * i + 2] = self.fid_field[i, 2]

    def init_color(self):
        for i in range(len(self.offsets)):
            r = random.randrange(0, 255) / 256
            g = random.randrange(0, 255) / 256
            b = random.randrange(0, 255) / 256

            size = 0
            if i < len(self.offsets) - 1:
                size = self.offsets[i+1] - self.offsets[i]
            else:
                size = self.num_verts - self.offsets[i]
            self.init_color_kernel(offset=self.offsets[i], size=size, color=ti.math.vec3(r,g,b))

    @ti.kernel
    def init_color_kernel(self, offset: ti.i32, size: ti.i32, color: ti.math.vec3):
        for i in range(size):
            self.colors[i+offset] = color

    @ti.kernel
    def init_l0_m_inv(self):
        for i in range(self.num_edges):
            v0, v1 = self.eid_field[i,0], self.eid_field[i,1]
            self.l0[i] = (self.x[v0] - self.x[v1]).norm()
            self.m_inv[v0] += 0.5 * self.l0[i]
            self.m_inv[v1] += 0.5 * self.l0[i]