import taichi as ti
import numpy as np
import meshio
import igl
import random
import os
from pathlib import Path

ti.init()

model_path = Path(__file__).resolve().parent.parent.parent / "models"
OBJ = "OBJ"
dir = str(model_path) + "/OBJ"

@ti.data_oriented
class TriMesh:
    def __init__(
            self,
            model_dir,
            model_name_list=[],
            offsets=[0],
            trans_list=[],
            rot=ti.math.vec3(0, 0, 0),
            scale_list=[],
            is_static=False):

        self.is_static = is_static

        self.num_verts = 0
        self.num_faces = 0
        self.num_edges = 0
        self.x_np = np.empty((0,3), dtype=float) # the vertices of mesh
        self.f_np = np.empty((0,3), dtype=int)   # the faces of mesh
        self.e_np = np.empty((0,2), dtype=int)   # the edges of mesh

        for i in range(len(model_name_list)):
            model_path = model_dir + "/" + model_name_list[i]
            mesh = meshio.read(model_path)
            scale_lf = lambda x, sc: sc * x
            trans_lf = lambda x, trans: x + trans

            x_np_temp = np.array(mesh.points, dtype=float)
            center = x_np_temp.sum(axis=0) / x_np_temp.shape[0] # center position of the mesh
            x_np_temp = np.apply_along_axis(lambda row: trans_lf(row, -center), 1, x_np_temp) # translate to origin
            x_np_temp = scale_lf(x_np_temp, scale_list[i]) # scale mesh to the particular ratio
            x_np_temp = np.apply_along_axis(lambda row: trans_lf(row, trans_list[i]), 1, x_np_temp) # translate to the particular position
            self.x_np = np.append(self.x_np, x_np_temp, axis=0)
            self.num_verts += mesh.points.shape[0]

            self.num_faces += len(mesh.cells_dict.get("triangle", []))
            self.f_np = np.array(mesh.cells_dict["triangle"])

            edges = set()
            for face in mesh.cells_dict["triangle"]:
                edges.add(tuple(sorted([face[0], face[1]])))
                edges.add(tuple(sorted([face[1], face[2]])))
                edges.add(tuple(sorted([face[2], face[0]])))
            self.num_edges = len(edges)
            self.e_np = np.array(list(edges))

        # fields about vertices
        self.y = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.x = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.x0 = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.y_origin = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.v = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.dx = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.dv = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.nc = ti.field(dtype=float, shape=self.num_verts)
        self.dup = ti.field(dtype=float, shape=self.num_verts)
        self.m_inv = ti.field(dtype=float, shape=self.num_verts)

        # fields about edges
        self.l0 = ti.field(dtype=float, shape=self.num_edges)
        self.eid_field = ti.Vector.field(n=2, dtype=int, shape=self.num_edges)

        # fields about faces
        self.aabb_min = ti.field(dtype=float, shape=self.num_faces)
        self.aabb_max = ti.field(dtype=float, shape=self.num_faces)
        self.morton_code = ti.field(dtype=ti.uint32, shape=self.num_faces)

        print(self.num_verts, self.num_faces, self.num_edges)
        print(self.x_np)
        print(self.f_np)
        print(self.e_np)


mesh = TriMesh(dir, ["plane.obj"], offsets=[0], trans_list=[(0, 5.0, 0)], scale_list=[10.0], is_static=False)