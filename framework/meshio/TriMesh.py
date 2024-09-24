import taichi as ti
import numpy as np
import os
import time
import meshio
import random
from pathlib import Path
from pyquaternion import Quaternion
import networkx as nx

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
        self.num_model = len(model_name_list)

        self.num_verts = 0
        self.num_faces = 0
        self.num_edges = 0
        self.x_np = np.empty((0, 3), dtype=float)  # the vertices of mesh
        self.f_np = np.empty((0, 3), dtype=int)  # the faces of mesh
        self.e_np = np.empty((0, 2), dtype=int)  # the edges of mesh

        self.vert_offsets = []  # offsets of each mesh
        self.edge_offsets = []  # offsets of each edge
        self.face_offsets = []  # offsets of each face
        self.vert_offsets.append(0)
        self.edge_offsets.append(0)
        self.face_offsets.append(0)

        model_name = model_name_list[0][:-4]
        print(model_name)

        path = '/home/mhkee/Desktop/workspace/starlab_physics/models/sampling/'+model_name+'.npy'
        load_array = np.load(path, allow_pickle=True)

        # print(load_array)
        self.sample_indices = ti.field(dtype=float, shape=load_array.shape)
        self.x_sample = ti.Vector.field(n=3, dtype=float, shape=load_array.shape[0])
        self.rho0_sample = ti.field(dtype=float, shape=load_array.shape[0])
        self.heat_map_sample = ti.Vector.field(n=3, dtype=float, shape=load_array.shape[0])

        if is_static is False:
            self.sample_indices.from_numpy(load_array)

        # concatenate all of meshes
        for i in range(self.num_model):
            model_path = model_dir + "/" + model_name_list[i]
            mesh = meshio.read(model_path)

            scale_lf = lambda x, sc: sc * x
            trans_lf = lambda x, trans: x + trans

            # rotate, scale, and translate all vertices
            x_np_temp = np.array(mesh.points, dtype=float)
            center = x_np_temp.sum(axis=0) / x_np_temp.shape[0]  # center position of the mesh
            x_np_temp = np.apply_along_axis(lambda row: trans_lf(row, -center), 1, x_np_temp)  # translate to origin
            x_np_temp = scale_lf(x_np_temp, scale_list[i])  # scale mesh to the particular ratio

            # rotate mesh if it is demanded...
            if len(rot_list) > 0:
                rot_quaternion = Quaternion(axis=[rot_list[i][0], rot_list[i][1], rot_list[i][2]], angle=rot_list[i][3])
                rot_matrix = rot_quaternion.rotation_matrix
                for j in range(x_np_temp.shape[0]):
                    x_np_temp[j] = rot_matrix @ x_np_temp[j]

            x_np_temp = np.apply_along_axis(lambda row: trans_lf(row, center), 1,
                                            x_np_temp)  # translate back to the original position
            x_np_temp = np.apply_along_axis(lambda row: trans_lf(row, trans_list[i]), 1,
                                            x_np_temp)  # translate again to the designated position!


            self.x_np = np.append(self.x_np, x_np_temp, axis=0)
            self.num_verts += mesh.points.shape[0]

            self.num_faces += len(mesh.cells_dict.get("triangle", []))

            face_temp = np.array(mesh.cells_dict["triangle"])
            face_temp = face_temp + self.vert_offsets[i]
            self.f_np = np.append(self.f_np, face_temp, axis=0)

            edges = set()
            for face in mesh.cells_dict["triangle"]:
                edges.add(tuple(sorted([face[0], face[1]])))
                edges.add(tuple(sorted([face[1], face[2]])))
                edges.add(tuple(sorted([face[2], face[0]])))
            self.num_edges += len(edges)

            edge_temp = np.array(list(edges))
            edge_temp = edge_temp + self.vert_offsets[i]
            self.e_np = np.append(self.e_np, edge_temp, axis=0)

            self.vert_offsets.append(self.vert_offsets[-1] + mesh.points.shape[0])
            self.edge_offsets.append(self.edge_offsets[-1] + len(edges))
            self.face_offsets.append(self.face_offsets[-1] + len(mesh.cells_dict.get("triangle", [])))

        #TODO
        # test_particle_np = np.zeros_like(self.x_np)
        # self.test_particles = ti.Vector.field(n=3, dtype=float)
        # size = 100
        # ti.root.dense(ti.i, size).place(self.test_particles)
        # self.test_particles.from_numpy(test_particle_np)
        # TODO


        # fields about vertices
        self.y = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.y_origin = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.x = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.x_e = ti.Vector.field(n=3, dtype=float, shape=self.num_edges)
        self.x_f = ti.Vector.field(n=3, dtype=float, shape=self.num_faces)
        self.x_test = ti.Vector.field(n=3, dtype=float, shape=self.num_verts + self.num_edges + self.num_faces)
        self.x0 = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.v = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.dx = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.dv = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.nc = ti.field(dtype=float, shape=self.num_verts)
        self.dup = ti.field(dtype=float, shape=self.num_verts)
        self.m_inv = ti.field(dtype=float, shape=self.num_verts)
        self.m = ti.field(dtype=float, shape=self.num_verts)
        self.fixed = ti.field(dtype=float, shape=self.num_verts)
        self.rho0 = ti.field(dtype=float, shape=self.num_verts)
        self.colors = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.heat_map = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)

        self.cache_size = 50
        self.num_neighbours = ti.field(dtype=int, shape=self.num_verts + self.num_edges + self.num_faces)
        self.neighbour_ids = ti.field(dtype=int, shape=(self.num_verts + self.num_edges + self.num_faces, self.cache_size))

        self.cache_size_rest = 100
        self.num_neighbours_rest = ti.field(dtype=int, shape=self.num_verts + self.num_edges + self.num_faces)
        self.neighbour_ids_rest = ti.field(dtype=int, shape=(self.num_verts + self.num_edges + self.num_faces, self.cache_size_rest))

        self.cache_size_dy = 100
        self.num_neighbours_dy = ti.field(dtype=int, shape=load_array.shape[0])
        self.neighbour_ids_dy = ti.field(dtype=int, shape=(load_array.shape[0], self.cache_size_dy))

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
        self.m.fill(0.0)
        if self.is_static is False:
            self.fixed.fill(1.0)
        else:
            self.fixed.fill(0.0)

        # fields about edges
        self.l0 = ti.field(dtype=float, shape=self.num_edges)

        self.hii = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.num_verts)
        self.hij = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.num_edges)

        self.Ax = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.r = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.r_next = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.b = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.p = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)

        self.eid_field = ti.field(dtype=int, shape=(self.num_edges, 2))
        # initialize the edge fields
        self.l0.fill(0.0)
        self.eid_field.from_numpy(self.e_np)
        self.edge_indices_flatten = ti.field(dtype=int, shape=self.num_edges * 2)
        # # fields about faces
        # self.aabb_min = ti.field(dtype=float, shape=self.num_faces)
        # self.aabb_max = ti.field(dtype=float, shape=self.num_faces)
        # # self.morton_code = ti.field(dtype=ti.uint32, shape=self.num_faces)
        self.fid_field = ti.field(dtype=int, shape=(self.num_faces, 3))
        self.face_indices_flatten = ti.field(dtype=ti.int32, shape=self.num_faces * 3)
        #
        # # initialize the face fields
        self.fid_field.from_numpy(self.f_np)
        # self.face_indices_flatten.from_numpy(self.f_np)
        self.init_edge_indices_flatten()
        self.init_face_indices_flatten()
        self.init_l0_m_inv()
        self.init_color()
        self.bending_indices = ti.field(dtype=ti.i32)
        self.bending_constraint_count = 0
        self.bending_l0 = ti.field(dtype=float)
        # self.num_bending = self.bending_l0.shape
        self.initBendingIndices()

    def initBendingIndices(self):
        # https://carmencincotti.com/2022-09-05/the-most-performant-bending-constraint-of-xpbd/
        self.bending_constraint_count, neighbor_set = self.findTriNeighbors()
        bending_indices_np = self.getBendingPair(self.bending_constraint_count, neighbor_set)
        # print(bending_indices_np)
        # print("# bending: ", self.bending_constraint_count)
        ti.root.dense(ti.i, bending_indices_np.shape[0] * 2).place(self.bending_indices)
        ti.root.dense(ti.i, bending_indices_np.shape[0]).place(self.bending_l0)

        for i in range(bending_indices_np.shape[0]):
            self.bending_indices[2 * i] = bending_indices_np[i][0]
            self.bending_indices[2 * i + 1] = bending_indices_np[i][1]

        self.init_bengding_l0()

    def findTriNeighbors(self):
        # print("initBend")
        # print(self.fid_np)
        # print(self.fid_np.shape)
        num_f = np.rint(self.f_np.shape[0]).astype(int)
        edgeTable = np.zeros((num_f * 3, self.f_np.shape[1]), dtype=int)
        # print(edgeTable.shape)

        for f in range(self.f_np.shape[0]):
            eT = np.zeros((3, 3))
            v1, v2, v3 = np.sort(self.f_np[f])
            e1, e2, e3 = 3 * f, 3 * f + 1, 3 * f + 2
            eT[0, :] = [v1, v2, e1]
            eT[1, :] = [v1, v3, e2]
            eT[2, :] = [v2, v3, e3]

            edgeTable[3 * f:3 * f + 3, :] = eT

        ind = np.lexsort((edgeTable[:, 1], edgeTable[:, 0]))
        edgeTable = edgeTable[ind]
        # print(edgeTable)

        neighbors = np.zeros((num_f * 3), dtype=int)
        neighbors.fill(-1)

        ii = 0
        bending_constraint_count = 0
        while (ii < 3 * num_f - 1):
            e0 = edgeTable[ii, :]
            e1 = edgeTable[ii + 1, :]

            if (e0[0] == e1[0] and e0[1] == e1[1]):
                neighbors[e0[2]] = e1[2]
                neighbors[e1[2]] = e0[2]

                bending_constraint_count = bending_constraint_count + 1
                ii = ii + 2
            else:
                ii = ii + 1

        # print(bending_constraint_count, "asdf!!")
        return bending_constraint_count, neighbors

    def getBendingPair(self, bend_count, neighbors):

        num_f = np.rint(self.f_np.shape[0]).astype(int)
        pairs = np.zeros((bend_count, 2), dtype=int)

        count = 0
        # print(neighbors)

        for f in range(num_f):
            for i in range(3):
                eid = 3 * f + i
                neighbor_edge = neighbors[eid]

                if neighbor_edge >= 0:
                    # print(eid,neighbor_edge,neighbors[neighbor_edge])
                    neighbors[neighbor_edge] = -1
                    # find not shared vertex in common edge of adjacent triangles
                    v = np.sort(self.f_np[f])

                    neighbor_fid = int(np.floor(neighbor_edge / 3.0 + 1e-4) + 1e-4)
                    neighbor_eid_local = neighbor_fid % 3

                    w = np.sort(self.f_np[neighbor_fid])

                    pairs[count, 0] = v[~np.isin(v, w)]
                    pairs[count, 1] = w[~np.isin(w, v)]

                    count = count + 1

        # print("검산!!!",count == bend_count)
        return pairs

    @ti.kernel
    def init_bengding_l0(self):
        for bi in ti.ndrange(self.bending_constraint_count):
            v0, v1 = self.bending_indices[2 * bi], self.bending_indices[2 * bi + 1]
            self.bending_l0[bi] = (self.x[v0] - self.x[v1]).norm()

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
        for i in range(len(self.vert_offsets)):
            r = float(random.randrange(0, 255) / 256)
            g = float(random.randrange(0, 255) / 256)
            b = float(random.randrange(0, 255) / 256)

            size = 0
            if i < len(self.vert_offsets) - 1:
                size = self.vert_offsets[i + 1] - self.vert_offsets[i]
            else:
                size = self.num_verts - self.vert_offsets[i]
            self.init_color_kernel(offset=self.vert_offsets[i], size=size, color=ti.math.vec3(r,g,b))

    @ti.kernel
    def init_color_kernel(self, offset: ti.i32, size: ti.i32, color: ti.math.vec3):
        for i in range(size):
            self.colors[i+offset] = color

    @ti.kernel
    def init_l0_m_inv(self):
        for i in range(self.num_edges):
            v0, v1 = self.eid_field[i,0], self.eid_field[i,1]
            self.l0[i] = (self.x[v0] - self.x[v1]).norm()
            self.m[v0] += 0.5 * self.l0[i]
            self.m[v1] += 0.5 * self.l0[i]

        for i in range(self.num_verts):
            self.m_inv[i] = 1.0 /  self.m[i]