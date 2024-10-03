import taichi as ti
import numpy as np
import networkx as nx
import meshio
import random
import os
import sys
import time
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
        self.num_model = len(model_name_list)

        self.num_verts = 0
        self.num_faces = 0
        self.num_edges = 0
        self.x_np = np.empty((0, 3), dtype=float) # the vertices of mesh
        self.f_np = np.empty((0, 3), dtype=int)   # the faces of mesh
        self.e_np = np.empty((0, 2), dtype=int)   # the edges of mesh

        self.vert_offsets = [] # offsets of each mesh
        self.edge_offsets = [] # offsets of each edge
        self.face_offsets = [] # offsets of each face
        self.vert_offsets.append(0)
        self.edge_offsets.append(0)
        self.face_offsets.append(0)

        # concatenate all of meshes
        for i in range(self.num_model):
            model_path = model_dir + "/" + model_name_list[i]
            mesh = meshio.read(model_path)

            scale_lf = lambda x, sc: sc * x
            trans_lf = lambda x, trans: x + trans

            # rotate, scale, and translate all vertices
            x_np_temp = np.array(mesh.points, dtype=float)
            center = x_np_temp.sum(axis=0) / x_np_temp.shape[0] # center position of the mesh
            x_np_temp = np.apply_along_axis(lambda row: trans_lf(row, -center), 1, x_np_temp) # translate to origin

            x_np_temp = scale_lf(x_np_temp, scale_list[i]) # scale mesh to the particular ratio

            # rotate mesh if it is demanded...
            if len(rot_list) > 0:
                rot_quaternion = Quaternion(axis=[rot_list[i][0], rot_list[i][1], rot_list[i][2]], angle=rot_list[i][3])
                rot_matrix = rot_quaternion.rotation_matrix
                for j in range(x_np_temp.shape[0]):
                    x_np_temp[j] = rot_matrix @ x_np_temp[j]

            x_np_temp = np.apply_along_axis(lambda row: trans_lf(row, center), 1, x_np_temp)  # translate back to the original position
            x_np_temp = np.apply_along_axis(lambda row: trans_lf(row, trans_list[i]), 1, x_np_temp) # translate again to the designated position!
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
        # print(self.num_verts, self.num_edges, self.num_faces)
        # print(self.vert_offsets, self.edge_offsets, self.face_offsets)

        ################################################################################################################
        # Graph Coloring (Crumpling Issue!)
        # make information about graph-coloring (only dynamic meshes are available!)

        self.color_max = 20
        self.original_edge_color_prefix_sum_np = np.empty(0, dtype=int)
        self.original_edge_color_np = np.empty((0, 3), dtype=int)

        if not self.is_static:
            print("\n=====================================================================================\n")
            for i in range(self.num_model):
                precomputed_dir = model_dir[:-len("models/OBJ")] + "color_graph"
                original_color_prefix_sum_name = precomputed_dir + "/" + model_name_list[i][:-len(".obj")] + "_original_prefix_sum.npy"
                original_color_graph_name = precomputed_dir + "/" + model_name_list[i][:-len(".obj")] + "_original_graph.npy"
                print(original_color_prefix_sum_name, original_color_graph_name)

                # if there is not the coloring graph directory, we should make that!
                if not os.path.exists(precomputed_dir):
                    print("The coloring graph dictionary does not exist. It will be made and then located in your path...")
                    os.mkdir(precomputed_dir)

                # if the existing data is in the direction, just import the data to reduce its initialization time
                should_make_graph = True
                if os.path.isfile(original_color_prefix_sum_name) and os.path.isfile(original_color_graph_name):
                    edges_color_prefix_sum = np.load(original_color_prefix_sum_name)
                    # if the number of prefix sum differs from the one of precomputed graph, we should make graph...
                    if edges_color_prefix_sum.shape[0] != self.color_max:
                        break
                    edges_color = np.load(original_color_graph_name)

                    self.original_edge_color_prefix_sum_np = np.append(self.original_edge_color_prefix_sum_np,
                                                                       edges_color_prefix_sum + self.edge_offsets[i],
                                                                       axis=0)

                    edges_color[:, 0] = edges_color[:, 0] + self.vert_offsets[i]
                    edges_color[:, 1] = edges_color[:, 1] + self.vert_offsets[i]
                    self.original_edge_color_np = np.append(self.original_edge_color_np, edges_color, axis=0)

                    print(f"The color graph and prefix sum files of <{model_name_list[i]}> exist!")
                    print("Importing these files...\n")
                    should_make_graph = False

                # otherwise, we should make graph in the first place...
                if should_make_graph:
                    print(f"The original color graph of <{model_name_list[i]}> does not exist.")
                    print("It will be made and then located in your path...")
                    print("Constructing an original graph...")
                    original_graph = nx.MultiGraph()
                    for j in range(self.edge_offsets[i], self.edge_offsets[i+1]):
                        original_graph.add_edge(self.e_np[j][0], self.e_np[j][1])

                    print("Executing original coloring... It might take a long time...")
                    start = time.time()
                    original_edge_colors = {}

                    for edge in original_graph.edges():
                        v0 = int(edge[0])
                        v1 = int(edge[1])
                        if v0 > v1:
                            v0, v1 = v1, v0
                        sorted_edge = (v0, v1)
                        available_colors = set(range(len(original_graph.edges())))
                        neighbors = list(set(original_graph.edges(edge[0])) | set(original_graph.edges(edge[1])))

                        for neighbor in neighbors:
                            vn0 = int(neighbor[0])
                            vn1 = int(neighbor[1])
                            if vn0 > vn1:
                                vn0, vn1 = vn1, vn0
                            neighbor_sorted = (vn0, vn1)

                            if neighbor_sorted in original_edge_colors:
                                available_colors.discard(original_edge_colors[neighbor_sorted])

                        original_edge_colors[sorted_edge] = min(available_colors)

                    end = time.time()
                    print("Euler graph construction is completed!")
                    print("Original Coloring Elapsed time:", round(end - start, 5), "sec.")

                    sorted_original_edge_colors = dict(sorted(original_edge_colors.items(), key=lambda x: x[1]))

                    original_edge_color_temp = []
                    for edge, color in sorted_original_edge_colors.items():
                        v0 = int(edge[0])
                        v1 = int(edge[1])
                        c = color
                        original_edge_color_temp.append([v0, v1, c])
                    original_edge_color = np.array(original_edge_color_temp, dtype=int)
                    self.original_edge_color_np = np.append(self.original_edge_color_np, original_edge_color, axis=0)

                    kind_of_color = original_edge_color[:, 2]
                    color_values, color_first_index = np.unique(kind_of_color, return_index=True)
                    original_prefix_sum_list = list(color_first_index)

                    num_model_edges = self.edge_offsets[i+1] - self.edge_offsets[i]
                    original_edge_color_prefix_sum = np.full(self.color_max, fill_value=num_model_edges, dtype=int)
                    original_edge_color_prefix_sum[:len(original_prefix_sum_list)] = original_prefix_sum_list
                    self.original_edge_color_prefix_sum_np = np.append(self.original_edge_color_prefix_sum_np,
                                                                       original_edge_color_prefix_sum + self.edge_offsets[i],
                                                                       axis=0)

                    print(f"Exporting the graph and prefix sum files of <{model_name_list[i]}>...\n")
                    original_edge_color[:, 0] = original_edge_color[:, 0] - self.vert_offsets[i]
                    original_edge_color[:, 1] = original_edge_color[:, 1] - self.vert_offsets[i]
                    np.save(original_color_graph_name, original_edge_color)
                    np.save(original_color_prefix_sum_name, original_edge_color_prefix_sum)

            self.original_edge_color_field = ti.field(dtype=int, shape=(self.num_edges, 3))
            self.original_edge_color_field.from_numpy(self.original_edge_color_np)
            print("=====================================================================================\n")

            # print(self.original_edge_color_prefix_sum_np, self.original_edge_color_prefix_sum_np.shape[0])
            # print(self.original_edge_color_np, self.original_edge_color_np.shape[0])

        ################################################################################################################
        # Euler Path

        self.duplicates_np = np.zeros(self.num_verts, dtype=int)
        self.euler_path_np = np.empty(0, dtype=int)
        self.duplicates_offsets = [0, ]
        self.euler_path_offsets = [0, ]

        if not is_static:
            for i in range(self.num_model):
                euler_dir = model_dir[:-len("models/OBJ")] + "euler_graph"
                precomputed_graph_file = euler_dir + "/" + model_name_list[i][:-len(".obj")] + ".edgelist"

                if not os.path.exists(euler_dir):
                    print("The ""euler_graph"" dictionary does not exist. It will be made and then located in your path...")
                    os.mkdir(dir)

                if not os.path.isfile(precomputed_graph_file):
                    print(f"The original color graph of <{model_name_list[i]}> does not exist.")
                    print("Constructing an Euler graph...")
                    start = time.time()

                    euler_graph = nx.MultiGraph()
                    for j in range(self.edge_offsets[i], self.edge_offsets[i+1]):
                        euler_graph.add_edge(self.e_np[j][0] - self.vert_offsets[i],
                                             self.e_np[j][1] - self.vert_offsets[i])
                    euler_graph = nx.eulerize(euler_graph)

                    if nx.is_eulerian(euler_graph):
                        print("Euler graph construction is completed!")

                        path_list = []
                        euler_path = list(nx.eulerian_path(euler_graph))

                        path_list.append(int(euler_path[0][0]) + self.vert_offsets[i])
                        path_list.append(int(euler_path[0][1]) + self.vert_offsets[i])
                        for j in range(1, len(euler_path)):
                            path_list.append(int(euler_path[j][1]) + self.vert_offsets[i])

                        path_len = len(path_list)
                        self.euler_path_np = np.append(self.euler_path_np, np.array(path_list), axis=0)
                        for j in range(path_len):
                            vid = path_list[j]
                            self.duplicates_np[vid] += 1

                        self.euler_path_offsets.append(self.euler_path_offsets[-1] + path_len)
                        self.duplicates_offsets.append(self.vert_offsets[i+1])

                    end = time.time()
                    print("Euler Graph Elapsed Time :", round(end - start, 5), "sec.")

                    print("Exporting the constructed Euler graph...")
                    nx.write_edgelist(euler_graph, precomputed_graph_file)
                    print("Export is completed!\n")
                    print(euler_graph.edges())

                else:
                    euler_dir = model_dir[:-len("models/OBJ")] + "euler_graph"
                    precomputed_graph_file = euler_dir + "/" + model_name_list[i][:-len(".obj")] + ".edgelist"

                    print("Importing a precomputed Euler graph...")
                    euler_graph = nx.read_edgelist(precomputed_graph_file, create_using=nx.MultiGraph)
                    print("Checking integrity... ", end='')
                    if nx.is_eulerian(euler_graph):
                        print("The imported graph is Eulerian!\n")

                        path_list = []
                        euler_path = list(nx.eulerian_path(euler_graph))

                        path_list.append(int(euler_path[0][0]) + self.vert_offsets[i])
                        path_list.append(int(euler_path[0][1]) + self.vert_offsets[i])
                        for j in range(1, len(euler_path)):
                            path_list.append(int(euler_path[j][1]) + self.vert_offsets[i])

                        path_len = len(path_list)
                        self.euler_path_np = np.append(self.euler_path_np, np.array(path_list), axis=0)
                        for j in range(path_len):
                            vid = path_list[j]
                            self.duplicates_np[vid] += 1

                        self.euler_path_offsets.append(self.euler_path_offsets[-1] + path_len)
                        self.duplicates_offsets.append(self.vert_offsets[i+1])
                    else:
                        print("The imported graph is not eulerian...")
                        print("Simulation ended!\n")
                        sys.exit()


            # print("Euler Path\n", self.euler_path_np)
            # print("Duplicate\n", self.duplicates_np)
            # print("Euler Path Length Offsets :", self.euler_path_offsets)
            # print("Duplicates Length Offsets :", self.duplicates_offsets)

            euler_path_offsets_np = np.array(self.euler_path_offsets)
            duplicates_offsets_np = np.array(self.duplicates_offsets)
            euler_offset_len = euler_path_offsets_np.shape[0]
            self.euler_path_offsets_field = ti.field(dtype=int, shape=euler_offset_len)
            self.duplicates_offsets_field = ti.field(dtype=int, shape=euler_offset_len)
            self.euler_path_offsets_field.from_numpy(euler_path_offsets_np)
            self.duplicates_offsets_field.from_numpy(duplicates_offsets_np)

            self.euler_path_len = self.euler_path_np.shape[0]
            self.euler_edge_len = self.euler_path_len - 1
            self.duplicates_len = self.duplicates_np.shape[0]
            self.euler_path_field = ti.field(dtype=int, shape=self.euler_path_len)
            self.duplicates_field = ti.field(dtype=int, shape=self.duplicates_len)
            self.euler_path_field.from_numpy(self.euler_path_np)
            self.duplicates_field.from_numpy(self.duplicates_np)

            self.euler_path_field_dim2 = ti.field(dtype=int, shape=(self.euler_edge_len, 2))
            for i in range(self.euler_edge_len - 1):
                self.euler_path_field_dim2[i,0] = self.euler_path_field[i]
                self.euler_path_field_dim2[i,1] = self.euler_path_field[i+1]

            # fields about euler
            self.y_euler = ti.Vector.field(n=3, dtype=float, shape=self.euler_path_len)
            self.y_tilde_euler = ti.Vector.field(n=3, dtype=float, shape=self.euler_path_len)
            self.x_euler = ti.Vector.field(n=3, dtype=float, shape=self.euler_path_len)
            self.v_euler = ti.Vector.field(n=3, dtype=float, shape=self.euler_path_len)
            self.dx_euler = ti.Vector.field(n=3, dtype=float, shape=self.euler_path_len)
            self.m_inv_euler = ti.field(dtype=float, shape=self.euler_path_len)
            self.fixed_euler = ti.field(dtype=float, shape=self.euler_path_len)

            self.l0_euler = ti.field(dtype=float, shape=self.euler_edge_len)
            self.colored_edge_pos_euler = ti.Vector.field(n=3, dtype=float, shape=self.euler_edge_len)
            self.edge_color_euler = ti.Vector.field(n=3, dtype=float, shape=self.euler_edge_len)

            print("=====================================================================================\n")

        ################################################################################################################
        # fields about vertices
        self.y = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.y_tilde = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
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
        self.y_tilde.fill(0.0)
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
        self.l0_original_graph = ti.field(dtype=float, shape=self.num_edges)

        # initialize the edge fields
        self.l0.fill(0.0)
        self.l0_original_graph.fill(0.0)
        self.eid_field.from_numpy(self.e_np)
        self.edge_indices_flatten = ti.field(dtype=int, shape=self.num_edges * 2)

        # fields about faces
        self.fid_field = ti.field(dtype=int, shape=(self.num_faces, 3))
        self.face_indices_flatten = ti.field(dtype=ti.int32, shape=self.num_faces * 3)

        # initialize the face fields
        self.fid_field.from_numpy(self.f_np)
        self.init_edge_indices_flatten()
        self.init_face_indices_flatten()
        self.init_color()

        if is_static is False:
            self.init_l0_m_inv()
            self.init_euler()
            self.v_euler.fill(0.0)

    ####################################################################################################################
    def reset(self):
        self.y.fill(0.0)
        self.y_tilde.fill(0.0)
        self.x.from_numpy(self.x_np)
        self.v.fill(0.0)
        self.dx.fill(0.0)
        self.dv.fill(0.0)
        self.nc.fill(0.0)

        if self.is_static is False:
            self.init_euler()
            self.y_euler.fill(0.0)
            self.y_tilde_euler.fill(0.0)
            self.v_euler.fill(0.0)

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
        for i in range(len(self.vert_offsets) - 1):
            r = float(random.randrange(0, 255) / 256)
            g = float(random.randrange(0, 255) / 256)
            b = float(random.randrange(0, 255) / 256)

            size = self.vert_offsets[i+1] - self.vert_offsets[i]
            self.init_color_kernel(offset=self.vert_offsets[i], size=size, color=ti.math.vec3(r, g, b))

    @ti.kernel
    def init_color_kernel(self, offset: ti.i32, size: ti.i32, color: ti.math.vec3):
        for i in range(size):
            self.colors[i+offset] = color

    @ti.kernel
    def init_l0_m_inv(self):
        for i in range(self.num_edges):
            v0, v1 = self.eid_field[i,0], self.eid_field[i,1]
            self.l0[i] = (self.x0[v0] - self.x0[v1]).norm()
            self.m_inv[v0] += 0.5 * self.l0[i]
            self.m_inv[v1] += 0.5 * self.l0[i]

        for i in range(self.num_edges):
            v0, v1 = self.original_edge_color_field[i,0], self.original_edge_color_field[i,1]
            self.l0_original_graph[i] = (self.x0[v0] - self.x0[v1]).norm()

    @ti.kernel
    def init_euler(self):
        for i in range(self.num_model):
            current_offset, next_offset = self.euler_path_offsets_field[i], self.euler_path_offsets_field[i+1]
            for j in range(current_offset, next_offset - 1):
                v0, v1 = self.euler_path_field[j], self.euler_path_field[j+1]
                self.l0_euler[j] = (self.x0[v0] - self.x0[v1]).norm()
                self.x_euler[j] = self.x0[v0]
                self.x_euler[j+1] = self.x0[v1]
                self.m_inv_euler[j] = self.m_inv[v0]
                self.m_inv_euler[j+1] = self.m_inv[v1]
                self.fixed_euler[j] = self.fixed[v0]
                self.fixed_euler[j+1] = self.fixed[v1]

        for i in range(self.num_model):
            current_offset, next_offset = self.euler_path_offsets_field[i], self.euler_path_offsets_field[i+1]
            for j in range(current_offset, next_offset - 1):
                v0, v1 = self.euler_path_field[j], self.euler_path_field[j+1]
                self.colored_edge_pos_euler[j] = 0.5 * (self.x0[v0] + self.x0[v1])

                if j % 2 == 0:
                    self.edge_color_euler[j] = ti.math.vec3(1.0, 0.0, 0.0)
                else:
                    self.edge_color_euler[j] = ti.math.vec3(0.0, 1.0, 0.0)

        # print("x_euler")
        # ti.loop_config(serialize=True)
        # for i in range(self.euler_path_len):
        #     print(f"{i}: {self.x_euler[i]}")
        # print("m_inv_euler")
        # ti.loop_config(serialize=True)
        # for i in range(self.euler_path_len):
        #     print(f"{i}: {self.m_inv_euler[i]}")
        # print("fixed_euler")
        # ti.loop_config(serialize=True)
        # for i in range(self.euler_path_len):
        #     print(f"{i}: {self.fixed_euler[i]}")
        # print("l0_euler")
        # ti.loop_config(serialize=True)
        # for i in range(self.euler_edge_len):
        #     print(i, i+1, self.euler_path_field[i], self.euler_path_field[i+1])
        # ti.loop_config(serialize=True)
        # for i in range(self.euler_edge_len):
        #     print(f"{i}: {self.l0_euler[i]}")