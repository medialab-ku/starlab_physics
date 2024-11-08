import taichi as ti
import numpy as np
import os
import sys
import time
import meshio
import random
from pathlib import Path
from pyquaternion import Quaternion
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt

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

        path = 'C:/Users/mhkee/Desktop/workspace/starlab_physics/models/sampling/'+model_name+'.npy'
        # load_array = np.load(path, allow_pickle=True)

        # print(load_array)
        # self.sample_indices = ti.field(dtype=float, shape=load_array.shape)
        # self.x_sample = ti.Vector.field(n=3, dtype=float, shape=load_array.shape[0])
        # self.rho0_sample = ti.field(dtype=float, shape=load_array.shape[0])
        # self.heat_map_sample = ti.Vector.field(n=3, dtype=float, shape=load_array.shape[0])
        #
        # if is_static is False:
        #     self.sample_indices.from_numpy(load_array)

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

        ################################################################################################################
        # Euler Path

        self.vertex_duplicates_count_np = np.zeros(self.num_verts, dtype=int)
        self.euler_path_np = np.empty(0, dtype=int)
        self.vertex_duplicates_offsets = [0, ]
        self.euler_path_offsets = [0, ]

        partition = []
        partition_offset = []
        partition_offset_vert = []

        # Euler path is only available on dynamic meshes!
        if not is_static:
            # The Euler path process is operated per mesh...
            for i in range(self.num_model):
                euler_dir = model_dir[:-len("models/OBJ")] + "euler_graph"
                precomputed_graph_file = euler_dir + "/" + model_name_list[i][:-len(".obj")] + ".edgelist"

                if not os.path.exists(euler_dir):
                    print("The ""euler_graph"" dictionary does not exist. "
                          "The dictionary will be made and then located in your path...")
                    os.mkdir(dir)

                # If the euler graph file does not exist, we should construct and export it!
                if not os.path.isfile(precomputed_graph_file):
                    print(f"The original color graph of <{model_name_list[i]}> does not exist.")
                    print("Constructing an Euler graph...")
                    start = time.time()

                    euler_graph = nx.MultiGraph()
                    for j in range(self.edge_offsets[i], self.edge_offsets[i + 1]):
                        # to ensure independency of the graph, we have to subtract vertex id offset from concatenated mesh!
                        euler_graph.add_edge(self.e_np[j][0] - self.vert_offsets[i],
                                             self.e_np[j][1] - self.vert_offsets[i])
                    euler_graph = nx.eulerize(euler_graph) # after adding edges, we eulerize the graph!

                    # Check whether the constructed graph is eulearian or not...
                    # and then make euler path and duplicates count!
                    if nx.is_eulerian(euler_graph):
                        print("Euler graph construction is completed!")
                        euler_path_list = []
                        euler_path = list(nx.eulerian_path(euler_graph))

                        # For example; euler graph : [[0,1],[1,3],[3,1],[1,2],[2,4]] -> euler path : [0, 1, 3, 1, 2, 4]
                        # Unlike above codes, we should add vertex id offset to the vid of euler path!
                        euler_path_list.append(int(euler_path[0][0]) + self.vert_offsets[i])
                        euler_path_list.append(int(euler_path[0][1]) + self.vert_offsets[i])
                        for j in range(1, len(euler_path)):
                            euler_path_list.append(int(euler_path[j][1]) + self.vert_offsets[i])

                        euler_path_len = len(euler_path_list)
                        self.euler_path_np = np.append(self.euler_path_np, np.array(euler_path_list), axis=0)

                        # Record how many times euler path have visited each vertex... (duplicates count)
                        for j in range(euler_path_len):
                            vid = euler_path_list[j]
                            self.vertex_duplicates_count_np[vid] += 1

                        # To divide the euler path and duplicates offsets from other meshes...
                        self.euler_path_offsets.append(self.euler_path_offsets[-1] + euler_path_len)
                        self.vertex_duplicates_offsets.append(self.vert_offsets[i + 1])

                    end = time.time()
                    print("Euler Graph Elapsed Time :", round(end - start, 5), "sec.")

                    # Export the constructed graph to the given directory!
                    print("Exporting the constructed Euler graph...")
                    nx.write_edgelist(euler_graph, precomputed_graph_file)
                    print("Export is completed!\n")
                    print(euler_graph.edges())

                # If we already have the existing euler graph, we can just import it!
                else:
                    print("Importing a precomputed Euler graph...")
                    euler_graph = nx.read_edgelist(precomputed_graph_file, create_using=nx.MultiGraph)
                    print("Checking integrity... ", end='')

                    if nx.is_eulerian(euler_graph):
                        print("The imported graph is Eulerian!\n")
                        euler_path_list = []
                        euler_path = list(nx.eulerian_path(euler_graph))

                        # For example; euler graph : [[0,1],[1,3],[3,1],[1,2],[2,4]] -> euler path : [0, 1, 3, 1, 2, 4]
                        # Unlike above codes, we should add vertex id offset to the vid of euler path!
                        euler_path_list.append(int(euler_path[0][0]) + self.vert_offsets[i])
                        euler_path_list.append(int(euler_path[0][1]) + self.vert_offsets[i])
                        for j in range(1, len(euler_path)):
                            euler_path_list.append(int(euler_path[j][1]) + self.vert_offsets[i])

                        euler_path_len = len(euler_path_list)
                        self.euler_path_np = np.append(self.euler_path_np, np.array(euler_path_list), axis=0)

                        # Record how many times the euler path has visited each vertex... (duplicates count)
                        for j in range(euler_path_len):
                            vid = euler_path_list[j]
                            self.vertex_duplicates_count_np[vid] += 1

                        # To divide the euler path and duplicates offsets from other meshes...
                        self.euler_path_offsets.append(self.euler_path_offsets[-1] + euler_path_len)
                        self.vertex_duplicates_offsets.append(self.vert_offsets[i + 1])

                    else:
                        print("Error : The imported graph is not Eulerian.")
                        print("Please remove the graph file and try again. It will be made again.")
                        print("Simulation ended!\n")
                        sys.exit()

                # To ensure stability of physics code, we split the euler path into several paths per duplicate edge
                # whenever we meet a duplicated edge in traversal, we should split the edge
                euler_path_edge_duplicates_count = {}
                euler_path_lb, euler_path_ub = self.euler_path_offsets[i], self.euler_path_offsets[i + 1]

                # Count how many times euler path edge have been visited
                # to prevent us from confusing edges, we unify vertex order of edges as ascending order!
                last_edge = (int(self.euler_path_np[euler_path_lb]), int(self.euler_path_np[euler_path_lb + 1]))
                if last_edge[0] > last_edge[1]:
                    last_edge[0], last_edge[1] = last_edge[1], last_edge[0]
                euler_path_edge_duplicates_count[last_edge] = 1

                for j in range(euler_path_lb + 1, euler_path_ub - 1):
                    v0, v1 = int(self.euler_path_np[j]), int(self.euler_path_np[j + 1])
                    if v0 > v1:
                        v0, v1 = v1, v0

                    if (v0, v1) in euler_path_edge_duplicates_count:
                        euler_path_edge_duplicates_count[(v0, v1)] += 1
                    else:
                        euler_path_edge_duplicates_count[(v0, v1)] = 1

                    last_edge = (v0, v1)

                # the list that contains edges that have been visited more than two times
                duplicate_edges_list = [k for k, v in euler_path_edge_duplicates_count.items() if v > 1]

                # the dictionary that informs the status whether a duplicate edge is visited or not
                # 0 : the edge is not visited yet / 1 : the edge is used
                check_used_duplicate_edges = {k: 0 for k in duplicate_edges_list}

                subpartition = [[]] # the partition of each mesh
                pid = 0 # partition id

                for j in range(euler_path_lb, euler_path_ub - 1):
                    v0, v1 = int(self.euler_path_np[j]), int(self.euler_path_np[j + 1])
                    if v0 > v1:
                        v0, v1 = v1, v0

                    if (v0, v1) in duplicate_edges_list:
                        # the duplicate edge is already used + the current partition have edge
                        if check_used_duplicate_edges[(v0, v1)] == 1 and len(subpartition[pid]) > 0:
                            subpartition.append([])
                            pid += 1
                        # the duplicate edge is already used + the current partition is not used yet
                        elif check_used_duplicate_edges[(v0, v1)] == 1 and len(subpartition[pid]) == 0:
                            continue
                        # the duplicate edge is not used yet
                        else:
                            subpartition[pid].append(int(v0))
                            subpartition[pid].append(int(v1))
                            check_used_duplicate_edges[(v0, v1)] = 1

                    else:
                        subpartition[pid].append(int(v0))
                        subpartition[pid].append(int(v1))

                # After all blocks are added to the partition of mesh, this subpartition is also added to the partition!
                partition.append(subpartition)

                # Split Algorithm
                # to prevent a block of partition that have excessively long length, we should split the partition!
                block_length_per_subpartition = [len(p) for p in subpartition]
                split_threshold = int(np.median(block_length_per_subpartition)) // 2 * 2 # the nearest small even number from the average
                print("Split threshold :", split_threshold)

                subpartition_split = []
                for block in subpartition:
                    if len(block) > split_threshold:
                        for j in range(0, len(block), split_threshold):
                            subpartition_split.append(block[i : i + split_threshold])
                    else:
                        subpartition_split.append(block)

                subpartition_split_length = [len(p) for p in subpartition_split]
                print("all length (before splitting) :", sum(block_length_per_subpartition))
                print("all length (after splitting) :", sum(subpartition_split_length))

                # Print plots
                plt.hist(block_length_per_subpartition, bins=max(block_length_per_subpartition))
                plt.show()
                plt.hist(subpartition_split_length, bins=max(subpartition_split_length))
                plt.show()

                print("the number of partition (before splitting):", len(partition))
                print("the number of partition (after splitting):", len(subpartition_split))
                print("partition (before splitting) :", partition)
                print("partition (after splitting) :", subpartition_split)

                # 여기서부터 시작하기
                subpartition_offset = [0] # partition [[1,2,2,3,3,4], [5,6,6,7]] -> [0, 6, 10]
                subpartition_vert_offset = [0] # partition [[1,2,2,3,3,4], [5,6,6,7]] -> [[1,2,3,4],[5,6,7]] -> [0,4,7]
                subpartition_flattened = []
                of = 0
                vert_of = 0

                for block in subpartition_split:
                    of += len(block)
                    vert_of += (len(block) // 2) + 1
                    subpartition_offset.append(of)
                    subpartition_vert_offset.append(vert_of)

                for block in subpartition_split:
                    for j in range(len(block)):
                        subpartition_flattened.append(block[j])

                subpartition_flattened_np = np.array(subpartition_flattened, dtype=int)
                subpartition_offset_np = np.array(subpartition_offset, dtype=int)
                subpartition_offset_vert_np = np.array(subpartition_vert_offset, dtype=int)

                dup_to_or = np.zeros(subpartition_vert_offset[-1], dtype=int)

                eid_dup = []
                for pi in range(len(subpartition_split)):
                    off_vert = subpartition_vert_offset[pi]
                    size = len(subpartition_split[pi]) // 2

                    for j in range(size):
                        vi = subpartition_split[pi][2 * j]
                        eid_dup.append(j + off_vert)
                        eid_dup.append(j + off_vert + 1)
                        dup_to_or[j + off_vert] = vi
                        # id += 1

                    vi = subpartition_split[pi][2 * (size - 1) + 1]
                    dup_to_or[off_vert + size] = vi
                    # id += 1

                # print("eid-dup : ", end="")
                # for i in range(len(eid_dup) - 1):
                #     print(int(eid_dup[i]), end=" ")
                # print()
                # print("the number of eid-dup :", len(eid_dup))

                # print("dup-to-origin : ", end="")
                # for i in dup_to_or:
                #     print(int(i), end=" ")
                # print()
                # print("the number of dup-to-origin :", len(dup_to_or))

                colors_np = np.zeros((offset_vert[-1], 3))

                for i in range(0, offset.shape[0] - 1):

                    r = float(random.randrange(0, 255) / 256)
                    g = float(random.randrange(0, 255) / 256)
                    b = float(random.randrange(0, 255) / 256)

                    off = offset_vert[i]
                    # print(off)
                    size = (offset_vert[i + 1] - offset_vert[i])
                    # print(size)

                    for j in range(size):
                        colors_np[off + j] = np.array([r, g, b])

                    # print("______")

                print(offset_vert)
                eid_dup = np.array(eid_dup)
                print("eid_dup // 2 =", eid_dup.shape[0] // 2)

        #data structures for partitioned euler path
        self.partition_offset =  ti.field(dtype=int, shape=(offset.shape[0]))
        self.eid_test = ti.field(dtype=int, shape=partition_flattened.shape[0])
        self.partition_offset.from_numpy(offset)
        self.eid_test.from_numpy(partition_flattened)


        self.vert_offset =  ti.field(dtype=int, shape=(offset_vert.shape[0]))
        self.eid_dup = ti.field(dtype=int, shape=eid_dup.shape[0])
        self.color_test = ti.Vector.field(n=3, dtype=float, shape=offset_vert[-1])

        self.vert_offset.from_numpy(offset_vert)

        # print(self.vert_offset)

        self.dup_to_ori = ti.field(dtype=int, shape=offset_vert[-1])
        self.dup_to_ori.from_numpy(dup_to_or)
        self.eid_dup.from_numpy(eid_dup)
        # print(self.dup_to_ori)
        # print(self.eid_test)
        # print(self.partition_offset)

        self.x_dup = ti.Vector.field(n=3, dtype=float, shape=offset_vert[-1])
        self.v_dup = ti.Vector.field(n=3, dtype=float, shape=offset_vert[-1])
        self.dx_dup = ti.Vector.field(n=3, dtype=float, shape=offset_vert[-1])
        self.a_dup = ti.field(dtype=float, shape=offset_vert[-1])
        self.b_dup = ti.field(dtype=float, shape=offset_vert[-1])
        self.c_dup = ti.field(dtype=float, shape=offset_vert[-1])
        self.c_dup_tilde = ti.field(dtype=float, shape=offset_vert[-1])
        # print(self.c_tilde.shape)
        self.d_dup = ti.Vector.field(n=3, dtype=float, shape=offset_vert[-1])
        self.d_dup = ti.Vector.field(n=3, dtype=float, shape=offset_vert[-1])
        self.d_dup_tilde = ti.Vector.field(n=3, dtype=float, shape=offset_vert[-1])
        # print(self.x_dup.shape)
        self.color_test.from_numpy(colors_np)
        # print(self.eid_test)
        # fields about vertices
        self.y = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.y_tilde = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.y_origin = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.x = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.num_dup = ti.field(dtype=float, shape=self.num_verts)
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
        # self.num_neighbours_dy = ti.field(dtype=int, shape=load_array.shape[0])
        # self.neighbour_ids_dy = ti.field(dtype=int, shape=(load_array.shape[0], self.cache_size_dy))

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
        self.hi = ti.field(dtype=float, shape=self.num_verts)
        self.hij = ti.Matrix.field(n=3, m=3, dtype=float, shape=self.num_edges)

        self.Ax = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.r = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.r_next = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.z = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.z_next = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.b = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)
        self.p = ti.Vector.field(n=3, dtype=float, shape=self.num_verts)


        self.eid_field = ti.field(dtype=int, shape=(self.num_edges, 2))
        # initialize the edge fields
        self.l0.fill(0.0)
        self.eid_field.from_numpy(self.e_np)
        self.edge_indices_flatten = ti.field(dtype=int, shape=self.num_edges * 2)
        self.edge_colores = ti.Vector.field(n=3, dtype=float, shape=self.num_edges)
        self.fid_field = ti.field(dtype=int, shape=(self.num_faces, 3))
        self.face_indices_flatten = ti.field(dtype=ti.int32, shape=self.num_faces * 3)
        #
        # # initialize the face fields

        self.fid_field.from_numpy(self.f_np)
        # self.face_indices_flatten.from_numpy(self.f_np)
        # self.init_edge_indices_flatten()
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
        self.v_dup.fill(0.0)
        self.dx.fill(0.0)
        self.dv.fill(0.0)
        self.nc.fill(0.0)

        self.init_num_dup()

        print(self.num_dup)

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
    def init_num_dup(self):

        self.num_dup.fill(0.0)
        for di in self.x_dup:
            vi = self.dup_to_ori[di]
            self.num_dup[vi] += 1.0

    @ti.kernel
    def init_l0_m_inv(self):
        for i in range(self.num_edges):
            v0_d, v1_d = self.eid_dup[2 * i + 0], self.eid_dup[2 * i + 1]
            v0, v1 = self.dup_to_ori[v0_d], self.dup_to_ori[v1_d]
            self.l0[i] = (self.x[v0] - self.x[v1]).norm()
            self.m[v0] += 0.5 * self.l0[i]
            self.m[v1] += 0.5 * self.l0[i]

        for i in range(self.num_verts):
            self.m_inv[i] = 1.0 /  self.m[i]

        # for i in range(self.euler_edge_len):
        #     v0, v1 = self.euler_path_field[i],  self.euler_path_field[i + 1]
        #     self.l0_euler[i] = (self.x[v0] - self.x[v1]).norm()
        #
        # for i in range(self.euler_path_len):
        #     v0 = self.euler_path_field[i]
        #     self.m_inv_euler[i] = self.m_inv[v0]
