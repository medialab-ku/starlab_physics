
import taichi as ti
import meshtaichi_patcher as patcher
import igl
import os
import numpy as np
import random
import framework.utilities.graph as graph_utils
import networkx as nx
import time
import copy
from collections import Counter

from framework.utilities.graph_coloring import GraphColoring

@ti.data_oriented
class MeshTaichiWrapper:
    def __init__(self,
                 model_dir,
                 model_name,
                 offsets,
                 trans=ti.math.vec3(0, 0, 0),
                 rot=ti.math.vec3(0, 0, 0),
                 scale=1.0,
                 is_static=False):

        self.is_static = is_static

        # print(model_dir + "/" + model_name)

        self.mesh = patcher.load_mesh(model_dir + "/" + model_name, relations=["FV", "EV", "FE"])
        self.mesh.verts.place({'fixed': ti.f32,
                               'm_inv': ti.f32,
                               'x0': ti.math.vec3,
                               'x': ti.math.vec3,
                               'y': ti.math.vec3,
                               'v': ti.math.vec3,
                               'dx': ti.math.vec3,
                               'dv': ti.math.vec3,
                               'dup': ti.i32,
                               'nc': ti.f32})
        self.mesh.edges.place({'l0': ti.f32})
        self.mesh.faces.place({'aabb_min': ti.math.vec3,
                               'aabb_max': ti.math.vec3,
                               'morton_code': ti.uint32})

        self.offsets = offsets
        self.mesh.verts.fixed.fill(1.0)
        self.mesh.verts.m_inv.fill(0.0)
        self.mesh.verts.v.fill(0.0)
        self.mesh.verts.x.from_numpy(self.mesh.get_position_as_numpy())
        self.num_verts = len(self.mesh.verts)

        self.faces = self.mesh.faces
        self.verts = self.mesh.verts

        # print(model_path, "# faces:", len(self.mesh.faces))

        # self.setCenterToOrigin()
        self.face_indices = ti.field(dtype=ti.int32, shape=len(self.mesh.faces) * 3)
        self.edge_indices = ti.field(dtype=ti.int32, shape=len(self.mesh.edges) * 2)

        self.face_edge_indices = ti.field(dtype=ti.int32, shape=len(self.mesh.faces) * 3)
        self.init_face_edge_indices()

        self.colors = ti.Vector.field(n=3, dtype=ti.f32, shape=len(self.mesh.verts))
        self.init_color()
        # self.colors.fill(0.1)
        self.verts = self.mesh.verts
        self.faces = self.mesh.faces
        self.edges = self.mesh.edges

        self.num_verts = len(self.mesh.verts)
        self.num_edges = len(self.mesh.edges)
        self.num_faces = len(self.mesh.faces)

        self.trans = trans
        self.rot = rot
        self.scale = scale

        self.applyTransform()
        self.initFaceIndices()
        self.fid_np = self.face_indices.to_numpy()
        self.fid_np = np.reshape(self.fid_np, (len(self.mesh.faces), 3))
        self.initEdgeIndices()
        self.eid_np = self.edge_indices.to_numpy()
        self.eid_np = np.reshape(self.eid_np, (len(self.mesh.edges), 2))
        # print(self.eid_np)

        self.mesh.verts.x0.copy_from(self.mesh.verts.x)
        self.eid_field = ti.field(dtype=ti.int32, shape=self.eid_np.shape)
        self.eid_field.from_numpy(self.eid_np)

        # extract OBJ mesh name
        # self.mesh_name = model_path[len("../models/OBJ/"):]
        self.mesh_name = model_name

        ###############################################################################################################
        # euler graph

        duplicates = np.zeros(self.num_verts, dtype=int)
        path_np = np.array([])
        # print(path_np)
        if is_static is False:
            # if open("./test.abjlist", "r") is False:
            dir = model_dir[:-len("models/OBJ")] + "euler_graph"
            if not os.path.exists(dir):
                print("The ""euler_graph"" dictionary does not exist.")
                print(" It will be made and then located in your path...")
                os.mkdir(dir)

            precomputed_graph = dir + "/" + model_name[:-len(".obj")] + ".edgelist"
            if not os.path.isfile(precomputed_graph):
                print("Constructing an Euler graph...")

                start = time.time()
                graph = graph_utils.construct_graph(self.eid_np)
                graph = nx.eulerize(graph)

                if nx.is_eulerian(graph):
                    print("Euler graph construction success...")

                end = time.time()
                print("Euler Graph Elapsed time:", round(end - start, 5), "sec.")

                # print("Elapsed time:", round(end - start, 5), "sec.")
                print("Exporting the constructed Euler graph...")
                nx.write_edgelist(graph, precomputed_graph)
                print("Export complete...")

            else:
                print("Importing a precomputed Euler graph...")
                graph = nx.read_edgelist(precomputed_graph, create_using=nx.MultiGraph)
                # print(graph)
                print("Checking integrity...")
                if nx.is_eulerian(graph):

                    path = []
                    print("The imported graph is Eulerian!")
                    euler_path = list(nx.eulerian_path(graph))
                    path.append(int(euler_path[0][0]))
                    path.append(int(euler_path[0][1]))
                    for i in range(1, len(euler_path)):
                        path.append(int(euler_path[i][1]))

                    path_len = len(path)
                    path_np = np.array(path)
                    # print(path_np.shape[0])
                    for i in range(path_len):
                        vid = path[i]
                        duplicates[vid] += 1

                    # degree_list = list(graph.degree)
                    # for i in range(self.num_verts):
                    #     vid = int(degree_list[i][0])
                    #     deg = int(degree_list[i][1])



                    #     duplicates[vid] = deg
                    # print(degree_list)

                    # print(duplicates)
                else:
                    print("The imported graph is not Eulerian...")



        # print(path_np[:4])
        path_len = path_np.shape[0]
        l0_len = path_len - 1
        # size0 = 2
        # size1 = 1
        # # if l0_len % 2 == 0:
        # #     size1 = size0 = l0_len // 2
        # # else:
        # #     size0 = l0_len // 2
        # #     size1 = l0_len // 2 + 1
        # #
        # # print(size0, size1, l0_len)
        #
        # print("_____________")
        #
        # for i in range(size0):
        #     print(path_np[2 * i], path_np[2 * i + 1])
        #     # print(2 * i)
        #
        # print("_____________")
        #
        # for i in range(size1):
        #     print(path_np[2 * i + 1], path_np[2 * i + 2])
        #     # print(2 * i + 1)
        #
        if path_len < 1:
            path_len = 1
            l0_len = 1

        print(round(path_len / self.num_edges, 3))
        # print(path_len - 1)
        self.x_euler = ti.Vector.field(n=3, dtype=ti.f32, shape=path_len)
        self.dx_euler = ti.Vector.field(n=3, dtype=ti.f32, shape=path_len)
        self.v_euler = ti.Vector.field(n=3, dtype=ti.f32, shape=path_len)
        self.y_euler = ti.Vector.field(n=3, dtype=ti.f32, shape=path_len)
        self.g_euler = ti.Vector.field(n=3, dtype=ti.f32, shape=path_len)

        self.m_inv_euler = ti.field(dtype=ti.f32, shape=path_len)
        self.fixed_euler = ti.field(dtype=ti.f32, shape=path_len)

        self.a_euler = ti.field(dtype=ti.f32, shape=path_len) # top 1st off-diagonal elements
        self.b_euler = ti.field(dtype=ti.f32, shape=path_len) # diag elements
        self.c_euler = ti.field(dtype=ti.f32, shape=path_len) # bottom 1st off-diagonal elements
        self.c_tilde_euler = ti.field(dtype=ti.f32, shape=path_len) # bottom 1st off-diagonal elements
        self.d_tilde_euler = ti.field(dtype=ti.f32, shape=path_len) # bottom 1st off-diagonal elements

        self.l0_euler = ti.field(dtype=ti.f32, shape=l0_len)
        self.colored_edge_pos_euler = ti.Vector.field(n=3, dtype=ti.f32, shape=l0_len)
        self.colors_edge_euler = ti.Vector.field(n=3, dtype=ti.f32, shape=l0_len)
        self.path_euler = ti.field(dtype=ti.i32, shape=path_len)
        self.edge_indices_euler = ti.field(dtype=ti.i32, shape=2 * l0_len)

        if is_static is False:
            self.path_euler.from_numpy(path_np)
            # print(self.path_euler)
            self.verts.dup.from_numpy(duplicates)
            # print(self.verts.dup)
            self.init_l0_euler()

        ################################################################################################################
        # phantom color graph

        self.max_color = 20
        self.cracking_threshold = 6
        self.phantom_len = self.num_edges
        self.edge_color_field = ti.field(dtype=ti.i32, shape=(self.phantom_len, 3))
        self.edge_color_prefix_sum_field = ti.field(dtype=ti.i32, shape=self.max_color)
        self.edge_color_np = self.edge_color_field.to_numpy()
        self.edge_color_prefix_sum_np = self.edge_color_prefix_sum_field.to_numpy()

        if is_static is False:
            dir = model_dir[:-len("models/OBJ")] + "color_graph"
            if not os.path.exists(dir):
                print("The ""color_graph"" dictionary does not exist.")
                print("It will be made and then located in your path...")
                os.mkdir(dir)

            precomputed_origin_graph_file = dir + "/" + model_name[:-len(".obj")] + "_origin_graph.edgelist"
            precomputed_phantom_graph_file = dir + "/" + model_name[:-len(".obj")] + "_phantom_graph.edgelist"
            precomputed_phantom_origin_numpy_file = dir + "/" + model_name[:-len(".obj")] + "_phantom_origin.npy"

            # If some files are not in the correct directory path, we have to make all the files and export them!
            if not (os.path.isfile(precomputed_origin_graph_file) or
                    os.path.isfile(precomputed_phantom_graph_file) or
                    os.path.isfile(precomputed_phantom_origin_numpy_file)):
                print("Constructing an phantom graph...")
                original_graph = graph_utils.construct_graph(self.eid_np)

                # extract vertex which degree is not lower than the threshold, and sorting those in descending order
                degree_vertices_list = list(original_graph.degree()) # [(vertex, degree), ...]
                sorted_degree_vertices_list = sorted(degree_vertices_list, key=lambda x: x[1], reverse=True)
                filtered_degree_vertices_list = [vertex for vertex, degree in sorted_degree_vertices_list
                                                 if degree >= self.cracking_threshold]

                # deep copy phantom coloring graph from the original graph
                phantom_graph = copy.deepcopy(original_graph)
                phantom_index = -1 # the phantom index starts from -1!
                phantom_original = {} # { phantom vertex index: original vertex index, ... }

                # crack those vertices by editing the graph edge information
                # ...by replacing an original vertex with several phantom vertices!!!
                for vertex in filtered_degree_vertices_list:
                    neighbors = list(phantom_graph.neighbors(vertex))
                    phantom_graph.remove_node(vertex)
                    for neighbor in neighbors:
                        phantom_graph.add_edge(phantom_index, neighbor)
                        phantom_original[phantom_index] = vertex
                        phantom_index -= 1

                # executing edge coloring on the phantom graph
                # print("Executing phantom coloring... It might take a long time...")
                # start = time.time()
                # edge_colors = {}
                # for edge in phantom_graph.edges():
                #     available_colors = set(range(len(phantom_graph.edges())))
                #     for neighbor in list(phantom_graph.edges(edge[0])) + list(phantom_graph.edges(edge[1])):
                #         if neighbor in edge_colors:
                #             available_colors.discard(edge_colors[neighbor])
                #     edge_colors[edge] = min(available_colors)
                # end = time.time()
                # print("Phantom Coloring Elapsed time:", round(end - start, 5), "sec.")

                # executing edge coloring on the original graph
                print("Executing original coloring... It might take a long time...")
                start = time.time()
                edge_colors = {}
                for edge in original_graph.edges():
                    available_colors = set(range(len(original_graph.edges())))
                    for neighbor in list(original_graph.edges(edge[0])) + list(original_graph.edges(edge[1])):
                        if neighbor in edge_colors:
                            available_colors.discard(edge_colors[neighbor])
                    edge_colors[edge] = min(available_colors)
                end = time.time()
                print("Phantom Coloring Elapsed time:", round(end - start, 5), "sec.")
                sorted_edge_colors = dict(sorted(edge_colors.items(), key=lambda x: x[1]))

                print("Original Edge length :", self.phantom_len)
                print("Length of colors :", len(sorted_edge_colors))

                edge_color_temp = []
                for edge, color in sorted_edge_colors.items():
                    v1 = edge[0] if edge[0] >= 0 else phantom_original[edge[0]]
                    v2 = edge[1] if edge[1] >= 0 else phantom_original[edge[1]]
                    c = color
                    edge_color_temp.append([v1, v2, c])
                self.edge_color_np = np.array(edge_color_temp, dtype=int)
                print("\nEdge-Color Field")
                print(self.edge_color_np)

                kind_of_color = self.edge_color_np[:, 2]
                color_values, color_first_index = np.unique(kind_of_color, return_index=True)
                prefix_sum_temp = list(color_first_index)

                self.edge_color_prefix_sum_np = np.full(self.max_color, fill_value=self.phantom_len - 1, dtype=int)
                self.edge_color_prefix_sum_np[:len(prefix_sum_temp)] = prefix_sum_temp
                print(self.edge_color_prefix_sum_np)

                # export the result into the directory
                # print("Exporting the graphs and the numpy file...")
                # nx.write_edgelist(original_graph, precomputed_origin_graph_file)
                # nx.write_edgelist(phantom_graph, precomputed_phantom_graph_file)
                # np.save(precomputed_phantom_origin_numpy_file, phantom_original_numpy)
                # print("Export complete...")

            # # otherwise, we can just import the files from the directory!
            # else:
            #     print("Importing a precomputed phantom coloring graph...")
            #     phantom_graph = nx.read_edgelist(precomputed_phantom_graph_file, create_using=nx.MultiGraph)
            #     phantom_original_numpy = np.load(precomputed_phantom_origin_numpy_file)

        self.edge_color_field.from_numpy(self.edge_color_np)
        self.edge_color_prefix_sum_field.from_numpy(self.edge_color_prefix_sum_np)

        self.l0_phantom = ti.field(dtype=ti.f32, shape=l0_len)
        self.init_l0_phantom()



        ################################################################################################################

        self.bending_indices = ti.field(dtype=ti.i32)
        self.bending_constraint_count = 0
        self.bending_l0 = ti.field(dtype=ti.f32)
        self.initBendingIndices()
        self.render_bending_vert = ti.Vector.field(3, dtype=ti.f32, shape=(len(self.mesh.verts),))
        self.init_render_bending_vert()

    @ti.kernel
    def init_l0_phantom(self):
        for i in range(self.phantom_len):
            v0, v1 = self.edge_color_field[i,0], self.edge_color_field[i,1]
            x01 = self.mesh.verts.x0[v0] - self.mesh.verts.x0[v1]
            self.l0_phantom[i] = x01.norm()

    @ti.kernel
    def init_l0_euler(self):
        # print("shape: ", self.path_euler.shape)
        len = self.path_euler.shape[0]
        for i in range(len - 1):
            v0, v1 = self.path_euler[i], self.path_euler[i + 1]
            x01 = self.mesh.verts.x0[v0] - self.mesh.verts.x0[v1]
            self.l0_euler[i] = x01.norm()
            self.x_euler[i] = self.verts.x0[v0]

        self.x_euler[len - 1] = self.verts.x0[self.path_euler[len - 1]]

        for i in range(len - 1):
            self.edge_indices_euler[2 * i] = i
            self.edge_indices_euler[2 * i + 1] = i + 1

        for i in range(len - 1):
            v0, v1 = self.edge_indices_euler[2 * i + 0], self.edge_indices_euler[2 * i + 1]
            self.colored_edge_pos_euler[i] = 0.5 * (self.x_euler[v0] + self.x_euler[v1])

            if i % 2 == 0:
                self.colors_edge_euler[i] = ti.math.vec3(1.0, 0.0, 0.0)
            else:
                self.colors_edge_euler[i] = ti.math.vec3(0.0, 0.0, 1.0)

    @ti.kernel
    def init_face_edge_indices(self):

        for f in self.mesh.faces:
            for d in range(3):
                self.face_edge_indices[3 * f.id + d] = f.edges[d].id

    @ti.kernel
    def init_render_bending_vert(self):
        for v in self.mesh.verts:
            self.render_bending_vert[v.id] = ti.Vector([v.x[0], v.x[1], v.x[2]])

    def initBendingIndices(self):
        # https://carmencincotti.com/2022-09-05/the-most-performant-bending-constraint-of-xpbd/
        self.bending_constraint_count, neighbor_set = self.findTriNeighbors()
        bending_indices_np = self.getBendingPair(self.bending_constraint_count, neighbor_set)
        # print("# bending: ", self.bending_constraint_count)
        ti.root.dense(ti.i, bending_indices_np.shape[0] * 2).place(self.bending_indices)
        ti.root.dense(ti.i, bending_indices_np.shape[0]).place(self.bending_l0)

        for i in range(bending_indices_np.shape[0]):
            self.bending_indices[2 * i] = bending_indices_np[i][0]
            self.bending_indices[2 * i + 1] = bending_indices_np[i][1]

        self.init_bengding_l0()

    @ti.kernel
    def init_bengding_l0(self):
        for bi in ti.ndrange(self.bending_constraint_count):
            v0,v1 = self.bending_indices[2*bi],self.bending_indices[2*bi+1]
            self.bending_l0[bi] = (self.mesh.verts.x[v0] - self.mesh.verts.x[v1]).norm()

    @ti.kernel
    def bendinIndi(self):
        for v in self.mesh.verts:
            self.render_bending_vert[v.id] = ti.Vector([v.x[0], v.x[1], v.x[2]])

    def findTriNeighbors(self):
        # print("initBend")
        # print(self.fid_np)
        # print(self.fid_np.shape)
        num_f = np.rint(self.fid_np.shape[0]).astype(int)
        edgeTable = np.zeros((num_f * 3, self.fid_np.shape[1]), dtype=int)
        # print(edgeTable.shape)

        for f in range(self.fid_np.shape[0]):
            eT = np.zeros((3, 3))
            v1, v2, v3 = np.sort(self.fid_np[f])
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

        num_f = np.rint(self.fid_np.shape[0]).astype(int)
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
                    v = np.sort(self.fid_np[f])

                    neighbor_fid = int(np.floor(neighbor_edge / 3.0 + 1e-4) + 1e-4)
                    neighbor_eid_local = neighbor_fid % 3

                    w = np.sort(self.fid_np[neighbor_fid])

                    pairs[count, 0] = v[~np.isin(v, w)]
                    pairs[count, 1] = w[~np.isin(w, v)]

                    count = count + 1

        # print("검산!!!",count == bend_count)
        return pairs

###########################
    def reset(self):
        self.mesh.verts.x.copy_from(self.mesh.verts.x0)
        self.mesh.verts.v.fill(0.)
        self.mesh.verts.fixed.fill(0.0)
        self.v_euler.fill(0.0)

        self.init_l0_euler()

    @ti.kernel
    def initFaceIndices(self):
        for f in self.mesh.faces:
            self.face_indices[f.id * 3 + 0] = f.verts[0].id
            self.face_indices[f.id * 3 + 1] = f.verts[1].id
            self.face_indices[f.id * 3 + 2] = f.verts[2].id

    @ti.kernel
    def initEdgeIndices(self):
        for e in self.mesh.edges:
            l0 = (e.verts[0].x - e.verts[1].x).norm()
            e.l0 = l0

            e.verts[0].m_inv += 0.5 * l0
            e.verts[1].m_inv += 0.5 * l0

            self.edge_indices[e.id * 2 + 0] = e.verts[0].id
            self.edge_indices[e.id * 2 + 1] = e.verts[1].id

        for v in self.mesh.verts:
            v.m_inv = 1.0 / v.m_inv

    # @ti.kernel
    # def Hierholzer(self):
    #
    #     edges = []
    #     first = 0
    #     stack = []
    #     stack.append(first)
    #     ret = []
    #     while len(stack) > 0:
    #         v = stack.top()
    #
    #     if !edges[v].size():
    #         ret.append(v)
    #         stack.pop()
    #     else:
    #         i = edges[v][edges[v].size() - 1]
    #         edges[v].pop_back()
    #         std::vector < int >::iterator it = std::find(edges[i].begin(), edges[i].end(), v);
    #         edges[i].erase(it);
    #         stack.push(i)
    #
    #     return ret

    def init_color(self):
        # print(self.offsets)
        for i in range(len(self.offsets)):

            r = random.randrange(0, 255) / 256
            g = random.randrange(0, 255) / 256
            b = random.randrange(0, 255) / 256
            size = 0
            if i < len(self.offsets) - 1:
                size = self.offsets[i + 1] - self.offsets[i]

            else:
                size = len(self.verts) - self.offsets[i]
            #
            self.init_colors(self.offsets[i], size, color=ti.math.vec3(r, g, b))

    @ti.kernel
    def init_colors(self, offset: ti.i32, size: ti.i32, color: ti.math.vec3):

        for i in range(size):
            self.colors[i + offset] = color

    @ti.kernel
    def setCenterToOrigin(self):

        center = ti.math.vec3(0, 0, 0)
        for v in self.verts:
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
            rot_rad_x = ti.math.radians(self.rot[0])
            rot_rad_y = ti.math.radians(self.rot[1])
            rot_rad_z = ti.math.radians(self.rot[2])
            rv = ti.math.rotation3d(rot_rad_x, rot_rad_y, rot_rad_z) @ v_4d
            v.x = ti.Vector([rv[0], rv[1], rv[2]])

        for v in self.mesh.verts:
            v.x += self.trans

    @ti.kernel
    def computeAABB(self, padding: ti.f32) -> (ti.math.vec3, ti.math.vec3):
        aabb_min = ti.math.vec3(1e5)
        aabb_max = ti.math.vec3(-1e5)

        for v in self.mesh.verts:
            temp = v.x
            ti.atomic_max(aabb_max, temp)
            ti.atomic_min(aabb_min, temp)

        aabb_min -= padding * ti.math.vec3(1.0)
        aabb_max += padding * ti.math.vec3(1.0)

        return aabb_min, aabb_max


    @ti.kernel
    def computeAABB_faces(self, padding: ti.f32):

        aabb_min = ti.math.vec3(1e5)
        aabb_max = ti.math.vec3(-1e5)
        ones = ti.math.vec3(1.0)
        for f in self.mesh.faces:
            f.aabb_min = ti.math.min(f.verts[0].x, f.verts[1].x, f.verts[2].x)
            f.aabb_max = ti.math.max(f.verts[0].x, f.verts[1].x, f.verts[2].x)

            f.aabb_min -= padding * ones
            f.aabb_max += padding * ones
        #     ti.atomic_max(aabb_max, f.aabb_max)
        #     ti.atomic_min(aabb_min, f.aabb_min)
        #
        # return aabb_min, aabb_max
    def export(self, scene_name, frame, is_static = False):
        if is_static:
            directory = os.path.join("../results/", scene_name, "StaticMesh_ID_")
        else:
            directory = os.path.join("../results/", scene_name, "Mesh_ID_")

        try:
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