import taichi as ti
import numpy as np
import os

@ti.data_oriented
class GraphColoring:
    def __init__(self, mesh_name, num_verts, num_edges, edges, eid_np, coloring_mode):
        self.mesh_name = mesh_name
        self.num_verts = num_verts
        self.num_edges = num_edges
        self.edges = edges
        self.eid_np = eid_np
        self.coloring_mode = coloring_mode # True : Phantom coloring / False : Greedy coloring

        self.eid_field = ti.field(dtype=ti.int32, shape=self.eid_np.shape)
        self.eid_field.from_numpy(self.eid_np)

        self.edge_indices_for_color = ti.field(dtype=ti.int32, shape=self.num_edges)
        self.edges_color = ti.field(dtype=ti.int32, shape=self.num_edges)
        self.edges_color.fill(-1)
        self.adj_edges_list = ti.field(dtype=ti.int32, shape=(self.num_edges, self.num_edges))
        self.adj_edges_list.fill(0)
        self.max_num_colors = 100
        self.available_colors = ti.field(dtype=ti.int32, shape=self.max_num_colors)  # temporary

        # these data will be ultimately used in XPBD.py
        self.sorted_edges_index = self.edge_indices_for_color
        self.sorted_edges_index_np = self.sorted_edges_index.to_numpy()
        self.color_prefix_sum = ti.field(dtype=ti.i32, shape=self.max_num_colors)
        self.color_prefix_sum.fill(0)
        self.color_prefix_sum_np = self.color_prefix_sum.to_numpy()

        self.color_prefix_sum_np_name = "./precomputed/" + self.mesh_name + "_prefix_sum.npy"
        self.sorted_edges_index_np_name = "./precomputed/" + self.mesh_name + "_sorted_edges_index.npy"

        # if the existing data is in the direction, just import the data to reduce its initialization time
        if os.path.isfile(self.color_prefix_sum_np_name) and os.path.isfile(self.sorted_edges_index_np_name):
            print("-------------------------------------------------------------")
            print("Importing the existing color list...")
            self.color_prefix_sum_np = np.load(self.color_prefix_sum_np_name)
            self.sorted_edges_index_np = np.load(self.sorted_edges_index_np_name)
            self.sorted_edges_index.from_numpy(self.sorted_edges_index_np)

        # otherwise, the initialization time should be consumed
        else:
            print("-------------------------------------------------------------")
            print("verts :", self.num_verts, "edges :", self.num_edges)
            self.initEdgeIndicesForColor()  # ensure the sequence of edge indices
            self.edge_indices_for_color_np = self.edge_indices_for_color.to_numpy()
            print(self.edge_indices_for_color_np)

            self.initAdjEdges() # construct graph that its nodes consist of edges of the mesh

            # before color
            self.edges_color_np = self.edges_color.to_numpy()
            self.adj_edges_list_np = self.adj_edges_list.to_numpy()
            self.available_colors_np = self.available_colors.to_numpy()

            self.colorEdgesGreedy() # coloring edges

            # after color
            self.edges_color.from_numpy(self.edges_color_np)
            self.adj_edges_list.from_numpy(self.adj_edges_list_np)
            self.available_colors.from_numpy(self.available_colors_np)

            # self.printEdgesColor()
            self.checkAdjColor() # checking integrity of the graph

            self.sorted_edges_index = self.edge_indices_for_color
            self.sorted_edges_color = self.edges_color
            self.color_prefix_sum = ti.field(dtype=ti.i32, shape=self.max_num_colors)
            self.color_prefix_sum.fill(0)

            # before sort
            self.sorted_edges_index_np = self.sorted_edges_index.to_numpy()
            self.sorted_edges_color_np = self.sorted_edges_color.to_numpy()
            self.color_prefix_sum_np = self.color_prefix_sum.to_numpy()

            self.colorCountingSort()  # sort

            # after sort
            self.sorted_edges_index.from_numpy(self.sorted_edges_index_np)
            self.sorted_edges_color.from_numpy(self.sorted_edges_color_np)
            self.color_prefix_sum.from_numpy(self.color_prefix_sum_np)

            self.insertPhantom() # insert phantom constraints and phantom particles in the edge graph
            self.exportColorResult() # export coloring result

    @ti.kernel
    def initEdgeIndicesForColor(self):
        # i = 0
        # for e in self.edges:
        #     self.edge_indices_for_color[i] = e.id
        #     i += 1

    @ti.kernel
    def initAdjEdges(self):
        print("Initializing the adjacency list...", end=" ")
        for i in range(self.num_edges):
            for j in range(self.num_edges):
                e1 = self.edge_indices_for_color[i]
                e2 = self.edge_indices_for_color[j]

                if e1 != e2:
                    v1 = self.eid_field[e1, 0]
                    v2 = self.eid_field[e1, 1]
                    v3 = self.eid_field[e2, 0]
                    v4 = self.eid_field[e2, 1]

                    if v1 == v3 or v1 == v4 or v2 == v3 or v2 == v4:
                        self.adj_edges_list[e1, e2] = 1
                        # print(e1, e2, "adjacent")
        print("Done.")

    def colorEdgesGreedy(self):
        print("Coloring edges...", end=" ")
        for i in range(self.num_edges):
            self.available_colors_np.fill(1)
            e1 = self.edge_indices_for_color_np[i]

            # determine forbidden colors to e1
            for j in range(self.num_edges):
                e2 = self.edge_indices_for_color_np[j]
                if self.adj_edges_list_np[e1, e2] == 1 and self.edges_color_np[e2] != -1:
                    self.available_colors_np[self.edges_color_np[e2]] = 0

            # find the first available color
            self.edges_color_np[e1] = np.argmax(self.available_colors_np)
        print("Done.")

    @ti.kernel
    def checkAdjColor(self):
        print("Checking Integrity of the adjacency list...", end=" ")
        for i in range(self.num_edges):
            for j in range(self.num_edges):
                e1 = self.edge_indices_for_color[i]
                e2 = self.edge_indices_for_color[j]
                if self.edges_color[e1] == -1 or self.edges_color[e2] == -1:
                    print("one of colors is -1 :", e1, e2, self.edges_color[e1], self.edges_color[e2])
                elif e1 != e2 and self.adj_edges_list[e1, e2] == 1 and self.edges_color[e1] == self.edges_color[e2]:
                    print("both colors are equivalent :", e1, e2, self.edges_color[e1], self.edges_color[e2])
        print("Done.")

    @ti.kernel
    def printEdgesColor(self):
        for e in self.edges:
            print(e.id, self.edges_color[e.id], end="\t")
        print()

    def colorCountingSort(self):
        print("Sorting colors...", end=" ")
        sorted_edges_index_temp = np.zeros_like(self.sorted_edges_index_np)
        sorted_edges_color_temp = np.zeros_like(self.sorted_edges_color_np)

        # count the number of times each color occurs in the input
        for c in self.edges_color_np:
            self.color_prefix_sum_np[c] += 1

        # modify the counting array to give the number of values smaller than index
        for i in range(1, self.max_num_colors):
            self.color_prefix_sum_np[i] += self.color_prefix_sum_np[i - 1]

        # transfer numbers from back to forth at locations provided by counting array
        for i in range(self.num_edges - 1, -1, -1):
            idx = self.sorted_edges_index_np[i]
            color = self.sorted_edges_color_np[i]
            self.color_prefix_sum_np[color] -= 1
            sorted_edges_color_temp[self.color_prefix_sum_np[color]] = color
            sorted_edges_index_temp[self.color_prefix_sum_np[color]] = idx

        self.sorted_edges_color_np = np.copy(sorted_edges_color_temp)
        self.sorted_edges_index_np = np.copy(sorted_edges_index_temp)
        # print("sorted_edges_color\n", self.sorted_edges_color_np)
        # print("color_prefix_sum\n", self.color_prefix_sum_np)
        print("Done.")

    def insertPhantom(self):
        print("\nPhantom Insertion")
        print("Creating cliques...", end=" ")
        # insert all cliques of the edge graph in the set S
        # using the Bron-Kerbosch algorithm
        cliques = self.BronKerboschIterative()
        print("Done.")

        print("Sorting cliques...", end=" ")
        # ... and sort S in order of decreasing size
        cliques.sort(key=lambda x: len(x), reverse=True)
        # print(cliques)
        # print(len(cliques))
        print("Done.")

        # traverse all cliques and insert phantom particles
        enough_clique_size = len(cliques[0]) // 2 # the good trade-off is choosing w(G)/2 as "q"
        print(enough_clique_size)
        # for clique in cliques:
        #     # execute the iteration until the size of clique is small enough
        #     if len(clique) < enough_clique_size:
        #         break
        #     # find a most shared particle in the clique
        #     for edge in clique:
        #         # How can I find the vertices in each edge???


    def BronKerboschIterative(self):
        n = self.num_edges  # num_edges
        all_nodes = set(range(n))  # (0 ~ n-1) edges of the mesh
        stack = [(set(), all_nodes, set())]  # init state of the stack (R, P, X)

        maximal_cliques = []
        while stack:
            R, P, X = stack.pop()

            if not P and not X:
                if len(R) >= 3:
                    maximal_cliques.append({int(node) for node in R})
                continue

            # select pivot node to reduce candidate nodes in P
            pivot = next(iter(P.union(X)))
            pivot_neighbors = set(np.nonzero(self.adj_edges_list_np[pivot])[0])

            for v in P - pivot_neighbors:
                v_neighbors = set(np.nonzero(self.adj_edges_list_np[v])[0])
                stack.append((R | {v}, P & v_neighbors, X & v_neighbors))
                P = P - {v}
                X = X | {v}

        return maximal_cliques

    def exportColorResult(self):
        print("\nExporting new color data...", end=" ")
        prefix_sum_name = self.mesh_name + "_prefix_sum.npy"
        sorted_edges_index_name = self.mesh_name + "_sorted_edges_index.npy"
        np.save("./precomputed/" + prefix_sum_name, self.color_prefix_sum_np)
        np.save("./precomputed/" + sorted_edges_index_name, self.sorted_edges_index_np)
        print("Done.")