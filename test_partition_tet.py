import numpy as np
import meshio
import networkx as nx
import pymetis
import taichi as ti


ti.init(arch=ti.cpu, kernel_profiler=True)

model_path ="models/MSH/32770_octocat.msh"
mesh = meshio.read(model_path)
scale_lf = lambda x, sc: sc * x
trans_lf = lambda x, trans: x + trans

x_np_temp = np.array(mesh.points, dtype=float)
num_particles = x_np_temp.shape[0]
center = x_np_temp.sum(axis=0) / x_np_temp.shape[0]

mass = ti.field(shape=num_particles, dtype=float)

tet_indices_np = np.array(mesh.cells[0].data, dtype=int)
num_tetras = tet_indices_np.shape[0]

print(num_tetras)

tet_adj_list = [np.array([], dtype=int) for i in range(num_tetras)]
tet_edge_set = []
for i in range(num_tetras - 1):
    set1 = set(tet_indices_np[i])
    for j in range(i + 1, num_tetras):
        set2 = set(tet_indices_np[j])
        if len(set1 & set2) > 0:
            tet_adj_list[i] = np.append(tet_adj_list[i], j)
            tet_adj_list[j] = np.append(tet_adj_list[j], i)
            tet_edge_set.append(sorted([i, j]))

# print(tet_edge_set)
G = nx.Graph()
G.add_edges_from(tet_edge_set)
colors = nx.greedy_color(G)
unique_elements, counts = np.unique(np.array(list(colors.values())), return_counts=True)
print("# colors-single element: ", len(unique_elements))


num_partition = 2000

print("# partition: ", num_partition)

n_cuts, membership = pymetis.part_graph(num_partition, adjacency=tet_adj_list)

# print(membership)
membership_np = np.array(membership)

tid_partition = []

for i in range(num_partition):
    tid_partition.append(np.where(membership_np == i)[0].tolist())

tet_vertex_set = [set() for i in range(num_partition)]

# print(tet_vertex_set)
for i in range(num_partition):
    for j in tid_partition[i]:
        tet_vertex_set[i] = tet_vertex_set[i] | set(tet_indices_np[j].tolist())

num_vertices_partition_np = [len(i) for i in tet_vertex_set]
max_num_vertices_partition = max(num_vertices_partition_np)
print(max(num_vertices_partition_np), min(num_vertices_partition_np))

partition_edge_set = []
for i in range(num_partition - 1):
    for j in range(i + 1, num_partition):
        if len(tet_vertex_set[i] & tet_vertex_set[j]) > 0:
            partition_edge_set.append(sorted([i, j]))

# print(tet_edge_set)
G_partition = nx.Graph()
G_partition.add_edges_from(partition_edge_set)
colors_partition = nx.greedy_color(G_partition)
unique_elements_partition, counts_partition = np.unique(np.array(list(colors_partition.values())), return_counts=True)

print("# colors-grouped elements: ", len(unique_elements_partition))

tet_indices = ti.field(shape=tet_indices_np.shape, dtype=int)
tet_indices.from_numpy(tet_indices_np)
