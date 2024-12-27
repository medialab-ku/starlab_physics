import numpy as np
import pymetis

def share(ei, ej):

    ret = 0
    for i in range(2):
        for j in range(2):
            if ei[i] == ej[j]:
                ret += 1

    return ret

edges = [(0, 1), (1, 2), (2, 3), (0, 2), (0, 3)]

edge_adj_list = [np.array([], dtype=int) for i in range(len(edges))]

for i in range(len(edges) - 1):
    for j in range(i + 1, len(edges)):
        if share(edges[i], edges[j]):
            # print(i, j)
            edge_adj_list[i] = np.append(edge_adj_list[i], j)
            edge_adj_list[j] = np.append(edge_adj_list[j], i)

print("edge_adj_list :", edge_adj_list)

n_parts = 2
n_cuts, membership = pymetis.part_graph(n_parts, adjacency=edge_adj_list)
print(f"n_cuts : {n_cuts} / membership : {membership}")

# membership [1, 0, 0, 1, 1] -> edges_part [[1,2], [0,3,4]]
partition_offsets = [0]
edges_part = []
for i in range(n_parts):
    partition = np.argwhere(np.array(membership) == i).ravel()
    partition_offsets.append(partition_offsets[-1] + len(partition))
    edges_part.append(partition)

print("edges_partition :", edges_part)
print("edges_offset :", partition_offsets)