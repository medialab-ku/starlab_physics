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

for i in range(len(edges)):
    for j in range(i + 1, len(edges)):
        if share(edges[i], edges[j]):
            # print(i, j)
            edge_adj_list[i] = np.append(edge_adj_list[i], j)
            edge_adj_list[j] = np.append(edge_adj_list[j], i)

print(edge_adj_list)

n_parts = 2
n_cuts, membership = pymetis.part_graph(n_parts, adjacency=edge_adj_list)
edges_part = []
for i in range(n_parts):
    edges_part.append(np.argwhere(np.array(membership) == i).ravel())


