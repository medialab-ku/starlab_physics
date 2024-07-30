import numpy as np

# adj_matrix = np.array([
#     [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
#     [1, 0, 1, 1, 1, 1, 0, 0, 0, 0],
#     [1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
#     [1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
#     [1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
#     [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
#     [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
#     [0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
#     [0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
#     [0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
# ])
adj_martix = np.array([
    [0, 1, 1, 1, 1],
    [1, 0, 1, 1, 1],
    [1, 1, 0, 1, 1],
    [1, 1, 1, 0, 1],
    [0, 1, 1, 1, 0],
])

eid = np.array([
    [0,1],
    [0,2],
    [0,3],
    [0,4],
    [0,5]
])


def insertPhantom():
    print("\nPhantom Insertion")
    # insert all cliques of the edge graph in the set S
    # using the Bron-Kerbosch algorithm : https://en.wikipedia.org/wiki/Bron%E2%80%93Kerbosch_algorithm
    print("Inserting cliques...", end=" ")
    cliques = BronKerboschIterative()
    print("Done.")

    # ... and sort S in order of decreasing size
    print("Sorting cliques...", end=" ")
    cliques.sort(key=lambda x: len(x), reverse=True)
    print("Cliques :", cliques)
    print("The number of cliques :", len(cliques))
    print("Done.")

    # traverse all cliques and insert phantom particles
    enough_clique_size = len(cliques[0]) // 2  # As the paper says, the good trade-off is choosing w(G)/2 as "q"
    print("Enough clique size :", enough_clique_size)
    for clique in cliques:
        # execute the iteration until the size of clique is small enough
        if len(clique) < enough_clique_size:
            break

        # find a most shared particle in the clique
        verts_of_edges = []
        for edge in clique:
            v1 = eid[edge, 0]
            v2 = eid[edge, 1]
            verts_of_edges.append(v1)
            verts_of_edges.append(v2)
            print(v1, v2)

        vertices_of_edges_np = np.array(verts_of_edges)
        unique_verts, counts = np.unique(vertices_of_edges_np, return_counts=True)
        print(unique_verts, counts)
        max_count_vert = unique_verts[np.argmax(counts)]
        print("The vertex that have the most number in this clique :", max_count_vert)


def BronKerboschIterative():
    n = adj_martix.shape[0] # num_edges
    all_nodes = set(range(n)) # (0 ~ n-1) edges of the mesh
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
        pivot_neighbors = set(np.nonzero(adj_martix[pivot])[0])

        for v in P - pivot_neighbors:
            v_neighbors = set(np.nonzero(adj_martix[v])[0])
            stack.append((R | {v}, P & v_neighbors, X & v_neighbors))
            P = P - {v}
            X = X | {v}

    return maximal_cliques

insertPhantom()