import numpy as np

adj_matrix = np.array([
    [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 0, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 0, 1, 1, 1, 0, 1, 1, 0],
    [1, 1, 1, 0, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 0, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
    [0, 0, 1, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
])

def colorEdgesPhantom():
    # Recursive method
    # n = adj_matrix.shape[0]
    # R = set()           # current clique set
    # P = set(range(n))   # candidate set
    # X = set()           # exclusion set

    # BronKerbosch(R, P, X, adj_matrix)

    # Non-recursive method
    cliques = BronKerboschIterative()
    print("cliques:", cliques)

def BronKerbosch(R, P, X, adj_matrix):
    if not any([P, X]):
        print(f"Maximal Clique: {R}")
        return

    for v in list(P):
        newR = R.union([v])
        newP = P.intersection(np.nonzero(adj_matrix[v])[0])
        newX = X.intersection(np.nonzero(adj_matrix[v])[0])

        BronKerbosch(newR, newP, newX, adj_matrix)

        P.remove(v)
        X.add(v)

def BronKerboschIterative():
    n = adj_matrix.shape[0] # num_edges
    all_nodes = set(range(n)) # (0 ~ n-1) edges of the mesh
    stack = [(set(), all_nodes, set())] # init state of the stack (R, P, X)

    maximal_cliques = []
    while stack:
        R, P, X = stack.pop()

        if not P and not X:
            if len(R) >= 3:
                maximal_cliques.append(R)
            continue

        # select pivot node to reduce candidate nodes
        pivot = next(iter(P.union(X)))
        pivot_neighbors = set(np.nonzero(adj_matrix[pivot])[0])

        for v in P - pivot_neighbors:
            v_neighbors = set(np.nonzero(adj_matrix[v])[0])
            stack.append((R | {v}, P & v_neighbors, X & v_neighbors))
            P = P - {v}
            X = X | {v}

    return maximal_cliques

colorEdgesPhantom()