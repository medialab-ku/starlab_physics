import numpy as np
def extract_edges(faces):

    num_faces = faces.shape[0]
    edge_set = set()
    for f in range(num_faces):
        face = faces[f]
        edge_set.add(tuple(sorted([face[0], face[1]])))
        edge_set.add(tuple(sorted([face[1], face[2]])))
        edge_set.add(tuple(sorted([face[2], face[0]])))
    # self.num_edges += len(edges)
    edges = np.array(list(edge_set))
    return edges