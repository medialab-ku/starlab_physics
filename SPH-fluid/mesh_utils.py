import numpy as np

def extract_vers(edges):

    num_edges = edges.shape[0]
    vert_set = set()
    for e in range(num_edges):
        edge = edges[e]
        vert_set.add(tuple(sorted([edge[0], edge[1]])))
        vert_set.add(tuple(sorted([edge[1], edge[2]])))
        vert_set.add(tuple(sorted([edge[2], edge[0]])))
    verts = np.array(list(vert_set))
    return verts

def extract_edges(faces):

    num_faces = faces.shape[0]
    edge_set = set()
    for f in range(num_faces):
        face = faces[f]
        edge_set.add(tuple(sorted([face[0], face[1]])))
        edge_set.add(tuple(sorted([face[1], face[2]])))
        edge_set.add(tuple(sorted([face[2], face[0]])))
    edges = np.array(list(edge_set))
    return edges

def extract_faces(tetras):

    num_tets = tetras.shape[0]
    face_set = set()
    for t in range(num_tets):
        tet = tetras[t]
        face_set.add(tuple(sorted([tet[0], tet[1], tet[2]])))
        face_set.add(tuple(sorted([tet[1], tet[2], tet[3]])))
        face_set.add(tuple(sorted([tet[2], tet[3], tet[0]])))
        face_set.add(tuple(sorted([tet[3], tet[0], tet[1]])))

    # print(face_set)

    faces = np.array(list(face_set))
    return faces