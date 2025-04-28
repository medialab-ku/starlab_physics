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
    edge_list = []
    for f in range(num_faces):
        face = faces[f]
        edge_list.append(tuple([face[0], face[1]]))
        edge_list.append(tuple([face[1], face[2]]))
        edge_list.append(tuple([face[2], face[0]]))

    edges = np.array(list(edge_list))
    edges_unique, counts= np.unique(edges, axis=0, return_counts=True)
    # edges_unique = edges_unique[counts == 1]
    return edges_unique

def extract_faces(tetras, points):
    num_tets = tetras.shape[0]
    count = {}
    face_list = []
    for t in range(num_tets):
        tet = tetras[t]
        for j in range(4):
            x, y, z = tuple(sorted([tet[j % 4], tet[(j + 1) % 4], tet[(j + 2) % 4]]))
            if (x, y, z) not in count:
                count[(x, y, z)] = 1
            else:
                count[(x, y, z)] += 1

            v12 = points[tet[(j + 1) % 4]] - points[tet[j % 4]]
            v13 = points[tet[(j + 2) % 4]] - points[tet[j % 4]]
            v14 = points[tet[(j + 3) % 4]] - points[tet[j % 4]]
            if np.dot(np.cross(v12, v13), v14) >= 0:
                face_list.append(tuple([tet[j % 4], tet[(j + 1) % 4], tet[(j + 2) % 4]]))
            else:
                face_list.append(tuple([tet[j % 4], tet[(j + 2) % 4], tet[(j + 1) % 4]]))

    faces = np.array(list(face_list))
    unique = []
    for face in faces:
        x, y, z = tuple(sorted(face))
        if count[(x, y, z)] == 1:
            unique.append(face)

    faces_unique = np.array(unique)
    return faces_unique