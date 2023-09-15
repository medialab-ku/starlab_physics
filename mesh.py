import taichi as ti
import numpy as np

class TotalMesh:
    def __init__(self):
        self.verts = []
        self.edges = []
        self.faces = []
        self.num_verts = 0
        self.num_edges = 0
        self.num_faces = 0
        self.num_verts_list = []
        self.verts_start = []
        self.verts_end = []
        self.edges_start = []
        self.edges_end = []
        self.faces_start = []
        self.faces_end = []

    def addMesh(self, mesh):
        self.verts.append(mesh.verts)
        self.edges.append(mesh.edges + self.num_verts)
        self.faces.append(mesh.faces + self.num_verts)
        self.num_verts += len(mesh.verts)
        self.num_edges += len(mesh.edges)
        self.num_faces += len(mesh.faces)
        self.num_verts_list.append(len(mesh.verts))
        self.verts_start.append(self.num_verts - len(mesh.verts))
        self.verts_end.append(self.num_verts)
        self.edges_start.append(self.num_edges - len(mesh.edges))
        self.edges_end.append(self.num_edges)
        self.faces_start.append(self.num_faces - len(mesh.faces))
        self.faces_end.append(self.num_faces)

    def getMesh(self, mesh_id):
        verts = self.verts[self.verts_start[mesh_id]:self.verts_end[mesh_id]]
        edges = self.edges[self.edges_start[mesh_id]:self.edges_end[mesh_id]]
        faces = self.faces[self.faces_start[mesh_id]:self.faces_end[mesh_id]]
        return verts, edges, faces

    def getMeshVert(self, mesh_id):
        return self.verts[self.verts_start[mesh_id]:self.verts_end[mesh_id]]

    def getMeshEdge(self, mesh_id):
        return self.edges[self.edges_start[mesh_id]:self.edges_end[mesh_id]]

    def getMeshFace(self, mesh_id):
        return self.faces[self.faces_start[mesh_id]:self.faces_end[mesh_id]]

    def getMeshVertNum(self, mesh_id):
        return self.num_verts_list[mesh_id]
    def getMeshVertNumSum(self, mesh_id):
        return sum(self.num_verts_list[:mesh_id])

    def deleteMesh(self, mesh_id):
        self.verts = self.verts[:self.verts_start[mesh_id]] + self.verts[self.verts_end[mesh_id]:]
        self.edges = self.edges[:self.edges_start[mesh_id]] + self.edges[self.edges_end[mesh_id]:]
        self.faces = self.faces[:self.faces_start[mesh_id]] + self.faces[self.faces_end[mesh_id]:]
        self.num_verts -= self.num_verts_list[mesh_id]
        self.num_edges -= self.num_verts_list[mesh_id]
        self.num_faces -= self.num_verts_list[mesh_id]
        self.num_verts_list = self.num_verts_list[:mesh_id] + self.num_verts_list[mesh_id + 1:]
        self.verts_start = self.verts_start[:mesh_id] + self.verts_start[mesh_id + 1:]
        self.verts_end = self.verts_end[:mesh_id] + self.verts_end[mesh_id + 1:]
        self.edges_start = self.edges_start[:mesh_id] + self.edges_start[mesh_id + 1:]
        self.edges_end = self.edges_end[:mesh_id] + self.edges_end[mesh_id + 1:]
        self.faces_start = self.faces_start[:mesh_id] + self.faces_start[mesh_id + 1:]
        self.faces_end = self.faces_end[:mesh_id] + self.faces_end[mesh_id + 1:]

    def updateMesh(self, mesh_id, verts, edges, faces):
        vert_start = sum(self.num_verts_list[:mesh_id])
        vert_end = sum(self.num_verts_list[:mesh_id + 1])
        edge_start = sum(self.num_verts_list[:mesh_id])
        edge_end = sum(self.num_verts_list[:mesh_id + 1])
        face_start = sum(self.num_verts_list[:mesh_id])
        face_end = sum(self.num_verts_list[:mesh_id + 1])
        self.verts[vert_start:vert_end] = verts
        self.edges[edge_start:edge_end] = edges
        self.faces[face_start:face_end] = faces

    def setIndices(self):
        self.indices = ti.field(ti.u32, shape=self.num_faces * 3)
        for f in self.faces:
            self.indices[f.id * 3 + 0] = f.verts[0].id
            self.indices[f.id * 3 + 1] = f.verts[1].id
            self.indices[f.id * 3 + 2] = f.verts[2].id