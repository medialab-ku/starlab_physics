import os
import meshio
import shutil
import numpy as np
import taichi as ti
import open3d as o3d

@ti.data_oriented
class Exporter:
    def __init__(self, output_vtk, output_obj, output_ply, frameInterval):
        print("Initializing Exporter")
        self.vertices = None
        self.faces = None
        self.frame = 0
        self.frameInterval = int(frameInterval)
        self.output_vtk = output_vtk
        self.output_obj = output_obj
        self.output_ply = output_ply
        folder = [self.output_vtk, self.output_obj + "/static", self.output_obj + "/dynamic", self.output_ply]

        for idx in range(len(folder)):
            if not os.path.exists(folder[idx]):
                os.makedirs(folder[idx])
            else:
                for filename in os.listdir(folder[idx]):
                    file_path = os.path.join(folder[idx], filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                    except Exception as e:
                        print(f'Failed to delete {file_path}. Reason: {e}')

    def set_faces(self, faces):
        self.faces = faces
        if hasattr(faces, 'to_numpy'):
            faces = faces.to_numpy()
        else:
            faces = np.array(faces)
        faces = faces.reshape(-1, 3)
        self.faces = faces + 1

    def export_mesh(self, filename, vertices, MODE="SINGLE"):
        self.vertices = vertices

        if self.frame % self.frameInterval != 0:
            return

        if hasattr(vertices, 'to_numpy'):
            vertices = vertices.to_numpy()
        else:
            vertices = np.array(vertices)

        if MODE == "SINGLE1" or MODE == "MULTI1":
            if MODE == "SINGLE1":
                output_filename = os.path.join(self.output_obj, "static", filename)
            elif MODE == "MULTI1":
                name, ext = os.path.splitext(filename)
                output_filename = os.path.join(self.output_obj, "static", f"{name}{self.frame}{ext}")

            with open(output_filename, 'w') as f:
                for v in vertices:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                for face in self.faces:
                    f.write(f"f {face[0]} {face[1]} {face[2]}\n")
            print(f"mesh1, Saved {len(vertices)} vertices, {len(self.faces)} faces to {output_filename}")

        if MODE == "SINGLE2" or MODE == "MULTI2":
            if MODE == "SINGLE2":
                output_filename = os.path.join(self.output_obj, "dynamic", filename)
            elif MODE == "MULTI2":
                name, ext = os.path.splitext(filename)
                output_filename = os.path.join(self.output_obj, "dynamic", f"{name}{self.frame}{ext}")

            with open(output_filename, 'w') as f:
                for v in vertices:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                for face in self.faces:
                    f.write(f"f {face[0]} {face[1]} {face[2]}\n")
            print(f"mesh2, Saved {len(vertices)} vertices, {len(self.faces)} faces to {output_filename}")

    def export_ply(self, filename, vertices, MODE="SINGLE"):
        self.vertices = vertices

        if self.frame % self.frameInterval != 0:
            return

        if hasattr(vertices, 'to_numpy'):
            vertices = vertices.to_numpy()
        else:
            vertices = np.array(vertices)

        name, ext = os.path.splitext(filename)
        if MODE == "SINGLE" or MODE == "MULTI":
            if MODE == "SINGLE":
                if ext == ".ply":
                    output_filename = os.path.join(self.output_ply, filename)
                if ext == ".obj":
                    output_filename = os.path.join(self.output_obj, filename)
            elif MODE == "MULTI":
                if ext == ".ply":
                    output_filename = os.path.join(self.output_ply, f"{name}{self.frame}{ext}")
                if ext == ".obj":
                    output_filename = os.path.join(self.output_obj, f"{name}{self.frame}{ext}")
            else:
                raise ValueError("MODE must be either 'SINGLE' or 'MULTI'")

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
            mesh.triangles = o3d.utility.Vector3iVector([])
            o3d.io.write_triangle_mesh(output_filename, mesh)
            print(f"Saved {len(vertices)} vertices to {output_filename}")
            print(f"Saved {vertices.shape[0]} particles to {output_filename}")

    def export_vtk(self, filename, vertices, MODE="SINGLE"):
        self.vertices = vertices

        if self.frame % self.frameInterval != 0:
            return

        if hasattr(vertices, 'to_numpy'):
            vertices = vertices.to_numpy()
        else:
            vertices = np.array(vertices)

        if MODE == "SINGLE" or MODE == "MULTI":
            if MODE == "SINGLE":
                output_filename = os.path.join(self.output_vtk, filename)
            elif MODE == "MULTI":
                name, ext = os.path.splitext(filename)
                output_filename = os.path.join(self.output_vtk, f"{name}{self.frame}{ext}")
            else:
                raise ValueError("MODE must be either 'SINGLE' or 'MULTI'")

        n = vertices.shape[0]

        cells = [("vertex", np.array([[i] for i in range(n)], dtype=np.int32))]
        mesh = meshio.Mesh(points=vertices, cells=cells)
        meshio.write(output_filename, mesh)

        print(f"Saved {len(vertices)} particles to {output_filename}")

    def update(self):
        self.frame += 1

