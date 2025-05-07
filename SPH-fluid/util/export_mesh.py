# data -> output -> mesh -> change
import os
import shutil
import numpy as np
import trimesh
import open3d as o3d

class Exporter:
    def __init__(self, folder, frameInterval):
        print("Initializing Exporter")
        self.vertices = None
        self.faces = None
        self.frame = 0
        self.folder = folder
        self.frameInterval = int(frameInterval)

        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        else:
            for filename in os.listdir(self.folder):
                file_path = os.path.join(self.folder, filename)
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
        self.frame += 1

        if self.frame % self.frameInterval != 0:
            return

        if hasattr(vertices, 'to_numpy'):
            vertices = vertices.to_numpy()
        else:
            vertices = np.array(vertices)

        if MODE == "SINGLE" or MODE == "MULTI":
            if MODE == "SINGLE":
                output_filename = os.path.join(self.folder, filename)
            elif MODE == "MULTI":
                name, ext = os.path.splitext(filename)
                output_filename = os.path.join(self.folder, f"{name}{self.frame}{ext}")
            else:
                raise ValueError("MODE must be either 'SINGLE' or 'MULTI'")

            with open(output_filename, 'w') as f:
                for v in vertices:
                    f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                for face in self.faces:
                    f.write(f"f {face[0]} {face[1]} {face[2]}\n")

        if MODE == "PARTICLE":
            name, ext = os.path.splitext(filename)
            output_filename = os.path.join(self.folder, f"{name}{self.frame}{ext}")

            # ToDo Trimesh
            # cloud = trimesh.points.PointCloud(vertices)
            # hull = cloud.convex_hull
            # hull.export(output_filename)

            # ToDo open3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(vertices)
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

            densities = np.asarray(densities)
            density_thresh = np.percentile(densities, 10)
            vertices_to_remove = densities < density_thresh
            mesh.remove_vertices_by_mask(vertices_to_remove)

            o3d.io.write_triangle_mesh(output_filename, mesh)

