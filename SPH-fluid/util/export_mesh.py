# data -> output -> mesh -> change
import sys
import os
sys.path.append(os.path.dirname(__file__))
import shutil
import partio
import numpy as np
import taichi as ti
from scipy.spatial import cKDTree
import mcubes
import open3d as o3d
# import trimesh
import pyvista as pv

@ti.data_oriented
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
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(vertices)
            # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            # mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
            #
            # densities = np.asarray(densities)
            # density_thresh = np.percentile(densities, 10)
            # vertices_to_remove = densities < density_thresh
            # mesh.remove_vertices_by_mask(vertices_to_remove)
            #
            # o3d.io.write_triangle_mesh(output_filename, mesh)

            # ToDo PyMcube
            grid_res = 128
            particle_pos = vertices.astype(np.float32)

            min_pos = np.min(particle_pos, axis=0)
            max_pos = np.max(particle_pos, axis=0)
            particle_pos = (particle_pos - min_pos) / (max_pos - min_pos + 1e-8)

            density_field = compute_density_kdtree(particle_pos, grid_res)
            min_d = np.min(density_field)
            max_d = np.max(density_field)

            alpha = 0.85
            iso_level = min_d + alpha * (max_d - min_d)

            # # ToDo make density plot
            # x = np.linspace(0, 1, grid_res + 1)
            # y = np.linspace(0, 1, grid_res + 1)
            # z = np.linspace(0, 1, grid_res + 1)
            # grid = pv.RectilinearGrid(x, y, z)
            # grid.cell_data["values"] = density_field.flatten(order="F")
            # plotter = pv.Plotter()
            # plotter.add_volume(grid, cmap="viridis", opacity="sigmoid")
            # plotter.show()

            vertices_mc, triangles_mc = mcubes.marching_cubes(density_field,  iso_level)
            vertices_mc *= 1.0 / grid_res

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices_mc)
            mesh.triangles = o3d.utility.Vector3iVector(triangles_mc)
            o3d.io.write_triangle_mesh(output_filename, mesh)

    def export_ply(self, filename, vertices, MODE="SINGLE"):
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

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices.astype(np.float64))
            mesh.triangles = o3d.utility.Vector3iVector([])
            o3d.io.write_triangle_mesh(output_filename, mesh)
            print(f"Saved {len(vertices)} vertices to {output_filename}")
            print(f"Saved {vertices.shape[0]} particles to {output_filename}")

    def export_bgeo(self, filename, vertices, MODE="SINGLE"):
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

            parts = partio.create()
            pos_attr = parts.addAttribute("position", partio.VECTOR, 3)

            for v in vertices:
                idx = parts.addParticle()
                parts.set(pos_attr, idx, [float(v[0]), float(v[1]), float(v[2])])

            partio.write(output_filename, parts)

            print(f"Saved {len(vertices)} particles to {output_filename}")


def compute_density_kdtree(particle_pos, grid_res=64, k=2, sigma=0.03):
    x = np.linspace(0, 1, grid_res, endpoint=False) + 0.5 * 1.0 / grid_res
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    grid_points = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

    tree = cKDTree(particle_pos)
    dists, _ = tree.query(grid_points, k=k, workers=-1)

    bandwidth = 1 / (2 * sigma ** 2)
    density_flat = np.sum(np.exp(-dists ** 2 * bandwidth), axis=1)

    density_field = density_flat.reshape((grid_res, grid_res, grid_res))
    return density_field

