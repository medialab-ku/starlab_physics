import open3d as o3d
import numpy as np
import os

def mesh_to_filled_particles(mesh_path, voxel_size=0.05, output_ply="filled_points.ply"):
    mesh_legacy = o3d.io.read_triangle_mesh(mesh_path)
    mesh_legacy.compute_vertex_normals()
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)

    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)

    aabb = mesh_legacy.get_axis_aligned_bounding_box()
    min_bound = aabb.min_bound
    max_bound = aabb.max_bound

    x = np.arange(min_bound[0], max_bound[0], voxel_size)
    y = np.arange(min_bound[1], max_bound[1], voxel_size)
    z = np.arange(min_bound[2], max_bound[2], voxel_size)
    grid = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1).reshape(-1, 3)

    query_points = o3d.core.Tensor(grid, dtype=o3d.core.Dtype.Float32)

    occupancy = scene.compute_occupancy(query_points)
    inside_mask = occupancy.numpy() == 1
    inside_points = query_points[inside_mask]

    pcd = o3d.t.geometry.PointCloud()
    pcd.point.positions = inside_points
    o3d.io.write_point_cloud(output_ply, pcd.to_legacy())
    print(f"Saved {inside_points.shape[0]} particles to {output_ply}")

    return inside_points.numpy()

if __name__ == "__main__":
    mesh_path = "input.obj"
    full_path = os.path.join(os.getcwd(), mesh_path)
    points = mesh_to_filled_particles(full_path, voxel_size=0.001, output_ply="filled_points.ply")
    print("Generated", len(points), "internal particles")