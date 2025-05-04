import trimesh
import numpy as np
import open3d as o3d

# 1. 직육면체 내부에 점 찍기
num_points = 5000  # 원하는 밀도만큼
box_min = np.array([-1, -1, -1])  # 직육면체 최소 좌표
box_max = np.array([1, 1, 1])     # 직육면체 최대 좌표

points = np.random.uniform(low=box_min, high=box_max, size=(num_points, 3))

# 2. Trimesh로 Convex Hull 생성
cloud = trimesh.points.PointCloud(points)
hull = cloud.convex_hull

# 3. 저장
hull.export('trimesh_box_output.obj')

# 4. 시각화 (Open3D)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.paint_uniform_color([0, 0, 1])  # 파란 점

mesh = o3d.io.read_triangle_mesh('trimesh_box_output.obj')
mesh.paint_uniform_color([0.7, 0.7, 0.7])  # 회색 메쉬

o3d.visualization.draw_geometries([pcd, mesh])