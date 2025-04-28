import open3d as o3d
import numpy as np

# 1. 구 위에 무작위 점 생성
points = np.random.randn(1000, 3)
points /= np.linalg.norm(points, axis=1, keepdims=True)  # 구 표면 정규화

# 2. 포인트 클라우드 객체 만들기
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)

# 3. 법선 추정
pcd.estimate_normals()

# 4. Poisson Surface Reconstruction
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)

# 5. 노이즈 제거 (밀도 낮은 점들 제거)
density_thresh = np.quantile(densities, 0.1)
mesh.remove_vertices_by_mask(densities < density_thresh)

# 6. 점들 컬러 설정 (보이기 쉽게)
pcd.paint_uniform_color([1, 0, 0])  # 빨간색 점

# 7. 메쉬 컬러 설정 (보이기 쉽게)
mesh.paint_uniform_color([0.7, 0.7, 0.7])  # 회색 메쉬

# 8. 둘 다 같이 띄우기
o3d.visualization.draw_geometries([pcd, mesh])