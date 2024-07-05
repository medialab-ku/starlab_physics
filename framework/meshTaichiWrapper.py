
import taichi as ti
import meshtaichi_patcher as patcher
import igl
import os
import numpy as np
@ti.data_oriented
class MeshTaichiWrapper:

    def __init__(self,
                 model_path,
                 trans=ti.math.vec3(0, 0, 0),
                 rot=ti.math.vec3(0, 0, 0),
                 scale=1.0,
                 is_static=False):

        self.is_static = is_static
        self.mesh = patcher.load_mesh(model_path, relations=["FV", "EV", "VV", "VE"])
        self.mesh.verts.place({'fixed': ti.f32,
                               'm_inv': ti.f32,
                               'x0': ti.math.vec3,
                               'x': ti.math.vec3,
                               'y': ti.math.vec3,
                               'v': ti.math.vec3,
                               'dx': ti.math.vec3,
                               'dv': ti.math.vec3,
                               'nc': ti.f32})
        self.mesh.edges.place({'l0': ti.f32})
        self.mesh.faces.place({'aabb_min': ti.math.vec3,
                               'aabb_max': ti.math.vec3,
                               'morton_code': ti.uint32})
        self.mesh.verts.fixed.fill(1.0)
        self.mesh.verts.m_inv.fill(0.0)
        self.mesh.verts.v.fill(0.0)
        self.mesh.verts.x.from_numpy(self.mesh.get_position_as_numpy())
        self.num_verts = len(self.mesh.verts)


        self.faces = self.mesh.faces
        self.verts = self.mesh.verts

        print(len(self.mesh.faces))

        # self.setCenterToOrigin()
        self.face_indices = ti.field(dtype=ti.int32, shape=len(self.mesh.faces) * 3)
        self.edge_indices = ti.field(dtype=ti.int32, shape=len(self.mesh.edges) * 2)
        self.fid_np = self.face_indices.to_numpy()
        self.fid_np = np.reshape(self.fid_np, (len(self.mesh.faces), 3))
        self.verts = self.mesh.verts
        self.faces = self.mesh.faces
        self.edges = self.mesh.edges

        self.trans = trans
        self.rot = rot
        self.scale = scale

        self.applyTransform()
        self.initFaceIndices()
        self.initEdgeIndices()
        self.mesh.verts.x0.copy_from(self.mesh.verts.x)

    def reset(self):
        self.mesh.verts.x.copy_from(self.mesh.verts.x0)
        self.mesh.verts.v.fill(0.)
        self.mesh.verts.fixed.fill(0.0)

    @ti.kernel
    def initFaceIndices(self):
        for f in self.mesh.faces:
            self.face_indices[f.id * 3 + 0] = f.verts[0].id
            self.face_indices[f.id * 3 + 1] = f.verts[1].id
            self.face_indices[f.id * 3 + 2] = f.verts[2].id

    @ti.kernel
    def initEdgeIndices(self):
        for e in self.mesh.edges:
            l0 = (e.verts[0].x - e.verts[1].x).norm()
            e.l0 = l0

            e.verts[0].m_inv += 0.5 * l0
            e.verts[1].m_inv += 0.5 * l0

            self.edge_indices[e.id * 2 + 0] = e.verts[0].id
            self.edge_indices[e.id * 2 + 1] = e.verts[1].id

        for v in self.mesh.verts:
            v.m_inv = 1.0 / v.m_inv

    @ti.kernel
    def setCenterToOrigin(self):

        center = ti.math.vec3(0, 0, 0)
        for v in self.verts:
            center += v.x

        center /= self.num_verts
        for v in self.mesh.verts:
            v.x -= center

    @ti.kernel
    def applyTransform(self):

        # self.setCenterToOrigin()

        for v in self.mesh.verts:
            v.x *= self.scale

        for v in self.mesh.verts:
            v_4d = ti.Vector([v.x[0], v.x[1], v.x[2], 1])
            rot_rad_x = ti.math.radians(self.rot[0])
            rot_rad_y = ti.math.radians(self.rot[1])
            rot_rad_z = ti.math.radians(self.rot[2])
            rv = ti.math.rotation3d(rot_rad_x, rot_rad_y, rot_rad_z) @ v_4d
            v.x = ti.Vector([rv[0], rv[1], rv[2]])

        for v in self.mesh.verts:
            v.x += self.trans

    @ti.kernel
    def computeAABB(self) -> (ti.math.vec3, ti.math.vec3):
        aabb_min = ti.math.vec3(1e5)
        aabb_max = ti.math.vec3(-1e5)

        for v in self.mesh.verts:
            temp = v.x
            ti.atomic_max(aabb_max, temp)
            ti.atomic_min(aabb_min, temp)

        return aabb_min, aabb_max


    @ti.kernel
    def computeAABB_faces(self, padding: ti.f32):

        for f in self.mesh.faces:
            f.aabb_min = ti.math.min(f.verts[0].x, f.verts[1].x, f.verts[2].x)
            f.aabb_max = ti.math.max(f.verts[0].x, f.verts[1].x, f.verts[2].x)

            f.aabb_min -= padding * ti.math.vec3(1.0)
            f.aabb_max += padding * ti.math.vec3(1.0)


    def export(self, scene_name, mesh_id, frame,is_static = False):
        if is_static:
            directory = os.path.join("results/", scene_name, "StaticMesh_ID_" + str(mesh_id))
        else:
            directory = os.path.join("results/", scene_name, "Mesh_ID_" + str(mesh_id))

        try:
            if not os.path.exists(directory):
                os.makedirs(directory)
        except OSError:
            print("Error: Failed to create folder" + directory)

        x_np = self.mesh.verts.x.to_numpy()
        file_name = "Mesh_obj_" + str(frame) + ".obj"
        file_path = os.path.join(directory, file_name)
        print("exporting ", file_path.__str__())
        igl.write_triangle_mesh(file_path, x_np, self.fid_np, force_ascii=True)
        print("done")
