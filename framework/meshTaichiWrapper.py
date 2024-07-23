
import taichi as ti
import meshtaichi_patcher as patcher
import igl
import os
import numpy as np
import random

@ti.data_oriented
class MeshTaichiWrapper:
    def __init__(self,
                 model_path,
                 offsets,
                 trans=ti.math.vec3(0, 0, 0),
                 rot=ti.math.vec3(0, 0, 0),
                 scale=1.0,
                 is_static=False):

        self.is_static = is_static
        self.mesh = patcher.load_mesh(model_path, relations=["FV", "EV"])
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

        self.offsets = offsets
        self.mesh.verts.fixed.fill(1.0)
        self.mesh.verts.m_inv.fill(0.0)
        self.mesh.verts.v.fill(0.0)
        self.mesh.verts.x.from_numpy(self.mesh.get_position_as_numpy())
        self.num_verts = len(self.mesh.verts)
        self.num_edges = len(self.mesh.edges)
        self.num_faces = len(self.mesh.faces)

        self.faces = self.mesh.faces
        self.verts = self.mesh.verts

        print(model_path, "# faces:", len(self.mesh.faces))

        # self.setCenterToOrigin()
        self.face_indices = ti.field(dtype=ti.int32, shape=len(self.mesh.faces) * 3)
        self.edge_indices = ti.field(dtype=ti.int32, shape=len(self.mesh.edges) * 2)

        self.colors = ti.Vector.field(n=3, dtype=ti.f32, shape=len(self.mesh.verts))
        self.init_color()
        # self.colors.fill(0.1)

        self.verts = self.mesh.verts
        self.faces = self.mesh.faces
        self.edges = self.mesh.edges

        self.trans = trans
        self.rot = rot
        self.scale = scale

        self.applyTransform()
        self.initFaceIndices()
        self.fid_np = self.face_indices.to_numpy()
        self.fid_np = np.reshape(self.fid_np, (len(self.mesh.faces), 3))
        self.initEdgeIndices()
        self.eid_np = self.edge_indices.to_numpy()
        self.eid_np = np.reshape(self.eid_np, (len(self.mesh.edges), 2))
        self.mesh.verts.x0.copy_from(self.mesh.verts.x)
        self.eid_field = ti.field(dtype=ti.int32, shape=self.eid_np.shape)
        self.eid_field.from_numpy(self.eid_np)

        self.edge_indices_for_color = ti.field(dtype=ti.int32, shape=self.num_edges)
        self.edges_color = ti.field(dtype=ti.int32, shape=self.num_edges)
        self.edges_color.fill(-1)
        self.adj_edges_list = ti.field(dtype=ti.int32, shape=(self.num_edges, self.num_edges))
        self.adj_edges_list.fill(0)
        self.num_colors = 10
        self.available_colors = ti.field(dtype=ti.int32, shape=self.num_colors) # temporary

        print("verts :", self.num_verts, "edges :", self.num_edges)
        self.initEdgeIndicesForColor() # ensure the sequence of edge indices
        self.initAdjEdges()

        self.edge_indices_for_color_np = self.edge_indices_for_color.to_numpy()
        self.edges_color_np = self.edges_color.to_numpy()
        self.adj_edges_list_np = self.adj_edges_list.to_numpy()
        self.available_colors_np = self.available_colors.to_numpy()

        self.colorEdgesGreedy()

        self.edges_color.from_numpy(self.edges_color_np)
        self.adj_edges_list.from_numpy(self.adj_edges_list_np)
        self.available_colors.from_numpy(self.available_colors_np)

        # self.printEdgesColor()
        self.checkAdjColor()

        self.sorted_edges_index = self.edge_indices_for_color
        self.sorted_edges_color = self.edges_color
        self.color_prefix_sum = ti.field(dtype=ti.i32, shape=self.num_colors)
        self.color_prefix_sum.fill(0)

        # before sort
        self.sorted_edges_index_np = self.sorted_edges_index.to_numpy()
        self.sorted_edges_color_np = self.sorted_edges_color.to_numpy()
        self.color_prefix_sum_np = self.color_prefix_sum.to_numpy()

        self.colorCountingSort() # sort

        # after sort
        self.sorted_edges_index.from_numpy(self.sorted_edges_index_np)
        self.sorted_edges_color.from_numpy(self.sorted_edges_color_np)
        self.color_prefix_sum.from_numpy(self.color_prefix_sum_np)

        self.bending_indices = ti.field(dtype=ti.i32)
        self.initBendingIndices()

        self.render_bending_vert = ti.Vector.field(3, dtype=ti.f32, shape=(len(self.mesh.verts),))
        self.init_render_bending_vert()


    @ti.kernel
    def init_render_bending_vert(self):
        for v in self.mesh.verts:
            self.render_bending_vert[v.id] = ti.Vector([v.x[0], v.x[1], v.x[2]])

    def initBendingIndices(self):
        # https://carmencincotti.com/2022-09-05/the-most-performant-bending-constraint-of-xpbd/
        bend_count, neighbor_set = self.findTriNeighbors()
        bending_indices_np = self.getBendingPair(bend_count, neighbor_set)

        ti.root.dense(ti.i, bending_indices_np.shape[0] * 2).place(self.bending_indices)

        for i in range(bending_indices_np.shape[0]):
            self.bending_indices[2 * i] = bending_indices_np[i][0]
            self.bending_indices[2 * i + 1] = bending_indices_np[i][1]

    @ti.kernel
    def bendinIndi(self):
        for v in self.mesh.verts:
            self.render_bending_vert[v.id] = ti.Vector([v.x[0], v.x[1], v.x[2]])

    def findTriNeighbors(self):
        # print("initBend")
        # print(self.fid_np)
        # print(self.fid_np.shape)
        num_f = np.rint(self.fid_np.shape[0]).astype(int)
        edgeTable = np.zeros((num_f * 3, self.fid_np.shape[1]), dtype=int)
        # print(edgeTable.shape)

        for f in range(self.fid_np.shape[0]):
            eT = np.zeros((3, 3))
            v1, v2, v3 = np.sort(self.fid_np[f])
            e1, e2, e3 = 3 * f, 3 * f + 1, 3 * f + 2
            eT[0, :] = [v1, v2, e1]
            eT[1, :] = [v1, v3, e2]
            eT[2, :] = [v2, v3, e3]

            edgeTable[3 * f:3 * f + 3, :] = eT

        ind = np.lexsort((edgeTable[:, 1], edgeTable[:, 0]))
        edgeTable = edgeTable[ind]
        # print(edgeTable)

        neighbors = np.zeros((num_f * 3), dtype=int)
        neighbors.fill(-1)

        ii = 0
        bending_constraint_count = 0
        while (ii < 3 * num_f - 1):
            e0 = edgeTable[ii, :]
            e1 = edgeTable[ii + 1, :]

            if (e0[0] == e1[0] and e0[1] == e1[1]):
                neighbors[e0[2]] = e1[2]
                neighbors[e1[2]] = e0[2]

                bending_constraint_count = bending_constraint_count + 1
                ii = ii + 2
            else:
                ii = ii + 1

        # print(bending_constraint_count, "asdf!!")
        return bending_constraint_count, neighbors

    def getBendingPair(self, bend_count, neighbors):

        num_f = np.rint(self.fid_np.shape[0]).astype(int)
        pairs = np.zeros((bend_count, 2), dtype=int)

        count = 0
        # print(neighbors)

        for f in range(num_f):
            for i in range(3):
                eid = 3 * f + i
                neighbor_edge = neighbors[eid]

                if neighbor_edge >= 0:
                    # print(eid,neighbor_edge,neighbors[neighbor_edge])
                    neighbors[neighbor_edge] = -1
                    # find not shared vertex in common edge of adjacent triangles
                    v = np.sort(self.fid_np[f])

                    neighbor_fid = int(np.floor(neighbor_edge / 3.0 + 1e-4) + 1e-4)
                    neighbor_eid_local = neighbor_fid % 3

                    w = np.sort(self.fid_np[neighbor_fid])

                    pairs[count, 0] = v[~np.isin(v, w)]
                    pairs[count, 1] = w[~np.isin(w, v)]

                    count = count + 1

        # print("검산!!!",count == bend_count)
        return pairs



###########################
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
    def initEdgeIndicesForColor(self):
        for e in self.mesh.edges:
            self.edge_indices_for_color[e.id] = e.id

    @ti.kernel
    def initAdjEdges(self):
        print("Initializing the adjacency list...", end=" ")
        for i in range(self.num_edges):
            for j in range(self.num_edges):
                e1 = self.edge_indices_for_color[i]
                e2 = self.edge_indices_for_color[j]

                if e1 != e2:
                    v1 = self.eid_field[e1,0]
                    v2 = self.eid_field[e1,1]
                    v3 = self.eid_field[e2,0]
                    v4 = self.eid_field[e2,1]

                    if v1 == v3 or v1 == v4 or v2 == v3 or v2 == v4:
                        self.adj_edges_list[e1,e2] = 1
                        # print(e1, e2, "adjacent")
        print("Done.")

    def colorEdgesGreedy(self):
        print("Coloring edges...", end=" ")
        for i in range(self.num_edges):
            self.available_colors_np.fill(1)
            e1 = self.edge_indices_for_color_np[i]

            # determine forbidden colors to e1
            for j in range(self.num_edges):
                e2 = self.edge_indices_for_color_np[j]
                if self.adj_edges_list_np[e1,e2] == 1 and self.edges_color_np[e2] != -1:
                    self.available_colors_np[self.edges_color_np[e2]] = 0

            # find the first available color
            self.edges_color_np[e1] = np.argmax(self.available_colors_np)
        print("Done.")

    @ti.kernel
    def checkAdjColor(self):
        print("Checking Integrity of the adjacency list...", end=" ")
        for i in range(self.num_edges):
            for j in range(self.num_edges):
                e1 = self.edge_indices_for_color[i]
                e2 = self.edge_indices_for_color[j]
                if self.edges_color[e1] == -1 or self.edges_color[e2] == -1:
                    print("one of colors is -1 :", e1, e2, self.edges_color[e1], self.edges_color[e2])
                elif e1 != e2 and self.adj_edges_list[e1,e2] == 1 and self.edges_color[e1] == self.edges_color[e2]:
                    print("both colors are equivalent :", e1, e2, self.edges_color[e1], self.edges_color[e2])
        print("Done.")

    @ti.kernel
    def printEdgesColor(self):
        for e in self.mesh.edges:
            print(e.id, self.edges_color[e.id], end="\t")
        print()

    def colorCountingSort(self):
        sorted_edges_index_temp = np.zeros_like(self.sorted_edges_index_np)
        sorted_edges_color_temp = np.zeros_like(self.sorted_edges_color_np)

        # count the number of times each color occurs in the input
        for c in self.edges_color_np:
            self.color_prefix_sum_np[c] += 1

        # modify the counting array to give the number of values smaller than index
        for i in range(1,self.num_colors):
            self.color_prefix_sum_np[i] += self.color_prefix_sum_np[i-1]

        # transfer numbers from back to forth at locations provided by counting array
        for i in range(self.num_edges-1, -1, -1):
            idx = self.sorted_edges_index_np[i]
            color = self.sorted_edges_color_np[i]
            self.color_prefix_sum_np[color] -= 1
            sorted_edges_color_temp[self.color_prefix_sum_np[color]] = color
            sorted_edges_index_temp[self.color_prefix_sum_np[color]] = idx

        self.sorted_edges_color_np = np.copy(sorted_edges_color_temp)
        self.sorted_edges_index_np = np.copy(sorted_edges_index_temp)
        for i in range(self.num_edges):
            print(i, self.sorted_edges_color_np[i], self.sorted_edges_index_np[i])
        print(self.sorted_edges_color_np)
        print(self.color_prefix_sum_np)
        print(self.sorted_edges_index_np)


    # @ti.kernel
    # def visualizeVertsColor(self):
    #     for e in range(self.num_edges):
    #         if self.edges_color[e] >= 0:
    #             if self.edges_color[e] == 0:
    #                 self.color_RGB[e] = ti.Vector([1.0, 0.0, 0.0])
    #             elif self.edges_color[e] == 1:
    #                 self.color_RGB[e] = ti.Vector([0.0, 1.0, 0.0])
    #             elif self.edges_color[e] == 2:
    #                 self.color_RGB[e] = ti.Vector([0.0, 0.0, 1.0])
    #             elif self.edges_color[e] == 3:
    #                 self.color_RGB[e] = ti.Vector([1.0, 1.0, 0.0])
    #             elif self.edges_color[e] == 4:
    #                 self.color_RGB[e] = ti.Vector([1.0, 0.0, 1.0])
    #             elif self.edges_color[e] == 5:
    #                 self.color_RGB[e] = ti.Vector([0.0, 1.0, 1.0])
    #             elif self.edges_color[e] == 6:
    #                 self.color_RGB[e] = ti.Vector([1.0, 1.0, 1.0])
    #             elif self.edges_color[e] == 7:
    #                 self.color_RGB[e] = ti.Vector([1.0, 0.5, 0.5])
    #             elif self.edges_color[e] == 8:
    #                 self.color_RGB[e] = ti.Vector([0.5, 1.0, 0.5])
    #             elif self.edges_color[e] == 9:
    #                 self.color_RGB[e] = ti.Vector([0.5, 0.5, 1.0])



    # @ti.kernel
    # def Hierholzer(self):
    #
    #     edges = []
    #     first = 0
    #     stack = []
    #     stack.append(first)
    #     ret = []
    #     while len(stack) > 0:
    #         v = stack.top()
    #
    #     if !edges[v].size():
    #         ret.append(v)
    #         stack.pop()
    #     else:
    #         i = edges[v][edges[v].size() - 1]
    #         edges[v].pop_back()
    #         std::vector < int >::iterator it = std::find(edges[i].begin(), edges[i].end(), v);
    #         edges[i].erase(it);
    #         stack.push(i)
    #
    #     return ret

    def init_color(self):
        # print(self.offsets)
        for i in range(len(self.offsets)):

            r = random.randrange(0, 255) / 256
            g = random.randrange(0, 255) / 256
            b = random.randrange(0, 255) / 256
            size = 0
            if i < len(self.offsets) - 1:
                size = self.offsets[i + 1] - self.offsets[i]

            else:
                size = len(self.verts) - self.offsets[i]
            #
            self.init_colors(self.offsets[i], size, color=ti.math.vec3(r, g, b))

    @ti.kernel
    def init_colors(self, offset: ti.i32, size: ti.i32, color: ti.math.vec3):

        for i in range(size):
            self.colors[i + offset] = color

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

            # f.aabb_min -= padding * ti.math.vec3(1.0)
            # f.aabb_max += padding * ti.math.vec3(1.0)


    def export(self, scene_name, frame, is_static = False):
        if is_static:
            directory = os.path.join("results/", scene_name, "StaticMesh_ID_")
        else:
            directory = os.path.join("results/", scene_name, "Mesh_ID_")

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
