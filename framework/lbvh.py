import taichi as ti

@ti.dataclass
class Node:
    object_id: ti.i32
    parent: ti.i32
    left: ti.i32
    right: ti.i32
    visited: ti.i32
    aabb_min: ti.math.vec3
    aabb_max: ti.math.vec3
    start: ti.i32
    end: ti.i32

@ti.data_oriented
class LBVH:
    def __init__(self, num_leafs):
        self.num_leafs = num_leafs
        self.num_nodes = 2 * self.num_leafs - 1
        # self.leaf_nodes = Node.field(shape=self.num_leafs)
        # self.internal_nodes = Node.field(shape=(self.num_leafs - 1))


        self.nodes = Node.field(shape=self.num_nodes)

        self.sorted_object_ids = ti.field(dtype=ti.i32, shape=self.num_leafs)
        self.object_ids = ti.field(dtype=ti.i32, shape=self.num_leafs)

        self.sorted_morton_codes = ti.field(dtype=ti.i32, shape=self.num_leafs)
        self.morton_codes = ti.field(dtype=ti.int32, shape=self.num_leafs)

        self.aabb_x = ti.Vector.field(n=3, dtype=ti.f32, shape=8 * self.num_nodes)
        self.aabb_indices = ti.field(dtype=ti.uint32, shape=24 * self.num_nodes)

        self.face_centers = ti.Vector.field(n=3, dtype=ti.f32, shape=self.num_leafs)
        self.zSort_line_idx = ti.field(dtype=ti.uint32, shape=self.num_nodes)

    # Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
    @ti.func
    def expand_bits(self, v):
        v = (v * ti.int64(0x00010001)) & ti.int64(0xFF0000FF)
        v = (v * ti.int64(0x00000101)) & ti.int64(0x0F00F00F)
        v = (v * ti.int64(0x00000011)) & ti.int64(0xC30C30C3)
        v = (v * ti.int64(0x00000005)) & ti.int64(0x49249249)
        return v

    @ti.func
    def morton_3d(self, x, y, z):
        x = ti.math.clamp(x * 1024., 0., 1023.)
        y = ti.math.clamp(y * 1024., 0., 1023.)
        z = ti.math.clamp(z * 1024., 0., 1023.)
        xx = self.expand_bits(ti.cast(x, ti.uint64))
        yy = self.expand_bits(ti.cast(y, ti.uint64))
        zz = self.expand_bits(ti.cast(z, ti.uint64))
        return xx * 4 + yy * 2 + zz

    @ti.kernel
    def assign_morton(self, mesh: ti.template(), aabb_min: ti.math.vec3, aabb_max: ti.math.vec3):

        for f in mesh.faces:
        # // obtain center of triangle
            u = f.verts[0]
            v = f.verts[1]
            w = f.verts[2]
            pos = (1. / 3.) * (u.x + v.x + w.x)
            self.face_centers[f.id] = pos

        # // normalize position
            x = (pos[0] - aabb_min[0]) / (aabb_max[0] - aabb_min[0])
            y = (pos[1] - aabb_min[1]) / (aabb_max[1] - aabb_min[1])
            z = (pos[2] - aabb_min[2]) / (aabb_max[2] - aabb_min[2])
        # // clamp to deal with numeric issues
            x = ti.math.clamp(x, 0., 1.)
            y = ti.math.clamp(y, 0., 1.)
            z = ti.math.clamp(z, 0., 1.)

    # // obtain and set morton code based on normalized position
            self.morton_codes[f.id] = self.morton_3d(x, y, z)
            self.object_ids[f.id] = f.id

    @ti.kernel
    def assign_leaf_nodes(self, mesh: ti.template()):
        # print(self.num_leafs)
        for f in mesh.faces:
            # // no need to set parent to nullptr, each child will have a parent
            id = self.object_ids[f.id]
            self.nodes[id + self.num_leafs - 1].object_id = f.id
            self.nodes[id + self.num_leafs - 1].left = -1
            self.nodes[id + self.num_leafs - 1].right = -1
            self.nodes[id + self.num_leafs - 1].aabb_min = f.aabb_min
            self.nodes[id + self.num_leafs - 1].aabb_max = f.aabb_max

            # // need to set for internal node parent to nullptr, for testing later
            # // there is one less internal node than leaf node, test for that
            # self.internal_nodes[i].parent = None

    @ti.func
    def delta(self, i, j):
        ret = -1
        if j <= (self.num_leafs - 1) and j >= 0:
            i_ = self.object_ids[i]
            j_ = self.object_ids[j]
            if i_ == j_:
                ret = 32 + ti.math.clz(self.morton_codes[i_] ^ self.morton_codes[j_])
            else:
                ret = ti.math.clz(self.morton_codes[i_] ^ self.morton_codes[j_])
        return ret

    @ti.func
    def find_split(self, l, r):
        first_code = self.morton_codes[self.object_ids[l]]
        last_code = self.morton_codes[self.object_ids[r]]

        ret = -1
        if first_code == last_code:
            ret = (l + r) // 2

        else:
            common_prefix = ti.math.clz(first_code ^ last_code)

            split = l
            step = r - l

            while step > 1:
                step = (step + 1) // 2
                new_split = split + step

                if new_split < r:
                    split_code = self.morton_codes[self.object_ids[new_split]]
                    split_prefix = ti.math.clz(first_code ^ split_code)
                    if split_prefix > common_prefix:
                        split = new_split

            ret = split
        return ret

    # @ti.func
    # def delta(self, a, b):
    #     return ti.math.clz(self.morton_codes[a] ^ self.morton_codes[b])

    @ti.func
    def determine_range(self, i, n):

        delta_l = self.delta(i, i - 1)
        delta_r = self.delta(i, i + 1)

        # print(delta_l, delta_r)
        d = 1

        delta_min = delta_l
        if delta_r < delta_l:
            d = - 1
            delta_min = delta_r

        # print(d)
        l_max = 2
        while self.delta(i, i + l_max * d) > delta_min:
            l_max <<= 2


        l = 0
        t = l_max // 2
        while t >= 1:
            if i + (l + t) * d >= 0 and i + (l + t) * d < n and self.delta(i, i + (l + t) * d) > delta_min:
                l += t
            t //= 2

        start = i
        end = i + l * d
        if d == -1:
            start = i + l * d
            end = i

        return start, end

    # @ti.kernel
    # def assign_internal_nodes_at_lev(self, num_lev: ti.int32):
    #     for i in range(num_lev):
    #         left = 2 * i + self.num_leafs
    #         right = left + 1
    #         parent = i + num_lev
    #         self.nodes[parent].left = left
    #         self.nodes[parent].right = right
    #         self.nodes[parent].aabb_min = ti.min(self.nodes[left].aabb_min, self.nodes[right].aabb_min)
    #         self.nodes[parent].aabb_max = ti.max(self.nodes[left].aabb_max, self.nodes[right].aabb_max)


    @ti.kernel
    def assign_internal_nodes(self):
        # start, end = self.determine_range(7, self.num_leafs)
        # print(start, end)
        # split = self.find_split(start, end)
        # left = split + self.num_leafs if split == start else split
        # right = split + 1 + self.num_leafs if split + 1 == end else split + 1
        # print(left, right)
        for i in range(self.num_leafs - 1):
            start, end = self.determine_range(i, self.num_leafs)
            split = self.find_split(start, end)
            # print(i, end - start, split)
            left = split + self.num_leafs - 1 if split == start else split
            right = split + 1 + self.num_leafs - 1 if split + 1 == end else split + 1

            self.nodes[i].left = left
            self.nodes[i].right = right
            self.nodes[left].parent = i
            self.nodes[right].parent = i
            self.nodes[i].start = start
            self.nodes[i].end = end



    def compute_node_aabbs(self):
        n = 2 * self.num_leafs - 1
        total = 0
        # for level in range(n.bit_length()):
        #     level_size = self.num_leafs >> level
        #     offset = (self.num_leafs >> (level + 1))
        #     total += level_size
        #     print(offset, level_size)
        #     # self.compute_node_aabbs_at_lev(offset, level_size)
        # print(total)

        for i in range(self.num_leafs - 1):
            start, end = self.nodes[i].start, self.nodes[i].end
            aabb_min = self.nodes[start + self.num_leafs - 1].aabb_min
            aabb_max = self.nodes[start + self.num_leafs - 1].aabb_max

            for j in ti.static(range(end - start)):
                aabb_min = ti.min(aabb_min, self.nodes[start + j].aabb_min)
                aabb_max = ti.max(aabb_max, self.nodes[start + j].aabb_max)

            self.nodes[i].aabb_min = aabb_min
            self.nodes[i].aabb_max = aabb_max
        print(self.nodes[0].aabb_min, self.nodes[0].aabb_max)

    @ti.kernel
    def compute_node_aabbs_at_lev(self, offset: ti.int32, size: ti.int32):

        for i in range(size):
            id = size + offset
            left, right = self.nodes[id].left, self.nodes[id].right
            min0, min1 = self.nodes[left].aabb_min, self.nodes[right].aabb_min
            max0, max1 = self.nodes[left].aabb_max, self.nodes[right].aabb_max

            self.nodes[id].aabb_min = ti.min(min0, min1)
            self.nodes[id].aabb_max = ti.max(max0, max1)

    def build(self, mesh, aabb_min_g, aabb_max_g):
        self.assign_morton(mesh, aabb_min_g, aabb_max_g)
        ti.algorithms.parallel_sort(keys=self.morton_codes, values=self.object_ids)

        self.assign_leaf_nodes(mesh)
        self.assign_internal_nodes()
        self.compute_node_aabbs()

    @ti.func
    def aabb_overlap(self, min1, max1, min2, max2):
        return (min1[0] <= max2[0] and max1[0] >= min2[0] and
                min1[1] <= max2[1] and max1[1] >= min2[1] and
                min1[2] <= max2[2] and max1[2] >= min2[2])

    @ti.kernel
    def update_zSort_face_centers_and_line(self):

        for i in range(self.num_leafs - 1):
            self.zSort_line_idx[2 * i + 0] = self.object_ids[i]
            self.zSort_line_idx[2 * i + 1] = self.object_ids[i + 1]

    def draw_zSort(self, scene):
        self.update_zSort_face_centers_and_line()
        scene.lines(self.face_centers, indices=self.zSort_line_idx, width=1.0, color=(1, 0, 0))
    #
    @ti.kernel
    def update_aabb_x_and_lines(self):
        for n in range(self.num_nodes):
            # i = n + self.num_leafs - 1
            aabb_min = self.nodes[n].aabb_min
            aabb_max = self.nodes[n].aabb_max

            self.aabb_x[8 * n + 0] = ti.math.vec3(aabb_max[0], aabb_max[1], aabb_max[2])
            self.aabb_x[8 * n + 1] = ti.math.vec3(aabb_min[0], aabb_max[1], aabb_max[2])
            self.aabb_x[8 * n + 2] = ti.math.vec3(aabb_min[0], aabb_max[1], aabb_min[2])
            self.aabb_x[8 * n + 3] = ti.math.vec3(aabb_max[0], aabb_max[1], aabb_min[2])

            self.aabb_x[8 * n + 4] = ti.math.vec3(aabb_max[0], aabb_min[1], aabb_max[2])
            self.aabb_x[8 * n + 5] = ti.math.vec3(aabb_min[0], aabb_min[1], aabb_max[2])
            self.aabb_x[8 * n + 6] = ti.math.vec3(aabb_min[0], aabb_min[1], aabb_min[2])
            self.aabb_x[8 * n + 7] = ti.math.vec3(aabb_max[0], aabb_min[1], aabb_min[2])

            self.aabb_indices[24 * n + 0] = 8 * n + 0
            self.aabb_indices[24 * n + 1] = 8 * n + 1
            self.aabb_indices[24 * n + 2] = 8 * n + 1
            self.aabb_indices[24 * n + 3] = 8 * n + 2
            self.aabb_indices[24 * n + 4] = 8 * n + 2
            self.aabb_indices[24 * n + 5] = 8 * n + 3
            self.aabb_indices[24 * n + 6] = 8 * n + 3
            self.aabb_indices[24 * n + 7] = 8 * n + 0
            self.aabb_indices[24 * n + 8] = 8 * n + 4
            self.aabb_indices[24 * n + 9] = 8 * n + 5
            self.aabb_indices[24 * n + 10] = 8 * n + 5
            self.aabb_indices[24 * n + 11] = 8 * n + 6
            self.aabb_indices[24 * n + 12] = 8 * n + 6
            self.aabb_indices[24 * n + 13] = 8 * n + 7
            self.aabb_indices[24 * n + 14] = 8 * n + 7
            self.aabb_indices[24 * n + 15] = 8 * n + 4
            self.aabb_indices[24 * n + 16] = 8 * n + 0
            self.aabb_indices[24 * n + 17] = 8 * n + 4
            self.aabb_indices[24 * n + 18] = 8 * n + 1
            self.aabb_indices[24 * n + 19] = 8 * n + 5
            self.aabb_indices[24 * n + 20] = 8 * n + 2
            self.aabb_indices[24 * n + 21] = 8 * n + 6
            self.aabb_indices[24 * n + 22] = 8 * n + 3
            self.aabb_indices[24 * n + 23] = 8 * n + 7

    def draw_bvh_aabb(self, scene):
        self.update_aabb_x_and_lines()
        scene.lines(self.aabb_x, indices=self.aabb_indices, width=1.0, color=(0, 0, 0))
        # scene.particles(self.aabb_x, radius=0.5, color=(0, 0, 0))