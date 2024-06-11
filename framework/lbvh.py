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
        self.morton_codes = ti.field(dtype=ti.uint64, shape=self.num_leafs)

        self.aabb_x = ti.Vector.field(n=3, dtype=ti.f32, shape=8 * self.num_nodes)
        self.aabb_indices = ti.field(dtype=ti.uint32, shape=12 * self.num_nodes)

        self.face_centers = ti.Vector.field(n=3, dtype=ti.f32, shape=self.num_leafs)
        self.zSort_line_idx = ti.field(dtype=ti.uint32, shape=self.num_nodes)

    # Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
    @ti.func
    def expand_bits(self, v):
        v = (v * ti.uint64(0x00010001)) & ti.uint64(0xFF0000FF)
        v = (v * ti.uint64(0x00000101)) & ti.uint64(0x0F00F00F)
        v = (v * ti.uint64(0x00000011)) & ti.uint64(0xC30C30C3)
        v = (v * ti.uint64(0x00000005)) & ti.uint64(0x49249249)
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
        for f in mesh.faces:
            # // no need to set parent to nullptr, each child will have a parent
            id = self.object_ids[f.id]
            self.nodes[id + self.num_leafs].object_id = f.id
            # // needed to recognize that this node is a leaf
            self.nodes[id + self.num_leafs].left = -1
            self.nodes[id + self.num_leafs].right = -1
            self.nodes[id + self.num_leafs].aabb_min = f.aabb_min
            self.nodes[id + self.num_leafs].aabb_max = f.aabb_max

            # // need to set for internal node parent to nullptr, for testing later
            # // there is one less internal node than leaf node, test for that
            # self.internal_nodes[i].parent = None

    @ti.func
    def delta(self, a, b, n, ka):
        # // this guard is for leaf nodes, not internal nodes (hence [0, n-1])
        if (b < 0) or (b > n - 1):
            return -1
        kb = self.morton_codes[self.object_ids[b]]
        if ka == kb:
            # // if keys are equal, use id as fallback
            # // (+32 because they have the same morton code)
            return 32 + ti.math.clz(ti.cast(a, ti.uint32) ^ ti.cast(b, ti.uint32))
        # // clz = count leading zeros
        return ti.math.clz(ka ^ kb)

    @ti.func
    def find_split(self, first, last, n):
        first_code = self.sorted_morton_codes[first]

        # calculate the number of highest bits that are the same
        # for all objects, using the count-leading-zeros intrinsic

        common_prefix = self.delta(first, last, n, first_code)

        # use binary search to find where the next bit differs
        # specifically, we are looking for the highest object that
        # shares more than commonPrefix bits with the first one

        # initial guess
        split = first

        step = last - first

        while step > 1:
            # exponential decrease
            step = (step + 1) >> 1
            # proposed new position
            new_split = split + step

            if new_split < last:
                split_prefix = self.delta(first, new_split, n, first_code)
                if split_prefix > common_prefix:
                    # accept proposal
                    split = new_split

        return split

    @ti.func
    def determine_range(self, n, i) -> ti.math.ivec2:
        ki = self.morton_codes[i]  # key of i

        # determine direction of the range(+1 or -1)

        delta_l = self.delta(i, i - 1, n, ki)
        delta_r = self.delta(i, i + 1, n, ki)

        d = 0  # direction

        # min of delta_r and delta_l
        delta_min = 0
        if delta_r < delta_l:
            d = -1
            delta_min = delta_r
        else:
            d = 1
            delta_min = delta_l

        # compute upper bound of the length of the range
        l_max = 2
        while self.delta(i, i + l_max * d, n, ki) > delta_min:
            l_max <<= 1

        # find other end using binary search
        l = 0
        t = l_max >> 1
        while t > 0:
            if self.delta(i, i + (l + t) * d, n, ki) > delta_min:
                l += t
            t >>= 1

        j = i + l * d

        # // ensure i <= j
        ret = ti.math.ivec2(0)
        if i < j:
            ret = ti.math.ivec2([i, j])
        else:
            ret = ti.math.ivec2([j, i])
        return ret

    @ti.kernel
    def assign_internal_nodes_at_lev(self, num_lev: ti.int32):
        for i in range(num_lev):
            left = 2 * i + self.num_leafs
            right = left + 1
            parent = i + num_lev
            self.nodes[parent].left = left
            self.nodes[parent].right = right
            self.nodes[parent].aabb_min = ti.min(self.nodes[left].aabb_min, self.nodes[right].aabb_min)
            self.nodes[parent].aabb_max = ti.max(self.nodes[left].aabb_max, self.nodes[right].aabb_max)

    def assign_internal_nodes(self):
        level_size = self.num_leafs
        while level_size > 1:
            half_level = level_size // 2
            self.assign_internal_nodes_at_lev(half_level)
            level_size = half_level

    def build(self, mesh, aabb_min_g, aabb_max_g):
        # self.internal_nodes.parent.fill(-1)
        self.assign_morton(mesh, aabb_min_g, aabb_max_g)
        ti.algorithms.parallel_sort(keys=self.morton_codes, values=self.object_ids)
        self.assign_leaf_nodes(mesh)
        self.assign_internal_nodes()
        # self.set_aabb()


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

            self.aabb_indices[12 * n + 0] = 8 * n + 0
            self.aabb_indices[12 * n + 1] = 8 * n + 1
            self.aabb_indices[12 * n + 2] = 8 * n + 1
            self.aabb_indices[12 * n + 3] = 8 * n + 2
            self.aabb_indices[12 * n + 4] = 8 * n + 2
            self.aabb_indices[12 * n + 5] = 8 * n + 3
            self.aabb_indices[12 * n + 6] = 8 * n + 3
            self.aabb_indices[12 * n + 7] = 8 * n + 0
            self.aabb_indices[12 * n + 8] = 8 * n + 4
            self.aabb_indices[12 * n + 9] = 8 * n + 5
            self.aabb_indices[12 * n + 10] = 8 * n + 5
            self.aabb_indices[12 * n + 11] = 8 * n + 6
            self.aabb_indices[12 * n + 12] = 8 * n + 6
            self.aabb_indices[12 * n + 13] = 8 * n + 7
            self.aabb_indices[12 * n + 14] = 8 * n + 7
            self.aabb_indices[12 * n + 15] = 8 * n + 4
            self.aabb_indices[12 * n + 16] = 8 * n + 0
            self.aabb_indices[12 * n + 17] = 8 * n + 4
            self.aabb_indices[12 * n + 18] = 8 * n + 1
            self.aabb_indices[12 * n + 19] = 8 * n + 5
            self.aabb_indices[12 * n + 20] = 8 * n + 2
            self.aabb_indices[12 * n + 21] = 8 * n + 6
            self.aabb_indices[12 * n + 22] = 8 * n + 3
            self.aabb_indices[12 * n + 23] = 8 * n + 7

    def draw_bvh_aabb(self, scene):
        self.update_aabb_x_and_lines()
        scene.lines(self.aabb_x, indices=self.aabb_indices, width=1.0, color=(0, 0, 0))