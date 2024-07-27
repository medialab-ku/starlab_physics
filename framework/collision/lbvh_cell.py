import taichi as ti
from framework.utilities.BlellochScan import BlellochScan
@ti.dataclass
class Node:
    object_id: ti.i32
    num_faces: ti.i32
    parent: ti.i32
    child_a: ti.i32
    child_b: ti.i32
    visited: ti.i32
    aabb_min: ti.math.vec3
    aabb_max: ti.math.vec3
    range_l: ti.i32
    range_r: ti.i32

@ti.data_oriented
class LBVH_CELL:
    def __init__(self, num_leafs):

        self.grid_res = ti.math.ivec3(0)
        self.grid_res[0] = 32
        self.grid_res[1] = 32
        self.grid_res[2] = 32
        self.cell_size = ti.math.vec3(0)
        self.origin = ti.math.vec3(0)

        self.num_cells = self.grid_res[0] * self.grid_res[1] * self.grid_res[2]
        print("# cells: ", self.num_cells)
        self.cell_centers = ti.Vector.field(n=3, dtype=ti.f32, shape=self.num_cells)
        self.cell_ids = ti.field(dtype=ti.i32, shape=self.num_cells)
        self.cell_morton_codes = ti.field(dtype=ti.int32, shape=self.num_cells)
        self.cell_nodes = Node.field(shape=(2 * self.num_cells - 1))

        self.cell_ids_sorted = ti.field(dtype=ti.i32, shape=self.num_cells)
        self.cell_morton_codes_sorted = ti.field(dtype=ti.i32, shape=self.num_cells)
        self.num_faces_in_cell = ti.field(dtype=ti.i32, shape=self.num_cells)
        self.prefix_sum_cell = ti.field(dtype=ti.i32, shape=self.num_cells)
        self.prefix_sum_cell_temp = ti.field(dtype=ti.i32, shape=self.num_cells)

        self.prefix_sum_executer_cell_Blelloch = BlellochScan(self.num_cells)
        self.prefix_sum_executer_cell = ti.algorithms.PrefixSumExecutor(self.num_cells)
        # self.prefix_sum_executer_cell = ti.algorithms.PrefixSumExecutor(self.num_cells)

        self.num_leafs = num_leafs
        self.face_ids = ti.field(dtype=ti.i32, shape=self.num_leafs)
        self.face_cell_ids = ti.field(dtype=ti.i32, shape=self.num_leafs)
        self.sorted_face_ids = ti.field(dtype=ti.i32, shape=self.num_leafs)
        self.sorted_face_cell_ids = ti.field(dtype=ti.i32, shape=self.num_leafs)
        self.sorted_to_origin_face_ids = ti.field(int, shape=self.num_leafs)

        self.face_aabb_min = ti.Vector.field(n=3, dtype=ti.f32, shape=self.num_leafs)
        self.face_aabb_max = ti.Vector.field(n=3, dtype=ti.f32, shape=self.num_leafs)

        # print("# leafs", num_leafs)
        self.leaf_offset = self.num_leafs - 1
        self.num_nodes = 2 * self.num_leafs - 1
        self.root = -1
        self.test = 1
        self.aabb_x = ti.Vector.field(n=3, dtype=ti.f32, shape=8 * self.test)
        self.aabb_x0 = ti.Vector.field(n=3, dtype=ti.f32, shape=8)

        self.aabb_indices = ti.field(dtype=ti.uint32, shape=24 * self.test)
        self.aabb_index0 = ti.field(dtype=ti.uint32, shape=24)
        # self.aabb_index1 = ti.field(dtype=ti.uint32, shape=24)

        self.zSort_line_idx = ti.field(dtype=ti.uint32, shape=self.num_nodes)
        self.zSort_line_idx_cells = ti.field(dtype=ti.uint32, shape=2 * self.num_cells)
        self.parent_ids = ti.field(dtype=ti.i32, shape=self.num_leafs)

        self.BITS_PER_PASS = 6
        self.RADIX = pow(2, self.BITS_PER_PASS)
        self.passes = (30 + self.BITS_PER_PASS - 1) // self.BITS_PER_PASS
        self.prefix_sum_executer = ti.algorithms.PrefixSumExecutor(self.RADIX)
        self.prefix_sum = ti.field(dtype=ti.i32, shape=self.RADIX)
        self.prefix_sum_temp = ti.field(dtype=ti.i32, shape=self.RADIX)


        self.cell_size = self.assign_cell_morton(ti.math.vec3(0.0), ti.math.vec3(1e3))
        self.radix_sort_cells()
        self.assign_internal_nodes_Karras12_cells()




    # Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
    @ti.func
    def expand_bits(self, v):
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8)) & 0x0300F00F
        v = (v | (v << 4)) & 0x030C30C3
        v = (v | (v << 2)) & 0x09249249
        return v

    @ti.func
    def morton_3d(self, x, y, z):
        x = ti.math.clamp(x * 1024., 0., 1023.)
        y = ti.math.clamp(y * 1024., 0., 1023.)
        z = ti.math.clamp(z * 1024., 0., 1023.)
        xx = self.expand_bits(ti.cast(x, ti.uint64))
        yy = self.expand_bits(ti.cast(y, ti.uint64))
        zz = self.expand_bits(ti.cast(z, ti.uint64))
        return ti.cast(xx | (yy << 1) | (zz << 2), ti.int32)

    @ti.kernel
    def assign_cell_centers(self, aabb_min: ti.math.vec3, aabb_max: ti.math.vec3) -> ti.math.vec3:

        cell_size = ti.math.vec3(0.)
        cell_size[0] = (aabb_max[0] - aabb_min[0]) / self.grid_res[0]
        cell_size[1] = (aabb_max[1] - aabb_min[1]) / self.grid_res[1]
        cell_size[2] = (aabb_max[2] - aabb_min[2]) / self.grid_res[2]

        for i in range(self.num_cells):
            x0 = i // (self.grid_res[1] * self.grid_res[2])
            y0 = (i % (self.grid_res[1] * self.grid_res[2])) // self.grid_res[2]
            z0 = i % self.grid_res[2]

            pos = ti.math.vec3(cell_size[0] * x0 + 0.5 * cell_size[0], cell_size[1] * y0 + 0.5 * cell_size[1], cell_size[2] * z0 + 0.5 * cell_size[2]) + aabb_min
            self.cell_centers[i] = pos

        return cell_size

    @ti.kernel
    def assign_cell_morton(self, aabb_min: ti.math.vec3, aabb_max: ti.math.vec3) -> ti.math.vec3:

        cell_size = ti.math.vec3(0.)
        cell_size[0] = (aabb_max[0] - aabb_min[0]) / self.grid_res[0]
        cell_size[1] = (aabb_max[1] - aabb_min[1]) / self.grid_res[1]
        cell_size[2] = (aabb_max[2] - aabb_min[2]) / self.grid_res[2]
        # cell_size = 0.5 * (cell_size_x + cell_size_y + cell_size_z) / 3.0

        for i in range(self.num_cells):
            x0 = i // (self.grid_res[1] * self.grid_res[2])
            y0 = (i % (self.grid_res[1] * self.grid_res[2])) // self.grid_res[2]
            z0 = i % self.grid_res[2]

            pos = ti.math.vec3(cell_size[0] * x0 + 0.5 * cell_size[0], cell_size[1] * y0 + 0.5 * cell_size[1], cell_size[2] * z0 + 0.5 * cell_size[2]) + aabb_min
            self.cell_centers[i] = pos

            x = (pos[0] - aabb_min[0]) / (aabb_max[0] - aabb_min[0])
            y = (pos[1] - aabb_min[1]) / (aabb_max[1] - aabb_min[1])
            z = (pos[2] - aabb_min[2]) / (aabb_max[2] - aabb_min[2])

            # // clamp to deal with numeric issues
            x = ti.math.clamp(x, 0., 1.)
            y = ti.math.clamp(y, 0., 1.)
            z = ti.math.clamp(z, 0., 1.)

            morton3d = self.morton_3d(x, y, z)
            self.cell_morton_codes[i] = morton3d
            self.cell_ids[i] = i

        return cell_size
    @ti.kernel
    def assign_morton(self, mesh: ti.template(), aabb_min: ti.math.vec3, aabb_max: ti.math.vec3) -> ti.i32:

        # max_value = -1
        # min0 = ti.math.vec3(1e4)
        # max0 = ti.math.vec3(-1e4)
        cnt = 0
        # aabb_min0 = ti.math.vec3(0.0)
        # aabb_max0 = ti.math.vec3(self.res_x * self.cell_size, self.res_y * self.cell_size, self.res_z * self.cell_size)

        cell_size_x = (aabb_max[0] - aabb_min[0]) / self.grid_res[0]
        cell_size_y = (aabb_max[1] - aabb_min[1]) / self.grid_res[1]
        cell_size_z = (aabb_max[2] - aabb_min[2]) / self.grid_res[2]

        cell_size = 0.5 * (cell_size_x + cell_size_y + cell_size_z) / 3.0

        for i in range(self.num_leafs):
            x0 = i // (self.grid_res[1] * self.grid_res[2])
            y0 = (i % (self.grid_res[1] * self.grid_res[2])) // self.grid_res[2]
            z0 = i % self.grid_res[2]

            pos = ti.math.vec3(cell_size_x * x0 + 0.5 * cell_size_x, cell_size_y * y0 + 0.5 * cell_size_y, cell_size_z * z0 + 0.5 * cell_size_z) + aabb_min
            self.face_centers[i] = pos

            x = (pos[0] - aabb_min[0]) / (aabb_max[0] - aabb_min[0])
            y = (pos[1] - aabb_min[1]) / (aabb_max[1] - aabb_min[1])
            z = (pos[2] - aabb_min[2]) / (aabb_max[2] - aabb_min[2])

                # // clamp to deal with numeric issues
            x = ti.math.clamp(x, 0., 1.)
            y = ti.math.clamp(y, 0., 1.)
            z = ti.math.clamp(z, 0., 1.)

            # // obtain and set morton code based on normalized position
            morton3d = self.morton_3d(x, y, z)
            # if morton3d < 0:
            #     cnt += 1
            self.morton_codes[i] = morton3d
            # ti.atomic_max(max_value, morton3d)
            self.object_ids[i] = i

        return cell_size

    #     for f in mesh.faces:
    #     # # // obtain center of triangle
    #     #     u = f.verts[0]
    #     #     v = f.verts[1]
    #     #     w = f.verts[2]
    #     #     pos = (1. / 3.) * (u.x + v.x + w.x)
    #
    #         pos = 0.5 * (f.aabb_max + f.aabb_min)
    #
    #         # if f.id < 10:
    #         #     print(pos[1])
    #         # pos[1] = 0.0
    #         # pos = ti.math.vec3(x, y, z)
    #         self.face_centers[f.id] = pos
    #         #
    #         # ti.atomic_max(max0, pos)
    #         # ti.atomic_min(min0, pos)
    #
    #     # for f in mesh.faces:
    #     #     pos = self.face_centers[f.id]
    #             # = 0.5 * (f.aabb_min + f.aabb_max)
    #     # // normalize position
    #         x = (pos[0] - aabb_min[0]) / (aabb_max[0] - aabb_min[0])
    #         y = (pos[1] - aabb_min[1]) / (aabb_max[1] - aabb_min[1])
    #         z = (pos[2] - aabb_min[2]) / (aabb_max[2] - aabb_min[2])
    #     # // clamp to deal with numeric issues
    #         x = ti.math.clamp(x, 0., 1.)
    #         y = ti.math.clamp(y, 0., 1.)
    #         z = ti.math.clamp(z, 0., 1.)
    #
    # # // obtain and set morton code based on normalized position
    #         morton3d = self.morton_3d(x, y, z)
    #         # if morton3d < 0:
    #         #     cnt += 1
    #         self.morton_codes[f.id] = morton3d
    #         # ti.atomic_max(max_value, morton3d)
    #         self.object_ids[f.id] = f.id
        # print(cnt)

        # return max_value

    @ti.kernel
    def assign_leaf_nodes(self, mesh: ti.template()):
        # print(self.num_leafs)
        for f in mesh.faces:
            # // no need to set parent to nullptr, each child will have a parents
            id = self.object_ids[f.id]
            self.nodes[f.id + self.num_leafs - 1].object_id = id
            self.nodes[f.id + self.num_leafs - 1].child_a = -1
            self.nodes[f.id + self.num_leafs - 1].child_b = -1
            self.nodes[f.id + self.num_leafs - 1].aabb_min = mesh.faces.aabb_min[id]
            self.nodes[f.id + self.num_leafs - 1].aabb_max = mesh.faces.aabb_max[id]

            # // need to set for internal node parent to nullptr, for testing later
            # // there is one less internal node than leaf node, test for that
            # self.internal_nodes[i].parent = None



    @ti.func
    def delta(self, i, j, num_leafs, morton_codes):
        ret = -1
        if j <= (num_leafs - 1) and j >= 0:
            xor = morton_codes[i] ^ morton_codes[j]
            if xor == 0:
                ret = 32
            else:
                ret = ti.math.clz(xor)
        return ret

    @ti.func
    def delta_Apetrei(self, i):
        # xor = 32
        xor = self.morton_codes[i] ^ self.morton_codes[i + 1]
        return xor

    @ti.func
    def find_split(self, l, r, morton_codes):
        first_code = morton_codes[l]
        last_code = morton_codes[r]

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
                    split_code = morton_codes[new_split]
                    split_prefix = ti.math.clz(first_code ^ split_code)
                    if split_prefix > common_prefix:
                        split = new_split

            ret = split
        return ret


    @ti.func
    def determine_range(self, i, n, morton_codes):

        delta_l = self.delta(i, i - 1, n, morton_codes)
        delta_r = self.delta(i, i + 1, n, morton_codes)

        d = 1

        delta_min = delta_l
        if delta_r < delta_l:
            d = - 1
            delta_min = delta_r

        # print(d)
        l_max = 2
        while self.delta(i, i + l_max * d, n, morton_codes) > delta_min:
            l_max <<= 2

        l = 0
        t = l_max // 2
        while t > 0:
            delta = -1
            if i + (l + t) * d >= 0 and i + (l + t) * d < n:
                delta = self.delta(i, i + (l + t) * d, n, morton_codes)

            if delta > delta_min:
                l += t
            t //= 2

        start = i
        end = i + l * d
        if d == -1:
            start = i + l * d
            end = i

        return start, end

    @ti.func
    def determine_range_test(self, i, n):

        delta_l = self.delta(i, i - 1)
        delta_r = self.delta(i, i + 1)

        d = 1

        delta_min = delta_l
        if delta_r < delta_l:
            d = - 1
            delta_min = delta_r

        delta_min = self.delta(i, i - d)

        print("delta_l: ", delta_l)
        print("delta_r: ", delta_r)
        print("d: ", d)
        # print(d)
        l_max = 2
        while self.delta(i, i + l_max * d) > delta_min:
            l_max <<= 2

        print("l_max: ", l_max)
        l = 0
        t = l_max // 2
        print("t: ", t)
        while t > 0:
            a = i + (l + t) * d
            b = self.delta(i, a)
            print("a: ", a)
            print("b: ", b)
            if a >= 0 and a < n and b > delta_min:
                l += t
            t //= 2
        print("l: ", l)
        start = i
        end = i + l * d
        if d == -1:
            start = i + l * d
            end = i

        return start, end


    @ti.kernel
    def assign_internal_nodes_Karras12(self):

        # ti.loop_config(block_dim=64)
        cnt = 0
        for i in range(self.num_leafs - 1):
            start, end = self.determine_range(i, self.num_leafs, self.morton_codes)
            split = self.find_split(start, end, self.morton_codes)
            left = split
            if split == start:
                left += self.leaf_offset

            right = split + 1
            if right == end:
                right += self.leaf_offset

            self.nodes[i].child_a = left
            self.nodes[i].child_b = right
            self.nodes[i].visited = 0
            self.nodes[left].parent = i
            self.nodes[right].parent = i
            self.nodes[i].range_l = start
            self.nodes[i].range_r = end

    @ti.kernel
    def assign_internal_nodes_Karras12_cells(self):

        # ti.loop_config(block_dim=64)
        cnt = 0
        for i in range(self.num_cells - 1):
            start, end = self.determine_range(i, self.num_cells, self.cell_morton_codes)
            split = self.find_split(start, end, self.cell_morton_codes)

            left = split
            if split == start:
                left += (self.num_cells - 1)

            right = split + 1
            if right == end:
                right += (self.num_cells - 1)

            self.cell_nodes[i].child_a = left
            self.cell_nodes[i].child_b = right
            self.cell_nodes[i].visited = 0
            self.cell_nodes[left].parent = i
            self.cell_nodes[right].parent = i
            self.cell_nodes[i].range_l = start
            self.cell_nodes[i].range_r = end

        # for i in range(self.num_cells - 1):
        #     print(i, self.cell_nodes[i].child_a, self.cell_nodes[i].child_b)

    @ti.func
    def choose_parent(self, left, right, current_node):

        if left == 0 or (right != self.num_leafs - 1 and self.delta_Apetrei(right) < self.delta_Apetrei(left - 1)):
            parent = right
            self.nodes[parent].child_a = current_node
            self.nodes[parent].range_l = left
        else:
            parent = left - 1
            self.nodes[parent].child_b = current_node
            self.nodes[parent].range_r = right

        return parent

    @ti.kernel
    def init_flag(self):
        for i in range(self.num_leafs):
            parent = self.nodes[i + self.num_leafs - 1].parent
            self.nodes[parent].visited += 1


    @ti.kernel
    def init_flag_cells(self):
        for i in range(self.num_cells):
            parent = self.cell_nodes[i + self.num_cells - 1].parent
            self.cell_nodes[parent].visited += 1





    @ti.kernel
    def init_flag(self):
        for i in range(self.num_leafs):
            parent = self.nodes[i + self.num_leafs - 1].parent
            self.nodes[parent].visited += 1

    def compute_bvh_aabbs(self):

        self.init_flag()
        while True:
            cnt = self.compute_node_aabbs()
            # print("cnt: ", cnt)
            if cnt == 0:
                break

    def compute_bvh_aabbs_cells(self):

        # self.init_flag_cells()
        while True:
            cnt = self.compute_cell_node_aabbs()
            # print("cnt: ", cnt)
            if cnt == 0:
                break

    @ti.kernel
    def compute_cell_node_aabbs(self) -> ti.int32:
        cnt = 0
        for i in range(self.num_cells - 1):
            if self.cell_nodes[i].visited == 2:
                left, right = self.cell_nodes[i].child_a, self.cell_nodes[i].child_b
                min0, min1 = self.cell_nodes[left].aabb_min, self.cell_nodes[right].aabb_min
                max0, max1 = self.cell_nodes[left].aabb_max, self.cell_nodes[right].aabb_max

                self.cell_nodes[i].num_faces = self.cell_nodes[left].num_faces + self.cell_nodes[right].num_faces
                if self.cell_nodes[left].num_faces > 0 and self.cell_nodes[right].num_faces > 0:
                    self.cell_nodes[i].aabb_min = ti.min(min0, min1)
                    self.cell_nodes[i].aabb_max = ti.max(max0, max1)

                elif self.cell_nodes[left].num_faces > 0:
                    self.cell_nodes[i].aabb_min = min0
                    self.cell_nodes[i].aabb_max = max0

                elif self.cell_nodes[right].num_faces > 0:
                    self.cell_nodes[i].aabb_min = min1
                    self.cell_nodes[i].aabb_max = max1

                else:
                    center = 0.25 * (min0 + max0 + min1 + max1)
                    self.cell_nodes[i].aabb_min = center
                    self.cell_nodes[i].aabb_max = center

                parent = self.cell_nodes[i].parent
                self.cell_nodes[i].visited += 1
                self.cell_nodes[parent].visited += 1
                ti.atomic_add(cnt, 1)

        return cnt

    @ti.kernel
    def count_frequency_cells(self, pass_num: ti.i32):
        for i in range(self.num_cells):
            mc_i = self.cell_morton_codes[i]
            digit = (mc_i >> (pass_num * self.BITS_PER_PASS)) & (self.RADIX - 1)
            ti.atomic_add(self.prefix_sum[digit], 1)


    @ti.kernel
    def sort_by_digit_cells(self, pass_num: ti.i32):

        ti.loop_config(serialize=True)
        for i in range(self.num_cells):
            I = self.num_cells - 1 - i
            mc_i = self.cell_morton_codes[I]
            digit = (mc_i >> (pass_num * self.BITS_PER_PASS)) & (self.RADIX - 1)
            idx = self.prefix_sum[digit] - 1
            # if idx >= 0:

            # I = self.num_leafs - 1 - fid
            # cell_id = self.face_cell_ids[I]
            # idx = ti.atomic_sub(self.prefix_sum_cell_temp[cell_id], 1) - 1
            # self.sorted_face_ids[idx] = self.face_ids[I]

            self.cell_ids_sorted[idx] = self.cell_ids[I]
            self.cell_morton_codes_sorted[idx] = self.cell_morton_codes[I]
            self.prefix_sum[digit] -= 1


    def radix_sort_cells(self):
        # print(self.passes)
        for pi in range(self.passes):
            self.prefix_sum.fill(0)
            self.count_frequency_cells(pi)
            self.prefix_sum_executer.run(self.prefix_sum)
            # self.blelloch_scan()
            self.sort_by_digit_cells(pi)
            self.cell_morton_codes.copy_from(self.cell_morton_codes_sorted)
            self.cell_ids.copy_from(self.cell_ids_sorted)

    @ti.func
    def pos_to_idx3d(self, pos: ti.math.vec3, grid_size: ti.math.vec3, origin: ti.math.vec3):
        return ((pos - origin) / grid_size).cast(int)

    @ti.func
    def flatten_cell_id(self, cell_id):
        return cell_id[0] * self.grid_res[1] * self.grid_res[2] + cell_id[1] * self.grid_res[2] + cell_id[2]

    @ti.func
    def get_flatten_cell_id(self, pos: ti.math.vec3, grid_size: ti.math.vec3, origin: ti.math.vec3):
        return self.flatten_cell_id(self.pos_to_idx3d(pos, grid_size, origin))

    @ti.kernel
    def assign_face_cell_ids(self, mesh: ti.template(), cell_size: ti.math.vec3, origin: ti.math.vec3):

        for f in mesh.faces:
            pos = 0.5 * (f.aabb_min + f.aabb_max)
            cell_id = self.get_flatten_cell_id(pos, cell_size, origin)
            self.face_aabb_min[f.id] = f.aabb_min
            self.face_aabb_max[f.id] = f.aabb_max
            self.face_ids[f.id] = f.id
            self.face_cell_ids[f.id] = cell_id
            ti.atomic_add(self.prefix_sum_cell[cell_id], 1)

    @ti.kernel
    def counting_sort_cells(self):

        # ti.loop_config(serialize=True)

        ti.loop_config(block_dim=64, block_dim_adaptive=True)
        for fid in range(self.num_leafs):
            I = self.num_leafs - 1 - fid
            cell_id = self.face_cell_ids[I]
            idx = ti.atomic_sub(self.prefix_sum_cell_temp[cell_id], 1) - 1
            self.sorted_face_ids[idx] = self.face_ids[I]
            # self.sorted_face_cell_ids[idx] = cell_id

    @ti.kernel
    def assign_leaf_cell_nodes(self, mesh: ti.template()):

        for i in range(self.num_cells):
            # // no need to set parent to nullptr, each child will have a parents
            if i == 0:
                self.cell_nodes[i + self.num_cells - 1].range_l = 0

            else:
                self.cell_nodes[i + self.num_cells - 1].range_l = self.prefix_sum_cell[i - 1]

            self.cell_nodes[i + self.num_cells - 1].range_r = self.prefix_sum_cell[i] - 1
            self.cell_nodes[i + self.num_cells - 1].child_a = -1
            self.cell_nodes[i + self.num_cells - 1].child_b = -1

            size = self.cell_nodes[i + self.num_cells - 1].range_r - self.cell_nodes[i + self.num_cells - 1].range_l + 1
            offset = self.cell_nodes[i + self.num_cells - 1].range_l

            aabb_min = self.cell_centers[i]
            aabb_max = self.cell_centers[i]

            self.cell_nodes[i + self.num_cells - 1].num_faces = size

            for j in range(size):
                fid = self.sorted_face_ids[j + offset]
                # print(mesh.faces.aabb_min[fid],  mesh.faces.aabb_max[fid])
                aabb_min = ti.math.min(aabb_min, mesh.faces.aabb_min[fid])
                aabb_max = ti.math.max(aabb_max, mesh.faces.aabb_max[fid])

            self.cell_nodes[i + self.num_cells - 1].aabb_min = aabb_min
            self.cell_nodes[i + self.num_cells - 1].aabb_max = aabb_max

            parent = self.cell_nodes[i + self.num_cells - 1].parent
            self.cell_nodes[parent].visited += 1


    def build(self, mesh, aabb_min_g, aabb_max_g):
        self.origin = aabb_min_g
        self.cell_size = self.assign_cell_centers(aabb_min_g, aabb_max_g)
        self.prefix_sum_cell.fill(0)
        self.assign_face_cell_ids(mesh, self.cell_size, aabb_min_g)

        # self.prefix_sum_executer_cell_Blelloch.run(self.prefix_sum_cell)
        self.prefix_sum_executer_cell.run(self.prefix_sum_cell)
        self.prefix_sum_cell_temp.copy_from(self.prefix_sum_cell)
        self.counting_sort_cells()
        if self.prefix_sum_cell[self.num_cells - 1] != self.num_leafs:
            print("[abort]: self.prefix_sum_cell[self.num_cells - 1] != self.num_leafs")

        self.cell_nodes.visited.fill(0)
        self.cell_nodes.num_faces.fill(0)
        self.assign_leaf_cell_nodes(mesh)
        self.compute_bvh_aabbs_cells()

    @ti.func
    def aabb_overlap(self, min1, max1, min2, max2):
        return (min1[0] <= max2[0] and max1[0] >= min2[0] and
                min1[1] <= max2[1] and max1[1] >= min2[1] and
                min1[2] <= max2[2] and max1[2] >= min2[2])

    @ti.func
    def traverse_cell_bvh_single(self, cell_size, origin, min0, max0, i, cache, nums):
        pos = 0.5 * (min0 + max0)
        cell_id = self.get_flatten_cell_id(pos, cell_size, origin)
        if cell_id <= self.num_cells - 1:
            root = 0
            # root = self.cell_nodes[cell_id + self.num_cells - 1].parent
            # while True:
            #
            #     if root == 0: break
            #
            #     left, right = self.cell_nodes[root].child_a, self.cell_nodes[root].child_b
            #     min_l, max_l = self.cell_nodes[left].aabb_min, self.cell_nodes[left].aabb_max
            #     min_r, max_r = self.cell_nodes[right].aabb_min, self.cell_nodes[right].aabb_max
            #
            #     if self.aabb_overlap(min_l, max_l, min_r, max_r):
            #         root = self.cell_nodes[root].parent
            #
            #     else: break
            # root = self.cell_nodes[cell_id + self.num_cells - 1].parent
            # root = self.cell_nodes[root].parent
            stack = ti.Vector([-1 for j in range(32)])
            stack[0] = root
            stack_counter = 1
            while stack_counter > 0:
                # print(stack)
                stack_counter -= 1
                idx = stack[stack_counter]
                min1, max1 = self.cell_nodes[idx].aabb_min, self.cell_nodes[idx].aabb_max
                # print(min1, max1)
                if self.cell_nodes[idx].num_faces > 0:
                    if self.aabb_overlap(min0, max0, min1, max1):
                        if idx >= self.num_cells - 1:
                            size = self.cell_nodes[idx].range_r - self.cell_nodes[idx].range_l + 1
                            offset = self.cell_nodes[idx].range_l
                            for j in range(size):
                                fid = self.sorted_face_ids[j + offset]
                                # print(mesh.faces.aabb_min[fid],  mesh.faces.aabb_max[fid])
                                aabb_min = self.face_aabb_min[fid]
                                aabb_max = self.face_aabb_max[fid]
                                if self.aabb_overlap(min0, max0, aabb_min, aabb_max):
                                    cache[i, nums[i]] = fid
                                    nums[i] += 1

                        else:
                            left, right = self.cell_nodes[idx].child_a, self.cell_nodes[idx].child_b

                            if self.cell_nodes[left].num_faces > 0:
                                stack[stack_counter] = left
                                stack_counter += 1

                            if self.cell_nodes[right].num_faces > 0:
                                stack[stack_counter] = right
                                stack_counter += 1

    @ti.kernel
    def update_zSort_cell_centers_and_line(self):
        for i in range(self.num_cells - 1):
            self.zSort_line_idx_cells[2 * i + 0] = self.cell_ids[i]
            self.zSort_line_idx_cells[2 * i + 1] = self.cell_ids[i + 1]

    def draw_zSort(self, scene):
        # self.update_zSort_face_centers_and_line()
        self.update_zSort_cell_centers_and_line()
        scene.lines(self.cell_centers, indices=self.zSort_line_idx_cells, width=1.0, color=(1, 0, 0))
        # scene.particles(self.face_centers, radius=self.cell_size, color=(0, 1, 0))


    @ti.kernel
    def update_cell_aabb_x_and_line0(self, n: ti.i32):

        aabb_min = self.cell_nodes[n].aabb_min
        aabb_max = self.cell_nodes[n].aabb_max


        self.aabb_x0[0] = ti.math.vec3(aabb_max[0], aabb_max[1], aabb_max[2])
        self.aabb_x0[1] = ti.math.vec3(aabb_min[0], aabb_max[1], aabb_max[2])
        self.aabb_x0[2] = ti.math.vec3(aabb_min[0], aabb_max[1], aabb_min[2])
        self.aabb_x0[3] = ti.math.vec3(aabb_max[0], aabb_max[1], aabb_min[2])

        self.aabb_x0[4] = ti.math.vec3(aabb_max[0], aabb_min[1], aabb_max[2])
        self.aabb_x0[5] = ti.math.vec3(aabb_min[0], aabb_min[1], aabb_max[2])
        self.aabb_x0[6] = ti.math.vec3(aabb_min[0], aabb_min[1], aabb_min[2])
        self.aabb_x0[7] = ti.math.vec3(aabb_max[0], aabb_min[1], aabb_min[2])

        self.aabb_index0[0] = 0
        self.aabb_index0[1] = 1
        self.aabb_index0[2] = 1
        self.aabb_index0[3] = 2
        self.aabb_index0[4] = 2
        self.aabb_index0[5] = 3
        self.aabb_index0[6] = 3
        self.aabb_index0[7] = 0
        self.aabb_index0[8] = 4
        self.aabb_index0[9] = 5
        self.aabb_index0[10] = 5
        self.aabb_index0[11] = 6
        self.aabb_index0[12] = 6
        self.aabb_index0[13] = 7
        self.aabb_index0[14] = 7
        self.aabb_index0[15] = 4
        self.aabb_index0[16] = 0
        self.aabb_index0[17] = 4
        self.aabb_index0[18] = 1
        self.aabb_index0[19] = 5
        self.aabb_index0[20] = 2
        self.aabb_index0[21] = 6
        self.aabb_index0[22] = 3
        self.aabb_index0[23] = 7


    def draw_bvh_cell_aabb_test(self, scene, n_leaf, n_internal):
        n_leaf += (self.num_cells - 1)
        self.update_cell_aabb_x_and_line0(n_leaf)
        scene.lines(self.aabb_x0, indices=self.aabb_index0, width=2.0, color=(0, 1, 0))
        # pos = self.face_centers[n_leaf]
        # scene.particles(, indices=self.aabb_index0, width=2.0, color=(0, 1, 0))

        self.update_cell_aabb_x_and_line0(n_internal)
        scene.lines(self.aabb_x0, indices=self.aabb_index0, width=2.0, color=(0, 0, 1))

        # left, right = self.nodes[n_internal].child_a, self.nodes[n_internal].child_b
        # self.update_aabb_x_and_line0(left)
        # scene.lines(self.aabb_x0, indices=self.aabb_index0, width=2.0, color=(1, 0, 0))
        #
        # self.update_aabb_x_and_line0(right)
        # scene.lines(self.aabb_x0, indices=self.aabb_index0, width=2.0, color=(0, 0, 1))