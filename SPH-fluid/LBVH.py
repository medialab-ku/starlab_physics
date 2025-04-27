import taichi as ti
import cupy as cp
import numpy as np
from trimesh.exchange.gltf import uint32


@ti.dataclass
class AABB:
    _min: ti.math.vec3
    _max: ti.math.vec3

@ti.dataclass
class Node:
    _id: ti.i32
    parent: ti.i32
    left: ti.i32
    right: ti.i32
    visited: ti.i32
    _min: ti.math.vec3
    _max: ti.math.vec3
    range_l: ti.i32
    range_r: ti.i32
    a: ti.i32

@ti.data_oriented
class LBVH:
    def __init__(self, num_leafs):

        print(num_leafs)

        self.num_leafs = num_leafs
        self.leaf_offset = self.num_leafs - 1
        self.num_nodes = 2 * self.num_leafs - 1
        self.root = -1
        self.nodes = Node.field(shape=self.num_nodes)
        self.aabb = AABB.field(shape=self.num_nodes)
        self.size = 5
        self.bits_per_pass = 8
        self._range = pow(2, self.bits_per_pass)
        # print(self._range)
        self.prefix_sum_executer = ti.algorithms.PrefixSumExecutor(self._range)
        self.count     = ti.field(dtype=int, shape=self._range)
        self.count_tmp = ti.field(dtype=int, shape=self._range)

        self.morton_code = ti.field(dtype=ti.uint32, shape=self.num_leafs)
        self.morton_code_bef = ti.field(dtype=ti.uint32, shape=self.num_leafs)

        self.keys            = ti.field(dtype=ti.uint32, shape=self.num_leafs)
        self.morton_code_buf = ti.field(dtype=ti.uint32, shape=self.num_leafs)
        self._ids         = ti.field(dtype=ti.uint32, shape=self.num_leafs)
        self._ids_buf     = ti.field(dtype=ti.uint32, shape=self.num_leafs)
        self._ids_new     = ti.field(dtype=ti.uint32, shape=self.num_leafs)

        self.pos     = ti.Vector.field(n=3, dtype=float, shape=self.num_leafs)
        self.pos_buf = ti.Vector.field(n=3, dtype=float, shape=self.num_leafs)


        #for debugging
        self.code_edge = ti.field(dtype=ti.uint32, shape=2 * (self.num_leafs - 1))

        self.test = 1
        self.aabb_x = ti.Vector.field(n=3, dtype=ti.f32, shape=8 * self.test)
        self.aabb_x0 = ti.Vector.field(n=3, dtype=ti.f32, shape=8)
        self.aabb_indices = ti.field(dtype=ti.uint32, shape=24 * self.test)
        self.aabb_index0 = ti.field(dtype=ti.uint32, shape=24)


    @ti.kernel
    def update_count(self, pass_idx: int):
        for m in range(self.num_leafs):
            digit = ti.cast((self.morton_code[m] >> (8 * pass_idx)) & 0xFF, ti.int32)
            ti.atomic_add(self.count[digit], 1)

    @ti.kernel
    def count_sort(self, pass_id: int):

        ti.loop_config(serialize=True)
        for m in range(self.num_leafs):
            I = self.num_leafs - 1 - m
            digit = ti.cast((self.morton_code[I] >> (8 * pass_id)) & 0xFF, ti.int32)
            a = ti.atomic_sub(self.count[digit], 1)
            self._ids_new[I] = a - 1

        for m in range(self.num_leafs):
            new_id = ti.cast(self._ids_new[m], ti.int32)

            self._ids_buf[new_id] = self._ids[m]
            self.morton_code_buf[new_id] = self.morton_code[m]
            self.pos_buf[new_id] = self.pos[m]

    @ti.kernel
    def init_id(self):

        for m in range(self.num_leafs):
            self._ids[m] = m

    def radix_sort(self):

        self.init_id()

        for i in range(4):
            self.count.fill(0)
            self.update_count(i)
            self.count_tmp.copy_from(self.count)
            self.prefix_sum_executer.run(self.count)
            self.count_sort(i)

            self.morton_code.copy_from(self.morton_code_buf)
            self.pos.copy_from(self.pos_buf)
            self._ids.copy_from(self._ids_buf)

        # for i in range(self.num_leafs - 1):
        #     if self.morton_code[i] > self.morton_code[i + 1]:
        #         d1 = ti.cast((self.morton_code[i] >> (8 * 3)) & 0xFF, ti.int32)
        #         d2 = ti.cast((self.morton_code[i + 1] >> (8 * 3)) & 0xFF, ti.int32)
        #         print(d1, d2)


    @ti.kernel
    def assign_leaf_nodes(self):

        for i in range(self.num_leafs):
            self.nodes[i + self.leaf_offset]._id = self._ids_new[i]
            self.nodes[i + self.leaf_offset].aabb._min = self.aabb[i]._min
            self.nodes[i + self.leaf_offset].aabb._max = self.aabb[i]._min
            self.nodes[i + self.leaf_offset]._a = -1
            self.nodes[i + self.leaf_offset]._b = -1

    @ti.func
    def expand_bits(self, v):
        v = (v | (v << 16)) & 0x030000FF
        v = (v | (v << 8))  & 0x0300F00F
        v = (v | (v << 4))  & 0x030C30C3
        v = (v | (v << 2))  & 0x09249249
        return v

    @ti.func
    def morton_3d(self, x, y, z):
        x = ti.math.clamp(x * 1024., 0., 1023.)
        y = ti.math.clamp(y * 1024., 0., 1023.)
        z = ti.math.clamp(z * 1024., 0., 1023.)
        xx = self.expand_bits(ti.cast(x, ti.uint32))
        yy = self.expand_bits(ti.cast(y, ti.uint32))
        zz = self.expand_bits(ti.cast(z, ti.uint32))
        return ti.cast(xx << 2 | (yy << 1) | zz, ti.uint32)

    @ti.kernel
    def assign_morton(self, x: ti.template(), primitives: ti.template(), pad: float):
        # print()

        dim = primitives.shape[0] // self.num_leafs

        _min_g = ti.math.vec3(1e5)
        _max_g = -ti.math.vec3(1e5)
        padding = ti.math.vec3(pad)

        for i in range(self.num_leafs):
            for j in range(dim):
                ti.atomic_min(_min_g, x[primitives[dim * i + j]])
                ti.atomic_max(_max_g, x[primitives[dim * i + j]])

        _min_g -= padding
        _max_g += padding

        extent = _max_g - _min_g

        for i in range(self.num_leafs):
            centroid = ti.math.vec3(0.0)

            for j in range(dim):
                centroid += x[primitives[dim * i + j]]

            centroid /= ti.cast(dim, float)
            self.pos[i] = centroid
            a = (centroid[0] - _min_g[0]) / extent[0]
            b = (centroid[1] - _min_g[1]) / extent[1]
            c = (centroid[2] - _min_g[2]) / extent[2]

            self.code_edge[2 * i + 0] = i #
            self.code_edge[2 * i + 1] = i + 1 #

            self.morton_code[i] = self.morton_3d(a, b, c)

    @ti.func
    def delta(self, i, j):
        ret = -1
        if j <= (self.num_leafs - 1) and j >= 0:
            xor = self.morton_code[i] ^ self.morton_code[j]
            if xor == 0:
                ret = 32
            else:
                ret = self.clz_soft(xor)
        return ret

    @ti.func
    def clz_soft(self, x: ti.i32) -> ti.i32:
        count = 32
        found = False
        if x > 0:
            count = 0
            for i in range(32):
                bit = (x >> (31 - i)) & 1
                is_leading_zero = (bit == 0) and (not found)
                count += ti.select(is_leading_zero, 1, 0)
                found = found or (bit == 1)
        return count

    @ti.func
    def determine_range(self, i):

        d = self.delta(i, i + 1) - self.delta(i, i - 1)
        d = 1 if d > 0 else -1
        # # print(d)
        #
        delta_min = self.delta(i, i - d)
        # print(delta_min)
        # # print(delta_min)
        #
        l_max = 2
        while self.delta(i, i + l_max * d) > delta_min:
            l_max *= 2

        l = 0
        t = l_max // 2
        while t >= 1:
            if self.delta(i, i + (l + t) * d) > delta_min:
                l += t
            t //= 2
        j = i + l * d



        return min(i, j), max(i, j)

    @ti.func
    def find_split(self, l, r):
        if l == r:
            split = l

        common_prefix = self.delta(l, r)
        split = l
        step = r - l

        while step > 1:
            step = (step + 1) // 2
            new_split = split + step

            if new_split < r:
                split_code = self.morton_code[new_split]
                split_prefix = self.clz_soft(self.morton_code[l] ^ split_code)
                if split_prefix > common_prefix:
                    split = new_split

        return split

    @ti.kernel
    def assign_internal_nodes(self):

        for i in range(self.num_leafs - 1):

            node_idx = i + self.num_leafs
            self.nodes[node_idx].parent = -1
            start, end = self.determine_range(i)
            split = self.find_split(start, end)

            left = split + self.num_leafs
            if split == start:
                left = split # leaf node

            right = split + 1 + self.num_leafs
            if split + 1 == end:
                right = split + 1


            self.nodes[node_idx].left = left
            self.nodes[node_idx].right = right
            self.nodes[node_idx].visited = 0
            self.nodes[node_idx].a = 1
            self.nodes[left].parent  = node_idx
            self.nodes[right].parent = node_idx

    @ti.kernel
    def check_integrity(self):

        cnt = 0
        # ti.loop_config(serialize=True)
        for i in range(self.num_leafs):
            idx = i
            # steps = 0
            while idx != self.num_leafs:
                idx = self.nodes[idx].parent
                # steps += 1

            # print(i, steps)
            ti.atomic_add(cnt, 1)
        # print(cnt)

    @ti.kernel
    def assign_aabb(self, x: ti.template(), primitives: ti.template(), pad: float):

        padding = ti.math.vec3(pad)
        # cnt = 0
        for i in range(self.num_leafs):
            dim = primitives.shape[0] // self.num_leafs

            if dim == 2:
                # print("test")
                v0, v1 = primitives[2 * self._ids[i] + 0], primitives[2 * self._ids[i] + 1]
                self.nodes[i]._min = ti.min(x[v0], x[v1]) - padding
                self.nodes[i]._max = ti.max(x[v0], x[v1]) + padding

            if dim == 3:
                # print("test")
                v0, v1, v2 = primitives[3 * self._ids[i] + 0], primitives[3 * self._ids[i] + 1], primitives[3 * self._ids[i] + 1]
                self.nodes[i]._min = ti.min(x[v0], x[v1], x[v2]) - padding
                self.nodes[i]._max = ti.max(x[v0], x[v1], x[v2]) + padding

            self.nodes[i].right = -1
            self.nodes[i].left = -1
            idx = self.nodes[i].parent
            while idx != -1:

                visited = ti.atomic_add(self.nodes[idx].visited, 1)
                if visited == 0: break

                left  = self.nodes[idx].left
                right = self.nodes[idx].right
                # ti.atomic_add(cnt, 1)

                self.nodes[idx]._min = ti.min(self.nodes[left]._min, self.nodes[right]._min)
                self.nodes[idx]._max = ti.max(self.nodes[left]._max, self.nodes[right]._max)

                idx = self.nodes[idx].parent
                self.nodes[idx].a = 1

    @ti.func
    def node_overlap(self, n0, n1):
        min0, max0 = self.nodes[n0]._min, self.nodes[n0]._min
        min1, max1 = self.nodes[n1]._min, self.nodes[n1]._max

        return self.aabb_overlap(min0, max0, min1, max1)

    @ti.func
    def aabb_overlap(self, min1, max1, min2, max2):
        return (min1[0] <= max2[0] and max1[0] >= min2[0] and
                min1[1] <= max2[1] and max1[1] >= min2[1] and
                min1[2] <= max2[2] and max1[2] >= min2[2])


    @ti.func
    def traverse_bvh_single(self, min0, max0, i, cache, nums):

        stack = ti.Vector([-1 for j in range(32)])
        stack[0] = self.num_leafs
        stack_counter = 1
        # cnt = 0
        while stack_counter > 0:
            # if stack_counter >= 32:
            #     print("loop")
            #     break
            # print(stack)
            stack_counter -= 1
            # if stack_counter < 0:
            #     print("test")
            idx = stack[stack_counter]
            min1, max1 = self.nodes[idx]._min, self.nodes[idx]._max
            # print(min1, max1)
            if self.aabb_overlap(min0, max0, min1, max1):
                if idx < self.num_leafs:
                    cache[i, nums[i]] = self._ids[idx]
                    nums[i] += 1
                else:
                    left, right = self.nodes[idx].left, self.nodes[idx].right

                    stack[stack_counter] = left
                    stack_counter += 1

                    stack[stack_counter] = right
                    stack_counter += 1

    @ti.func
    def traverse_bvh_single_test(self, min0, max0, type, i, cache, num):

        stack = ti.Vector([-1 for j in range(32)])
        stack[0] = self.num_leafs
        stack_counter = 1

        while stack_counter > 0:

            stack_counter -= 1
            idx = stack[stack_counter]
            min1, max1 = self.nodes[idx]._min, self.nodes[idx]._max

            if self.aabb_overlap(min0, max0, min1, max1):
                if idx < self.num_leafs:
                    n = ti.atomic_add(num[None], 1)
                    cache[n] = ti.math.ivec3([i, self._ids[idx], type])
                    # ti.atomic_add(num[None], 1)
                else:
                    left, right = self.nodes[idx].left, self.nodes[idx].right

                    stack[stack_counter] = left
                    stack_counter += 1

                    stack[stack_counter] = right
                    stack_counter += 1


    @ti.func
    def traverse_bvh_test(self, min0, max0):

        stack = ti.Vector([-1 for j in range(32)])
        stack[0] = self.num_leafs
        stack_counter = 1
        cnt = 0
        # stack = ti.Vector([-1 for j in range(32)])

        while stack_counter > 0:

            # print(stack)
            stack_counter -= 1
            idx = stack[stack_counter]
            min1, max1 = self.nodes[idx]._min, self.nodes[idx]._max
            # print(min1, max1)
            if self.aabb_overlap(min0, max0, min1, max1):
                if idx < self.num_leafs:
                    cnt += 1
            else:
                left, right = self.nodes[idx].left, self.nodes[idx].right
                stack[stack_counter] = left
                stack_counter += 1
                stack[stack_counter] = right
                stack_counter += 1

        print(cnt)

    @ti.func
    def traverse_brute(self, min0, max0, i, cache, nums):
        cnt = 0
        for n in range(self.num_leafs):
            min1, max1 = self.nodes[n]._min, self.nodes[n]._max
            if self.aabb_overlap(min0, max0, min1, max1):
                # print("test")
                cache[i, nums[i]] = self._ids[n]
                nums[i] += 1


    def build(self, x, primitives, pad):

        self.assign_morton(x, primitives, pad)

        a = self.morton_code.to_numpy()
        cp_arr = cp.asarray(a)
        idx = cp.argsort(a)
        values = cp_arr[idx]
        a = cp.asnumpy(values)
        b = cp.asnumpy(idx)
        self.morton_code.from_numpy(a)
        self._ids.from_numpy(b)

        # self.init_id()
        # self.radix_sort()
        self.assign_internal_nodes()
        self.assign_aabb(x, primitives, pad)

    def draw_bvh_aabb_test(self, scene, n):
        self.update_aabb_x_and_line0(n)
        scene.lines(self.aabb_x0, indices=self.aabb_index0, width=2.0, color=(0, 1, 0))


    @ti.kernel
    def update_aabb_x_and_line0(self, n: ti.i32):

        aabb_min = self.nodes[n]._min
        aabb_max = self.nodes[n]._max

        # print(aabb_min, aabb_max)

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