import taichi as ti
import numpy as np
import time

ti.init(arch=ti.cuda, kernel_profiler=False)

num_leaf = 1000
morton_codes = ti.field(dtype=ti.i32, shape=num_leaf)
sorted_morton_codes = ti.field(dtype=ti.i32, shape=num_leaf)
indices = ti.field(dtype=ti.i32, shape=num_leaf)
sorted_indices = ti.field(dtype=ti.i32, shape=num_leaf)
BITS_PER_PASS = 16
RADIX = pow(2, BITS_PER_PASS)
prefix_sum_executer = ti.algorithms.PrefixSumExecutor(RADIX)
prefix_sum = ti.field(dtype=ti.i32, shape=RADIX)
count = ti.field(dtype=ti.i32, shape=RADIX)
count_temp = ti.field(dtype=ti.i32, shape=RADIX)

a = ti.field(dtype=ti.int32, shape=num_leaf)
temp = ti.field(dtype=ti.int32, shape=num_leaf)

@ti.kernel
def test():
    for i in range(num_leaf):
        morton_codes[i] = num_leaf - i
        indices[i] = i

    # morton_codes[0] = 7
    # morton_codes[1] = 8
    # morton_codes[2] = 4
    # morton_codes[3] = 3
    # morton_codes[4] = 2
    # morton_codes[5] = 5
    # morton_codes[6] = 9
    # morton_codes[7] = 0
    # morton_codes[8] = 1
    # morton_codes[9] = 6

@ti.kernel
def get_max_value() -> ti.i32:

    max_value = -1
    for i in range(num_leaf):
        ti.atomic_max(max_value, morton_codes[i])

    return max_value

@ti.kernel
def count_frequency(pass_num: ti.i32):
    for i in range(num_leaf):
        digit = (morton_codes[i] >> (pass_num * BITS_PER_PASS)) & (RADIX - 1)
        ti.atomic_add(count[digit], 1)

@ti.kernel
def sort_by_digit(pass_num: ti.i32):

    ti.loop_config(block_dim=64)
    for i in range(num_leaf):
        I = num_leaf - 1 - i
        digit = (morton_codes[I] >> (pass_num * BITS_PER_PASS)) & (RADIX - 1)
        idx = ti.atomic_sub(count[digit], 1) - 1
        if idx >= 0:
            sorted_indices[idx] = indices[I]
            sorted_morton_codes[idx] = morton_codes[I]

def radix_sort():
    max_value = get_max_value()
    passes = (max_value.bit_length() + BITS_PER_PASS - 1) // BITS_PER_PASS
    print(passes)

    for pi in range(passes):
        count.fill(0)
        count_frequency(pi)

        # blelloch_scan()
        prefix_sum_executer.run(count)
        sort_by_digit(pi)
        morton_codes.copy_from(sorted_morton_codes)
        indices.copy_from(sorted_indices)

@ti.kernel
def upsweep(step: ti.int32, size: ti.int32):
    offset = step - 1
    for i in range(size):
        id = offset + step * i
        count[id] += count[id - (step >> 1)]

@ti.kernel
def downsweep(step: ti.int32, size: ti.int32):
    offset = step - 1
    offset_rev = (step >> 1)
    for i in range(size):
        id = offset + step * i
        temp = count[id - offset_rev]
        count[id - offset_rev] = count[id]
        count[id] += temp


@ti.kernel
def add_count():

    for i in range(RADIX):
        count[i] += count_temp[i]

def blelloch_scan():

    count_temp.copy_from(count)

    d = 0
    test = RADIX
    while test > 1:
        step = 1 << (d + 1)
        size = RADIX // step
        upsweep(step, size)

        d += 1
        test //= 2

    count[RADIX - 1] = 0
    d = BITS_PER_PASS - 1

    while d >= 0:
        step = 1 << (d + 1)
        size = RADIX // step
        downsweep(step, size)
        d -= 1

    add_count()


# count.fill(1)
# start_time = time.time()
# blelloch_scan()
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(elapsed_time)
#
# start_time = time.time()
# prefix_sum_executer.run(count)
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(elapsed_time)

test()
start_time = time.time()
radix_sort()
end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)


test()
start_time = time.time()
ti.algorithms.parallel_sort(keys=morton_codes, values=indices)
end_time = time.time()
elapsed_time = end_time - start_time
print(elapsed_time)

#
#
# test()
#
# prefix_sum_executer.run(vec)