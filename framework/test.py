import taichi as ti

ti.init(arch=ti.cuda)  # or ti.cuda
#

num_leaf = 10
morton_codes = ti.field(dtype=ti.i32, shape=num_leaf)
sorted_morton_codes = ti.field(dtype=ti.i32, shape=num_leaf)
indices = ti.field(dtype=ti.i32, shape=num_leaf)
sorted_indices = ti.field(dtype=ti.i32, shape=num_leaf)
BITS_PER_PASS = 1
RADIX = pow(2, BITS_PER_PASS)
prefix_sum_executer = ti.algorithms.PrefixSumExecutor(RADIX)
prefix_sum = ti.field(dtype=ti.i32, shape=RADIX)
count = ti.field(dtype=ti.i32, shape=RADIX)

@ti.kernel
def test():
    for i in range(num_leaf):
        morton_codes[i] = num_leaf - i
        indices[i] = i


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

    for i in range(num_leaf):
        I = num_leaf - 1 - i
        digit = (morton_codes[I] >> (pass_num * BITS_PER_PASS)) & (RADIX - 1)
        idx = ti.atomic_sub(prefix_sum[digit], 1) - 1
        if idx >= 0:
            sorted_morton_codes[idx] = morton_codes[I]

def radix_sort():
    test()
    print(morton_codes)
    max_value = get_max_value()
    passes = (max_value.bit_length() + BITS_PER_PASS - 1) // BITS_PER_PASS

    print(passes)
    for pi in range(passes):
        count.fill(0)
        count_frequency(pi)
        prefix_sum.copy_from(count)
        prefix_sum_executer.run(prefix_sum)
        sort_by_digit(pi)
        morton_codes.copy_from(sorted_morton_codes)


radix_sort()
print(sorted_morton_codes)


#
# prefix_sum_executer = ti.algorithms.PrefixSumExecutor(10)
#
# test()
#
# prefix_sum_executer.run(vec)