import taichi as ti

enable_profiler = False
ti.init(arch=ti.cuda, device_memory_GB=6, kernel_profiler=enable_profiler)

# morton_codes = []
# morton_codes.append(int('00101101101101101101101101101101', 2))
# morton_codes.append(int('00001001001001001001001001001001', 2))
# morton_codes.append(int('00111111111111111111111111111111', 2))
# morton_codes.append(int('00010010010010010010010010010010', 2))
# pass_num = 0
BITS_PER_PASS = 6
RADIX = pow(2, BITS_PER_PASS)
RADIX = 10
passes = (30 + BITS_PER_PASS - 1) // BITS_PER_PASS
passes = 3
prefix_sum_executer = ti.algorithms.PrefixSumExecutor(RADIX)
# prefix_sum = ti.field(dtype=ti.int32, shape=RADIX)

num_leafs = 10
morton_codes = ti.field(dtype=ti.int32, shape=num_leafs)
ids = ti.field(dtype=ti.int32, shape=num_leafs)
# sorted_morton_codes = []

prefix_sum = ti.field(dtype=ti.int32, shape=RADIX)
sorted_morton_codes = ti.field(dtype=ti.int32, shape=num_leafs)
# for i in range(pass_num):
#     digit = (mc_i >> (i * BITS_PER_PASS)) & (RADIX - 1)
#     print(bin(digit))

test = 329

i = 2

# print((test % pow(10, i + 1)) // pow(10, i))

@ti.kernel
def init():
    morton_codes[0] = 9
    morton_codes[1] = 7
    morton_codes[2] = 7
    morton_codes[3] = 9
    morton_codes[4] = 6
    morton_codes[5] = 5
    morton_codes[6] = 6
    morton_codes[7] = 6
    morton_codes[8] = 0
    morton_codes[9] = 5

@ti.kernel
def count_frequency(pass_num: ti.i32):
    for i in range(num_leafs):
        mc_i = morton_codes[i]
        # digit = (mc_i >> (pass_num * BITS_PER_PASS)) & (RADIX - 1)
        digit = (mc_i % pow(10, pass_num + 1)) // pow(10, pass_num)
        # digit = mc_i
        ti.atomic_add(prefix_sum[digit], 1)

@ti.kernel
def sort_by_digit(pass_num: ti.i32):

    ti.loop_config(serialize=True)
    for i in range(num_leafs):
        I = num_leafs - 1 - i
        mc_i = morton_codes[I]
        # digit = (mc_i >> (pass_num * BITS_PER_PASS)) & (RADIX - 1)
        digit = (mc_i % pow(10, pass_num + 1)) // pow(10, pass_num)
        # digit = mc_i
        # idx = ti.atomic_sub(prefix_sum[digit], 1)
        idx = prefix_sum[digit] - 1
        # if idx >= 0:
        sorted_morton_codes[idx] = morton_codes[I]
        ti.atomic_sub(prefix_sum[digit], 1)



#the prefix_sum range must be less then the size of the array
def radix_sort():

    for pi in range(1):
        # print(pi)
        prefix_sum.fill(0)

        # print(prefix_sum)
        # for i in range(num_leafs):
        #     mc_i = morton_codes[i]
        #     digit = (mc_i % pow(10, pi + 1)) // pow(10, pi)
        #     prefix_sum[digit] += 1
        # print(prefix_sum)
        count_frequency(pi)
        # print(prefix_sum)
        prefix_sum_executer.run(prefix_sum)
        # print(prefix_sum)
        sort_by_digit(pi)
        # print(prefix_sum)
        morton_codes.copy_from(sorted_morton_codes)
        # for i in range(9):
        #     prefix_sum[i + 1] = prefix_sum[i] + prefix_sum[i + 1]
        #     # print(prefix_sum[i])
        #
        # print(prefix_sum)
        # for i in range(num_leafs):
        #     I = num_leafs - 1 - i
        #
        #     mc_i = morton_codes[I]
        #
        #     digit = (mc_i % pow(10, pi + 1)) // pow(10, pi)
        #
        #     idx = prefix_sum[digit] - 1
        #     sorted_morton_codes[idx] = morton_codes[I]
        #     prefix_sum[digit] -= 1

        # for i in range(num_leafs):
        #     morton_codes[i] = sorted_morton_codes[i]

    print(morton_codes)


# init()
# morton_codes.append(329)
# morton_codes.append(457)
# morton_codes.append(657)
# morton_codes.append(657)
# morton_codes.append(657)
# morton_codes.append(839)
# morton_codes.append(436)
# morton_codes.append(720)
# morton_codes.append(355)
# morton_codes.append(355)
#
# sorted_morton_codes.append(329)
# sorted_morton_codes.append(457)
# sorted_morton_codes.append(657)
# sorted_morton_codes.append(657)
# sorted_morton_codes.append(657)
# sorted_morton_codes.append(839)
# sorted_morton_codes.append(436)
# sorted_morton_codes.append(720)
# sorted_morton_codes.append(355)
# sorted_morton_codes.append(355)
init()
radix_sort()