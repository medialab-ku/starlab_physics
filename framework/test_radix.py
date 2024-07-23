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
NUM_COLORS = pow(2, BITS_PER_PASS)
NUM_COLORS = 10
passes = (30 + BITS_PER_PASS - 1) // BITS_PER_PASS
passes = 3
prefix_sum_executer = ti.algorithms.PrefixSumExecutor(NUM_COLORS)
# prefix_sum = ti.field(dtype=ti.int32, shape=RADIX)

num_edges = 10
color = ti.field(dtype=ti.int32, shape=num_edges)
edge_idx = ti.field(dtype=ti.int32, shape=num_edges)
sorted_edge_idx = ti.field(dtype=ti.int32, shape=num_edges)
sorted_to_origin = ti.field(dtype=ti.int32, shape=num_edges)
# sorted_morton_codes = []

prefix_sum = ti.field(dtype=ti.int32, shape=NUM_COLORS)
prefix_sum_temp = ti.field(dtype=ti.int32, shape=NUM_COLORS)
sorted_color = ti.field(dtype=ti.int32, shape=num_edges)
# for i in range(pass_num):
#     digit = (mc_i >> (i * BITS_PER_PASS)) & (RADIX - 1)
#     print(bin(digit))

test = 329

i = 2

# print((test % pow(10, i + 1)) // pow(10, i))

@ti.kernel
def init():
    color[0] = 9
    color[1] = 7
    color[2] = 7
    color[3] = 9
    color[4] = 6
    color[5] = 5
    color[6] = 6
    color[7] = 6
    color[8] = 0
    color[9] = 5

    edge_idx[0] = 0
    edge_idx[1] = 1
    edge_idx[2] = 2
    edge_idx[3] = 3
    edge_idx[4] = 4
    edge_idx[5] = 5
    edge_idx[6] = 6
    edge_idx[7] = 7
    edge_idx[8] = 8
    edge_idx[9] = 9

@ti.kernel
def count_frequency(pass_num: ti.i32):
    for i in range(num_edges):
        col_i = color[i]
        ti.atomic_add(prefix_sum[col_i], 1)

@ti.kernel
def counting_sort(pass_num: ti.i32):

    ti.loop_config(serialize=True)
    for i in range(num_edges):
        I = num_edges - 1 - i
        col_i = color[I]
        idx = prefix_sum[col_i] - 1
        sorted_color[idx] = color[I]
        sorted_edge_idx[idx] = edge_idx[I]
        ti.atomic_sub(prefix_sum[col_i], 1)

    for i in range(num_edges):
        sorted_idx = sorted_edge_idx[i]
        sorted_to_origin[sorted_idx] = i



#the prefix_sum range must be less then the size of the array
def radix_sort():


    print("unsorted color:", color)
    # print(pi)
    prefix_sum.fill(0)
    count_frequency(0)
    print("prefix sum(before): ", prefix_sum)
    prefix_sum_executer.run(prefix_sum)
    prefix_sum_temp.copy_from(prefix_sum)
    print("prefix sum(after): ", prefix_sum_temp)
    counting_sort(0)
    color.copy_from(sorted_color)


    print("sorted color: ", color)

    print("unsorted index:", edge_idx)
    print("sorted index:", sorted_edge_idx)

    print("sorted to origin:", sorted_to_origin)


    # print("original index")
    # print(sorted_index)


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