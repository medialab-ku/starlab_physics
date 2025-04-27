import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

num_bits = 30
_range = pow(2, num_bits)
bits_per_pass = 8
num_pass = num_bits // bits_per_pass + (1 if  num_bits % bits_per_pass > 0 else 0)

# print(num_pass)
# num_pass = num_bits

# print(num_pass)



buf_size  = pow(2, bits_per_pass)
# buf_size  = 10
count     = ti.field(ti.i32, shape=buf_size)
count_tmp = ti.field(ti.i32, shape=buf_size)

size     = int(1e2)
code     = ti.field(ti.i32, shape=size)
code_buf = ti.field(ti.i32, shape=size)
id_new   = ti.field(ti.i32, shape=size)
prefix_sum_exec = ti.algorithms.PrefixSumExecutor(buf_size)

@ti.kernel
def count_values(pass_id: int):

    for i in range(size):
        key = (code[i] >> (bits_per_pass * pass_id)) & 0xFF
        # digit = (code[i] // pow(10, pass_id)) % 10
        # digit = code[i]
        ti.atomic_add(count[key], 1)

@ti.kernel
def sort(pass_id: int):

    ti.loop_config(serialize=True)
    for i in range(size):
        I = size - 1 - i  # reverse for stability
        key = (code[I] >> (bits_per_pass * pass_id)) & 0xFF
        a = ti.atomic_sub(count_tmp[key], 1)
        id_new[I] = a - 1

    for i in range(size):
        new_id = id_new[i]
        code_buf[new_id] = code[i]

@ti.kernel
def validate(pass_id: int) -> int:

    cnt = 0
    for i in range(size - 1):
        digit_i = (code[i] >> (bits_per_pass * pass_id)) & 0xFF
        digit_j = (code[i + 1] >> (bits_per_pass * pass_id)) & 0xFF
        if digit_i > digit_j:
            ti.atomic_add(cnt, 1)
    return cnt

@ti.kernel
def validate_last() -> int:

    cnt = 0
    for i in range(size - 1):
        if code[i] > code[i + 1]:
            digit_i = (code[i] >> (bits_per_pass * 1)) & 0xFF
            digit_j = (code[i + 1] >> (bits_per_pass * 1)) & 0xFF
            print(digit_i, digit_j)
            ti.atomic_add(cnt, 1)

    return cnt

def radix_sort():
    for i in range(num_pass):

        count.fill(0)
        count_values(i)
        prefix_sum_exec.run(count)
        count_tmp.copy_from(count)
        sort(i)
        code.copy_from(code_buf)
        # print(code)
        print("after pass ", i, ": ", validate(i))

code.from_numpy(np.random.randint(0, _range, size=size))

# print(code)
# a = 4321
# n = 2
# print((a // pow(10, n)) % 10)

# # print(code)
# radix_sort()

# print(code)
#
# print("final: ", validate_last())


575219354
709436562
728610003
786130326

arr = np.array([172567730, 575219354, 709436562, 728610003, 786130326])

a = int(172567730)
b = int(575219354)

def binary_32bit(n):

    return format(n & 0xFFFFFFFF, '032b')

def count_leading_zeros(num):
    bit = binary_32bit(num)
    return len(bit) - len(bit.lstrip('0'))

@ti.func
def clz_soft(x: ti.i32) -> ti.i32:
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


for i in range(len(arr) - 1):
    print(count_leading_zeros(arr[i] ^ arr[4]))

print()

for i in range(len(arr) - 1):
    print(count_leading_zeros(arr[0] ^ arr[i + 1]))


# print(binary_32bit(a^b))
# print(count_leading_zeros(a^b))




# print(foo(a, b))
