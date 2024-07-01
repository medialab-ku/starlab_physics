import taichi as ti

import matplotlib.pyplot as plt


def spread_bits(v):
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v << 8)) & 0x0300F00F
    v = (v | (v << 4)) & 0x030C30C3
    v = (v | (v << 2)) & 0x09249249
    return v

def interleave_bits_2d(x, y, z):

    return spread_bits(x) | (spread_bits(y) << 1) | (spread_bits(z) << 2)


def float_to_morton_code_2d(x, y, bits=10):
    max_val = (1 << bits) - 1
    x_int = int(x * max_val)
    y_int = int(y * max_val)
    return interleave_bits_2d(x_int, y_int)

print(bin(interleave_bits_2d(0, 0, 0)))

