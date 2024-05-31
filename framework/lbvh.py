import taichi as ti

# Expands a 10-bit integer into 30 bits by inserting 2 zeros after each bit.
@ti.func
def expand_bits(v):
    v = (v * 0x00010001) & 0xFF0000FF
    v = (v * 0x00000101) & 0x0F00F00F
    v = (v * 0x00000011) & 0xC30C30C3
    v = (v * 0x00000005) & 0x49249249
    return v

@ti.func
def morton_3d(x, y, z):
    x = ti.math.clamp(x * 1024., 0., 1023.)
    y = ti.math.clamp(y * 1024., 0., 1023.)
    z = ti.math.clamp(z * 1024., 0., 1023.)
    xx = expand_bits(ti.cast(x, ti.uint32))
    yy = expand_bits(ti.cast(y, ti.uint32))
    zz = expand_bits(ti.cast(z, ti.uint32))
    return xx * 4 + yy * 2 + zz

@ti.func
def delta(a, b, n, c, ka):
    # // this guard is for leaf nodes, not internal nodes (hence [0, n-1])
    if (b < 0) or (b > n - 1):
        return -1
    kb = c[b]
    if ka == kb:
        # // if keys are equal, use id as fallback
        # // (+32 because they have the same morton code)
        return 32 + ti.math.clz(ti.cast(a, ti.uint32) ^ ti.cast(b, ti.uint32))
    # // clz = count leading zeros
    return ti.math.clz(ka ^ kb)

@ti.func
def determine_range(sorted_morton_codes, n, i):
    c = sorted_morton_codes
    ki = c[i]   # key of i

    # determine direction of the range(+1 or -1)

    delta_l = delta(i, i - 1, n, c, ki)
    delta_r = delta(i, i + 1, n, c, ki)

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
    while delta(i, i + l_max * d, n, c, ki) > delta_min:
        l_max <<= 1

    # find other end using binary search
    l = 0
    t = l_max >> 1
    while t > 0:
        if delta(i, i + (l + t) * d, n, c, ki) > delta_min:
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

@ti.func
def find_split(sorted_morton_codes, first, last, n):
    first_code = sorted_morton_codes[first]

    # calculate the number of highest bits that are the same
    # for all objects, using the count-leading-zeros intrinsic

    common_prefix = delta(first, last, n, sorted_morton_codes, first_code)

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
            split_prefix = delta(first, new_split, n, sorted_morton_codes, first_code)
            if split_prefix > common_prefix:
                # accept proposal
                split = new_split

    return split