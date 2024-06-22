import numpy as np


def blelloch_scan_inclusive(arr):
    n = len(arr)
    if n == 0:
        return arr

    # Ensure the length is a power of two by padding with zeros
    m = 1
    while m < n:
        m <<= 1
    padded_arr = arr + [0] * (m - n)

    # if n % 2 == 0:
    #     m = n.bit_length()
    # else:
    #     m = n.bit_length() + 1

    test = len(arr)


    # Step 1: Upsweep (reduce) phase
    upsweep = padded_arr.copy()

    d = 0
    while test > 1:
        step = 1 << (d + 1)
        size = n // step
        offset = step - 1
        for i in range(size):
            id = offset + step * i
            upsweep[id] += upsweep[id - (step >> 1)]
        # print(upsweep)
        d += 1
        test //= 2

    # Step 2: Downsweep phase
    downsweep = upsweep.copy()
    a = downsweep[n - 1]
    downsweep[n - 1] = 0  # Set the last element to 0 for the downsweep phase

    d = n.bit_length() - 1
    while d >= 0:
        print(downsweep)
        step = 1 << (d + 1)
        size = m // step
        offset = step - 1
        offset_rev = (step >> 1)
        for i in range(size):
            id = offset + step * i
            # if id < n:
            temp = downsweep[id - offset_rev]
            downsweep[id - offset_rev] = downsweep[id]
            downsweep[id] += temp
            # else:
                # temp = downsweep[id - offset_rev]
                # downsweep[id - offset_rev] = downsweep[id]
                # downsweep[id] += a
            #
            # if id - offset_rev >= n:
            #     print("fuck")

        # print(downsweep)
        d -= 1
    # downsweep[n - 1] = a + downsweep[n - 2]
    # Step 3: Convert to inclusive scan and trim padding

    for i in range(m):
        downsweep[i] += padded_arr[i]

    print(downsweep)
    return downsweep[:n]


size = 31
arr = [0] * size

for i in range(size):
    arr[i] = 1

# Example usage
# arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1]
result = blelloch_scan_inclusive(arr)
print(result)

# Test with non-power of two length
# arr = [1, 1, 1, 1, 1, 1]
# result = blelloch_scan_inclusive(arr)
# print(result)
