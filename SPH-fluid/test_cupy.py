import cupy as cp

# Create a random array of integers
arr = cp.random.randint(0, 10000, size=1000000, dtype=cp.uint32)
print("Original array:", arr)

# Perform radix-sort (internally)
sorted_arr = cp.sort(arr)
print("Sorted array:", sorted_arr)

# To get the indices that would sort the array (argsort)
sorted_indices = cp.argsort(arr)
# print("Indices that sort the array:", sorted_indices)

# Sorted manually using indices
# print("Manually sorted:", arr[sorted_indices])