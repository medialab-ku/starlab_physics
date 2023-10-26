import taichi as ti


ti.init(arch=ti.cpu)
n = 4
K = ti.linalg.SparseMatrixBuilder(n, n, max_num_triplets=100)
b = ti.ndarray(ti.f32, shape=n)

@ti.kernel
def fill(A: ti.types.sparse_matrix_builder(), b: ti.types.ndarray(), interval: ti.i32):
    for i in range(n):
        A[i, i] += 2.0

        if i % interval == 0:
            b[i] += 1.0

@ti.kernel
def fill2(A: ti.types.sparse_matrix_builder(), b: ti.types.ndarray(), interval: ti.i32):
    for i in range(n):
        A[i, i] += 1.0

    for i in range(n):
        A[i, i] += 3.0


fill(K, b, 3)

A = K.build()
print(">>>> Matrix A:")
print(A)
B = fill2(K, b, 3)

B = K.build()
print(B)

print(">>>> Vector b:")
print(b)
# outputs:
# >>>> Matrix A:
# [2, 0, 0, 0]
# [0, 2, 0, 0]
# [0, 0, 2, 0]
# [0, 0, 0, 2]
# >>>> Vector b:
# [1. 0. 0. 1.]
solver = ti.linalg.SparseSolver(solver_type="LLT")
solver.analyze_pattern(A)
solver.factorize(A)
x = solver.solve(b)
# isSuccess = solver.info()
print(">>>> Solve sparse linear systems Ax = b with the solution x:")
print(x)
# print(f">>>> Computation was successful?: {isSuccess}")
# outputs:
# >>>> Solve sparse linear systems Ax = b with the solution x:
# [0.5 0.  0.  0.5]
# >>>> Computation was successful?: True