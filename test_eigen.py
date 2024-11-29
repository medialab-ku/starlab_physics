import taichi as ti

ti.init(arch=ti.cuda, device_memory_GB=8, default_fp=ti.f32, kernel_profiler=False)

A = ti.math.mat2([2.0, 1.0, 1.0, 2.0])

# v1 = ti.math.vec2([1, 1])
# v2 = ti.math.vec2([-1, 1])
#
# P = ti.math.mat2([1, 1, -1, 1])
# D = ti.math.mat2([1, 0, 0, 3])
#
# print(P @ D @ P.inverse())

# id2 = ti.Matrix.identity(dt=float, n=2)
#
# x01 = ti.math.vec2(2.0, 0.0)
# r = x01.norm()
# # n =
# B = (id2 - (1.0 / r) * (id2 - (x01.outer_product(x01)) / (r * r)))



# print(v1.outer_product(v1))
# print(v2.outer_product(v2))

@ti.kernel
def foo():

    r0 = 1.0
    id2 = ti.Matrix.identity(dt=float, n=3)
    x01 = ti.math.vec3(1.0, 1.0, 1.0)
    r = x01.norm()
    n = x01 / r
    t1 = ti.math.vec3(n[1], -n[0], 0.0)
    t2 = n.cross(t1)

    nnT = n.outer_product(n)
    alpha = r0 / r

    if 1.0 - alpha < 0.0:
        print("indefinite")

    stiffness = 1e3
    B = stiffness * ((1.0 - alpha) * id2 + alpha * nnT)
    print(B)
    # B = 2 * id2

    D = ti.math.mat3([stiffness, 0.0, 0.0, 0.0, stiffness * abs(1.0 - alpha), 0.0, 0.0, 0.0, stiffness * abs(1.0 - alpha)])
    P = ti.math.mat3([n[0], t1[0], t2[0], n[1], t1[1], t2[1], n[2], t1[2], t2[2]])
    # print(B)
    # eigenvalues, eigenvectors = ti.eig(B)
    # print(eigenvalues)
    # print(eigenvectors)

    #
    # P = ti.math.mat2(0.0)
    # D = ti.math.mat2(0.0)
    # for i in range(2):
    #     D[i, i] = eigenvalues[i, 0]
    #     for j in range(2):
    #         P[i, j] = eigenvectors[i, 2 * i + j]
    # print(P)
    print(P @ D @ P.inverse())

foo()
n = 3
K = ti.linalg.SparseMatrixBuilder(3 * n, 3 * n, max_num_triplets=100)

@ti.kernel
def fill(A: ti.types.sparse_matrix_builder()):
    test = ti.math.mat3([1, 1, 1, 1, 1, 1, 1, 1, 1])
    ids = ti.Vector([0, 2], ti.i32)
    for o, p in ti.ndrange(3, 3):
        A[3 * ids[0] + o, 3 * ids[1] + p] -= test[o, p]
        A[3 * ids[1] + o, 3 * ids[0] + p] -= test[o, p]

        A[3 * ids[0] + o, 3 * ids[0] + p] += test[o, p]
        A[3 * ids[1] + o, 3 * ids[1] + p] += test[o, p]

fill(K)

A = K.build()
print(A)