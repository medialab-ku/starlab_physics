import taichi as ti
import ccd as ccd
ti.init(arch=ti.cuda, kernel_profiler=True)
a = ti.math.vec3([1, 1, 1])
b = ti.math.vec3([1, 1, 1])

@ti.func
def test_abT(a: ti.math.vec3, b: ti.math.vec3) -> ti.math.mat3:

    abT = ti.math.mat3([[0, 0, 0], [0, 0, 0], [0, 0, 0]])

    abT[0, 0] = a[0] * b[0]
    abT[0, 1] = a[0] * b[1]
    abT[0, 2] = a[0] * b[2]

    abT[1, 0] = a[1] * b[0]
    abT[1, 1] = a[1] * b[1]
    abT[1, 2] = a[1] * b[2]

    abT[2, 0] = a[2] * b[0]
    abT[2, 1] = a[2] * b[1]
    abT[2, 2] = a[2] * b[2]

    return abT


@ti.kernel
def test():

    a = ti.math.vec3([0.0, 0.0, 0.0])
    b = ti.math.vec3([1.0, 0.0, 0.0])

    c = ti.math.vec3([0.5, 0.3, 0.5])
    d = ti.math.vec3([0.5, 0.3, -1.0])

    ab = b - a
    cd = d - c
    ac = c - a

    a11 = ab.dot(ab)
    a12 = -cd.dot(ab)
    a21 = cd.dot(ab)
    a22 = -cd.dot(cd)
    det = a11 * a22 - a12 * a21

    mat = ti.math.mat2([[ab.dot(ab), -cd.dot(ab)], [cd.dot(ab), -cd.dot(cd)]])

    #
    gg = ti.math.vec2([ab.dot(ac), cd.dot(ac)])

    t = mat.inverse() @ gg
    #
    # s = (a22 * gg[0] - a12 * gg[1]) / det
    # t = (-a21 * gg[0] + a11 * gg[1]) / det
    # t2 = ti.min(1, ti.max(t2, 0))

    t1 = t[0]
    t2 = t[1]

    if t1 < 0.0:
        t1 = 0.0

    if t1 > 1.0:
        t1 = 1.0

    if t2 < 0.0:
        t2 = 0.0

    if t2 > 1.0:
        t2 = 1.0


    print(t)

a = ti.math.vec3([0.0, 0.0, 0.0])
b = ti.math.vec3([1.0, 1.0, 1.0])

c = ti.math.vec3([0.5, 0.5, 1.1])
d = ti.math.vec3([1.5, 1.5, 1.5])

e = ti.math.mat3([a, b, c])

dof = 16000
n = 6  # Size of the matrices
m = dof // n  # Number of matrices to factorize

A = ti.field(ti.f32, shape=(m, n * n))
L = ti.field(ti.f32, shape=(m, n * n))
x = ti.Vector.field(n=3, shape=(m, n), dtype=ti.f32)
y = ti.Vector.field(n=3, shape=(m, n), dtype=ti.f32)
b = ti.Vector.field(n=3, shape=(m, n), dtype=ti.f32)

L.fill(0.0)
x.fill(0.0)
b.fill(1.0)

@ti.kernel
def init():
    for i in range(m):
        # A[i] = ti.Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        for j in range(n):
            A[i, j + n * j] = (i + 1)

@ti.kernel
def cholesky():
    for bat in range(m):
        for j in range(n):
            sum = 0.0
            for k in range(j):
                sum += L[bat, j * n + k] * L[bat, j * n + k]

            L[bat, j * n + j] = ti.sqrt(A[bat, j * n + j] - sum)
            for i in range(j + 1, n):
                sum = 0.0
                for k in range(j):
                    sum += L[bat, i * n + k] * L[bat, j * n + k]

                L[bat, i * n + j] = (1.0 / L[bat, j * n + j] * (A[bat, i * n + j] - sum))

@ti.kernel
def solve_backward():
    for bat in range(m):
        for i in range(n):
            x[bat, i] = b[bat, i]
            for j in range(n):
                x[bat, i] -= L[bat, j * n + i] * x[bat, j]
            x[bat, i] /= L[bat, i * n + i]

@ti.kernel
def solve_forward():
    for bat in range(m):
        for inv_i in range(n):
            i = n - inv_i - 1
            x[bat, i] = b[bat, i]
            for j in range(i + 1, n):
                x[bat, i] -= L[bat, j * n + i] * x[bat, j]
            x[bat, i] /= L[bat, i * n + i]
# init()
#
# ti.profiler.clear_kernel_profiler_info()
# cholesky()
# solve_backward()
# solve_forward()
# query_result1 = ti.profiler.query_kernel_profiler_info(cholesky.__name__)
# query_result2 = ti.profiler.query_kernel_profiler_info(solve_backward.__name__)
# query_result3 = ti.profiler.query_kernel_profiler_info(solve_forward.__name__)
# print("cholesky elapsed time: ", query_result1.avg)
# print("back elapsed time: ", query_result2.avg)
# print("forward elapsed time: ", query_result3.avg)
# print("total elapsed time: ", query_result1.avg + query_result2.avg + query_result3.avg)

@ti.kernel
def test():
    for offset in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2)))):
        print(offset[0], offset[1])

test()