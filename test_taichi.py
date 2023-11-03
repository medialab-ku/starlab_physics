import taichi as ti
import ccd as ccd
ti.init(arch=ti.cuda)
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

n = 9  # Size of the matrices
m = 1  # Number of matrices to factorize

A = ti.field(ti.f32, shape=(m, n * n))
L = ti.field(ti.f32, shape=(m, n * n))

L.fill(0.0)
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

init()
cholesky()

print(L)