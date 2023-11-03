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

print(e)

@ti.func
def ret():
    a = ti.math.vec3([0.0, 0.0, 0.0])
    b = ti.math.vec3([1.0, 1.0, 1.0])
    return a, b

@ti.kernel
def test_max():

    a = ti.math.vec3(1,2,3)
    max = ti.max(a)
    print(max)

@ti.kernel
def aabb_intersect(a_min: ti.math.vec3, a_max: ti.math.vec3, b_min: ti.math.vec3, b_max: ti.math.vec3) -> ti.i32:

        return  a_min[0] <= b_max[0] and \
                a_max[0] >= b_min[0] and \
                a_min[1] <= b_max[1] and \
                a_max[1] >= b_min[1] and \
                a_min[2] <= b_max[2] and \
                a_max[2] >= b_min[2]


S = ti.root.dynamic(ti.i, 1024, chunk_size=32)
x = ti.field(ti.math.uvec4)
S.place(x)

@ti.func
def test(i: ti.int32)-> ti.int32:
    return i
@ti.kernel
def add_data():


    min = 4.0
    for i in range(10):
        a = test(i)
        if min > a:
            min = a

    print(min)
    # for i in range(10):
    #     x.append(ti.math.uvec4([4 * i + 0, 4 * i + 1, 4 * i + 2, 4 * i + 3]))
    # #
    #
    # for xi in x:
    #     print(x[xi][0])
    #
    # print(x.length())
    # print(x[0])
    # x.deactivate()
    # print(x.length())

@ti.kernel
def test_ccd():
    x0 = ti.math.vec3([0.5, 0.5, 0.5])
    dx0 = ti.math.vec3([0.0, -1.0, 0.0])


    x1 = ti.math.vec3([1.0, 0.0, 0.])
    x2 = ti.math.vec3([0., 0.0, 1.])
    x3 = ti.math.vec3([0., 0.0, 0.])

    dx_zero = ti.math.vec3([0.0, 0.0, 0.0])

    alpha_ccd = ccd.point_triangle_ccd(x0, x1, x2, x3, dx0, dx_zero, dx_zero, dx_zero, 0.0, 0.01, 1.0)

    x = x0 + alpha_ccd * dx0

    print(x.y)
    print(alpha_ccd)
# print(aabb_intersect(a, b, c, d))

# add_data()
test_ccd()

num_mat = 10
a_field = ti.Matrix.field(m=4, n=4, shape=(num_mat), dtype=ti.f32)
l_field = ti.Matrix.field(m=4, n=4, shape=(num_mat), dtype=ti.f32)

batch_size = 10
num_bats = 10
tmat_size = (batch_size * (batch_size + 1) / 2)

mat = ti.Matrix(m=3, n=3, shape=(tmat_size * num_bats), dtype=ti.f32)
x = ti.Vector.field(n=3, shape=(batch_size * num_bats), dtype=ti.f32)

@ti.kernel
def test_llt():

   print()


test_llt()