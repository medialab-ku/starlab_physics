import taichi as ti

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

@ti.kernel
def add_data():

    a, b = ret()
    print(a)
    print(b)

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


# print(aabb_intersect(a, b, c, d))

# test_max()
add_data()
