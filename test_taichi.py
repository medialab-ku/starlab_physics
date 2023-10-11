import taichi as ti

ti.init(arch=ti.cuda)

S = ti.root.dense(ti.i, 10).dynamic(ti.j, 1024, chunk_size=32)
x = ti.field(int)
S.place(x)

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

    A = ti.math.vec3([0.594281, 0.383333, 0.528867])
    B = ti.math.vec3([0.452860, 0.312623, 0.488043])

    C = ti.math.vec3([0.500000, 0.270000, 0.413397])
    D = ti.math.vec3([0.452860, 0.274044, 0.569692])


    AB = B - A
    CD = D - C
    AC = C - A

    mat = ti.math.mat2([[-CD.dot(AB), AB.dot(AB)],
                        [-CD.dot(CD), CD.dot(AB)]])

    b = ti.math.vec2([AB.dot(AC), CD.dot(AC)])

    t = mat.inverse() @ b

    t1 = t[0]
    t2 = t[1]

    p1 = A + t1 * AB
    p2 = C + t2 * CD

    dist = (p1 - p2).norm()
    print(t)
    print(dist)

@ti.kernel
def add_data():
    for i in range(10):
        for j in range(i):
            x[i].append(j)
            print(i, x[i, j])  # will print i

    for i in range(10):
        x[i].deactivate()
        print(x[i].length())  # will print 0


abT = test_abT(a, b)

print(abT)