import taichi as ti

@ti.func
def edge_edge_dist(a: ti.math.vec3, b: ti.math.vec3, c: ti.math.vec3, d: ti.math.vec3, t: ti.math.vec2) -> ti.f32:
    ab = b - a
    cd = d - c
    ac = c - a

    mat = ti.math.mat2([[-cd.dot(ab), ab.dot(ab)],
                        [-cd.dot(cd), cd.dot(ab)]])

    b = ti.math.vec2([ab.dot(ac), cd.dot(ac)])

    t = mat.inverse() @ b

    t1 = t[0]
    t2 = t[1]

    p1 = a + t1 * ab
    p2 = c + t2 * cd

    dist = (p1 - p2).norm()

    return dist