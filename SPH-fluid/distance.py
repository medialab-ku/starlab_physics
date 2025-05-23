import taichi as ti

@ti.func
def d_type_PT(v0: ti.math.vec3, v1: ti.math.vec3, v2: ti.math.vec3, v3: ti.math.vec3) -> int:

    v12 = v2 - v1
    v13 = v3 - v1
    n = v12.cross(v13)
    v13n = v12.cross(n)
    v10 = v0 - v1

    A = ti.math.mat2([[v12.dot(v12), v12.dot(v13n)],
                      [v13n.dot(v12), v13n.dot(v13n)]])

    v10_proj = ti.math.vec2([v12.dot(v10), v13n.dot(v10)])
    param0 = A.inverse() @ v10_proj

    dtype = -1
    if param0[0] > 0.0 and param0[0] < 1.0 and param0[1] >= 0: dtype = 3
    else:
        v32 = v3 - v2
        v32n = v32.cross(n)
        v20 = v0 - v2
        A = ti.math.mat2([[v32.dot(v32), v32.dot(v32n)],
                          [v32n.dot(v32), v32n.dot(v32n)]])

        v20_proj = ti.math.vec2([v32.dot(v20), v32n.dot(v20)])
        param1 = A.inverse() @ v20_proj
        if param1[0]>0.0 and param1[0]<1.0 and param1[1]>=0: dtype = 4
        else:
            v31 = v1 - v3
            v31n = v31.cross(n)
            v30 = v0 - v3
            A = ti.math.mat2([[v31.dot(v31), v31.dot(v31n)],
                              [v31n.dot(v31), v31n.dot(v31n)]])
            v30_proj = ti.math.vec2([v31.dot(v30), v31n.dot(v30)])
            param2 = A.inverse() @ v30_proj

            if param2[0]>0.0 and param2[0]<1.0 and param2[1]>=0: dtype = 5
            else:
                if   param0[0] <= 0.0 and param2[0] >= 1.0: dtype = 0
                elif param1[0] <= 0.0 and param0[0] >= 1.0: dtype = 1
                elif param2[0] <= 0.0 and param1[0] >= 1.0: dtype = 2
                else:                                       dtype = 6

    return dtype


@ti.func
def d_PT( p: ti.math.vec3, x0: ti.math.vec3, x1: ti.math.vec3, x2: ti.math.vec3)->ti.math.vec3:
    # 1) Build vectors in the plane
    E0 = x1 - x0
    E1 = x2 - x0

    # 2) Plane normal (unnormalized)
    n = E0.cross(E1)

    # 3) Project p onto the plane
    #    dist = how far along n we must move to get to plane
    dist = (p - x0).dot(n) / n.dot(n)  # might be negative depending on orientation
    p_proj = p - dist * n

    # 4) Vector from x0 to the projected point
    v2 = p_proj - x0

    # 5) Barycentric formula
    dot00 = E0.dot(E0)
    dot01 = E0.dot(E1)
    dot02 = E0.dot(v2)
    dot11 = E1.dot(E1)
    dot12 = E1.dot(v2)

    denom = dot00 * dot11 - dot01 * dot01

    # Compute 'u' and 'v' in the usual formula
    u = (dot11 * dot02 - dot01 * dot12) / denom
    v = (dot00 * dot12 - dot01 * dot02) / denom

    # a0 + a1 + a2 = 1
    a0 = 1.0 - u - v
    a1 = u
    a2 = v

    return ti.math.vec3([a0, a1, a2])


@ti.func
def d_PE(p: ti.math.vec3, x0: ti.math.vec3, x1: ti.math.vec3) -> ti.math.vec2:
    """
    Return (a0, a1) = barycentric coords of p’s projection on the line x0->x1,
    with a0 + a1 = 1.
    If you clamp t to [0, 1], then p_proj is on the *segment* x0--x1.
    """
    E = x1 - x0
    denom = E.dot(E)

    a0 = a1 = 0
    # Degenerate check (if x0 == x1)
    if denom < 1e-12:
        # Both "barycentric" weights collapse to x0 anyway.
        a0 = 1

    # Unclamped t in the infinite line sense
    t = (p - x0).dot(E) / denom
    # print(t)
    # Clamp t to [0,1] for a line *segment*
    t_clamped = ti.max(0.0, ti.min(1.0, t))
    # print(t_clamped)
    # a0=1-t, a1=t
    # a1 = t_clamped
    # a0 = 1.0 - a1
    # print(a0, a1)
    test = ti.math.vec2([1.0 - t_clamped, t_clamped])
    # print(test)
    return test

@ti.func
def d_type_EE(v0: ti.math.vec3, v1: ti.math.vec3, v2: ti.math.vec3, v3: ti.math.vec3) -> ti.int32:
    u, v, w = v1 - v0, v3 - v2, v0 - v2

    a = u.dot(u)
    b = u.dot(v)
    c = v.dot(v)
    d = u.dot(w)
    e = v.dot(w)

    D = a * c - b * b
    tD = D
    sN = 0.0
    tN = 0.0
    default_case = 8

    sN = b * e - c * d
    if sN <= 0.0:
        tN = e
        tD = c
        default_case = 2

    elif sN >= D:
        tN = e + b
        tD = c
        default_case = 5
    else:
        tN = (a * e - b * d)
        if tN > 0.0 and tN < tD and (u.cross(v).dot(w) < 1e-4 or u.cross(v).dot(u.cross(v)) < 1.0e-20 * a * c):
            if sN < D / 2:
                tN = e
                tD = c
                default_case = 2
            else:
                tN = e + b
                tD = c
                default_case = 5


    if tN <= 0.0:
        if -d <= 0.0: default_case = 0
        elif -d >= a: default_case = 3
        else: default_case = 6


    elif tN >= tD:
        if (-d + b) <= 0.0: default_case = 1
        elif (-d + b) >= a: default_case = 4
        else: default_case = 7

    return default_case
