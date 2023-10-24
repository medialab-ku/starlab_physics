import taichi as ti


@ti.func
def d_type_PT(v0: ti.math.vec3, v1: ti.math.vec3, v2: ti.math.vec3, v3: ti.math.vec3) -> ti.int32:

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
    if param0[0]>0.0 and param0[0]<1.0 and param0[1]>=0: dtype = 3
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

@ti.func
def d_PP(v0: ti.math.vec3, v1: ti.math.vec3) -> ti.f32:
    return (v0 - v1).dot(v0 - v1)

@ti.func
def d_PE(v0: ti.math.vec3, v1: ti.math.vec3, v2: ti.math.vec3) -> ti.f32:
    a = (v1 - v0).cross(v2 - v0)
    v12 = v2 - v1
    return a.dot(a) / v12.dot(v12)

@ti.func
def d_PT(v0: ti.math.vec3, v1: ti.math.vec3, v2: ti.math.vec3, v3: ti.math.vec3) -> ti.f32:

    b = (v2 - v1).cross(v3 - v1)
    aTb = (v0 - v1).dot(b)
    return aTb * aTb / b.dot(b)

@ti.func
def d_EE(v0: ti.math.vec3, v1: ti.math.vec3, v2: ti.math.vec3, v3: ti.math.vec3) -> ti.f32:

    b = (v1 - v0).cross(v3 - v2)
    aTb = (v2 - v0).dot(b)

    return aTb * aTb / b.dot(b)
@ti.func
def g_PP(v0: ti.math.vec3, v1: ti.math.vec3):
    grad = 2.0 * (v0 - v1)
    return grad, -grad
@ti.func
def g_PE(v0: ti.math.vec3, v1: ti.math.vec3, v2: ti.math.vec3):
    v01, v02, v03 = v0[0], v0[1], v0[2]
    v11, v12, v13 = v1[0], v1[1], v1[2]
    v21, v22, v23 = v2[0], v2[1], v2[2]

    t17 = -v11 + v01
    t18 = -v12 + v02
    t19 = -v13 + v03
    t20 = -v21 + v01
    t21 = -v22 + v02
    t22 = -v23 + v03
    t23 = -v21 + v11
    t24 = -v22 + v12
    t25 = -v23 + v13
    t42 = 1.0 / ((t23 * t23 + t24 * t24) + t25 * t25)
    t44 = t17 * t21 + -(t18 * t20)
    t45 = t17 * t22 + -(t19 * t20)
    t46 = t18 * t22 + -(t19 * t21)
    t43 = t42 * t42
    t50 = (t44 * t44 + t45 * t45) + t46 * t46
    t51 = (v11 * 2.0 + -(v21 * 2.0)) * t43 * t50
    t52 = (v12 * 2.0 + -(v22 * 2.0)) * t43 * t50
    t43 = (v13 * 2.0 + -(v23 * 2.0)) * t43 * t50

    g0 = t42 * (t24 * t44 * 2.0 + t25 * t45 * 2.0)
    g1 = -t42 * (t23 * t44 * 2.0 - t25 * t46 * 2.0)
    g2 = -t42 * (t23 * t45 * 2.0 + t24 * t46 * 2.0)

    gvec31 = ti.math.vec3([g0, g1, g2])

    g3 = -t51 - t42 * (t21 * t44 * 2.0 + t22 * t45 * 2.0)
    g4 = -t52 + t42 * (t20 * t44 * 2.0 - t22 * t46 * 2.0)
    g5 = -t43 + t42 * (t20 * t45 * 2.0 + t21 * t46 * 2.0)
    gvec32 = ti.math.vec3([g3, g4, g5])
    g6 = t51 + t42 * (t18 * t44 * 2.0 + t19 * t45 * 2.0)
    g7 = t52 - t42 * (t17 * t44 * 2.0 - t19 * t46 * 2.0)
    g8 = t43 - t42 * (t17 * t45 * 2.0 + t18 * t46 * 2.0)
    gvec33 = ti.math.vec3([g6, g7, g8])

    return gvec31, gvec32, gvec33

@ti.func
def g_PT(v0: ti.math.vec3, v1: ti.math.vec3, v2: ti.math.vec3, v3: ti.math.vec3):
    v01, v02, v03 = v0[0], v0[1], v0[2]
    v11, v12, v13 = v1[0], v1[1], v1[2]
    v21, v22, v23 = v2[0], v2[1], v2[2]
    v31, v32, v33 = v3[0], v3[1], v3[2]

    t11 = -v11 + v01
    t12 = -v12 + v02
    t13 = -v13 + v03
    t14 = -v21 + v11
    t15 = -v22 + v12
    t16 = -v23 + v13
    t17 = -v31 + v11
    t18 = -v32 + v12
    t19 = -v33 + v13
    t20 = -v31 + v21
    t21 = -v32 + v22
    t22 = -v33 + v23
    t32 = t14 * t18 + -(t15 * t17)
    t33 = t14 * t19 + -(t16 * t17)
    t34 = t15 * t19 + -(t16 * t18)
    t43 = 1.0 / ((t32 * t32 + t33 * t33) + t34 * t34)
    t45 = (t13 * t32 + t11 * t34) + -(t12 * t33)
    t44 = t43 * t43
    t46 = t45 * t45
    g0 = t34 * t43 * t45 * 2.0
    g1 = t33 * t43 * t45 * -2.0
    g2 = t32 * t43 * t45 * 2.0

    gvec31 = ti.math.vec3([g0, g1, g2])

    t45 *= t43
    g3 = -t44 * t46 * (t21 * t32 * 2.0 + t22 * t33 * 2.0) - t45 * ((t34 + t12 * t22) - t13 * t21) * 2.0
    t43 = t44 * t46
    g4 = t43 * (t20 * t32 * 2.0 - t22 * t34 * 2.0) + t45 * ((t33 + t11 * t22) - t13 * t20) * 2.0
    g5 = t43 * (t20 * t33 * 2.0 + t21 * t34 * 2.0) - t45 * ((t32 + t11 * t21) - t12 * t20) * 2.0
    gvec32 = ti.math.vec3([g3, g4, g5])
    g6 = t45 * (t12 * t19 - t13 * t18) * 2.0 + t43 * (t18 * t32 * 2.0 + t19 * t33 * 2.0)
    g7 = t45 * (t11 * t19 - t13 * t17) * -2.0 - t43 * (t17 * t32 * 2.0 - t19 * t34 * 2.0)
    g8 = t45 * (t11 * t18 - t12 * t17) * 2.0 - t43 * (t17 * t33 * 2.0 + t18 * t34 * 2.0)
    gvec33 = ti.math.vec3([g6, g7, g8])
    g9 = t45 * (t12 * t16 - t13 * t15) * -2.0 - t43 * (t15 * t32 * 2.0 + t16 * t33 * 2.0)
    g10 = t45 * (t11 * t16 - t13 * t14) * 2.0 + t43 * (t14 * t32 * 2.0 - t16 * t34 * 2.0)
    g11 = t45 * (t11 * t15 - t12 * t14) * -2.0 + t43 * (t14 * t33 * 2.0 + t15 * t34 * 2.0)
    gvec34 = ti.math.vec3([g9, g10, g11])

    return gvec31, gvec32, gvec33, gvec34

@ti.func
def g_EE(v0: ti.math.vec3, v1: ti.math.vec3, v2: ti.math.vec3, v3: ti.math.vec3):
    v01, v02, v03 = v0[0], v0[1], v0[2]
    v11, v12, v13 = v1[0], v1[1], v1[2]
    v21, v22, v23 = v2[0], v2[1], v2[2]
    v31, v32, v33 = v3[0], v3[1], v3[2]

    t11 = -v11 + v01
    t12 = -v12 + v02
    t13 = -v13 + v03
    t14 = -v21 + v01
    t15 = -v22 + v02
    t16 = -v23 + v03
    t17 = -v31 + v21
    t18 = -v32 + v22
    t19 = -v33 + v23
    t32 = t14 * t18
    t33 = t15 * t17
    t34 = t14 * t19
    t35 = t16 * t17
    t36 = t15 * t19
    t37 = t16 * t18
    t44 = t11 * t18 + -(t12 * t17)
    t45 = t11 * t19 + -(t13 * t17)
    t46 = t12 * t19 + -(t13 * t18)
    t75 = 1.0 / ((t44 * t44 + t45 * t45) + t46 * t46)
    t77 = (t16 * t44 + t14 * t46) + -(t15 * t45)
    t76 = t75 * t75
    t78 = t77 * t77
    t79 = (t12 * t44 * 2.0 + t13 * t45 * 2.0) * t76 * t78
    t80 = (t11 * t45 * 2.0 + t12 * t46 * 2.0) * t76 * t78
    t81 = (t18 * t44 * 2.0 + t19 * t45 * 2.0) * t76 * t78
    t18 = (t17 * t45 * 2.0 + t18 * t46 * 2.0) * t76 * t78
    t83 = (t11 * t44 * 2.0 + -(t13 * t46 * 2.0)) * t76 * t78
    t19 = (t17 * t44 * 2.0 + -(t19 * t46 * 2.0)) * t76 * t78
    t76 = t75 * t77
    g0 = -t81 + t76 * ((-t36 + t37) + t46) * 2.0
    g1 = t19 - t76 * ((-t34 + t35) + t45) * 2.0
    g2 = t18 + t76 * ((-t32 + t33) + t44) * 2.0

    gvec31 = ti.math.vec3(g0, g1, g2)

    g3 = t81 + t76 * (t36 - t37) * 2.0
    g4 = -t19 - t76 * (t34 - t35) * 2.0
    g5 = -t18 + t76 * (t32 - t33) * 2.0

    gvec32 = ti.math.vec3(g3, g4, g5)

    t17 = t12 * t16 + -(t13 * t15)
    g6 = t79 - t76 * (t17 + t46) * 2.0
    t18 = t11 * t16 + -(t13 * t14)
    g7 = -t83 + t76 * (t18 + t45) * 2.0
    t19 = t11 * t15 + -(t12 * t14)
    g8 = -t80 - t76 * (t19 + t44) * 2.0

    gvec33 = ti.math.vec3(g6, g7, g8)

    g9 = -t79 + t76 * t17 * 2.0
    g10 = t83 - t76 * t18 * 2.0
    g11 = t80 + t76 * t19 * 2.0

    gvec34 = ti.math.vec3(g9, g10, g11)

    return gvec31, gvec32, gvec33, gvec34



@ti.func
def compute_q(input: ti.f32, eps_x: ti.f32)-> ti.f32:

    input_div_eps_x = input / eps_x
    return (-input_div_eps_x + 2.0) * input_div_eps_x


@ti.func
def compute_q_g(input: ti.f32, eps_x: ti.f32)-> ti.f32:

    one_div_eps_x = 1.0 / eps_x
    return 2.0 * one_div_eps_x * (-one_div_eps_x * input + 1.0)


@ti.func
def compute_q_H(input: ti.f32, eps_x: ti.f32) -> ti.f32:

    return -2.0 / (eps_x * eps_x)



@ti.func
def compute_eps_x(v0: ti.math.vec3, v1: ti.math.vec3, v2: ti.math.vec3, v3: ti.math.vec3) -> ti.f32:
    return 1.0e-3 * (v0 - v1).dot(v0 - v1) * (v2 - v3).dot(v2 - v3)

@ti.func
def compute_e(v0: ti.math.vec3, v1: ti.math.vec3, v2: ti.math.vec3, v3: ti.math.vec3, eps_x: ti.f32) -> ti.f32:

    cross = (v1 - v0).cross(v3 - v2)
    sq_norm = cross.dot(cross)

    e = 1.0
    if sq_norm < eps_x:
        compute_q(sq_norm, eps_x)

    return e

@ti.func
def compute_e_g(v0: ti.math.vec3, v1: ti.math.vec3, v2: ti.math.vec3, v3: ti.math.vec3, eps_x: ti.f32):

    cross = (v1 - v0).cross(v3 - v2)
    sq_norm = cross.dot(cross)

    g0 = ti.math.vec3(0.0, 0.0, 0.0)
    g1 = ti.math.vec3(0.0, 0.0, 0.0)
    g2 = ti.math.vec3(0.0, 0.0, 0.0)
    g3 = ti.math.vec3(0.0, 0.0, 0.0)

    if sq_norm < eps_x:
        q_g = compute_q_g(sq_norm, eps_x)
        g0, g1, g2, g3 = computeEECrossSqNormGradient(v0, v1, v2, v3)
        g0 *= q_g
        g1 *= q_g
        g2 *= q_g
        g3 *= q_g

    return g0, g1, g2, g3

@ti.func
def computeEECrossSqNormGradient(v0: ti.math.vec3, v1: ti.math.vec3, v2: ti.math.vec3, v3: ti.math.vec3):

    t8 = -v1.x + v0.x
    t9 = -v1.y + v0.y
    t10 = -v1.z + v0.z
    t11 = -v3.x + v2.x
    t12 = -v3.y + v2.y
    t13 = -v3.z + v2.z
    t23 = t8 * t12 + -(t9 * t11)
    t24 = t8 * t13 + -(t10 * t11)
    t25 = t9 * t13 + -(t10 * t12)
    t26 = t8 * t23 * 2.0
    t27 = t9 * t23 * 2.0
    t28 = t8 * t24 * 2.0
    t29 = t10 * t24 * 2.0
    t30 = t9 * t25 * 2.0
    t31 = t10 * t25 * 2.0
    t32 = t11 * t23 * 2.0
    t33 = t12 * t23 * 2.0
    t23 = t11 * t24 * 2.0
    t10 = t13 * t24 * 2.0
    t9 = t12 * t25 * 2.0
    t8 = t13 * t25 * 2.0

    g0 = t33 + t10
    g1 = -t32 + t8
    g2 = -t23 - t9

    gvec0 = ti.math.vec3(g0, g1, g2)

    g3 = -t33 - t10
    g4 = t32 - t8
    g5 = t23 + t9
    gvec1 = ti.math.vec3(g3, g4, g5)
    g6 = -t27 - t29
    g7 = t26 - t31
    g8 = t28 + t30
    gvec2 = ti.math.vec3(g6, g7, g8)
    g9 = t27 + t29
    g10 = -t26 + t31
    g11 = -t28 - t30
    gvec3 = ti.math.vec3(g9, g10, g11)

    return gvec0, gvec1, gvec2, gvec3