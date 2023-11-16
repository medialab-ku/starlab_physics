import taichi as ti

@ti.func
def b_C0(d: ti.f32, dHat: ti.f32)-> ti.f32:
    return -ti.math.log(d / dHat)

@ti.func
def g_bC0(d1: ti.f32)-> ti.f32:
    return -1.0 / d1

@ti.func
def H_bC0(d1: ti.f32) -> ti.f32:
    return 1.0 / d1 * d1

@ti.func
def b_C1(d: ti.f32, dHat: ti.f32)-> ti.f32:
    return (d - dHat) * ti.math.log(d / dHat)

@ti.func
def g_bC1(d1: ti.f32, dHat1: ti.f32)-> ti.f32:
    return ti.math.log(d1 / dHat1) + (d1 - dHat1) / d1


@ti.func
def H_bC1(d1: ti.f32, dHat1: ti.f32)-> ti.f32:
    return 2.0 / d1 - 1.0 / (d1 * d1) * (d1 - dHat1)


@ti.func
def b_C2(d: ti.f32, dHat: ti.f32)-> ti.f32:
    return -(d - dHat) * (d - dHat) * ti.math.log(d / dHat)


@ti.func
def g_bC2(d1: ti.f32, dHat1: ti.f32)-> ti.f32:
    t2 = d1 - dHat1
    return t2 * ti.math.log(d1 / dHat1) * -2.0 - (t2 * t2)


@ti.func
def H_bC2(d1: ti.f32, dHat1: ti.f32)-> ti.f32:

    t2 = d1 - dHat1
    return (ti.math.log(d1 / dHat1) * -2.0 - t2 * 4.0 / d1) + 1.0 / (d1 * d1) * (t2 * t2)

@ti.func
def compute_b(d: ti.f32, dHat: ti.f32) -> ti.f32:
    return b_C2(d, dHat)

@ti.func
def compute_g_b(d: ti.f32, dHat: ti.f32) -> ti.f32:
  return g_bC2(d, dHat)

@ti.func
def compute_H_b(d: ti.f32, dHat: ti.f32) -> ti.f32:

    return H_bC2(d, dHat)


