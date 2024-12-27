import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

num_partition = 3
dim = 1
max_size = 3 * dim

@ti.func
def llt(A: ti.template()):
    L = ti.Matrix.zero(ti.f32, max_size, max_size)

    for i in ti.static(range(max_size)):
        for j in ti.static(range(i + 1)):
            # sum_ = A[i, j] - Σ(L[i, k] * L[j, k]), (k = 0, 1, ... , j-1)
            sum_ = A[i, j]
            for k in ti.static(range(j)):
                sum_ -= L[i, k] * L[j, k]

            if i == j:
                L[i, j] = ti.sqrt(sum_)
            else:
                L[i, j] = sum_ / L[j, j]

    return L

@ti.func
def forward_substitution(L: ti.template(), b: ti.template()):
    y = ti.Vector.zero(ti.f32, max_size)

    for i in ti.static(range(max_size)):
        sum_ = 0.0
        for j in ti.static(range(i)):
            sum_ += L[i, j] * y[j]
        y[i] = (b[i] - sum_) / L[i, i]

    return y

@ti.func
def backward_substitution(Lt: ti.template(), y: ti.template()):
    x = ti.Vector.zero(ti.f32, max_size)

    for i in ti.static(range(max_size - 1, -1, -1)):
        sum_ = 0.0
        for j in ti.static(range(i + 1, max_size)):
            sum_ += Lt[i, j] * x[j]
        x[i] = (y[i] - sum_) / Lt[i, i]

    return x

@ti.kernel
def launch_kernel():
    A = ti.Matrix([
        [4.0, 2.0, -2.0],
        [2.0, 6.0,  0.0],
        [-2.0, 0.0, 5.0],
    ])
    b = ti.Vector([2.0, 1.0, 3.0])

    # L L^t = A
    L = llt(A)

    #L (L^t) x = b
    #  (L^t) x = L(^-1) b
    y = forward_substitution(L, b)

    #  (L^t) x = L(^-1) b
    #        x = (L^t)^(-1) L(^-1) b
    x = backward_substitution(L.transpose(), y)

    print("Result")
    print("A = ", A)
    print("b = ", b)
    print("L = ", L)
    print("y = ", y)
    print("x = ", x)

def kernel_wrapper():
    launch_kernel()

kernel_wrapper()