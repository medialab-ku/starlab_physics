import taichi as ti
import numpy as np
ti.init(arch=ti.gpu)

num_partition = 2
dim = 1
max_size = 2 * dim

@ti.func
def llt(A: ti.template()):

    L = ti.Matrix.identity(dt=float, n=max_size)
    return L

@ti.func
def forward_substitution(L: ti.template(), b: ti.template()):

    y = ti.Vector.one(dt=float, n=max_size)
    return y

@ti.func
def backward_substitution(L: ti.template(), y: ti.template()):

    x = ti.Vector.one(dt=float, n=max_size)
    return x

@ti.kernel
def launch_kernel():

  # for i in range(num_partition):
  A = ti.Matrix.identity(dt=float, n=max_size)
  b = ti.Vector.one(dt=float, n=max_size)

  # L L^t = A
  L = llt(A)

  #L (L^t) x = b
  #  (L^t) x = L(^-1) b
  y = forward_substitution(L, b)

  #  (L^t) x = L(^-1) b
  #        x = (L^t)^(-1) L(^-1) b
  x = backward_substitution(L.transpose(), y)



launch_kernel()