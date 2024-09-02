import numpy as np
import taichi as ti

ti.init(arch=ti.cuda, default_fp=ti.f64)

numVerts = 20

a = np.array([-2 for i in range(numVerts - 1)], dtype=np.float64)
b = np.array([10 for i in range(numVerts)], dtype=np.float64)
c = np.array([-2 for i in range(numVerts - 1)], dtype=np.float64)
tri_diag_mat = np.diag(b) + np.diag(a, k=1) + np.diag(c, k=-1)
x = np.ones(shape=(numVerts, 3), dtype=np.float64)
d = np.matmul(tri_diag_mat, x)
a = np.insert(a, 0, -1.0)

a_ti = ti.field(dtype=float, shape=a.shape)
a_ti.from_numpy(a)

b_ti = ti.field(dtype=float, shape=b.shape)
b_ti.from_numpy(b)

c_ti = ti.field(dtype=float, shape=c.shape)
c_ti.from_numpy(c)

x_ti = ti.Vector.field(n=3, dtype=float, shape=numVerts)

d_ti = ti.Vector.field(n=3, dtype=float, shape=numVerts)
d_ti.from_numpy(d)

# print(d_ti)

c_tilde = ti.field(dtype=float, shape=c.shape)
d_tilde = ti.Vector.field(n=3, dtype=ti.f64, shape=numVerts)

# print(d_tilde)
@ti.kernel
def ThomasAlgorithm_ti():

    # print(a[1])
    # print(b[0])
    # print(c[0])

    c_tilde[0] = c_ti[0] / b_ti[0]

    ti.loop_config(serialize=True)
    for i in range(numVerts - 2):
        # print(i)
        id = i + 1
        c_tilde[id] = c_ti[id] / (b_ti[id] - a_ti[id] * c_tilde[id - 1])
        # print(id)

    d_tilde[0] = d_ti[0] / b_ti[0]

    ti.loop_config(serialize=True)
    for i in range(numVerts - 1):
        # print(i)
        id = i + 1
        d_tilde[id] = (d_ti[id] - a_ti[id] * d_tilde[id - 1]) / (b_ti[id] - a_ti[id] * c_tilde[id - 1])
        # print(id)

    x_ti[numVerts - 1] = d_tilde[numVerts - 1]

    ti.loop_config(serialize=True)
    for i in range(numVerts - 1):
        # print(i)
        id = numVerts - 2 - i
        x_ti[id] = d_tilde[id] - c_tilde[id] * x_ti[id + 1]
        # print(id)
    # print(x_ti[0])

# def ThomasAlgorithm(a: np.array, b: np.array, c: np.array, x: np.array, d: np.array):
#     numVerts = x.shape[0]
#     # print(numVerts)
#
#     c_tilde = np.zeros(numVerts - 1)
#     d_tilde = np.zeros(shape=(numVerts, 3))
#
#     c_tilde[0] = c[0] / b[0]
#
#     for i in range(1, numVerts - 1):
#         # print(i)
#         c_tilde[i] = c[i] / (b[i] - a[i] * c_tilde[i - 1])
#
#     d_tilde[0] = d[0] / b[0]
#
#     for i in range(1, numVerts):
#         # print(i)
#         d_tilde[i] = (d[i] - a[i] * d_tilde[i - 1]) / (b[i] - a[i] * c_tilde[i - 1])
#
#     x[numVerts - 1] = d_tilde[numVerts - 1]
#     for i in range(numVerts - 2, -1, -1):
#         # print(i)
#         x[i] = d_tilde[i] - c_tilde[i] * x[i + 1]


# x = np.zeros(numVerts)
ThomasAlgorithm_ti()
print(x_ti)