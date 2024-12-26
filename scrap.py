import taichi as ti

ti.init(arch=ti.gpu)



A = ti.Matrix.field(n=3,m=3, shape=(10, 3, 3), dtype=float)

@ti.func
def svd3x2(F):

    U, sigma2, V_T = ti.svd(F @ F.transpose())

    if U.determinant() < 0:
        for i in ti.static(range(3)):
            U[i, 2] *= -1
        # sig[2, 2] = -sig[2, 2]
        # sig[2, 2] = -sig[2, 2]

    V, _, U_T = ti.svd(F.transpose() @ F)
    if V.determinant() < 0:
        for i in ti.static(range(2)):
            V[i, 1] *= -1

    sigma = ti.Matrix.cols([[ti.sqrt(sigma2[0, 0]), 0, 0], [0, ti.sqrt(sigma2[1, 1]), 0]])

    return U, sigma, V.transpose()

@ti.kernel
def foo():

    # print(A[0, 0, 0])
    v0_uv = ti.Vector([1., 0.], float)
    v1_uv = ti.Vector([0., 1.], float)

    Dm_2d = ti.Matrix.cols([v0_uv, v1_uv])

    v0 = ti.Vector([1., 0., 0.0], float)
    v1 = ti.Vector([0., 0., 2.0], float)
    Ds_3d = ti.Matrix.cols([v0, v1])
    F = Ds_3d @ Dm_2d.inverse()
    #
    U, sigma, V = svd3x2(F)

    # print(sigma)

    V_ext = ti.Matrix.cols([[V[0, 0], V[1, 0], 0.0], [V[0, 1], V[1, 1], 0.0]])
    R = U @ V_ext

    print(R @ Dm_2d)
    # print(R @ v1_uv)
    #
    # print(U @ sigma @ V_T)
    # print(R.transpose() @ R)

foo()