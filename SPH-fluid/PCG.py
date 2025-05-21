from math import isnan

from math_utils import *
import matplotlib.pyplot as plt
import numpy as np

@ti.data_oriented
class PCG:

    def __init__(self, n, n_dy):

        self.Ax = ti.Vector.field(3, float)
        self.Ap = ti.Vector.field(3, float)
        # self.s = ti.Vector.field(3, float)
        self.p = ti.Vector.field(3, float)
        self.z = ti.Vector.field(3, float)
        self.r = ti.Vector.field(3, float)

        self.Ax_dy = ti.Vector.field(3, float)
        self.Ap_dy = ti.Vector.field(3, float)
        # self.s = ti.Vector.field(3, float)
        self.p_dy = ti.Vector.field(3, float)
        self.z_dy = ti.Vector.field(3, float)
        self.r_dy = ti.Vector.field(3, float)

        ti.root.dense(ti.i, n).place(self.Ax, self.Ap, self.p, self.z, self.r)
        ti.root.dense(ti.i, n_dy).place(self.Ax_dy, self.Ap_dy, self.p_dy, self.z_dy, self.r_dy)

    @ti.kernel
    def applyPrecondition(self, z: ti.template(), hii: ti.template(), r: ti.template()):
        for i in z:
            z[i] = hii[i].inverse() @ r[i]

    def solve(self, x, b, hii, x_dy, b_dy, hii_dy, tol, matFreeAx):


        x.fill(0)
        x_dy.fill(0)

        self.r.copy_from(b)
        self.r_dy.copy_from(b_dy)
        self.applyPrecondition(self.z, hii, self.r)
        self.applyPrecondition(self.z_dy, hii_dy, self.r_dy)

        self.p.copy_from(self.z)
        self.p_dy.copy_from(self.z_dy)
        rs_old = dot(self.r, self.z)
        rs_old += dot(self.r_dy, self.z_dy)

        itrCnt = 0
        if rs_old < 1e-16:
            return itrCnt

        log_debug = []
        maxPCGIter = int(1e4)
        for i in range(maxPCGIter):

            matFreeAx(self.Ap, self.p, self.Ap_dy, self.p_dy)
            pAp = dot(self.p, self.Ap)
            pAp += dot(self.p_dy, self.Ap_dy)

            # if pAp < 0:
            #     print("A is negative definite!!")

            if pAp < 1e-16:
                break

            alpha = rs_old / pAp

            if isnan(alpha):
                print("PCG iter: ", itrCnt)
                print("pAp: ", pAp)
                print("alpha is a NaN!!")
                return  itrCnt
                # exit()
                # break
            # print("alpha: ", alpha)
            add(x, x, alpha, self.p)
            add(x_dy, x_dy, alpha, self.p_dy)
            add(self.r, self.r, -alpha, self.Ap)
            add(self.r_dy, self.r_dy, -alpha, self.Ap_dy)

            self.applyPrecondition(self.z, hii, self.r)
            self.applyPrecondition(self.z_dy, hii_dy, self.r_dy)
            rs_new = dot(self.r, self.z)
            rs_new += dot(self.r_dy, self.z_dy)
            #
            # if isnan(rs_new):
            #     print("rs_new is a NaN!!")

            itrCnt += 1
            log_debug.append(rs_new)
            if rs_new < tol:
                # print("rs_new: ", rs_new)
                break
            beta = rs_new / rs_old
            add(self.p, self.z, beta, self.p)
            add(self.p_dy, self.z_dy, beta, self.p_dy)
            rs_old = rs_new

        # if itrCnt == maxPCGIter:
        #     print("PCG failed to converge...")
        #     plt.plot(np.array(log_debug))
        #     # plt.yscale('log')
        #     plt.show()
        #     exit()

        # print("PCG iter: ", itrCnt)
        # x.copy_from(self.s)

        # add(x, x, alpha, self.p)
        scale(x, -1.0, x)
        scale(x_dy, -1.0, x_dy)

        return itrCnt