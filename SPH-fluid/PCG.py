from math_utils import *
import matplotlib.pyplot as plt
import numpy as np

@ti.data_oriented
class PCG:

    def __init__(self, n):

        self.Ax = ti.Vector.field(3, float)
        self.Ap = ti.Vector.field(3, float)
        # self.s = ti.Vector.field(3, float)
        self.p = ti.Vector.field(3, float)
        self.z = ti.Vector.field(3, float)
        self.r = ti.Vector.field(3, float)

        ti.root.dense(ti.i, n).place(self.Ax, self.Ap, self.p, self.z, self.r)

    @ti.kernel
    def applyPrecondition(self, z: ti.template(), hii: ti.template(), r: ti.template()):
        for i in z:
            z[i] = hii[i].inverse() @ r[i]

    def solve(self, x, b, hii, tol, matFreeAx):

        x.fill(0.0)
        # self.applyPrecondition(x, hii, b)
        # # self.s.copy_from(x)
        # matFreeAx(self.Ax, x)
        #
        self.r.copy_from(b)
        # add(self.r, b, -1.0, self.Ax)
        self.applyPrecondition(self.z, hii, self.r)

        self.p.copy_from(self.z)
        rs_old = dot(self.r, self.z)

        # if rs_old > tol:
        log_debug = []
        itrCnt = 0
        maxPCGIter = int(1e4)
        for i in range(maxPCGIter):

            matFreeAx(self.Ap, self.p)
            pAp = dot(self.p, self.Ap)

            # print(pAp)
            #
            if abs(pAp) < 1e-4:
                break

            alpha = rs_old / pAp
            add(x, x, alpha, self.p)
            add(self.r, self.r, -alpha, self.Ap)

            self.applyPrecondition(self.z, hii, self.r)
            rs_new = dot(self.r, self.z)

            itrCnt += 1
            log_debug.append(rs_new)
            # print(rs_new)
            if rs_new < tol:
                break
            beta = rs_new / rs_old
            add(self.p, self.z, beta, self.p)
            rs_old = rs_new

        if itrCnt == maxPCGIter:
            print("PCG failed to converge...")
            plt.plot(np.array(log_debug))
            plt.yscale('log')
            plt.show()
            exit()

        # print("PCG iter: ", itrCnt)
        # x.copy_from(self.s)