import taichi as ti

@ti.data_oriented
class ConjugateGradient:

    def __init__(self):

        print("ConjugateGradient initialized...")


    @ti.kernel
    def vector_add(self, ret: ti.template(), x: ti.template(), y: ti.template(), scalar: float):
        for i in x:
            ret[i] = x[i] + scalar * y[i]

    @ti.kernel
    def dot_product(self, x: ti.template(), y: ti.template()) -> float:
        ret = 0.0
        for i in x:
            ret += x[i].dot(y[i])
        return ret

    @ti.kernel
    def compute_mat_free_Ax(self, mesh: ti.template(), Ax: ti.template(), x: ti.template()):

        Ax.fill(0.0)
        for i in range(mesh.num_edges):
            Ax[i] += mesh.hii[i] @ x[i]

        for i in range(mesh.num_edges):
            vi, vj = mesh.eid_field[i, 0], mesh.eid_field[i, 1]
            hij = mesh.hij[i]
            xij = x[vi] - x[vj]
            hijxij = hij @ xij
            Ax[vi] += hijxij
            Ax[vj] -= hijxij

    def run(self, mesh: ti.template(), max_cg_iter, threshold):

        mesh.dx.fill(0.0)
        # r_0 = b - Ax_0
        mesh.r.copy_from(mesh.b)
        # print(mesh.b)
        # p_0 = r_0
        mesh.p.copy_from(mesh.r)
        cg_iter = 0
        # r_normSq = self.dot_product(mesh.r, mesh.r)
        # if r_normSq > 1e-6:
        while True:
            # Ap_k = A * p_k
            self.compute_mat_free_Ax(mesh, mesh.Ax, mesh.p)

            # alpha = r_k ^T r_k / p_k ^T (Ap_k)
            r_normSq = self.dot_product(mesh.r, mesh.r)

            # print(r_normSq)
            if r_normSq < threshold or cg_iter >= max_cg_iter:
                return r_normSq, cg_iter

            alpha = r_normSq / self.dot_product(mesh.p, mesh.Ax)

            # dx_k+1 = dx_k + alpha * p_k
            self.vector_add(mesh.dx, mesh.dx, mesh.p, alpha)

            # r_k+1 = r_k + alpha * Ap_k
            self.vector_add(mesh.r_next, mesh.r, mesh.Ax, -alpha)

            # beta = r_k+1 ^T r_k+1 / r_k ^T r_k
            beta = self.dot_product(mesh.r_next, mesh.r_next) / r_normSq

            # p_k+1 = r_k+1 + beta * p_k+1
            self.vector_add(mesh.p, mesh.r_next, mesh.p, beta)

            # r_k = r_k+1
            mesh.r.copy_from(mesh.r_next)
            cg_iter += 1