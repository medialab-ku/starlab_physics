import taichi as ti

@ti.data_oriented
class ConjugateGradient:

    def __init__(self):

        print("ConjugateGradient initialized...")
        self.cg_iter = 0
        self.cg_err = 0.0

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
    def preconditioning_jacobi(self, mesh: ti.template(), z: ti.template(), r: ti.template()):
        for i in z:
            h = mesh.hii[i] + mesh.hii_e[i]
            z[i] = h.inverse() @ r[i]

    @ti.kernel
    def preconditioning_tri(self, mesh: ti.template(), z: ti.template(), r: ti.template()):

        for di in mesh.x_dup:
            vi = mesh.dup_to_ori[di]

            mesh.b_dup[di] = mesh.hii[vi] + mesh.hii_e[vi]
            mesh.d_dup[di] = r[vi]

        n_part = (mesh.partition_offset.shape[0] - 1)

        # for di in mesh.x_dup:
        #     b = mesh.b_dup[di]
        #     mesh.dx_dup[di] = b.inverse() @ mesh.d_dup[di]


        for pi in range(n_part):

            size   = mesh.vert_offset[pi + 1] - mesh.vert_offset[pi]
            offset = mesh.vert_offset[pi]

            # Thomas algorithm
            # https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm

            # for j in ti.static(range(3)):
            mesh.c_dup_tilde[offset] = ti.math.inverse(mesh.b_dup[offset]) @ mesh.c_dup[offset]
            ti.loop_config(serialize=True)
            for id in range(size):  # lb+1 ~ ub-1
                i = id + offset
                tmp = ti.math.inverse(mesh.b_dup[i] - mesh.a_dup[i] * mesh.c_dup_tilde[i - 1])

                mesh.c_dup_tilde[i] = tmp @ mesh.c_dup[i]
            #
            mesh.d_dup_tilde[offset] = ti.math.inverse(mesh.b_dup[offset]) @ mesh.d_dup[offset]
            ti.loop_config(serialize=True)
            for id in range(1, size):  # lb+1 ~ ub
                i = id + offset
                tmp = ti.math.inverse(mesh.b_dup[i] - mesh.a_dup[i] * mesh.c_dup_tilde[i - 1])
                mesh.d_dup_tilde[i] = tmp @ (mesh.d_dup[i] - mesh.a_dup[i] @ mesh.d_dup_tilde[i - 1])

            mesh.dx_dup[offset + size - 1] = mesh.d_dup_tilde[offset + size - 1]
            ti.loop_config(serialize=True)
            for i in range(0, size - 1):
                idx = size - 2 - i + offset  # ub-1 ~ lb
                mesh.dx_dup[idx] = mesh.d_dup_tilde[idx] - mesh.c_dup_tilde[idx] @ mesh.dx_dup[idx + 1]

        z.fill(0.0)
        for di in mesh.x_dup:
            vi = mesh.dup_to_ori[di]
            z[vi] += mesh.dx_dup[di]

        for i in z:
            z[i] /= mesh.num_dup[i]

    @ti.kernel
    def preconditioning_identity(self, mesh: ti.template(), z: ti.template(), r: ti.template()):
        for i in z:
            z[i] = r[i]

    @ti.kernel
    def compute_mat_free_Ax(self, mesh: ti.template(), Ax: ti.template(), x: ti.template()):

        Ax.fill(0.0)
        for i in x:
            Ax[i] += mesh.hii[i] @ x[i]

        for i in range(mesh.num_edges):
            vi_d, vj_d = mesh.eid_dup[2 * i + 0], mesh.eid_dup[2 * i + 1]
            vi, vj = mesh.dup_to_ori[vi_d], mesh.dup_to_ori[vj_d]
            hij = mesh.hij[i]
            xij = x[vi] - x[vj]
            hijxij = hij @ xij
            Ax[vi] += hijxij
            Ax[vj] -= hijxij

    def run(self, precond_type, mesh: ti.template(), max_cg_iter, threshold):

        mesh.dx.fill(0.0)
        # r_0 = b - Ax_0
        mesh.r.copy_from(mesh.b)

        if precond_type == 0:
            self.preconditioning_tri(mesh, mesh.z, mesh.r)
        elif precond_type == 1:
            self.preconditioning_jacobi(mesh, mesh.z, mesh.r)
        # print(mesh.b)
        # p_0 = r_0
        mesh.p.copy_from(mesh.z)
        self.cg_iter = 0
        # r_normSq = self.dot_product(mesh.r, mesh.r)
        # if r_normSq > 1e-6:
        while True:
            # Ap_k = A * p_k
            self.compute_mat_free_Ax(mesh, mesh.Ax, mesh.p)

            # alpha = r_k ^T r_k / p_k ^T (Ap_k)
            self.cg_err = self.dot_product(mesh.r, mesh.r)
            # print(r_normSq)
            if self.cg_err < threshold or self.cg_iter >= max_cg_iter:
                break

            rTz = self.dot_product(mesh.r, mesh.z)
            alpha = rTz / self.dot_product(mesh.p, mesh.Ax)

            # dx_k+1 = dx_k + alpha * p_k
            self.vector_add(mesh.dx, mesh.dx, mesh.p, alpha)
            # r_k+1 = r_k + alpha * Ap_k
            self.vector_add(mesh.r_next, mesh.r, mesh.Ax, -alpha)

            if precond_type == 0:
                self.preconditioning_tri(mesh, mesh.z_next, mesh.r_next)
            elif precond_type == 1:
                self.preconditioning_jacobi(mesh, mesh.z_next, mesh.r_next)
            # beta = r_k+1 ^T r_k+1 / z_k ^T r_k
            beta = self.dot_product(mesh.z_next, mesh.r_next) / rTz

            # p_k+1 = z_k+1 + beta * p_k+1
            self.vector_add(mesh.p, mesh.z_next, mesh.p, beta)

            # r_k = r_k+1
            mesh.r.copy_from(mesh.r_next)
            mesh.z.copy_from(mesh.z_next)
            self.cg_iter += 1