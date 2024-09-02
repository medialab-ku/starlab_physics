import taichi as ti
import framework.collision.distance as di


@ti.func
def __vt_st(compliance, vi_d, fi_s, mesh_dy, mesh_st, dHat, vt_st_cache_size, vt_st_pair, vt_st_pair_num, vt_st_pair_g, vt_st_pair_schur):

    v0 = vi_d
    v1 = mesh_st.face_indices[3 * fi_s + 0]
    v2 = mesh_st.face_indices[3 * fi_s + 1]
    v3 = mesh_st.face_indices[3 * fi_s + 2]

    x0 = mesh_dy.verts.y[v0]
    x1 = mesh_st.verts.x[v1]
    x2 = mesh_st.verts.x[v2]
    x3 = mesh_st.verts.x[v3]

    g0 = ti.math.vec3(0.0)
    g1 = ti.math.vec3(0.0)
    g2 = ti.math.vec3(0.0)
    g3 = ti.math.vec3(0.0)

    d = dHat
    dtype = di.d_type_PT(x0, x1, x2, x3)

    if dtype == 0:
        d = di.d_PP(x0, x1)
        g0, g1 = di.g_PP(x0, x1)

    elif dtype == 1:
        d = di.d_PP(x0, x2)
        g0, g2 = di.g_PP(x0, x2)

    elif dtype == 2:
        d = di.d_PP(x0, x3)
        g0, g3 = di.g_PP(x0, x3)

    elif dtype == 3:
        d = di.d_PE(x0, x1, x2)
        g0, g1, g2 = di.g_PE(x0, x1, x2)

    elif dtype == 4:
        d = di.d_PE(x0, x2, x3)
        g0, g2, g3 = di.g_PE(x0, x2, x3)

    elif dtype == 5:
        d = di.d_PE(x0, x1, x3)
        g0, g1, g3 = di.g_PE(x0, x1, x3)

    elif dtype == 6:
        d = di.d_PT(x0, x1, x2, x3)
        g0, g1, g2, g3 = di.g_PT(x0, x1, x2, x3)

    if d < dHat:

        schur = mesh_dy.verts.m_inv[v0] * g0.dot(g0)
        k = 1e8
        ld = k * (dHat - d) / (k * schur + 1.0)
        p0 = mesh_dy.verts.y[v0] + mesh_dy.verts.m_inv[v0] * ld * g0
        mesh_dy.verts.gii[v0] += mesh_dy.verts.m_inv[v0] * compliance * (mesh_dy.verts.y[v0] - p0)
        mesh_dy.verts.hii[v0] += mesh_dy.verts.m_inv[v0] * compliance
        # mesh_dy.verts.nc[v0] += 1

        if vt_st_pair_num[vi_d] < vt_st_cache_size:
            vt_st_pair[vi_d, vt_st_pair_num[vi_d], 0] = fi_s
            vt_st_pair[vi_d, vt_st_pair_num[vi_d], 1] = dtype
            vt_st_pair_g[vi_d, vt_st_pair_num[vi_d], 0] = g0
            vt_st_pair_g[vi_d, vt_st_pair_num[vi_d], 1] = g1
            vt_st_pair_g[vi_d, vt_st_pair_num[vi_d], 2] = g2
            vt_st_pair_g[vi_d, vt_st_pair_num[vi_d], 3] = g3
            vt_st_pair_schur[vi_d, vt_st_pair_num[vi_d]] = schur
            ti.atomic_add(vt_st_pair_num[vi_d], 1)

@ti.func
def __tv_st(compliance, fi_d, vi_s, mesh_dy, mesh_st, dHat, tv_st_cache_size, tv_st_pair, tv_st_pair_num, tv_st_pair_g, tv_st_pair_schur):

    v0 = vi_s
    v1 = mesh_dy.face_indices[3 * fi_d + 0]
    v2 = mesh_dy.face_indices[3 * fi_d + 1]
    v3 = mesh_dy.face_indices[3 * fi_d + 2]

    x0 = mesh_st.verts.x[v0]
    x1 = mesh_dy.verts.y[v1]
    x2 = mesh_dy.verts.y[v2]
    x3 = mesh_dy.verts.y[v3]

    dtype = di.d_type_PT(x0, x1, x2, x3)
    d = dHat
    g0, g1, g2, g3 = ti.math.vec3(0.0), ti.math.vec3(0.0), ti.math.vec3(0.0), ti.math.vec3(0.0)
    schur = 0.0
    # if dtype == 0:
    #     d = di.d_PP(x0, x1)
    #     if d < dHat:
    #         g0, g1 = di.g_PP(x0, x1)
    #         schur = mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1.dot(g1)
    #         k = 1e8
    #         ld = k * (dHat - d) / (k * schur + 1.0)
    #
    #         p0 = mesh_dy.verts.y[v1] + mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * ld * g1
    #         mesh_dy.verts.gii[v1] += compliance * mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * (mesh_dy.verts.y[v1] - p0)
    #         mesh_dy.verts.hii[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * compliance
    #
    # elif dtype == 1:
    #     d = di.d_PP(x0, x2)
    #     if d < dHat:
    #         g0, g2 = di.g_PP(x0, x2)
    #         schur = mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2.dot(g2)
    #         k = 1e8
    #         ld = k * (dHat - d) / (k * schur + 1.0)
    #         p1 = mesh_dy.verts.y[v2] + mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * ld * g2
    #         mesh_dy.verts.gii[v2] += compliance * mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * (mesh_dy.verts.y[v2] - p1)
    #         mesh_dy.verts.hii[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * compliance
    #
    #
    # elif dtype == 2:
    #     d = di.d_PP(x0, x3)
    #     if d < dHat:
    #         g0, g3 = di.g_PP(x0, x3)
    #         schur = mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3.dot(g3)
    #         k = 1e8
    #         ld = k * (dHat - d) / (k * schur + 1.0)
    #
    #         p2 = mesh_dy.verts.y[v3] + mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * ld * g3
    #         mesh_dy.verts.gii[v3] += compliance * mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * (mesh_dy.verts.y[v3] - p2)
    #         mesh_dy.verts.hii[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * compliance
    #
    # elif dtype == 3:
    #     d = di.d_PE(x0, x1, x2)
    #     if d < dHat:
    #         g0, g1, g2 = di.g_PE(x0, x1, x2)
    #         schur = mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1.dot(g1) + mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2.dot(g2)
    #         k = 1e8
    #         ld = k * (dHat - d) / (k * schur + 1.0)
    #
    #         p0 = mesh_dy.verts.y[v1] + mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * ld * g1
    #         p1 = mesh_dy.verts.y[v2] + mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * ld * g2
    #
    #         mesh_dy.verts.gii[v1] += compliance * mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * (mesh_dy.verts.y[v1] - p0)
    #         mesh_dy.verts.gii[v2] += compliance * mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * (mesh_dy.verts.y[v2] - p1)
    #
    #         mesh_dy.verts.hii[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * compliance
    #         mesh_dy.verts.hii[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * compliance
    #
    #
    # elif dtype == 4:
    #     d = di.d_PE(x0, x2, x3)
    #     if d < dHat:
    #         g0, g2, g3 = di.g_PE(x0, x2, x3)
    #         schur = mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2.dot(g2) + mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3.dot(g3)
    #         k = 1e8
    #         ld = k * (dHat - d) / (k * schur + 1.0)
    #
    #         p1 = mesh_dy.verts.y[v2] + mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * ld * g2
    #         p2 = mesh_dy.verts.y[v3] + mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * ld * g3
    #
    #         mesh_dy.verts.gii[v2] += compliance * mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * (mesh_dy.verts.y[v2] - p1)
    #         mesh_dy.verts.gii[v3] += compliance * mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * (mesh_dy.verts.y[v3] - p2)
    #
    #         mesh_dy.verts.hii[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * compliance
    #         mesh_dy.verts.hii[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * compliance
    #
    # elif dtype == 5:
    #     d = di.d_PE(x0, x1, x3)
    #     if d < dHat:
    #         g0, g1, g3 = di.g_PE(x0, x1, x3)
    #         schur = mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1.dot(g1) + mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3.dot(g3)
    #         k = 1e8
    #         ld = k * (dHat - d) / (k * schur + 1.0)
    #
    #         p0 = mesh_dy.verts.y[v1] + mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * ld * g1
    #         p2 = mesh_dy.verts.y[v3] + mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * ld * g3
    #
    #         mesh_dy.verts.gii[v1] += compliance * mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * (mesh_dy.verts.y[v1] - p0)
    #         mesh_dy.verts.gii[v3] += compliance * mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * (mesh_dy.verts.y[v3] - p2)
    #
    #         mesh_dy.verts.hii[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * compliance
    #         mesh_dy.verts.hii[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * compliance

    if dtype == 6:
        d = di.d_PT(x0, x1, x2, x3)
        if d < dHat:
            g0, g1, g2, g3 = di.g_PT(x0, x1, x2, x3)
            schur = (mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1.dot(g1) +
                     mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2.dot(g2) +
                     mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3.dot(g3))

            k = 1e8
            ld = k * (dHat - d) / (k * schur + 1.0)
            p0 = mesh_dy.verts.y[v1] + mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * ld * g1
            p1 = mesh_dy.verts.y[v2] + mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * ld * g2
            p2 = mesh_dy.verts.y[v3] + mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * ld * g3

            mesh_dy.verts.gii[v1] += compliance * mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * (mesh_dy.verts.y[v1] - p0)
            mesh_dy.verts.gii[v2] += compliance * mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * (mesh_dy.verts.y[v2] - p1)
            mesh_dy.verts.gii[v3] += compliance * mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * (mesh_dy.verts.y[v3] - p2)

            mesh_dy.verts.hii[v1] += compliance * mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1]
            mesh_dy.verts.hii[v2] += compliance * mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2]
            mesh_dy.verts.hii[v3] += compliance * mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3]

    if tv_st_pair_num[vi_s] < tv_st_cache_size:
        tv_st_pair[vi_s, tv_st_pair_num[vi_s], 0] = fi_d
        tv_st_pair[vi_s, tv_st_pair_num[vi_s], 1] = dtype
        tv_st_pair_g[vi_s, tv_st_pair_num[vi_s], 0] = g0
        tv_st_pair_g[vi_s, tv_st_pair_num[vi_s], 1] = g1
        tv_st_pair_g[vi_s, tv_st_pair_num[vi_s], 2] = g2
        tv_st_pair_g[vi_s, tv_st_pair_num[vi_s], 3] = g3
        tv_st_pair_schur[vi_s, tv_st_pair_num[vi_s]] = schur
        tv_st_pair_num[vi_s] += 1

@ti.func
def __vt_dy(vi_d, fi_d, mesh_dy, dHat, vt_dy_cache_size, vt_dy_pair, vt_dy_pair_num, vt_dy_pair_g, vt_dy_pair_schur):

    v0 = vi_d
    v1 = mesh_dy.face_indices[3 * fi_d + 0]
    v2 = mesh_dy.face_indices[3 * fi_d + 1]
    v3 = mesh_dy.face_indices[3 * fi_d + 2]

    x0 = mesh_dy.verts.y[v0]
    x1 = mesh_dy.verts.y[v1]
    x2 = mesh_dy.verts.y[v2]
    x3 = mesh_dy.verts.y[v3]

    dtype = di.d_type_PT(x0, x1, x2, x3)
    d = dHat
    g0, g1, g2, g3 = ti.math.vec3(0.0), ti.math.vec3(0.0), ti.math.vec3(0.0), ti.math.vec3(0.0)
    schur = 0.0
    if dtype == 0:
        d = di.d_PP(x0, x1)
        if d < dHat:
            g0, g1 = di.g_PP(x0, x1)
            schur = (mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0.dot(g0) +
                     mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1.dot(g1) + 1e-4)
            ld = (dHat - d) / schur
            mesh_dy.verts.dx[v0] += mesh_dy.verts.m_inv[v0] * ld * g0
            mesh_dy.verts.dx[v1] += mesh_dy.verts.m_inv[v1] * ld * g1

            mesh_dy.verts.nc[v0] += 1
            mesh_dy.verts.nc[v1] += 1

    elif dtype == 1:
        d = di.d_PP(x0, x2)
        if d < dHat:
            g0, g2 = di.g_PP(x0, x2)
            schur = (mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0.dot(g0) +
                     mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2.dot(g2) + 1e-4)
            ld = (dHat - d) / schur
            mesh_dy.verts.dx[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * ld * g0
            mesh_dy.verts.dx[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * ld * g2
            mesh_dy.verts.nc[v0] += 1
            mesh_dy.verts.nc[v2] += 1


    elif dtype == 2:
        d = di.d_PP(x0, x3)
        if d < dHat:
            g0, g3 = di.g_PP(x0, x3)
            schur = (mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0.dot(g0) +
                     mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3.dot(g3) + 1e-4)
            ld = (dHat - d) / schur

            mesh_dy.verts.dx[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * ld * g0
            mesh_dy.verts.dx[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * ld * g3

            mesh_dy.verts.nc[v0] += 1
            mesh_dy.verts.nc[v3] += 1


    elif dtype == 3:
        d = di.d_PE(x0, x1, x2)
        if d < dHat:
            g0, g1, g2 = di.g_PE(x0, x1, x2)
            schur = (mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0.dot(g0) +
                     mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1.dot(g1) +
                     mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2.dot(g2) + 1e-4)
            ld = (dHat - d) / schur

            mesh_dy.verts.dx[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * ld * g0
            mesh_dy.verts.dx[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * ld * g1
            mesh_dy.verts.dx[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * ld * g2

            mesh_dy.verts.nc[v0] += 1
            mesh_dy.verts.nc[v1] += 1
            mesh_dy.verts.nc[v2] += 1

    elif dtype == 4:
        d = di.d_PE(x0, x2, x3)

        if d < dHat:
            g0, g2, g3 = di.g_PE(x0, x2, x3)
            schur = (mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0.dot(g0) +
                     mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2.dot(g2) +
                     mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3.dot(g3) + 1e-4)
            ld = (dHat - d) / schur

            mesh_dy.verts.dx[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * ld * g0
            mesh_dy.verts.dx[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * ld * g2
            mesh_dy.verts.dx[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * ld * g3
            mesh_dy.verts.nc[v0] += 1
            mesh_dy.verts.nc[v2] += 1
            mesh_dy.verts.nc[v3] += 1

    elif dtype == 5:
        d = di.d_PE(x0, x1, x3)
        if d < dHat:
            g0, g1, g3 = di.g_PE(x0, x1, x3)
            schur = (mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0.dot(g0) +
                     mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1.dot(g1) +
                     mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3.dot(g3) + 1e-4)
            ld = (dHat - d) / schur

            mesh_dy.verts.dx[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * ld * g0
            mesh_dy.verts.dx[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * ld * g1
            mesh_dy.verts.dx[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * ld * g3
            mesh_dy.verts.nc[v0] += 1
            mesh_dy.verts.nc[v1] += 1
            mesh_dy.verts.nc[v3] += 1

    elif dtype == 6:
        d = di.d_PT(x0, x1, x2, x3)

        if d < dHat:
            g0, g1, g2, g3 = di.g_PT(x0, x1, x2, x3)
            schur = (mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0.dot(g0) +
                     mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1.dot(g1) +
                     mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2.dot(g2) +
                     mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3.dot(g3) + 1e-4)
            ld = (dHat - d) / schur
            mesh_dy.verts.dx[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * ld * g0
            mesh_dy.verts.dx[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * ld * g1
            mesh_dy.verts.dx[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * ld * g2
            mesh_dy.verts.dx[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * ld * g3
            mesh_dy.verts.nc[v0] += 1
            mesh_dy.verts.nc[v1] += 1
            mesh_dy.verts.nc[v2] += 1
            mesh_dy.verts.nc[v3] += 1
    #
    if vt_dy_pair_num[vi_d] < vt_dy_cache_size:
        vt_dy_pair[vi_d, vt_dy_pair_num[vi_d], 0] = fi_d
        vt_dy_pair[vi_d, vt_dy_pair_num[vi_d], 1] = dtype
        vt_dy_pair_g[vi_d, vt_dy_pair_num[vi_d], 0] = g0
        vt_dy_pair_g[vi_d, vt_dy_pair_num[vi_d], 1] = g1
        vt_dy_pair_g[vi_d, vt_dy_pair_num[vi_d], 2] = g2
        vt_dy_pair_g[vi_d, vt_dy_pair_num[vi_d], 3] = g3
        vt_dy_pair_schur[vi_d, vt_dy_pair_num[vi_d]] = schur
        vt_dy_pair_num[vi_d] += 1

@ti.func
def __ee_dy(compliance_col, ei0, ei1, mesh_dy, dHat, ee_dy_cache_size, ee_dy_pair, ee_dy_pair_num, ee_dy_pair_g, ee_dy_pair_schur):

    v0 = mesh_dy.edge_indices[2 * ei0 + 0]
    v1 = mesh_dy.edge_indices[2 * ei0 + 1]

    v2 = mesh_dy.edge_indices[2 * ei1 + 0]
    v3 = mesh_dy.edge_indices[2 * ei1 + 1]

    x0 = mesh_dy.verts.y[v0]
    x1 = mesh_dy.verts.y[v1]

    x2 = mesh_dy.verts.y[v2]
    x3 = mesh_dy.verts.y[v3]

    dtype = di.d_type_EE(x0, x1, x2, x3)
    d = dHat
    g0, g1, g2, g3 = ti.math.vec3(0.0), ti.math.vec3(0.0), ti.math.vec3(0.0), ti.math.vec3(0.0)
    schur = 0.0

    # if dtype == 0:
    #     d = di.d_PP(x0, x2)
    #     if d < dHat:
    #         g0, g2 = di.g_PP(x0, x2)
    #         schur = mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0.dot(g0) + mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2.dot(g2)
    #         ld = compliance_col * (dHat - d) / (compliance_col * schur + 1.0)
    #         mesh_dy.verts.dx[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * ld * g0
    #         mesh_dy.verts.dx[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * ld * g2
    #         mesh_dy.verts.nc[v0] += 1
    #         mesh_dy.verts.nc[v2] += 1
    #
    # elif dtype == 1:
    #     d = di.d_PP(x0, x3)
    #     if d < dHat:
    #         g0, g3 = di.g_PP(x0, x3)
    #         schur = mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0.dot(g0) + mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3.dot(g3)
    #         ld = compliance_col * (dHat - d) / (compliance_col * schur + 1.0)
    #         mesh_dy.verts.dx[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * ld * g0
    #         mesh_dy.verts.dx[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * ld * g3
    #         mesh_dy.verts.nc[v0] += 1
    #         mesh_dy.verts.nc[v3] += 1
    #
    # elif dtype == 2:
    #     d = di.d_PE(x0, x2, x3)
    #     if d < dHat:
    #         g0, g2, g3 = di.g_PE(x0, x2, x3)
    #         schur = mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0.dot(g0) + mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2.dot(g2) + mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3.dot(g3)
    #         ld = compliance_col * (dHat - d) / (compliance_col * schur + 1.0)
    #         mesh_dy.verts.dx[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * ld * g0
    #         mesh_dy.verts.dx[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * ld * g2
    #         mesh_dy.verts.dx[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * ld * g3
    #         mesh_dy.verts.nc[v0] += 1
    #         mesh_dy.verts.nc[v2] += 1
    #         mesh_dy.verts.nc[v3] += 1
    #
    # elif dtype == 3:
    #     d = di.d_PP(x1, x2)
    #     if d < dHat:
    #         g1, g2 = di.g_PP(x1, x2)
    #         schur = mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1.dot(g1) + mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2.dot(g2)
    #         ld = compliance_col * (dHat - d) / (compliance_col * schur + 1.0)
    #         mesh_dy.verts.dx[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * ld * g1
    #         mesh_dy.verts.dx[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * ld * g2
    #         mesh_dy.verts.nc[v1] += 1
    #         mesh_dy.verts.nc[v2] += 1
    #
    # elif dtype == 4:
    #     d = di.d_PP(x1, x3)
    #     if d < dHat:
    #         g1, g3 = di.g_PP(x1, x3)
    #         schur = mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1.dot(g1) + mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3.dot(g3)
    #         ld = compliance_col * (dHat - d) / (compliance_col * schur + 1.0)
    #         mesh_dy.verts.dx[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * ld * g1
    #         mesh_dy.verts.dx[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * ld * g3
    #         mesh_dy.verts.nc[v1] += 1
    #         mesh_dy.verts.nc[v3] += 1
    #
    # elif dtype == 5:
    #     d = di.d_PE(x1, x2, x3)
    #     if d < dHat:
    #         g1, g2, g3 = di.g_PE(x1, x2, x3)
    #         schur = mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1.dot(g1) + mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2.dot(g2) + mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3.dot(g3)
    #         ld = compliance_col * (dHat - d) / (compliance_col * schur + 1.0)
    #
    #         mesh_dy.verts.dx[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * ld * g1
    #         mesh_dy.verts.dx[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * ld * g2
    #         mesh_dy.verts.dx[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * ld * g3
    #         mesh_dy.verts.nc[v1] += 1
    #         mesh_dy.verts.nc[v2] += 1
    #         mesh_dy.verts.nc[v3] += 1
    #
    #
    # elif dtype == 6:
    #     d = di.d_PE(x2, x0, x1)
    #     if d < dHat:
    #         g2, g0, g1 = di.g_PE(x2, x0, x1)
    #         schur = mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0.dot(g0) + mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1.dot(g1) + mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2.dot(g2)
    #         ld = compliance_col * (dHat - d) / (compliance_col * schur + 1.0)
    #
    #         mesh_dy.verts.dx[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * ld * g0
    #         mesh_dy.verts.dx[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * ld * g1
    #         mesh_dy.verts.dx[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * ld * g2
    #
    #         mesh_dy.verts.nc[v0] += 1
    #         mesh_dy.verts.nc[v1] += 1
    #         mesh_dy.verts.nc[v2] += 1
    #
    #
    # elif dtype == 7:
    #     d = di.d_PE(x3, x0, x1)
    #     if d < dHat:
    #         g3, g0, g1 = di.g_PE(x3, x0, x1)
    #         schur = mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0.dot(g0) + mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1.dot(g1) + mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3.dot(g3)
    #         ld = compliance_col * (dHat - d) / (compliance_col * schur + 1.0)
    #
    #         mesh_dy.verts.dx[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * ld * g0
    #         mesh_dy.verts.dx[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * ld * g1
    #         mesh_dy.verts.dx[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * ld * g3
    #         mesh_dy.verts.nc[v0] += 1
    #         mesh_dy.verts.nc[v1] += 1
    #         mesh_dy.verts.nc[v3] += 1

    if dtype == 8:
        x01 = x0 - x1
        x23 = x2 - x3

        metric_para_EE = x01.cross(x23).norm()
        if metric_para_EE > 1e-3:
            d = di.d_EE(x0, x1, x2, x3)
            if d < dHat:
                g0, g1, g2, g3 = di.g_EE(x0, x1, x2, x3)
                schur = (mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0.dot(g0) + mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1.dot(g1)
                         + mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2.dot(g2) + mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3.dot(g3))
                ld = compliance_col * (dHat - d) / (compliance_col * schur + 1.0)

                mesh_dy.verts.dx[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * ld * g0
                mesh_dy.verts.dx[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * ld * g1
                mesh_dy.verts.dx[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * ld * g2
                mesh_dy.verts.dx[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * ld * g3

                mesh_dy.verts.nc[v0] += 1
                mesh_dy.verts.nc[v1] += 1
                mesh_dy.verts.nc[v2] += 1
                mesh_dy.verts.nc[v3] += 1

    if d < dHat:
        ee_dy_pair[ei0, ee_dy_pair_num[ei0], 0] = ei1
        ee_dy_pair[ei0, ee_dy_pair_num[ei0], 1] = dtype
        ee_dy_pair_g[ei0, ee_dy_pair_num[ei0], 0] = g0
        ee_dy_pair_g[ei0, ee_dy_pair_num[ei0], 1] = g1
        ee_dy_pair_g[ei0, ee_dy_pair_num[ei0], 2] = g2
        ee_dy_pair_g[ei0, ee_dy_pair_num[ei0], 3] = g3
        ee_dy_pair_schur[ei0, ee_dy_pair_num[ei0]] = schur
        ee_dy_pair_num[ei0] += 1

@ti.func
def __ee_st(compliance, ei_d, ei_s, mesh_dy, mesh_st, dHat, ee_st_cache_size, ee_st_pair, ee_st_pair_num, ee_st_pair_g, ee_st_pair_schur):

    v0 = mesh_dy.edge_indices[2 * ei_d + 0]
    v1 = mesh_dy.edge_indices[2 * ei_d + 1]

    v2 = mesh_st.edge_indices[2 * ei_s + 0]
    v3 = mesh_st.edge_indices[2 * ei_s + 1]

    x0, x1 = mesh_dy.verts.y[v0], mesh_dy.verts.y[v1]
    x2, x3 = mesh_st.verts.x[v2], mesh_st.verts.x[v3]

    dtype = di.d_type_EE(x0, x1, x2, x3)
    # print("ee type: ", dtype)
    g0, g1, g2, g3 = ti.math.vec3(0.0), ti.math.vec3(0.0), ti.math.vec3(0.0), ti.math.vec3(0.0)
    d = dHat
    schur = 0.0
    # if dtype == 0:
    #     d = di.d_PP(x0, x2)
    #     if d < dHat:
    #         g0, g2 = di.g_PP(x0, x2)
    #         schur = mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0.dot(g0)
    #         ld = compliance * (dHat - d) / (compliance * schur + 1.0)
    #         mesh_dy.verts.dx[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * ld * g0
    #         mesh_dy.verts.nc[v0] += 1
    #
    # elif dtype == 1:
    #     d = di.d_PP(x0, x3)
    #     if d < dHat:
    #         g0, g3 = di.g_PP(x0, x3)
    #         schur = mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0.dot(g0)
    #         ld = compliance * (dHat - d) / (compliance * schur + 1.0)
    #         mesh_dy.verts.dx[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * ld * g0
    #         mesh_dy.verts.nc[v0] += 1
    #
    # elif dtype == 2:
    #     d = di.d_PE(x0, x2, x3)
    #     if d < dHat:
    #         g0, g2, g3 = di.g_PE(x0, x2, x3)
    #         schur = mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0.dot(g0)
    #         ld = compliance * (dHat - d) / (compliance * schur + 1.0)
    #         mesh_dy.verts.dx[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * ld * g0
    #         mesh_dy.verts.nc[v0] += 1
    #
    # elif dtype == 3:
    #     d = di.d_PP(x1, x2)
    #     if d < dHat:
    #         g1, g2 = di.g_PP(x1, x2)
    #         schur = mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1.dot(g1)
    #         ld = compliance * (dHat - d) / (compliance * schur + 1.0)
    #
    #         mesh_dy.verts.dx[v1] += mesh_dy.verts.m_inv[v1] * ld * g1
    #         mesh_dy.verts.nc[v1] += 1
    #
    # elif dtype == 4:
    #     d = di.d_PP(x1, x3)
    #     if d < dHat:
    #         g1, g3 = di.g_PP(x1, x3)
    #         schur = mesh_dy.verts.m_inv[v1] * g1.dot(g1)
    #         ld = compliance * (dHat - d) / (compliance * schur + 1.0)
    #         mesh_dy.verts.dx[v1] += mesh_dy.verts.m_inv[v0] * ld * g1
    #         mesh_dy.verts.nc[v1] += 1
    #
    # elif dtype == 5:
    #     d = di.d_PE(x1, x2, x3)
    #     if d < dHat:
    #         g1, g2, g3 = di.g_PE(x1, x2, x3)
    #         schur = mesh_dy.verts.m_inv[v1] * g1.dot(g1)
    #         ld = compliance * (dHat - d) / (compliance * schur + 1.0)
    #         mesh_dy.verts.dx[v1] += mesh_dy.verts.m_inv[v0] * ld * g1
    #         mesh_dy.verts.nc[v1] += 1
    #
    # elif dtype == 6:
    #     d = di.d_PE(x2, x0, x1)
    #     if d < dHat:
    #         g2, g0, g1 = di.g_PE(x2, x0, x1)
    #         schur = mesh_dy.verts.m_inv[v0] * g0.dot(g0) + mesh_dy.verts.m_inv[v1] * g1.dot(g1)
    #         ld = compliance * (dHat - d) / (compliance * schur + 1.0)
    #
    #         mesh_dy.verts.dx[v0] += mesh_dy.verts.m_inv[v0] * ld * g0
    #         mesh_dy.verts.dx[v1] += mesh_dy.verts.m_inv[v1] * ld * g1
    #         mesh_dy.verts.nc[v0] += 1
    #         mesh_dy.verts.nc[v1] += 1
    #
    # elif dtype == 7:
    #     d = di.d_PE(x3, x0, x1)
    #     if d < dHat:
    #         g3, g0, g1 = di.g_PE(x3, x0, x1)
    #         schur = mesh_dy.verts.m_inv[v0] * g0.dot(g0) + mesh_dy.verts.m_inv[v1] * g1.dot(g1)
    #         ld = compliance * (dHat - d) / (compliance * schur + 1.0)
    #
    #         mesh_dy.verts.dx[v0] += mesh_dy.verts.m_inv[v0] * ld * g0
    #         mesh_dy.verts.dx[v1] += mesh_dy.verts.m_inv[v1] * ld * g1
    #         mesh_dy.verts.nc[v0] += 1
    #         mesh_dy.verts.nc[v1] += 1

    if dtype == 8:
        x01 = x0 - x1
        x23 = x2 - x3
        metric_para_EE = x01.cross(x23).norm()
        if metric_para_EE > 1e-3:
            # print("parallel!")
            d = di.d_EE(x0, x1, x2, x3)
            if d < dHat:
                g0, g1, g2, g3 = di.g_EE(x0, x1, x2, x3)
                schur = (mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0.dot(g0) +
                         mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1.dot(g1))
                ld = compliance * (dHat - d) / (compliance * schur + 1.0)
                mesh_dy.verts.dx[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * ld * g0
                mesh_dy.verts.dx[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * ld * g1
                mesh_dy.verts.nc[v0] += 1
                mesh_dy.verts.nc[v1] += 1
    #
    if d < dHat:
        ee_st_pair[ei_d, ee_st_pair_num[ei_d], 0] = ei_s
        ee_st_pair[ei_d, ee_st_pair_num[ei_d], 1] = dtype
        ee_st_pair_g[ei_d, ee_st_pair_num[ei_d], 0] = g0
        ee_st_pair_g[ei_d, ee_st_pair_num[ei_d], 1] = g1
        ee_st_pair_g[ei_d, ee_st_pair_num[ei_d], 2] = g2
        ee_st_pair_g[ei_d, ee_st_pair_num[ei_d], 3] = g3
        ee_st_pair_schur[ei_d, ee_st_pair_num[ei_d]] = schur
        ee_st_pair_num[ei_d] += 1