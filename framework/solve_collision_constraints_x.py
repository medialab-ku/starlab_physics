import taichi as ti
import distance as di


@ti.func
def __vt_st(vi_d, fi_s, mesh_dy, mesh_st, dHat):
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
        schur = mesh_st.verts.m_inv[v0] * g0.dot(g0) + 1e-4
        ld = (dHat - d) / schur
        mesh_dy.verts.dx[v0] += mesh_st.verts.m_inv[v0] * ld * g0
        mesh_dy.verts.nc[v0] += 1
@ti.func
def __tv_st(fi_d, vi_s, mesh_dy, mesh_st, dHat):

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
    if dtype == 0:
        d = di.d_PP(x0, x1)
        if d < dHat:
            g0, g1 = di.g_PP(x0, x1)
            schur = mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1.dot(g1) + 1e-4
            ld = (dHat - d) / schur
            mesh_dy.verts.dx[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * ld * g1
            mesh_dy.verts.nc[v1] += 1

    elif dtype == 1:
        d = di.d_PP(x0, x2)
        if d < dHat:
            g0, g2 = di.g_PP(x0, x2)
            schur = mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2.dot(g2) + 1e-4
            ld = (dHat - d) / schur
            mesh_dy.verts.dx[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * ld * g2
            mesh_dy.verts.nc[v2] += 1


    elif dtype == 2:
        d = di.d_PP(x0, x3)
        if d < dHat:
            g0, g3 = di.g_PP(x0, x3)
            schur = mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3.dot(g3) + 1e-4
            ld = (dHat - d) / schur
            mesh_dy.verts.dx[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * ld * g3
            mesh_dy.verts.nc[v3] += 1

    elif dtype == 3:
        d = di.d_PE(x0, x1, x2)
        if d < dHat:
            g0, g1, g2 = di.g_PE(x0, x1, x2)
            schur = mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1.dot(g1) + mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2.dot(g2) + 1e-4
            ld = (dHat - d) / schur
            mesh_dy.verts.dx[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * ld * g1
            mesh_dy.verts.dx[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * ld * g2
            mesh_dy.verts.nc[v1] += 1
            mesh_dy.verts.nc[v2] += 1

    elif dtype == 4:
        d = di.d_PE(x0, x2, x3)
        if d < dHat:
            g0, g2, g3 = di.g_PE(x0, x2, x3)
            schur = mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2.dot(g2) + mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3.dot(g3) + 1e-4
            ld = (dHat - d) / schur
            mesh_dy.verts.dx[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * ld * g2
            mesh_dy.verts.dx[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * ld * g3

            mesh_dy.verts.nc[v2] += 1
            mesh_dy.verts.nc[v3] += 1

    elif dtype == 5:
        d = di.d_PE(x0, x1, x3)
        if d < dHat:
            g0, g1, g3 = di.g_PE(x0, x1, x3)
            schur = mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1.dot(g1) + mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3.dot(g3) + 1e-4
            ld = (dHat - d) / schur
            mesh_dy.verts.dx[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * ld * g1
            mesh_dy.verts.dx[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * ld * g3
            mesh_dy.verts.nc[v1] += 1
            mesh_dy.verts.nc[v3] += 1

    elif dtype == 6:
        d = di.d_PT(x0, x1, x2, x3)
        if d < dHat:
            g0, g1, g2, g3 = di.g_PT(x0, x1, x2, x3)
            schur = (mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1.dot(g1) +
                     mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2.dot(g2) +
                     mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3.dot(g3) + 1e-4)
            ld = (dHat - d) / schur
            mesh_dy.verts.dx[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * ld * g1
            mesh_dy.verts.dx[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * ld * g2
            mesh_dy.verts.dx[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * ld * g3
            mesh_dy.verts.nc[v1] += 1
            mesh_dy.verts.nc[v2] += 1
            mesh_dy.verts.nc[v3] += 1

        # if d < dHat and tv_static_pair_num[fid_d] < tv_static_pair_cache_size:
        #     tv_static_pair[fid_d, tv_static_pair_num[fid_d], 0] = vid_s
        #     tv_static_pair[fid_d, tv_static_pair_num[fid_d], 1] = dtype
        #     tv_static_pair_g[fid_d, tv_static_pair_num[fid_d], 0] = g0
        #     tv_static_pair_g[fid_d, tv_static_pair_num[fid_d], 1] = g1
        #     tv_static_pair_g[fid_d, tv_static_pair_num[fid_d], 2] = g2
        #     tv_static_pair_g[fid_d, tv_static_pair_num[fid_d], 3] = g3
        #     tv_static_pair_schur[fid_d, tv_static_pair_num[fid_d]] = schur
        #     tv_static_pair_num[fid_d] += 1

@ti.func
def __tv_dy(fi_d, vi_d, mesh_dy, dHat):

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
            mesh_dy.verts.dx[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * ld * g0
            mesh_dy.verts.dx[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * ld * g1

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
            mesh_dy.verts.dx[v2] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v2] * ld * g2
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
    # if d < dHat and tv_dynamic_pair_num[fid] < tv_dynamic_pair_cache_size:
    #     tv_dynamic_pair[fid, tv_dynamic_pair_num[fid], 0] = vid
    #     tv_dynamic_pair[fid, tv_dynamic_pair_num[fid], 1] = dtype
    #     tv_dynamic_pair_g[fid, tv_dynamic_pair_num[fid], 0] = g0
    #     tv_dynamic_pair_g[fid, tv_dynamic_pair_num[fid], 1] = g1
    #     tv_dynamic_pair_g[fid, tv_dynamic_pair_num[fid], 2] = g2
    #     tv_dynamic_pair_g[fid, tv_dynamic_pair_num[fid], 3] = g3
    #     tv_dynamic_pair_schur[fid, tv_dynamic_pair_num[fid]] = schur
    #     tv_dynamic_pair_num[fid] += 1

@ti.func
def __ee_dy(ei0, ei1, mesh_dy, dHat):

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

    if dtype == 0:
        d = di.d_PP(x0, x2)
        if d < dHat:
            g0, g2 = di.g_PP(x0, x2)
            schur = mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0.dot(g0) + mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2.dot(
                g2) + 1e-4
            ld = (dHat - d) / schur
            mesh_dy.verts.dx[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * ld * g0
            mesh_dy.verts.dx[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * ld * g2
            mesh_dy.verts.nc[v0] += 1
            mesh_dy.verts.nc[v2] += 1

    elif dtype == 1:
        d = di.d_PP(x0, x3)
        if d < dHat:
            g0, g3 = di.g_PP(x0, x3)
            schur = mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0.dot(g0) + mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3.dot(
                g3) + 1e-4
            ld = (dHat - d) / schur
            mesh_dy.verts.dx[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * ld * g0
            mesh_dy.verts.dx[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * ld * g3
            mesh_dy.verts.nc[v0] += 1
            mesh_dy.verts.nc[v3] += 1

    elif dtype == 2:
        d = di.d_PE(x0, x2, x3)
        if d < dHat:
            g0, g2, g3 = di.g_PE(x0, x2, x3)
            schur = mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0.dot(g0) + mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2.dot(g2) + \
                    mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3.dot(g3) + 1e-4
            ld = (dHat - d) / schur
            mesh_dy.verts.dx[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * ld * g0
            mesh_dy.verts.dx[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * ld * g2
            mesh_dy.verts.dx[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * ld * g3
            mesh_dy.verts.nc[v0] += 1
            mesh_dy.verts.nc[v2] += 1
            mesh_dy.verts.nc[v3] += 1



    elif dtype == 3:
        d = di.d_PP(x1, x2)
        if d < dHat:
            g1, g2 = di.g_PP(x1, x2)
            schur = mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1.dot(g1) + mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2.dot(
                g2) + 1e-4
            ld = (dHat - d) / schur
            mesh_dy.verts.dx[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * ld * g1
            mesh_dy.verts.dx[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * ld * g2
            mesh_dy.verts.nc[v1] += 1
            mesh_dy.verts.nc[v2] += 1

    elif dtype == 4:
        d = di.d_PP(x1, x3)
        if d < dHat:
            g1, g3 = di.g_PP(x1, x3)
            schur = mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1.dot(g1) + mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3.dot(
                g3) + 1e-4
            ld = (dHat - d) / schur
            mesh_dy.verts.dx[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * ld * g1
            mesh_dy.verts.dx[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * ld * g3
            mesh_dy.verts.nc[v1] += 1
            mesh_dy.verts.nc[v3] += 1


    elif dtype == 5:
        d = di.d_PE(x1, x2, x3)
        if d < dHat:
            g1, g2, g3 = di.g_PE(x1, x2, x3)
            schur = mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1.dot(g1) + mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2.dot(g2) + \
                    mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3.dot(g3) + 1e-4
            ld = (dHat - d) / schur

            mesh_dy.verts.dx[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * ld * g1
            mesh_dy.verts.dx[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * ld * g2
            mesh_dy.verts.dx[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * ld * g3
            mesh_dy.verts.nc[v1] += 1
            mesh_dy.verts.nc[v2] += 1
            mesh_dy.verts.nc[v3] += 1


    elif dtype == 6:
        d = di.d_PE(x2, x0, x1)
        if d < dHat:
            g2, g0, g1 = di.g_PE(x2, x0, x1)
            schur = mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0.dot(g0) + mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1.dot(g1) + \
                    mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2.dot(g2) + 1e-4
            ld = (dHat - d) / schur

            mesh_dy.verts.dx[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * ld * g0
            mesh_dy.verts.dx[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * ld * g1
            mesh_dy.verts.dx[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * ld * g2

            mesh_dy.verts.nc[v0] += 1
            mesh_dy.verts.nc[v1] += 1
            mesh_dy.verts.nc[v2] += 1


    elif dtype == 7:
        d = di.d_PE(x3, x0, x1)
        if d < dHat:
            g3, g0, g1 = di.g_PE(x3, x0, x1)
            schur = mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0.dot(g0) + mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1.dot(g1) + \
                    mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3.dot(g3) + 1e-4
            ld = (dHat - d) / schur

            mesh_dy.verts.dx[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * ld * g0
            mesh_dy.verts.dx[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * ld * g1
            mesh_dy.verts.dx[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * ld * g3
            mesh_dy.verts.nc[v0] += 1
            mesh_dy.verts.nc[v1] += 1
            mesh_dy.verts.nc[v3] += 1

    elif dtype == 8:
        x01 = x0 - x1
        x23 = x2 - x3

        metric_para_EE = x01.cross(x23).norm()
        if metric_para_EE > 1e-6:
            d = di.d_EE(x0, x1, x2, x3)
            if d < dHat:
                g0, g1, g2, g3 = di.g_EE(x0, x1, x2, x3)
                schur = mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0.dot(g0) + mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1.dot(
                    g1) + mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2.dot(g2) + mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3.dot(
                    g3) + 1e-4
                ld = (dHat - d) / schur

                mesh_dy.verts.dx[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * ld * g0
                mesh_dy.verts.dx[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * ld * g1
                mesh_dy.verts.dx[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * ld * g2
                mesh_dy.verts.dx[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * ld * g3

                mesh_dy.verts.nc[v0] += 1
                mesh_dy.verts.nc[v1] += 1
                mesh_dy.verts.nc[v2] += 1
                mesh_dy.verts.nc[v3] += 1

    # if d < dHat and ee_dynamic_pair_num[ei0] < ee_dynamic_pair_cache_size:
    #     ee_dynamic_pair[ei0, ee_dynamic_pair_num[ei0], 0] = ei1
    #     ee_dynamic_pair[ei0, ee_dynamic_pair_num[ei0], 1] = dtype
    #     ee_dynamic_pair_g[ei0, ee_dynamic_pair_num[ei0], 0] = g0
    #     ee_dynamic_pair_g[ei0, ee_dynamic_pair_num[ei0], 1] = g1
    #     ee_dynamic_pair_g[ei0, ee_dynamic_pair_num[ei0], 2] = g2
    #     ee_dynamic_pair_g[ei0, ee_dynamic_pair_num[ei0], 3] = g3
    #     ee_dynamic_pair_schur[ei0, ee_dynamic_pair_num[ei0]] = schur
    #     ee_dynamic_pair_num[ei0] += 1

@ti.func
def __ee_st(ei_d, ei_s, mesh_dy, mesh_st, dHat):

    v0 = mesh_dy.edge_indices[2 * ei_d + 0]
    v1 = mesh_dy.edge_indices[2 * ei_d + 1]

    v2 = mesh_st.edge_indices[2 * ei_s + 0]
    v3 = mesh_st.edge_indices[2 * ei_s + 1]

    x0, x1 = mesh_dy.verts.y[v0], mesh_dy.verts.y[v1]
    x2, x3 = mesh_st.verts.x[v2], mesh_st.verts.x[v3]

    dtype = di.d_type_EE(x0, x1, x2, x3)
    g0, g1, g2, g3 = ti.math.vec3(0.0), ti.math.vec3(0.0), ti.math.vec3(0.0), ti.math.vec3(0.0)
    d = dHat
    schur = 0.0
    if dtype == 0:
        d = di.d_PP(x0, x2)
        if d < dHat:
            g0, g2 = di.g_PP(x0, x2)
            schur = mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0.dot(g0) + 1e-4
            ld = (dHat - d) / schur
            mesh_dy.verts.dx[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * ld * g0
            mesh_dy.verts.nc[v0] += 1

    elif dtype == 1:
        d = di.d_PP(x0, x3)
        if d < dHat:
            g0, g3 = di.g_PP(x0, x3)
            schur = mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0.dot(g0) + 1e-4
            ld = (dHat - d) / schur
            mesh_dy.verts.dx[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * ld * g0
            mesh_dy.verts.nc[v0] += 1

    elif dtype == 2:
        d = di.d_PE(x0, x2, x3)
        if d < dHat:
            g0, g2, g3 = di.g_PE(x0, x2, x3)
            schur = mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0.dot(g0) + 1e-4
            ld = (dHat - d) / schur
            mesh_dy.verts.dx[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * ld * g0
            mesh_dy.verts.nc[v0] += 1

    elif dtype == 3:
        d = di.d_PP(x1, x2)
        if d < dHat:
            g1, g2 = di.g_PP(x1, x2)
            schur = mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1.dot(g1) + 1e-4
            ld = (dHat - d) / schur

            mesh_dy.verts.dx[v1] += mesh_dy.verts.m_inv[v1] * ld * g1
            mesh_dy.verts.nc[v1] += 1

    elif dtype == 4:
        d = di.d_PP(x1, x3)
        if d < dHat:
            g1, g3 = di.g_PP(x1, x3)
            schur = mesh_dy.verts.m_inv[v1] * g1.dot(g1) + 1e-4
            ld = (dHat - d) / schur

            mesh_dy.verts.dx[v1] += mesh_dy.verts.m_inv[v0] * ld * g1
            mesh_dy.verts.nc[v1] += 1

    elif dtype == 5:
        d = di.d_PE(x1, x2, x3)
        if d < dHat:
            g1, g2, g3 = di.g_PE(x1, x2, x3)
            schur = mesh_dy.verts.m_inv[v1] * g1.dot(g1) + 1e-4
            ld = (dHat - d) / schur
            mesh_dy.verts.dx[v1] += mesh_dy.verts.m_inv[v0] * ld * g1
            mesh_dy.verts.nc[v1] += 1

    elif dtype == 6:
        d = di.d_PE(x2, x0, x1)
        if d < dHat:
            g2, g0, g1 = di.g_PE(x2, x0, x1)
            schur = mesh_dy.verts.m_inv[v0] * g0.dot(g0) + mesh_dy.verts.m_inv[v1] * g1.dot(g1) + 1e-4
            ld = (dHat - d) / schur

            mesh_dy.verts.dx[v0] += mesh_dy.verts.m_inv[v0] * ld * g0
            mesh_dy.verts.dx[v1] += mesh_dy.verts.m_inv[v1] * ld * g1
            mesh_dy.verts.nc[v0] += 1
            mesh_dy.verts.nc[v1] += 1

    elif dtype == 7:
        d = di.d_PE(x3, x0, x1)
        if d < dHat:
            g3, g0, g1 = di.g_PE(x3, x0, x1)
            schur = mesh_dy.verts.m_inv[v0] * g0.dot(g0) + mesh_dy.verts.m_inv[v1] * g1.dot(g1) + 1e-4
            ld = (dHat - d) / schur

            mesh_dy.verts.dx[v0] += mesh_dy.verts.m_inv[v0] * ld * g0
            mesh_dy.verts.dx[v1] += mesh_dy.verts.m_inv[v1] * ld * g1
            mesh_dy.verts.nc[v0] += 1
            mesh_dy.verts.nc[v1] += 1


    elif dtype == 8:
        x01 = x0 - x1
        x23 = x2 - x3
        metric_para_EE = x01.cross(x23).norm()
        if metric_para_EE > 1e-3:
            d = di.d_EE(x0, x1, x2, x3)
            if d < dHat:
                g0, g1, g2, g3 = di.g_EE(x0, x1, x2, x3)
                schur = mesh_dy.verts.m_inv[v0] * g0.dot(g0) + mesh_dy.verts.m_inv[v1] * g1.dot(g1) + 1e-4
                ld = (dHat - d) / schur
                mesh_dy.verts.dx[v0] += mesh_dy.verts.m_inv[v0] * ld * g0
                mesh_dy.verts.dx[v1] += mesh_dy.verts.m_inv[v1] * ld * g1
                mesh_dy.verts.nc[v0] += 1
                mesh_dy.verts.nc[v1] += 1
    #
    # if d < dHat and ee_static_pair_num[eid_d] < ee_static_pair_cache_size:
    #     ee_static_pair[eid_d, ee_static_pair_num[eid_d], 0] = eid_s
    #     ee_static_pair[eid_d, ee_static_pair_num[eid_d], 1] = dtype
    #     ee_static_pair_g[eid_d, ee_static_pair_num[eid_d], 0] = g0
    #     ee_static_pair_g[eid_d, ee_static_pair_num[eid_d], 1] = g1
    #     ee_static_pair_g[eid_d, ee_static_pair_num[eid_d], 2] = g2
    #     ee_static_pair_g[eid_d, ee_static_pair_num[eid_d], 3] = g3
    #     ee_static_pair_schur[eid_d, ee_static_pair_num[eid_d]] = schur
    #     ee_static_pair_num[eid_d] += 1
