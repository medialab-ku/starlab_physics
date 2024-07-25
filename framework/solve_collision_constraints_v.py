import taichi as ti
import distance as di


@ti.func
def __vt_st(vid_d, fid_s, dtype, mesh_dy, mesh_st, g0, g1, g2, g3, schur, mu):

    v0 = vid_d
    v1 = mesh_st.face_indices[3 * fid_s + 0]
    v2 = mesh_st.face_indices[3 * fid_s + 1]
    v3 = mesh_st.face_indices[3 * fid_s + 2]

    Cv = g0.dot(mesh_dy.verts.v[v0]) + g1.dot(mesh_st.verts.v[v1]) + g2.dot(mesh_st.verts.v[v2]) + g3.dot(mesh_st.verts.v[v3])
    if dtype == 0:
        if Cv < 0.:
            ld_v = -Cv / schur
            mesh_dy.verts.dv[v0] += mesh_dy.verts.m_inv[v0] * ld_v * g0
            mesh_dy.verts.nc[v0] += 1
            vTan0 = mesh_dy.verts.v[v0] + mesh_dy.verts.m_inv[v0] * ld_v * g0
            g0Tan = mesh_st.verts.v[v1] - vTan0
            cTan = 0.5 * g0Tan.dot(g0Tan)
            schur = mesh_dy.verts.m_inv[v0] * g0Tan.dot(g0Tan) + 1e-4
            ldTan = cTan / schur
            dvTan = mesh_dy.verts.m_inv[v0] * ldTan * g0Tan

            if mu * abs(Cv) > cTan:
                mu = 1.0
            mesh_dy.verts.v[v0] += mu * dvTan

    elif dtype == 1:
        if Cv < 0.:
            ld_v = -Cv / schur
            mesh_dy.verts.dv[v0] += mesh_dy.verts.m_inv[v0] * ld_v * g0
            mesh_dy.verts.nc[v0] += 1
            vTan0 = mesh_dy.verts.v[v0] + mesh_dy.verts.m_inv[v0] * ld_v * g0
            g0Tan = mesh_st.verts.v[v2] - vTan0
            cTan = 0.5 * g0Tan.dot(g0Tan)
            schur = mesh_dy.verts.m_inv[v0] * g0Tan.dot(g0Tan) + 1e-4
            ldTan = cTan / schur
            dvTan = mesh_dy.verts.m_inv[v0] * ldTan * g0Tan
            if mu * abs(Cv) > cTan:
                mu = 1.0
            mesh_dy.verts.v[v0] += mu * dvTan

    elif dtype == 2:
        if Cv < 0.:
            ld_v = -Cv / schur
            mesh_dy.verts.dv[v0] += mesh_dy.verts.m_inv[v0] * ld_v * g0
            mesh_dy.verts.nc[v0] += 1
            vTan0 = mesh_dy.verts.v[v0] + mesh_dy.verts.m_inv[v0] * ld_v * g0
            g0Tan = mesh_st.verts.v[v3] - vTan0
            cTan = 0.5 * g0Tan.dot(g0Tan)
            schur = mesh_dy.verts.m_inv[v0] * g0Tan.dot(g0Tan) + 1e-4
            ldTan = cTan / schur
            dvTan = mesh_dy.verts.m_inv[v0] * ldTan * g0Tan
            if mu * abs(Cv) > cTan:
                mu = 1.0
            mesh_dy.verts.v[v0] += mu * dvTan


    elif dtype == 3:
        if Cv < 0.:
            ld_v = -Cv / schur
            mesh_dy.verts.dv[v0] += mesh_st.verts.m_inv[v0] * ld_v * g0
            mesh_dy.verts.nc[v0] += 1
            vTan0 = mesh_dy.verts.v[v0] + mesh_dy.verts.m_inv[v0] * ld_v * g0
            a, b = g1.norm(), g2.norm()
            p = (a * mesh_st.verts.v[v1] + b * mesh_st.verts.v[v2]) / (a + b)
            g0Tan = p - vTan0
            cTan = 0.5 * g0Tan.dot(g0Tan)
            schur = mesh_dy.verts.m_inv[v0] * g0Tan.dot(g0Tan) + 1e-4
            ldTan = cTan / schur
            dvTan = mesh_dy.verts.m_inv[v0] * ldTan * g0Tan
            if mu * abs(Cv) > cTan:
                mu = 1.0
            mesh_dy.verts.v[v0] += mu * dvTan

    elif dtype == 4:
        if Cv < 0.:
            ld_v = -Cv / schur
            mesh_dy.verts.dv[v0] += mesh_dy.verts.m_inv[v0] * ld_v * g0
            mesh_dy.verts.nc[v0] += 1
            vTan0 = mesh_dy.verts.v[v0] + mesh_dy.verts.m_inv[v0] * ld_v * g0
            a, b = g2.norm(), g3.norm()
            p = (a * mesh_st.verts.v[v2] + b * mesh_st.verts.v[v3]) / (a + b)
            g0Tan = p - vTan0
            cTan = 0.5 * g0Tan.dot(g0Tan)
            schur = mesh_dy.verts.m_inv[v0] * g0Tan.dot(g0Tan) + 1e-4
            ldTan = cTan / schur
            dvTan = mesh_dy.verts.m_inv[v0] * ldTan * g0Tan
            if mu * abs(Cv) > cTan:
                mu = 1.0
            mesh_dy.verts.v[v0] += mu * dvTan

    elif dtype == 5:
        if Cv < 0.:
            ld_v = -Cv / schur
            mesh_dy.verts.dv[v0] += mesh_dy.verts.m_inv[v0] * ld_v * g0
            mesh_dy.verts.nc[v0] += 1
            vTan0 = mesh_dy.verts.v[v0] + mesh_dy.verts.m_inv[v0] * ld_v * g0
            a, b = g1.norm(), g3.norm()
            p = (a * mesh_st.verts.v[v1] + b * mesh_st.verts.v[v3]) / (a + b)
            g0Tan = vTan0 - p
            cTan = 0.5 * g0Tan.dot(g0Tan)
            schur = mesh_dy.verts.m_inv[v0] * g0Tan.dot(g0Tan) + 1e-4
            ldTan = cTan / schur
            dvTan = mesh_dy.verts.m_inv[v0] * ldTan * g0Tan
            if mu * abs(Cv) > cTan:
                mu = 1.0
            mesh_dy.verts.v[v0] += mu * dvTan

    elif dtype == 6:
        if Cv < 0.:
            ld_v = -Cv / schur
            mesh_dy.verts.dv[v0] += mesh_dy.verts.m_inv[v0] * ld_v * g0
            mesh_dy.verts.nc[v0] += 1
            vTan0 = mesh_dy.verts.v[v0] + mesh_dy.verts.m_inv[v0] * ld_v * g0
            a, b, c = g1.norm(), g2.norm(), g3.norm()
            p = (a * mesh_st.verts.v[v1] + b * mesh_st.verts.v[v2] + c * mesh_st.verts.v[v3]) / (a + b + c)
            g0Tan = p - vTan0
            cTan = 0.5 * g0Tan.dot(g0Tan)
            schur = mesh_dy.verts.m_inv[v0] * g0Tan.dot(g0Tan) + 1e-4
            ldTan = cTan / schur
            dvTan = mesh_dy.verts.m_inv[v0] * ldTan * g0Tan
            if mu * abs(Cv) > cTan:
                mu = 1.0
            mesh_dy.verts.v[v0] += mu * dvTan

@ti.func
def __tv_st(fid_d, vid_s, dtype, mesh_dy, mesh_st, g0, g1, g2, g3, schur, mu):

    v0 = vid_s
    v1 = mesh_dy.face_indices[3 * fid_d + 0]
    v2 = mesh_dy.face_indices[3 * fid_d + 1]
    v3 = mesh_dy.face_indices[3 * fid_d + 2]

    Cv = g0.dot(mesh_st.verts.v[v0]) + g1.dot(mesh_dy.verts.v[v1]) + g3.dot(mesh_dy.verts.v[v2]) + g3.dot(mesh_dy.verts.v[v3])
    if dtype == 0:
        if Cv < 0.0:
            ld_v = - Cv / schur
            mesh_dy.verts.dx[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * ld_v * g1
            mesh_dy.verts.nc[v1] += 1
            vTan1 = mesh_dy.verts.v[v1] + mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1 * ld_v
            g1Tan = mesh_st.verts.v[v0] - vTan1
            cTan = 0.5 * g1Tan.dot(g1Tan)
            schur = mesh_dy.verts.m_inv[v1] * g1Tan.dot(g1Tan) + 1e-4
            ldTan = cTan / schur
            dvTan1 = mesh_dy.verts.m_inv[v1] * ldTan * g1Tan

            if mu * abs(Cv) > cTan:
                mu = 1.0
            mesh_dy.verts.dv[v1] += mu * dvTan1

    elif dtype == 1:
        if Cv < 0.0:
            ld_v = - Cv / schur
            mesh_dy.verts.dx[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * ld_v * g2
            mesh_dy.verts.nc[v2] += 1

            vTan2 = mesh_dy.verts.v[v2] + mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2 * ld_v
            g2Tan = mesh_st.verts.v[v0] - vTan2
            cTan = 0.5 * g2Tan.dot(g2Tan)
            schur = mesh_dy.verts.m_inv[v2] * g2Tan.dot(g2Tan) + 1e-4
            ldTan = cTan / schur
            dvTan2 = mesh_dy.verts.m_inv[v2] * ldTan * g2Tan

            if mu * abs(Cv) > cTan:
                mu = 1.0
            mesh_dy.verts.dv[v2] += mu * dvTan2

    elif dtype == 2:
        if Cv < 0.0:
            ld_v = - Cv / schur
            mesh_dy.verts.dx[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * ld_v * g3
            mesh_dy.verts.nc[v3] += 1

            vTan3 = mesh_dy.verts.v[v3] + mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3 * ld_v
            g3Tan = mesh_st.verts.v[v0] - vTan3
            cTan = 0.5 * g3Tan.dot(g3Tan)
            schur = mesh_dy.verts.m_inv[v3] * g3Tan.dot(g3Tan) + 1e-4
            ldTan = cTan / schur
            dvTan3 = mesh_dy.verts.m_inv[v3] * ldTan * g3Tan

            if mu * abs(Cv) > cTan:
                mu = 1.0
            mesh_dy.verts.dv[v3] += mu * dvTan3

    elif dtype == 3:

        if Cv < 0.0:
            ld_v = - Cv / schur
            mesh_dy.verts.dx[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * ld_v * g1
            mesh_dy.verts.dx[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * ld_v * g2
            mesh_dy.verts.nc[v1] += 1
            mesh_dy.verts.nc[v2] += 1

            vTan1 = mesh_dy.verts.v[v1] + mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1 * ld_v
            vTan2 = mesh_dy.verts.v[v2] + mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2 * ld_v

            a, b = g1.norm(), g2.norm()
            ab = (a + b)
            p = (a * vTan1 + b * vTan2) / (a + b)
            g1Tan = (a / ab) * (mesh_st.verts.v[v0] - p)
            g2Tan = (b / ab) * (mesh_st.verts.v[v0] - p)
            cTan = 0.5 * (mesh_st.verts.v[v0] - p).dot(mesh_st.verts.v[v0] - p)
            schur = mesh_dy.verts.m_inv[v1] * g1Tan.dot(g1Tan) + mesh_dy.verts.m_inv[v2] * g2Tan.dot(g2Tan) + 1e-4
            ldTan = cTan / schur
            dvTan1 = mesh_dy.verts.m_inv[v1] * ldTan * g1Tan
            dvTan2 = mesh_dy.verts.m_inv[v2] * ldTan * g2Tan

            if mu * abs(Cv) > cTan:
                mu = 1.0

            mesh_dy.verts.dv[v1] += mu * dvTan1
            mesh_dy.verts.dv[v2] += mu * dvTan2

    elif dtype == 4:
        if Cv < 0.0:
            ld_v = -Cv / schur
            mesh_dy.verts.dx[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * ld_v * g2
            mesh_dy.verts.dx[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * ld_v * g3

            mesh_dy.verts.nc[v2] += 1
            mesh_dy.verts.nc[v3] += 1

            vTan2 = mesh_dy.verts.v[v2] + mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2 * ld_v
            vTan3 = mesh_dy.verts.v[v3] + mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3 * ld_v

            a, b = g2.norm(), g3.norm()
            ab = (a + b)
            p = (a * vTan2 + b * vTan3) /ab
            g2Tan = (a/ab) * (mesh_st.verts.v[v0] - p)
            g3Tan = (b/ab) * (mesh_st.verts.v[v0] - p)
            cTan = 0.5 * (mesh_st.verts.v[v0] - p).dot(mesh_st.verts.v[v0] - p)
            schur = mesh_dy.verts.m_inv[v2] * g2Tan.dot(g2Tan) + mesh_dy.verts.m_inv[v3] * g3Tan.dot(g3Tan) + 1e-4
            ldTan = cTan / schur
            dvTan2 = mesh_dy.verts.m_inv[v2] * ldTan * g2Tan
            dvTan3 = mesh_dy.verts.m_inv[v3] * ldTan * g3Tan


            if mu * abs(Cv) > cTan:
                mu = 1.0

            mesh_dy.verts.dv[v2] += mu * dvTan2
            mesh_dy.verts.dv[v3] += mu * dvTan3

    elif dtype == 5:
        if Cv < 0.0:
            ld_v = -Cv / schur
            mesh_dy.verts.dx[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * ld_v * g1
            mesh_dy.verts.dx[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * ld_v * g3
            mesh_dy.verts.nc[v1] += 1
            mesh_dy.verts.nc[v3] += 1

            vTan1 = mesh_dy.verts.v[v1] + mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1 * ld_v
            vTan3 = mesh_dy.verts.v[v3] + mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3 * ld_v
            a, b = g1.norm(), g3.norm()
            ab = (a + b)
            p = (a * vTan1 + b * vTan3) / ab
            g1Tan = (a/ab) * (mesh_st.verts.v[v0] - p)
            g3Tan = (b/ab) * (mesh_st.verts.v[v0] - p)
            cTan = 0.5 * (mesh_st.verts.v[v0] - p).dot(mesh_st.verts.v[v0] - p)
            schur = mesh_dy.verts.m_inv[v1] * g1Tan.dot(g1Tan) + mesh_dy.verts.m_inv[v3] * g3Tan.dot(g3Tan) + 1e-4
            ldTan = cTan / schur
            dvTan1 = mesh_dy.verts.m_inv[v1] * ldTan * g1Tan
            dvTan3 = mesh_dy.verts.m_inv[v3] * ldTan * g3Tan

            if mu * abs(Cv) > cTan:
                mu = 1.0

            mesh_dy.verts.dv[v1] += mu * dvTan1
            mesh_dy.verts.dv[v3] += mu * dvTan3

    elif dtype == 6:
        if Cv < 0.0:
            ld_v = -Cv / schur

            mesh_dy.verts.dx[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * ld_v * g1
            mesh_dy.verts.dx[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * ld_v * g2
            mesh_dy.verts.dx[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * ld_v * g3
            mesh_dy.verts.nc[v1] += 1
            mesh_dy.verts.nc[v2] += 1
            mesh_dy.verts.nc[v3] += 1

            vTan1 = mesh_dy.verts.v[v1] + mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1 * ld_v
            vTan2 = mesh_dy.verts.v[v2] + mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2 * ld_v
            vTan3 = mesh_dy.verts.v[v3] + mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3 * ld_v

            a, b, c = g1.norm(), g2.norm(), g3.norm()
            abc = (a + b + c)
            p = (a * vTan1 + b * vTan2 + c * vTan3) / abc
            g1Tan = (a / abc) * (mesh_st.verts.v[v0] - p)
            g2Tan = (b / abc) * (mesh_st.verts.v[v0] - p)
            g3Tan = (c / abc) * (mesh_st.verts.v[v0] - p)
            cTan = 0.5 * (mesh_st.verts.v[v0] - p).dot(mesh_st.verts.v[v0] - p)
            schur = mesh_dy.verts.m_inv[v1] * g1Tan.dot(g1Tan) + mesh_dy.verts.m_inv[v2] * g2Tan.dot(g2Tan) + mesh_dy.verts.m_inv[v3] * g3Tan.dot(g3Tan) + 1e-4
            ldTan = cTan / schur
            dvTan1 = mesh_dy.verts.m_inv[v1] * ldTan * g1Tan
            dvTan2 = mesh_dy.verts.m_inv[v2] * ldTan * g2Tan
            dvTan3 = mesh_dy.verts.m_inv[v3] * ldTan * g3Tan

            if mu * abs(Cv) > cTan:
                mu = 1.0

            mesh_dy.verts.dv[v1] += mu * dvTan1
            mesh_dy.verts.dv[v2] += mu * dvTan2
            mesh_dy.verts.dv[v3] += mu * dvTan3


@ti.func
def __vt_dy(vid_d, fid_d, dtype, mesh_dy, g0, g1, g2, g3, schur, mu):

    v0 = vid_d
    v1 = mesh_dy.face_indices[3 * fid_d + 0]
    v2 = mesh_dy.face_indices[3 * fid_d + 1]
    v3 = mesh_dy.face_indices[3 * fid_d + 2]

    Cv = g0.dot(mesh_dy.verts.v[v0]) + g1.dot(mesh_dy.verts.v[v1]) + g2.dot(mesh_dy.verts.v[v2]) + g3.dot(mesh_dy.verts.v[v3])

    if Cv < 0.0:
        ld_v = -Cv / schur
        if dtype == 0:

            mesh_dy.verts.dv[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0 * ld_v
            mesh_dy.verts.dv[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1 * ld_v

            mesh_dy.verts.nc[v0] += 1
            mesh_dy.verts.nc[v1] += 1

            vTan0 = mesh_dy.verts.v[v0] + mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0 * ld_v
            vTan1 = mesh_dy.verts.v[v1] + mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1 * ld_v
            g0Tan = -(vTan0 - vTan1)
            g1Tan = vTan0 - vTan1
            cTan = 0.5 * (g1Tan.dot(g1Tan) + g0Tan.dot(g0Tan))
            schur = mesh_dy.verts.m_inv[v0] * g0Tan.dot(g0Tan) + mesh_dy.verts.m_inv[v1] * g1Tan.dot(g1Tan) + 1e-4
            ldTan = cTan / schur
            dvTan0 = mesh_dy.verts.m_inv[v0] * ldTan * g0Tan
            dvTan1 = mesh_dy.verts.m_inv[v1] * ldTan * g1Tan

            if mu * abs(Cv) > cTan:
                mu = 1.0
            mesh_dy.verts.dv[v0] += mu * dvTan0
            mesh_dy.verts.dv[v1] += mu * dvTan1

        elif dtype == 1:

            mesh_dy.verts.dv[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0 * ld_v
            mesh_dy.verts.dv[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2 * ld_v

            mesh_dy.verts.nc[v0] += 1
            mesh_dy.verts.nc[v2] += 1

            vTan0 = mesh_dy.verts.v[v0] + mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0 * ld_v
            vTan2 = mesh_dy.verts.v[v2] + mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2 * ld_v

            g0Tan = -(vTan0 - vTan2)
            g2Tan = vTan0 - vTan2
            cTan = 0.5 * (g2Tan.dot(g2Tan) + g0Tan.dot(g0Tan))
            schur = mesh_dy.verts.m_inv[v0] * g0Tan.dot(g0Tan) + mesh_dy.verts.m_inv[v2] * g2Tan.dot(g2Tan) + 1e-4
            ldTan = cTan / schur
            dvTan0 = mesh_dy.verts.m_inv[v0] * ldTan * g0Tan
            dvTan2 = mesh_dy.verts.m_inv[v1] * ldTan * g2Tan

            if mu * abs(Cv) > cTan:
                mu = 1.0
            mesh_dy.verts.dv[v0] += mu * dvTan0
            mesh_dy.verts.dv[v2] += mu * dvTan2

        elif dtype == 2:
            mesh_dy.verts.dv[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0 * ld_v
            mesh_dy.verts.dv[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3 * ld_v
            mesh_dy.verts.nc[v0] += 1
            mesh_dy.verts.nc[v3] += 1

            vTan0 = mesh_dy.verts.v[v0] + mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0 * ld_v
            vTan3 = mesh_dy.verts.v[v3] + mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3 * ld_v

            g0Tan = -(vTan0 - vTan3)
            g3Tan = vTan0 - vTan3
            cTan = 0.5 * (g3Tan.dot(g3Tan) + g0Tan.dot(g0Tan))
            schur = mesh_dy.verts.m_inv[v0] * g0Tan.dot(g0Tan) + mesh_dy.verts.m_inv[v3] * g3Tan.dot(g3Tan) + 1e-4
            ldTan = cTan / schur
            dvTan0 = mesh_dy.verts.m_inv[v0] * ldTan * g0Tan
            dvTan3 = mesh_dy.verts.m_inv[v3] * ldTan * g3Tan

            if mu * abs(Cv) > cTan:
                mu = 1.0
            mesh_dy.verts.dv[v0] += mu * dvTan0
            mesh_dy.verts.dv[v3] += mu * dvTan3

        elif dtype == 3:

            mesh_dy.verts.dv[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0 * ld_v
            mesh_dy.verts.dv[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1 * ld_v
            mesh_dy.verts.dv[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2 * ld_v
            mesh_dy.verts.nc[v0] += 1
            mesh_dy.verts.nc[v1] += 1
            mesh_dy.verts.nc[v2] += 1

            vTan0 = mesh_dy.verts.v[v0] + mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0 * ld_v
            vTan1 = mesh_dy.verts.v[v1] + mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1 * ld_v
            vTan2 = mesh_dy.verts.v[v2] + mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2 * ld_v

            a, b = g1.norm(), g2.norm()
            ab = a + b
            p = (a * vTan1 + b * vTan2) / ab
            g0Tan = p - vTan0
            g1Tan = -( a / ab) * g0Tan
            g2Tan = -( b / ab) * g0Tan
            cTan = 0.5 * (g0Tan.dot(g0Tan) + g1Tan.dot(g1Tan) + g2Tan.dot(g2Tan))
            schur = mesh_dy.verts.m_inv[v0] * g0Tan.dot(g0Tan) + mesh_dy.verts.m_inv[v1] * g1Tan.dot(g1Tan) + mesh_dy.verts.m_inv[v2] * g2Tan.dot(g2Tan) + 1e-4
            ldTan = cTan / schur
            dvTan0 = mesh_dy.verts.m_inv[v0] * ldTan * g0Tan
            dvTan1 = mesh_dy.verts.m_inv[v1] * ldTan * g1Tan
            dvTan2 = mesh_dy.verts.m_inv[v2] * ldTan * g2Tan
            if mu * abs(Cv) > cTan:
                mu = 1.0
            mesh_dy.verts.dv[v0] += mu * dvTan0
            mesh_dy.verts.dv[v1] += mu * dvTan1
            mesh_dy.verts.dv[v2] += mu * dvTan2

        elif dtype == 4:

            mesh_dy.verts.dv[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0 * ld_v
            mesh_dy.verts.dv[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2 * ld_v
            mesh_dy.verts.dv[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3 * ld_v
            mesh_dy.verts.nc[v0] += 1
            mesh_dy.verts.nc[v2] += 1
            mesh_dy.verts.nc[v3] += 1

            vTan0 = mesh_dy.verts.v[v0] + mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0 * ld_v
            vTan2 = mesh_dy.verts.v[v2] + mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2 * ld_v
            vTan3 = mesh_dy.verts.v[v3] + mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3 * ld_v

            a, b = g2.norm(), g3.norm()
            ab = a + b
            p = (a * vTan2 + b * vTan3) /ab
            g0Tan = p - vTan0
            g2Tan = -(a/ab) * g0Tan
            g3Tan = -(b/ab) * g0Tan
            cTan = 0.5 * (g0Tan.dot(g0Tan) + g2Tan.dot(g2Tan) + g3Tan.dot(g3Tan))
            schur = mesh_dy.verts.m_inv[v0] * g0Tan.dot(g0Tan) + mesh_dy.verts.m_inv[v2] * g2Tan.dot(g2Tan) + mesh_dy.verts.m_inv[v3] * g3Tan.dot(g3Tan) + 1e-4
            ldTan = cTan / schur
            dvTan0 = mesh_dy.verts.m_inv[v0] * ldTan * g0Tan
            dvTan2 = mesh_dy.verts.m_inv[v2] * ldTan * g2Tan
            dvTan3 = mesh_dy.verts.m_inv[v3] * ldTan * g3Tan
            if mu * abs(Cv) > cTan:
                mu = 1.0
            mesh_dy.verts.dv[v0] += mu * dvTan0
            mesh_dy.verts.dv[v2] += mu * dvTan2
            mesh_dy.verts.dv[v3] += mu * dvTan3


        elif dtype == 5:

            mesh_dy.verts.dv[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0 * ld_v
            mesh_dy.verts.dv[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1 * ld_v
            mesh_dy.verts.dv[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3 * ld_v
            mesh_dy.verts.nc[v0] += 1
            mesh_dy.verts.nc[v1] += 1
            mesh_dy.verts.nc[v3] += 1

            vTan0 = mesh_dy.verts.v[v0] + mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0 * ld_v
            vTan1 = mesh_dy.verts.v[v1] + mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1 * ld_v
            vTan3 = mesh_dy.verts.v[v3] + mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3 * ld_v

            a, b = g1.norm(), g3.norm()
            ab = a + b
            p = (a * vTan1 + b * vTan3) / ab
            g0Tan = p - vTan0
            g1Tan = -( a/ab)* g0Tan
            g3Tan = -( b/ab)* g0Tan
            cTan = 0.5 * (g0Tan.dot(g0Tan) + g1Tan.dot(g1Tan) + g3Tan.dot(g3Tan))
            schur = mesh_dy.verts.m_inv[v0] * g0Tan.dot(g0Tan) + mesh_dy.verts.m_inv[v1] * g1Tan.dot(g1Tan) + mesh_dy.verts.m_inv[v3] * g3Tan.dot(g3Tan) + 1e-4
            ldTan = cTan / schur
            dvTan0 = mesh_dy.verts.m_inv[v0] * ldTan * g0Tan
            dvTan1 = mesh_dy.verts.m_inv[v1] * ldTan * g1Tan
            dvTan3 = mesh_dy.verts.m_inv[v3] * ldTan * g3Tan
            if mu * abs(Cv) > cTan:
                mu = 1.0
            mesh_dy.verts.dv[v0] += mu * dvTan0
            mesh_dy.verts.dv[v1] += mu * dvTan1
            mesh_dy.verts.dv[v3] += mu * dvTan3

        elif dtype == 6:

            mesh_dy.verts.dv[v0] += mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0 * ld_v
            mesh_dy.verts.dv[v1] += mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1 * ld_v
            mesh_dy.verts.dv[v2] += mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2 * ld_v
            mesh_dy.verts.dv[v3] += mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3 * ld_v

            mesh_dy.verts.nc[v0] += 1
            mesh_dy.verts.nc[v1] += 1
            mesh_dy.verts.nc[v2] += 1
            mesh_dy.verts.nc[v3] += 1

            vTan0 = mesh_dy.verts.v[v0] + mesh_dy.verts.fixed[v0] * mesh_dy.verts.m_inv[v0] * g0 * ld_v
            vTan1 = mesh_dy.verts.v[v1] + mesh_dy.verts.fixed[v1] * mesh_dy.verts.m_inv[v1] * g1 * ld_v
            vTan2 = mesh_dy.verts.v[v2] + mesh_dy.verts.fixed[v2] * mesh_dy.verts.m_inv[v2] * g2 * ld_v
            vTan3 = mesh_dy.verts.v[v3] + mesh_dy.verts.fixed[v3] * mesh_dy.verts.m_inv[v3] * g3 * ld_v

            a, b, c = g1.norm(), g2.norm(), g3.norm()
            abc = a + b + c
            p = (a * vTan1 + b * vTan2 + c * vTan3) / abc
            g0Tan = p - vTan0
            g1Tan = - (a/abc) * g0Tan
            g2Tan = - (b/abc) * g0Tan
            g3Tan = - (c/abc) * g0Tan
            cTan = 0.5 * (g0Tan.dot(g0Tan) + g1Tan.dot(g1Tan) + g2Tan.dot(g2Tan) + g3Tan.dot(g3Tan))
            schur = mesh_dy.verts.m_inv[v0] * g0Tan.dot(g0Tan) + mesh_dy.verts.m_inv[v1] * g1Tan.dot(g1Tan) + mesh_dy.verts.m_inv[v2] * g2Tan.dot(g2Tan) + mesh_dy.verts.m_inv[v3] * g3Tan.dot(g3Tan) + 1e-4
            ldTan = cTan / schur
            dvTan0 = mesh_dy.verts.m_inv[v0] * ldTan * g0Tan
            dvTan1 = mesh_dy.verts.m_inv[v1] * ldTan * g1Tan
            dvTan2 = mesh_dy.verts.m_inv[v2] * ldTan * g2Tan
            dvTan3 = mesh_dy.verts.m_inv[v3] * ldTan * g3Tan
            if mu * abs(Cv) > cTan:
                mu = 1.0
            mesh_dy.verts.dv[v0] += mu * dvTan0
            mesh_dy.verts.dv[v1] += mu * dvTan1
            mesh_dy.verts.dv[v2] += mu * dvTan2
            mesh_dy.verts.dv[v3] += mu * dvTan3

@ti.func
def solve_collision_ee_static_v(self, eid_d, eid_s, dHat, mu):

    v0 = self.edge_indices_dynamic[2 * eid_d + 0]
    v1 = self.edge_indices_dynamic[2 * eid_d + 1]
    v2 = self.edge_indices_static[2 * eid_s + 0]
    v3 = self.edge_indices_static[2 * eid_s + 1]

    x0, x1 = self.y[v0], self.y[v1]
    x2, x3 = self.x_static[v2], self.x_static[v3]

    dtype = di.d_type_EE(x0, x1, x2, x3)

    if dtype == 0:
        d = di.d_PP(x0, x2)
        if d < dHat:
            g0, g2 = di.g_PP(x0, x2)
            schur = self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) + 1e-4
            Cv = g0.dot(self.v[v0]) + g2.dot(self.v_static[v2])
            if Cv < 0.0:
                ld_v = -Cv / schur

                self.dv[v0] += self.fixed[v0] * self.m_inv[v0] * ld_v * g0
                self.nc[v0] += 1

                vTan0 = self.v[v0] + self.fixed[v0] * self.m_inv[v0] * ld_v * g0
                g0Tan = self.v_static[v2] - vTan0
                cTan = 0.5 * g0Tan.dot(g0Tan)
                schur = self.m_inv[v0] * g0Tan.dot(g0Tan) + 1e-4
                ldTan = cTan / schur
                dvTan0 = self.m_inv[v0] * ldTan * g0Tan

                if mu * abs(Cv) > cTan:
                    mu = 1.0

                self.dv[v0] += mu * dvTan0

    elif dtype == 1:
        d = di.d_PP(x0, x3)
        if d < dHat:
            g0, g3 = di.g_PP(x0, x3)
            schur = self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) + 1e-4
            Cv = g0.dot(self.v[v0]) + g3.dot(self.v_static[v3])
            if Cv < 0.0:
                ld_v = -Cv / schur
                self.dv[v0] += self.fixed[v0] * self.m_inv[v0] * ld_v * g0
                self.nc[v0] += 1

                vTan0 = self.v[v0] + self.fixed[v0] * self.m_inv[v0] * ld_v * g0
                g0Tan = self.v_static[v3] - vTan0
                cTan = 0.5 * g0Tan.dot(g0Tan)
                schur = self.m_inv[v0] * g0Tan.dot(g0Tan) + 1e-4

                ldTan = cTan / schur
                dvTan0 = self.m_inv[v0] * ldTan * g0Tan
                if mu * abs(Cv) > cTan:
                    mu = 1.0

                self.dv[v0] += mu * dvTan0

    elif dtype == 2:
        d = di.d_PE(x0, x2, x3)
        if d < dHat:
            g0, g2, g3 = di.g_PE(x0, x2, x3)
            schur = self.fixed[v0] * self.m_inv[v0] * g0.dot(g0) + 1e-4
            Cv = g0.dot(self.v[v0]) + g3.dot(self.v_static[v3])
            if Cv < 0.0:
                ld_v = -Cv / schur
                self.dv[v0] += self.fixed[v0] * self.m_inv[v0] * ld_v * g0
                self.nc[v0] += 1

                vTan0 = self.v[v0] + self.fixed[v0] * self.m_inv[v0] * ld_v * g0
                c, d = g2.norm(), g3.norm()
                c1 = c / (c + d)
                d1 = d / (c + d)

                p1 = c1 * self.v_static[v2] + d1 * self.v_static[v3]

                g0Tan = (p1 - vTan0)
                cTan = 0.5 * (p1 - vTan0).dot(p1 - vTan0)
                schur = self.m_inv[v0] * g0Tan.dot(g0Tan) + 1e-4
                ldTan = cTan / schur
                dvTan0 = self.m_inv[v0] * ldTan * g0Tan

                if mu * abs(Cv) > cTan:
                    mu = 1.0

                self.dv[v0] += mu * dvTan0

    elif dtype == 3:
        d = di.d_PP(x1, x2)
        if d < dHat:
            g1, g2 = di.g_PP(x1, x2)
            schur = self.fixed[v1] * self.m_inv[v1] * g1.dot(g1) + 1e-4
            Cv = g1.dot(self.v[v1]) + g2.dot(self.v_static[v2])
            if Cv < 0.0:
                ld_v = -Cv / schur
                self.dv[v1] += self.fixed[v1] * self.m_inv[v1] * ld_v * g1
                self.nc[v1] += 1

                vTan1 = self.v[v1] + self.fixed[v1] * self.m_inv[v1] * ld_v * g1
                g1Tan = self.v_static[v2] - vTan1
                cTan = 0.5 * g1Tan.dot(g1Tan)
                schur = self.m_inv[v1] * g1Tan.dot(g1Tan) + 1e-4
                ldTan = cTan / schur
                dvTan1 = self.m_inv[v1] * ldTan * g1Tan
                if mu * abs(Cv) > cTan:
                    mu = 1.0
                self.dv[v1] += mu * dvTan1

    elif dtype == 4:
        d = di.d_PP(x1, x3)
        if d < dHat:
            g1, g3 = di.g_PP(x1, x3)
            schur = self.fixed[v1] * self.m_inv[v1] * g1.dot(g1) + 1e-4
            Cv = g1.dot(self.v[v1]) + g3.dot(self.v_static[v3])
            if Cv < 0.0:
                ld_v = -Cv / schur
                self.dv[v1] += self.fixed[v1] * self.m_inv[v1] * ld_v * g1
                self.nc[v1] += 1

                vTan1 = self.v[v1] + self.fixed[v1] * self.m_inv[v1] * ld_v * g1
                g1Tan = self.v_static[v3] - vTan1
                cTan = 0.5 * g1Tan.dot(g1Tan)
                schur = self.m_inv[v1] * g1Tan.dot(g1Tan) + 1e-4
                ldTan = cTan / schur
                dvTan1 = self.m_inv[v1] * ldTan * g1Tan
                if mu * abs(Cv) > cTan:
                    mu = 1.0
                self.dv[v1] += mu * dvTan1

    elif dtype == 5:
        d = di.d_PE(x1, x2, x3)
        if d < dHat:
            g1, g2, g3 = di.g_PE(x1, x2, x3)
            schur = self.m_inv[v1] * g1.dot(g1) + 1e-4
            Cv = g1.dot(self.v[v1]) + g2.dot(self.v_static[v2]) + g3.dot(self.v_static[v3])
            if Cv < 0.0:
                ld_v = -Cv / schur
                self.dv[v1] += self.fixed[v1] * self.m_inv[v1] * ld_v * g1
                self.nc[v1] += 1

                c, d = g2.norm(), g3.norm()
                c1 = c / (c + d)
                d1 = d / (c + d)

                p1 = c1 * self.v_static[v2] + d1 * self.v_static[v3]

                vTan1 = self.v[v1] + self.fixed[v1] * self.m_inv[v1] * ld_v * g1
                g1Tan = p1 - vTan1
                cTan = 0.5 * (p1 - vTan1).dot(p1 - vTan1)
                schur = self.m_inv[v1] * g1Tan.dot(g1Tan) + 1e-4
                ldTan = cTan / schur
                dvTan1 = self.m_inv[v1] * ldTan * g1Tan
                if mu * abs(Cv) > cTan:
                    mu = 1.0
                self.dv[v1] += mu * dvTan1

    elif dtype == 6:
        d = di.d_PE(x2, x0, x1)
        if d < dHat:
            g2, g0, g1 = di.g_PE(x2, x0, x1)
            schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v1] * g1.dot(g1) + 1e-4
            Cv = g0.dot(self.v_static[v0]) + g1.dot(self.v[v1]) + g2.dot(self.v_static[v2])
            if Cv < 0.0:
                ld_v = -Cv / schur
                self.dv[v0] += self.fixed[v0] * self.m_inv[v0] * ld_v * g0
                self.dv[v1] += self.fixed[v1] * self.m_inv[v1] * ld_v * g1
                self.nc[v0] += 1
                self.nc[v1] += 1

                vTan0 = self.v[v0] + self.fixed[v0] * self.m_inv[v0] * ld_v * g0
                vTan1 = self.v[v1] + self.fixed[v1] * self.m_inv[v1] * ld_v * g1
                a, b = g0.norm(), g1.norm()

                a1 = a / (a + b)
                b1 = b / (a + b)
                p0 = a1 * vTan0 + b1 * vTan1

                g0Tan = a1 * (self.v_static[v2] - p0)
                g1Tan = b1 * (self.v_static[v2] - p0)
                cTan = 0.5 * (self.v_static[v2] - p0).dot(self.v_static[v2] - p0)
                schur = self.m_inv[v0] * g0Tan.dot(g0Tan) + self.m_inv[v1] * g1Tan.dot(g1Tan) + 1e-4
                ldTan = cTan / schur
                dvTan0 = self.m_inv[v0] * ldTan * g0Tan
                dvTan1 = self.m_inv[v1] * ldTan * g1Tan
                if mu * abs(Cv) > cTan:
                    mu = 1.0

                self.dv[v0] += mu * dvTan0
                self.dv[v1] += mu * dvTan1

    elif dtype == 7:
        d = di.d_PE(x3, x0, x1)
        if d < dHat:
            g3, g0, g1 = di.g_PE(x3, x0, x1)
            schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v1] * g1.dot(g1) + 1e-4
            Cv = g0.dot(self.v_static[v0]) + g1.dot(self.v[v1]) + g3.dot(self.v_static[v3])
            if Cv < 0.0:
                ld_v = -Cv / schur

                self.dv[v0] += self.fixed[v0] * self.m_inv[v0] * ld_v * g0
                self.dv[v1] += self.fixed[v1] * self.m_inv[v1] * ld_v * g1

                self.nc[v0] += 1
                self.nc[v1] += 1

                vTan0 = self.v[v0] + self.fixed[v0] * self.m_inv[v0] * ld_v * g0
                vTan1 = self.v[v1] + self.fixed[v1] * self.m_inv[v1] * ld_v * g1
                a, b = g0.norm(), g1.norm()

                a1 = a / (a + b)
                b1 = b / (a + b)
                p0 = a1 * vTan0 + b1 * vTan1

                g0Tan = a1 * (self.v_static[v3] - p0)
                g1Tan = b1 * (self.v_static[v3] - p0)
                cTan = 0.5 * (self.v_static[v3] - p0).dot(self.v_static[v3] - p0)
                schur = self.m_inv[v0] * g0Tan.dot(g0Tan) + self.m_inv[v1] * g1Tan.dot(g1Tan) + 1e-4
                ldTan = cTan / schur
                dvTan0 = self.m_inv[v0] * ldTan * g0Tan
                dvTan1 = self.m_inv[v1] * ldTan * g1Tan

                if mu * abs(Cv) > cTan:
                    mu = 1.0

                self.dv[v0] += mu * dvTan0
                self.dv[v1] += mu * dvTan1

    if dtype == 8:
        d = di.d_EE(x0, x1, x2, x3)
        if d < dHat:
            g0, g1, g2, g3 = di.g_EE(x0, x1, x2, x3)
            schur = self.m_inv[v0] * g0.dot(g0) + self.m_inv[v1] * g1.dot(g1) + 1e-4
            Cv = g0.dot(self.v_static[v0]) + g1.dot(self.v[v1]) + g3.dot(self.v_static[v3])
            if Cv < 0.0:
                ld_v = -Cv / schur

                self.dv[v0] += self.fixed[v0] * self.m_inv[v0] * ld_v * g0
                self.dv[v1] += self.fixed[v1] * self.m_inv[v1] * ld_v * g1

                self.nc[v0] += 1
                self.nc[v1] += 1

                vTan0 = self.v[v0] + self.fixed[v0] * self.m_inv[v0] * ld_v * g0
                vTan1 = self.v[v1] + self.fixed[v1] * self.m_inv[v1] * ld_v * g1

                a, b, c, d = g0.norm(), g1.norm(), g2.norm(), g3.norm()

                a1 = a / (a + b)
                b1 = b / (a + b)
                c1 = c / (c + d)
                d1 = d / (c + d)

                p0 = a1 * vTan0 + b1 * vTan1
                p1 = c1 * self.v_static[v2] + d1 * self.v_static[v3]

                g0Tan = a1 * (p1 - p0)
                g1Tan = b1 * (p1 - p0)
                cTan = 0.5 * (p1 - p0).dot(p1 - p0)
                schur = self.m_inv[v0] * g0Tan.dot(g0Tan) + self.m_inv[v1] * g1Tan.dot(g1Tan) + 1e-4
                ldTan = cTan / schur
                dvTan0 = self.m_inv[v0] * ldTan * g0Tan
                dvTan1 = self.m_inv[v1] * ldTan * g1Tan

                if mu * abs(Cv) > cTan:
                    mu = 1.0

                self.dv[v0] += mu * dvTan0
                self.dv[v1] += mu * dvTan1

@ti.func
def solve_collision_ee_dynamic_v(self, ei0, ei1, dtype, g0, g1, g2, g3, schur, mu):

    v0 = self.edge_indices_dynamic[2 * ei0 + 0]
    v1 = self.edge_indices_dynamic[2 * ei0 + 1]

    v2 = self.edge_indices_dynamic[2 * ei1 + 0]
    v3 = self.edge_indices_dynamic[2 * ei1 + 1]

    dvn = g0.dot(self.v[v0]) + g1.dot(self.v[v1]) + g2.dot(self.v[v2]) + g3.dot(self.v[v3])
    if dvn < 0.0:
        ld = dvn / schur

        if dtype == 0:

            self.dv[v0] -= self.fixed[v0] * self.m_inv[v0] * g0 * ld
            self.dv[v2] -= self.fixed[v2] * self.m_inv[v2] * g2 * ld

            self.nc[v0] += 1
            self.nc[v2] += 1

        elif dtype == 1:

            self.dv[v0] -= self.fixed[v0] * self.m_inv[v0] * g0 * ld
            self.dv[v3] -= self.fixed[v3] * self.m_inv[v3] * g3 * ld

            self.nc[v0] += 1
            self.nc[v3] += 1

        elif dtype == 2:

            self.dv[v0] -= self.fixed[v0] * self.m_inv[v0] * g0 * ld
            self.dv[v2] -= self.fixed[v2] * self.m_inv[v2] * g2 * ld
            self.dv[v3] -= self.fixed[v3] * self.m_inv[v3] * g3 * ld

            self.nc[v0] += 1
            self.nc[v2] += 1
            self.nc[v3] += 1

        elif dtype == 3:

            self.dv[v1] -= self.fixed[v1] * self.m_inv[v1] * g1 * ld
            self.dv[v2] -= self.fixed[v2] * self.m_inv[v2] * g2 * ld

            self.nc[v1] += 1
            self.nc[v2] += 1

        elif dtype == 4:

            self.dv[v1] -= self.fixed[v1] * self.m_inv[v1] * g1 * ld
            self.dv[v3] -= self.fixed[v3] * self.m_inv[v3] * g3 * ld

            self.nc[v1] += 1
            self.nc[v3] += 1

        elif dtype == 5:

            self.dv[v1] -= self.fixed[v1] * self.m_inv[v1] * g1 * ld
            self.dv[v2] -= self.fixed[v2] * self.m_inv[v2] * g2 * ld
            self.dv[v3] -= self.fixed[v3] * self.m_inv[v3] * g3 * ld

            self.nc[v1] += 1
            self.nc[v2] += 1
            self.nc[v3] += 1

        elif dtype == 6:

            self.dv[v0] -= self.fixed[v0] * self.m_inv[v0] * g0 * ld
            self.dv[v1] -= self.fixed[v1] * self.m_inv[v1] * g1 * ld
            self.dv[v2] -= self.fixed[v2] * self.m_inv[v2] * g2 * ld

            self.nc[v0] += 1
            self.nc[v1] += 1
            self.nc[v2] += 1

        elif dtype == 7:

            self.dv[v0] -= self.fixed[v0] * self.m_inv[v0] * g0 * ld
            self.dv[v1] -= self.fixed[v1] * self.m_inv[v1] * g1 * ld
            self.dv[v3] -= self.fixed[v3] * self.m_inv[v3] * g3 * ld

            self.nc[v0] += 1
            self.nc[v1] += 1
            self.nc[v3] += 1

        elif dtype == 8:

            self.dv[v0] -= self.fixed[v0] * self.m_inv[v0] * g0 * ld
            self.dv[v1] -= self.fixed[v1] * self.m_inv[v1] * g1 * ld
            self.dv[v2] -= self.fixed[v2] * self.m_inv[v2] * g2 * ld
            self.dv[v3] -= self.fixed[v3] * self.m_inv[v3] * g3 * ld

            self.nc[v0] += 1
            self.nc[v1] += 1
            self.nc[v2] += 1
            self.nc[v3] += 1

