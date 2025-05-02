import taichi as ti

@ti.data_oriented
class CCDModule:
    def __init__(self):
        pass

    @ti.func
    def point_point_distance(self, v0, v1):
        return (v0 - v1).norm_sqr()

    @ti.func
    def point_edge_distance(self, v0, v1, v2):
        num = (v1 - v0).cross(v2 - v0).norm_sqr()
        denom = (v2 - v1).norm_sqr()
        return num / denom

    @ti.func
    def point_triangle_distance(self, v0, v1, v2, v3):
        b = (v2 - v1).cross(v3 - v1)
        aTb = (v0 - v1).dot(b)
        return (aTb * aTb) / b.norm_sqr()

    @ti.func
    def edge_edge_distance(self, v0, v1, v2, v3):
        u = v1 - v0; v = v3 - v2; b = u.cross(v)
        aTb = (v2 - v0).dot(b)
        return (aTb * aTb) / b.norm_sqr()

    @ti.func
    def classify_edge(self, p, a, b, normal):
        basis0 = b - a
        basis1 = basis0.cross(normal)
        basis2 = p - a

        D = ti.Matrix.cols([basis0, basis1, normal])
        D1 = ti.Matrix.cols([basis2, basis1, normal])
        D2 = ti.Matrix.cols([basis0, basis2, normal])

        detD = D.determinant()
        u = u = ti.cast(0.0, ti.f64); v = ti.cast(0.0, ti.f64)
        if ti.abs(detD) >= 1e-12:
            u = D1.determinant() / detD
            v = D2.determinant() / detD

        return ti.Vector([u, v])

    @ti.func
    def dtype_point_triangle(self, p, v1, v2, v3):
        normal = (v2 - v1).cross(v3 - v1)
        uv0 = self.classify_edge(p, v1, v2, normal)
        u0, v0 = uv0[0], uv0[1]
        uv1 = self.classify_edge(p, v2, v3, normal)
        u1, v1_ = uv1[0], uv1[1]
        uv2 = self.classify_edge(p, v3, v1, normal)
        u2, v2_ = uv2[0], uv2[1]

        res = 6
        if 0 < u0 < 1 and v0 >= 0: res = 3
        elif 0 < u1 < 1 and v1_ >= 0: res = 4
        elif 0 < u2 < 1 and v2_ >= 0: res = 5
        elif u0 <= 0 and u2 >= 1: res = 0
        elif u1 <= 0 and u0 >= 1: res = 1
        elif u2 <= 0 and u1 >= 1: res = 2

        return res

    @ti.func
    def _dtype_edge_edge(self, v0, v1, v2, v3):
        _v0 = ti.cast(v0, ti.f64); _v1 = ti.cast(v1, ti.f64); _v2 = ti.cast(v2, ti.f64); _v3 = ti.cast(v3, ti.f64)
        u = _v1 - _v0; v = _v3 - _v2; w = _v0 - _v2
        a = u.dot(u); b = u.dot(v); c = v.dot(v)
        d = u.dot(w); e = v.dot(w)

        D = a * c - b * b; tD = D
        sN = b * e - c * d
        tN = ti.cast(0.0, ti.f64)
        result = 8

        if sN <= 0.0:
            tN = e; tD = c
            result = 2
        elif sN >= D:
            tN = e + b; tD = c
            result = 5
        else:
            tN = a * e - b * d
            cross_uv = u.cross(v)
            if tN > 0.0 and tN < tD and (w.dot(cross_uv) == 0.0 or cross_uv.norm_sqr() < 1.0e-20 * a * c):
                if sN < D * 0.5:
                    tN = e; tD = c
                    result = 2
                else:
                    tN = e + b; tD = c
                    result = 5

        if tN <= 0.0:
            if -d <= 0.0: result = 0
            elif -d >= a: result = 3
            else: result = 6
        elif tN >= tD:
            if -d + b <= 0.0: result = 1
            elif -d + b >= a: result = 4
            else: result = 7

        return result

    @ti.func
    def point_triangle_distance_unclassified(self, p, t0, t1, t2):
        dtype = self.dtype_point_triangle(p, t0, t1, t2)
        res = ti.cast(1e32, ti.f64)
        if dtype == 0: res = self.point_point_distance(p, t0)
        elif dtype == 1: res = self.point_point_distance(p, t1)
        elif dtype == 2: res = self.point_point_distance(p, t2)
        elif dtype == 3: res = self.point_edge_distance(p, t0, t1)
        elif dtype == 4: res = self.point_edge_distance(p, t1, t2)
        elif dtype == 5: res = self.point_edge_distance(p, t2, t0)
        elif dtype == 6: res = self.point_triangle_distance(p, t0, t1, t2)
        return res

    @ti.func
    def edge_edge_distance_unclassified(self, ea0, ea1, eb0, eb1):
        dtype = self._dtype_edge_edge(ea0, ea1, eb0, eb1)
        res = ti.cast(1e32, ti.f64)
        if dtype == 0: res = self.point_point_distance(ea0, eb0)
        elif dtype == 1: res = self.point_point_distance(ea0, eb1)
        elif dtype == 2: res = self.point_edge_distance(ea0, eb0, eb1)
        elif dtype == 3: res = self.point_point_distance(ea1, eb0)
        elif dtype == 4: res = self.point_point_distance(ea1, eb1)
        elif dtype == 5: res = self.point_edge_distance(ea1, eb0, eb1)
        elif dtype == 6: res = self.point_edge_distance(eb0, ea0, ea1)
        elif dtype == 7: res = self.point_edge_distance(eb1, ea0, ea1)
        elif dtype == 8: res = self.edge_edge_distance(ea0, ea1, eb0, eb1)
        return res

    @ti.func
    def point_triangle_ccd(self, p, t0, t1, t2, dp, dt0, dt1, dt2, eta: ti.f64, thickness: ti.f64):
        _p = ti.cast(p, ti.f64); _t0 = ti.cast(t0, ti.f64); _t1 = ti.cast(t1, ti.f64); _t2 = ti.cast(t2, ti.f64)
        _dp = ti.cast(dp, ti.f64); _dt0 = ti.cast(dt0, ti.f64); _dt1 = ti.cast(dt1, ti.f64); _dt2 = ti.cast(dt2, ti.f64)

        mov = (_dt0 + _dt1 + _dt2 + _dp) * (-0.25)
        _dp = _dp + mov; _dt0 = _dt0 + mov; _dt1 = _dt1 + mov; _dt2 = _dt2 + mov

        max_disp_mag = _dp.norm() + ti.sqrt(ti.max(_dt0.norm_sqr(), ti.max(_dt1.norm_sqr(), _dt2.norm_sqr())))
        toc = ti.cast(1.0, ti.f64)

        if max_disp_mag != 0:
            dist2_cur = self.point_triangle_distance_unclassified(_p, _t0, _t1, _t2)
            dist_cur = ti.sqrt(dist2_cur)
            gap = eta * (dist2_cur - thickness * thickness) / (dist_cur + thickness)

            toc = 0.0
            count = 0

            while count <= 50000:
                count += 1
                toc_lower_bound = (1 - eta) * (dist2_cur - thickness * thickness) / (
                            (dist_cur + thickness) * max_disp_mag)

                _p = _p + _dp * toc_lower_bound
                _t0 = _t0 + _dt0 * toc_lower_bound
                _t1 = _t1 + _dt1 * toc_lower_bound
                _t2 = _t2 + _dt2 * toc_lower_bound

                dist2_cur = self.point_triangle_distance_unclassified(_p, _t0, _t1, _t2)
                dist_cur = ti.sqrt(dist2_cur)

                if toc > 0 and (dist2_cur - thickness * thickness) / (dist_cur + thickness) < gap:
                    break

                toc += toc_lower_bound
                if toc > 1.0: toc = 1.0; break

        return toc

    @ti.func
    def edge_edge_ccd(self, ea0, ea1, eb0, eb1, dea0, dea1, deb0, deb1, eta: ti.f64, thickness: ti.f64):
        ea0 = ti.cast(ea0, ti.f64); ea1 = ti.cast(ea1, ti.f64); eb0 = ti.cast(eb0, ti.f64); eb1 = ti.cast(eb1, ti.f64)
        dea0 = ti.cast(dea0, ti.f64); dea1 = ti.cast(dea1, ti.f64); deb0 = ti.cast(deb0, ti.f64); deb1 = ti.cast(deb1, ti.f64)

        mov = -0.25 * (dea0 + dea1 + deb0 + deb1)
        dea0 += mov; dea1 += mov; deb0 += mov; deb1 += mov

        max_disp_mag = (ti.sqrt(ti.max(dea0.norm_sqr(), dea1.norm_sqr()))
                        + ti.sqrt(ti.max(deb0.norm_sqr(), deb1.norm_sqr())))
        toc = ti.cast(1.0, ti.f64)

        if max_disp_mag != 0.0:
            dist2_cur = self.edge_edge_distance_unclassified(ea0, ea1, eb0, eb1)
            dFunc = dist2_cur - thickness * thickness

            if dFunc <= 0.0:
                dist2_cur = min((ea0 - eb0).norm_sqr(), (ea0 - eb1).norm_sqr(),
                                (ea1 - eb0).norm_sqr(), (ea1 - eb1).norm_sqr())
                dFunc = dist2_cur - thickness * thickness
            dist_cur = ti.sqrt(dist2_cur)
            gap = eta * dFunc / (dist_cur + thickness)

            toc = 0.0
            count = 0

            while count <= 50000:
                count += 1
                toc_lower_bound = (1.0 - eta) * dFunc / ((dist_cur + thickness) * max_disp_mag)

                ea0 += dea0 * toc_lower_bound; ea1 += dea1 * toc_lower_bound
                eb0 += deb0 * toc_lower_bound; eb1 += deb1 * toc_lower_bound

                dist2_cur = self.edge_edge_distance_unclassified(ea0, ea1, eb0, eb1)
                dFunc = dist2_cur - thickness * thickness

                if dFunc <= 0.0:
                    dist2_cur = min((ea0 - eb0).norm_sqr(), (ea0 - eb1).norm_sqr(),
                                    (ea1 - eb0).norm_sqr(), (ea1 - eb1).norm_sqr())
                    dFunc = dist2_cur - thickness * thickness
                dist_cur = ti.sqrt(dist2_cur)

                if toc > 0.0 and (dFunc / (dist_cur + thickness)) < gap:break
                toc += toc_lower_bound
                if toc > 1.0: toc = 1.0; break
        return toc

    # ToDo temp kernel code
    @ti.kernel
    def point_mesh_ccd(self, p: ti.template(), dp: ti.template(),
                       triangles: ti.template(), tri_moves: ti.template(),
                       eta: ti.f64, thickness: ti.f64) -> ti.f64:

        N = triangles.shape[0]
        min_toc = ti.cast(1.0, ti.f64)

        for i in range(N):
            t0 = triangles[i, 0]; t1 = triangles[i, 1]; t2 = triangles[i, 2]
            dt0 = tri_moves[i, 0]; dt1 = tri_moves[i, 1]; dt2 = tri_moves[i, 2]

            toc = self.point_triangle_ccd(p, t0, t1, t2, dp, dt0, dt1, dt2, eta, thickness)
            if toc < min_toc:
                min_toc = toc

        return min_toc

    # ToDo temp kernel code
    @ti.kernel
    def edges_edges_ccd(self, edges_a: ti.template(), moves_a: ti.template(),
                        edges_b: ti.template(), moves_b: ti.template(),
                        eta: ti.f64, thickness: ti.f64) -> ti.f64:

        N = edges_a.shape[0]; M = edges_b.shape[0]
        min_toc = ti.cast(1.0, ti.f64)

        for i, j in ti.ndrange(N, M):  # 2차원 병렬 루프
            ea0 = edges_a[i, 0]; ea1 = edges_a[i, 1]; dea0 = moves_a[i, 0]; dea1 = moves_a[i, 1]
            eb0 = edges_b[j, 0]; eb1 = edges_b[j, 1]; deb0 = moves_b[j, 0]; deb1 = moves_b[j, 1]

            toc = self.edge_edge_ccd(ea0, ea1, eb0, eb1, dea0, dea1, deb0, deb1, eta, thickness)
            if toc < min_toc:
                min_toc = toc

        return min_toc