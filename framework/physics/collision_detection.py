import taichi as ti
import numpy as np
import meshio

from framework.meshio.TriMesh import *

@ti.data_oriented
class CollisionDetection:
    def __init__(self, mesh: TriMesh, dHat=1e-4, grid_size=1e-1, max_num_objects=100000):
        self.mesh = mesh
        self.dHat = dHat
        self.grid_size = grid_size
        self.max_num_objects = max_num_objects

        self.inv_grid_size = 1 / self.grid_size
        self.epsilon = 1e-6
        self.table_size = self.max_num_objects * 2

        ################################################################################################################
        # Jacobi

        # vertex-face(PT) collision
        self.cell_start_idx_verts_jacobi = ti.field(dtype=ti.i32)
        self.cell_entries_verts_jacobi = ti.field(dtype=ti.i32)
        ti.root.dense(ti.i, self.table_size + 1).place(self.cell_start_idx_verts_jacobi)
        ti.root.dense(ti.i, self.table_size).place(self.cell_entries_verts_jacobi)

        # edge-edge(EE) collision
        self.cell_start_idx_edges_jacobi = ti.field(dtype=ti.i32)
        self.cell_entries_edges_jacobi = ti.field(dtype=ti.i32)
        ti.root.dense(ti.i, self.table_size + 1).place(self.cell_start_idx_edges_jacobi)
        ti.root.dense(ti.i, self.table_size).place(self.cell_entries_edges_jacobi)

        self.max_constraints = 2 ** 21
        self.pair_jacobi = ti.types.struct(a=ti.types.vector(n=4, dtype=ti.i32), # four indices w.r.t. a constraint
                                           b=ti.f32,                             # constraint distance
                                           c=ti.types.vector(n=4, dtype=ti.f32), # constraint coords
                                           d=ti.types.vector(n=3, dtype=ti.f32)) # direction vector t
        self.cid_jacobi = self.pair_jacobi.field() # store the pair data
        self.cid_root_jacobi = ti.root.bitmasked(ti.ij, (2, self.max_constraints)).place(self.cid_jacobi)
        self.pse_verts_jacobi = ti.algorithms.PrefixSumExecutor(self.table_size + 1)
        self.pse_edges_jacobi = ti.algorithms.PrefixSumExecutor(self.table_size + 1)

        self.cell_start_idx_verts_jacobi.fill(0)
        self.cell_entries_verts_jacobi.fill(-1)

        self.cell_start_idx_edges_jacobi.fill(0)
        self.cell_entries_edges_jacobi.fill(-1)

        ################################################################################################################
        # Euler

    ####################################################################################################################

    @ti.kernel
    def find_collision_jacobi(self):
        self.cid_root_jacobi.deactivate_all()

        # vert-face (PT)
        self.cell_start_idx_verts_jacobi.fill(0)
        self.cell_entries_verts_jacobi.fill(-1)
        self.count_cells_verts_jacobi()
        self.pse_verts_jacobi.run(self.cell_start_idx_verts_jacobi)
        self.fill_cells_verts_jacobi()
        self.find_constraints_vert_face_jacobi()

        # edge-edge (EE)
        self.cell_start_idx_edges_jacobi.fill(0)
        self.cell_entries_edges_jacobi.fill(-1)
        self.count_cells_edges_jacobi()
        self.pse_edges_jacobi.run(self.cell_start_idx_edges_jacobi)
        self.fill_cells_edges_jacobi()
        self.find_constraints_edge_edge_jacobi()

    @ti.kernel
    def count_cells_verts_jacobi(self):
        for i in range(self.mesh.num_verts):
            x = self.mesh.x[i]
            hash_id = self.get_hash(x)
            ti.atomic_add(self.cell_start_idx_verts_jacobi[hash_id], 1)
            print(x[0], x[1], x[2], hash_id)

    @ti.kernel
    def fill_cells_verts_jacobi(self):
        for i in range(self.mesh.num_verts):
            x = self.mesh.x[i]
            hash_id = self.get_hash(x)
            id_old = ti.atomic_sub(self.cell_start_idx_verts_jacobi[hash_id], 1)
            self.cell_entries_verts_jacobi[id_old - 1] = i

    @ti.kernel
    def find_constraints_vert_face_jacobi(self):
        for i in range(self.mesh.num_faces):
            t0, t1, t2 = self.mesh.fid_field[i, 0], self.mesh.fid_field[i, 1], self.mesh.fid_field[i, 2]
            x0, x1, x2 = self.mesh.x[t0], self.mesh.x[t1], self.mesh.x[t2]
            # compute AABB
            lower = ti.floor((ti.min(x0, x1, x2) - self.dHat) * self.inv_grid_size)
            upper = ti.floor((ti.max(x0, x1, x2) + self.dHat) * self.inv_grid_size) + 1
            # traverse all grid cells in the AABB region
            for I in ti.grouped(ti.ndrange((lower[0], upper[0]), (lower[1], upper[1]), (lower[2], upper[2]))):
                hash_id = self.get_hash(I)
                start, end = self.cell_start_idx_verts_jacobi[hash_id], self.cell_start_idx_verts_jacobi[hash_id + 1]
                # traverse all points in the current cell to inspect whether each point have collision to the triangle
                for j in range(start, end):
                    pid = self.cell_entries_verts_jacobi[j]
                    x_pid = self.mesh.x[pid]
                    self.attempt_vert_face_jacobi(i, pid, t0, t1, t2, x_pid, x0, x1, x2)

    @ti.func
    def attempt_vert_face_jacobi(self, triangle_id, pid, t0, t1, t2, xp, x0, x1, x2):
        # check whether the point belongs to the face, and the point is too far to inspect
        if pid != t0 and pid != t1 and pid != t2 and self.vert_face_ccd_broadphase(xp, x0, x1, x2, self.dHat):
            # if it is not, get barycentric coordinate of the projection point of xp, and the projection point itself
            cord0, cord1, cord2 = self.dist3D_vert_face(xp, x0, x1, x2)
            xt = cord0 * x0 + cord1 * x1 + cord2 * x2
            t_pt = xp - xt
            dist = t_pt.norm()

            # if the distance between xp and xt is lower than dHat, add this to the constraint struct(self.cid)
            if self.epsilon < ti.abs(dist) and ti.abs(dist) < self.dHat:
                ids = ti.Vector([pid, t0, t1, t2], ti.i32)
                cord = ti.Vector([1.0, -cord0, -cord1, -cord2], ti.f32)
                hash_id = self.two_int_to_hash(triangle_id, pid)
                self.cid_jacobi[0, hash_id] = self.pair_jacobi(ids, dist, cord, t_pt)

    @ti.kernel
    def count_cells_edges_jacobi(self):
        for e in range(self.mesh.num_edges):
            e0, e1 = self.mesh.eid_field[e, 0], self.mesh.eid_field[e, 1]
            x_e0, x_e1 = self.mesh.x[e0], self.mesh.x[e1]
            e0_grid_coord, e1_grid_coord = self.coord_to_grid(x_e0), self.coord_to_grid(x_e1)

            et = e0_grid_coord
            x0, y0, z0 = e0_grid_coord[0], e0_grid_coord[1], e0_grid_coord[2]
            x1, y1, z1 = e1_grid_coord[0], e1_grid_coord[1], e1_grid_coord[2]
            dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
            x_increase, y_increase, z_increase = (1 if dx > 0 else -1), (1 if dy > 0 else -1), (1 if dz > 0 else -1)
            l, m, n = ti.abs(dx), ti.abs(dy), ti.abs(dz)
            dx1, dy1, dz1 = 2*l, 2*m, 2*n

            if (l >= m) and (l >= n):
                err_1, err_2 = dy1 - l, dz1 - l
                for i in range(l):
                    hash_id = self.get_hash(et)
                    ti.atomic_add(self.cell_start_idx_edges_jacobi[hash_id], 1)
                    if err_1 > 0:
                        et[1] += y_increase
                        err_1 -= dx1
                    if err_2 > 0:
                        et[2] += z_increase
                        err_2 -= dx1
                    err_1 += dy1
                    err_2 += dz1
                    et[0] += x_increase

            elif (m >= l) and (m >= n):
                err_1, err_2 = dx1 - m, dz1 - m
                for i in range(m):
                    hash_id = self.get_hash(et)
                    ti.atomic_add(self.cell_start_idx_edges_jacobi[hash_id], 1)
                    if err_1 > 0:
                        et[0] += x_increase
                        err_1 -= dy1
                    if err_2 > 0:
                        et[2] += z_increase
                        err_2 -= dy1
                    err_1 += dx1
                    err_2 += dz1
                    et[1] += y_increase

            else:
                err_1, err_2 = dy1 - n, dx1 - n
                for i in range(n):
                    hash_id = self.get_hash(et)
                    ti.atomic_add(self.cell_start_idx_edges_jacobi[hash_id], 1)
                    if err_1 > 0:
                        et[1] += y_increase
                        err_1 -= dz1
                    if err_2 > 0:
                        et[0] += x_increase
                        err_2 -= dz1
                    err_1 += dy1
                    err_2 += dx1
                    et[2] += z_increase

            hash_id = self.get_hash(et)
            ti.atomic_add(self.cell_start_idx_edges_jacobi[hash_id], 1)

    @ti.kernel
    def fill_cells_edges_jacobi(self):
        for e in range(self.mesh.num_edges):
            e0, e1 = self.mesh.eid_field[e, 0], self.mesh.eid_field[e, 1]
            x_e0, x_e1 = self.mesh.x[e0], self.mesh.x[e1]
            e0_grid_coord, e1_grid_coord = self.coord_to_grid(x_e0), self.coord_to_grid(x_e1)

            et = e0_grid_coord
            x0, y0, z0 = e0_grid_coord[0], e0_grid_coord[1], e0_grid_coord[2]
            x1, y1, z1 = e1_grid_coord[0], e1_grid_coord[1], e1_grid_coord[2]
            dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
            x_increase, y_increase, z_increase = (1 if dx > 0 else -1), (1 if dy > 0 else -1), (1 if dz > 0 else -1)
            l, m, n = ti.abs(dx), ti.abs(dy), ti.abs(dz)
            dx1, dy1, dz1 = 2*l, 2*m, 2*n

            if (l >= m) and (l >= n):
                err_1, err_2 = dy1 - l, dz1 - l
                for i in range(l):
                    hash_id = self.get_hash(et)
                    id_old = ti.atomic_sub(self.cell_start_idx_edges_jacobi[hash_id], 1)
                    self.cell_entries_edges_jacobi[id_old - 1] = e
                    if err_1 > 0:
                        et[1] += y_increase
                        err_1 -= dx1
                    if err_2 > 0:
                        et[2] += z_increase
                        err_2 -= dx1
                    err_1 += dy1
                    err_2 += dz1
                    et[0] += x_increase

            elif (m >= l) and (m >= n):
                err_1, err_2 = dx1 - m, dz1 - m
                for i in range(m):
                    hash_id = self.get_hash(et)
                    id_old = ti.atomic_sub(self.cell_start_idx_edges_jacobi[hash_id], 1)
                    self.cell_entries_edges_jacobi[id_old - 1] = e
                    if err_1 > 0:
                        et[0] += x_increase
                        err_1 -= dy1
                    if err_2 > 0:
                        et[2] += z_increase
                        err_2 -= dy1
                    err_1 += dx1
                    err_2 += dz1
                    et[1] += y_increase

            else:
                err_1, err_2 = dy1 - n, dx1 - n
                for i in range(n):
                    hash_id = self.get_hash(et)
                    id_old = ti.atomic_sub(self.cell_start_idx_edges_jacobi[hash_id], 1)
                    self.cell_entries_edges_jacobi[id_old - 1] = e
                    if err_1 > 0:
                        et[1] += y_increase
                        err_1 -= dz1
                    if err_2 > 0:
                        et[0] += x_increase
                        err_2 -= dz1
                    err_1 += dy1
                    err_2 += dx1
                    et[2] += z_increase

            hash_id = self.get_hash(et)
            id_old = ti.atomic_sub(self.cell_start_idx_edges_jacobi[hash_id], 1)
            self.cell_entries_edges_jacobi[id_old - 1] = e

    @ti.kernel
    def find_constraints_edge_edge_jacobi(self):
        for e in range(self.mesh.num_edges):
            e0, e1 = self.mesh.eid_field[e, 0], self.mesh.eid_field[e, 1]
            x_e0, x_e1 = self.mesh.x[e0], self.mesh.x[e1]
            e0_grid_coord, e1_grid_coord = self.coord_to_grid(x_e0), self.coord_to_grid(x_e1)

            et = e0_grid_coord
            x0, y0, z0 = e0_grid_coord[0], e0_grid_coord[1], e0_grid_coord[2]
            x1, y1, z1 = e1_grid_coord[0], e1_grid_coord[1], e1_grid_coord[2]
            dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
            x_increase, y_increase, z_increase = (1 if dx > 0 else -1), (1 if dy > 0 else -1), (1 if dz > 0 else -1)
            l, m, n = ti.abs(dx), ti.abs(dy), ti.abs(dz)
            dx1, dy1, dz1 = 2 * l, 2 * m, 2 * n

            if (l >= m) and (l >= n):
                err_1, err_2 = dy1 - l, dz1 - l
                for i in range(l):
                    hash_id = self.get_hash(et)
                    self.attempt_edge_edge_jacobi(hash_id, e, e0, e1, x_e0, x_e1)
                    if err_1 > 0:
                        et[1] += y_increase
                        err_1 -= dx1
                    if err_2 > 0:
                        et[2] += z_increase
                        err_2 -= dx1
                    err_1 += dy1
                    err_2 += dz1
                    et[0] += x_increase

            elif (m >= l) and (m >= n):
                err_1, err_2 = dx1 - m, dz1 - m
                for i in range(m):
                    hash_id = self.get_hash(et)
                    self.attempt_edge_edge_jacobi(hash_id, e, e0, e1, x_e0, x_e1)
                    if err_1 > 0:
                        et[0] += x_increase
                        err_1 -= dy1
                    if err_2 > 0:
                        et[2] += z_increase
                        err_2 -= dy1
                    err_1 += dx1
                    err_2 += dz1
                    et[1] += y_increase

            else:
                err_1, err_2 = dy1 - n, dx1 - n
                for i in range(n):
                    hash_id = self.get_hash(et)
                    self.attempt_edge_edge_jacobi(hash_id, e, e0, e1, x_e0, x_e1)
                    if err_1 > 0:
                        et[1] += y_increase
                        err_1 -= dz1
                    if err_2 > 0:
                        et[0] += x_increase
                        err_2 -= dz1
                    err_1 += dy1
                    err_2 += dx1
                    et[2] += z_increase

            hash_id = self.get_hash(et)
            self.attempt_edge_edge_jacobi(hash_id, e, e0, e1, x_e0, x_e1)

    @ti.func
    def attempt_edge_edge_jacobi(self, hash_id, eid_0, a0, a1, x_a0, x_a1):
        start, end = self.cell_start_idx_edges_jacobi[hash_id], self.cell_start_idx_edges_jacobi[hash_id + 1]
        for k in range(start, end):
            j = self.cell_entries_edges_jacobi[k]

            if eid_0 < j:
                b0, b1 = self.mesh.eid_field[eid_0, 0], self.mesh.eid_field[eid_0, 1]
                x_b0, x_b1 = self.mesh.x[b0], self.mesh.x[b1]

                if (a0 != b0 and a0 != b1 and a1 != b0 and a1 != b1 and
                    self.edge_edge_ccd_broadphase(x_a0, x_a1, x_b0, x_b1, self.dHat)):
                    t_ee, sc, tc = self.dist3D_edge_edge(x_a0, x_a1, x_b0, x_b1)
                    dist = t_ee.norm()
                    if self.epsilon < ti.abs(dist) and dist < self.dHat:
                        cord = ti.Vector([sc - 1.0, -sc, 1.0 - tc, tc], ti.f32)
                        ids = ti.Vector([a0, a1, b0, b1], ti.i32)
                        hash_index = self.two_int_to_hash(eid_0, j)
                        self.cid_jacobi[1, hash_index] = self.pair_jacobi(ids, dist, cord, t_ee)

    ####################################################################################################################
    # Utility function

    @ti.func
    def coord_to_grid(self, coord: ti.template()) -> ti.template():
        x = ti.floor(self.inv_grid_size * coord)
        x_int = ti.cast(x, ti.int32)
        return x_int

    @ti.func
    def coord_to_hash(self, x: ti.template()) -> ti.int32:
        h = ((x[0] + 92837111) * 92837111) ^ ((x[1] + 689287499) * 689287499) ^ ((x[2] + 283923481) * 283923481)
        return ti.abs(h) % self.table_size

    @ti.func
    def two_int_to_hash(self, x: ti.template(), y: ti.template()) -> ti.int32:
        h = ((x * 92837111) + 92837111) ^ ((y * 689287499) + 689287499)
        return ti.abs(h) % self.max_constraints

    @ti.func
    def get_hash(self, coord: ti.template()) -> ti.int32:
        return self.coord_to_hash(self.coord_to_grid(coord))

    @ti.func
    def vert_face_ccd_broadphase(self, p0, t0, t1, t2, dHat):
        min_t = ti.min(ti.min(t0, t1), t2)
        max_t = ti.max(ti.max(t0, t1), t2)
        return (min_t - dHat < p0).all() and (p0 < max_t + dHat).all()

    @ti.func
    def edge_edge_ccd_broadphase(self, a0, a1, b0, b1, dHat):
        max_a, min_a = ti.max(a0, a1), ti.min(a0, a1)
        max_b, min_b = ti.max(b0, b1), ti.min(b0, b1)
        return (min_a < max_b + dHat).all() and (min_b - dHat < max_a).all()

    @ti.func
    def dist3D_vert_face(self, P, V0, V1, V2):
        cord0, cord1, cord2 = 0.0, 0.0, 0.0
        u, v = V1 - V0, V2 - V0
        n = u.cross(v)
        s_p = (n.dot(P - V0)) / (n.dot(n))
        P0 = P - s_p * n
        w = P0 - V0
        n_cross_v, n_cross_u = n.cross(v), n.cross(u)
        s, t = w.dot(n_cross_v) / (u.dot(n_cross_v)), w.dot(n_cross_u) / (v.dot(n_cross_u))

        if s >= 0.0 and t >= 0.0:
            if s + t <= 1.0:
                cord0, cord1, cord2 = 1.0 - s - t, s, t
            else:
                q = V2 - V1
                k = (P - V1).dot(q) / (q.dot(q))
                if k > 1.0:
                    cord2 = 1.0
                elif k < 0.0:
                    cord1 = 1.0
                else:
                    cord1 = 1.0 - k
                    cord2 = k

        elif s >= 0.0 and t < 0.0:
            k = w.dot(u) / (u.dot(u))
            if k > 1.0:
                cord1 = 1.0
            elif k < 0.0:
                cord0 = 1.0
            else:
                cord0 = 1.0 - k
                cord1 = k

        elif s < 0.0 and t >= 0.0:
            k = w.dot(v) / (v.dot(v))
            if k > 1.0:
                cord2 = 1.0
            elif k < 0.0:
                cord0 = 1.0
            else:
                cord0 = 1.0 - k
                cord2 = k

        else:  # s < 0 and t < 0
            cord0 = 1.0

        return cord0, cord1, cord2

    @ti.func
    def dist3D_edge_edge(self, A0, A1, B0, B1):
        u, v, w = A1 - A0, B1 - B0, A0 - B0
        a, b, c, d, e = u.norm_sqr(), u.dot(v), v.norm_sqr(), u.dot(w), v.dot(w)

        D = a * c - b * b
        sc, sN, sD = D, D, D
        tc, tN, tD = D, D, D

        if D < 1e-7:
            sN, sD = 0.0, 1.0
            tN, tD = e, c
        else:
            sN = b * e - c * d
            tN = a * e - b * d
            if sN < 0.0:
                sN, tN, tD = 0.0, e, c
            elif sN > sD:
                sN, tN, tD = sD, e + b, c

        if tN < 0.0:
            tN = 0.0
            if -d < 0.0:
                sN = 0.0
            elif -d > a:
                sN = sD
            else:
                sN, sD = -d, a
        elif tN > tD:
            tN = tD
            if -d + b < 0.0:
                sN = 0.0
            elif -d + b > a:
                sN = sD
            else:
                sN, sD = -d + b, a

        if ti.abs(sN) < 1e-7:
            sc = 0.0
        else:
            sc = sN / sD

        if ti.abs(tN) < 1e-7:
            tc = 0.0
        else:
            tc = tN / tD

        dP = - w - (sc * u) + (tc * v)  # Qc - Pc
        return dP, sc, tc

ti.init(arch=ti.gpu)
collision = CollisionDetection(TriMesh(
    model_dir = "D:\Sehyeon\Projects\Research\starlab_physics\models\OBJ",
    model_name_list=[
                     "square.obj"
                    ],
    trans_list=[
                (0.0, 1.0, 0.0)
               ],
    scale_list=[
                2.0,
               ],
    rot_list=[
              (1.0, 0.0, 0.0, 0.0)
            ], # (axis.x, axis.y, axis.z, radian)
    is_static=False))