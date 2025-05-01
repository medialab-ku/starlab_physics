import taichi as ti

from CCDModule import CCDModule

if __name__ == '__main__':
    ti.init(arch=ti.cpu)
    ccd = CCDModule()

    ####################################
    # ToDo initial setting
    # one triangle
    N = 1
    triangles = ti.Vector.field(3, dtype=ti.f64, shape=(N, 3))
    tri_moves = ti.Vector.field(3, dtype=ti.f64, shape=(N, 3))

    # initial point
    p = ti.Vector([0.0, 0.0, 1.0], dt=ti.f64)
    t0 = ti.Vector([0.0, -1.0, 0.0], dt=ti.f64)
    t1 = ti.Vector([1.0, 1.0, 1.0], dt=ti.f64)
    t2 = ti.Vector([-1.0, 1.0, 0.0], dt=ti.f64)

    # movement (not moving)
    dp = ti.Vector([0.0, 3.0, -2.0], dt=ti.f64)
    dt0 = ti.Vector([0.0, 3.0, 0.0], dt=ti.f64)
    dt1 = ti.Vector([0.0, 3.0, -1.0], dt=ti.f64)
    dt2 = ti.Vector([0.0, 3.0, 0.0], dt=ti.f64)

    # array initialization
    triangles[0, 0] = t0; triangles[0, 1] = t1; triangles[0, 2] = t2
    tri_moves[0, 0] = dt0; tri_moves[0, 1] = dt1; tri_moves[0, 2] = dt2

    # CCD parameter
    eta = 0.01
    thickness = 0.01
    ####################################

    toc = ccd.point_mesh_ccd(p, dp, triangles, tri_moves, eta, thickness)
    print("Time of Contact (toc):", toc)