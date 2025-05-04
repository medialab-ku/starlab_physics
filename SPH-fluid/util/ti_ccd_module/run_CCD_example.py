import taichi as ti

from CCDModule import CCDModule

if __name__ == '__main__':
    ti.init(arch=ti.cpu)
    ccd = CCDModule()

    ####################################
    # ToDo initial setting (point-triangle)
    # one triangle
    N = 1
    triangles = ti.Vector.field(3, dtype=ti.f64, shape=(N, 3))
    tri_moves = ti.Vector.field(3, dtype=ti.f64, shape=(N, 3))

    # initial points
    p = ti.Vector([0.0, 0.0, 1.0], dt=ti.f64)
    t0 = ti.Vector([0.0, -1.0, 0.0], dt=ti.f64)
    t1 = ti.Vector([1.0, 1.0, 1.0], dt=ti.f64)
    t2 = ti.Vector([-1.0, 1.0, 0.0], dt=ti.f64)

    # movement
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
    print("point-triangle Time of Contact (toc):", toc)

    ####################################
    # ToDo: initial setting (edge-edge)
    # one edge pair
    N = 1; M = 1
    edges_a = ti.Vector.field(3, dtype=ti.f64, shape=(N, 2))
    edges_b = ti.Vector.field(3, dtype=ti.f64, shape=(M, 2))
    moves_a = ti.Vector.field(3, dtype=ti.f64, shape=(N, 2))
    moves_b = ti.Vector.field(3, dtype=ti.f64, shape=(M, 2))

    # initial edges
    ea0 = ti.Vector([0.0, -1.0, 0.0], dt=ti.f64)
    ea1 = ti.Vector([0.0, 1.0, 0.0], dt=ti.f64)
    eb0 = ti.Vector([-1.0, 0.0, -0.3], dt=ti.f64)
    eb1 = ti.Vector([1.0, 0.0, -0.7], dt=ti.f64)

    # movement
    dea0 = ti.Vector([0.0, 0.0, -1.0], dt=ti.f64)
    dea1 = ti.Vector([0.0, 0.0, -1.0], dt=ti.f64)
    deb0 = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
    deb1 = ti.Vector([-1.0, 0.0, 0.0], dt=ti.f64)

    # array initialization
    edges_a[0, 0] = ea0; edges_a[0, 1] = ea1
    edges_b[0, 0] = eb0; edges_b[0, 1] = eb1
    moves_a[0, 0] = dea0; moves_a[0, 1] = dea1
    moves_b[0, 0] = deb0; moves_b[0, 1] = deb1

    # CCD parameter
    eta = 0.01
    thickness = 0.01
    ####################################

    # CCD 실행
    toc = ccd.edges_edges_ccd(edges_a, moves_a, edges_b, moves_b, eta, thickness)
    print("edge-edge Time of Contact (toc):", toc)