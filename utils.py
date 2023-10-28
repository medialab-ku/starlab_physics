import taichi as ti


ti.init(arch=ti.gpu)

S = ti.root.dynamic(ti.i, 1024, chunk_size=32)
mat9x9 = ti.types.matrix(n=9, m=9, dtype=ti.f32)
x = mat9x9.field()
S.place(x)

@ti.func
def test2(i: ti.f32):
    mat = ti.Matrix([[i, 0, 0, 0, 0, 0, 0, 0, 0],
                     [0, i, 0, 0, 0, 0, 0, 0, 0],
                     [0, 0, i, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, i, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, i, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, i, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, i, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, i, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0, i]])

    return mat


# ti.linalg.taichi_cg_solver()

@ti.kernel
def test():
   center = ti.math.vec3(0.5, 0.5, 0.5)

   vec = ti.math.vec3(0.5, 0.8, 0.5)

   rad = 0.4

   normal = (vec - center).normalized()

   p = center + rad * normal

   print(p)

test()