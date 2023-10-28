import taichi as ti


ti.init(arch=ti.gpu)

S = ti.root.dynamic(ti.i, 1024, chunk_size=32)
SphereType = ti.types.struct(center=ti.math.vec3, radius=float)
x = SphereType.field()
S.place(x)

test = ti.Matrix.field

@ti.kernel
def test():

    x.append(SphereType(center=ti.math.vec3(1, 1, 1), radius=0.1))
    print(x.length())
    print(x[0].center)

test()