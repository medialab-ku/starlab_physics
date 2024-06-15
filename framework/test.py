import taichi as ti

ti.init(arch=ti.cuda)  # or ti.cuda

# Define a Taichi field to store results
@ti.kernel
def test() -> ti.uint32:

    a = ti.cast(0, ti.int32)

    return ti.math.clz(a)


print(test())