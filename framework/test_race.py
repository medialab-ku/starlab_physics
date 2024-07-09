import taichi as ti

ti.init(arch=ti.cuda, kernel_profiler=False)

size = 10
test = ti.field(dtype=ti.i32, shape=size)

@ti.kernel
def foo():
    for i in range(2 * size):
        j = i // 2
        test[j] = i

foo()
print(test)


