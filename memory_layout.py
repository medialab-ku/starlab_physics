import taichi as ti

ti.init(arch=ti.gpu)

x = ti.field(int)
# x_snode = ti.root.dense(ti.ijk, (3, 3, 3)).place(x)
# x_snode = ti.root.dense(ti.i, 8).place(x)
x_snode = ti.root.pointer(ti.ijk, (4, 4, 4)).dense(ti.ijk, (2, 2, 2)).place(x)

@ti.kernel
def foo():
    print("foo")
    for I in ti.grouped(x):
        print(I)

x[0, 0, 0] = 1
foo()

ti.deactivate_all_snodes()
foo()

x[2, 0, 0] = 1
foo()
# print(x)