import taichi as ti

ti.init(arch=ti.cuda)

S = ti.root.dense(ti.i, 10).dynamic(ti.j, 1024, chunk_size=32)
x = ti.field(int)
S.place(x)


@ti.kernel
def add_data():
    for i in range(10):
        for j in range(i):
            x[i].append(j)
            print(i, x[i, j])  # will print i

    for i in range(10):
        x[i].deactivate()
        print(x[i].length())  # will print 0


add_data()
