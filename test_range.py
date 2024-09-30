import taichi as ti

ti.init(arch=ti.gpu)

@ti.kernel
def test_range():

    ti.loop_config(serialize=True)
    for i in range(5, 10):
        print(i)


test_range()