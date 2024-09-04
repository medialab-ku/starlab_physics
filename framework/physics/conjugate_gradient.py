import taichi as ti

@ti.data_oriented
class ConjugateGradient:

    def __init__(self):

        print("ConjugateGradient initialized...")


    @ti.kernel
    def vector_add(self, ret: ti.template(), x: ti.template(), y: ti.template(), scalar: float):
        for i in x:
            ret[i] = x[i] + scalar * y[i]

    @ti.kernel
    def dot_product(self, x: ti.template(), y: ti.template()) -> float:
        ret = 0.0
        for i in x:
            ret += x[i].dot(y[i])
        return ret