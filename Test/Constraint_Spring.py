import taichi as ti

@ti.data_oriented
class SpringConstraint:
    def __init__(self, x0, edge_indices):
