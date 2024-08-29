import numpy as np
import meshio
from taichi.examples.simulation.laplace_equation import points


def create_particle_block(size_x, size_y, size_z, radius):
    x = np.empty((size_x * size_y * size_z, 3))
    x_id = np.empty((size_x * size_y * size_z, 1))
    trans_lf = lambda x, trans: x + trans
    off = 1.1 * radius
    for i in range(size_x):
        for j in range(size_y):
            for k in range(size_z):
                id = i * size_y * size_z + j * size_z + k
                x[id] = [i * off, j * off, k * off]
                x_id[id] = [id]

    center = x.sum(axis=0) / x.shape[0]  # center position of the mesh
    x = np.apply_along_axis(lambda row: trans_lf(row, -center), 1, x)  # translate to origin
    x = x.astype(np.float32)
    x_id = x_id.astype(np.int32)

    return x, x_id

x, x_id = create_particle_block(80, 2, 80, 0.3)
# print(x_id)
cell_block = meshio.CellBlock(cell_type="vertex", data=x_id, tags=[])
mesh = meshio.Mesh(points=x, cells=[cell_block])
# print(mesh.cells)
meshio.write("../../models/VTK/very_thin_sheet.vtk", mesh)
# meshio.read("particle_block.vtk")