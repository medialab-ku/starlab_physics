import taichi as ti
@ti.data_oriented
class SpatialHash:
    def __init__(self, grid_resolution=(64, 64, 64)):

        self.grid_resolution = grid_resolution
        self.num_particles_in_cell = ti.field(int)
        self.particle_ids_in_cell = ti.field(int)

        self.cell_cache_size = 50
        grid_node = ti.root.dense(ti.ijk, (4, 4, 4)).dense(ti.ijk, (4, 4, 4)).dense(ti.ijk, (4, 4, 4))
        grid_node.place(self.num_particles_in_cell)
        grid_node.dense(ti.l,  self.cell_cache_size).place(self.particle_ids_in_cell)
        # self.cell_size = 1.0
        # self.num_particle_neighbors = ti.field(int)
        # self.nb_cache_size = 50
        # self.particle_neighbor_ids = ti.field(int)

        self.bbox_vertices = ti.Vector.field(n=3, dtype=float, shape=8)
        self.bbox_edge_indices_flattened = ti.field(dtype=int, shape=24)

        self.bbox_min = -ti.math.vec3(5.0)
        self.bbox_max = ti.math.vec3(5.0)

        self.cell_size = (self.bbox_max - self.bbox_min)[0] / self.grid_resolution[0]
        print(self.cell_size)
        self.init_bbox(self.bbox_min, self.bbox_max)

    @ti.kernel
    def init_bbox(self, bd_min: ti.math.vec3, bd_max: ti.math.vec3):

        self.bbox_vertices[0] = ti.math.vec3(bd_max[0], bd_max[1], bd_max[2])
        self.bbox_vertices[1] = ti.math.vec3(bd_min[0], bd_max[1], bd_max[2])
        self.bbox_vertices[2] = ti.math.vec3(bd_min[0], bd_max[1], bd_min[2])
        self.bbox_vertices[3] = ti.math.vec3(bd_max[0], bd_max[1], bd_min[2])

        self.bbox_vertices[4] = ti.math.vec3(bd_max[0], bd_min[1], bd_max[2])
        self.bbox_vertices[5] = ti.math.vec3(bd_min[0], bd_min[1], bd_max[2])
        self.bbox_vertices[6] = ti.math.vec3(bd_min[0], bd_min[1], bd_min[2])
        self.bbox_vertices[7] = ti.math.vec3(bd_max[0], bd_min[1], bd_min[2])

        self.bbox_edge_indices_flattened[0] = 0
        self.bbox_edge_indices_flattened[1] = 1
        self.bbox_edge_indices_flattened[2] = 1
        self.bbox_edge_indices_flattened[3] = 2
        self.bbox_edge_indices_flattened[4] = 2
        self.bbox_edge_indices_flattened[5] = 3
        self.bbox_edge_indices_flattened[6] = 3
        self.bbox_edge_indices_flattened[7] = 0
        self.bbox_edge_indices_flattened[8] = 4
        self.bbox_edge_indices_flattened[9] = 5
        self.bbox_edge_indices_flattened[10] = 5
        self.bbox_edge_indices_flattened[11] = 6
        self.bbox_edge_indices_flattened[12] = 6
        self.bbox_edge_indices_flattened[13] = 7
        self.bbox_edge_indices_flattened[14] = 7
        self.bbox_edge_indices_flattened[15] = 4
        self.bbox_edge_indices_flattened[16] = 0
        self.bbox_edge_indices_flattened[17] = 4
        self.bbox_edge_indices_flattened[18] = 1
        self.bbox_edge_indices_flattened[19] = 5
        self.bbox_edge_indices_flattened[20] = 2
        self.bbox_edge_indices_flattened[21] = 6
        self.bbox_edge_indices_flattened[22] = 3
        self.bbox_edge_indices_flattened[23] = 7

    @ti.func
    def pos_to_cell_id(self, y: ti.math.vec3) -> ti.math.ivec3:
        test = (y - self.bbox_min) / self.cell_size
        return ti.cast(test, int)

    @ti.func
    def is_in_grid(self, c):
        # @c: Vector(i32)

        is_in_grid = True
        for i in ti.static(range(3)):
            is_in_grid = is_in_grid and (0 <= c[i] < self.grid_resolution[i])

        return is_in_grid


    @ti.kernel
    def insert_particles_in_grid(self, x: ti.template()):
        self.num_particles_in_cell.fill(0)
        for pi in x:
            cell_id = self.pos_to_cell_id(x[pi])
            counter = ti.atomic_add(self.num_particles_in_cell[cell_id], 1)
            if counter < self.cell_cache_size:
                self.particle_ids_in_cell[cell_id, counter] = pi

    @ti.kernel
    def search_neighbours(self, x: ti.template(), num_neighbours: ti.template(), neighbour_ids: ti.template(), cache_size: int):

        num_neighbours.fill(0)
        for pi in x:
            xi = x[pi]
            cell_id = self.pos_to_cell_id(xi)
            for offs in ti.static(ti.grouped(ti.ndrange((-1, 2), (-1, 2), (-1, 2)))):
                cell_to_check = cell_id + offs
                if self.is_in_grid(cell_to_check):
                    for j in range(self.num_particles_in_cell[cell_to_check]):
                        pj = self.particle_ids_in_cell[cell_to_check, j]
                        if pi == pj:
                            continue

                        xj = x[pi]
                        n = num_neighbours[pi]
                        if n < cache_size:
                            neighbour_ids[pi, n] = pj
                            num_neighbours[pi] += 1
