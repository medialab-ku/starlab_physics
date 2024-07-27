import taichi as ti
enable_profiler = False
ti.init(arch=ti.cuda, device_memory_GB=6, kernel_profiler=enable_profiler)

num_particles = 10
pos2d = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)
res2d = ti.math.ivec2(2, 2)
cell_size = ti.math.vec2(1.0, 1.0)
prefix_sum_exec = ti.algorithms.PrefixSumExecutor(res2d[0] * res2d[1])
prefix_sum = ti.field(dtype=ti.i32, shape=res2d[0] * res2d[1])
prefix_sum_temp = ti.field(dtype=ti.i32, shape=res2d[0] * res2d[1])
cell_id = ti.field(dtype=ti.i32, shape=num_particles)
_id = ti.field(dtype=ti.i32, shape=num_particles)
sorted_id = ti.field(dtype=ti.i32, shape=num_particles)
sorted_to_origin_id = ti.field(dtype=ti.i32, shape=num_particles)
sorted_cell_id = ti.field(dtype=ti.i32, shape=num_particles)
particles_in_cell = ti.field(dtype=ti.i32, shape=res2d[0] * res2d[1])

@ti.kernel
def init_pos2d():

    pos2d[0] = ti.math.vec2([0.1, 1.0])
    pos2d[1] = ti.math.vec2([1.1, 1.5])
    pos2d[2] = ti.math.vec2([1.2, 0.5])
    pos2d[3] = ti.math.vec2([0.2, 0.3])
    pos2d[4] = ti.math.vec2([1.3, 0.2])
    pos2d[5] = ti.math.vec2([0.1, 0.1])
    pos2d[6] = ti.math.vec2([0.5, 0.9])
    pos2d[7] = ti.math.vec2([0.5, 1.9])
    pos2d[8] = ti.math.vec2([1.5, 0.9])
    pos2d[9] = ti.math.vec2([0.5, 0.8])

@ti.kernel
def get_flattened_cell_id():
    prefix_sum.fill(0)
    for i in range(num_particles):
        cell_id_2d = (pos2d[i] / cell_size).cast(ti.int32)
        flattened_cell_id = cell_id_2d[1] * res2d[1] + cell_id_2d[0]
        cell_id[i] = flattened_cell_id
        _id[i] = i
        ti.atomic_add(prefix_sum[flattened_cell_id], 1)

@ti.kernel
def counting_sort_by_cell():

    # ti.loop_config(serialize=True)
    for i in range(num_particles):
        I = num_particles - 1 - i
        cid = cell_id[I]
        idx = ti.atomic_sub(prefix_sum_temp[cid], 1) - 1
        sorted_cell_id[idx] = cid
        sorted_id[idx] = I

    for i in range(num_particles):
        sid = sorted_id[i]
        sorted_to_origin_id[sid] = i

init_pos2d()
get_flattened_cell_id()
# print(_id)
# print(cell_id)
# print(prefix_sum)
prefix_sum_exec.run(prefix_sum)
prefix_sum_temp.copy_from(prefix_sum)

print(prefix_sum_temp)
counting_sort_by_cell()
print(sorted_id)
print(sorted_cell_id)
# print(sorted_to_origin_id)


@ti.kernel
def for_each_color(color_offset: ti.int32, size: ti.i32):

    for i in range(size):
        original_edge_id = sorted_id[i + color_offset]

def foo():
    colors = 4
    for c in range(colors):
        for_each_color(c, prefix_sum_temp[c])

