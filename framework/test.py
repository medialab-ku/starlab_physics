import taichi as ti

ti.init(arch=ti.cuda, device_memory_GB=3)

# Node = ti.types.struct(parent=ti.i32, child_a=ti.i32, child_b=ti.i32)

@ti.dataclass
class Node:
    object_id: ti.i32
    parent:  ti.i32
    # child_a:  ti.i32
    # child_b:  ti.i32
    # visited: ti.i32
    # aabb_min: ti.math.vec3
    # aabb_max: ti.math.vec3


num_nodes = 3
nodes = Node.field(shape=num_nodes)
codes = ti.field(dtype=ti.uint32, shape=num_nodes)
@ti.kernel
def init():

    for i in range(3):
        nodes[i].object_id = i
        nodes[i].parent = i
        codes[i] = num_nodes - i

init()

print(nodes)
ti.algorithms.parallel_sort(keys=codes, values=nodes)
print(nodes)