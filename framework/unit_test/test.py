# edges = [[1, 2], [0, 2], [0, 1, 3, 4], [2, 4], [2, 3]]
# edges[0].remove(1)
# print(len(edges[0]))

def Hierholzer(adjacency_list):
    first = 0
    stack = []
    stack.append(first)
    ret = []
    while len(stack) > 0:
        v = stack[-1]
        if len(adjacency_list[v]) == 0:
            ret.append(v)
            stack.pop()
        else:
            i = adjacency_list[v][- 1]
            adjacency_list[v].pop()
            # if v in edges[v]:
            adjacency_list[i].remove(v)
            stack.append(i)

    return ret


def compute_adjacency_list(num_verts, edges):
    adjacency_list = [[] for _ in range(num_verts)]

    for i in range(len(edges) // 2):
       id0 = edges[2 * i + 0]
       id1 = edges[2 * i + 1]
       adjacency_list[id0].append(id1)
       adjacency_list[id1].append(id0)

    return adjacency_list

edges = [0, 1, 1, 2, 2, 0]
adjacency_list = compute_adjacency_list(3, edges)
# print(Hierholzer(adjacency_list))

print(int('11111111', 2))
mc_i = int('00101101101101101101101101101101', 2)

morton_codes = []
morton_codes.append(int('00101101101101101101101101101101', 2))
morton_codes.append(int('00001001001001001001001001001001', 2))
morton_codes.append(int('00111111111111111111111111111111', 2))
morton_codes.append(int('00010010010010010010010010010010', 2))
pass_num = 0
BITS_PER_PASS = 6
RADIX = pow(2, BITS_PER_PASS)
pass_num = (30 + BITS_PER_PASS - 1) // BITS_PER_PASS
for i in range(pass_num):
    digit = (mc_i >> (i * BITS_PER_PASS)) & (RADIX - 1)
    print(bin(digit))