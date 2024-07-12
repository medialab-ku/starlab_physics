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
print(Hierholzer(adjacency_list))