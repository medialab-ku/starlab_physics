edges = [[1, 2], [0, 2], [0, 1, 3, 4], [2, 4], [2, 3]]
# edges[0].remove(1)
# print(len(edges[0]))

def Hierholzer(edges):
    first = 0
    stack = []
    stack.append(first)
    ret = []
    while len(stack) > 0:
        print("ret: ", ret)
        v = stack[-1]
        print("v: ", v)
        print("len(edges[v]): ", len(edges[v]))
        if len(edges[v]) == 0:
            ret.append(v)
            print("ret: ", ret)
            stack.pop()
            print("stack: ", stack)
        else:
            i = edges[v][- 1]
            edges[v].pop()
            # if v in edges[v]:
            edges[i].remove(v)
            stack.append(i)

    return ret

print(Hierholzer(edges))