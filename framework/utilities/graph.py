from collections import deque
import numpy as np
def construct_graph(num_verts, edges):
    adjacency_list = [[] for _ in range(num_verts)]

    for i in range(len(edges)):
       id0 = edges[i][0]
       id1 = edges[i][1]
       adjacency_list[id0].append(id1)
       adjacency_list[id1].append(id0)

    return adjacency_list

# def Dijkstra(graph, start):
def bfs_shortest_path(graph, start, end):
    # Keep track of explored nodes
    explored = []
    # Keep track of all the paths to be checked
    queue = deque([[start]])

    # Return path if start is goal
    if start == end:
        return [start]

    # Loop to traverse the graph with the help of the queue
    while queue:
        # Pop the first path from the queue
        path = queue.popleft()
        # Get the last node from the path
        node = path[-1]
        if node not in explored:
            neighbours = graph[node]
            # Go through all neighbour nodes, construct a new path and
            # push it into the queue
            for neighbour in neighbours:
                new_path = list(path)
                new_path.append(neighbour)
                queue.append(new_path)
                # Return path if neighbour is goal
                if neighbour == end:
                    return new_path
            # Mark node as explored
            explored.append(node)

    # In case there's no path between the two nodes
    return "Path not found"

def floyd_warshall(graph):
    # Number of vertices in the graph
    n = len(graph)

    # Create a distance matrix and initialize it with infinity
    dist = np.full((n, n), np.inf)

    # Distance from a node to itself is zero
    for i in range(n):
        dist[i][i] = 0

    # Initialize the distance matrix with edge weights
    for u in range(n):
        for v in graph[u]:
            dist[u][v] = 1  # Since the graph is unweighted, the distance is 1

    # Floyd-Warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    dist = dist.astype(int)

    return dist

def eulerization(graph):

    # print(graph)
    while True:
        dist = floyd_warshall(graph)
        odd_degree_nodes = []
        num_nodes = len(graph)
        for i in range(num_nodes):

            if len(graph[i]) % 2 == 1:
                odd_degree_nodes.append(i)

        # print(odd_degree_nodes)
        if len(odd_degree_nodes) == 0:
            break

        dist_min = 1e9
        pair = [-1, -1]
        for i in range(0, len(odd_degree_nodes) - 1):
            i_odd = odd_degree_nodes[i]
            for j in range(i + 1, len(odd_degree_nodes)):
                j_odd = odd_degree_nodes[j]
                dist_ij = dist[i_odd][j_odd]
                if dist_ij < dist_min:
                    dist_min = dist_ij
                    pair = [i_odd, j_odd]

        path = bfs_shortest_path(graph, pair[0], pair[1])

        for i in range(0, len(path) - 1):
            node_i, node_j = path[i], path[i + 1]
            graph[node_i].append(node_j)
            graph[node_j].append(node_i)

    return graph

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
