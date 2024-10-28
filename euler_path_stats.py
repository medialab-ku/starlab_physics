from pathlib import Path
import networkx as nx
import os
from collections import Counter
import matplotlib.pyplot as plt
from scipy.special import euler

model_path = Path(__file__).resolve().parent / "models"
model_dir = str(model_path) + "/OBJ"

print(model_path)

def euler_path(model_name):
    euler_dir = model_dir[:-len("models/OBJ")] + "euler_graph"
    print(euler_dir)
    precomputed_graph_file = euler_dir + "/" + model_name[:-len(".obj")] + ".edgelist"
    if not os.path.exists(euler_dir):
        print("The ""euler_graph"" dictionary does not exist. It will be made and then located in your path...")
        os.mkdir(dir)


    print("Importing a precomputed Euler graph...")
    euler_graph = nx.read_edgelist(precomputed_graph_file, create_using=nx.MultiGraph)
    print("Checking integrity... ", end='')
    if nx.is_eulerian(euler_graph):

        print("The imported graph is Eulerian!\n")

        # print(euler_graph)
        euler_path = list(nx.eulerian_path(euler_graph))
        # print(euler_path)
        # print(euler_path)
        edge_count = {}
        duplicate = list()
        for u, v in euler_path:
            if (u, v) in edge_count:
                edge_count[(u, v)] += 1
                duplicate.append((u, v))
            elif (v, u) in edge_count:  # Ensure undirected pairs are counted correctly
                edge_count[(v, u)] += 1
                duplicate.append((v, u))
            else:
                edge_count[(u, v)] = 1

        count_of_duplicates = Counter(edge_count.values())
        print(count_of_duplicates)


        for u, v in duplicate:
            euler_path.remove((u, v))

        # print(euler_path)

        partition = [[]]

        pid = 0
        for u, v in euler_path:
            if (u, v) in edge_count:
                if edge_count[(u, v)] == 2:
                    partition.append([])
                    pid += 1
                partition[pid].append((v, u))
            elif (v, u) in edge_count:  # Ensure undirected pairs are counted correctly
                if edge_count[(v, u)] == 2:
                    partition.append([])
                    pid += 1
                partition[pid].append((v, u))

        values = []
        total = 0
        for i in range(len(partition)):
            values.append(len(partition[i]))
            total += len(partition[i])

        count = Counter(values)
        print(partition)
        # print(total)

        # print(len(partition))
        # graph = nx.Graph(euler_graph)
        # print(len(graph.edges))

        # euler_path = list(nx.eulerian_path(euler_graph))
        # print(len(euler_path))

        # x = [1, 2, 2, 3, 3, 4, 5, 5, 6]
        # y = [2, 3, 3, 4, 4, 5, 5, 6, 7]

        # Create separate histograms for x and y
        plt.hist(values, bins=5, alpha=0.5, label='x', edgecolor='black')
        # plt.hist(y, bins=5, alpha=0.5, label='y', edgecolor='black')
        plt.title("Separate Histograms for x and y")
        plt.show()


euler_path("square.obj")


