import taichi as ti
# import vtk
import networkx as nx
import numpy as np
import framework.utilities.graph as graph_utils
from framework.meshio.meshTaichiWrapper import MeshTaichiWrapper
import os
from pathlib import Path
import time

def MESH(file_path):
    with open(file_path, mode="r") as file:
        lines = file.readlines()

    header = lines[0].strip().split()
    num_edges = int(header[0])

    eid_np = np.zeros((num_edges, 2), dtype=int)

    edge_count = 0
    for line in lines[1: (num_edges + 1)]:
        parts = line.strip().split()
        v1, v2 = int(parts[1]), int(parts[2])

        eid_np[edge_count][0] = v1
        eid_np[edge_count][1] = v2
        edge_count += 1

    print("The number of edges :", edge_count)

    return eid_np

def make_graph(precomputed_graph, eid_np):
    if not os.path.isfile(precomputed_graph):
        print(f"The Euler graph file '{precomputed_graph}' does not exist.")
        print(f"Constructing an Euler graph... It might take a long time...: {model[0]}")

        start = time.time()
        graph = graph_utils.construct_graph(eid_np)
        graph = nx.eulerize(graph)

        if nx.is_eulerian(graph):
            print("Euler graph construction success...")

        end = time.time()
        print("Euler Graph Elapsed time:", round(end - start, 5), "sec.")

        # print("Elapsed time:", round(end - start, 5), "sec.")
        print("Exporting the constructed Euler graph...")
        nx.write_edgelist(graph, precomputed_graph)
        print("Export complete...\n")

    else:
        print("Importing a precomputed Euler graph...")
        graph = nx.read_edgelist(precomputed_graph, create_using=nx.MultiGraph)
        # print(graph)

        print("Checking integrity...")
        if nx.is_eulerian(graph):
            print("The imported graph is Eulerian!\n")
        else:
            print("The imported graph is not Eulerian...\n")

#######################################################################################################################

enable_profiler = False
ti.init(arch=ti.cuda, device_memory_GB=8, kernel_profiler=enable_profiler)

model_path = Path(__file__).resolve().parent.parent.parent / "models"
model_directories = ["MESH", "OBJ"]

for model_type in model_directories:
    print()
    print(model_type)
    model_dir = str(model_path) + "/" + model_type + "/"
    graph_dir = model_dir[:-len("models/" + model_type + "/")] + "euler_graph/"
    print("Model directory :", model_dir)
    print("Graph directory :", graph_dir)
    print()

    models = []
    try:
        entries = list(os.scandir(model_dir))

        file_found = False
        for entry in entries:
            if entry.is_file():
                file_name, file_extension = os.path.splitext(entry.name)
                if ((model_type == "MESH" and file_extension == ".edge") or
                    (model_type == "OBJ" and file_extension == ".obj") or
                    (model_type == "VTK" and file_extension == ".vtk")):
                    models.append([file_name, file_extension])
                    file_found = True

        if not file_found:
            print(f"There are no files in the directory '{model_dir}'.")

    except FileNotFoundError:
        print(f"Error: The directory '{model_dir}' does not exist.")
    except PermissionError:
        print(f"Error: Permission denied for the directory '{model_dir}'.")

    for model in models:
        precomputed_graph = graph_dir + model[0] + ".edgelist"
        print("Precomputed Graph :", precomputed_graph)

        if model_type == "MESH":
            eid_np = MESH(model_dir + model[0] + model[1])
            make_graph(precomputed_graph, eid_np)

        elif model_type == "OBJ":
            # model[0] : file name, model[1] : file extension
            # call "is_static=True" to prevent constructing graph in the MeshTaichiWrapper object
            mesh = MeshTaichiWrapper(model_dir, model_name=(model[0] + model[1]), offsets=[0], scale=10.0, trans=ti.math.vec3(0, 0.0, 0), rot=ti.math.vec3(0.0, 0.0, 0.0), is_static=True)
            make_graph(precomputed_graph, mesh.eid_np)