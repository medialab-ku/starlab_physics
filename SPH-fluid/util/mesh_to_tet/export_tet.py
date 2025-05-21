import trimesh
import tetgen
import os


def generate_tet_mesh_from_obj(obj_path):
    mesh = trimesh.load(obj_path)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError("Only single Trimesh objects supported.")

    vertices = mesh.vertices
    faces = mesh.faces

    tgen = tetgen.TetGen(vertices, faces)

    nodes, tets = tgen.tetrahedralize(order=1, mindihedral=5, minratio=1.5)

    return nodes, tets

def save_as_medit_mesh(filename, nodes, tets):
    with open(filename, 'w') as f:
        f.write("MeshVersionFormatted 1\n")
        f.write("Dimension 3\n\n")

        f.write("Vertices\n")
        f.write(f"{len(nodes)}\n")
        for v in nodes:
            f.write(f"{v[0]} {v[1]} {v[2]} 1\n")

        f.write("\nTetrahedra\n")
        f.write(f"{len(tets)}\n")
        for tet in tets:
            f.write(f"{tet[0] + 1} {tet[1] + 1} {tet[2] + 1} {tet[3] + 1} 1\n")  # 1-based index

        f.write("\nEnd\n")


def save_as_gmsh_msh(filename, nodes, tets):
    with open(filename, 'w') as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")

        f.write("$Nodes\n")
        f.write(f"{len(nodes)}\n")
        for i, v in enumerate(nodes):
            f.write(f"{i + 1} {v[0]} {v[1]} {v[2]}\n")
        f.write("$EndNodes\n")

        f.write("$Elements\n")
        f.write(f"{len(tets)}\n")
        for i, tet in enumerate(tets):
            f.write(f"{i + 1} 4 2 0 1 {tet[0] + 1} {tet[1] + 1} {tet[2] + 1} {tet[3] + 1}\n")
        f.write("$EndElements\n")


if __name__ == "__main__":
    obj_path = "solidify.obj"
    out_prefix = os.path.splitext(os.path.basename(obj_path))[0]

    nodes, tets = generate_tet_mesh_from_obj(obj_path)

    save_as_medit_mesh(f"{out_prefix}.mesh", nodes, tets)
    print(f"Saved Medit .mesh to {out_prefix}.mesh")

    save_as_gmsh_msh(f"{out_prefix}.msh", nodes, tets)
    print(f"Saved Gmsh .msh to {out_prefix}.msh")