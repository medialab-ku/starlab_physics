# XPBD Cloth Simulation

# Environment

![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-brightgreen)
![Python](https://img.shields.io/badge/Python-3.11-blue)

# Install

    pip install -r requirements.txt

# Current Features

* XPBD
  * Constraints
    * Spring Constraints
    * Bending Constraints 
    * Non-penetration Constraints 
      
  * Solver
    * Jacobi
    * **[WIP]** Parallel Gauss-Seidel


* Collision detection
  * Broad phase
    * Parallel AABB BVH Construction/Traversal
    * Spatial Hashing 
  * Narrow phase
    * Vertex-Triangle
    * **[WIP]** Edge-Edge

* Utilities
  * GUI for setting XPBD parameters in runtime
  * Vertex selection (for setting fixed)
  * Exporting simulated meshes

# run demo

    python main.py

![demo_short](https://github.com/user-attachments/assets/eb8bf322-3b65-4c87-bbc3-d96940c1853a)


# Frequently used keys
* **'r'** : reset simulation
* **' '** : run/stop simulation

# Changing the simulation mesh

The mesh data(i.e., *.obj) are in the **"starlab_physics/models/OBJ"** folder.

In **"starlab_physics/Scenes/concat_test.py"** file, 

    model_dir = str(model_path / "OBJ")
    model_names = []
    trans_list = []
    scale_list = []
    
    ...

    offsets = concat_mesh(concat_model_name, model_dir, model_names, trans_list, scale_list)

fill in **...** with your mesh data, as follows:

    model_names.append("your-obj-name.obj")
    trans_list.append([x, y, z])
    scale_list.append(size)

# References
1. Lauterbach, Christian, et al. "Fast BVH construction on GPUs." Computer Graphics Forum. Vol. 28. No. 2. Oxford, UK: Blackwell Publishing Ltd, 2009.
2. Macklin, Miles, Matthias Müller, and Nuttapong Chentanez. "XPBD: position-based simulation of compliant constrained dynamics." Proceedings of the 9th International Conference on Motion in Games. 2016.
3Lauterbach, Christian, et al. "Fast BVH construction on GPUs." Computer Graphics Forum. Vol. 28. No. 2. Oxford, UK: Blackwell Publishing Ltd, 2009.