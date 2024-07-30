# XPBD Cloth Simulation

# Environment

![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-brightgreen)
![Python](https://img.shields.io/badge/Python-3.11-blue)

# Install

    pip install -r requirements.txt

# run demo

    python main.py

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
1.
2. Macklin, Miles, Matthias Müller, and Nuttapong Chentanez. "XPBD: position-based simulation of compliant constrained dynamics." Proceedings of the 9th International Conference on Motion in Games. 2016.
3Lauterbach, Christian, et al. "Fast BVH construction on GPUs." Computer Graphics Forum. Vol. 28. No. 2. Oxford, UK: Blackwell Publishing Ltd, 2009.