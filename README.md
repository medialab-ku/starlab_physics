# VTON

![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-brightgreen)
![Python](https://img.shields.io/badge/Python-3.11-blue)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white">

# Environment
    
    Ubuntu 22.04

# How to Install Physics

## 1. Add a git submodule

    git submodule add https://github.com/medialab-ku/starlab_physics

## 2. Use a specific branch

Add the following tag:

**branch = v0.1**

in ".gitmodules" as follows:
    
    [submodule "starlab_physics"]
	path = starlab_physics
	url = https://github.com/medialab-ku/starlab_physics
	branch = v0.1  

## 3. Pull a git submodule

When you first clone the project, the submodule directory(i.e, "starlab_physics") is empty.

Therefore, you should execute the following commends to clone the submodule project.
    
    git submodule init
    git submodule update --remote


## 4. Install required packages

    cd starlab-physics
    pip install -r requirements.txt
