import numpy as np
import taichi as ti

# from framework.utilities.makeGraphFromMesh import file_name
# import torch
from framework.utilities.sampling.Particle import ClothParticle
import os


class GenerateParticle:
    def __init__(self, file_name):
        # self.MODE = MODE
        print("Start")

        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.subFolder = '/home/mhkee/Desktop/workspace/starlab_physics/models/OBJ/'
        self.fileobj = file_name + ".obj"
        self.filenpy = file_name + ".npy"
        # file_path = os.path.join(current_dir, 'cloth_models', self.subFolder , self.filenpy)

        output_path = '/home/mhkee/Desktop/workspace/starlab_physics/models/sampling/' + self.filenpy
        self.part = ClothParticle(subFolder=self.subFolder , filename=self.fileobj)

        self.part.file_stage(output_path)

        if not os.path.exists(output_path):
            print("not generated")
            print("loading")
            self.part.obj_load()
            self.part.transform(scale=10, translate=np.array([0.0, 0.0, 0.0]), PRINT=False)  # scale, translate
            self.part.viewchange(elev=90, azim=270, size=5)
            self.part.file_generate(distance=0.15, subFolder=self.subFolder , filename=self.filenpy, LOG=False, SHOW=False)
            print("file generated")

        # else:
        #     print("already generated")
        #     self.part.obj_load()
        #     if self.MODE == "MATPLOT":
        #         self.part.transform(scale=10, translate=np.array([0.0, 0.0, 0.0]), PRINT=False)
        #         self.part.viewchange(elev=45, azim=180, size=1)
        #
        #     if self.MODE == "TAICHI":
        #         self.part.transform(scale=1, translate=np.array([0.0, 0.0, 0.0]), PRINT=False)
        #         self.part.viewchange(elev=45, azim=180, size=0.003)
        #
        #     self.part.file_load(normal_distance=0.5, subFolder=self.subFolder , filename=self.filenpy, SHOW=True, MODE=self.MODE)

    def setscene(self, scene: ti.ui.Scene):
        self.scene = scene
        self.part.setscene(scene)

    def render(self):
        self.part.taichi_display(distance=0.05)

if __name__ == '__main__':
    gen = GenerateParticle(file_name="smplH-male")
