import numpy as np
import pickle as pkl

name = input("file name?\n")
load_data = np.load(f'/home/media/workspace/vton_tkhmr/VTON/UI/cloth_rigging/cloth_models/long_shirt/{name}.npy', allow_pickle=True)
count=0
for i in load_data:
    print(i)
print(len(load_data))

