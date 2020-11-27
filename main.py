from ProjectiveDynamics import pyPD 
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import meshplot

if __name__ == '__main__':

    # unit test with a single tet element
    #e : Number of elements
    #N : Number of vertices 
    #initial position of vertices : N x 3 

    x = torch.Tensor([  [1,0,0],
                        [0,1,0],
                        [0,0,1],
                        [0,0,0],
                        [1,1,1]])
    #initial position of vertices : N x 3 
    v = torch.Tensor([  [3,0,0],
                        [3,0,0],
                        [3,0,0],
                        [3,0,0],
                        [3,0,0]])

    #selection matrix for all tetrahedrons : 4*e x N 
    S = torch.Tensor([  [1,0,0,0,0],
                        [0,1,0,0,0],
                        [0,0,1,0,0],
                        [0,0,0,1,0],
                        [1,0,0,0,0],
                        [0,1,0,0,0],
                        [0,0,1,0,0],
                        [0,0,0,0,1]])
    #lumped mass matrix : N x N   
    M = torch.eye(5)       

    #time step 
    dt = 0.03           
    pypd = pyPD(x,v,S,M,dt)

    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    test = ax.scatter([],[],[])



