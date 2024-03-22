# Macklin, M. and Müller, M., 2013. Position based fluids. ACM Transactions on Graphics (TOG), 32(4), p.104.
# Taichi implementation by Ye Kuang (k-ye)

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
plt.rcParams['font.family'] = 'Times New Roman'

import taichi as ti

ti.init(arch=ti.gpu)

pbd_num_iters = 1

time_delta = 1.0 / 60.0
dim = 2
density = 1.0

num_vert = -1
num_ele = -1
num_edge = -1


rest_positions = ti.Vector.field(dim, float)

old_positions = ti.Vector.field(dim, float)
positions = ti.Vector.field(dim, float)
positions_render = ti.Vector.field(dim, float)

velocities = ti.Vector.field(dim, float)


force_ext = ti.Vector.field(dim,float)
mass_particles = ti.field(float)
ele_count = ti.field(int)

pos_delta = ti.Vector.field(dim, float)

Dm = ti.Matrix.field(n=dim,m=dim,dtype=ti.f32)
Dm_inv = ti.Matrix.field(n=dim,m=dim,dtype=ti.f32)
ele_vol = ti.field(dtype=ti.f32)
ele_delta_x = ti.Vector.field(dim * (dim + 1), float)

ele_vert = ti.field(ti.uint32)
ele_vert_inv = ti.field(ti.uint32)
edge_vert = ti.field(ti.uint32)

world_side_len = 1000
particle_rad_in_world = 2

particle_rad_render = particle_rad_in_world/world_side_len


def init_ti_fields(num_vert,num_ele,num_edge) :

    ti.root.dense(ti.i, num_vert).place(old_positions, positions, positions_render, velocities,
                                        force_ext, mass_particles, rest_positions, ele_count, pos_delta)

    ti.root.dense(ti.i,num_ele).place(Dm, Dm_inv, ele_vol, ele_delta_x)

    ti.root.dense(ti.i, (dim+1) * num_ele).place(ele_vert,ele_vert_inv)

    ti.root.dense(ti.i, 2 * num_edge).place(edge_vert)

def gen_rect2D():

    global num_vert,num_ele,num_edge

    ld = ti.Vector([100,300])  # left down

    dispx = ti.Vector([250,0])  # side len x
    dispy = ti.Vector([0,500])  # side len y

    nx,ny=35 + 1, 70 + 1 # number of vertex

    num_vert = nx*ny
    num_ele = 2* (nx-1) * (ny-1)
    num_edge = num_ele * 3

    init_ti_fields(num_vert,num_ele,num_edge)

    # set vertex, element, edge
    for iy in range(ny) :
        for jx in range(nx) :
            positions[iy * nx + jx] = ld + jx/(nx-1)*dispx + iy/(ny-1)*dispy

    for eiy in range(ny-1) :
        for ejx in range(nx-1) :
            vld = eiy * (nx) + ejx
            vrd = eiy * (nx) + ejx + 1
            vlu = (eiy+1) * (nx) + ejx
            vru = (eiy+1) * (nx) + ejx + 1

            ele_vert[3 * ( 2*(eiy * (nx-1) + ejx) ) + 0] = vld
            ele_vert[3 * ( 2*(eiy * (nx-1) + ejx) ) + 1] = vru
            ele_vert[3 * ( 2*(eiy * (nx-1) + ejx) ) + 2] = vlu

            ele_vert[3 * ( 2*(eiy * (nx-1) + ejx) +1) + 0] = vld
            ele_vert[3 * ( 2*(eiy * (nx-1) + ejx) +1) + 1] = vru
            ele_vert[3 * ( 2*(eiy * (nx-1) + ejx) +1) + 2] = vrd

            edge_vert[6 * (2 * (eiy * (nx - 1) + ejx))] = vld
            edge_vert[6 * (2 * (eiy * (nx - 1) + ejx)) + 1] = vru
            edge_vert[6 * (2 * (eiy * (nx - 1) + ejx)) + 2] = vru
            edge_vert[6 * (2 * (eiy * (nx - 1) + ejx)) + 3] = vlu
            edge_vert[6 * (2 * (eiy * (nx - 1) + ejx)) + 4] = vlu
            edge_vert[6 * (2 * (eiy * (nx - 1) + ejx)) + 5] = vld

            edge_vert[6 * (2 * (eiy * (nx - 1) + ejx) + 1)] = vld
            edge_vert[6 * (2 * (eiy * (nx - 1) + ejx) + 1) + 1] = vru
            edge_vert[6 * (2 * (eiy * (nx - 1) + ejx) + 1) + 2] = vru
            edge_vert[6 * (2 * (eiy * (nx - 1) + ejx) + 1) + 3] = vrd
            edge_vert[6 * (2 * (eiy * (nx - 1) + ejx) + 1) + 4] = vrd
            edge_vert[6 * (2 * (eiy * (nx - 1) + ejx) + 1) + 5] = vld

def setup_mesh() :
    gen_rect2D()

@ti.kernel
def setup_rest_quantities_tet():
    for e in ti.ndrange(num_ele) :

        node0 = ele_vert[3 * e + 0]
        node1 = ele_vert[3 * e + 1]
        node2 = ele_vert[3 * e + 2]
        displ1 = positions[node0] - positions[node2]
        displ2 = positions[node1] - positions[node2]

        Dm_local = ti.Matrix.cols([displ1, displ2])

        Dm[e] = Dm_local
        Dm_inv[e] = ti.math.inverse(Dm_local)
        ele_vol[e] = 0.5*ti.abs(ti.math.determinant(Dm_local))

def setup_rest_quantities():
    setup_rest_quantities_tet()

    for e in range(num_ele):
        node0 = ele_vert[3 * e + 0]
        node1 = ele_vert[3 * e + 1]
        node2 = ele_vert[3 * e + 2]

        mass_frac = density*ele_vol[e]/3

        mass_particles[node0] += mass_frac
        mass_particles[node1] += mass_frac
        mass_particles[node2] += mass_frac

        ele_count[node0] +=1
        ele_count[node1] +=1
        ele_count[node2] +=1

def init_config() :
    for i in range(num_vert) :
        # positions[i] = ti.Vector([500.0,500.0])
        velocities[i] = ti.Vector([0.0,-30.0])

    for i in range(num_vert) :
        rest_positions[i] = positions[i]

def init() :
    setup_mesh()
    setup_rest_quantities()
    init_config()

@ti.kernel
def prologue():
    for v in positions:
        old_positions[v] = positions[v]
        force_ext[v] = ti.Vector([0, -9.81]) * mass_particles[v]

    # apply gravity within boundary
    for v in positions:
        pos, vel = positions[v], velocities[v]
        vel += force_ext[v] * time_delta / mass_particles[v]
        pos += vel * time_delta

        if(pos[1]<0.0) :
            pos[1] = 0.0
        if(pos[1]>world_side_len) :
            pos[1] = world_side_len
        if(pos[0]<0.0) :
            pos[0] = 0.0
        if(pos[0]>world_side_len) :
            pos[0] = world_side_len

        # if v==0 or v==70:
        #     pos = rest_positions[v] + ti.Vector([0.0,600.0])

        positions[v] = pos
        velocities[v] = (pos-old_positions[v])/time_delta

@ti.func
def get_F(e):
    node0, node1, node2 = ele_vert[3 * e + 0], ele_vert[3 * e + 1], ele_vert[3 * e + 2]
    pos0, pos1, pos2 = positions[node0], positions[node1], positions[node2]

    displ1 = pos0 - pos2
    displ2 = pos1 - pos2
    return ti.Matrix.cols([displ1, displ2]) @ Dm_inv[e]

@ti.func
def get_R(F):
    U, sig, V = ti.svd(F)
    if U.determinant() < 0:
        for i in ti.static(range(2)): U[i, 1] *= -1
        sig[1, 1] = -sig[1, 1]
    if V.determinant() < 0:
        for i in ti.static(range(2)): V[i, 1] *= -1
        sig[1, 1] = -sig[1, 1]

    R = U@V.transpose()
    return R

@ti.func
def calc_ele_delta_pos(e,C,gradC):

    v0, v1, v2 = ele_vert[3 * e + 0], ele_vert[3 * e + 1], ele_vert[3 * e + 2]
    m0 = mass_particles[v0]
    m1 = mass_particles[v1]
    m2 = mass_particles[v2]

    lam_denom = 1e-8

    lam_denom += (gradC[0] * gradC[0] + gradC[1] * gradC[1]) / m0
    lam_denom += (gradC[2] * gradC[2] + gradC[3] * gradC[3]) / m1
    lam_denom += (gradC[4] * gradC[4] + gradC[5] * gradC[5]) / m2

    lamb = - C / lam_denom

    delta_x = gradC * lamb

    delta_x[0] /= m0
    delta_x[1] /= m0
    delta_x[2] /= m1
    delta_x[3] /= m1
    delta_x[4] /= m2
    delta_x[5] /= m2

    return delta_x

@ti.kernel
def calc_proj_pos():
    for e in ti.ndrange(num_ele):
        F = get_F(e)
        R = get_R(F)

        # get constr
        C = (F-R).norm()
        C = 0.5*C*C

        # get grad constr
        dCdx = (F-R) @ Dm_inv[e].transpose()
        gradC = ti.Vector([dCdx[0,0], dCdx[1,0], dCdx[0,1], dCdx[1,1], -dCdx[0,0]-dCdx[0,1], -dCdx[1,0]-dCdx[1,1]])

        ele_delta_x[e] = calc_ele_delta_pos(e,C,gradC)

@ti.kernel
def agg_delta_x() :
    pos_delta.fill(0.0)
    ti.loop_config(serialize=True)
    for e in range(num_ele) :

        vidx0 = ele_vert[3*e + 0]
        vidx1 = ele_vert[3*e + 1]
        vidx2 = ele_vert[3*e + 2]

        dx0 = ti.Vector([ele_delta_x[e][0], ele_delta_x[e][1]]) / ele_count[vidx0]
        dx1 = ti.Vector([ele_delta_x[e][2], ele_delta_x[e][3]]) / ele_count[vidx1]
        dx2 = ti.Vector([ele_delta_x[e][4], ele_delta_x[e][5]]) / ele_count[vidx2]

        pos_delta[vidx0]+= dx0
        pos_delta[vidx1]+= dx1
        pos_delta[vidx2]+= dx2

@ti.kernel
def apply_delta_x():
    for v in positions :
        pos = positions[v] + pos_delta[v]

        if(pos[1]<0.0) :
            pos[1] = 0.0
        if(pos[1]>world_side_len) :
            pos[1] = world_side_len
        if(pos[0]<0.0) :
            pos[0] = 0.0
        if(pos[0]>world_side_len) :
            pos[0] = world_side_len

        # if v==0 or v==70:
        #     pos = rest_positions[v]+ ti.Vector([0.0,600.0])

        positions[v] = pos
        velocities[v] = (pos-old_positions[v])/time_delta



def substep():
    calc_proj_pos()
    agg_delta_x()
    apply_delta_x()


def run_pbd():
    prologue()
    for _ in range(pbd_num_iters):
        substep()

@ti.kernel
def set_render_pos():
    for v in range(num_vert):
        for _dim in ti.static(range(dim)):
            positions_render[v][_dim] = positions[v][_dim] / world_side_len

    ele_vert_inv.fill(0)
    for e in range(num_ele) :
        if (get_F(e)).determinant() < 0.0 : # redundantly calc
            ele_vert_inv[3 * e + 0] = ele_vert[3 * e + 0]
            ele_vert_inv[3 * e + 1] = ele_vert[3 * e + 1]
            ele_vert_inv[3 * e + 2] = ele_vert[3 * e + 2]


if __name__ == "__main__":

    init()

    window = ti.ui.Window('Window Title', (world_side_len,world_side_len))
    scene = ti.ui.Scene()
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))

    while window.running:
        #dynamics
        run_pbd()
        #render
        set_render_pos()
        # canvas.circles(centers=positions_render,radius = particle_rad_render)

        canvas.triangles(vertices=positions_render, indices=ele_vert, color=(0.2, 1.0, 0.4))
        canvas.triangles(vertices=positions_render, indices=ele_vert_inv, color=(1.0, 0.2, 0.4))

        # canvas.lines(vertices=positions_render,indices=edge_vert,color=(0,0,0),width=0.0007)

        window.show()
