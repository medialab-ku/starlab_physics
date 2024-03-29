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
time_delta_gpu = ti.field(ti.f32,shape = ())
time_delta_gpu[None] = time_delta

frame = 0
frame_gpu = ti.field(ti.i32,shape = ())

world_side_len = 1300
particle_rad_in_world = 2

particle_rad_render = particle_rad_in_world/world_side_len

density = 1.0
dim = 2

num_vert = -1
num_ele = -1
num_edge = -1

rest_positions = ti.Vector.field(dim, float)
handle_positions = ti.Vector.field(dim, float)
handle_positions_render = ti.Vector.field(dim, float)

old_positions = ti.Vector.field(dim, float)
positions = ti.Vector.field(dim, float)
positions_render = ti.Vector.field(dim, float)

velocities = ti.Vector.field(dim, float)

force_ext = ti.Vector.field(dim,float)
mass_particles = ti.field(float)
ele_count = ti.field(int)

state_delta = ti.Vector.field(dim, float)

Dm = ti.Matrix.field(n=dim,m=dim,dtype=ti.f32)
Dm_inv = ti.Matrix.field(n=dim,m=dim,dtype=ti.f32)
ele_vol = ti.field(dtype=ti.f32)
ele_delta_state = ti.Vector.field(dim * (dim + 1), float) # auxiliary variable

ele_vert = ti.field(ti.uint32)
ele_vert_inv = ti.field(ti.uint32)
edge_vert = ti.field(ti.uint32)


handle_animation_idx = ti.field(ti.uint32)
handle_animation = ti.Vector.field(n=5, dtype=ti.f32, shape=(2,)) # vel x,vel y, start_frame, frame, destroy_frame

def init_ti_fields(num_vert,num_ele,num_edge) :

    ti.root.dense(ti.i, num_vert).place(old_positions, positions, positions_render, velocities,
                                        force_ext, mass_particles, rest_positions, handle_positions, ele_count, state_delta,
                                        handle_animation_idx, handle_positions_render)

    ti.root.dense(ti.i,num_ele).place(Dm, Dm_inv, ele_vol, ele_delta_state)

    ti.root.dense(ti.i, (dim+1) * num_ele).place(ele_vert,ele_vert_inv)

    ti.root.dense(ti.i, 2 * num_edge).place(edge_vert)

def set_animation(nx,ny) :

    # for i in range(ny):
    #     handle_animation_idx[i * nx] = 1
    #
    # for i in range(ny):
    #     handle_animation_idx[i * nx + nx - 1] = 2

    handle_animation[0] = ti.Vector([-110, 0, 100, 200, 500])
    handle_animation[1] = ti.Vector([110, 0, 100, 200, 500])


def gen_rect2D():

    global num_vert,num_ele,num_edge

    ld = ti.Vector([600,400])  # left down

    dispx = ti.Vector([50,0])  # side len x
    dispy = ti.Vector([0,50])  # side len y


    num_vert = 3
    num_ele = 1
    num_edge = num_ele * 3

    init_ti_fields(num_vert,num_ele,num_edge)

    # set_animation(nx,ny)

    # set vertex, element, edge
    positions[0] = ld
    positions[1] = ld + dispx
    positions[2] = ld + dispy

    ele_vert[0] = 0
    ele_vert[1] = 1
    ele_vert[2] = 2

    edge_vert[0] = 0
    edge_vert[1] = 1
    edge_vert[2] = 0
    edge_vert[3] = 2
    edge_vert[4] = 1
    edge_vert[5] = 2



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
    velocities[0] = ti.Vector([0.0,0.0])
    for i in range(num_vert) :
        rest_positions[i] = positions[i]

def init() :
    setup_mesh()
    setup_rest_quantities()
    init_config()

@ti.func
def set_handle_position(v):

    if(handle_animation_idx[v]>0) :
        anim_type = handle_animation_idx[v] - 1

        anim_vel = ti.Vector([handle_animation[anim_type][0], handle_animation[anim_type][1]])

        startIdx = handle_animation[anim_type][2]
        activeIdx = handle_animation[anim_type][3]
        destroyIdx = handle_animation[anim_type][4]

        cur_frame = frame_gpu[None]

        if(cur_frame < startIdx) :
            handle_positions[v]= rest_positions[v]
        elif (startIdx<=cur_frame and cur_frame< startIdx + activeIdx) :
            handle_positions[v] = rest_positions[v] + anim_vel * time_delta_gpu[None] * (cur_frame-startIdx)
        elif (startIdx + activeIdx <= cur_frame and cur_frame < destroyIdx ) :
            handle_positions[v] = rest_positions[v] + anim_vel * time_delta_gpu[None] * activeIdx
        else :
            handle_animation_idx[v] = 0

@ti.kernel
def prologue():
    for v in positions:
        if v==0:
            frame_gpu[None] = frame_gpu[None] + 1
        old_positions[v] = positions[v]
        force_ext[v] = ti.Vector([0, 0]) * mass_particles[v]

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

        positions[v] = pos
        velocities[v] = (pos-old_positions[v])/time_delta

    for v in positions :
        set_handle_position(v)

@ti.func
def get_F(e):
    node0, node1, node2 = ele_vert[3 * e + 0], ele_vert[3 * e + 1], ele_vert[3 * e + 2]
    pos0, pos1, pos2 = positions[node0], positions[node1], positions[node2]

    displ1 = pos0 - pos2
    displ2 = pos1 - pos2
    return ti.Matrix.cols([displ1, displ2]) @ Dm_inv[e]
@ti.func
def get_dFdt(e):
    node0, node1, node2 = ele_vert[3 * e + 0], ele_vert[3 * e + 1], ele_vert[3 * e + 2]
    vel0, vel1, vel2 = velocities[node0], velocities[node1], velocities[node2]

    displ1 = vel0 - vel2
    displ2 = vel1 - vel2
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

@ti.func
def calc_length_constraint(e, F, R, J):
    C = (F-R).norm()
    C = 0.5*C*C
    return C

@ti.func
def calc_grad_length_constraint(e, F, R, J) :
    dCdx = (F - R)
    dCdx = dCdx @ Dm_inv[e].transpose()
    gradC = ti.Vector(
        [dCdx[0, 0], dCdx[1, 0], dCdx[0, 1], dCdx[1, 1], -dCdx[0, 0] - dCdx[0, 1], -dCdx[1, 0] - dCdx[1, 1]])
    return gradC

@ti.func
def calc_volume_constraint(e, F, R, J):
    volC = J - 1
    C= 0.5 * volC * volC
    return C

@ti.func
def calc_grad_volume_constraint(e, F, R, J) :
    col1 = ti.Vector([F[1,1],-F[0,1]])
    col2 = ti.Vector([-F[1,0],F[0,0]])
    dJdF = ti.Matrix.cols([col1,col2])
    dCdx = (J-1) * dJdF
    dCdx = dCdx @ Dm_inv[e].transpose()

    gradC = ti.Vector([dCdx[0, 0], dCdx[1, 0], dCdx[0, 1], dCdx[1, 1], -dCdx[0, 0] - dCdx[0, 1], -dCdx[1, 0] - dCdx[1, 1]])

    return gradC

@ti.func
def calc_attachment_constraint(e):
    C = 0.0
    # attachment constraint
    for i in range(3) :
        vv = ele_vert[3*e+i]
        if(handle_animation_idx[vv]!=0) :
            len = (positions[vv] - handle_positions[vv]).norm()
            C+= 0.5*len*len
    return C
@ti.func
def calc_grad_attachment_constraint(e) :

    gradC = ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0])
    # attachment constraint
    for i in range(3) :
        vv = ele_vert[3*e+i]
        if(handle_animation_idx[vv]!=0) :
            grad_v = (positions[vv] - handle_positions[vv])
            gradC[2*i] += grad_v[0]
            gradC[2*i+1] +=grad_v[1]

    return gradC


@ti.kernel
def calc_proj_pos():
    for e in ti.ndrange(num_ele):
        F = get_F(e)
        J = F.determinant()
        R = get_R(F)

        C = calc_length_constraint(e, F, R, J)
        gradC = calc_grad_length_constraint(e, F, R, J)
        ele_delta_state[e] = calc_ele_delta_pos(e, C, gradC)

        C = calc_volume_constraint(e, F, R, J)
        gradC = calc_grad_volume_constraint(e, F, R, J)
        ele_delta_state[e] += calc_ele_delta_pos(e, C, gradC)

        C = calc_attachment_constraint(e)
        gradC = calc_grad_attachment_constraint(e)
        ele_delta_state[e] += calc_ele_delta_pos(e, C, gradC)

@ti.kernel
def agg_delta_state() :
    state_delta.fill(0.0)
    ti.loop_config(serialize=True)
    for e in range(num_ele) :

        vidx0 = ele_vert[3*e + 0]
        vidx1 = ele_vert[3*e + 1]
        vidx2 = ele_vert[3*e + 2]

        dx0 = ti.Vector([ele_delta_state[e][0], ele_delta_state[e][1]]) / ele_count[vidx0]
        dx1 = ti.Vector([ele_delta_state[e][2], ele_delta_state[e][3]]) / ele_count[vidx1]
        dx2 = ti.Vector([ele_delta_state[e][4], ele_delta_state[e][5]]) / ele_count[vidx2]

        state_delta[vidx0]+= dx0
        state_delta[vidx1]+= dx1
        state_delta[vidx2]+= dx2

@ti.kernel
def apply_delta_pos():
    for v in positions :
        pos = positions[v] + 0.1*state_delta[v]

        if(pos[1]<0.0) :
            pos[1] = 0.0
        if(pos[1]>world_side_len) :
            pos[1] = world_side_len
        if(pos[0]<0.0) :
            pos[0] = 0.0
        if(pos[0]>world_side_len) :
            pos[0] = world_side_len

        positions[v] = pos
        velocities[v] = (pos-old_positions[v])/time_delta

@ti.func
def calc_grad_volumeRate_constraint(e, dFdt, C) :
    col1 = ti.Vector([dFdt[1,1],-dFdt[0,1]])
    col2 = ti.Vector([-dFdt[1,0],dFdt[0,0]])
    dCd_dFdt_ = ti.Matrix.cols([col1,col2])
    dCdv = dCd_dFdt_ @ Dm_inv[e].transpose()

    gradvC = ti.Vector([dCdv[0, 0], dCdv[1, 0], dCdv[0, 1], dCdv[1, 1], -dCdv[0, 0] - dCdv[0, 1], -dCdv[1, 0] - dCdv[1, 1]])

    return gradvC


@ti.kernel
def calc_proj_vel():
    for e in ti.ndrange(num_ele):
        F = get_F(e)
        J = F.determinant()
        dFdt = get_dFdt(e)

        C = dFdt.determinant()

        ele_delta_state[e] = ti.Vector([0.0,0.0,0.0,0.0,0.0,0.0])

        print(J,C,"====")
        if J<0.9 and C < 0.0:
            gradvC = calc_grad_volumeRate_constraint(e,dFdt,C)
            ele_delta_state[e] = calc_ele_delta_pos(e,C,gradvC)

@ti.kernel
def apply_delta_vel():
    for v in positions :
        velocities[v] += state_delta[v]

@ti.kernel
def tifunccaller() :
    print(get_dFdt(0).determinant(),velocities[0],velocities[1],velocities[2])
    print(get_dFdt(0).determinant(),velocities[1]-velocities[0],velocities[2]-velocities[0])

def substep():

    calc_proj_pos()
    agg_delta_state()
    apply_delta_pos()
    # pos update 됨. velocity도 있음 ㅇㅇ 일반적인 PBD

    calc_proj_vel()
    agg_delta_state()

    tifunccaller()
    apply_delta_vel()
    tifunccaller()



def run_pbd():
    prologue()
    for _ in range(pbd_num_iters):
        substep()

@ti.kernel
def set_render_pos():
    for v in range(num_vert):
        for _dim in ti.static(range(dim)):
            positions_render[v][_dim] = positions[v][_dim] / world_side_len

        if handle_animation_idx[v]!=0 :
            handle_positions_render[v] = handle_positions[v]/ world_side_len
        else :
            handle_positions_render[v] = ti.Vector([-1.0,-1.0])

    ele_vert_inv.fill(0)
    for e in range(num_ele) :
        if (get_F(e)).determinant() < 0.0 : # redundantly calc
            ele_vert_inv[3 * e + 0] = ele_vert[3 * e + 0]
            ele_vert_inv[3 * e + 1] = ele_vert[3 * e + 1]
            ele_vert_inv[3 * e + 2] = ele_vert[3 * e + 2]


if __name__ == "__main__":
    print("frame" + str(frame))
    init()

    window = ti.ui.Window('Window Title', (world_side_len,world_side_len))
    scene = ti.ui.Scene()
    canvas = window.get_canvas()
    canvas.set_background_color((1, 1, 1))

    while window.running:
        if(frame == 100) :
            velocities[0] = ti.Vector([800,800])

        print("frame cpu: ",frame)
        #dynamics
        run_pbd()
        #render
        set_render_pos()

        canvas.triangles(vertices=positions_render, indices=ele_vert, color=(0.2, 1.0, 0.4))
        canvas.triangles(vertices=positions_render, indices=ele_vert_inv, color=(1.0, 0.2, 0.4))

        canvas.lines(vertices=positions_render,indices=edge_vert,color=(0,0,0),width=0.0007)

        canvas.circles(centers=handle_positions_render,radius = particle_rad_render, color = (0,0,1))

        window.show()

        window.save_image("./img/asdf" + str(frame) + ".jpg")

        frame = frame+1
