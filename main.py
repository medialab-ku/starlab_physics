import numpy as np
import taichi as ti
import meshtaichi_patcher as Patcher
import argparse

import solver
import mf

mf = mf.mathFunctions()

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="obj_models/square_big.obj")
parser.add_argument('--sequence', default="seq_models/Kyra_DVStandClubbing/Kyra_DVStandClubbing_0000.obj")
parser.add_argument('--primitive', default="obj_models/square.obj")
parser.add_argument('--arch', default='gpu')
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

ti.init(arch=getattr(ti, args.arch))

mesh = Patcher.load_mesh(args.sequence, relations=["FV"])
mesh.verts.place({'x': ti.math.vec3,
                  'ox': ti.math.vec3,
                  'new_x': ti.math.vec3,
                  'v': ti.math.vec3,
                  'f': ti.math.vec3,
                  'n': ti.math.vec3})

mesh.edges.place({'rest_len': ti.f32})

mesh.verts.x.from_numpy(mesh.get_position_as_numpy())
mesh.verts.ox.copy_from(mesh.verts.x)
mesh.verts.new_x.copy_from(mesh.verts.x)
mesh.verts.v.fill([0.0, 0.0, 0.0])

primitive_mesh = Patcher.load_mesh(args.primitive, relations=["FV"])
primitive_mesh.verts.place({'x': ti.math.vec3,
                            'f': ti.math.vec3,
                            'n': ti.math.vec3})
primitive_mesh.verts.x.from_numpy(primitive_mesh.get_position_as_numpy())

solver = solver.solver(mesh, primitive_mesh, dt=1e-3, max_iter=1000)
solver.setMeshPos(1.0, (0, 2, 0))
solver.computeNormal()
solver.initialize()
solver.initIndices()

# Rendering
window = ti.ui.Window("Mass Spring", (1024, 768))

canvas = window.get_canvas()
canvas.set_background_color(color=(1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(20, 15, 20)
camera.up(0, 1.0, 0)
camera.lookat(0, 0, 0)
camera.fov(55)

frame = 0
### seq_mesh ###
# seq_mesh_path = args.sequence + "_%04d.obj" % (frame % 350)
# seq_mesh = Patcher.load_mesh(seq_mesh_path, relations=["FV"])
# seq_mesh.verts.place({'x': ti.math.vec3,
#                         'f': ti.math.vec3,
#                         'n': ti.math.vec3})
# seq_mesh.verts.x.from_numpy(seq_mesh.get_position_as_numpy())
# s_indices = ti.field(ti.u32, shape=len(seq_mesh.faces) * 3)
#
# @ti.kernel
# def initSeqIndices():
#     for f in seq_mesh.faces:
#         s_indices[f.id * 3 + 0] = f.verts[0].id
#         s_indices[f.id * 3 + 1] = f.verts[1].id
#         s_indices[f.id * 3 + 2] = f.verts[2].id
# initSeqIndices()

while window.running:
    ### seq mesh ###
    # seq_mesh_path = args.sequence + "_%04d.obj" % (frame % 350)
    # seq_mesh = Patcher.load_mesh(seq_mesh_path, relations=["FV"])
    # seq_mesh.verts.place({'x': ti.math.vec3,
    #                         'f': ti.math.vec3,
    #                         'n': ti.math.vec3})
    # seq_mesh.verts.x.from_numpy(seq_mesh.get_position_as_numpy() * 5)
    solver.applyExtForce()

    solver.update()
    # solver.collision_detection()

    frame += 1

    camera.track_user_inputs(window, movement_speed=0.0, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0.6, 0.6, 0.6))
    scene.point_light(pos=(3, 4, 2), color=(0.4, 0.4, 0.4))
    scene.point_light(pos=(2, 4, 3), color=(0.4, 0.4, 0.4))
    scene.mesh(solver.mesh.verts.x, solver.indices, normals=solver.mesh.verts.n, color=(0.5, 0.5, 0.5), show_wireframe=True)
    scene.mesh(solver.primitive.verts.x, solver.p_indices, normals=solver.primitive.verts.n, color=(0.5, 0.5, 0.5), show_wireframe=True)
    # scene.mesh(seq_mesh.verts.x, s_indices, normals=seq_mesh.verts.n, color=(0.5, 0.5, 0.5), show_wireframe=True)
    canvas.scene(scene)
    window.show()
    for event in window.get_events(ti.ui.PRESS):
        if event.key in [ti.ui.ESCAPE]:
            window.running = False
