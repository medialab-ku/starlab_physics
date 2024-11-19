import random

import numpy as np
import matplotlib.pyplot as plt
import taichi as ti
import json
import random as rd

from Scenes import concat_test as scene1
import os
from pathlib import Path
from framework.physics import XPBD
from framework.physics import XPBFEM
from framework.utilities import selection_tool as st
from framework.collision import SpatialHash as shash
from framework.utilities.make_plot import make_plot

import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# initialize the tri UI params
config_path = Path(__file__).resolve().parent / "framework" / "sim_config.json"
config_data = {}
default_data = {
    "sim_type": 0,
    "solver_type": 0,
    "dt_tri": 0.03,
    "dt_tet": 0.02,
    "n_substep": 20,
    "n_iter": 1,
    "dHat": 0.05,
    "fric_coef": 0.8,
    "damping": 0.001,
    "YM": 5e3,
    "YM_b": 5e3,
    "PCG_threshold": 1e-4,
    "PCG_definiteness_fix": True,
    "PCG_use_line_search": True,
    "PCG_print_stats": False,
    "PCG_max_cg_iter": 100,

    "PR": 0.2
}

# plot
plot_export_path = str(Path(__file__).resolve().parent / "results") + "/"
# x_name == "frame" and y_name == "iteration" -> line graph
# x_name == "frame" and y_name == "energy" -> line graph
plot = make_plot(plot_export_path, "frame", "iteration")
characters = 'ABCDEF0123456789'
plot_data_temp = {}

if not os.path.exists(config_path):
    with open(config_path, 'w') as json_file:
        json.dump(default_data, json_file, indent=4)

with open(config_path, 'r') as json_file:
    config_data = json.load(json_file)

sim_type_ui = config_data["sim_type"]
solver_type_ui = config_data["solver_type"]
precond_type_ui = config_data["precond_type"]
# print(solver_type_ui)
dt_tri_ui = config_data["dt_tri"]
dt_tet_ui = config_data["dt_tet"]
n_substep = config_data["n_substep"]
n_iter = config_data["n_iter"]
dHat_ui = config_data["dHat"]
friction_coeff_ui = config_data["fric_coef"]
damping_ui = config_data["damping"]
YM_ui = config_data["YM"]
YM_b_ui = config_data["YM_b"]


# PR_ui = config_data["PR"]

sh_st = shash.SpatialHash(grid_resolution=(64, 64, 64))
sh_dy = shash.SpatialHash(grid_resolution=(64, 64, 64))

sim_tri = XPBD.Solver(scene1.obj_mesh_dy,
                      # scene1.obj_mesh_st,
                      scene1.particles_st,
                      g=ti.math.vec3(0.0, -7.0, 0.0),
                      dt=dt_tri_ui,
                      stiffness_stretch=YM_ui,
                      stiffness_bending=YM_b_ui,
                      dHat=dHat_ui,
                      sh_st=sh_st,
                      sh_dy=sh_dy)


sim_tri.selected_solver_type = solver_type_ui
sim_tri.selected_precond_type = precond_type_ui

df = pd.DataFrame({'x': [], 'y': []})

# Create figure and axis
fig, ax = plt.subplots()
line, = ax.plot([], [], 'b')
# Set axis limits
ax.set_ylim(1e-4, 0.01)
# Function to update the plot dynamically
def update_plot(df):
    line.set_xdata(df['x'])
    line.set_ydata(df['y'])
    fig.canvas.draw()  # Redraw the plot
    fig.canvas.flush_events()  # Process GUI events


window = ti.ui.Window("PBD framework", (1024, 768), fps_limit=200)
gui = window.get_gui()
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = window.get_scene()
camera = ti.ui.Camera()
camera.position(2., 4.0, 7.0)
camera.fov(40)
camera.up(0, 1, 0)

run_sim = False
MODE_WIREFRAME = False
LOOKAt_ORIGIN = True
SHOW_GRAPH = False
USE_HEATMAP = True
USE_PNCG = True
PARTICLE = True
#selector
g_selector_tri = st.SelectionTool(sim_tri.num_verts_dy, sim_tri.mesh_dy.x, window, camera)

frame_end = 100
mesh_export = False
result_export = False
frame_cpu = 0

def show_options_tri():

    global n_substep
    global n_iter
    global dt_tri_ui
    global sim_type_ui
    global solver_type_ui
    global precond_type_ui
    global damping_ui
    global YM_ui
    global YM_b_ui
    global sim_tri
    global dHat_ui
    global friction_coeff_ui
    global MODE_WIREFRAME
    global SHOW_GRAPH
    global LOOKAt_ORIGIN
    global USE_HEATMAP
    global PARTICLE
    global USE_PNCG
    global mesh_export
    global result_export
    global frame_end

    old_dt = dt_tri_ui
    old_solver_type_ui = solver_type_ui
    old_precond_type_ui = precond_type_ui
    old_dHat = dHat_ui
    old_friction_coeff = dHat_ui
    old_damping = damping_ui
    YM_old = YM_ui
    YM_b_old = YM_b_ui

    with gui.sub_window("XPBD Settings", 0., 0., 0.3, 0.7) as w:

        precond_type_ui = w.slider_int("precond. type", precond_type_ui, 0, 1)
        if precond_type_ui == 0:
            w.text("precond. type: Euler")
        elif precond_type_ui == 1:
            w.text("precond. type: Jacobi")
        sim_tri.threshold = w.slider_float("CG threshold", sim_tri.threshold, 0.00001, 0.101)
        sim_tri.definiteness_fix = w.checkbox("definiteness fix", sim_tri.definiteness_fix)
        sim_tri.enable_line_search = w.checkbox("line search", sim_tri.enable_line_search)
        sim_tri.print_stats = w.checkbox("print stats.", sim_tri.print_stats)
        sim_tri.enable_pncg = w.checkbox("enable pncg",  sim_tri.enable_pncg)
        # sim_tri.max_cg_iter = w.slider_int("CG max iter", sim_tri.max_cg_iter, 1, 1000)


        dt_tri_ui = w.slider_float("dt", dt_tri_ui, 0.001, 0.101)
        n_substep = w.slider_int("# sub", n_substep, 1, 100)
        n_iter = w.slider_int("# iter", n_iter, 1, 100)
        dHat_ui = w.slider_float("dHat", dHat_ui, 0.0001, 1.101)
        friction_coeff_ui = w.slider_float("fric. coef.", friction_coeff_ui, 0.0, 1.0)
        damping_ui = w.slider_float("damping", damping_ui, 0.0, 1.0)
        YM_ui = w.slider_float("stretch stiff.", YM_ui, 0.0, 1e5)
        YM_b_ui = w.slider_float("bending stiff.", YM_b_ui, 0.0, 1e5)

        frame_str = "# frame: " + str(frame_cpu)
        w.text(frame_str)
        SHOW_GRAPH = w.checkbox("Show graph", SHOW_GRAPH)
        LOOKAt_ORIGIN = w.checkbox("Look at origin", LOOKAt_ORIGIN)

        sim_tri.enable_velocity_update = w.checkbox("velocity constraint", sim_tri.enable_velocity_update)
        # sim.enable_collision_handling = w.checkbox("handle collisions", sim.enable_collision_handling)
        mesh_export = w.checkbox("export mesh", mesh_export)
        result_export = w.checkbox("export result", result_export)

        if mesh_export or result_export:
            frame_end = w.slider_int("end frame", frame_end, 1, 2000)

        w.text("")
        w.text("dynamic mesh stats.")
        verts_str = "# verts: " + str(sim_tri.num_verts_dy)
        edges_str = "# edges: " + str(sim_tri.num_edges_dy)
        faces_str = "# faces: " + str(sim_tri.num_faces_dy)

        w.text(verts_str)
        w.text(edges_str)
        w.text(faces_str)


    if not old_dt == dt_tri_ui:
        sim_tri.dt = dt_tri_ui

    if not old_solver_type_ui == solver_type_ui:
        sim_tri.selected_solver_type = solver_type_ui

    if not old_precond_type_ui == precond_type_ui:
        sim_tri.selected_precond_type = precond_type_ui

    if not old_dHat == dHat_ui:
        sim_tri.dHat = dHat_ui

    if not old_friction_coeff == friction_coeff_ui:
        sim_tri.mu = friction_coeff_ui

    if not YM_old == YM_ui:
        sim_tri.stiffness_stretch = YM_ui

    if not YM_b_old == YM_b_ui:
        sim_tri.stiffness_bending = YM_b_ui

    if not old_damping == damping_ui:
        sim_tri.damping = damping_ui


while window.running:

    if LOOKAt_ORIGIN:
        camera.lookat(0.0, 0.0, 0.0)

    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.3, 0.3, 0.3))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.3, 0.3, 0.3))

    if window.get_event(ti.ui.PRESS):

        if window.event.key == 'x':  # export selection
            print("==== Vertex EXPORT!! ====")

            g_selector_tri.export_selection()


        if window.event.key == 'i':
            print("==== IMPORT!! ====")

            g_selector_tri.import_selection()
            sim_tri.set_fixed_vertices(g_selector_tri.is_selected)

            # load_animation()

        if window.event.key == 'k':
            print("== current selected vertices ==")
            for i in range(g_selector_tri.max_numverts_dynamic):
                if g_selector_tri.is_selected[i] >= 1:
                    print(i, end=' ')
            print()

        if window.event.key == 't':
            g_selector_tri.sewing_selection()

        if window.event.key == 'y':
            g_selector_tri.pop_sewing()

        if window.event.key == 'u':
            sim_tri.move_particle_x(-0.05)

        if window.event.key == 'o':
            sim_tri.move_particle_x(0.05)

        # if window.event.key == 'u':
        #     g_selector.remove_all_sewing()

        if window.event.key == ' ':
            run_sim = not run_sim

        if window.event.key == 'r':
            frame_cpu = 0
            sim_tri.reset()
            df = pd.DataFrame({'x': [], 'y': []})
            config_data = {
                "sim_type": sim_type_ui,
                "solver_type": solver_type_ui,
                "precond_type": precond_type_ui,
                "dt_tri": dt_tri_ui,
                "dt_tet": dt_tet_ui,
                "n_substep": n_substep,
                "n_iter": n_iter,
                "dHat": dHat_ui,
                "fric_coef": friction_coeff_ui,
                "damping": damping_ui,
                "YM": YM_ui,
                "YM_b": YM_b_ui,
                "PCG_threshold": sim_tri.threshold,
                "PCG_definiteness_fix": sim_tri.definiteness_fix,
                "PCG_use_line_search": sim_tri.enable_line_search,
                "PCG_print_stats": sim_tri.print_stats,
                "PCG_max_cg_iter": sim_tri.max_cg_iter,

            }
            with open(config_path, 'w') as json_file:
                json.dump(config_data, json_file, indent=4)

            # if sim_type_ui == 0:
            g_selector_tri.is_selected.fill(0.0)
            sim_tri.set_fixed_vertices(g_selector_tri.is_selected)



            # sim_tet.set_fixed_vertices(g_selector.is_selected)
            run_sim = False

        if window.event.key == 'v':
            sim_tri.enable_velocity_update = not sim_tri.enable_velocity_update
            if sim_tri.enable_velocity_update is True:
                print("velocity update on")
            else:
                print("velocity update off")

        if window.event.key == 'z':
            sim_tri.enable_collision_handling = not sim_tri.enable_collision_handling
            if sim_tri.enable_collision_handling is True:
                print("collision handling on")
            else:
                print("collision handling off")

        if window.event.key == 'h':
            print("fix vertices")
            sim_tri.set_fixed_vertices(g_selector_tri.is_selected)

        if window.event.key == 'l':
            p = plot.make_graph()
            if p is None:
                print("The graph is not correctly created!")
                print("You should not change the end frame number during collecting data...")
            else:
                plot.export_result(p)
                print("The graph is successfully exported!")

        if window.event.key == ti.ui.BACKSPACE:
            g_selector_tri.is_selected.fill(0)

        if window.event.key == ti.ui.LMB:
            # if sim_type_ui == 0:
            g_selector_tri.LMB_mouse_pressed = True
            g_selector_tri.mouse_click_pos[0], g_selector_tri.mouse_click_pos[1] = window.get_cursor_pos()


        if window.event.key == ti.ui.TAB:
            # if sim_type_ui == 0:
            g_selector_tri.MODE_SELECTION = not g_selector_tri.MODE_SELECTION

    if window.get_event(ti.ui.RELEASE):
        if window.event.key == ti.ui.LMB:
            g_selector_tri.LMB_mouse_pressed = False
            g_selector_tri.mouse_click_pos[2], g_selector_tri.mouse_click_pos[3] = window.get_cursor_pos()
            g_selector_tri.Select()


    if g_selector_tri.LMB_mouse_pressed:
        g_selector_tri.mouse_click_pos[2], g_selector_tri.mouse_click_pos[3] = window.get_cursor_pos()
        g_selector_tri.update_ti_rect_selection()


    if run_sim:
        sim_tri.forward(n_substeps=n_substep, n_iter=n_iter)
        frame_cpu += 1

    if SHOW_GRAPH:
        plt.ion()
        ax.set_ylim(0, sim_tri.max_cg_iter)

        if frame_cpu < 100:
            ax.set_xlim(0, 100)
        else:
            ax.set_xlim(0, frame_cpu)
        new_data = {'x': [frame_cpu], 'y': [sim_tri.PCG.cg_iter]}
        df = df._append(pd.DataFrame(new_data), ignore_index=True)
        update_plot(df)
        plt.show()

    show_options_tri()


    if mesh_export and run_sim:
        if frame_cpu < frame_end:
            E = sim_tri.compute_spring_energy(YM_ui)
            print(frame_cpu, E)

        else:
            run_sim = False
        # sim_tri.mesh_dy.export(os.path.basename(scene1.__file__), frame_cpu)

    if result_export:
        if not run_sim:
            plot_data_temp = {
                "name": ''.join(random.choices(characters, k=16)), # Hash name
                "label": "Euler" if precond_type_ui == 0 else "Jacobi",
                "conditions": {
                    "precond_type": "Euler" if precond_type_ui == 0 else "Jacobi",
                    "dt": dt_tri_ui,
                    "substep": n_substep,
                    "iter": n_iter,
                    "damping": damping_ui,
                    "YM": YM_ui,
                    "YM_b": YM_b_ui,
                },
                "data": {}
            }
        else:
            if frame_cpu < frame_end:
                # print(sim_tri.PCG.cg_iter)
                E = sim_tri.compute_spring_energy(YM_ui)
                # plot_data_temp["data"][frame_cpu] = E
                plot_data_temp["data"][frame_cpu] = sim_tri.PCG.cg_iter

            else:
                plot.collect_data(plot_data_temp)
                run_sim = False

    scene.mesh(sim_tri.mesh_dy.x, indices=sim_tri.mesh_dy.face_indices_flatten, per_vertex_color=sim_tri.mesh_dy.colors)
    scene.mesh(sim_tri.mesh_dy.x, indices=sim_tri.mesh_dy.face_indices_flatten, color=(0, 0.0, 0.0), show_wireframe=True)

    scene.lines(sh_st.bbox_vertices, width=1.0, indices=sh_st.bbox_edge_indices_flattened, color=(0, 0, 0))


    g_selector_tri.renderTestPos()
    scene.particles(g_selector_tri.renderTestPosition, radius=0.02, color=(1, 0, 1))
    canvas.lines(g_selector_tri.ti_mouse_click_pos, width=0.002, indices=g_selector_tri.ti_mouse_click_index, color=(1, 0, 1) if g_selector_tri.MODE_SELECTION else (0, 0, 1))

    # scene.lines(sim_tet.aabb_x0, indices=sim_tet.aabb_index0, width=1.0, color=(0.0, 0.0, 0.0))
    camera.track_user_inputs(window, movement_speed=0.4, hold_key=ti.ui.RMB)
    canvas.scene(scene)
    window.show()