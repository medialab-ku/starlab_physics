import numpy as np
import matplotlib.pyplot as plt
import taichi as ti
import json

from Scenes import concat_test as scene1
import os
from pathlib import Path
from framework.physics import XPBD
from framework.physics import XPBFEM
from framework.utilities import selection_tool as st
from framework.collision import SpatialHash as shash

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

if not os.path.exists(config_path):
    with open(config_path, 'w') as json_file:
        json.dump(default_data, json_file, indent=4)

with open(config_path, 'r') as json_file:
    config_data = json.load(json_file)

sim_type_ui = config_data["sim_type"]
solver_type_ui = config_data["solver_type"]
dt_tri_ui = config_data["dt_tri"]
dt_tet_ui = config_data["dt_tet"]
n_substep = config_data["n_substep"]
n_iter = config_data["n_iter"]
dHat_ui = config_data["dHat"]
friction_coeff_ui = config_data["fric_coef"]
damping_ui = config_data["damping"]
YM_ui = config_data["YM"]
YM_b_ui = config_data["YM_b"]
PR_ui = config_data["PR"]

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
sim_tet = XPBFEM.Solver(scene1.msh_mesh_dy,
                        g=ti.math.vec3(0.0, -9.8, 0.0),
                        dt=dt_tet_ui)

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
USE_HEATMAP = True
PARTICLE = True
#selector
g_selector_tri = st.SelectionTool(sim_tri.num_verts_dy, sim_tri.mesh_dy.x, window, camera)
g_selector_tet = st.SelectionTool(sim_tet.num_verts_dy, sim_tet.mesh_dy.x, window, camera)


frame_end = 100



mesh_export = False
frame_cpu = 0

def show_options_tri():

    global n_substep
    global n_iter
    global dt_tri_ui
    global sim_type_ui
    global solver_type_ui
    global damping_ui
    global YM_ui
    global YM_b_ui
    global sim_tri
    global dHat_ui
    global friction_coeff_ui
    global MODE_WIREFRAME
    global LOOKAt_ORIGIN
    global USE_HEATMAP
    global PARTICLE
    global mesh_export
    global frame_end

    old_dt = dt_tri_ui
    old_solver_type_ui = solver_type_ui
    old_dHat = dHat_ui
    old_friction_coeff = dHat_ui
    old_damping = damping_ui
    YM_old = YM_ui
    YM_b_old = YM_b_ui

    with gui.sub_window("XPBD Settings", 0., 0., 0.3, 0.7) as w:

        sim_type_ui = w.slider_int("sim type", sim_type_ui, 0, 1)
        solver_type_ui = w.slider_int("solver type", solver_type_ui, 0, 5)
        if solver_type_ui == 0:
            w.text("solver type: Jacobi")
        elif solver_type_ui == 1:
            w.text("solver type: PD-diag")
        elif solver_type_ui == 2:
            w.text("solver type: H-diag")
        elif solver_type_ui == 3:
            w.text("solver type: Euler Path")
        elif solver_type_ui == 4:
            w.text("solver type: Newton PCG")
            sim_tri.threshold = w.slider_float("CG threshold", sim_tri.threshold, 0.0001, 0.101)
            sim_tri.definiteness_fix = w.checkbox("definiteness fix", sim_tri.definiteness_fix)
            sim_tri.use_line_search = w.checkbox("line search", sim_tri.use_line_search)
            sim_tri.print_stats = w.checkbox("print stats.", sim_tri.print_stats)
            sim_tri.max_cg_iter = w.slider_int("CG max iter", sim_tri.max_cg_iter, 1, 100)
        elif solver_type_ui == 5:
            w.text("solver type: PD PCG")
            sim_tri.threshold = w.slider_float("CG threshold", sim_tri.threshold, 0.0001, 0.101)
            sim_tri.use_line_search = w.checkbox("line search", sim_tri.use_line_search)
            sim_tri.print_stats = w.checkbox("print stats.", sim_tri.print_stats)
            sim_tri.max_cg_iter = w.slider_int("CG max iter", sim_tri.max_cg_iter, 1, 100)


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

        LOOKAt_ORIGIN = w.checkbox("Look at origin", LOOKAt_ORIGIN)
        USE_HEATMAP = w.checkbox("heatmap", USE_HEATMAP)
        PARTICLE = w.checkbox("particle", PARTICLE)
        sim_tri.enable_velocity_update = w.checkbox("velocity constraint", sim_tri.enable_velocity_update)
        # sim.enable_collision_handling = w.checkbox("handle collisions", sim.enable_collision_handling)
        # mesh_export = w.checkbox("export mesh", mesh_export)

        if mesh_export is True:
            frame_end = w.slider_int("end frame", frame_end, 1, 2000)

        w.text("")
        w.text("dynamic mesh stats.")
        verts_str = "# verts: " + str(sim_tri.num_verts_dy)
        edges_str = "# edges: " + str(sim_tri.num_edges_dy)
        faces_str = "# faces: " + str(sim_tri.num_faces_dy)
        w.text(verts_str)
        w.text(edges_str)
        w.text(faces_str)
        # w.text("")
        # w.text("static mesh stats.")
        # verts_str = "# verts: " + str(sim_tri.num_verts_st)
        # edges_str = "# edges: " + str(sim_tri.num_edges_st)
        # faces_str = "# faces: " + str(sim_tri.num_faces_st)
        # w.text(verts_str)
        # w.text(edges_str)
        # w.text(faces_str)

    if not old_dt == dt_tri_ui:
        sim_tri.dt = dt_tri_ui

    if not old_solver_type_ui == solver_type_ui:
        sim_tri.selected_solver_type = solver_type_ui

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

def show_options_tet():

    global n_substep
    global dt_tet_ui
    global solver_type_ui
    global sim_type_ui
    global damping_ui
    global sim_tri
    global dHat_ui
    global YM_ui
    global PR_ui
    global MODE_WIREFRAME
    global LOOKAt_ORIGIN
    global PARTICLE
    global mesh_export
    global frame_end

    old_dt = dt_tet_ui
    old_solver_type_ui = solver_type_ui
    old_dHat = dHat_ui
    old_damping = damping_ui
    YM_old = YM_ui
    PR_old = PR_ui

    with gui.sub_window("XPBD Settings", 0., 0., 0.4, 0.35) as w:

        sim_type_ui = w.slider_int("sim type", sim_type_ui, 0, 1)

        solver_type_ui = w.slider_int("solver type", solver_type_ui, 0, 1)
        if solver_type_ui == 0:
            w.text("solver type: XPBD Jacobi")
        elif solver_type_ui == 1:
            w.text("solver type: PD diag")

        dt_tet_ui = w.slider_float("Time Step Size", dt_tet_ui, 0.001, 0.101)
        n_substep = w.slider_int("# Substepping", n_substep, 1, 100)
        dHat_ui = w.slider_float("dHat", dHat_ui, 0.001, 1.01)
        damping_ui = w.slider_float("Damping Ratio", damping_ui, 0.0, 1.0)
        YM_ui = w.slider_float("Young's Modulus", YM_ui, 0.0, 1e8)
        PR_ui = w.slider_float("Poisson's Ratio", PR_ui, 0.0, 0.49)

        frame_str = "# frame: " + str(frame_cpu)
        w.text(frame_str)

        LOOKAt_ORIGIN = w.checkbox("Look at origin", LOOKAt_ORIGIN)
        PARTICLE = w.checkbox("particle", PARTICLE)
        # sim.enable_velocity_update = w.checkbox("velocity constraint", sim.enable_velocity_update)
        # sim.enable_collision_handling = w.checkbox("handle collisions", sim.enable_collision_handling)
        # mesh_export = w.checkbox("export mesh", mesh_export)

        # if mesh_export is True:
        #     frame_end = w.slider_int("end frame", frame_end, 1, 2000)

        w.text("stats.")
        verts_str = "# verts: " + str(sim_tet.num_verts_dy)
        w.text(verts_str)
        tets_str = "# tets: " + str(sim_tet.num_tets)
        w.text(tets_str)
        # w.text("")
        # particles_st_str = "# static particles: " + str(sim.num_particles - sim.num_particles_dy)
        # w.text(particles_st_str)

    if not old_dt == dt_tet_ui:
        sim_tet.dt = dt_tet_ui

    if not old_dHat == dHat_ui:
        sim_tet.particle_rad = dHat_ui

    if not old_solver_type_ui == solver_type_ui:
        sim_tet.solver_type = solver_type_ui

    # if not old_friction_coeff == friction_coeff_ui:
    #     sim.mu = friction_coeff_ui
    #
    if not YM_old == YM_ui:
        sim_tet.YM = YM_ui

    if not PR_old == PR_ui:
        sim_tet.PR = PR_ui

    if not old_damping == damping_ui:
        sim_tet.damping = damping_ui

# def load_animation():
#     global sim_tri
#
#     with open('framework/animation/animation.json') as f:
#         animation_raw = json.load(f)
#     animation_raw = {int(k): v for k, v in animation_raw.items()}
#
#     animationDict = {(i+1):[] for i in range(4)}
#
#     for i in range(4):
#         ic = i + 1
#         icAnimation = animation_raw[ic]
#         listLen = len(icAnimation)
#         # print(listLen)
#         assert listLen % 7 == 0,str(ic)+"th Animation SETTING ERROR!! ======"
#
#         num_animation = listLen // 7
#
#         for a in range(num_animation) :
#             animationFrag = [animation_raw[ic][k + 7*a] for k in range(7)] # [vx,vy,vz,rx,ry,rz,frame]
#             animationDict[ic].append(animationFrag)

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
            if sim_type_ui == 0:
                g_selector_tri.export_selection()
            elif sim_type_ui == 1:
                g_selector_tet.export_selection()

        if window.event.key == 'i':
            print("==== IMPORT!! ====")
            if sim_type_ui == 0:
                g_selector_tri.import_selection()
                sim_tri.set_fixed_vertices(g_selector_tri.is_selected)
            elif sim_type_ui == 1:
                g_selector_tet.import_selection()
                sim_tet.set_fixed_vertices(g_selector_tet.is_selected)
            # load_animation()

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
            sim_tet.reset()

            config_data = {
                "sim_type": sim_type_ui,
                "solver_type": solver_type_ui,
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
                "PCG_use_line_search": sim_tri.use_line_search,
                "PCG_print_stats": sim_tri.print_stats,
                "PCG_max_cg_iter": sim_tri.max_cg_iter,
                "PR": sim_tet.PR
            }
            with open(config_path, 'w') as json_file:
                json.dump(config_data, json_file, indent=4)

            if sim_type_ui == 0:
                g_selector_tri.is_selected.fill(0.0)
                sim_tri.set_fixed_vertices(g_selector_tri.is_selected)

            elif sim_type_ui == 1:
                g_selector_tet.is_selected.fill(0.0)
                sim_tet.set_fixed_vertices(g_selector_tet.is_selected)

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

        if window.event.key == ti.ui.BACKSPACE:
            g_selector_tri.is_selected.fill(0)

        if window.event.key == ti.ui.LMB:
            if sim_type_ui == 0:
                g_selector_tri.LMB_mouse_pressed = True
                g_selector_tri.mouse_click_pos[0], g_selector_tri.mouse_click_pos[1] = window.get_cursor_pos()
            elif sim_type_ui == 1:
                g_selector_tet.LMB_mouse_pressed = True
                g_selector_tet.mouse_click_pos[0], g_selector_tet.mouse_click_pos[1] = window.get_cursor_pos()

        if window.event.key == ti.ui.TAB:
            if sim_type_ui == 0:
                g_selector_tri.MODE_SELECTION = not g_selector_tri.MODE_SELECTION
            elif sim_type_ui == 1:
                g_selector_tet.MODE_SELECTION = not g_selector_tet.MODE_SELECTION

    if window.get_event(ti.ui.RELEASE):
        if window.event.key == ti.ui.LMB:
            if sim_type_ui == 0:
                g_selector_tri.LMB_mouse_pressed = False
                g_selector_tri.mouse_click_pos[2], g_selector_tri.mouse_click_pos[3] = window.get_cursor_pos()
                g_selector_tri.Select()
            elif sim_type_ui == 1:
                g_selector_tet.LMB_mouse_pressed = False
                g_selector_tet.mouse_click_pos[2], g_selector_tet.mouse_click_pos[3] = window.get_cursor_pos()
                g_selector_tet.Select()
    if sim_type_ui == 0:
        if g_selector_tri.LMB_mouse_pressed:
            g_selector_tri.mouse_click_pos[2], g_selector_tri.mouse_click_pos[3] = window.get_cursor_pos()
            g_selector_tri.update_ti_rect_selection()
    elif sim_type_ui == 1:
        if g_selector_tet.LMB_mouse_pressed:
            g_selector_tet.mouse_click_pos[2], g_selector_tet.mouse_click_pos[3] = window.get_cursor_pos()
            g_selector_tet.update_ti_rect_selection()

    if run_sim:
        # sim.animate_handle(g_selector.is_selected)

        if sim_type_ui == 0:
            if frame_cpu == 0:
                sim_tri.particle_st.x_prev.copy_from(sim_tri.particle_st.x)

            sim_tri.move_particle_x(0.003)
            sim_tri.forward(n_substeps=n_substep, n_iter=n_iter)
            sim_tri.particle_st.x_prev.copy_from(sim_tri.particle_st.x_current)
        elif sim_type_ui == 1:
            sim_tet.forward(n_substeps=n_substep, n_iter=n_iter)

        frame_cpu += 1

    if sim_type_ui == 0:
        show_options_tri()
    elif sim_type_ui == 1:
        show_options_tet()

    if mesh_export and run_sim and frame_cpu < frame_end:
        sim_tri.mesh_dy.export(os.path.basename(scene1.__file__), frame_cpu)

    if sim_type_ui == 0:
        # scene.mesh(sim_tri.mesh_dy.x, indices=sim_tri.mesh_dy.face_indices_flatten, color=(1, 0.5, 0.0))

        if PARTICLE:
            if USE_HEATMAP:
                rho0_np = sim_tri.mesh_dy.rho0_sample.to_numpy()
                colormap = plt.colormaps['viridis']
                norm = plt.Normalize(vmin=np.min(rho0_np), vmax=np.max(rho0_np))
                rgb_array = colormap(norm(rho0_np))[:, :3]
                # print(rgb_array.shape)
                sim_tri.mesh_dy.heat_map_sample.from_numpy(rgb_array)
                scene.particles(sim_tri.mesh_dy.x_sample, radius=sim_tri.dHat, per_vertex_color=sim_tri.mesh_dy.heat_map_sample)
                #TODO
                # scene.particles(sim_tri.mesh_dy.test_particles, radius=sim_tri.dHat, per_vertex_color=sim_tri.mesh_dy.heat_map)
                # scene.particles(sim_tri.mesh_dy.x_sample, radius=sim_tri.dHat, per_vertex_color=sim_tri.mesh_dy.heat_map)
                rho0_np = sim_tri.particle_st.rho0.to_numpy()
                colormap = plt.colormaps['viridis']
                norm = plt.Normalize(vmin=np.min(rho0_np), vmax=np.max(rho0_np))
                rgb_array = colormap(norm(rho0_np))[:, :3]
                # print(rgb_array.shape)
                sim_tri.particle_st.heat_map.from_numpy(rgb_array)
                scene.particles(sim_tri.particle_st.x, radius=sim_tri.dHat, per_vertex_color=sim_tri.particle_st.heat_map)
            else:
                # scene.particles(sim_tri.mesh_dy.x, radius=sim_tri.dHat, color=(1.0, 0.0, 0.0))

                #TODO
                scene.particles(sim_tri.mesh_dy.x_sample, radius=sim_tri.dHat, color=(0.0, 1.0, 0.0))

                # scene.particles(sim_tri.mesh_dy.x_e, radius=sim_tri.dHat,  color=(0.0, 1.0, 0.0))
                # scene.particles(sim_tri.mesh_dy.x_f, radius=sim_tri.dHat,  color=(0.0, 0.0, 1.0))
                scene.particles(sim_tri.particle_st.x, radius=sim_tri.dHat, color=(0.3, 0.3, 0.3))


            # scene.mesh(sim_tri.mesh_dy.x, indices=sim_tri.mesh_dy.face_indices_flatten, color=(0, 0.0, 0.0), show_wireframe=True)
            # scene.mesh(sim_tri.mesh_st.x, indices=sim_tri.mesh_st.face_indices_flatten, color=(0, 0.0, 0.0),show_wireframe=True)
            # scene.particles(sim_tri.mesh_st.x_test, radius=sim_tri.dHat, color=(0.3, 0.3, 0.3))
            #     scene.particles(sim_tri.particle_st.x, radius=sim_tri.dHat, color=(0.3, 0.3, 0.3))
        else:

            scene.mesh(sim_tri.mesh_dy.x, indices=sim_tri.mesh_dy.face_indices_flatten, per_vertex_color=sim_tri.mesh_dy.colors)
            scene.mesh(sim_tri.mesh_dy.x, indices=sim_tri.mesh_dy.face_indices_flatten, color=(0, 0.0, 0.0), show_wireframe=True)

            # scene.mesh(sim_tri.mesh_st.x, indices=sim_tri.mesh_st.face_indices_flatten, color=(0.3, 0.3, 0.3))
            # scene.mesh(sim_tri.mesh_st.x, indices=sim_tri.mesh_st.face_indices_flatten, color=(0, 0.0, 0.0), show_wireframe=True)
            scene.particles(sim_tri.particle_st.x, radius=sim_tri.dHat, color=(0.3, 0.3, 0.3))


        scene.lines(sh_st.bbox_vertices, width=1.0, indices=sh_st.bbox_edge_indices_flattened, color=(0, 0, 0))


    elif sim_type_ui == 1:
        scene.mesh(sim_tet.x, indices=sim_tet.faces, per_vertex_color=sim_tet.mesh_dy.color)
        scene.mesh(sim_tet.x, indices=sim_tet.faces, color=(0.0, 0.0, 0.0), show_wireframe=True)
        scene.lines(sim_tet.aabb_x0, indices=sim_tet.aabb_index0, width=1.0, color=(0.0, 0.0, 0.0))

    # scene.lines(sim.mesh_dy.x_euler, indices=sim.mesh_dy.edge_indices_euler, width=1.0, color=(0., 0., 0.))
    # scene.particles(sim.mesh_dy.x_euler, radius=0.02, color=(0., 0., 0.))
    # sim.mesh_dy.colors_edge_euler.fill(ti.math.vec3([1.0, 0.0, 0.0]))
    # scene.particles(sim.mesh_dy.colored_edge_pos_euler, radius=0.05,  per_vertex_color=sim.mesh_dy.colors_edge_euler)
    # if sim.mesh_st != None:
    #     scene.mesh(sim.mesh_st.x, indices=sim.mesh_st.face_indices_flatten, color=(0, 0.0, 0.0), show_wireframe=True)
    #     scene.mesh(sim.mesh_st.x, indices=sim.mesh_st.face_indices_flatten, per_vertex_color=sim.mesh_st.colors)

    if sim_type_ui == 0:
        g_selector_tri.renderTestPos()
        scene.particles(g_selector_tri.renderTestPosition, radius=0.02, color=(1, 0, 1))
        canvas.lines(g_selector_tri.ti_mouse_click_pos, width=0.002, indices=g_selector_tri.ti_mouse_click_index, color=(1, 0, 1) if g_selector_tet.MODE_SELECTION else (0, 0, 1))
    elif sim_type_ui == 1:
        g_selector_tet.renderTestPos()
        scene.particles(g_selector_tet.renderTestPosition, radius=0.02, color=(1, 0, 1))
        canvas.lines(g_selector_tet.ti_mouse_click_pos, width=0.002, indices=g_selector_tet.ti_mouse_click_index, color=(1, 0, 1) if g_selector_tet.MODE_SELECTION else (0, 0, 1))

    # scene.lines(sim_tet.aabb_x0, indices=sim_tet.aabb_index0, width=1.0, color=(0.0, 0.0, 0.0))
    camera.track_user_inputs(window, movement_speed=0.8, hold_key=ti.ui.RMB)
    canvas.scene(scene)
    window.show()