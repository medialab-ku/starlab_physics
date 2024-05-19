import taichi as ti
import json

from Scenes import test_fem as scene1
# from Scenes import cloth_swing as scene1
# from Scenes import cloth_slide as scene1
# from Scenes import collition_unit_test as scene1
from Scenes import cloth_stack as scene1
# from Scenes import scene_cylinder_crossing as scene1
# from Scenes import scene_cylinder_crossing_4 as scene1
# from Scenes import scene_thin_shell_twist as scene1
# from Scenes import scene_cube_stretch as scene1
# from Scenes import scene_fluid_compression as scene1
# from Scenes import moving_obstacle as scene1
# from Scenes import scene_torus_chain as scene1
import os
import XPBD
import selection_tool as st

sim = XPBD.Solver(scene1.enable_profiler, scene1.meshes_dynamic, scene1.meshes_static, scene1.tet_meshes_dynamic, scene1.particles, g=ti.math.vec3(0.0, -9.81, 0.0), dt=0.03, grid_size=ti.math.vec3(8., 8., 8.), YM=1e6, PR=0.45, particle_radius=0.02, dHat=4e-3)
window = ti.ui.Window("PBD framework", (1024, 768), fps_limit=200)
gui = window.get_gui()
canvas = window.get_canvas()
canvas.set_background_color((1, 1, 1))
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(2., 4.0, 7.0)
camera.fov(40)
camera.up(0, 1, 0)

colors_static = []
colors_static.append((1.0, 0.0, 0))
colors_static.append((0, 0.5, 0))
colors_static.append((1, 0.5, 0))
colors_static.append((0, 0.5, 1))

run_sim = False

MODE_WIREFRAME = False
LOOKAt_ORIGIN = True

#selector
g_selector = st.SelectionTool(sim.max_num_verts_dynamic, sim.x, window, camera)
print("sim.max_num_verts_dynamic", sim.max_num_verts_dynamic)

n_substep = 25
frame_end = 100
dt_ui = sim.dt[0]
dHat_ui = sim.dHat[0]
strain_limit_ui = sim.strain_limit[0]
ang_vel_x_ui = sim.obs_ang_vel[0][0]
ang_vel_y_ui = sim.obs_ang_vel[0][1]
ang_vel_z_ui = sim.obs_ang_vel[0][2]

lin_vel_x_ui = sim.obs_lin_vel[0][0]
lin_vel_y_ui = sim.obs_lin_vel[0][1]
lin_vel_z_ui = sim.obs_lin_vel[0][2]

PR_ui = sim.PR[0]
YM_ui = sim.YM[0]
friction_coeff_ui = sim.friction_coeff[0]
mesh_export = False
frame_cpu = 0

def show_options():
    global n_substep
    global dt_ui
    global strain_limit_ui
    global YM_ui
    global PR_ui
    global sim
    global dHat_ui
    global friction_coeff_ui
    global MODE_WIREFRAME
    global LOOKAt_ORIGIN
    global mesh_export
    global frame_end

    old_dt = dt_ui
    old_dHat = dHat_ui
    old_friction_coeff = dHat_ui
    old_strain_limit = strain_limit_ui
    YM_old = YM_ui
    PR_old = PR_ui

    with gui.sub_window("Time Step", 0., 0., 0.3, 0.5) as w:
        # dt_ui = w.slider_float("dt", dt_ui, 0.0, 0.1)
        dt_ui = w.slider_float("dt", dt_ui, 0.001, 0.101)

        n_substep = w.slider_int("# sub", n_substep, 1, 100)
        dHat_ui = w.slider_float("dHat", dHat_ui, 0.0001, 0.0101)
        friction_coeff_ui = w.slider_float("fric. coef.", friction_coeff_ui, 0.0, 1.0)
        strain_limit_ui = w.slider_float("strain limit", strain_limit_ui, 0.0, 1.0)
        YM_ui = w.slider_float("YM", YM_ui, 0.0, 1e8)
        if sim.max_num_tetra_dynamic > 0:
            PR_ui = w.slider_float("PR", PR_ui, 0.0, 0.495)

        MODE_WIREFRAME = w.checkbox("wireframe", MODE_WIREFRAME)
        LOOKAt_ORIGIN = w.checkbox("Look at origin", LOOKAt_ORIGIN)

        sim.enable_velocity_update = w.checkbox("velocity constraint", sim.enable_velocity_update)
        sim.enable_collision_handling = w.checkbox("handle collisions", sim.enable_collision_handling)
        mesh_export = w.checkbox("export mesh", mesh_export)

        if mesh_export is True:
            frame_end = w.slider_int("end frame", frame_end, 1, 2000)

        frame_str = "# frame: " + str(frame_cpu)
        verts_str = "# verts: " + str(sim.max_num_verts_dynamic)
        edges_str = "# edges: " + str(sim.max_num_edges_dynamic)

        w.text(frame_str)
        w.text(verts_str)
        w.text(edges_str)

        if sim.max_num_tetra_dynamic > 0:
            tetra_str = "# tetrs: " + str(sim.max_num_tetra_dynamic)
            w.text(tetra_str)
            rest_volume = sim.rest_volume[0]
            current_volume = sim.current_volume[0]

            volume_ratio = round(100.0 * current_volume / rest_volume, 2)
            volume_ratio_str = "volume ratio(%): " + str(volume_ratio) + "%"
            w.text(volume_ratio_str)

            num_inverted_tetrs = sim.num_inverted_elements[0]
            num_inverted_tetrs_str = "# inverted tetrs: " + str(num_inverted_tetrs)
            w.text(num_inverted_tetrs_str)

    if not old_dt == dt_ui:
        sim.dt[0] = dt_ui

    if not old_dHat == dHat_ui:
        sim.dHat[0] = dHat_ui

    if not old_friction_coeff == friction_coeff_ui:
        sim.friction_coeff[0] = friction_coeff_ui

    if not YM_old == YM_ui:
        sim.YM[0] = YM_ui

    if not PR_old == PR_ui:
        sim.PR[0] = PR_ui

    if not old_strain_limit == strain_limit_ui:
        sim.strain_limit[0] = strain_limit_ui

    global ang_vel_x_ui
    global ang_vel_y_ui
    global ang_vel_z_ui

    global lin_vel_x_ui
    global lin_vel_y_ui
    global lin_vel_z_ui

    old_ang_vel_x_ui = ang_vel_x_ui
    old_ang_vel_y_ui = ang_vel_y_ui
    old_ang_vel_z_ui = ang_vel_z_ui

    old_lin_vel_x_ui = lin_vel_x_ui
    old_lin_vel_y_ui = lin_vel_y_ui
    old_lin_vel_z_ui = lin_vel_z_ui

    with gui.sub_window("Move obstacle", 0.8, 0.8, 0.3, 0.5) as w:
        sim.enable_move_obstacle = w.checkbox("move obstacle", sim.enable_move_obstacle)
        ang_vel_x_ui = w.slider_float("ang vel x", ang_vel_x_ui, -40, 40.)
        ang_vel_y_ui = w.slider_float("ang vel y", ang_vel_y_ui, -40, 40.)
        ang_vel_z_ui = w.slider_float("ang vel z", ang_vel_z_ui, -40, 40.)

        lin_vel_x_ui = w.slider_float("lin vel x", lin_vel_x_ui, -40.0, 40.)
        lin_vel_y_ui = w.slider_float("lin vel y", lin_vel_y_ui, -40.0, 40.)
        lin_vel_z_ui = w.slider_float("lin vel z", lin_vel_z_ui, -40.0, 40.)

    if not old_ang_vel_x_ui == ang_vel_x_ui:
        sim.obs_ang_vel[0][0] = ang_vel_x_ui

    if not old_ang_vel_y_ui == ang_vel_y_ui:
        sim.obs_ang_vel[0][1] = ang_vel_y_ui

    if not old_ang_vel_z_ui == ang_vel_z_ui:
        sim.obs_ang_vel[0][2] = ang_vel_z_ui

    if not old_lin_vel_x_ui == lin_vel_x_ui:
        sim.obs_lin_vel[0][0] = lin_vel_x_ui

    if not old_lin_vel_y_ui == lin_vel_y_ui:
        sim.obs_lin_vel[0][1] = lin_vel_y_ui

    if not old_lin_vel_z_ui == lin_vel_z_ui:
        sim.obs_lin_vel[0][2] = lin_vel_z_ui

    with gui.sub_window("Debug", 0.8, 0.8, 0.3, 0.5) as w:

        if sim.max_num_tetra_dynamic > 0:
            # rest_volume = sim.rest_volume[0]
            # current_volume = sim.current_volume[0]
            #
            # volume_ratio = round(100.0 * current_volume / rest_volume, 2)
            # volume_ratio_str = "volume ratio(%): " + str(volume_ratio) + "%"
            # w.text(volume_ratio_str)

            num_inverted_elements_str = "# inverted elements: " + str(sim.num_inverted_elements[0])
            w.text(num_inverted_elements_str)

        # num_vt_dynamic_str = "# vt_dynamic: " + str(sim.vt_active_set_num_dynamic[0])
        # w.text(num_vt_dynamic_str)
        #
        # num_ee_static_str = "# ee_static: " + str(sim.ee_active_set_num[0])
        # w.text(num_ee_static_str)
        #
        # num_vt_static_str = "# vt_static: " + str(sim.vt_active_set_num[0])
        # w.text(num_vt_static_str)
        #
        # num_tv_static_str = "# tv_static: " + str(sim.tv_active_set_num[0])
        # w.text(num_tv_static_str)

def load_animation():
    global sim

    with open('animation/animation.json') as f:
        animation_raw = json.load(f)
    animation_raw = {int(k): v for k, v in animation_raw.items()}

    # 4 = (g_selector.num_maxCounter)
    animationDict = {(i+1):[] for i in range(4)}

    # 4 = (g_selector.num_maxCounter)
    for i in range(4):
        ic = i+1
        icAnimation = animation_raw[ic]
        listLen = len(icAnimation)
        # print(listLen)
        assert listLen % 7 == 0,str(ic)+"th Animation SETTING ERROR!! ======"

        num_animation = listLen // 7

        for a in range(num_animation) :
            animationFrag = [animation_raw[ic][k + 7*a] for k in range(7)] # [vx,vy,vz,rx,ry,rz,frame]
            animationDict[ic].append(animationFrag)

    # print(animationDict)
    sim._set_animation(animationDict, g_selector.is_selected)

while window.running:

    if LOOKAt_ORIGIN :
        camera.lookat(0.0, 0.0, 0.0)

    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.3, 0.3, 0.3))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.3, 0.3, 0.3))

    if window.get_event(ti.ui.PRESS):
        if window.event.key == 'c':
            g_selector.selection_Count_Up()

        if window.event.key == 'x':  # export selection
            print("==== Vertex EXPORT!! ====")
            g_selector.export_selection()

        if window.event.key == 'i':
            print("==== IMPORT!! ====")
            g_selector.import_selection()
            load_animation()

        if window.event.key == ' ':
            run_sim = not run_sim

        if window.event.key == 'r':
            frame_cpu = 0
            sim.reset()
            g_selector.is_selected.fill(0)
            sim.set_fixed_vertices(g_selector.is_selected)
            run_sim = False

        if window.event.key == 'v':
            sim.enable_velocity_update = not sim.enable_velocity_update
            if sim.enable_velocity_update is True:
                print("velocity update on")
            else:
                print("velocity update off")


        if window.event.key == 'g':
            sim.random_noise()

        if window.event.key == 'z':
            sim.enable_collision_handling = not sim.enable_collision_handling
            if sim.enable_collision_handling is True:
                print("collision handling on")
            else:
                print("collision handling off")

        if window.event.key == 'h':
            sim.set_fixed_vertices(g_selector.is_selected)


        if window.event.key == ti.ui.BACKSPACE:
            g_selector.is_selected.fill(0)

        if window.event.key == ti.ui.LMB:
            g_selector.LMB_mouse_pressed = True
            g_selector.mouse_click_pos[0], g_selector.mouse_click_pos[1] = window.get_cursor_pos()

        if window.event.key == ti.ui.TAB:
            g_selector.MODE_SELECTION = not g_selector.MODE_SELECTION

    if window.get_event(ti.ui.RELEASE):
        if window.event.key == ti.ui.LMB:
            g_selector.LMB_mouse_pressed = False
            g_selector.mouse_click_pos[2], g_selector.mouse_click_pos[3] = window.get_cursor_pos()
            g_selector.Select()

    if g_selector.LMB_mouse_pressed:
        g_selector.mouse_click_pos[2], g_selector.mouse_click_pos[3] = window.get_cursor_pos()
        g_selector.update_ti_rect_selection()

    if run_sim:
        sim.animate_handle(g_selector.is_selected)
        sim.forward(n_substeps=n_substep)
        frame_cpu = frame_cpu+1

    show_options()

    for mid in range(len(scene1.meshes_dynamic)):
        scene.mesh(sim.meshes_dynamic[mid].mesh.verts.x, indices=sim.meshes_dynamic[mid].face_indices, color=scene1.colors_tri_dynamic[mid] if not MODE_WIREFRAME else (0,0,0),show_wireframe=MODE_WIREFRAME)

    for mid in range(len(scene1.meshes_static)):
        scene.mesh(sim.meshes_static[mid].mesh.verts.x, indices=sim.meshes_static[mid].face_indices, color=colors_static[mid] if not MODE_WIREFRAME else (0, 0, 0), show_wireframe=MODE_WIREFRAME)

    for tid in range(len(scene1.tet_meshes_dynamic)):
        scene.mesh(sim.tet_meshes_dynamic[tid].verts.x, indices=sim.tet_meshes_dynamic[tid].face_indices, color=scene1.colors_tet_dynamic[tid] if not MODE_WIREFRAME else (0,0,0),show_wireframe=MODE_WIREFRAME)

    for pid in range(len(scene1.particles)):
        scene.particles(sim.particles[pid].x, radius=sim.particle_radius, color=(1, 0, 0))

    if mesh_export and run_sim and frame_cpu < frame_end:
        sim.export_mesh = True
        for mid in range(len(scene1.meshes_dynamic)):
            sim.meshes_dynamic[mid].export(os.path.basename(scene1.__file__), mid, frame_cpu)

        for tid in range(len(scene1.tet_meshes_dynamic)):
            sim.tet_meshes_dynamic[tid].export(os.path.basename(scene1.__file__), tid, frame_cpu)

        for pid in range(len(scene1.particles)):
            sim.particles[pid].export(os.path.basename(scene1.__file__), pid, frame_cpu)

        for sid in range(len(scene1.meshes_static)):
            sim.meshes_static[sid].export(os.path.basename(scene1.__file__), sid, frame_cpu, is_static=True)

    # scene.lines(sim.grid_vertices, indices=sim.grid_edge_indices, width=1.0, color=(0, 0, 0))
    scene.lines(sim.aabb_vertices, indices=sim.grid_edge_indices, width=1.0, color=(0, 0, 0))
    scene.mesh(sim.x_static,  indices=sim.face_indices_static, color=(0, 0, 0), show_wireframe=True)
    scene.mesh(sim.x,  indices=sim.face_indices_dynamic, color=(0, 0, 0), show_wireframe=True)

    g_selector.renderTestPos()
    scene.particles(g_selector.renderTestPosition,radius=0.01, color=(1, 0, 1))

    canvas.lines(g_selector.ti_mouse_click_pos, width=0.002, indices=g_selector.ti_mouse_click_index, color=(1, 0, 1) if g_selector.MODE_SELECTION else (0, 0, 1))

    camera.track_user_inputs(window, movement_speed=0.15, hold_key=ti.ui.RMB)
    canvas.scene(scene)
    window.show()
