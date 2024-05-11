import taichi as ti
import json
# from Scenes import test_fem as scene1
from Scenes import cloth_swing as scene1
# from Scenes import scene_cylinder_crossing as scene1
# from Scenes import scene_cylinder_crossing_4 as scene1
# from Scenes import scene_thin_shell_twist as scene1
# from Scenes import scene_cube_stretch as scene1
# from Scenes import scene_fluid_compression as scene1

import XPBD
import selection_tool as st

sim = XPBD.Solver(scene1.meshes_dynamic, scene1.meshes_static, scene1.tet_meshes_dynamic, scene1.particles, g=ti.math.vec3(0.0, -9.81, 0.0), dt=0.03, grid_size=ti.math.vec3(5., 5., 5.), particle_radius=0.02, dHat=6e-3)
# sim = XPBD.Solver(scene1.meshes_dynamic, scene1.meshes_static, scene1.tet_meshes_dynamic, scene1.particles, g=scene1.gravity, dt=scene1.dt, grid_size=scene1.grid_size, particle_radius=scene1.particle_radius, dHat=scene1.dHat)

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
colors_static.append((0, 0.5, 0))
colors_static.append((0, 0.5, 0))

run_sim = False

MODE_WIREFRAME = False
LOOKAt_ORIGIN = True

#selector
g_selector = st.SelectionTool(sim.max_num_verts_dynamic, sim.x, window, camera)
print("sim.max_num_verts_dynamic", sim.max_num_verts_dynamic)

n_substep = 20
dt_ui = sim.dt[0]
dHat_ui = sim.dHat[0]
friction_coeff_ui = sim.friction_coeff[0]

frame_cpu = 0

def show_options():
    global n_substep
    global dt_ui
    global sim
    global dHat_ui
    global friction_coeff_ui
    global MODE_WIREFRAME
    global LOOKAt_ORIGIN

    old_dt = dt_ui
    old_dHat = dHat_ui
    with gui.sub_window("Time Step", 0.05, 0.1, 0.2, 0.3) as w:
        # dt_ui = w.slider_float("dt", dt_ui, 0.0, 0.1)
        dt_ui = w.slider_float("dt", dt_ui, 0.001, 0.101)

        n_substep = w.slider_int("substeps", n_substep, 1, 100)
        dHat_ui = w.slider_float("dHat", dHat_ui, 0.0001, 0.0101)
        friction_coeff_ui = w.slider_float("friction coeff.", friction_coeff_ui, 0.0, 1.0)


        MODE_WIREFRAME = w.checkbox("wireframe",MODE_WIREFRAME)
        LOOKAt_ORIGIN = w.checkbox("Look at origin",LOOKAt_ORIGIN)

        sim.enable_velocity_update = w.checkbox("velocity constraint",sim.enable_velocity_update)

        frame_str = "frame : " + str(frame_cpu)
        w.text(frame_str)

        # if(frame_cpu != sim.frame[0]) :
        #     print("cpu_gpu frame different!!",frame_cpu,sim.frame[0])

    if not old_dt == dt_ui :
        # sim.dt[0] = dt_ui if dt_ui > 0.00001 else 0.00001
        sim.dt[0] = dt_ui

    if not old_dHat == dHat_ui:
        sim.dHat[0] = dHat_ui

    if not friction_coeff_ui == friction_coeff_ui:
        sim.friction_coeff_ui[0] = friction_coeff_ui

def load_animation() :
    global sim

    with open('animation/animation.json') as f:
        animation_raw = json.load(f)
    animation_raw = {int(k): v for k, v in animation_raw.items()}

    # 4 = (g_selector.num_maxCounter)
    animationDict = {(i+1):[] for i in range(4)}

    # 4 = (g_selector.num_maxCounter)
    for i in range(4) :
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
    sim._set_animation(animationDict,g_selector.is_selected)

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
            print("==== EXPORT!! ====")
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


        if window.event.key == 'z':
            sim.enable_collision_handling = not sim.enable_collision_handling
            if sim.enable_collision_handling is True:
                print("collision handling on")
            else:
                print("collision handling off")

        if window.event.key == 'h':
            sim.set_fixed_vertices(g_selector.is_selected)

        if window.event.key == 'z':
            sim.frame[0] = 0
            frame_cpu = 0

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

    for tid in range(len(scene1.tet_meshes_dynamic)):
        scene.mesh(sim.tet_meshes_dynamic[tid].verts.x, indices=sim.tet_meshes_dynamic[tid].face_indices, color=scene1.colors_tet_dynamic[tid] if not MODE_WIREFRAME else (0,0,0),show_wireframe=MODE_WIREFRAME)

    for pid in range(len(scene1.particles)):
        scene.particles(sim.particles[pid].x, radius=sim.particle_radius, color=(1, 0, 0))

    # for mid in range(len(scene1.meshes_static)):
    #     scene.mesh(sim.meshes_static[mid].mesh.verts.x, indices=sim.meshes_static[mid].face_indices, color=colors_static[mid] if not MODE_WIREFRAME else (0,0,0),show_wireframe=MODE_WIREFRAME)


    scene.lines(sim.grid_vertices, indices=sim.grid_edge_indices, width=1.0, color=(0, 0, 0))
    scene.mesh(sim.x_static,  indices=sim.face_indices_static, color=(0, 0, 0), show_wireframe=True)
    scene.mesh(sim.x,  indices=sim.face_indices_dynamic, color=(0, 0, 0), show_wireframe=True)

    g_selector.renderTestPos()
    scene.particles(g_selector.renderTestPosition,radius=0.01, color=(1, 0, 1))

    canvas.lines(g_selector.ti_mouse_click_pos, width=0.002, indices=g_selector.ti_mouse_click_index, color=(1, 0, 1) if g_selector.MODE_SELECTION else (0, 0, 1))

    camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
    canvas.scene(scene)
    window.show()

# print("end frame : ", sim.frame[0])