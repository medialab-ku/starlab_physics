import taichi as ti
import json

from Scenes import fem_test as scene1
import os

from framework.physics import XPBFEM
from framework.utilities import selection_tool as st

# sim = XPBF.Solver(scene1.particles_dy, g=ti.math.vec3(0.0, -7., 0.0), dt=0.020)
sim = XPBFEM.Solver(scene1.msh_mesh_dy, g=ti.math.vec3(0.0, -9.81, 0.0), dt=0.020)
window = ti.ui.Window("XPBD framework", (1024, 768), fps_limit=200)
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
#selector
# g_selector = st.SelectionTool(sim.max_num_verts_dy, sim.mesh_dy.verts.x, window, camera)

n_substep = 5
frame_end = 100

dt_ui = sim.dt
solver_type_ui = sim.solver_type
dHat_ui = sim.dHat
damping_ui = sim.damping

YM_ui = sim.YM
PR_ui = sim.PR

mesh_export = False
frame_cpu = 0

def show_options():

    global n_substep
    global dt_ui
    global solver_type_ui
    global damping_ui
    global sim
    global dHat_ui
    global YM_ui
    global PR_ui
    global MODE_WIREFRAME
    global LOOKAt_ORIGIN
    global mesh_export
    global frame_end

    old_dt = dt_ui
    old_solver_type_ui = solver_type_ui
    old_dHat = dHat_ui
    old_damping = damping_ui
    YM_old = YM_ui
    PR_old = PR_ui

    with gui.sub_window("XPBD Settings", 0., 0., 0.4, 0.35) as w:

        solver_type_ui = w.slider_int("solver type", solver_type_ui, 0, 1)
        if solver_type_ui == 0:
            w.text("solver type: XPBD Jacobi")
        elif solver_type_ui == 1:
            w.text("solver type: PD diag")

        dt_ui = w.slider_float("Time Step Size", dt_ui, 0.001, 0.101)
        n_substep = w.slider_int("# Substepping", n_substep, 1, 100)
        dHat_ui = w.slider_float("dHat", dHat_ui, 0.001, 0.101)
        damping_ui = w.slider_float("Damping Ratio", damping_ui, 0.0, 1.0)
        YM_ui = w.slider_float("Young's Modulus", YM_ui, 0.0, 1e8)
        PR_ui = w.slider_float("Poisson's Ratio", PR_ui, 0.0, 1e8)

        frame_str = "# frame: " + str(frame_cpu)
        w.text(frame_str)

        LOOKAt_ORIGIN = w.checkbox("Look at origin", LOOKAt_ORIGIN)
        # sim.enable_velocity_update = w.checkbox("velocity constraint", sim.enable_velocity_update)
        # sim.enable_collision_handling = w.checkbox("handle collisions", sim.enable_collision_handling)
        # mesh_export = w.checkbox("export mesh", mesh_export)

        # if mesh_export is True:
        #     frame_end = w.slider_int("end frame", frame_end, 1, 2000)

        w.text("stats.")
        verts_str = "# verts: " + str(sim.num_verts)
        w.text(verts_str)
        tets_str = "# tets: " + str(sim.num_tets)
        w.text(tets_str)
        # w.text("")
        # particles_st_str = "# static particles: " + str(sim.num_particles - sim.num_particles_dy)
        # w.text(particles_st_str)



    if not old_dt == dt_ui:
        sim.dt = dt_ui

    if not old_dHat == dHat_ui:
        sim.particle_rad = dHat_ui

    if not old_solver_type_ui == solver_type_ui:
        sim.solver_type = solver_type_ui

    # if not old_friction_coeff == friction_coeff_ui:
    #     sim.mu = friction_coeff_ui
    #
    if not YM_old == YM_ui:
        sim.YM = YM_ui

    if not PR_old == PR_ui:
        sim.PR = PR_ui

    if not old_damping == damping_ui:
        sim.damping = damping_ui

def load_animation():
    global sim

    with open('framework/animation/animation.json') as f:
        animation_raw = json.load(f)
    animation_raw = {int(k): v for k, v in animation_raw.items()}

    animationDict = {(i+1):[] for i in range(4)}

    for i in range(4):
        ic = i + 1
        icAnimation = animation_raw[ic]
        listLen = len(icAnimation)
        # print(listLen)
        assert listLen % 7 == 0, str(ic) + "th Animation SETTING ERROR!! ======"

        num_animation = listLen // 7

        for a in range(num_animation) :
            animationFrag = [animation_raw[ic][k + 7*a] for k in range(7)] # [vx,vy,vz,rx,ry,rz,frame]
            animationDict[ic].append(animationFrag)

while window.running:

    if LOOKAt_ORIGIN:
        camera.lookat(0, 0, 0)

    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.3, 0.3, 0.3))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.3, 0.3, 0.3))

    if window.get_event(ti.ui.PRESS):

        # if window.event.key == 'x':  # export selection
        #     print("==== Vertex EXPORT!! ====")
        #     g_selector.export_selection()
        #
        # if window.event.key == 'i':
        #     print("==== IMPORT!! ====")
        #     g_selector.import_selection()
        #     sim.set_fixed_vertices(g_selector.is_selected)
        #     # load_animation()
        #
        # if window.event.key == 't':
        #     g_selector.sewing_selection()
        #
        # if window.event.key == 'y':
        #     g_selector.pop_sewing()

        # if window.event.key == 'u':
        #     g_selector.remove_all_sewing()

        if window.event.key == ' ':
            run_sim = not run_sim

        if window.event.key == 'r':
            frame_cpu = 0
            sim.reset()
            # g_selector.is_selected.fill(0.0)
            # sim.set_fixed_vertices(g_selector.is_selected)
            run_sim = False

        # if window.event.key == 'v':
        #     sim.enable_velocity_update = not sim.enable_velocity_update
        #     if sim.enable_velocity_update is True:
        #         print("velocity update on")
        #     else:
        #         print("velocity update off")
        #
        # if window.event.key == 'z':
        #     sim.enable_collision_handling = not sim.enable_collision_handling
        #     if sim.enable_collision_handling is True:
        #         print("collision handling on")
        #     else:
        #         print("collision handling off")

        # if window.event.key == 'h':
        #     print("fix vertices")
        #     sim.set_fixed_vertices(g_selector.is_selected)
        #
        # if window.event.key == ti.ui.BACKSPACE:
        #     g_selector.is_selected.fill(0)
        #
        # if window.event.key == ti.ui.LMB:
        #     g_selector.LMB_mouse_pressed = True
        #     g_selector.mouse_click_pos[0], g_selector.mouse_click_pos[1] = window.get_cursor_pos()
        #
        # if window.event.key == ti.ui.TAB:
        #     g_selector.MODE_SELECTION = not g_selector.MODE_SELECTION


    # if window.get_event(ti.ui.RELEASE):
    #     if window.event.key == ti.ui.LMB:
    #         g_selector.LMB_mouse_pressed = False
    #         g_selector.mouse_click_pos[2], g_selector.mouse_click_pos[3] = window.get_cursor_pos()
    #         g_selector.Select()
    #
    # if g_selector.LMB_mouse_pressed:
    #     g_selector.mouse_click_pos[2], g_selector.mouse_click_pos[3] = window.get_cursor_pos()
    #     g_selector.update_ti_rect_selection()

    if run_sim:
        # sim.animate_handle(g_selector.is_selected)
        sim.forward(n_substeps=n_substep)
        frame_cpu += 1

    show_options()

    # if mesh_export and run_sim and frame_cpu < frame_end:
    #     sim.mesh_dy.export(os.path.basename(scene1.__file__), frame_cpu)

    # scene.mesh(sim.mesh_dy.verts.x,  indices=sim.mesh_dy.face_indices, per_vertex_color=sim.mesh_dy.colors)
    # scene.mesh(sim.mesh_dy.verts.x, indices=sim.mesh_dy.face_indices, color=(0, 0.0, 0.0), show_wireframe=True)
    #
    # if sim.mesh_st != None:
    #     scene.mesh(sim.mesh_st.verts.x, indices=sim.mesh_st.face_indices, color=(0, 0.0, 0.0), show_wireframe=True)
    #     scene.mesh(sim.mesh_st.verts.x, indices=sim.mesh_st.face_indices, color=(1, 1.0, 1.0))

    # g_selector.renderTestPos()

    #draw selected particles
    # scene.particles(g_selector.renderTestPosition, radius=0.01, color=(1, 0, 1))
    # scene.particles(g_selector.renderTestPosition, radius=0.01, color=(1, 0, 1))

    # if run_sim :
    #     for i in range(sim.particle.num_sets):
    #         sim.particle.export(os.path.basename(scene1.__file__),i,frame_cpu)

    # scene.particles(sim.x, radius=sim.padding, color=(1.0, 0.0, 0.0))
    scene.mesh(sim.x, indices=sim.faces, per_vertex_color=sim.tet_mesh.color)
    scene.mesh(sim.x, indices=sim.faces, color=(0.0, 0.0, 0.0), show_wireframe=True)
    scene.lines(sim.aabb_x0, indices=sim.aabb_index0, width=1.0, color=(0.0, 0.0, 0.0))
    camera.track_user_inputs(window, movement_speed=0.8, hold_key=ti.ui.RMB)
    canvas.scene(scene)
    window.show()