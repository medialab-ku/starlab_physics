import taichi as ti
import json

from Scenes import fluid_test as scene1
import os

from Scenes.fluid_test import particles
from framework.physics import XPBF
from framework.utilities import selection_tool as st
from framework.collision import SpatialHash as shash

sh = shash.SpatialHash(grid_resolution=(64, 64, 64))
sim = XPBF.Solver(scene1.particles, g=ti.math.vec3(0.0, -9.81, 0.0), dt=0.020, sh=sh)

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
#selector
# g_selector = st.SelectionTool(sim.num_particles_dy, sim.x, window, camera)

n_substep = 20
frame_end = 100

dt_ui = 0.01
solver_type_ui = 1
g_ui = -9.81

dt_ui = sim.dt
solver_type_ui = sim.solver_type
g_ui = sim.g[1]


dHat_ui = sim.particle_rad
dHat_ui = sim.particle_rad

damping_ui = sim.damping
YM_ui = sim.YM
PR_ui = sim.PR
ZE_ui = sim.ZE
mesh_export = False
frame_cpu = 0

def show_options():

    global n_substep
    global dt_ui
    global solver_type_ui
    global g_ui
    global damping_ui
    global sim
    global dHat_ui
    global YM_ui
    global PR_ui
    global ZE_ui
    global MODE_WIREFRAME
    global LOOKAt_ORIGIN
    global mesh_export
    global frame_end

    old_dt = dt_ui
    old_solver_type_ui = solver_type_ui
    old_g = g_ui
    old_dHat = dHat_ui
    old_damping = damping_ui
    YM_old = YM_ui
    PR_old = PR_ui
    ZE_old = ZE_ui

    with gui.sub_window("XPBD Settings", 0., 0., 0.3, 0.7) as w:

        solver_type_ui = w.slider_int("solver type", solver_type_ui, 0, 2)
        if solver_type_ui == 0:
            w.text("solver type: XPBD Jacobi")
        elif solver_type_ui == 1:
            w.text("solver type: PD diag")
        elif solver_type_ui == 2:
            w.text("solver type: Kugelstadt 21)")

        dt_ui = w.slider_float("dt", dt_ui, 0.001, 0.101)
        g_ui = w.slider_float("g", g_ui, -20.0, 20.0)
        n_substep = w.slider_int("# sub", n_substep, 1, 100)
        dHat_ui = w.slider_float("particle rad.", dHat_ui, 0.001, 0.101)
        damping_ui = w.slider_float("damping", damping_ui, 0.0, 1.0)

        YM_ui = w.slider_int("Young's Modulus", YM_ui, -1, 5)
        PR_ui = w.slider_float("Poisson's Ratio", PR_ui, 0.0, 0.49)
        ZE_ui = w.slider_float("Zero Energy", ZE_ui, 0.0, 1e8)

        frame_str = "# frame: " + str(frame_cpu)
        w.text(frame_str)

        LOOKAt_ORIGIN = w.checkbox("Look at origin", LOOKAt_ORIGIN)
        # sim.enable_velocity_update = w.checkbox("velocity constraint", sim.enable_velocity_update)
        # sim.enable_collision_handling = w.checkbox("handle collisions", sim.enable_collision_handling)
        # mesh_export = w.checkbox("export mesh", mesh_export)

        # if mesh_export is True:
        #     frame_end = w.slider_int("end frame", frame_end, 1, 2000)

        w.text("")
        particles_dy_str = "# dynamic particles: " + str(sim.num_particles_dy)
        w.text(particles_dy_str)

        w.text("")
        particles_st_str = "# static particles: " + str(sim.num_particles - sim.num_particles_dy)
        w.text(particles_st_str)

    if not old_solver_type_ui == solver_type_ui:
        sim.solver_type = solver_type_ui

    if not old_g == g_ui:
        sim.g[1] = g_ui

    if not old_dt == dt_ui:
        sim.dt = dt_ui

    if not old_dHat == dHat_ui:
        sim.particle_rad = dHat_ui

    # if not old_friction_coeff == friction_coeff_ui:
    #     sim.mu = friction_coeff_ui

    if not YM_old == YM_ui:
        sim.YM = YM_ui

    if not PR_old == PR_ui:
        sim.PR = PR_ui

    if not ZE_old == ZE_ui:
        sim.ZE = PR_ui

    # if not YM_old == YM_ui:
    #     sim.stiffness_bending = YM_ui
    #
    # if not YM_b_old == YM_b_ui:
    #     sim.stiffness_stretch = YM_b_ui

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

        if window.event.key == 'x':  # export selection
            print("==== Vertex EXPORT!! ====")
            # g_selector.export_selection()

        if window.event.key == 'i':
            print("==== IMPORT!! ====")
            # g_selector.import_selection()
            # sim.set_fixed_vertices(g_selector.is_selected)
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

        if window.event.key == 't':
            sim.randomize()
            run_sim = False

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
    #     if window.event.key == ti.ui.LMB:
    #         g_selector.LMB_mouse_pressed = True
    #         g_selector.mouse_click_pos[0], g_selector.mouse_click_pos[1] = window.get_cursor_pos()
    #
    #     if window.event.key == ti.ui.TAB:
    #         g_selector.MODE_SELECTION = not g_selector.MODE_SELECTION
    #
    #
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
        sim.forward(n_substeps=1, n_iter=n_substep)
        frame_cpu += 1

    # show_options()

    # if mesh_export and run_sim and frame_cpu < frame_end:
    #     sim.mesh_dy.export(os.path.basename(scene1.__file__), frame_cpu)
    # draw selected particles
    # scene.particles(g_selector.renderTestPosition, radius=sim.particle_rad, color=(1, 0, 1))
    # scene.particles(g_selector.renderTestPosition, radius=0.01, color=(1, 0, 1))

    # if run_sim :
    #     for i in range(sim.particle.num_sets):
    #         sim.particle.export(os.path.basename(scene1.__file__),i,frame_cpu)

    scene.particles(sim.particle.x, radius=sim.particle_rad, per_vertex_color=sim.particle.color)
    # scene.particles
    # scene.lines(sim.aabb_x0, indices=sim.aabb_index0, width=1.0, color=(0.0, 0.0, 0.0))

    # g_selector.renderTestPos()
    # scene.particles(g_selector.renderTestPosition, radius=sim.particle_rad, color=(1, 0, 0))
    # canvas.lines(g_selector.ti_mouse_click_pos, width=0.002, indices=g_selector.ti_mouse_click_index, color=(1, 0, 1) if g_selector.MODE_SELECTION else (0, 0, 1))
    # scene.particles(particles.x0, radius=0.2, color=(1, 0, 0))
    scene.lines(sh.bbox_vertices, width=1.0, indices=sh.bbox_edge_indices_flattened, color=(0, 0, 0))
    camera.track_user_inputs(window, movement_speed=0.8, hold_key=ti.ui.RMB)
    canvas.scene(scene)
    window.show()