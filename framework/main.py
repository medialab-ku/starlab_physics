import taichi as ti
from Scenes import test_fem as scene1

import selection_tool as st

sim = XPBD.Solver(scene1.meshes_dynamic, scene1.meshes_static, scene1.tet_meshes_dynamic, scene1.particles, g=ti.math.vec3(0.0, -9.81, 0.0), dt=0.03, grid_size=ti.math.vec3(3.5, 3.5, 3.5), particle_radius=0.02, dHat=1e-3)
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

run_sim = True

#selector
g_selector = st.SelectionTool(sim.max_num_verts_dynamic,sim.x,window,camera)
print("sim.max_num_verts_dynamic", sim.max_num_verts_dynamic)

n_substep = 20
dt_ui = sim.dt[0]
dHat_ui = sim.dHat[0]

def show_options():
    global n_substep
    global dt_ui
    global sim
    global dHat_ui

    old_dt = dt_ui
    old_dHat = dHat_ui
    with gui.sub_window("Time Step", 0.05, 0.1, 0.2, 0.15) as w:
        # dt_ui = w.slider_float("dt", dt_ui, 0.0, 0.1)
        dt_ui = w.slider_float("dt", dt_ui, 0.001, 0.101)

        n_substep = w.slider_int("substeps", n_substep, 1, 100)
        dHat_ui = w.slider_float("dHat", dHat_ui, 0.0001, 0.0101)

    if not old_dt == dt_ui :
        # sim.dt[0] = dt_ui if dt_ui > 0.00001 else 0.00001
        sim.dt[0] = dt_ui

    if not old_dHat == dHat_ui:
        sim.dHat[0] = dHat_ui


while window.running:

    camera.lookat(0.0, 0.0, 0.0)
    scene.set_camera(camera)
    scene.ambient_light((0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.3, 0.3, 0.3))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.3, 0.3, 0.3))

    if window.get_event(ti.ui.PRESS):
        if window.event.key == ' ':
            run_sim = not run_sim

        if window.event.key == 'r':
            sim.reset()
            run_sim = False

        if window.event.key == 'v':
            sim.enable_velocity_update = not sim.enable_velocity_update
            if sim.enable_velocity_update is True:
                print("enable velocity update")
            else:
                print("disable velocity update")

        if window.event.key == 'h':
            sim.set_fixed_vertices(g_selector.is_selected)

        if window.event.key == ti.ui.LMB:
            g_selector.LMB_mouse_pressed = True
            g_selector.mouse_click_pos[0], g_selector.mouse_click_pos[1] = window.get_cursor_pos()

        if window.event.key == ti.ui.RMB:
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
        sim.forward(n_substeps=n_substep)

    show_options()

    for mid in range(len(scene1.meshes_dynamic)):
        scene.mesh(sim.meshes_dynamic[mid].mesh.verts.x, indices=sim.meshes_dynamic[mid].face_indices, color=scene1.colors_tri_dynamic[mid])

    for tid in range(len(scene1.tet_meshes_dynamic)):
        scene.mesh(sim.tet_meshes_dynamic[tid].verts.x, indices=sim.tet_meshes_dynamic[tid].face_indices, color=scene1.colors_tet_dynamic[tid])

    for pid in range(len(scene1.particles)):
        scene.particles(sim.particles[pid].x, radius=sim.particle_radius, color=(1, 0, 0))

    for mid in range(len(scene1.meshes_static)):
        scene.mesh(sim.meshes_static[mid].mesh.verts.x, indices=sim.meshes_static[mid].face_indices, color=colors_static[mid])


    scene.lines(sim.grid_vertices, indices=sim.grid_edge_indices, width=1.0, color=(0, 0, 0))
    scene.mesh(sim.x_static,  indices=sim.face_indices_static, color=(0, 0, 0), show_wireframe=True)
    scene.mesh(sim.x,  indices=sim.face_indices_dynamic, color=(0, 0, 0), show_wireframe=True)

    g_selector.renderTestPos()
    scene.particles(g_selector.renderTestPosition,radius=0.01, color=(1, 0, 1))

    canvas.lines(g_selector.ti_mouse_click_pos, width=0.002, indices=g_selector.ti_mouse_click_index, color=(1, 0, 1) if g_selector.MODE_SELECTION else (0, 0, 1))

    camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
    canvas.scene(scene)
    window.show()