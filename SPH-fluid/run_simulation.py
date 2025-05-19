import os
import argparse
import taichi as ti
import numpy as np
from config_builder import SimConfig
from particle_system import ParticleSystem
from export_mesh import Exporter

import open3d as o3d

ti.init(arch=ti.gpu, device_memory_fraction=0.5)

def mesh_to_filled_particles(mesh_path, voxel_size=0.05):
    mesh_legacy = o3d.io.read_triangle_mesh(mesh_path)
    mesh_legacy.compute_vertex_normals()
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)

    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh)

    aabb = mesh_legacy.get_axis_aligned_bounding_box()
    min_bound = aabb.min_bound
    max_bound = aabb.max_bound

    x = np.arange(min_bound[0], max_bound[0], voxel_size)
    y = np.arange(min_bound[1], max_bound[1], voxel_size)
    z = np.arange(min_bound[2], max_bound[2], voxel_size)
    grid = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1).reshape(-1, 3)

    query_points = o3d.core.Tensor(grid, dtype=o3d.core.Dtype.Float32)

    occupancy = scene.compute_occupancy(query_points)
    inside_mask = occupancy.numpy() == 1
    inside_points = query_points[inside_mask]

    return inside_points.numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SPH Taichi')
    parser.add_argument('--scene_file',
                        default='',
                        help='scene file')
    args = parser.parse_args()
    scene_path = args.scene_file
    config = SimConfig(scene_file_path=scene_path)
    scene_name = scene_path.split("/")[-1].split(".")[0]

    substeps = config.get_cfg("numberOfStepsPerRenderUpdate")
    output_frames = config.get_cfg("exportFrame")
    output_interval = int(0.016 / config.get_cfg("timeStepSize"))
    output_ply = config.get_cfg("exportPly")
    output_obj = config.get_cfg("exportObj")
    output_vtk = config.get_cfg("exportVtk")
    series_prefix = "{}_output/particle_object_{}.ply".format(scene_name, "{}")
    if output_frames:
        os.makedirs(f"{scene_name}_output_img", exist_ok=True)
    if output_ply:
        os.makedirs(f"{scene_name}_output_ply", exist_ok=True)
    if output_obj:
        os.makedirs(f"{scene_name}_output_obj", exist_ok=True)
    if output_vtk:
        os.makedirs(f"{scene_name}_output_vtk", exist_ok=True)

    ps = ParticleSystem(config, GGUI=True)
    solver = ps.build_solver()
    solver.initialize()

    exporter = Exporter(f"{scene_name}_output_vtk", f"{scene_name}_output_obj", f"{scene_name}_output_ply", frameInterval=output_interval)

    window = ti.ui.Window('SPH', (1024, 1024), show_window = True, vsync=False)
    gui = window.get_gui()

    method = config.get_cfg("simulationMethod")

    frame_cnt = 0
    cnt_ply = 0
    runSim = False

    stop_frame = False
    end_frame = 1000

    def show_options():
        global ps
        global frame_cnt
        global solver
        global method
        global end_frame
        global stop_frame
        global output_obj
        global output_ply
        global output_vtk

        # global use_gn
        # global eta
        # global pbf_num_iters
        # global mass_ratio
        # global use_heatmap
        # global solver_type
        # global dt_ui
        # global g_ui
        # global damping_ui
        # global YM_ui
        # global PR

        # old_dHat = dHat_ui
        # old_damping = damping_ui
        # YM_old = YM_ui
        # PR_old = PR
        # mass_ratio_old = mass_ratio

        with gui.sub_window("Settings", 0., 0., 0.4, 0.31) as w:
            # dt_ui = w.slider_float("dt", dt_ui, 0.001, 0.101)
            ps.eta = w.slider_float("eta ", ps.eta, 0.0, 1.0)
            solver.viscosity = w.slider_float("viscosity", solver.viscosity, 0.0, 1.0)

            if method == 5:
                solver.tol = w.slider_float("opt tol", solver.tol, 0.0, 1.0)
                solver.k_rho = w.slider_float("k rho", solver.k_rho, 0.0, 1e6)
                solver.da_ratio = w.slider_float("da_ratio", solver.da_ratio, 0.0, 2.0)
                solver.use_gn = w.checkbox("use gn", solver.use_gn)
                solver.use_div = w.checkbox("use div", solver.use_div)

            # pbf_num_iters = w.slider_int("# iter", pbf_num_iters, 1, 100)
            # solver_type = w.slider_int("solver type", solver_type, 0, 2)


            stop_frame = w.checkbox("stop frame ?", stop_frame)
            output_ply = w.checkbox("output_ply", output_ply)
            output_obj = w.checkbox("output_obj", output_obj)
            output_vtk = w.checkbox("output_vtk", output_vtk)

            if stop_frame:
                end_frame = w.slider_int("end frame", end_frame, 0, int(1e5))

            gui.text(f"# particle: {ps.particle_num}")
            gui.text(f"Current frame: {frame_cnt}")

    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    camera.position(5.5, 2.5, 4.0)
    camera.up(0.0, 1.0, 0.0)
    camera.lookat(-1.0, 0.0, 0.0)
    camera.fov(70)
    scene.set_camera(camera)

    canvas = window.get_canvas()
    radius = 0.002
    movement_speed = 0.02
    background_color = (0, 0, 0)  # 0xFFFFFF
    particle_color = (1, 1, 1)

    # Invisible objects
    invisible_objects = config.get_cfg("invisibleObjects")
    if not invisible_objects:
        invisible_objects = []

    # Draw the lines for domain
    x_max, y_max, z_max = config.get_cfg("domainEnd")
    box_anchors = ti.Vector.field(3, dtype=ti.f32, shape = 8)
    box_anchors[0] = ti.Vector([0.0, 0.0, 0.0])
    box_anchors[1] = ti.Vector([0.0, y_max, 0.0])
    box_anchors[2] = ti.Vector([x_max, 0.0, 0.0])
    box_anchors[3] = ti.Vector([x_max, y_max, 0.0])

    box_anchors[4] = ti.Vector([0.0, 0.0, z_max])
    box_anchors[5] = ti.Vector([0.0, y_max, z_max])
    box_anchors[6] = ti.Vector([x_max, 0.0, z_max])
    box_anchors[7] = ti.Vector([x_max, y_max, z_max])

    box_lines_indices = ti.field(int, shape=(2 * 12))

    for i, val in enumerate([0, 1, 0, 2, 1, 3, 2, 3, 4, 5, 4, 6, 5, 7, 6, 7, 0, 4, 1, 5, 2, 6, 3, 7]):
        box_lines_indices[i] = val

    if ps.cfg.get_cfg("simulationMethod") == 5:
        solver.build_static_LBVH()

    mesh_path= "./data/models/cube.obj"
    points = mesh_to_filled_particles(mesh_path, voxel_size=0.2)
    n_particles = points.shape[0]
    mesh_particles = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
    color = ti.Vector.field(3, dtype=ti.f32, shape=n_particles)
    mesh_particles.from_numpy(points)

    @ti.kernel
    def init_color():
        for i in mesh_particles:
            color[i] = ti.Vector([0.2, 0.6, 1.0])  # 밝은 파란색

    init_color()

    while window.running:

        show_options()

        if window.get_event(ti.ui.PRESS):
            if window.event.key == ' ':
                runSim = not runSim
                # print("test")

            if window.event.key == 'r':
                frame_cnt = 0
                ps.x.copy_from(ps.x_0)
                ps.v.fill(0.0)
                solver.initialize()
                if ps.cfg.get_cfg("simulationMethod") == 5:
                    solver.build_static_LBVH()
                runSim = False

        if stop_frame and frame_cnt > end_frame:
            runSim = False

        # for i in range(substeps):
        if runSim:
            # print(runSim)
            for i in range(substeps):
               optIter, pcgIter_total = solver.step()
               print(optIter, pcgIter_total)
            frame_cnt += 1

        ps.copy_to_vis_buffer(invisible_objects=invisible_objects)
        if ps.dim == 2:
            canvas.set_background_color(background_color)
            canvas.circles(ps.x_vis_buffer, radius=ps.particle_radius, color=particle_color)
        elif ps.dim == 3:
            camera.track_user_inputs(window, movement_speed=movement_speed, hold_key=ti.ui.LMB)
            scene.set_camera(camera)
            scene.point_light((2.0, 2.0, 2.0), color=(1.0, 1.0, 1.0))
            scene.particles(ps.x_vis_buffer, radius=ps.particle_radius, per_vertex_color=ps.color_vis_buffer)


            scene.particles(ps.xTmp, radius=ps.particle_radius, per_vertex_color=ps.color_vis_buffer)
            scene.lines(box_anchors, indices=box_lines_indices, color = (0.99, 0.68, 0.28), width = 1.0)
            canvas.scene(scene)
            # solver.LBVH.draw_bvh_aabb_test(scene,  solver.LBVH.num_leafs)
            # solver.LBVH.draw_bvh_aabb_test(scene,  2)

            if ps.num_static_vertices > 0:
                # scene.mesh(ps.x_st, ps.faces_st, color=(1.5, 1.0, 0.0))
                # scene.mesh(ps.x_st, ps.faces_st, color=(1.0, 1.0, 1.0), show_wireframe= True)
                scene.lines(vertices=ps.x_st, indices=ps.edges_st, color=(1.0, 1.0, 1.0), width=1.0)
                # scene.lines(vertices=solver.LBVH.pos, indices=solver.LBVH.code_edge, color=(1.0, 0.0, 0.0), width=1.0)

        if output_frames:
            if frame_cnt % output_interval == 0:
                window.write_image(f"{scene_name}_output_img/{frame_cnt:06}.png")

        if output_obj:
            exporter.export_ply("scene.obj", ps.x, MODE="MULTI")
        if output_vtk:
            exporter.export_vtk("scene.vtk", ps.x, MODE="MULTI")
        if output_ply:
            exporter.export_ply("scene.ply", ps.x, MODE="MULTI")

        scene.particles(mesh_particles, radius=0.02, per_vertex_color=color)
        
        # if frame_cnt % output_interval == 0:
        #     if output_ply:
        #         obj_id = 0
        #         obj_data = ps.dump(obj_id=obj_id)
        #         np_pos = obj_data["position"]
        #         writer = ti.tools.PLYWriter(num_vertices=ps.object_collection[obj_id]["particleNum"])
        #         writer.add_vertex_pos(np_pos[:, 0], np_pos[:, 1], np_pos[:, 2])
        #         writer.export_frame_ascii(cnt_ply, series_prefix.format(0))
        #     if output_obj:
        #         for r_body_id in ps.object_id_rigid_body:
        #             with open(f"{scene_name}_output/obj_{r_body_id}_{cnt_ply:06}.obj", "w") as f:
        #                 e = ps.object_collection[r_body_id]["mesh"].export(file_type='obj')
        #                 f.write(e)
        #     cnt_ply += 1

        # if cnt > 6000:
        #     break
        window.show()