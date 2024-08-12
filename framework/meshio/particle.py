import taichi as ti
import meshio
import numpy as np
import random

import os

@ti.data_oriented
class Particle:

    def __init__(self,
                 model_dir,
                 model_names=[],
                 translations=[],
                 scales=[],
                 rotations=[],
                 is_static=[],
                 radius=0.01):

        num_sets = len(model_names)
        self.num_particles = 0

        self.offsets = [0]
        points = np.empty((0, 3))
        m_inv_np = np.empty((0))
        # print(points.shape)
        # points.reshape(0, 3)
        self.num_static = 0
        for i in range(num_sets):
            model_path = model_dir + "/" + model_names[i]
            p = meshio.read(model_path)
            pos_temp = np.array(p.points, dtype=np.float32)
            scale = lambda x, sc: sc * x
            translate = lambda x, trans: x + trans

            center = pos_temp.sum(axis=0) / pos_temp.shape[0]

            pos_temp = np.apply_along_axis(lambda row: translate(row, -center), 1, pos_temp)
            pos_temp = scale(pos_temp, scales[i])
            pos_temp = np.apply_along_axis(lambda row: translate(row, translations[i]), 1, pos_temp)

            if is_static[i] is True:
                m_inv_temp = np.zeros(pos_temp.shape[0])
                self.num_static += pos_temp.shape[0]
            else:
                m_inv_temp = np.ones(pos_temp.shape[0])

            points = np.append(points, pos_temp, axis=0)
            m_inv_np = np.append(m_inv_np, m_inv_temp, axis=0)

            self.num_particles += pos_temp.shape[0]
            self.offsets.append(self.num_particles)

        self.num_dynamic = self.num_particles - self.num_static
        # print(self.num_static)
        self.x0 = ti.Vector.field(n=3, dtype=float)
        self.y = ti.Vector.field(n=3, dtype=float)
        self.dx = ti.Vector.field(n=3, dtype=float)
        self.x = ti.Vector.field(n=3, dtype=float)
        self.c_den = ti.Vector.field(n=3, dtype=float)
        self.ld_den = ti.Vector.field(n=3, dtype=float)
        self.v = ti.Vector.field(n=3, dtype=float)
        self.m_inv = ti.field(dtype=float)
        self.color = ti.Vector.field(n=3, dtype=float)

        particle_snode = ti.root.dense(ti.i, self.num_particles).place(self.x0, self.y, self.dx, self.x, self.v, self.m_inv, self.color)
        particle_snode.place(self.c_den, self.ld_den)


        self.x.from_numpy(points)
        self.m_inv.from_numpy(m_inv_np)
        # self.m_inv.fill(-1.0)
        self.v.fill(0.0)
        self.x0.copy_from(self.x)

        self.init_color(is_static=is_static)
        self.radius = radius
        self.x0.copy_from(self.x)

    def reset(self):
        self.x.copy_from(self.x0)
        self.v.fill(0.)

    def init_color(self, is_static):
        # print(self.offsets)
        for i in range(len(self.offsets) - 1):
            size = self.offsets[i + 1] - self.offsets[i]
            if is_static[i] is True:
                self.init_colors(self.offsets[i], size, color=ti.math.vec3(0.5, 0.5, 0.5))
            else:
                r = random.randrange(0, 255) / 256
                g = random.randrange(0, 255) / 256
                b = random.randrange(0, 255) / 256

                self.init_colors(self.offsets[i], size, color=ti.math.vec3(r, g, b))

    @ti.kernel
    def init_colors(self, offset: ti.i32, size: ti.i32, color: ti.math.vec3):

        for i in range(size):
            self.color[i + offset] = color

    # @ti.kernel
    # def setCenterToOrigin(self):
    #
    #     center = ti.math.vec3(0, 0, 0)
    #     for i in range(self.num_particles):
    #         center += self.x[i]
    #
    #     center /= self.num_particles
    #     for i in range(self.num_particles):
    #         self.x[i] -= center
    #
    # @ti.kernel
    # def applyTransform(self):
    #     for i in range(self.num_particles):
    #         self.x[i] *= self.scale
    #
    #
    #     for i in range(self.num_particles):
    #         v_4d = ti.Vector([self.x[i][0], self.x[i][1], self.x[i][2], 1])
    #         rot_rad = ti.math.radians(self.rot)
    #         rv = ti.math.rotation3d(rot_rad[0], rot_rad[1], rot_rad[2]) @ v_4d
    #         self.x[i] = ti.Vector([rv[0], rv[1], rv[2]])
    #
    #     for i in range(self.num_particles):
    #         self.x[i] += self.trans

    # def export(self, scene_name, mesh_id, frame):
    #     directory = os.path.join("results/", scene_name, "Particle_ID_" + str(mesh_id))
    #
    #     try :
    #         if not os.path.exists(directory):
    #             os.makedirs(directory)
    #     except OSError:
    #         print("Error: Failed to create folder" + directory)
    #
    #     x_np = self.x.to_numpy()
    #     print(x_np.shape)
    #     file_name = "Particle_vtk_" + str(frame) + ".vtk"
    #     file_path = os.path.join(directory, file_name)
    #
    #     print("exporting ", file_path.__str__())
    #     meshio.write_points_cells(file_path,x_np,self.vtkcells)
    #     print("done")
