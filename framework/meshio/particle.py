import taichi as ti
import meshio
import numpy as np
import random
from pyquaternion import Quaternion
from pathlib import Path
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
                 radius=0.01,
                 rho0=[]):

        num_sets = len(model_names)
        self.num_particles = 0

        self.offsets = [0]
        points = np.empty((0, 3))

        rest_density = []
        m_inv_np = np.empty((0))
        # m = np.empty((0))
        m_np = np.empty((0))
        type_np = np.empty((0))
        type = 0
        self.num_static = 0
        for i in range(num_sets):
            model_path = model_dir + "/" + model_names[i]
            p = meshio.read(model_path)
            # print(p.cells[0].data)
            pos_temp = np.array(p.points, dtype=np.float32)
            scale = lambda x, sc: sc * x
            translate = lambda x, trans: x + trans
            center = pos_temp.sum(axis=0) / pos_temp.shape[0]
            pos_temp = np.apply_along_axis(lambda row: translate(row, -center), 1, pos_temp)
            pos_temp = scale(pos_temp, scales[i])
            if len(rotations) > 0: # rotate mesh if it is demanded...
                rot_quaternion = Quaternion(axis=[rotations[i][0], rotations[i][1], rotations[i][2]], angle=rotations[i][3])
                rot_matrix = rot_quaternion.rotation_matrix
                for j in range(pos_temp.shape[0]):
                    pos_temp[j] = rot_matrix @ pos_temp[j]
            pos_temp = np.apply_along_axis(lambda row: translate(row, translations[i]), 1, pos_temp)

            if is_static[i] is True:
                m_inv_temp = np.zeros(pos_temp.shape[0])
                m_temp = np.zeros(pos_temp.shape[0])
                self.num_static += pos_temp.shape[0]
            else:
                m_inv_temp = (1.0 / rho0[i]) * np.ones(pos_temp.shape[0])
                m_temp = rho0[i] * np.ones(pos_temp.shape[0])

            type_temp = type * np.ones(pos_temp.shape[0])
            # m_temp = rho0[i] * np.ones(pos_temp.shape[0])
            points = np.append(points, pos_temp, axis=0)
            m_inv_np = np.append(m_inv_np, m_inv_temp, axis=0)
            m_np = np.append(m_np, m_temp, axis=0)
            type_np = np.append(type_np, type_temp, axis=0)
            type += 1
            # m_np = np.append(m_np, m_inv_temp, axis=0)

            self.num_particles += pos_temp.shape[0]

            self.offsets.append(self.num_particles)
            rest_density.extend([rho0[i]] * pos_temp.shape[0])

        rest_density = np.array(rest_density)
        density_np = np.loadtxt('smpl_particles_density.csv', delimiter=',')


        self.num_dynamic = self.num_particles - self.num_static
        # print(self.num_static)
        self.type = ti.field(dtype=int)
        self.x0 = ti.Vector.field(n=3, dtype=float)
        self.x_prev = ti.Vector.field(n=3, dtype=float)
        self.x_current = ti.Vector.field(n=3, dtype=float)
        self.y = ti.Vector.field(n=3, dtype=float)
        self.dx = ti.Vector.field(n=3, dtype=float)
        self.nc = ti.field(dtype=float)
        self.x = ti.Vector.field(n=3, dtype=float)
        self.V0 = ti.field(dtype=float)
        self.F = ti.Matrix.field(n=3, m=3, dtype=float)
        self.L = ti.Matrix.field(n=3, m=3, dtype=float)
        self.c_den = ti.Vector.field(n=3, dtype=float)
        self.ld = ti.field(dtype=float)
        self.v = ti.Vector.field(n=3, dtype=float)
        self.m_inv = ti.field(dtype=float)
        self.m = ti.field(dtype=float)
        self.is_fixed = ti.field(dtype=float)
        # self.m = ti.field(dtype=float)
        self.color = ti.Vector.field(n=3, dtype=float)
        self.rho0 = ti.field(dtype=float)
        self.heat_map = ti.Vector.field(n=3, dtype=float)
        particle_snode = ti.root.dense(ti.i, self.num_particles)
        # particle_snode.place(self.V0, self.F, self.L, self.x0)
        particle_snode.place(self.dx, self.nc)
        particle_snode.place(self.y, self.x, self.x_prev,  self.x_current, self.v, self.m_inv, self.m, self.is_fixed, self.color, self.rho0, self.heat_map)
        particle_snode.place(self.c_den, self.ld)

        self.num_particle_neighbours_rest = ti.field(dtype=int)
        self.particle_neighbours_ids_rest = ti.field(dtype=int)

        self.num_particle_neighbours = ti.field(dtype=int)
        self.particle_neighbours_ids = ti.field(dtype=int)

        particle_snode.place(self.type, self.V0, self.F, self.L, self.x0, self.num_particle_neighbours_rest)
        self.particle_cache_size = 15
        particle_snode.dense(ti.j, self.particle_cache_size).place(self.particle_neighbours_ids_rest)

        particle_snode.place(self.num_particle_neighbours)

        self.nb_cache_size = 50
        particle_snode.dense(ti.j, self.nb_cache_size).place(self.particle_neighbours_ids)

        self.type.from_numpy(type_np)
        self.x.from_numpy(points)
        self.m_inv.from_numpy(m_inv_np)
        self.m.from_numpy(m_np)
        self.is_fixed.fill(1.0)
        # self.m_inv.fill(-1.0)
        self.v.fill(0.0)
        self.x0.copy_from(self.x)

        self.init_color(is_static=is_static)
        self.radius = radius
        self.rho0.fill(1.0)
        # self.rho0.from_numpy(density_np)
        # print(self.rho0)
        self.x0.copy_from(self.x)

    def reset(self):
        self.x.copy_from(self.x0)
        self.v.fill(0.)
        # self.reset_rho()

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
    def move(self):
        a = ti.math.vec3(0.001)
        for i in range(self.x0):
            self.x0[i] += a

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
