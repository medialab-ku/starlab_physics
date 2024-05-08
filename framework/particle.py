import taichi as ti
import meshio
import numpy as np

@ti.data_oriented
class Particle:

    def __init__(self,
                 model_path,
                 trans=ti.math.vec3(0, 0, 0),
                 rot=ti.math.vec3(0, 0, 0),
                 scale=1.0,
                 radius=0.01):

        p = meshio.read(model_path)
        points = p.points
        points = np.array(points, dtype=np.float32)
        self.num_particles = points.shape[0]

        print(self.num_particles)

        self.x = ti.Vector.field(n=3, dtype=ti.f32, shape=self.num_particles)
        self.x0 = ti.Vector.field(n=3, dtype=ti.f32, shape=self.num_particles)
        self.y = ti.Vector.field(n=3, dtype=ti.f32, shape=self.num_particles)
        self.x.from_numpy(points)

        self.v = ti.Vector.field(n=3, dtype=ti.f32, shape=self.num_particles)
        self.v.fill(0.0)
        self.nc = ti.field(dtype=ti.int32, shape=self.num_particles)
        self.m_inv = ti.field(dtype=ti.f32, shape=self.num_particles)
        self.m_inv.fill(1.0)
        self.trans = trans
        self.rot = rot
        self.scale = scale
        self.radius = radius

        self.setCenterToOrigin()
        self.applyTransform()

        self.x0.copy_from(self.x)

    def reset(self):
        self.x.copy_from(self.x0)
        self.v.fill(0.)

    @ti.kernel
    def setCenterToOrigin(self):

        center = ti.math.vec3(0, 0, 0)
        for i in range(self.num_particles):
            center += self.x[i]

        center /= self.num_particles
        for i in range(self.num_particles):
            self.x[i] -= center

    @ti.kernel
    def applyTransform(self):
        for i in range(self.num_particles):
            self.x[i] *= self.scale


        for i in range(self.num_particles):
            v_4d = ti.Vector([self.x[i][0], self.x[i][1], self.x[i][2], 1])
            rot_rad = ti.math.radians(self.rot)
            rv = ti.math.rotation3d(rot_rad[0], rot_rad[1], rot_rad[2]) @ v_4d
            self.x[i] = ti.Vector([rv[0], rv[1], rv[2]])

        for i in range(self.num_particles):
            self.x[i] += self.trans
