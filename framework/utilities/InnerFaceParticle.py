import taichi as ti
import numpy as np

@ti.data_oriented
class InnerFaceParticle:
    def __init__(self, num_faces, x_np, fid_np, is_static):
        self.num_faces = num_faces
        self.x0_np = x_np
        self.fid_np = fid_np
        self.is_static = is_static

        self.num_particles_per_edge = 5
        self.num_particles_per_face = np.sum(np.arange(1, self.num_particles_per_edge + 1))

        self.ub = 0.9
        self.lb = 1 - self.ub

        if is_static is True:
            print("Static mesh doesn't need to have particles!")
            return

        self.particles_per_face_np_x0 = np.zeros(shape=(self.num_faces, self.num_particles_per_face, 3), dtype=float)
        self.particles_per_face_field = ti.Vector.field(n=3, shape=(self.num_faces * self.num_particles_per_face), dtype=float)
        self.particles_per_face_field_x0 = ti.Vector.field(n=3, shape=(self.num_faces * self.num_particles_per_face), dtype=float)

        self.make_particles()
        self.np_to_field()

    def make_particles(self):
        for i in range(self.num_faces):
            # initialize particles which is inside their face...
            vid_0, vid_1, vid_2 = self.fid_np[i,0], self.fid_np[i,1], self.fid_np[i,2]
            v0, v1, v2 = self.x0_np[vid_0], self.x0_np[vid_1], self.x0_np[vid_2]

            count = 0
            for j in range(self.num_particles_per_edge):
                for k in range(j+1):
                    self.particles_per_face_np_x0[i,count] = (
                        (self.ub - ((self.ub - self.lb) * (j / (self.num_particles_per_edge - 1)))) * v0 +
                        ((self.lb / 2) + ((self.ub - self.lb) * ((j-k) / (self.num_particles_per_edge - 1)))) * v1 +
                        ((self.lb / 2) + ((self.ub - self.lb) * (k / (self.num_particles_per_edge - 1)))) * v2
                    )
                    count += 1

    def np_to_field(self):
        for i in range(self.num_faces):
            for j in range(self.num_particles_per_face):
                self.particles_per_face_field[self.num_particles_per_face * i + j] = self.particles_per_face_np_x0[i, j]
                self.particles_per_face_field_x0[self.num_particles_per_face * i + j] = self.particles_per_face_np_x0[i, j]

    def reset(self):
        self.particles_per_face_field.from_numpy(self.particles_per_face_np_x0.reshape(self.num_faces * self.num_particles_per_face, 3))