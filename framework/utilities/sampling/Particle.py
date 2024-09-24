import meshio
import math
import os
import numpy as np
import random
import taichi as ti
import matplotlib.pyplot as plt


# from UI.sampling.Particle import ClothParticle
# part = ClothParticle('long_shirt', 'long_shirt.obj')

class ClothParticle:
    def __init__(self, subFolder, filename):
        self.filename = filename

        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_path = os.path.join(self.current_dir, 'cloth_models', subFolder, filename)


        self.obj = None
        self.grid = None
        self.barycentric = []
        self.elev = 0
        self.azim = 0
        self.size = 5

        self.particle_field = None

    def obj_load(self):
        mesh = meshio.read(self.file_path)
        vertices = np.array(mesh.points, dtype=np.float32)

        try:
            triangles = np.array(mesh.cells_dict["triangle"], dtype=np.int32)
        except:
            triangles = []
        try:
            quads = np.array(mesh.cells_dict["quad"], dtype=np.int32)
        except:
            quads = []

        indices = []
        if len(quads) > 0:
            for quad in quads:
                v0, v1, v2, v3 = quad
                indices.append([v0, v1, v2])
                indices.append([v1, v2, v3])

            indices = np.array(indices, dtype=np.int32)


        # '''
        # Non = []
        #
        # if len(triangles) > 0 and not len(indices):
        #     indices = np.vstack([triangles, indices])
        # '''


        self.obj = {
            'vertices': vertices,
            'indices': indices
        }

        self.obj['indices'] = triangles

    def transform(self, scale, translate, PRINT: bool = False):
        self.obj['vertices'] *= scale
        self.obj['vertices'] += np.vstack([translate] * self.obj['vertices'].shape[0])
        print("transformed\n")

        if PRINT:
            print("\nvertices\n", self.obj['vertices'])
            print("\nindices\n", self.obj['indices'])

    def canvas_generate(self, distance: float, iteration: int = 30, canvas_size: tuple = (100, 100)) -> list:
        canvas_width, canvas_height = canvas_size

        grid_size = distance / math.sqrt(2)
        grid_width = math.floor(canvas_width / grid_size) + 1
        grid_height = math.floor(canvas_height / grid_size) + 1
        self.grid = np.ndarray(shape=(grid_width, grid_height, 2), dtype=np.float32)
        self.grid.fill(-100.0)

        # random generate
        start_point = np.array((random.random() * canvas_width, random.random() * canvas_height), dtype=np.float32)
        start_grid_x = math.floor(start_point[0] / grid_size)
        start_grid_y = math.floor(start_point[1] / grid_size)
        self.grid[start_grid_x, start_grid_y] = start_point

        # generate trigger
        active = [start_point]
        points = [start_point]

        while len(active) > 0:
            rand_idx = random.randint(0, len(active) - 1)
            selected_vector = active[rand_idx]

            found = False

            for temp in range(iteration):
                rand_theta = random.random() * math.pi * 2.0
                rand_unit_vector = np.array((np.cos(rand_theta), np.sin(rand_theta)), dtype=np.float32)

                rand_vector_magnitude = (random.random() + 1) * distance  # [r, 2r]
                rand_vector = selected_vector + (rand_unit_vector * rand_vector_magnitude)

                rand_grid_x, rand_grid_y = math.floor(rand_vector[0] / grid_size), math.floor(
                    rand_vector[1] / grid_size)

                if not self.isvalid(rand_vector, rand_grid_x, rand_grid_y, grid_width, grid_height, canvas_width,
                                    canvas_height, distance):
                    continue

                found = True
                self.grid[rand_grid_x, rand_grid_y] = rand_vector
                points.append(rand_vector)
                active.append(rand_vector)

                break

            if not found:
                del (active[rand_idx])

        return points

    def isvalid(self, vector, rand_grid_x, rand_grid_y, grid_width, grid_height, width, height, distance) -> bool:
        if vector[0] < 0 or vector[0] > width or vector[1] < 0 or vector[1] > height:
            return False

        search_x_min = max(rand_grid_x - 1, 0)
        search_x_max = min(rand_grid_x + 1, grid_width - 1)

        search_y_min = max(rand_grid_y - 1, 0)
        search_y_max = min(rand_grid_y + 1, grid_height - 1)

        for i in range(search_x_min, search_x_max + 1, 1):
            for j in range(search_y_min, search_y_max + 1, 1):
                if (self.grid[i][j] - [-100.0, -100.0]).all():
                    if np.linalg.norm(self.grid[i][j] - vector) < distance:
                        return False

        return True

    def project(self, v1, v2, v3):  # apply to grid
        l13 = v3 - v1
        l12 = v2 - v1

        norm_l12 = np.linalg.norm(l12)

        proj_v1 = np.array([0, 0])
        proj_v2 = np.array([norm_l12, 0])

        proj_l13_to_l12 = np.dot(l12, l13) / norm_l12
        proj_v3 = np.array(
            [max(proj_l13_to_l12, norm_l12 - proj_l13_to_l12), np.linalg.norm(l13 - proj_l13_to_l12 * l12 / norm_l12)])

        return np.array([proj_v1, proj_v2, proj_v3], dtype=np.float32)

    def is_point_in_triangle(self, point, v1, v2, v3) -> bool:
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        sign1 = sign(point, v1, v2)
        sign2 = sign(point, v2, v3)
        sign3 = sign(point, v3, v1)

        has_neg = (sign1 < 0) or (sign2 < 0) or (sign3 < 0)
        has_pos = (sign1 > 0) or (sign2 > 0) or (sign3 > 0)

        return not (has_neg and has_pos)

    def cartesian_to_barycentric(self, x1, y1, x2, y2, x3, y3, x, y):

        area = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        u = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / area
        v = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / area
        w = 1 - u - v
        return u, v, w

    def viewchange(self, elev, azim, size):
        self.elev = elev
        self.azim = azim
        self.size = size

    def temp_display(self, points):

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], marker='o', s=self.size)

        max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                              points[:, 1].max() - points[:, 1].min(),
                              points[:, 2].max() - points[:, 2].min()]).max() / 2.0

        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.view_init(elev=self.elev, azim=self.azim)
        plt.show()

    def file_generate(self, distance: float, subFolder, filename, LOG=True, SHOW=False):
        line = []
        numpy_array = np.ndarray(shape=(0, 4), dtype=np.float32)
        print(numpy_array)
        '''
        for v in self.obj['vertices']:
            line.append("v " + str(v[0]) + " " + str(v[1]) + " " + str(v[2]))
        '''

        generated = []
        length = self.obj['indices'].shape[0]

        for face in range(length):
            if LOG:
                print(f'iteration - {face} total - {length - 1}')

            vertex = self.obj['vertices'][self.obj['indices'][face]]
            projected_face = self.project(vertex[0], vertex[1], vertex[2])
            generated_canvas = self.canvas_generate(distance, iteration=30, canvas_size=(
                projected_face[:, 0].max(), projected_face[:, 1].max()))

            #line.append("f " + str(self.obj['indices'][face][0]) + " " + str(self.obj['indices'][face][1]) + " " + str(self.obj['indices'][face][2]))

            for point in generated_canvas:
                if self.is_point_in_triangle(point, projected_face[0], projected_face[1], projected_face[2]):
                    u, v, w = self.cartesian_to_barycentric(*projected_face[0], *projected_face[1], *projected_face[2],
                                                            *point)
                    position = np.array(u * vertex[0] + v * vertex[1] + w * vertex[2], dtype=np.float32)
                    generated.append(position)
                    sample = np.array([[face, u, v, w]], dtype=np.float32)
                    numpy_array = np.append(numpy_array, sample, axis=0)
                    line.append(str(face) + " " + str(u) + " " + str(v) + " " + str(w))

        print(numpy_array.shape)
        generated = np.array(generated, dtype=np.float32)

        if SHOW:
            self.temp_display(generated)

        self.file_path = os.path.join(self.current_dir, 'cloth_models', subFolder, filename)

        string_array = np.array(line, dtype=object)
        np.save(self.file_path, numpy_array)
        np.save(self.file_path_2, numpy_array)

    def file_stage(self, path):
        self.file_path_2 = path

    def calculate_points(self, distance, normal_pointing):
        point = []

        for index in self.barycentric:
            face_idx = index[0]
            vertex_idx = self.obj['indices'][int(face_idx)]
            vertex = [self.obj['vertices'][vertex_idx[0]], self.obj['vertices'][vertex_idx[1]],
                      self.obj['vertices'][vertex_idx[2]]]

            normal = np.cross(vertex[1] - vertex[0], vertex[2] - vertex[0])
            normal = normal / np.linalg.norm(normal) * distance
            vertex += np.vstack([normal] * 3) * normal_pointing
            point.append(vertex[0] * index[1] + vertex[1] * index[2] + vertex[2] * index[3])

        point = np.array(point, dtype=np.float32)
        return point

    def matplot_display(self, distance):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')

        for pointing in range(-1, 2, 1):
            points = self.calculate_points(distance, pointing)
            print(points)
            print("points: "+ str(points.shape))
            # if pointing == -1:
            #     ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='purple', marker='o', s=self.size)
            if pointing == 0:
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='green', marker='o', s=self.size)
            # if pointing == 1:
            #     ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', marker='o', s=self.size)

        max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                              points[:, 1].max() - points[:, 1].min(),
                              points[:, 2].max() - points[:, 2].min()]).max() / 2.0

        mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
        mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
        mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.view_init(elev=self.elev, azim=self.azim)
        plt.show()

    def setscene(self, scene: ti.ui.Scene):
        self.scene = scene

    def initialize(self):
        self.particle_field = ti.Vector.field(3, dtype=ti.f32, shape=len(self.barycentric))

    def taichi_display(self, distance):
        for pointing in range(-1, 2, 1):
            points = self.calculate_points(distance, pointing)
            self.particle_field.from_numpy(points)

            if pointing == -1:
                self.scene.particles(self.particle_field, radius=self.size, color=(1.0, 0.0, 1.0))
            if pointing == 0:
                self.scene.particles(self.particle_field, radius=self.size, color=(0.0, 1.0, 0.0))
            if pointing == 1:
                self.scene.particles(self.particle_field, radius=self.size, color=(0.0, 0.0, 1.0))

    def file_load(self, normal_distance, subFolder, filename, SHOW=True, MODE="MATPLOT"):
        self.file_path = os.path.join(self.current_dir, 'cloth_models', subFolder, filename)

        load_array = np.load(self.file_path_2, allow_pickle=True)

        print(load_array)
        np.set_printoptions(threshold=np.inf)

        for line in load_array:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.startswith("uvw "):
                weight = list(map(float, line.split()[1:]))
                self.barycentric.append(weight)

        if SHOW and MODE == "MATPLOT":
            self.matplot_display(normal_distance)

        if SHOW and MODE == "TAICHI":
            self.initialize()
