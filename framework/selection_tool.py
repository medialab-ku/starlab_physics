import taichi as ti
import numpy as np
import json

@ti.data_oriented
class SelectionTool :

    def __init__(self,max_num_verts_dynamic, simulation_x, window, camera):
        # self.selected_indices = ti.field(dtype = ti.uint32,shape = (max_num_verts_dynamic,))
        self.num_selected = 0

        self.selectionCounter = ti.field(dtype = ti.int32,shape = ())
        self.selectionCounter[None] = 1
        self.num_maxCounter = 4

        self.LMB_mouse_pressed = False

        self.MODE_SELECTION = True #True add, False sub

        self.selected_indices_cpu = {}
        self.is_selected = ti.field(dtype=ti.f32, shape=max_num_verts_dynamic)
        self.window = window
        self.camera = camera
        self.max_num_verts_dynamic = max_num_verts_dynamic

        self.ti_viewTransform = ti.Matrix.field(n=4, m=4, dtype = ti.float32, shape = ())
        self.ti_projTransform = ti.Matrix.field(n=4, m=4, dtype = ti.float32, shape = ())
        self.simulation_x = simulation_x

        self.ti_mouse_click_index = ti.Vector.field(2, ti.int32,shape = (4,))
        self.ti_mouse_click_index[0][0] = 0
        self.ti_mouse_click_index[0][1] = 1
        self.ti_mouse_click_index[1][0] = 1
        self.ti_mouse_click_index[1][1] = 2
        self.ti_mouse_click_index[2][0] = 2
        self.ti_mouse_click_index[2][1] = 3
        self.ti_mouse_click_index[3][0] = 3
        self.ti_mouse_click_index[4][1] = 0

        self.ti_mouse_click_pos = ti.Vector.field(2,ti.float32,shape=(4,))
        self.mouse_click_pos = [0, 0, 0, 0]# press x,y release x,y

        self.renderTestPosition = ti.Vector.field(n=3, dtype=ti.f32, shape=(max_num_verts_dynamic,))

        self.Select()
    def Select(self):
        aspect = self.window.get_window_shape()[0]/self.window.get_window_shape()[1];
        xmin = self.mouse_click_pos[0] if self.mouse_click_pos[0] < self.mouse_click_pos[2] else self.mouse_click_pos[2]
        xmax = self.mouse_click_pos[2] if self.mouse_click_pos[0] < self.mouse_click_pos[2] else self.mouse_click_pos[0]
        ymin = self.mouse_click_pos[1] if self.mouse_click_pos[1] < self.mouse_click_pos[3] else self.mouse_click_pos[3]
        ymax = self.mouse_click_pos[3] if self.mouse_click_pos[1] < self.mouse_click_pos[3] else self.mouse_click_pos[1]

        projMatrix = self.camera.get_projection_matrix(aspect).T
        viewTransform = self.camera.get_view_matrix().T

        self.ti_viewTransform[None] = viewTransform.tolist()
        self.ti_projTransform[None] = projMatrix.tolist()

        self._check_inside_selection_box(xmin,xmax,ymin,ymax,self.MODE_SELECTION)
        ti.sync()

        # self.get_selection_array()
        # self.export_selection()

        self.mouse_click_pos = [0, 0, 0, 0]
        self.reset_ti_rect_selection()

    @ti.kernel
    def _check_inside_selection_box(self, xmin : float,xmax : float,ymin : float,ymax : float,mode : int):
        for i in self.is_selected :
            # self.is_selected[i] = False

            pos = self.simulation_x[i]
            pos_h = ti.Vector([pos[0], pos[1], pos[2], 1])
            pos_h_in_clipSpace = self.ti_projTransform[None] @ self.ti_viewTransform[None] @ pos_h
            pos_h_in_clipSpace = pos_h_in_clipSpace/pos_h_in_clipSpace[3]

            x_c,y_c = pos_h_in_clipSpace[0] * 0.5 + 0.5,pos_h_in_clipSpace[1] * 0.5 + 0.5 # viewport transform

            if x_c > xmin and x_c < xmax and y_c > ymin and y_c < ymax:
                self.is_selected[i] = mode if mode == 0 else self.selectionCounter[None]


    def update_ti_rect_selection(self):

        self.ti_mouse_click_pos[0][0] = self.mouse_click_pos[0]
        self.ti_mouse_click_pos[0][1] = self.mouse_click_pos[1]
        self.ti_mouse_click_pos[1][0] = self.mouse_click_pos[2]
        self.ti_mouse_click_pos[1][1] = self.mouse_click_pos[1]
        self.ti_mouse_click_pos[2][0] = self.mouse_click_pos[2]
        self.ti_mouse_click_pos[2][1] = self.mouse_click_pos[3]
        self.ti_mouse_click_pos[3][0] = self.mouse_click_pos[0]
        self.ti_mouse_click_pos[3][1] = self.mouse_click_pos[3]
    def reset_ti_rect_selection(self):
        self.ti_mouse_click_pos[0][0] = 0
        self.ti_mouse_click_pos[0][1] = 0
        self.ti_mouse_click_pos[1][0] = 0
        self.ti_mouse_click_pos[1][1] = 0
        self.ti_mouse_click_pos[2][0] = 0
        self.ti_mouse_click_pos[2][1] = 0
        self.ti_mouse_click_pos[3][0] = 0
        self.ti_mouse_click_pos[3][1] = 0

    def get_selection_array(self):
        is_selected_np = self.is_selected.to_numpy()
        self.selected_indices_cpu = np.argwhere(is_selected_np == True)
        self.selected_indices_cpu = self.selected_indices_cpu[:,0]
        self.num_selected = self.selected_indices_cpu.shape[0]
        print("selection_tool.get_selection_array()::"," num_selected : ",self.num_selected," :: idx : ",self.selected_indices_cpu)




    def export_selection(self):
        is_selected_np = self.is_selected.to_numpy()
        # self.selected_indices_cpu = {}
        count_sum = 0

        # print("==== selection count ====")

        for i in range(self.num_maxCounter) :
            i_selected = np.argwhere(is_selected_np == (i+1))
            i_selected =  list(i_selected[:,0])
            self.selected_indices_cpu[i+1] = list([int(x) for x in i_selected])

            # print(i+1,"th selection",self.selected_indices_cpu[i+1])
            # print(i+1,"th selection count",len(self.selected_indices_cpu[i+1]))

            count_sum = count_sum + len(self.selected_indices_cpu[i+1])

        # print("total selection count : ", count_sum)
        # print("==========================")

        # print(self.selected_indices_cpu)

        with open('animation/handle.json', 'w', encoding='utf-8') as f:
            json.dump(self.selected_indices_cpu, f, ensure_ascii=False, indent=4)


        # with open('handle.json') as f:
        #     testRead = json.load(f)
        # testRead = {int(k): v for k, v in testRead.items()}
        #
        # for i in range(self.num_maxCounter) :
        #     for j in range(len(testRead[i+1])) :
        #         if not testRead[i+1][j] == self.selected_indices_cpu[i+1][j] :
        #             print("nononoonononoonononoonononoonononoonononoonononoonononoonononoonononoonononoo",testRead[i+1][j],self.selected_indices_cpu[i+1][j])

    def import_selection(self):
        with open('animation/handle.json') as f:
            self.selected_indices_cpu = json.load(f)
        self.selected_indices_cpu = {int(k): v for k, v in self.selected_indices_cpu.items()}

        # for i in range(self.num_maxCounter) :
        #     print(i+1,"th selection",self.selected_indices_cpu[i+1])
        #     print(i+1,"th selection count",len(self.selected_indices_cpu[i+1]))

        self.is_selected.fill(0)

        for i in range(self.num_maxCounter) :
            idxCount = i+1
            idx = self.selected_indices_cpu[idxCount]
            for j in idx :
                self.is_selected[j] = idxCount # 1,2,3,4


    @ti.kernel
    def renderTestPos(self):
        for i in self.simulation_x:
            if self.is_selected[i] < 1.0:
                self.renderTestPosition[i] = ti.Vector([-999, -999, -999])
            else :
                self.renderTestPosition[i] = self.simulation_x[i]

    def selection_Count_Up(self):
        self.selectionCounter[None] = (self.selectionCounter[None] % self.num_maxCounter) + 1
        print("current_selectionCounter = ", self.selectionCounter[None])