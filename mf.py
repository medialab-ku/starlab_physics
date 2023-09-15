import taichi as ti
import meshtaichi_patcher as Patcher
from scipy.spatial.transform import Rotation as R

@ti.data_oriented
class mathFunctions:
    def __init__(self):
        pass

    @ti.kernel
    def add(self, ans: ti.template(), a: ti.template(), k: ti.f32, b: ti.template()):
        for i in ans:
            ans[i] = a[i] + k * b[i]

    @ti.kernel
    def dot(self, a: ti.template(), b: ti.template()) -> ti.f32:
        ans = 0.0
        ti.loop_config(block_dim=32)
        for i in a: ans += a[i].dot(b[i])
        return ans

    @ti.kernel
    def cross(self, a: ti.template(), b: ti.template()) -> ti.f32:
        ans = 0.0
        ti.loop_config(block_dim=32)
        for i in a: ans += a[i].cross(b[i])
        return ans

    @ti.kernel
    def sub(self, ans: ti.template(), a: ti.template(), b: ti.template()):
        for i in ans:
            ans[i] = a[i] - b[i]