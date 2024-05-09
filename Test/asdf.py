import json

#
# data = {}
#
# data[0] = [0,1,2]
# print(data)
#
# with open('data.json', 'w', encoding='utf-8') as f:
#     json.dump(data, f, ensure_ascii=False, indent=4)
#
#
# with open('data.json') as f :
#     asdf = json.load(f)
# asdf = {int(k):v for k,v in asdf.items()}

import taichi as ti

ti.init(ti.gpu)

asdf = ti.Vector.field(6, dtype=ti.i32, shape=(4, 40))

asdf[3,0] = ti.Vector([1,1,1,1,1,3])


for i in range(1,5):
    print(i)

# q = ti.Vector([1.0,0.0,0.0,0.0])
q = ti.Vector([1.0,2.0,3.0,4.0])

M = ti.math.mat4(0.0)
M[3,3] = 1.0
M[0,0] = 1-2*(q[1]**2 + q[2]**2)
M[1,1] = 1-2*(q[0]**2 + q[2]**2)
M[2,2] = 1-2*(q[0]**2 + q[1]**2)
M[2,1] = 2*(q[1]*q[2] - q[3] * q[0])
M[1,2] = 2*(q[1]*q[2] - q[3] * q[0])
M[0,2] = 2*(q[0]*q[2] + q[3] * q[1])
M[2,0] = 2*(q[0]*q[2] + q[3] * q[1])
M[0,1] = 2*(q[0]*q[1] - q[3] * q[2])
M[1,0] = 2*(q[0]*q[1] - q[3] * q[2])

aaa = ti.Vector([1.0,1.0,1.0,1.0])

print(M)
print(M@aaa)