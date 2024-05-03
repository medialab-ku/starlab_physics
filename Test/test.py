import taichi as ti
import numpy as np

ti.init(ti.gpu)

a = ti.field(dtype = bool,shape = (3,))
b = a
print(a)
print(b)

a[2]=True

print(a)
print(b)

aa = ti.Vector([1,2,3])

#################

n = 100000000
val = ti.field(ti.i32, shape=n)
val_serial = np.zeros((n))

s = ti.field(dtype = ti.i32,shape=())

@ti.kernel
def fill():
    ti.loop_config(parallelize=8, block_dim=256)
    # If the kernel is run on the CPU backend, 8 threads will be used to run it
    # If the kernel is run on the CUDA backend, each block will have 16 threads.
    for i in range(n):
        val[i] = i

fill()



@ti.kernel
def redsum() :
    for i in val :
        if val[i]==i :
            s[None] +=1

redsum()

print("Done!",s[None])

@ti.kernel
def check():
    ti.loop_config(parallelize=8, block_dim=256)
    # If the kernel is run on the CPU backend, 8 threads will be used to run it
    # If the kernel is run on the CUDA backend, each block will have 16 threads.
    for i in range(n):
        if not val[i] == i :
            print(i)

check()


print("=====")


############################
qwer = np.array([1,1,0,0,1,0,1,1,0,1,1,1,0,0,1])

asdf = np.argwhere(qwer==1)

print(np.squeeze(asdf))