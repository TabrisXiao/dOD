
import numpy as np

def bkg_sample(size):
   return np.random.rand(size[0],size[1])

def signal_sample(r, size, start = [0,0]):
    sig = np.zeros((size[0],size[1]))
    width = size[0]
    height = size[1]
    x0=[0,0]
    p_ul = [0,0]
    p_dr = [0,0]
    if start[0]-r>=0: p_ul[0]=start[0]-r
    else : p_ul[0]=0
    if start[1]-r>=0: p_ul[1]=start[1]-r
    else : p_ul[1]=0

    if start[0]+r>width-1 : p_dr[0]=width-1
    else : p_dr[0]=start[0]+r
    if start[1]+r>height-1: p_dr[1]=height-1
    else : p_dr[1]=start[1]+r

    for i in range(p_ul[0],p_dr[0]+1):
        for j in range(p_ul[1],p_dr[1]+1):
            if pow(pow(x0[0]+i-start[0],2)+pow(x0[1]+j-start[1],2),0.5) > r: continue
            sig[x0[0]+i,x0[1]+j] = 1
            
    return sig

def signal_motion_sample(r,size, ntime):
    x = int(np.random.randint(size[0]-1))
    y = int(np.random.randint(size[1]-1))

    bkg = bkg_sample(size)
   
    vmax_x = 10
    vmax_y = 10
    dx=0 
    dy=0
    sig = []
    data = []
    for i in range(ntime):
        d0 = signal_sample(r, size, [x,y])
#        data = d0
        sig.append(d0)
        data.append(np.add(d0,bkg))
        dx = int(np.random.randint(-vmax_x,vmax_x))
        dy = int(np.random.randint(-vmax_y,vmax_y))
        x = x+dx
        y = y+dy
        if x < 0 or x > size[0]-1 : x = x-2*dx
        if y < 0 or y > size[1]-1 : y = y-2*dy
    return data, sig

t = [200,200]
#signal_motion_sample(2,t,1)
ntime = 50
data,sig = signal_motion_sample(5,t,ntime)


import matplotlib.pyplot as plt
import time
for i in range(ntime):
    plt.imshow(data[i])
    plt.pause(1.0/ntime)

plt.show()
