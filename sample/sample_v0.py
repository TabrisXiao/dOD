
import numpy as np
from dataWrapper import data_wrapper as drp

def randomPoint(size, shift= [0,0]):
	x = shift[0]+int(np.random.randint(size[0]-2*shift[0]-1))
	y = shift[1]+int(np.random.randint(size[1]-2*shift[1]-1))
	return [x, y]


def bkg_sample(size,dtype):
	return np.random.rand(size[0],size[1]).astype(dtype)

def bkg_sample_v1(size,dtype):
	bkg = np.random.rand(size[0],size[1]).astype(dtype)
	do_shape = np.random.uniform(0,1)
	#if do_shape < 0.3: return bkg
	rad = int(np.random.uniform(10,60))
	val = np.random.uniform(0.5,3)
	
	start = randomPoint(size)
	bound = []
	bx = start[0] - rad
	if bx < 0 : bx = 0	
	by = start[1] - rad
	if by < 0 : bx = 0	
	for i in range(bx,bx+2*rad-1):
		if i > size[0]-1 : continue
		for j in range(by,by+2*rad-1):
			if j > size[1]-1 : continue
			bkg[i,j] = 1
			#bkg[i,j] = val
	return bkg.astype(dtype)
	

def signal_sample(r, size,dtype, start = [0,0]):
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
	return sig.astype(dtype)


def signal_motion_sample(r,size, shift,ntime, dtype):
    x = shift[0]+int(np.random.randint(size[0]-2*shift[0]-1))
    y = shift[1]+int(np.random.randint(size[1]-2*shift[1]-1))
	
    bkg = bkg_sample_v1(size, dtype)
    #bkg = bkg_sample(size, dtype)
    makeSig = 1
     
    vmax_x = size[0]//20
    vmax_y = size[1]//20
    dx=0 
    dy=0
    sig = []
    data = []
    boundary_x = [shift[0], size[0]-2*shift[0]-1]
    boundary_y = [shift[1], size[1]-2*shift[1]-1]
    for i in range(ntime):
        d0 = np.zeros((size[0],size[1]))
        if makeSig > 0.3:
             d0 = signal_sample(r, size, dtype = dtype, start = [x,y])
        #sig.append(d0)
        data.append(np.add(d0,bkg))
        sig.append(d0[shift[0]:size[0]-shift[0], shift[1]:size[1]-shift[1]])
        dx = int(np.random.randint(-vmax_x,vmax_x))
        dy = int(np.random.randint(-vmax_y,vmax_y))
        x = x+dx
        y = y+dy
        if x < boundary_x[0] or x > boundary_x[1] : x = x-2*dx
        if y < boundary_y[0] or y > boundary_y[1] : y = y-2*dy
    return data, sig 

class sample_v0(drp):
	def __init__(self, name, px, py, crop, FPS, radius, buffSize = 24, dtype = 'float32'):
		super(sample_v0, self).__init__( name, px,py,dtype,FPS)
		self.crop = crop # shift the crop
		self.r = radius
		self.buff_size= buffSize
		self.buff_ptr = self.buff_size
		self.buff =[]

	def pop(self, fps):
		if fps > self.buff_size : self.buff_size = fps
		if self.buff_ptr == self.buff_size: 
			self.buff = signal_motion_sample(self.r,[self.p_w, self.p_h], self.crop, self.buff_size, self.dtype)
			self.buff_ptr =fps
		data = self.buff[0][self.buff_ptr-fps:self.buff_ptr]
		mask = [self.buff[1][self.buff_ptr-1]]
		self.buff_ptr+=1
		return data, mask


	def adjust(self,x):
		x = np.swapaxes(x, 1, 2)
		x = np.swapaxes(x, 2, 3)
		return x
