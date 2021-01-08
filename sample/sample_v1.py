
import numpy as np
from dataWrapper import data_wrapper as drp

def randomPoint(size, shift= [0,0]):
	x = shift[0]+int(np.random.randint(size[0]-2*shift[0]-1))
	y = shift[1]+int(np.random.randint(size[1]-2*shift[1]-1))
	return [x, y]


def bkg_sample(size,dtype):
	return np.random.rand(size[0],size[1]).astype(dtype)

def draw_square(point, r, array, value):
	size = array.shape
	bx = point[0] - r
	if bx < 0 : bx = 0	
	by = point[1] - r
	if by < 0 : bx = 0	
	for i in range(bx,bx+2*r-1):
		if i > size[0]-1 : continue
		for j in range(by,by+2*r-1):
			if j > size[1]-1 : continue
			array[i,j] = value
	return array

def draw_disc(point, r, array, value):
	size = array.shape
	bx = point[0] - r
	if bx < 0 : bx = 0	
	by = point[1] - r
	if by < 0 : bx = 0	
	for i in range(bx,bx+2*r-1):
		if i > size[0]-1 : continue
		for j in range(by,by+2*r-1):
			if j > size[1]-1 : continue
			if pow(pow(i-point[0],2)+pow(j-point[1],2),0.5) > r: continue
			array[i,j] = value
	return array

def draw_dummy(point, r, array, value):
	return array

signal_map = [draw_dummy, draw_disc, draw_square]

def dice_draw_map():
	dice = int(np.random.uniform(1,3))
	if dice == 3 : dice = 0
	return signal_map[dice]	
	
def bkg_sample_v1(point, size,dtype, bkg_noise, bkg_obj):
	"""
	point is the [x,y] coordinate of the upper left shift point
	"""
	if bkg_noise: bkg = np.random.rand(size[0],size[1]).astype(dtype)
	else: bkg = np.zeros((size[0],size[1])).astype(dtype)
	do_shape = np.random.uniform(0,1)
	#if do_shape < 0.3: return bkg
	rad = int(np.random.uniform(10,60))
	val = np.random.uniform(0.6,6)
	
	start = randomPoint(size, point)
	if bkg_obj:
		sigF = dice_draw_map()
		bkg = sigF(start, rad, bkg, val)
	return bkg.astype(dtype)

def signal_gen(point, r, size, sig_f, value, dtype): 
	sig = np.zeros((size[0],size[1]))
	sig_f(point, r, sig, value)
	return sig.astype(dtype)

def signal_motion_sample_v1(r,size, shift,ntime, dtype, **kwargs):
	reverseRate = 0
	bkgScale = [0.9, 1.1]
	keys = kwargs.keys()
	if "bkgScale" in keys: bkgScale = kwargs["bkgScale"]
	if "bkg_noise" in keys: bkg_noise = True
	else : bkg_noise = False
	if "bkg_obj" in kwargs.keys(): bkg_obj = True
	else : bkg_obj = False
	if "reverseRate" in kwargs.keys(): reverseRate = kwargs["reverseRate"]
	x = shift[0]+int(np.random.randint(size[0]-2*shift[0]-1))
	y = shift[1]+int(np.random.randint(size[1]-2*shift[1]-1))
	
	bkg = bkg_sample_v1(shift, size, dtype, bkg_noise,bkg_obj)
	scale = np.random.uniform(bkgScale[0],bkgScale[1])
	bkg = np.multiply(bkg,scale)
	#bkg = bkg_sample(size, dtype)
	reverseDice = np.random.uniform(0,1)
	 
	vmax_x = size[0]//20
	vmax_y = size[1]//20
	dx=0 
	dy=0
	sig = []
	data = []
	boundary_x = [shift[0], size[0]-2*shift[0]-1]
	boundary_y = [shift[1], size[1]-2*shift[1]-1]
	sig_f = dice_draw_map()
	val = np.random.uniform(0.2,1)
	for i in range(ntime):
		if reverseDice < reverseRate:
			d = signal_gen([x,y], r, size, sig_f, -val, dtype)
		else:
			d = signal_gen([x,y], r, size, sig_f, val, dtype)
		d0 = signal_gen([x,y], r, size, sig_f, 1, dtype)
		data.append(np.add(d,bkg))
		sig.append(d0[shift[0]:size[0]-shift[0], shift[1]:size[1]-shift[1]])
		dx = int(np.random.randint(-vmax_x,vmax_x))
		dy = int(np.random.randint(-vmax_y,vmax_y))
		x = x+dx
		y = y+dy
		if x < boundary_x[0] or x > boundary_x[1] : x = x-2*dx
		if y < boundary_y[0] or y > boundary_y[1] : y = y-2*dy
	return data, sig 

class sample_v1(drp):
	def __init__(self, name, px, py, crop, FPS, radius, buffSize = 24, dtype = 'float32', **kwargs):
		super(sample_v1, self).__init__( name, px,py,dtype,FPS)
		self.crop = crop # shift the crop
		self.r = radius
		self.buff_size= buffSize
		self.buff_ptr = self.buff_size
		self.buff =[]
		self.sg_control = kwargs
#		self.bkg_noise = True
#		self.bkg_obj = True
#		self.reverseRate = reverseRate

	def pop(self, fps):
		if fps > self.buff_size : self.buff_size = fps
		if self.buff_ptr == self.buff_size: 
			self.buff = signal_motion_sample_v1(self.r,[self.p_w, self.p_h], self.crop, self.buff_size, self.dtype, **self.sg_control)
			self.buff_ptr =fps
		data = self.buff[0][self.buff_ptr-fps:self.buff_ptr]
		mask = [self.buff[1][self.buff_ptr-1]]
		self.buff_ptr+=1
		return data, mask


	def adjust(self,x):
		x = np.swapaxes(x, 1, 2)
		x = np.swapaxes(x, 2, 3)
		return x
