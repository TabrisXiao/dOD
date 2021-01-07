
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

class data_wrapper(object):
	
	"""
	a base class for wrapper the samples into a generator feeding to nn for trainning
	FUNC:
		pop: (vritual) generate one sample
		adjust: (vritual) adjust the data shape for trainning
		generate: generate a batch of samples
		show: current draw the sample by the matplotlib
	"""
	def __init__(self,name, pixel_width, pixel_height, dtype, FPS=10):
		"""
		FPS (frames per sample): number of frames for each sample = pop()
		"""
		self.name = name 
		self.p_w = pixel_width
		self.p_h = pixel_height
		self.fps = FPS
		if dtype == 'float32': self.dtype = np.float32
		elif dtype == 'float16': self.dtype = np.float16
	
	def generate(self, size):
		""" 
		The virutal function pop here suppose to be implemented in child class with the output:
		data, sig, logist = self.pop()
		where 'data' is the sig+bkg	
		"""
		data = []
		sig = []
		for i in range(size):
			d, s = self.pop(self.fps)
			data.append(d)
			sig.append(s)
		#return self.adjust(np.array(data)), np.array(sig)
		return self.adjust(np.array(data)), self.adjust(np.array(sig))
	
	def update(self, i, data, fg, ax, ntrail):
		label = 'Frame step: {0}/{1}'.format(i+1, ntrail*self.fps)
		fg.set_data(data[i])
		ax.set_xlabel(label)
		return fg, ax

	
	def show(self):
		ntrails = int(40/self.fps);
		if ntrails*self.fps < 40 : ntrails+=1
		data = []
		mask = []
		d, sig = self.pop(self.fps)
		for i in range(self.fps):
			data.append(d[i])
			mask.append(sig[0]) # the first fps-1 frames are still
		for j in range((ntrails-1)*self.fps):
			d, sig = self.pop(self.fps)
			data.append(d[self.fps-1])
			mask.append(sig[0])
		fig,(ax1,ax2) = plt.subplots(ncols=2)
		fg1 = ax1.imshow(data[0], interpolation='none')
		fg2 = ax2.imshow(mask[0], interpolation='none')
		ax1.set_title("data")
		ax2.set_title("mask")
		for i in range(1, ntrails*self.fps):
			self.update(i, data, fg1, ax1, ntrails)
			self.update(i, mask, fg2, ax2, ntrails)
			plt.pause(2/self.fps/ntrails)
			fig.canvas.draw()
		plt.show()

	def show_each_pop(self, ntrails):
		if ntrails*self.fps < 40 : ntrails+=1
		data = []
		mask = []
		for j in range(ntrails):
			d, sig = self.pop(self.fps)
			for i in range(self.fps):
				data.append(d[i])
				mask.append(sig[0]) # the first fps-1 frames are still
		fig,(ax1,ax2) = plt.subplots(1,2)
		#fig,(ax1,ax2) = plt.subplots(1,2, sharey=True)
		print(len(mask))
		fg1 = ax1.imshow(data[0])
		fg2 = ax2.imshow(mask[0])
		ax1.set_title("data")
		ax2.set_title("mask")
		for i in range(1, ntrails*self.fps):
			self.update(i, data, fg1, ax1, ntrails)
			self.update(i, mask, fg2, ax2, ntrails)
			plt.pause(2/self.fps/ntrails)
		plt.show()

	def normalization(self, array):
		m = np.amax(array)
		if m != 0: array /=m
		return array
		
	def check(self, net):
		ntrails = int(42/self.fps);
		if ntrails*self.fps < 42 : ntrails+=1
		data = []
		mask = []
		predict = []
		difference = []
		d, sig = self.pop(self.fps)
		pre = net.predict(self.adjust([d]))[0,:,:,0]
		pre = self.normalization(pre)
		diff = np.subtract(pre, sig[0])
		for i in range(self.fps):
			data.append(d[i])
			mask.append(sig[0]) # the first fps-1 frames are still
			predict.append(pre) # the first fps-1 frames are still
			difference.append(diff) # the first fps-1 frames are still
		for j in range((ntrails-1)*self.fps):
			d, sig = self.pop(self.fps)
			data.append(d[self.fps-1])
			pre = net.predict(self.adjust([d]))[0,:,:,0]
			pre = self.normalization(pre)
			mask.append(sig[0])
			predict.append(pre) 
			diff = np.subtract(pre, sig[0])
			difference.append(diff) 
		fig,ax, = plt.subplots(1,4)
		print(len(mask))
		fg1 = ax[0].imshow(data[0])
		fg2 = ax[1].imshow(mask[0])
		fg3 = ax[2].imshow(predict[0])
		fg4 = ax[3].imshow(difference[0])
		ax[0].set_title("data")
		ax[1].set_title("mask")
		ax[2].set_title("predict")
		ax[3].set_title("difference")
		for i in range(1, ntrails*self.fps):
			self.update(i, data,       fg1, ax[0], ntrails)
			self.update(i, mask,       fg2, ax[1], ntrails)
			self.update(i, predict,    fg3, ax[2], ntrails)
			self.update(i, difference, fg4, ax[3], ntrails)
			plt.pause(2/self.fps/ntrails)
			fig.canvas.draw()
		plt.show()

