
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
	def __init__(self,name, pixel_width, pixel_height, FPS=10):
		"""
		FPS (frames per sample): number of frames for each sample = pop()
		"""
		self.name = name 
		self.p_w = pixel_width
		self.p_h = pixel_height
		self.fps = FPS
	
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
		return self.adjust(np.array(data)), np.array(sig)
		#return self.adjust(np.array(data)), self.adjust(np.array(sig))
	
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
			mask.append(sig) # the first fps-1 frames are still
		for j in range((ntrails-1)*self.fps):
			d, sig = self.pop(self.fps)
			data.append(d[self.fps-1])
			mask.append(sig)
		fig,(ax1,ax2) = plt.subplots(1,2, sharey=True)
		print(data[0].shape)
		fg1 = ax1.imshow(data[0])
		fg2 = ax2.imshow(mask[0])
		ax1.set_title("data")
		ax2.set_title("mask")
		for i in range(1, ntrails*self.fps):
			self.update(i, data, fg1, ax1, ntrails)
			self.update(i, mask, fg2, ax2, ntrails)
			plt.pause(2/self.fps/ntrails)
		plt.show()
			
