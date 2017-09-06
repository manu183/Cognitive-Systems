import numpy as np

def median(shape=(3,3)):
	return np.ones(shape,np.float32)/(shape[0] * shape[1])

def gauss_2d(shape=(3,3),sigma=0.5):
	m, n = [(ss - 1.) / 2. for ss in shape]
	y, x = np.ogrid[-m : m+1, -n : n+1]
	h = np.exp( -(x * x + y * y) / (2. * sigma * sigma) )
	h[ h < np.finfo(h.dtype).eps * h.max() ] = 0
	sumh = h.sum()
	if sumh != 0:
		h /= sumh
	return h

def prewitt_x():
	return np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

def prewitt_y():
	return np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

def sobel_x():
	return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

def sobel_y():
	return np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

def roberts_x():
	return np.array([[-1, 0], [0, 1]])

def roberts_y():
	return np.array([[0, -1], [1, 0]])

def laplace():
	return np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])

def laplace_alt():
	return np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])