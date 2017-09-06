import numpy as np

def affine_dot_operator(image, a, b):
	'''
	Algorithmus fuer die Anwendung eines affinen Punktoperators
	'''
	for index, pixel in np.ndenumerate(image):
		g = round(a * image[index] + b)
		if g < 0:
			image[index] = 0
		elif g > 255:
			image[index] = 255
		else:
			image[index] = g

	return image

def calculate_histogram(image):
	'''
	Berechne Histogramm
	'''

	h = np.zeros(256)

	for index, pixel in np.ndenumerate(image):
		h[image[index]] += 1

	return h

def calculate_accumulated_histogram(h):
	'''
	Berechne akkumuliertes Histogramm
	'''

	h_a = np.zeros(256)
	h_a[0] = h[0]

	for x in xrange (1,256):
		h_a[x] = h_a[x-1] + h[x]

	return h_a

def calculate_min_max(image):
	'''
	Berechne minimalen und maximalen Grauwert
	'''

	min = 255
	max = 0
	for index, pixel in np.ndenumerate(image):
		if image[index] < min:
			min = image[index]
		if image[index] > max:
			max = image[index]

	return (min,max)

def calculate_quantile(accumulated_histogram, threshold):
	for x in xrange (0,255):
		if accumulated_histogram[x] >= threshold * accumulated_histogram[-1]:
			return x

def histogram_stretching(image):
	'''
	Histogramm-Spreizung
	'''

	min, max = calculate_min_max(image)
	if min == max:
		return

	a = 255 / (max - min)
	b = - (255 * min) / (max - min)
	return affine_dot_operator(image, a, b)

def histogram_stretching_quantile(image, p_min = 0.1, p_max = 0.9):
	''' 
	Histogramm-Dehnung
	'''

	h = calculate_histogram(image)
	h_a = calculate_accumulated_histogram(h)

	min = calculate_quantile(h_a, p_min)
	max = calculate_quantile(h_a, p_max)

	if min == max:
		return

	a = 255 / (max - min)
	b = - (255 - min) / (max - min)

	return affine_dot_operator(image, a, b)

def histogram_equalization(image):
	'''
	Histogramm-Ausgleich
	'''

	h = calculate_histogram(image)
	h_a= calculate_accumulated_histogram(h)
	h_n = np.zeros(256)

	for x in xrange(0,255):
		h_n[x] = round((255*h_a[x])/(h_a[255]))

	for index, pixel in np.ndenumerate(image):
		image[index] = h_n[image[index]]

	return image
