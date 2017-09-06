import cv2
import numpy as np
from histogram import histogram_stretching, histogram_stretching_quantile, histogram_equalization
from filter import gauss_2d, median, prewitt_x, prewitt_y, sobel_x, sobel_y, roberts_x, roberts_y, laplace, laplace_alt
from os.path import basename, dirname, join, splitext

def stretch_histogram(image_path):
	filename = basename(splitext(image_path)[0])
	extension = splitext(image_path)[1]
	folder = dirname(splitext(image_path)[0])
	im = cv2.imread(image_path) 
	im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	im_fixed_contrast = histogram_stretching(im_grey)
	cv2.imwrite(join(folder, ''.join((filename, '-hist_stretched', extension))), im_fixed_contrast)

def stretch_histogram_quantile(image_path):
	filename = basename(splitext(image_path)[0])
	extension = splitext(image_path)[1]
	folder = dirname(splitext(image_path)[0])
	im = cv2.imread(image_path) 
	im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	im_fixed_contrast = histogram_stretching_quantile(im_grey)
	cv2.imwrite(join(folder, ''.join((filename, '-hist_stretched_quantile', extension))), im_fixed_contrast)

def equalize_histogram(image_path):
	filename = basename(splitext(image_path)[0])
	extension = splitext(image_path)[1]
	folder = dirname(splitext(image_path)[0])
	im = cv2.imread(image_path) 
	im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	im_fixed_contrast = histogram_equalization(im_grey)
	cv2.imwrite(join(folder, ''.join((filename, '-equalized_hist', extension))), im_fixed_contrast)

def discrete_fourier_transform(image_path):
	filename = basename(splitext(image_path)[0])
	extension = splitext(image_path)[1]
	folder = dirname(splitext(image_path)[0])
	im = cv2.imread(image_path, 0)
	dft = cv2.dft(np.float32(im), flags = cv2.DFT_COMPLEX_OUTPUT)
	dft_shift = np.fft.fftshift(dft)
	im_ft = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
	cv2.imwrite(join(folder, ''.join((filename, '-fourier', extension))), im_ft)

def convolve(image_path, filter):
	filename = basename(splitext(image_path)[0])
	extension = splitext(image_path)[1]
	folder = dirname(splitext(image_path)[0])
	im = cv2.imread(image_path, 0)
	cv2.imwrite(join(folder, ''.join((filename, '-convolved', extension))), cv2.filter2D(im, -1, filter))

# stretch_histogram("imgs/family-bad_contrast.jpg")
# stretch_histogram_quantile("imgs/family-bad_contrast.jpg")
# equalize_histogram("imgs/family-bad_contrast.jpg")
# discrete_fourier_transform("imgs/family-bad_contrast.jpg")
convolve("imgs/waterfall_jam.jpg", laplace_alt())