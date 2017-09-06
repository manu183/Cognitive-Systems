import cv2
import matplotlib.pyplot as plt
from histogram import histogram_stretching, histogram_stretching_quantile, histogram_equalization
from os.path import basename, dirname, join, splitext

def stretch_histogram(image_path):
	filename = basename(splitext(image_path)[0])
	extension = splitext(image_path)[1]
	folder = dirname(splitext(image_path)[0])
	im = cv2.imread(image_path) 
	im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	im_fixed_contrast = histogram_stretching(im_grey)
	cv2.imwrite(join(folder, ''.join((filename + '-hist_stretched', extension))), im_fixed_contrast)

def stretch_histogram_quantile(image_path):
	filename = basename(splitext(image_path)[0])
	extension = splitext(image_path)[1]
	folder = dirname(splitext(image_path)[0])
	im = cv2.imread(image_path) 
	im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	im_fixed_contrast = histogram_stretching_quantile(im_grey)
	cv2.imwrite(join(folder, ''.join((filename + '-hist_stretched_quantile', extension))), im_fixed_contrast)

def equalize_histogram(image_path):
	filename = basename(splitext(image_path)[0])
	extension = splitext(image_path)[1]
	folder = dirname(splitext(image_path)[0])
	im = cv2.imread(image_path) 
	im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	im_fixed_contrast = histogram_equalization(im_grey)
	cv2.imwrite(join(folder, ''.join((filename + '-equalized_hist', extension))), im_fixed_contrast)

stretch_histogram("imgs/family-bad_contrast.jpg")
stretch_histogram_quantile("imgs/family-bad_contrast.jpg")
equalize_histogram("imgs/family-bad_contrast.jpg")