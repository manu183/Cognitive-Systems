import cv2
import numpy as np
from os.path import basename, dirname, join, splitext

from histogram import histogram_stretching, histogram_stretching_quantile, histogram_equalization
from filter import gauss_2d, median, prewitt_x, prewitt_y, sobel_x, sobel_y, roberts_x, roberts_y, laplace, laplace_alt
from levenshtein import levenshtein_distance
from rotation import rotation_matrix_to_quaternion, rotation_matrix

def get_file_info(image_path):
	filename = basename(splitext(image_path)[0])
	extension = splitext(image_path)[1]
	folder = dirname(splitext(image_path)[0])
	return (folder, filename, extension)

def stretch_histogram(image_path):
	im = cv2.imread(image_path) 
	im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	im_fixed_contrast = histogram_stretching(im_grey)
	folder, filename, extension = get_file_info(image_path)
	cv2.imwrite(join(folder, ''.join((filename, '-hist_stretched', extension))), im_fixed_contrast)

def stretch_histogram_quantile(image_path):
	im = cv2.imread(image_path) 
	im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	im_fixed_contrast = histogram_stretching_quantile(im_grey)
	folder, filename, extension = get_file_info(image_path)
	cv2.imwrite(join(folder, ''.join((filename, '-hist_stretched_quantile', extension))), im_fixed_contrast)

def equalize_histogram(image_path):
	im = cv2.imread(image_path) 
	im_grey = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
	im_fixed_contrast = histogram_equalization(im_grey)
	folder, filename, extension = get_file_info(image_path)
	cv2.imwrite(join(folder, ''.join((filename, '-equalized_hist', extension))), im_fixed_contrast)

def discrete_fourier_transform(image_path):
	im = cv2.imread(image_path, 0)
	dft = cv2.dft(np.float32(im), flags = cv2.DFT_COMPLEX_OUTPUT)
	dft_shift = np.fft.fftshift(dft)
	im_ft = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))
	folder, filename, extension = get_file_info(image_path)
	cv2.imwrite(join(folder, ''.join((filename, '-fourier', extension))), im_ft)

def convolve(image_path, filter):
	im = cv2.imread(image_path, 0)
	folder, filename, extension = get_file_info(image_path)
	cv2.imwrite(join(folder, ''.join((filename, '-convolved', extension))), cv2.filter2D(im, -1, filter))

def main():
	# Histogram normalization
	stretch_histogram("imgs/family-bad_contrast.jpg")
	stretch_histogram_quantile("imgs/family-bad_contrast.jpg")
	equalize_histogram("imgs/family-bad_contrast.jpg")

	# Fourier transformation
	discrete_fourier_transform("imgs/waterfall_jam.jpg")

	# Apply filters
	convolve("imgs/waterfall_jam.jpg", laplace_alt())

	# Levenshtein distance
	s = ['if there is no rain in April you will have a great summer','no rain in april then great summer come','there is rain in April you have summer','in April no rain you have summer great','there is no rain in apple a great summer comes','you have a great summer comes if there is no rain in April']
	t = [_.split() for _ in s]
	print "The word edit distances are %s ,the character distances are %s and the word distances having double-punishment for substitutions are %s" % ([levenshtein_distance(t[0],t[1]),levenshtein_distance(t[0],t[2]),levenshtein_distance(t[0],t[3]),levenshtein_distance(t[0],t[4]),levenshtein_distance(t[0],t[5])],[levenshtein_distance(s[0],s[1]),levenshtein_distance(s[0],s[2]),levenshtein_distance(s[0],s[3]),levenshtein_distance(s[0],s[4]),levenshtein_distance(s[0],s[5])],[levenshtein_distance(t[0],t[1],2),levenshtein_distance(t[0],t[2],2),levenshtein_distance(t[0],t[3],2),levenshtein_distance(t[0],t[4],2),levenshtein_distance(t[0],t[5],2)])

	# Rotation matrix around x with 180 degrees
	mat = rotation_matrix(180,0,0)
	print("Rotation matrix: %s" % mat)

	# Get quaternion from existing rotation matrix
	m = np.array([[0, 0.5 * np.sqrt(3), 0.5], [0, -0.5, 0.5 * np.sqrt(3)], [1, 0, 0]]) # matrix of kogsys exam SS2012 task 2
	quaternion = rotation_matrix_to_quaternion(m)
	print("Quaternion: %s" % (quaternion,))

if __name__ == "__main__":
    main()