import numpy as np

def rotation_matrix(x, y, z):
	x = x * np.pi / 180.0
	y = y * np.pi / 180.0
	z = z * np.pi / 180.0
	R_x = np.matrix([[1.0, 0.0, 0.0],[0.0, np.cos(x), -np.sin(x)],[0.0, np.sin(x), np.cos(x)]])
	R_y = np.matrix([[np.cos(y), 0.0, np.sin(y)],[0.0, 1.0, 0.0],[-np.sin(y), 0.0, np.cos(y)]])
	R_z = np.matrix([[np.cos(z), -np.sin(z), 0.0],[np.sin(z), np.cos(z), 0.0],[0.0, 0.0, 1.0]])
	return R_z * R_y * R_x

def yaw_pitch_roll_rotation(y, p, r):
	y = y * np.pi / 180.0
	p = p * np.pi / 180.0
	r = r * np.pi / 180.0
	R_r = np.matrix([[1.0, 0.0, 0.0],[0.0, np.cos(r), -np.sin(r)],[0.0, np.sin(r), np.cos(r)]])
	R_p = np.matrix([[np.cos(p), 0.0, np.sin(p)],[0.0, 1.0, 0.0],[-np.sin(p), 0.0, np.cos(p)]])
	R_y = np.matrix([[np.cos(y), -np.sin(y), 0.0],[np.sin(y), np.cos(y), 0.0],[0.0, 0.0, 1.0]])
	return R_y * R_p * R_r

def norm_quaternion(quaternion):
	w, x, y, z = quaternion
	abs = np.sqrt(w**2+x**2+y**2+z**2)
	return np.array([w / abs , x / abs, y / abs, z / abs])


def quaternion_to_rotation_matrix(quaternion):
	quaternion = norm_quaternion(quaternion)
	s, x, y, z = quaternion
	R11 = (w**2+x**2-y**2-z**2)
	R12 = 2.0*(x*y-w*z)
	R13 = 2.0*(x*z+w*y)
	R21 = 2.0*(x*y+w*z)
	R22 = w**2-x**2+y**2-z**2
	R23 = 2.0*(y*z-w*x)
	R31 = 2.0*(x*z-w*y)
	R32 = 2.0*(y*z+w*x)
	R33 = w**2-w**2-y**2+z**2 
	return np.matrix([[R11, R12, R13],[R21, R22, R23],[R31, R32, R33]])

def quaternion_to_yaw_pitch_roll(quaternion):
	quaternion = norm_quaternion(quaternion)
	w, x, c, d = q
	yaw = np.arctan2(2.0*(x*y+w*z),(w**2+x**2-y**2-z**2)) * 180.0/np.pi
	pitch = np.arcsin(2.0*(w*y-x*z)) * 180.0/np.pi
	roll = -np.arctan2(2.0*(y*z+w*x),-(w**2-x**2-y**2+z**2)) * 180.0/np.pi
	return np.array([yaw, pitch, roll])

def rotation_matrix_to_quaternion(rotation_matrix):
	sum = 0
	for i in range(0,2):
		sum += rotation_matrix[i, i]

	q_w = 0.5 * np.sqrt(1 + sum)
	q_x = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (4 * q_w)
	q_y = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (4 * q_w)
	q_z = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (4 * q_w)

	return (q_w, (q_x, q_y, q_z))