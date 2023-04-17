import numpy as np

def carthesian_to_simion_angles(x, y, z):
	el = np.arctan(y/np.sqrt(x**2 + z**2))
	az = np.arctan2(-z, x)

	return az, el

def carthesian_to_polar_angles(x, y, z):
	theta = np.arccos(z/np.sqrt(x**2 + y**2 + z**2))
	phi = np.arctan2(y, x)

	return phi, theta



def simion_to_spherical_angles(az, el, deg=True):
	if np.size(az) > 1 or np.size(el) > 1:
		az = np.atleast_1d(az)
		el = np.atleast_1d(el)

	if deg is True:
		az = az*np.pi/180
		el = el*np.pi/180

	# convert to x, y, z
	x = np.cos(el)*np.cos(az)
	y = np.sin(el)
	z = -np.cos(el)*np.sin(az)

	# compute spherical angles
	phi, theta = carthesian_to_polar_angles(x, y, z)

	if deg is True:
		phi = phi*180/np.pi
		theta = theta*180/np.pi

	return phi, theta

def spherical_to_simion_angles(phi, theta, deg=True):
	if np.size(phi) > 1 or np.size(theta) > 1:
		phi = np.atleast_1d(phi)
		theta = np.atleast_1d(theta)

	if deg is True:
		phi = phi*np.pi/180
		theta = theta*np.pi/180

	# convert to x, y, z
	x = np.cos(phi) * np.sin(theta)
	y = np.sin(phi) * np.sin(theta)
	z = np.cos(theta)

	# compute spherical angles
	az, el = carthesian_to_simion_angles(x, y, z)

	if deg is True:
		az = az*180/np.pi
		el = el*180/np.pi

	return az, el