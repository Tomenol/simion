import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import gaussian_kde
import scipy

from tqdm import tqdm

import os

def sample_sphere_simion(az_min, az_max, el_min, el_max):
	phi = np.random.uniform(az_min, az_max)
	theta = np.arcsin(np.random.uniform(np.sin(el_min), np.sin(el_max)))

	return phi, theta

def compute_az_el_bounds(x, y, z):
	az_min = np.min(np.arctan2(-z[None, :], x[:, None]))
	az_max = np.max(np.arctan2(-z[None, :], x[:, None]))

	if x[0] > 0 or x[1] < 0:
		r_min = np.min(np.sqrt(x[:, None]**2 + z[None, :]**2))
	else:
		r_min = np.min(z)
	el_min = np.arctan(y/r_min)
	el_max = np.arctan(y/np.max(np.sqrt(x[:, None]**2 + z[None, :]**2)))

	return az_min, az_max, el_min, el_max 


def transport_entrance_to_start_surface(x_cs, y_cs, z_cs, x_e, y_e, z_e, z_start):
	x_e_at_start = x_e + (x_cs - np.flip(x_e))/(z_cs[0] - z_e) * (z_e - z_start)
	y_e_at_start = y_e + (y_e - y_cs)/(z_cs[0] - z_e) * (z_e - z_start)

	return x_e_at_start, y_e_at_start


def transport_to_cs(x, y, z, x0, y0, z0, y_cs):
	x_at_cs = x - (x - x0)/(y - y0)*(y - y_cs)
	z_at_cs = z - (z - z0)/(y - y0)*(y - y_cs)

	return x_at_cs, z_at_cs


def transport_to_entrance(x, y, z, x0, y0, z0, z_e):
	x_at_e = x - (x - x0)/(z - z0)*(z - z_e)
	y_at_e = y - (y - y0)/(z - z0)*(z - z_e)

	return x_at_e, y_at_e


def compute_direction(az, el, x0, y0, z0):
	x = np.cos(el)*np.cos(az) + x0
	y = np.sin(el) + y0
	z = -np.cos(el)*np.sin(az) + z0

	return x, y, z


def plot_lnt_surfaces(x_cs, y_cs, z_cs, x_e, y_e, z_e, z_start, ax):
	# Conversion surface
	X = np.linspace(*x_cs, 10)
	Z = np.linspace(*z_cs, 10)
	X, Z = np.meshgrid(X, Z)
	ax.plot_surface(X, np.ones(np.shape(Z))*y_cs, Z) 

	# Entrance surface
	X = np.linspace(*x_e, 10)
	Y = np.linspace(*y_e, 10)
	X, Y = np.meshgrid(X, Y)
	ax.plot_surface(X, Y, np.ones(np.shape(Z))*z_e)

	return ax


def compute_max_solid_angle(x_cs, y_cs, z_cs, x_e, y_e, z_e, z_start, tol=0.001, N=100):
	""" Computes the maximum sampling solid angle enclosing the CS from the initial start surface.

	This function computes the maximum sampling solid angle enclosing the CS from the initial start surface by computing 
	the solid angle enclosing CS for multiple evenly spaced points on a grid (start surface).

	Note: 
	 - It does not take into account the CS obstruction by the entrance surface.
  	 - The sampling solid angle is bigger than the actual CS solid angle because sampling is performed with respect to constant
  	   azimuth and elevation bounds

	Parameters
	----------
		 x_cs : numpy.ndarray (2,)
			Minimum/maximum x-coordinates of the conversion surface.
		 y_cs : float
			y-coordinate of the conversion surface.
		 z_cs : numpy.ndarray (2,)
			Minimum/maximum z-coordinates of the conversion surface.
		 x_e : numpy.ndarray (2,)
			Minimum/maximum x-coordinates of the entrance surface.
		 y_e : numpy.ndarray (2,)
			Minimum/maximum y-coordinates of the entrance surface.
		 z_e : float
			z-coordinate of the entrance surface.
		 z_start
			z-coordinate of the start surface.
		 tol : float, default=0.001
		 	Value added to the result to account for the start surface discretization.
		 N : int, default=100
			Number of values over x and y used to construct the initial start surface sampling grid. 
	
	Returns
	-------
		sangle_max : float
			Value of the maximum solid angle enclosing CS as seen from the initial start surface.
	"""
	sangle_max = 0

	x_e_at_start, y_e_at_start = transport_entrance_to_start_surface(x_cs, y_cs, z_cs, x_e, y_e, z_e, z_start) # Move sampling surface at z_start
	x = np.linspace(*x_e_at_start, N)
	y = np.linspace(*y_e_at_start, N)

	pbar = tqdm(total=len(x)*len(y))
	pbar.set_description("Computinf max solid angle")
	for i in range(len(x)):
		for j in range(len(y)):
			# compute min/max elevation on plate
			az_min, az_max, el_min, el_max  = compute_az_el_bounds(x_cs - x[i], y_cs - y[j], z_cs - z_start)

			sangle = (az_max - az_min)*(np.sin(el_max) - np.sin(el_min))
			if sangle > sangle_max: 
				sangle_max = sangle

			pbar.update(1)
	pbar.close()

	return sangle_max + tol


def integrate_solid_angle_grid(x_cs, y_cs, z_cs, x_e, y_e, z_e, z_start, N=100, n_sub_samples=1000):
	""" Integrates the true CS solid angle on the start surface using a hybrid grid/MC integration scheme.

	This function integrates the true CS solid angle on the start surface using a hybrid grid/MC integration scheme.
  	 
	Parameters
	----------
	x_cs : numpy.ndarray (2,)
		Minimum/maximum x-coordinates of the conversion surface.
	y_cs : float
		y-coordinate of the conversion surface.
	z_cs : numpy.ndarray (2,)
		Minimum/maximum z-coordinates of the conversion surface.
	x_e : numpy.ndarray (2,)
		Minimum/maximum x-coordinates of the entrance surface.
	y_e : numpy.ndarray (2,)
		Minimum/maximum y-coordinates of the entrance surface.
	z_e : float
		z-coordinate of the entrance surface.
	z_start
		z-coordinate of the start surface.
	N : int, default=100
		Number of values over x and y used to construct the initial start surface sampling grid. 
	n_sub_samples : int, default=1000
		Number of MC samples used to integrate the solid angle at each grid point.

	Returns
	-------
		integral : float
			Integral of the CS solid angle on the start surface.
	"""
	# Move sampling surface at z_start
	x_e_at_start, y_e_at_start = transport_entrance_to_start_surface(x_cs, y_cs, z_cs, x_e, y_e, z_e, z_start)
	dx0 = np.diff(x_e_at_start)/N
	dy0 = np.diff(y_e_at_start)/N

	sangle_max = compute_max_solid_angle(x_cs, y_cs, z_cs, x_e, y_e, z_e, z_start)
	integral = 0

	pbar = tqdm(total=N**2)
	pbar.set_description("Integrating solid angle on surface (grid)")	
	for i in range(N):
		for j in range(N):
			x0 = x_e_at_start[0] + i*dx0
			y0 = y_e_at_start[0] + j*dy0
			z0 = z_start

			# compute az el bounds
			az_min, az_max, el_min, el_max  = compute_az_el_bounds(x_cs - z0, y_cs - y0, z_cs - z0)

			# compute min/max elevation on plate
			omega_s_ij = (az_max - az_min)*(np.sin(el_max) - np.sin(el_min))

			# sample the solid angles multiple times and count number of accepted values
			n_sampled = 0
			n_accepted = 0
			while n_accepted < n_sub_samples:		
				n_sampled += 1

				if n_sampled > n_sub_samples*10 and n_accepted == 0: break # useless to continue, negligeable
				if np.random.uniform(0, sangle_max) > omega_s_ij: continue

				az, el = sample_sphere_simion(az_min, az_max, el_min, el_max) # compute azimuth and elevation
				x, y, z = compute_direction(az, el, x0, y0, z0) # compute direction

				# check if the point intersects both cs and entrance
				x_at_cs, z_at_cs = transport_to_cs(x, y, z, x0, y0, z0, y_cs)
				if x_at_cs <= x_cs[0] or x_at_cs >= x_cs[1] or z_at_cs <= z_cs[0] or z_at_cs >= z_cs[1]: continue
				x_at_e, y_at_e = transport_to_entrance(x, y, z, x0, y0, z0, z_e)
				if x_at_e <= x_e[0] or x_at_e >= x_e[1] or y_at_e <= y_e[0] or y_at_e >= y_e[1]: continue

				n_accepted += 1

			integral += omega_s_ij * n_accepted/n_sampled * dx0 * dy0
			
			pbar.update(1)
	pbar.close()

	# omega_arr = np.array(omega_arr)
	# mean_solid_angle = np.mean(omega_arr)
	# std_solid_angle = np.sqrt(1/(n_generated_particles - 1)*np.sum((omega_arr - mean_solid_angle)**2))

	return integral * 1e-2

def integrate_solid_angle_mc(x_cs, y_cs, z_cs, x_e, y_e, z_e, z_start, n_samples=10000):
	""" Integrates the true CS solid angle on the start surface using a MC integration scheme.

	This function integrates the true CS solid angle on the start surface using a MC integration scheme.
  	 
	Parameters
	----------
	x_cs : numpy.ndarray (2,)
		Minimum/maximum x-coordinates of the conversion surface.
	y_cs : float
		y-coordinate of the conversion surface.
	z_cs : numpy.ndarray (2,)
		Minimum/maximum z-coordinates of the conversion surface.
	x_e : numpy.ndarray (2,)
		Minimum/maximum x-coordinates of the entrance surface.
	y_e : numpy.ndarray (2,)
		Minimum/maximum y-coordinates of the entrance surface.
	z_e : float
		z-coordinate of the entrance surface.
	z_start
		z-coordinate of the start surface.
	n_samples : int, default=1000
		Total number of MC samples used to integrate the solid angle.

	Returns
	-------
		integral : float
			Integral of the CS solid angle on the start surface.
	"""
	integral = 0
	n_sampled = 0
	n_tot = 0
	omega_arr = []
	
	# Move sampling surface at z_start
	x_e_at_start, y_e_at_start = transport_entrance_to_start_surface(x_cs, y_cs, z_cs, x_e, y_e, z_e, z_start)
	S0 = np.diff(x_e_at_start)*np.diff(y_e_at_start)*1e-2

	sangle_max = compute_max_solid_angle(x_cs, y_cs, z_cs, x_e, y_e, z_e, z_start)

	pbar = tqdm(total=n_samples)
	pbar.set_description("Integrating solid angle on surface (MC)")
	while n_sampled < n_samples:
		n_tot += 1

		# Sample random point on conversion surface
		x0 = np.random.uniform(*x_e_at_start)
		y0 = np.random.uniform(*y_e_at_start)
		z0 = z_start

		# compute min/max elevation on plate
		dx = x_cs - x0
		dy = y_cs - y0
		dz = z_cs - z0

		az_min, az_max, el_min, el_max  = compute_az_el_bounds(dx, dy, dz)

		omega_i = (az_max - az_min)*(np.sin(el_max) - np.sin(el_min))
		v = np.random.uniform(0, sangle_max)
		if v > omega_i:
			continue

		#compute azimuth and elevation
		az, el = sample_sphere_simion(az_min, az_max, el_min, el_max)
		x, y, z = compute_direction(az, el, x0, y0, z0) # compute direction

		# check if the point intersects both cs and entrance
		x_at_cs, z_at_cs = transport_to_cs(x, y, z, x0, y0, z0, y_cs)
		if x_at_cs <= x_cs[0] or x_at_cs >= x_cs[1] or z_at_cs <= z_cs[0] or z_at_cs >= z_cs[1]: continue
		x_at_e, y_at_e = transport_to_entrance(x, y, z, x0, y0, z0, z_e)
		if x_at_e <= x_e[0] or x_at_e >= x_e[1] or y_at_e <= y_e[0] or y_at_e >= y_e[1]: continue

		integral += omega_i
		omega_arr += [omega_i]

		n_sampled += 1

		pbar.update(1)
	pbar.close()

	omega_arr = np.array(omega_arr)
	integral = integral*S0/n_tot
	std_integral = np.sqrt(1/(n_tot - 1) * np.sum((omega_arr - np.mean(omega_arr))**2))
	std_integral = S0 * std_integral/np.sqrt(n_tot)

	# print(f"Integral = {integral[0]} +/- {2*std_integral[0]}")

	return integral[0], std_integral[0]


def plot_N_to_J_factor(x_cs, y_cs, z_cs, x_e, y_e, z_e, z_start, N_min_log=1, N_max_log=6, n_N_points=10):
	""" Computes and plots the N/JdtdE ratio using the hybrid grid/MC and full MC integration schemes.

	This function computes and plots the N/JdtdE ratio using the hybrid grid/MC and full MC integration schemes.
  	 
	Parameters
	----------
	x_cs : numpy.ndarray (2,)
		Minimum/maximum x-coordinates of the conversion surface.
	y_cs : float
		y-coordinate of the conversion surface.
	z_cs : numpy.ndarray (2,)
		Minimum/maximum z-coordinates of the conversion surface.
	x_e : numpy.ndarray (2,)
		Minimum/maximum x-coordinates of the entrance surface.
	y_e : numpy.ndarray (2,)
		Minimum/maximum y-coordinates of the entrance surface.
	z_e : float
		z-coordinate of the entrance surface.
	z_start
		z-coordinate of the start surface.
	N_min_log : int, default=1
		Min number of points (log10) sampled on the initial start surface.
	N_max_log : int, default=1
		Max number of points (log10) sampled on the initial start surface.
	n_N_points : int, default=1
		Number of N point values used to plot the integrals' convergence.  

	Returns
	-------
		None
	"""
	int_omega_mc = []
	int_omega_grid = []

	N 		= np.logspace(N_min_log, N_max_log, n_N_points).astype(int)
	data 	= np.ndarray(n_N_points, 3)

	for i in range(n_N_points):
		int_omega_grid 	+= [integrate_solid_angle_grid(x_cs, y_cs, z_cs, x_e, y_e, z_e, z_start, N=20, n_sub_samples=int(N[i]/(20*20)))]
		int_omega_mc 	+= [integrate_solid_angle_mc(x_cs, y_cs, z_cs, x_e, y_e, z_e, z_start, n_samples=N[i])[0]]

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.loglog(N, int_omega_grid, "-k", label="Rectangular integration (grid)")
	ax.loglog(N, int_omega_mc, "--b", label="MC integration")
	ax.set_xlabel("$N_{sim}$ [-]")
	ax.set_ylabel(r"$N_{sim} \; / \; \left(\int J_{ENA} dE_0 \cdot \Delta t \right)$   [$sr \cdot m^2$]")
	ax.legend()


def sample_square_opening(
	e_min, 
	e_max, 
	x_cs, 
	y_cs, 
	z_cs, 
	x_e, 
	y_e, 
	z_e, 
	z_start, 
	mass, 
	charge, 
	n_particles,
	az_bounds=None,
	el_bounds=None, 
	tol=1, 
	tol_entrance=.05
):
	""" Samples N points from the inital start surface within the FOV of the instrument.

	This function samples N points from the inital start surface within the FOV of the instrument.
  	 
	Parameters
	----------
	x_cs : numpy.ndarray (2,)
		Minimum/maximum x-coordinates of the conversion surface.
	y_cs : float
		y-coordinate of the conversion surface.
	z_cs : numpy.ndarray (2,)
		Minimum/maximum z-coordinates of the conversion surface.
	x_e : numpy.ndarray (2,)
		Minimum/maximum x-coordinates of the entrance surface.
	y_e : numpy.ndarray (2,)
		Minimum/maximum y-coordinates of the entrance surface.
	z_e : float
		z-coordinate of the entrance surface.
	z_start
		z-coordinate of the start surface.
	mass : float
		Mass of the sampled particles.
	charge : float
		Charge of the sampled particles.
	n_particles : int
		Number of sampled particles within the FOV of the instrument.
	tol : float
		Tolerance used for start surface sampling.
	tol_entrance : float
		Tolerance used for surface intersection sample rejection.

	Returns
	-------
		data : numpy.ndarray (n_particles, 9)
			- data[0] -> TBD
			- data[1] -> Particle mass
			- data[2] -> Particle charge
			- data[3] -> Particle Initial X position
			- data[4] -> Particle Initial Y position
			- data[5] -> Particle Initial Z position
			- data[6] -> Particle Initial azimuth
			- data[7] -> Particle Initial elevation
			- data[8] -> Particle Initial energy
	"""
	data = np.ndarray((n_particles, 9))
	
	particle_id = 0
	n_generated_particles = 0
	mean_solid_angle = 0

	x_e_at_start, y_e_at_start = transport_entrance_to_start_surface(x_cs, y_cs, z_cs, x_e, y_e, z_e, z_start)
	sangle_max = compute_max_solid_angle(x_cs, y_cs, z_cs, x_e, y_e, z_e, z_start)

	pbar = tqdm(total=n_particles)
	pbar.set_description("Sampling FOV")
	while particle_id < n_particles:
		n_generated_particles += 1

		# Sample random point on conversion surface
		x0 = np.random.uniform(*(np.array([-tol, tol]) + x_e_at_start))
		y0 = np.random.uniform(*(np.array([-tol, tol]) + y_e_at_start))
		z0 = z_start

		# compute min/max elevation on plate
		dx = x_cs - x0
		dy = y_cs - y0
		dz = z_cs - z0

		az_min, az_max, el_min, el_max  = compute_az_el_bounds(dx, dy, dz)
		if az_bounds is not None:
			az_min, az_max = az_bounds
		if el_bounds is not None:
			el_min, el_max = el_bounds

		mean_solid_angle += (az_max - az_min)*(np.sin(el_max) - np.sin(el_min))

		v = np.random.uniform(0, sangle_max)
		if v > (az_max - az_min)*(np.sin(el_max) - np.sin(el_min)):
			continue

		#compute azimuth and elevation
		az, el = sample_sphere_simion(az_min, az_max, el_min, el_max)
		x, y, z = compute_direction(az, el, x0, y0, z0) # compute direction

		# check if the point intersects both cs and entrance
		x_at_cs, z_at_cs = transport_to_cs(x, y, z, x0, y0, z0, y_cs)
		if x_at_cs <= x_cs[0] or x_at_cs >= x_cs[1] or z_at_cs <= z_cs[0] or z_at_cs >= z_cs[1]: continue
		x_at_e, y_at_e = transport_to_entrance(x, y, z, x0, y0, z0, z_e)
		if x_at_e <= x_e[0] or x_at_e >= x_e[1] or y_at_e <= y_e[0] or y_at_e >= y_e[1]: continue

		data[particle_id, 0] = 0
		data[particle_id, 1] = mass
		data[particle_id, 2] = charge
		data[particle_id, 3] = x0
		data[particle_id, 4] = y0
		data[particle_id, 5] = z0
		data[particle_id, 6] = az*180/np.pi
		data[particle_id, 7] = el*180/np.pi 
		data[particle_id, 8] = np.random.uniform(e_min, e_max)

		pbar.update(1)
		particle_id += 1

	pbar.close()

	return data

def main(x_cs, y_cs, z_cs, x_e, z_e, z_start, ion_cache_file="init_ions.npy", n_particles=100000):
	if not os.path.exists(ion_cache_file):
		data = sample_square_opening(
			10, 
			10, 
			x_cs, 
			y_cs, 
			z_cs, 
			x_e, 
			y_e, 
			z_e, 
			z_start, # z_start
			1.00,
			0.0,
			n_particles, 
			tol=10
		)
		np.save(ion_cache_file, data)
	else:
		data = np.load(ion_cache_file)

	# plot results
	az = data[:, 6]*np.pi/180
	el = data[:, 7]*np.pi/180
	x0 = data[:, 3]
	y0 = data[:, 4]
	z0 = data[:, 5]

	r = 200
	vecs = compute_direction(az, el, x0, y0, z0)*r
	x = vecs[0]
	y = vecs[1]
	z = vecs[2]
	
	x_at_cs, z_at_cs = transport_to_cs(x, y, z, x0, y0, z0, y_cs)
	x_at_cs = x_at_cs - np.mean(x_cs)
	x_at_cs = np.append(x_at_cs, np.flip(x_at_cs))
	z_at_cs = z_at_cs - z_cs[0]
	z_at_cs = np.append(z_at_cs, z_at_cs)

	x_at_e, y_at_e = transport_to_entrance(x, y, z, x0, y0, z0, z_e)

	fig = plt.figure()
	ax = fig.add_subplot(111)
	hist = ax.hist2d(x_at_cs, z_at_cs, bins=[int(np.diff(x_cs)/2), int(np.diff(z_cs)/2)])
	cb = fig.colorbar(hist[3], ax=ax, pad=0.07)
	cb.set_label('Counts [-]')
	fig.suptitle("Conversion surface hits")
	ax.set_xlabel("$x$ [mm]")
	ax.set_ylabel("$z$ [mm]")

	# scatter plot
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.plot(x_at_cs, z_at_cs, "+b", markersize=2)
	fig.suptitle("Conversion surface hits")
	ax.set_xlabel("$x$ [mm]")
	ax.set_ylabel("$z$ [mm]")

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.hist(data[:, 6] + 90, color="blue", bins=35)
	fig.suptitle("Conversion surface hits")
	ax.set_xlabel("Azimuth [deg]")
	ax.set_ylabel("CS Counts [-]")

	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.hist(-data[:, 7], color="blue", bins=35)
	fig.suptitle("Conversion surface hits")
	ax.set_xlabel("Elevation [deg]")
	ax.set_ylabel("CS Counts [-]")


def compare_both_methods(x_cs, y_cs, z_cs, x_e, z_e, z_start):
	""" Compares the two FOV sampling method results (2pi sampling and rejections sampling).

	This function computes and plots the results of the two FOV sampling methods implemented in 
	this script.

	Note: 
		The function plots the results cached within the .npy files, therefore the two 
		sampling methods must be ran beforehand.

	Parameters
	----------
		x_cs : numpy.ndarray (2,)
			Minimum/maximum x-coordinates of the conversion surface.
		y_cs : float
			y-coordinate of the conversion surface.
		z_cs : numpy.ndarray (2,)
			Minimum/maximum z-coordinates of the conversion surface.
		x_e : numpy.ndarray (2,)
			Minimum/maximum x-coordinates of the entrance surface.
		y_e : numpy.ndarray (2,)
			Minimum/maximum y-coordinates of the entrance surface.
		z_e : float
			z-coordinate of the entrance surface.
		z_start
			z-coordinate of the start surface.

	Returns
	-------
		None
	"""

	ion_cache_files = ["init_ions_2pi.npy", "init_ions_rej_omega.npy"]
	labels = [r"$2\pi$ sampling", r"$\Omega/\Omega_{max}$ sampling"]
	colors = ["blue", "red"]

	fig = plt.figure()
	ax11 = fig.add_subplot(221)
	ax12 = fig.add_subplot(222)
	ax21 = fig.add_subplot(223)
	ax22 = fig.add_subplot(224)

	fig.suptitle("Conversion surface hits")
	ax11.set_xlabel("$x$ [mm]")
	ax11.set_ylabel("$z$ [mm]")
	ax12.set_xlabel("$x$ [mm]")
	ax12.set_ylabel("$z$ [mm]")
	ax21.set_xlabel("Azimuth [deg]")
	ax21.set_ylabel("CS Counts [-]")
	ax22.set_xlabel("Elevation [deg]")
	ax22.set_ylabel("CS Counts [-]")

	for i, file in enumerate(ion_cache_files):
		data = np.load(file)

		az = data[:, 6]*np.pi/180
		el = data[:, 7]*np.pi/180
		x0 = data[:, 3]
		y0 = data[:, 4]
		z0 = data[:, 5]

		dir_ = compute_direction(az, el, x0, y0, z0) # compute direction
	
		x_at_cs, z_at_cs = transport_to_cs(*dir_, x0, y0, z0, y_cs)
		x_at_cs = x_at_cs - np.mean(x_cs)
		x_at_cs = np.append(x_at_cs, np.flip(x_at_cs))
		z_at_cs = z_at_cs - z_cs[0]
		z_at_cs = np.append(z_at_cs, z_at_cs)

		x_at_e, y_at_e = transport_to_entrance(*dir_, x0, y0, z0, z_e)

		if i == 0:
			ax = ax11
		else:
			ax = ax12
		
		ax.set_title(labels[i])
		hist = ax.hist2d(x_at_cs, z_at_cs, bins=[int(np.diff(x_cs)/2), int(np.diff(z_cs)/2)])
		cb = fig.colorbar(hist[3], ax=ax, pad=0.08)
		cb.set_label('Counts [-]')
		
		# scatter plot
		ax21.hist(data[:, 6]+90, color=colors[i], bins=35, label=labels[i], histtype='step')
		ax22.hist(-data[:, 7], color=colors[i], bins=35, label=labels[i], histtype='step')

	ax21.legend()
	ax22.legend()

if __name__ == '__main__':
	ion_cache_file = "init_ions.npy"

	x_cs = np.array([38, 102])
	y_cs = 51.5
	z_cs = np.array([68.5, 100])
	x_e = np.array([40, 100])
	y_e = np.array([51.5, 66.5])
	z_e = 17
	z_start = 0.1

	# Sample FOV
	N = 1000000
	main(x_cs, y_cs, z_cs, x_e, z_e, z_start, ion_cache_file="init_ions.npy", n_particles=N)


	# Compare the two FOV computation methods
	N = 100000

	# 1st method - 2pi sampling
	if not os.path.exists("init_ions_2pi.npy"):
		data = sample_square_opening(10, 10, x_cs, y_cs, z_cs, x_e, y_e, z_e, z_start, 1.00, 0.0, N, az_bounds=[-np.pi, 0], el_bounds=[-np.pi/2, np.pi/2], tol=1)
		np.save("init_ions_2pi.npy", data)
	else:
		data = np.load("init_ions_2pi.npy")

	# 2nd method - rejection sampling
	if not os.path.exists("init_ions_rej_omega.npy"):
		data = sample_square_opening(10, 10, x_cs, y_cs, z_cs, x_e, y_e, z_e, z_start, 1.00, 0.0, N, tol=1)
		np.save("init_ions_rej_omega.npy", data)
	else:
		data = np.load("init_ions_rej_omega.npy")

	# Plot both results
	compare_both_methods(x_cs, y_cs, z_cs, x_e, z_e, z_start)


	# Compare Omega.dS integral method results
	# plot_N_to_J_factor(x_cs, y_cs, z_cs, x_e, z_e, z_start, N_min_log=1, N_max_log=5, n_N_points=15)

	# Compute kappa ratio and uncertainty
	N = 100000 # only 10^5 points needed
	integral, std = integrate_solid_angle_mc(x_cs, y_cs, z_cs, x_e, y_e, z_e, z_start, n_samples=N)
	print(f"N/JdtdE = {integral} +/- {2*std}")

	plt.show()