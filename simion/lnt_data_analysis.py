from subprocess import run
import sys
import os
import numpy as np
import math
import matplotlib.pyplot as plt 
from random import uniform
import csv
import os.path
import re
from scipy.optimize import curve_fit
from matplotlib import gridspec
from scipy.signal import argrelextrema
import time
from glob import glob
import tqdm
import scipy.stats

import simion.io
import simion

def sort_nicely(l):
	""" Sort the given list in the way that humans expect.
	"""

	def alphanum_key(s):
		""" Turn a string into a list of string and number chunks.
			"z23a" -> ["z", 23, "a"]
		"""
		def tryint(s):
			try:
				return int(s)
			except:
				return s
				
		return [ tryint(c) for c in re.split('([0-9]+)', s) ]

	l.sort(key=alphanum_key)

def get_sorted_log_file_list(suffix):
	file_list = glob(suffix)
	sort_nicely(file_list)

	return file_list

def add_data_to_array(array, data):
	if array is None:
		array = data
	else:
		array = np.append(array, data)
	return array


def get_logfiles_data(logfile_path, suffix, headers, energy, subenergy, only_success=False):
	if not isinstance(headers, list):
		headers = list(headers)

	arrays = [None]*(len(headers) + 1)

	# get all log files from log folder
	if subenergy is None:
		file_list = get_sorted_log_file_list(f"{logfile_path}/*_KE_*_{math.floor(energy)}_*{suffix}.csv")
	else:
		file_list = get_sorted_log_file_list(f"{logfile_path}/*_KE_*_{math.floor(energy)}_{str(math.floor(subenergy))}{suffix}.csv")

	if len(file_list) == 0:
		raise Exception("No files found in 'get_logfiles_data' for the given parameters: " + f"{logfile_path}/*_KE_*_{math.floor(energy)}_{str(math.floor(subenergy))}{suffix}.csv")

	# read file content
	for file in file_list:
		header, data = simion.io.read_logfile(file)
		if only_success is True:
			if len(data) > 0:
				data = keep_only_successful_hits(logfile_path, data[:, header['Ion N']], data, energy, subenergy)
				
				if len(data) == 0:
					print(f"No successful hits found in {file}, skipping file.")
			else:
				print(f"No data found in {file}, skipping file.")

		if len(data) > 1:
			arrays[0] = add_data_to_array(arrays[0], data[:, header['Ion N']])

			for i in range(1, len(arrays)):
				arrays[i] = add_data_to_array(arrays[i], data[:, header[headers[i-1]]])

	return arrays


def get_all_energy_settings_and_initial_energies(logfile_path):
	file_list = get_sorted_log_file_list(f"{logfile_path}/*_start.csv")
	if len(file_list) == 0:
		raise Exception("No simulation results found in the specified folder.")

	energy_settings = np.sort(list(set([int(filename.split('_')[-3]) for filename in file_list])))
	initial_energies = [np.sort(list(set([int(filename.split('_')[-2]) for filename in file_list if int(filename.split('_')[-3]) == energy_setting]))).tolist() for energy_setting in energy_settings]

	return energy_settings, initial_energies

def keep_only_successful_hits(logfile_path, ion_inds, data, energy, subenergy):
	ion_inds_hits, hits_start, hits_stop = get_logfiles_data(logfile_path, "_hits", ['Hit Start Surface', 'Hit Stop Surface'], energy, subenergy, only_success=False)
	
	mask_signal = np.logical_and(hits_start == 1, hits_stop == 1)
	inds_signal = np.where(np.in1d(ion_inds, ion_inds_hits[mask_signal]))[0]

	data = data[inds_signal, :]

	return data

def generate_hitmap_fig_names(output_folder, surface_type, energy, subenergy, only_success):
	# fig names
	if only_success is True:
		fig_title = f'Ion {surface_type} surface success countrate\n($E_b$ = {str(math.floor(energy))}eV energy setting)'
		fig_filename = output_folder + f'LNT_WAVE_ENA_PERFORMANCE_SUCCESS_HIT_MAP_CONTOUR_{surface_type}_ES_' + str(math.floor(energy))
		fig_filename_scatter = output_folder + f'LNT_WAVE_ENA_PERFORMANCE_SUCCESS_HIT_MAP_SCATTER_{surface_type}_ES_' + str(math.floor(energy))
	else:
		fig_title = f'Ion {surface_type} surface countrate\n($E_b$ = {str(math.floor(energy))}eV energy setting)'
		fig_filename = output_folder + f'LNT_WAVE_ENA_PERFORMANCE_HIT_MAP_CONTOUR_{surface_type}_ES_' + str(math.floor(energy))
		fig_filename_scatter = output_folder + f'LNT_WAVE_ENA_PERFORMANCE_HIT_MAP_SCATTER_{surface_type}_ES_' + str(math.floor(energy))

	
	if subenergy is not None:
		fig_filename_scatter += '_E0_' + str(math.floor(subenergy))
		fig_filename += '_E0_' + str(math.floor(subenergy))
		fig_title += f' @ $E_0$ = {math.floor(subenergy)}eV'

	return fig_title, fig_filename, fig_filename_scatter


def compute_false_detection_rate(tmpfile_path, energy_settings):
	false_detections 		= np.zeros(len(energy_settings))
	total_stop_surface_hits = np.zeros(len(energy_settings))

	for energy_setting_id, energy_setting in enumerate(energy_settings):
		# tmp_hit_files = get_sorted_log_file_list(f"{tmpfile_path}hits_KE_*_{math.floor(energy_setting)}_{str(math.floor(initial_particle_energies[se]))}.txt")
		tmp_hit_files = get_sorted_log_file_list(f"{tmpfile_path}hits_KE_*_{math.floor(energy_setting)}_*.txt")
		initial_particle_energies = np.sort(np.array([tmp_hit_file.split('_')[-1].split('.')[0] for tmp_hit_file in tmp_hit_files]).astype(int))

		for initial_energy_id, initial_particle_energy in enumerate(initial_particle_energies):
			tmp_hit_files = get_sorted_log_file_list(f"{tmpfile_path}hits_KE_*_{math.floor(energy_setting)}_{str(math.floor(initial_particle_energy))}.txt")

			for file in tmp_hit_files:
				subhits = simion.io.read_tempfile(file)

				total_stop_surface_hits[energy_setting_id] 	+= subhits[9] # Start and stop hits

				
				total_stop_surface_hits[energy_setting_id] 	+= subhits[3] + subhits[4] # only start CEM hits

		# factor 2 because of system symmetry
		integrated_coinc_hits[energy_setting_id] 	= np.sum(coinc_hits)
		integrated_start_hits[energy_setting_id] 	= np.sum(start_hits)
		integrated_stop_hits[energy_setting_id] 	= np.sum(stop_hits)

	# compute snr
	snr = np.sqrt(integrated_coinc_hits)*np.sqrt(2) / np.sqrt(integrated_start_hits + integrated_stop_hits)
	print("False detection rate : ")
	print("    Energy bands : 		", energy_settings)
	print("    SNR : 				", snr)

def contour_plot_start(logfile_path, output_folder, energy, subenergy, only_success=False, scatter=True):
	""" Read the .csv and do a contour plot of the hitmap for one subenergy and plot it."""
	x_min 		= 0
	x_max 		= 60
	z_min 		= 0
	z_max 		= 27

	center_x 	= int((x_max + x_min)/2)
	bins_x 		= np.arange(x_min-center_x, x_max-center_x + 1, 0.5)
	bins_z		= np.arange(z_min, z_max + 1, 0.5)

	# fig names
	fig_title, fig_filename, fig_filename_scatter = generate_hitmap_fig_names(output_folder, "START", energy, subenergy, only_success)

	ion_ids_xz, x, z = get_logfiles_data(logfile_path, "_start", ['X Ion Start Splat[mm]', 'Z Ion Start Splat[mm]'], energy, subenergy, only_success=only_success)

	if x is not None and z is not None:
		# use x symetry to duplicated data
		x = np.append(x, 2*center_x - x)
		z = np.append(z, z)

		fig = plt.figure()
		ax = fig.add_subplot(111)
		hist = ax.hist2d(x - center_x, z, bins=[bins_x, bins_z], range=[[x_min-center_x, x_max-center_x], [z_min, z_max]])
		cb = fig.colorbar(hist[3], ax=ax)
		cb.set_label('Counts [-]')
		ax.set_title(fig_title)
		ax.set_xlabel('$x$ [mm]')
		ax.set_ylabel('$z$ [mm]')
		ax.set_xlim([x_min-center_x, x_max-center_x])
		ax.set_ylim([z_min, z_max])  
		plt.savefig(fig_filename + ".png", format="png", dpi=400)
		plt.close(fig)

		if scatter is True:
			xz = np.vstack([x - center_x,  z])
			density = scipy.stats.gaussian_kde(xz)(xz)
			
			fig = plt.figure()
			ax = fig.add_subplot(111)
			scatter_plot = ax.scatter(x - center_x,  z, c=density, s=2, marker="o")
			cb = fig.colorbar(scatter_plot, label="Density [-]")
			ax.set_title(fig_title)
			ax.set_xlabel('$x$ [mm]')
			ax.set_ylabel('$z$ [mm]')
			ax.set_xlim([x_min-center_x, x_max-center_x])
			ax.set_ylim([z_min, z_max])  
			plt.savefig(fig_filename_scatter + ".png", format="png", dpi=400)
			plt.close(fig)


def contour_plot_start_velocity(logfile_path, output_folder, energy, subenergy, only_success=False):
	""" Read the .csv and do a contour plot of the hitmap for one subenergy and plot it."""
	x_min 		= 0
	x_max 		= 60
	z_min 		= 0
	z_max 		= 27

	center_x 	= int((x_max + x_min)/2)
	bins_x 		= np.arange(x_min-center_x, x_max-center_x + 1, 0.5)
	bins_z		= np.arange(z_min, z_max + 1, 0.5)
	X, Z = np.meshgrid(bins_x, bins_z)

	# fig names
	fig_title, fig_filename, fig_filename_scatter = generate_hitmap_fig_names(output_folder, "START_2D_VELOCITY", energy, subenergy, only_success)

	ion_ids_xz, x, z, vx, vy, vz = get_logfiles_data(logfile_path, "_start", ['X Ion Start Splat[mm]', 'Z Ion Start Splat[mm]', 'Vx Ion Start Splat[mm/usec]', 'Vy Ion Start Splat[mm/usec]', 'Vz Ion Start Splat[mm/usec]'], energy, subenergy, only_success=only_success)

	if x is not None and z is not None:
		v_norm = np.max(np.sqrt(vx**2 + vy**2 + vz**2))
		vx = vx/v_norm
		vz = vz/v_norm

		x = x - center_x

		# use x symetry to duplicated data
		x = np.append(x, -x)
		z = np.append(z, z)
		vx = np.append(vx, -vx)
		vz = np.append(vz, vz)


		vx_grid = np.zeros((len(bins_z), len(bins_x)))
		vz_grid = np.zeros((len(bins_z), len(bins_x)))
		for zi in range(len(X)-1):
			for xi in range(len(X[0])-1):
				msk = np.logical_and(np.logical_and(x >= X[zi, xi], x < X[zi, xi+1]), np.logical_and(z >= Z[zi, xi], z < Z[zi+1, xi]))
				if len(np.where(msk)[0]) > 0:
					vx_grid[zi, xi] = np.mean(vx[msk])
					vz_grid[zi, xi] = np.mean(vz[msk])

		fig = plt.figure()
		ax = fig.add_subplot(111)
		# ax.plot(x, z, "+b")
		hist = ax.hist2d(x, z, bins=[bins_x, bins_z], range=[[x_min-center_x, x_max-center_x], [z_min, z_max]])
		ax.quiver(X+np.diff(bins_x)[0]/2, Z+np.diff(bins_z)[0]/2, vx_grid, vz_grid, scale=30, color="cyan", headwidth=8, headlength=8, width=0.00075)
		cb = fig.colorbar(hist[3], ax=ax)
		cb.set_label('Counts [-]')
		ax.set_title(fig_title)
		ax.set_xlabel('$x$ [mm]')
		ax.set_ylabel('$z$ [mm]')
		ax.set_xlim([x_min-center_x, x_max-center_x])
		ax.set_ylim([z_min, z_max])  
		plt.savefig(fig_filename + ".png", format="png", dpi=400)
		plt.close(fig)


def contour_plot_start_all(logfile_path, output_folder, energy_settings, initial_particle_energies, only_success=False, scatter=True):
	""" Read the .csv and do a contour plot of the hitmap for one subenergy and plot it."""
	x_min 		= 0
	x_max 		= 60
	z_min 		= 0
	z_max 		= 27

	center_x 	= int((x_max + x_min)/2)
	bins_x 		= np.arange(x_min-center_x, x_max-center_x + 1, 0.5)
	bins_z		= np.arange(z_min, z_max + 1, 0.5)

	plt.rc('font', size=3)

	plt.rc("axes", linewidth=.25)
	plt.rc('xtick.major', size=2, width=.25)
	plt.rc('xtick.minor', size=1, width=.25)

	plt.rc('ytick.major', size=2, width=.25)
	plt.rc('ytick.minor', size=1, width=.25)


	# fig names
	fig_title = f'Ion START surface success countrates'
	fig_filename = output_folder + f'LNT_WAVE_ENA_PERFORMANCE_SUCCESS_HIT_MAP_CONTOUR_START'
	fig_filename_scatter = output_folder + f'LNT_WAVE_ENA_PERFORMANCE_SUCCESS_HIT_MAP_SCATTER_START'
	cache_file = output_folder + f'lnt_start_hist_cache'

	n_initial_energies = max([len(sub_initial_energies) for sub_initial_energies in initial_particle_energies])
	fig, axes = plt.subplots(len(energy_settings), n_initial_energies+1, dpi=50)
	fig.suptitle(fig_title)

	hists = []
	if not os.path.exists(cache_file + ".pickle"):
		for energy_setting_id, energy_setting in enumerate(energy_settings):
			print(f"Processing energy setting {energy_setting_id+1}/{len(energy_settings)}")

			hists.append([])
			for initial_energy_id, initial_energy in enumerate(initial_particle_energies[energy_setting_id]):			
				ion_ids_xz, x, z = get_logfiles_data(logfile_path, "_start", ['X Ion Start Splat[mm]', 'Z Ion Start Splat[mm]'], energy_setting, initial_energy, only_success=only_success)

				if x is not None and z is not None:
					# use x symetry to duplicated data
					x = np.append(x, 2*center_x - x)
					z = np.append(z, z)

					H, xedges, yedges = np.histogram2d(x - center_x, z, bins=[bins_x, bins_z], range=[[x_min-center_x, x_max-center_x], [z_min, z_max]])
					hists[energy_setting_id].append(H)
				else:
					hists[energy_setting_id].append(None)

			ion_ids_xz, x, z = get_logfiles_data(logfile_path, "_start", ['X Ion Start Splat[mm]', 'Z Ion Start Splat[mm]'], energy_setting, None, only_success=only_success)

			if x is not None and z is not None:
				# use x symetry to duplicated data
				x = np.append(x, 2*center_x - x)
				z = np.append(z, z)

				H, xedges, yedges = np.histogram2d(x - center_x, z, bins=[bins_x, bins_z], range=[[x_min-center_x, x_max-center_x], [z_min, z_max]])
				hists[energy_setting_id].append(H)
			else:
				hists[energy_setting_id].append(None)

		simion.io.write_pickle(cache_file, hists)
	else:
		print("Using cache file", cache_file)
		hists = simion.io.read_pickle(cache_file)		


	for energy_setting_id, energy_setting in enumerate(energy_settings):
		# normalize plot along all initial energies for the given energy setting 
		vmax = np.max([np.max(hists[energy_setting_id][i]) for i in range(len(hists[energy_setting_id])-1) if hists[energy_setting_id][i] is not None])

		for i in range(len(hists[energy_setting_id])):
			axes[energy_setting_id][i].tick_params(axis="both", which="both", labelright=False, labelleft=False, labelbottom=False, labeltop=False)
			
			if energy_setting_id == 0:
				axes[energy_setting_id][i].tick_params(axis="both", which="both", labeltop=True)
			if energy_setting_id == len(energy_settings)-1:
				axes[energy_setting_id][i].tick_params(axis="both", which="both", labelbottom=True)
			if i == 0:
				axes[energy_setting_id][i].tick_params(axis="both", which="both", labelleft=True)
			if i == len(hists[energy_setting_id])-1:
				axes[energy_setting_id][i].tick_params(axis="both", which="both", labelright=True)

			if hists[energy_setting_id][i] is not None:
				if i < len(hists[energy_setting_id])-1:
					im = axes[energy_setting_id][i].imshow(np.rot90(hists[energy_setting_id][i]), vmin=0, vmax=vmax, extent=[x_min-center_x, x_max-center_x, z_min, z_max])
				else:
					axes[energy_setting_id][i].imshow(np.rot90(hists[energy_setting_id][i]), vmin=0, extent=[x_min-center_x, x_max-center_x, z_min, z_max])

			axes[energy_setting_id][i].set_xlim([x_min-center_x, x_max-center_x])
			axes[energy_setting_id][i].set_ylim([z_min, z_max])

	fig.text(0.5, 0.02, '$x$ [mm]', ha='center', va='center')
	fig.text(0.02, 0.5, '$z$ [mm]', ha='center', va='center', rotation='vertical')

	fig.subplots_adjust(
	    top=0.92,
	    bottom=0.05,
	    left=0.05,
	    right=0.95,
	    hspace=0.08,
	    wspace=0.1
	)

	plt.savefig(fig_filename + ".pdf", format="pdf", dpi=800)
	print("Figure saved : ", fig_filename + ".pdf")
	plt.close(fig)


def contour_plot_stop(logfile_path, output_folder, energy, subenergy, only_success=False, scatter=True):
	""" Read the .csv and do a contour plot of the hitmap for one subenergy and plot it. """
	x_min 		= 40
	x_max 		= 100
	y_min 		= 19.75
	y_max 		= 94

	center_x	= int((x_max + x_min)/2)
	bins_x 		= np.arange(x_min-center_x, x_max-center_x + 1, 0.5)
	bins_y 		= np.arange(y_min, y_max + 1, 0.5)

	# fig names
	fig_title, fig_filename, fig_filename_scatter = generate_hitmap_fig_names(output_folder, "STOP", energy, subenergy, only_success)

	ion_ids_xy, x, y = get_logfiles_data(logfile_path, "_stop", ['X Ion Start Splat[mm]', 'Y Ion Start Splat[mm]'], energy, subenergy, only_success=only_success)

	if x is not None and y is not None:
		# use x symetry to duplicated data
		x = np.append(x, 2*center_x - x)
		y = np.append(y, y)

		fig = plt.figure()
		ax = fig.add_subplot(111)
		hist = ax.hist2d(x - center_x, y, bins=[bins_x, bins_y], range=[[x_min-center_x, x_max-center_x], [y_min, y_max]])
		cb = fig.colorbar(hist[3], ax=ax)
		cb.set_label('Counts [-]')
		ax.set_title(fig_title)
		ax.set_xlabel('$x$ [mm]')
		ax.set_ylabel('$y$ [mm]')
		ax.set_xlim([x_min-center_x, x_max-center_x])
		ax.set_ylim([y_min, y_max])  
		plt.savefig(fig_filename + ".png", format="png", dpi=400)
		plt.close(fig)


		if scatter is True:
			xy = np.vstack([x - center_x,  y])
			density = scipy.stats.gaussian_kde(xy)(xy)

			fig = plt.figure()
			ax = fig.add_subplot(111)
			scatter_plot = ax.scatter(x - center_x,  y, c=density, s=2, marker="o")
			cb = fig.colorbar(scatter_plot, label="Density [-]")
			ax.set_title(fig_title)
			ax.set_xlabel('$x$ [mm]')
			ax.set_ylabel('$y$ [mm]')
			ax.set_xlim([x_min-center_x, x_max-center_x])
			ax.set_ylim([y_min, y_max])  
			plt.savefig(fig_filename_scatter + ".png", format="png", dpi=400)
			plt.close(fig)

def scatter_stop(logfile_path, output_folder, energy, subenergy):
	""" Read the .csv and hit file and do a contour plot of the hitmap for one subenergy and plot it. """
	x_min = 40
	x_max = 100
	y_min = 19.75
	y_max = 94

	center_x = int((x_max + x_min)/2)
	bins_x = np.arange(x_min-center_x, x_max-center_x + 1, 0.5)
	bins_y = np.arange(y_min, y_max + 1, 0.5)

	# fig names
	fig_title = f'Ion STOP surface S-N ($E_b$ = {str(math.floor(energy))}eV energy setting)'
	fig_filename = output_folder + 'LNT_WAVE_ENA_PERFORMANCE_HIT_MAP_SN_STOP_ES_' + str(math.floor(energy))
	if subenergy is not None:
		fig_filename += '_E0_' + str(math.floor(subenergy))
		fig_title += f' @ $E_0$ = {math.floor(subenergy)}eV'


	ion_ids_hits, hits_start, hits_stop 	= get_logfiles_data(logfile_path, "_hits", ['Hit Start Surface', 'Hit Stop Surface'], energy, subenergy, only_success=False)
	ion_ids_xy, x, y 						= get_logfiles_data(logfile_path, "_stop", ['X Ion Start Splat[mm]', 'Y Ion Start Splat[mm]'], energy, subenergy, only_success=False)

	# Compute signal and noise 
	mask_signal = np.logical_and(hits_start == 1, hits_stop == 1)
	mask_noise = np.logical_or(hits_start == 0, hits_stop == 0)

	if x is not None and y is not None:
		inds_signal = np.nonzero(ion_ids_hits[mask_signal][:, None] == ion_ids_xy)
		inds_noise = np.nonzero(ion_ids_hits[mask_noise][:, None] == ion_ids_xy)

		N = len(x)
		x = np.append(x, 2*center_x - x)
		y = np.append(y, y)

		fig = plt.figure()
		ax = fig.add_subplot(111)

		# hist = ax.hist2d(x - center_x, y, bins=[bins_x, bins_y], range=[[x_min-center_x, x_max-center_x], [y_min, y_max]])
		if len(inds_noise) > 1:
			inds_noise = np.append(inds_noise[1], inds_noise[1] + N)
			ax.plot(x[inds_noise] - center_x, y[inds_noise], "+r", markersize=1)

		if len(inds_signal) > 1:
			inds_signal = np.append(inds_signal[1], inds_signal[1] + N)
			ax.plot(x[inds_signal] - center_x, y[inds_signal], "+b", markersize=1)

		ax.legend(["Noise", "Signal"])
		# cb = fig.colorbar(hist[3], ax=ax)
		# cb.set_label('Counts [-]')
		ax.set_title(fig_title)
		ax.set_xlabel('$x$ [mm]')
		ax.set_ylabel('$y$ [mm]')
		ax.set_xlim([x_min-center_x, x_max-center_x])
		ax.set_ylim([y_min, y_max])
		plt.savefig(fig_filename + ".png", format="png", dpi=400)
		plt.close(fig)

def contour_plot_cs(logfile_path, output_folder, energy, subenergy):
	""" Read the .csv and do a contour plot of the hitmap for one subenergy and plot it. """
	x_min 		= 38
	x_max 		= 102
	z_min 		= 68.5
	z_max 		= 100

	center_x 	= int((x_max + x_min)/2)
	bins_x 		= np.arange(x_min-center_x, x_max-center_x + 1, 0.5)
	bins_z 		= np.arange(0, z_max - z_min + 1, 0.5)

	fig_title, fig_filename, fig_filename_scatter = generate_hitmap_fig_names(output_folder, "CS", energy, subenergy, only_success)

	ion_ids_xz, x, z = get_logfiles_data(logfile_path, "_cs", ['X Ion Start Splat[mm]', 'Z Ion Start Splat[mm]'], energy, subenergy, only_success=False)

	if x is not None and z is not None:
		# use x symetry to duplicated data
		x = np.append(x, 2*center_x - x)
		z = np.append(z, z)

		fig = plt.figure()
		ax = fig.add_subplot(111)
		hist = ax.hist2d(x - center_x, z - z_min, bins=[bins_x, bins_z], range=[[x_min-center_x, x_max-center_x], [0, z_max - z_min]])
		cb = fig.colorbar(hist[3], ax=ax)
		cb.set_label('Counts [-]')
		ax.set_title(fig_title)
		ax.set_xlabel('$x$ [mm]')
		ax.set_ylabel('$z$ [mm]')
		ax.set_xlim([x_min-center_x, x_max-center_x])
		ax.set_ylim([0, z_max - z_min])
		plt.savefig(fig_filename + ".png", format="png", dpi=400)
		plt.close(fig)

		if len(x) < 50000:
			xz = np.vstack([x - center_x,  z - z_min])
			density = scipy.stats.gaussian_kde(xz)(xz)
			
			fig = plt.figure()
			ax = fig.add_subplot(111)
			scatter_plot = ax.scatter(x - center_x,  z - z_min, c=density, s=2, marker="o")
			cb = fig.colorbar(scatter_plot, label="Density [-]")
			ax.set_title(fig_title)
			ax.set_xlabel('$x$ [mm]')
			ax.set_ylabel('$z$ [mm]')
			ax.set_xlim([x_min-center_x, x_max-center_x])
			ax.set_ylim([0, z_max - z_min])
			plt.savefig(fig_filename_scatter + ".png", format="png", dpi=400)
			plt.close(fig)


def coincidence_plot(tmpfile_path, output_folder, energy_setting, initial_particle_energies, min_energy=None, max_energy=None):
	""" Plot the coincidicence hits against the energy of the neutral beam and save it. """
	coinc_hits = []
	start_hits = []
	stop_hits = []

	for se in range(len(initial_particle_energies)):
		subhit_coin = 0
		subhit_start = 0
		subhit_stop = 0

		tmp_hit_files = get_sorted_log_file_list(f"{tmpfile_path}hits_KE_*_{math.floor(energy_setting)}_{str(math.floor(initial_particle_energies[se]))}.txt")
		for file in tmp_hit_files:
			subhits = simion.io.read_tempfile(file)
			subhit_coin += subhits[9] # Start and stop hits
			subhit_start += subhits[3] + subhits[4] # only start CEM hits
			subhit_stop += subhits[7] + subhits[8] # only stop CEM hits

		# factor 2 because of system symmetry
		coinc_hits.append(2*subhit_coin)
		start_hits.append(2*subhit_start)
		stop_hits.append(2*subhit_stop)

	if min_energy is None: min_energy = min(initial_particle_energies)
	if max_energy is None: max_energy = max(initial_particle_energies)
	
	# Plot results		
	fig = plt.figure()
	fig.suptitle('Coincidence counts @ ' + str(math.floor(energy_setting)) + 'eV Energy setting')
	ax = fig.add_subplot(111)
	ax.semilogx(initial_particle_energies, coinc_hits)
	ax.set_xlabel('Energy [eV]')
	ax.set_ylabel('Counts [-]')  
	ax.set_xlim([min_energy, max_energy])
	plt.savefig(output_folder + 'counts_KE_'+str(math.floor(energy_setting)),dpi=400)
	plt.close(fig)

	
	fig = plt.figure()
	fig.suptitle('Coincidence counts @ ' + str(math.floor(energy_setting)) + 'eV Energy setting')
	ax = fig.add_subplot(111)
	ax.semilogx(initial_particle_energies, coinc_hits, "-r")
	ax.semilogx(initial_particle_energies, start_hits, "--g")
	ax.semilogx(initial_particle_energies, stop_hits, "--b")
	ax.set_xlim([min_energy, max_energy])
	ax.set_xlabel('Energy [eV]')
	ax.set_ylabel('Counts [-]')
	ax.legend(['Coincidence counts', 'Start CEM counts', 'Stop CEM counts'])
	plt.savefig(output_folder + 'counts_KE_start_stop_'+str(math.floor(energy_setting)),dpi=400)
	plt.close(fig)


def scatter_coincidence_hits_start(logfile_path, output_folder, energy, subenergy):
	""" Read the .csv and do a contour plot of the hitmap for one subenergy and plot it."""
	x = z = ion_ids_xz = None
	ion_ids_hits = hits_start = hits_stop = None

	x_min 		= 0
	x_max 		= 60
	z_min 		= 0
	z_max 		= 27
	center_x 	= int((x_max + x_min)/2)

	# fig names
	fig_title = f'Ion START surface coincidence hits \n ($E_b$ = {str(math.floor(energy))}eV energy setting)'
	fig_filename = output_folder + 'LNT_WAVE_ENA_PERFORMANCE_COINCIDENCE_SCATTER_START_ES_' + str(math.floor(energy))
	if subenergy is not None:
		fig_filename += '_E0_' + str(math.floor(subenergy))
		fig_title += f' @ $E_0$ = {math.floor(subenergy)}eV'

	# read data from log files
	ion_ids_xz, x, z 						= get_logfiles_data(logfile_path, "_start", ['X Ion Start Splat[mm]', 'Z Ion Start Splat[mm]'], energy, subenergy, only_success=False)
	ion_ids_hits, hits_start, hits_stop 	= get_logfiles_data(logfile_path, "_hits", ['Hit Start Surface', 'Hit Stop Surface'], energy, subenergy, only_success=False)

	# Compute signal and noise 
	mask_signal = np.logical_and(hits_start == 1, hits_stop == 1)
	mask_noise = np.logical_or(hits_start == 0, hits_stop == 0)

	if x is not None and z is not None:
		inds_signal = np.nonzero(ion_ids_hits[mask_signal][:, None] == ion_ids_xz)
		inds_noise = np.nonzero(ion_ids_hits[mask_noise][:, None] == ion_ids_xz)

		N = len(x)
		x = np.append(x, 2*center_x - x)
		z = np.append(z, z)

		fig = plt.figure()
		ax = fig.add_subplot(111)

		if len(inds_noise) > 1:
			inds_noise = np.append(inds_noise[1], inds_noise[1] + N)
			ax.plot(x[inds_noise] - center_x, z[inds_noise], "+r", markersize=1)

		if len(inds_signal) > 1:
			inds_signal = np.append(inds_signal[1], inds_signal[1] + N)
			ax.plot(x[inds_signal] - center_x, z[inds_signal], "+b", markersize=1)

		ax.legend(["Noise", "Signal"])
		ax.set_title(fig_title)
		ax.set_xlabel('$x$ [mm]')
		ax.set_ylabel('$y$ [mm]')
		ax.set_xlim([x_min-center_x, x_max-center_x])
		ax.set_ylim([z_min, z_max])
		plt.savefig(fig_filename + ".png", format="png", dpi=400)
		plt.close(fig)


	
def tof_spectrum(logfile_path, output_folder, energy_setting, initial_particle_energies):
	""" Plots the measured TOF spectrum for one energy setting. """
	tof_mean  = []
	tof_res = []

	m = 19
	N = 100
	tof_min = 0.05
	tof_max = 0.05
	
	# Defining the step so all plots have the same X-axis properties for the same energy setting
	for se in range(len(initial_particle_energies)):
		suffix = str(math.floor(energy_setting)) + '_' + str(math.floor(initial_particle_energies[se]))+'_tof.csv'
		l=os.listdir(logfile_path)
		sort_nicely(l)
		for file in l:
			if file.endswith(suffix):
				try: 
					header, data = simion.io.read_logfile(logfile_path+file)
					if len(data) > 1:
						tof_m=min(data[:,4])
						tof_M=max(data[:,4])
						if tof_m<tof_min:
							tof_min=tof_m
						if tof_M>tof_max:
							tof_max=tof_M
				except:
					continue
	
	tof_step = np.linspace(tof_min,tof_max,num=N)

	# Plot the TOF spectrum for each of the subenergies
	print('Computing TOF spectrums for each initial_particle_energy')

	for se in range(len(initial_particle_energies)):
		suffix=str(math.floor(energy_setting)) + '_' + str(math.floor(initial_particle_energies[se]))+'_tof.csv'
		l=os.listdir(logfile_path)
		sort_nicely(l)
		tof_heat=[0] * N
		
		for file in l:
			if file.endswith(suffix):
				
				header, data = simion.io.read_logfile(logfile_path+file)
				try:
					for i in range(len(tof_step)):
						for j in range(len(data)):
							if data[j,4] > tof_step[i]-((tof_max-tof_min)/N)/2 and data[j,4] < tof_step[i] +((tof_max-tof_min)/N)/2:
								tof_heat[i]=tof_heat[i]+1
				except:
					#print('There was no coincidence hits at this configuration \n')
					continue

		fig = plt.figure()
		fig.suptitle('TOF spectrum of test_3_KE_' + str(math.floor(energy_setting)) + '_' + str(math.floor(initial_particle_energies[se])))
		ax = fig.add_subplot(111)

		ax.plot(tof_step, tof_heat)

		ax.set_xlabel('TOF / us')
		ax.set_ylabel('Intensity /a.u.')

		plt.savefig(output_folder + 'tof_spectrum_test_3_KE_' + str(math.floor(energy_setting)) + "_" + str(math.floor(initial_particle_energies[se])),dpi=400)
		plt.close(fig)
		
	print('Computing integrated TOF spectrum for the central energy')
	tof_heat_full = [0] * N
	suffix = str(math.floor(energy_setting))+'_tof.csv'

	for file in l:
		if file.endswith(suffix):
			header, data = simion.io.read_logfile(logfile_path+file)
			try:
				for i in range(len(tof_step)):
					for j in range(len(data)):
						if data[j,4] > tof_step[i]-((tof_max-tof_min)/N)/2 and data[j,4] < tof_step[i] + ((tof_max-tof_min)/N)/2:
							tof_heat_full[i] = tof_heat_full[i] + 1
			except:
				#print('There was no coincidence hits at this configuration \n')
				continue

	x = np.asarray(tof_step)
	y = np.asarray(tof_heat_full)
	
	mean=sum(x*y)/sum(y)
	sigma=np.sqrt((sum(y*(x-mean)**2))/sum(y))
	parameters, covariance = curve_fit(gauss,x,y,p0=[max(y),mean,sigma])
	tof_mean.append(parameters[1])
	resolution=2*math.sqrt(2*np.log(2))*parameters[2]
	tof_res.append(resolution)
	
	fig = plt.figure()
	fig.suptitle('LNT TOF spectrum for ' + str(math.floor(energy_setting)) + 'eV')
	ax = fig.add_subplot(111)

	ax.plot(tof_step, tof_heat_full, label='TOF')
	ax.plot(tof_step, gauss(tof_step,*parameters), 'r', label='Gaussian fit')

	ax.set_xlabel('TOF [us]')
	ax.set_ylabel('Intensity [a.u.]')

	ax.legend()

	plt.savefig(output_folder + 'tof_spectrum_KE_' + str(math.floor(energy_setting)),dpi=400)
	print('Integrated TOF spectrum done')
	plt.close(fig)

	# with open(output_folder + 'TOF_resolution.csv', 'w+', newline="") as f:
	# 	writer = csv.writer(f,delimiter=';')
	# 	writer.writerow(['Energy setting [eV]','Central TOF [us]','TOF Resolution (FWHM) [us]'])

	# 	for i in range(len(initial_energies)):
	# 		data=[]
	# 		data.append(energy_setting)
	# 		data.append(tof_mean[i])
	# 		data.append(tof_res[i])
	# 		writer.writerow(data)
	# 		del data


def tof_spectrum_v2(logfile_path, output_folder, energy_setting, initial_particle_energies):
	""" Plots the measured TOF spectrum for one energy setting. """
	tof_mean  = []
	tof_res = []

	m = 19
	N = 100
	tof_min = 0.05
	tof_max = 0.05

	# Defining the step so all plots have the same X-axis properties for the same energy setting
	for se in range(len(initial_particle_energies)):
		suffix = str(math.floor(energy_setting)) + '_' + str(math.floor(initial_particle_energies[se]))+'_tof.csv'
		l=os.listdir(logfile_path)
		sort_nicely(l)
		for file in l:
			if file.endswith(suffix):
				try: 
					header, data = simion.io.read_logfile(logfile_path+file)
					tof_m=min(data[:, 4])
					tof_M=max(data[:, 4])
					if tof_m<tof_min:
						tof_min=tof_m
					if tof_M>tof_max:
						tof_max=tof_M
				except:
					continue
	
	tof_step = np.linspace(tof_min,tof_max,num=N)

	# Plot the TOF spectrum for each of the subenergies
	print('Computing TOF spectrums for each initial_particle_energy\n')

	for se in range(len(initial_particle_energies)):
		suffix=str(math.floor(energy_setting)) + '_' + str(math.floor(initial_particle_energies[se]))+'_tof.csv'
		l=os.listdir(logfile_path)
		sort_nicely(l)
		tof_heat=[0] * N
		
		for file in l:
			if file.endswith(suffix):
				
				header, data = simion.io.read_logfile(logfile_path+file)
				try:
					for i in range(len(tof_step)):
						for j in range(len(data)):
							if data[j,4] > tof_step[i]-((tof_max-tof_min)/N)/2 and data[j,4] < tof_step[i] +((tof_max-tof_min)/N)/2:
								tof_heat[i]=tof_heat[i]+1
				except:
					#print('There was no coincidence hits at this configuration \n')
					continue
 	
		fig = plt.figure()
		fig.suptitle('TOF spectrum of test_3_KE_' + str(math.floor(energy_setting)) + '_' + str(math.floor(initial_particle_energies[se])))
		ax = fig.add_subplot(111)
		ax.plot(tof_step, tof_heat)
		ax.set_xlabel('TOF / us')
		ax.set_ylabel('Intensity /a.u.')
		plt.savefig(output_folder + 'tof_spectrum_test_3_KE_' + str(math.floor(energy_setting)) + "_" + str(math.floor(initial_particle_energies[se])),dpi=400)
		plt.close(fig)
		
	print('Computing integrated TOF spectrum for the central energy\n')
	tof_heat_full = [0] * N
	suffix = str(math.floor(energy_setting))+'_tof.csv'

	for file in l:
		if file.endswith(suffix):
			header, data = simion.io.read_logfile(logfile_path+file)
			try:
				for i in range(len(tof_step)):
					for j in range(len(data)):
						if data[j,4] > tof_step[i]-((tof_max-tof_min)/N)/2 and data[j,4] < tof_step[i] + ((tof_max-tof_min)/N)/2:
							tof_heat_full[i] = tof_heat_full[i] + 1
			except:
				#print('There was no coincidence hits at this configuration \n')
				continue

	x = np.asarray(tof_step)
	y = np.asarray(tof_heat_full)
	
	mean=sum(x*y)/sum(y)
	sigma=np.sqrt((sum(y*(x-mean)**2))/sum(y))
	parameters, covariance = curve_fit(gauss,x,y,p0=[max(y),mean,sigma])
	tof_mean.append(parameters[1])
	resolution=2*math.sqrt(2*np.log(2))*parameters[2]
	tof_res.append(resolution)
	
	fig = plt.figure()
	fig.suptitle('LNT TOF spectrum for ' + str(math.floor(energy_setting)) + 'eV')
	ax = fig.add_subplot(111)
	ax.plot(tof_step, tof_heat_full, label='TOF')
	ax.plot(tof_step, gauss(tof_step,*parameters), 'r', label='Gaussian fit')
	ax.set_xlabel('TOF [us]')
	ax.set_ylabel('Intensity [a.u.]')
	ax.legend()
	plt.savefig(output_folder + 'tof_spectrum_KE_' + str(math.floor(energy_setting)),dpi=400)
	plt.close(fig)



def tof_spectrum_comparison(logfile_path, energy):
	""" Plot the TOF spectrum along with the neutral TOF one. """
	tof_mean=[]
	tof_res=[]
	for i in range(len(energy)):
		m=19
		subenergy=np.linspace(0.25*energy[i],2.5*energy[i],m)
		
		N=100
		tof_min= 0.05
		tof_max=0.05
		#tof_min_N=1
		#tof_max_N=0
		"Defining the step so all plots have the same X-axis properties for the same energy setting"
		for se in range(len(subenergy)):
			suffix=str(math.floor(energy[i])) + "_" + str(math.floor(subenergy[se])) + '_tof.csv'
			l=os.listdir(logfile_path)
			sort_nicely(l)
			for file in l:
				if file.endswith(suffix):
					try: 
						header, data = simion.io.read_logfile(logfile_path+file)
						tof_m=min(data[:,4])
						tof_M=max(data[:,4])
						# tof_m_N=min(data[:,2])
						# tof_M_N=max(data[:,2])
						if tof_m<tof_min:
							tof_min=tof_m
						if tof_M>tof_max:
							tof_max=tof_M
						# if tof_m_N<tof_min_N:
						#     tof_min_N=tof_m_N
						# if tof_M_N>tof_max_N:
						#     tof_max_N=tof_M_N
					except:
						continue
		
		tof_step=np.linspace(tof_min,tof_max,num=N)
		# tof_step_N=np.linspace(tof_min_N,tof_max_N,num=N)
		
		print('Performing integrated measured and neutral TOF spectrum for the central energy\n')
		tof_heat_full=[0] * N
		tof_heat_full_N=[0] * N
		suffix=str(math.floor(energy[i]))+'_tof.csv'
		for file in l:
			if file.endswith(suffix):
				header, data = simion.io.read_logfile(logfile_path+file)
				try:
					
					for k in range(len(tof_step)):
						for j in range(len(data)):
							if data[j,4] > tof_step[k]-((tof_max-tof_min)/N)/2 and data[j,4] < tof_step[k] +((tof_max-tof_min)/N)/2:
								tof_heat_full[k]=tof_heat_full[k]+1
					for k in range(len(tof_step)):
						for j in range(len(data)):
							if data[j,2] > tof_step[k]-((tof_max-tof_min)/N)/2 and data[j,2] < tof_step[k] +((tof_max-tof_min)/N)/2:
								tof_heat_full_N[k]=tof_heat_full_N[k]+1
				except:
					#print('There was no coincidence hits at this configuration \n')
					continue
		
		
		x=np.asarray(tof_step)
		y=np.asarray(tof_heat_full)
		
		mean=sum(x*y)/sum(y)
		sigma=np.sqrt((sum(y*(x-mean)**2))/sum(y))
		parameters, covariance = curve_fit(gauss,x,y,p0=[max(y),mean,sigma])
		tof_mean.append(parameters[1])
		resolution=2*math.sqrt(2*np.log(2))*parameters[2]
		tof_res.append(resolution)
		
		
		plt.figure()
		plt.rcParams["figure.figsize"] = (10,6)
		plt.plot(tof_step,tof_heat_full,label='TOF')
		plt.plot(tof_step,gauss(tof_step,*parameters),'r',label='Gaussian fit')
		plt.title('LNT TOF spectrum for '+str(math.floor(energy[i]))+' eV set')
		plt.xlabel('TOF / us')
		plt.ylabel('Intensity / a.u.')
		plt.legend()
		plt.savefig(output_folder+'tof_spectrum_KE_'+str(math.floor(energy[i])),dpi=400)
		plt.close(fig)
		
		plt.figure()
		plt.rcParams["figure.figsize"] = (10,6)
		plt.plot(tof_step,tof_heat_full,label='Measured')
		plt.plot(tof_step,tof_heat_full_N,label='Neutral')
		plt.title('LNT TOF spectrum for '+str(math.floor(energy[i]))+' eV setting')
		plt.legend()
		plt.xlabel('TOF / us')
		plt.ylabel('Intensity / a.u.') 
		plt.savefig(output_folder+'tof_spectrum_comparison_KE_'+str(math.floor(energy[i])),dpi=400)
		plt.close(fig)
		
		plt.figure()
		plt.rcParams["figure.figsize"] = (10,6)
		plt.plot(tof_heat_full,tof_heat_full_N,'o',color='blue')
		plt.plot(tof_heat_full,tof_heat_full,linestyle=(0,(5,20)),color='black')
		plt.title('LNT TOF scatter plot for '+str(math.floor(energy[i]))+' eV setting')
		plt.xlabel('Measured intensity / a.u')
		plt.ylabel('Neutral intensity / a.u.') 
		plt.savefig(output_folder+'tof_spectrum_dispersion_KE_'+str(math.floor(energy[i])),dpi=400)
		plt.close(fig)
		
	with open(output_folder + 'TOF_resolution.csv', 'w', newline="") as f:
		writer = csv.writer(f,delimiter=';')
		writer.writerow(['Energy setting/eV','Central TOF/us','TOF Resolution (FWHM)/us'])

		for i in range(len(initial_energies)):
			data=[]

			data.append(energy[i])
			data.append(tof_mean[i])
			data.append(tof_res[i])

			writer.writerow(data)
			del data

		f.close()
		
def tof_scatter_plot(logfile_path, energy, output_folder):
	""" Plots the different TOF combinations. """    
	l=os.listdir(logfile_path)
	sort_nicely(l)

	for i in range(len(energy)):
		suffix = str(math.floor(energy[i])) + '_tof.csv'

		tof_meas = []
		tof_neutral = []
		tof_e_1 = []
		tof_e_2 = []
		
		# compute TOF for each particle type
		for file in l:
			if file.endswith(suffix):
				header, data = simion.io.read_logfile(logfile_path + file)

				for j in range(len(data)):
					tof_meas.append(data[j,header['Measured TOF[us]']])
					tof_neutral.append(data[j,header['Neutral TOF[us]']])
					tof_e_1.append(data[j,header['Electron 1 TOF[us]']])
					tof_e_2.append(data[j,header['Electron 2 TOF[us]']])
		
		diag_straight=np.linspace(min(tof_meas),max(tof_meas),50)  

		# Plots Neutral TOF against Measured TOF
		fig = plt.figure()
		fig.sup_title('LNT TOF scatter plot for the ' + str(math.floor(energy[i])) + 'eV energy setting')
		ax = fig.add_subplot(111)
		plt.plot(tof_meas, tof_neutral, 'o', color='blue', markersize=2)
		plt.plot(diag_straight, diag_straight, linestyle=(0,(5,20)), color='black')
		plt.xlabel('Measured TOF / us')
		plt.ylabel('Neutral TOF / us') 
		ax.set_aspect('equal')
		plt.xlim(min(tof_meas), max(tof_meas))
		plt.ylim(min(tof_meas), max(tof_meas))
		plt.savefig(output_folder + 'tof_spectrum_dispersion_KE_' 		+ str(math.floor(energy[i])), dpi=400)
		
		plt.close(fig)

		# Plots Neutral TOF against Electron 1 TOF
		fig = plt.figure()
		fig.sup_title('LNT TOF scatter plot for the ' + str(math.floor(energy[i])) + 'eV energy setting')
		ax = fig.add_subplot(111)
		ax.plot(tof_e_1, tof_neutral, 'o', color='blue', markersize=2)
		ax.plot(diag_straight, diag_straight, linestyle=(0,(5,20)), color='black')
		ax.xlabel('Electron 1 TOF / us')
		ax.ylabel('Neutral TOF / us') 
		ax.set_aspect('equal')
		ax.xlim(0, max(tof_neutral))
		ax.ylim(0, max(tof_neutral))
		plt.savefig(output_folder + 'tof_spectrum_dispersion_e1_KE_' 	+ str(math.floor(energy[i])), dpi=400)
		plt.close(fig)



		# Plots Neutral TOF against Electron 2 TOF
		fig = plt.figure()
		fig.sup_title('LNT TOF scatter plot for the ' + str(math.floor(energy[i])) + 'eV energy setting')
		ax = fig.add_subplot(111)
		ax.plot(tof_e_2, tof_neutral, 'o', color='blue', markersize=2)
		ax.plot(diag_straight, diag_straight, linestyle=(0,(5,20)), color='black')
		ax.xlabel('Electron 2 TOF / us')
		ax.ylabel('Neutral TOF / us') 
		ax.set_aspect('equal')
		ax.xlim(0, max(tof_neutral))
		ax.ylim(0, max(tof_neutral))
		plt.savefig(output_folder + 'tof_spectrum_dispersion_e2_KE_' 	+ str(math.floor(energy[i])), dpi=400)
		plt.close(fig)
		

		# Plots Neutral TOF against Electron 1 TOF and Electron 2 TOF
		fig = plt.figure()
		fig.sup_title('LNT TOF scatter plot for the ' + str(math.floor(energy[i])) + 'eV energy setting')
		ax = fig.add_subplot(111)
		ax.plot(tof_e_2, tof_neutral, 'o', color='blue', markersize=2, label='Electron 2')
		ax.plot(tof_e_1, tof_neutral, 'o', color='red', markersize=2, label='Electron 1')
		ax.plot(diag_straight, diag_straight, linestyle=(0,(5,20)), color='black')
		ax.xlabel('Electron TOF / us')
		ax.ylabel('Neutral TOF / us') 
		ax.set_aspect('equal')
		ax.legend()
		ax.xlim(0, max(tof_neutral))
		ax.ylim(0, max(tof_neutral))
		plt.savefig(output_folder + 'tof_spectrum_dispersion_e1_e2_KE_' + str(math.floor(energy[i])), dpi=400)
		plt.close(fig)
	

def gauss(x,a,mu,sigma):
	""" Returns the Gaussian Curve. """
	return a*np.exp(-(x-mu)**2/(2.*sigma**2))
	  

def energy_resolution(logfile_path, energy, output_folder):
	""" Computes the energy response curve along with a Gaussian fit and get the resolution. """
	e_res = []
	e_central = []

	m = 19
	M = 200

	for i in range(len(energy)):
		subenergy = np.linspace(0.25*energy[i],2.5*energy[i],m)
		subenergy_gauss = np.linspace(0.25*energy[i],2.5*energy[i],M)

		# Take the coincidence hits curve and fit it into a gaussian curve, giving the mean, variance and FWHM
		coinc_hits=[]
		
		l=os.listdir(logfile_path)
		sort_nicely(l)
		for se in range(len(subenergy)):
			subhit_coin=0
			for file in l:
				suffix='KE_' + str(math.floor(energy[i])) + "_" + str(math.floor(subenergy[se])) + '.txt'
				if file.endswith(suffix):
					subhits=read_tempfile(logfile_path+file)
					subhit_coin=subhit_coin+subhits[9]
	
			coinc_hits.append(subhit_coin)
		
		# Parameters for the gaussian fit
		x = np.asarray(subenergy)
		y = np.asarray(coinc_hits)
		
		mean = sum(x*y)/sum(y)
		sigma = np.sqrt((sum(y*(x-mean)**2))/sum(y))
		parameters, covariance = curve_fit(gauss,x,y,p0=[max(y),mean,sigma])
		
		# Plot results
		fig = plt.figure()
		ax = fig.add_subplot(111)

		ax.plot(subenergy, coinc_hits, label='Simulation results')
		ax.plot(subenergy_gauss, gauss(subenergy_gauss ,*parameters), 'r', label='Gaussian fit')

		ax.legend()

		ax.title('LNT energy response for energy setting at '+str(math.floor(energy[i]))+' eV')
		ax.xlabel('ENA Energy [eV]')
		ax.ylabel('Coincidence Counts [-]') 

		plt.savefig(output_folder + 'energy_resolution_' + str(math.floor(energy[i])), dpi=400)
		plt.close(fig)

		e_central.append(parameters[1])
		resolution = 2*math.sqrt(2*np.log(2))*parameters[2]
		e_res.append(resolution)

		print('FWHM at '+str(energy[i])+': '+str(abs(2*math.sqrt(2*np.log(2))*parameters[2])))
	
	# Save results to output file
	with open(output_folder + 'energy_resolution.csv', 'w', newline="") as f:
		writer = csv.writer(f, delimiter=';')
		writer.writerow(['Energy setting/eV','Central Energy/eV','Energy Resolution (FWHM)/eV'])

		for i in range(len(initial_energies)):
			data=[]
			data.append(energy[i])
			data.append(e_central[i])
			data.append(e_res[i])

			writer.writerow(data)
			del data

		f.close()

def mass_resolution(logfile_path, output_folder):
	""" 
		Obtains the energy of the neutral using the ENA mean energy and TOF for each energy setting
		Obtains the mean distance by taking the neutral energy, mass and TOF
	"""

	energy_path 	= 'energy_resolution/'
	tof_path		= 'tof_spectra/'
	mass_path		= 'mass_resolution/'

	header_e, energy = simion.io.read_logfile(output_folder + energy_path + 'energy_resolution.csv')
	header_tof, tof = simion.io.read_logfile(output_folder + tof_path + 'TOF_resolution.csv')

	loss=0.85

	n_energy = []
	distance = []
	distance_heat_l = []
	distance_heat_r = []

	imass = 1.007276
	emass = 0.00054857990946 	# electron mass in amu
	nmass=imass+emass
	amu = 1.660538921e-27 # kg
	ec = 1.60217657e-19 	# elementary charge in Coulomb
	rot= math.radians(-25)
	
	for i in range(len(energy)):
		n_energy.append(loss*(2400+loss*energy[i,1]))
		distance.append(np.sqrt(2*n_energy[i]*ec/(nmass*amu))*tof[i,1]*1e-3) 
		
		start_max = contour_plot_start_success(logfile_path,energy[i,0])
		stop_max = contour_plot_stop_sucess(logfile_path, energy[i,0])
		
		start_max_rot = list([tuple([start_max[0][0],math.cos(-rot)*start_max[0][1]-math.sin(-rot)*start_max[0][2],math.sin(-rot)*start_max[0][1]+math.cos(-rot)*start_max[0][2]]),tuple([start_max[1][0],math.cos(-rot)*start_max[1][1]-math.sin(-rot)*start_max[1][2],math.sin(-rot)*start_max[1][1]+math.cos(-rot)*start_max[1][2]])])
		distance_heat_l.append(math.sqrt((stop_max[0][0]-start_max_rot[0][0])**2+(stop_max[0][1]-start_max_rot[0][1])**2+(stop_max[0][2]-start_max_rot[0][2])**2))        
		distance_heat_r.append(math.sqrt((stop_max[1][0]-start_max_rot[1][0])**2+(stop_max[1][1]-start_max_rot[1][1])**2+(stop_max[1][2]-start_max_rot[1][2])**2))        


	with open(output_folder + mass_path + 'mass_resolution.csv', 'w+', newline="") as f:
		writer = csv.writer(f,delimiter=';')
		writer.writerow(['Energy setting/eV','Mean distance/mm','Mean distance left side/mm','Mean distance right side/mm'])

		for i in range(len(initial_energies)):
			data=[]
			data.append(energy[i,0])
			data.append(distance[i])
			data.append(distance_heat_l[i])
			data.append(distance_heat_r[i])
			writer.writerow(data)
			del data

		f.close()
			

def compute_min_max_hist_bounds(hist, bin_edges, level, comparison_type='strict'):
	bin_center = (bin_edges[:-1] + bin_edges[1:])/2

	# get all values above the specified level
	if comparison_type == 'strict':
		mask_hist_greater_that_level = hist > level
	elif comparison_type == 'nonstrict':
		mask_hist_greater_that_level = hist >= level

	# No values found, returns None
	if len(np.where(mask_hist_greater_that_level)[0]) == 0:
		 return None

	# compute indices
	inds = np.where(mask_hist_greater_that_level)[0]
	ind_max = inds[-1]
	ind_min = inds[0]

	uncertainties = np.diff(bin_center)

	# compute mean values and uncertainty
	if ind_max < len(bin_center)-1:
		max_bound = 0.5*(bin_center[ind_max+1] + bin_center[ind_max])
		uncertainty_max = uncertainties[ind_max]
	else:
		max_bound = bin_center[-1]
		uncertainty_max = uncertainties[-1]
	if ind_min > 0:
		min_bound = 0.5*(bin_center[ind_min] + bin_center[ind_min-1])
		uncertainty_min = uncertainties[ind_min-1]
	else:
		min_bound = bin_center[0]
		uncertainty_min = uncertainties[0]

	return min_bound, max_bound, uncertainty_min, uncertainty_max

			
def angular_resolution(logfile_path, output_folder, energy_settings, initial_particle_energies, only_success=False):
	""" Plots the distribution of az and el on the start surface. """

	# for initial_energy in initial_particle_energies:
	# 	ion_ids_xz, x, z, az_init, el_init = get_logfiles_data(logfile_path, "_start", ['X Ion Start Splat[mm]', 'Z Ion Start Splat[mm]', 'Azm Ion Start[deg]', 'Elv Ion Start[deg]'], energy_setting, initial_energy, only_success=only_success)
	# 	if only_success is True:
	# 		x, z, az_init, el_init = keep_only_successful_hits(logfile_path, ion_ids_xz, [x, z, az_init, el_init], energy_setting, initial_energy)
	# 	print(x)
	# 	print(z)
	# 	print(az_init)
	# 	print(el_init)
	# 	if x is not None and z is not None and az_init is not None and el_init is not None:
	# 		# use x symetry to duplicated data
	# 		az_init 	= np.append(az_init, az_init)
	# 		el_init 	= np.append(el_init, el_init)

	# 		fig = plt.figure()
	# 		ax11 = fig.add_subplot(121)
	# 		ax12 = fig.add_subplot(122)

	# 		fig.suptitle("LNT FOV")
	# 		ax11.set_xlabel("Azimuth [deg]")
	# 		ax11.set_ylabel("Counts [-]")
	# 		ax12.set_xlabel("Elevation [deg]")
	# 		ax12.set_ylabel("Counts [-]")

	# 		ax11.hist(az_init + 90, color="blue", bins=35, histtype='step')
	# 		ax12.hist(-el_init, color="blue", bins=35, histtype='step')

	# 		plt.close(fig)			

	
	fig = plt.figure(dpi=100, figsize=(13, 6))
	ax11 = fig.add_subplot(121)
	ax12 = fig.add_subplot(122)
	fig.suptitle("LNT FOV")

	ax11.set_xlabel("Azimuth [deg]")
	ax11.set_ylabel("Counts [-]")
	ax12.set_xlabel("Elevation [deg]")

	fmt = [
		["-", "red"],
		["-", "blue"],
		["-", "green"],
		["-", "cyan"],
		["-", "orange"],
		["-", "black"],
		["-", "magenta"],
		["--", "red"],
		["--", "blue"],
		["--", "green"],
		["--", "cyan"],
		["--", "orange"],
		["--", "black"],
		["--", "magenta"],
	]

	fov_az = []
	fov_az_uncertainty = []
	fov_el = []
	fov_el_uncertainty = []
	resolution_az = []
	resolution_az_uncertainty = []
	resolution_el = []
	resolution_el_uncertainty = []

	energy_settings_plot = []

	az_bins = np.arange(-45, 46, 4)
	el_bins = np.arange(0, 21, 1)
	
	for energy_setting_id, energy_setting in enumerate(energy_settings):
		print(f"Processing energy setting {energy_setting_id+1}/{len(energy_settings)}")
		az_init = None
		el_init = None
	
		for initial_energy in initial_particle_energies[energy_setting_id]:
			ion_ids_xz, x, z, az_init_sub, el_init_sub = get_logfiles_data(logfile_path, "_start", ['X Ion Start Splat[mm]', 'Z Ion Start Splat[mm]', 'Azm Ion Start[deg]', 'Elv Ion Start[deg]'], energy_setting, initial_energy, only_success=only_success)

			if az_init_sub is not None and el_init_sub is not None:
				if az_init is None:
					az_init = az_init_sub
				else:
					az_init = np.append(az_init, az_init_sub)

				if el_init is None:
					el_init = el_init_sub
				else:
					el_init = np.append(el_init, el_init_sub)

		if az_init is not None and el_init is not None:
			az_hist, az_bin_edges = np.histogram(np.array(az_init) + 90, bins=az_bins)
			el_hist, el_bin_edges = np.histogram(-np.array(el_init), bins=el_bins)

			ax11.stairs(az_hist, az_bin_edges, linestyle=fmt[energy_setting_id][0], color=fmt[energy_setting_id][1], label=f"$E_s = {energy_setting}$eV")
			ax12.stairs(el_hist, el_bin_edges, linestyle=fmt[energy_setting_id][0], color=fmt[energy_setting_id][1], label=f"$E_s = {energy_setting}$eV")

			# Compute FOV and angular resolution 
			az_min, az_max, daz_min, daz_max = compute_min_max_hist_bounds(az_hist, az_bin_edges, 0, comparison_type='strict')
			el_min, el_max, del_min, del_max = compute_min_max_hist_bounds(el_hist, el_bin_edges, 0, comparison_type='strict')
			fov_az += [az_max - az_min]
			fov_az_uncertainty += [np.sqrt(daz_min**2 + daz_max**2)]
			fov_el += [el_max - el_min]
			fov_el_uncertainty += [np.sqrt(del_min**2 + del_max**2)]

			az_fhwm_min, az_fhwm_max, daz_fhwm_min, daz_fhwm_max = compute_min_max_hist_bounds(az_hist, az_bin_edges, np.max(az_hist)/2, comparison_type='nonstrict')
			el_fhwm_min, el_fhwm_max, del_fhwm_min, del_fhwm_max = compute_min_max_hist_bounds(el_hist, el_bin_edges, np.max(el_hist)/2, comparison_type='nonstrict')
			resolution_az += [az_fhwm_max - az_fhwm_min]
			resolution_az_uncertainty += [np.sqrt(daz_fhwm_min**2 + daz_fhwm_max**2)]
			resolution_el += [el_fhwm_max - el_fhwm_min]
			resolution_el_uncertainty += [np.sqrt(del_fhwm_min**2 + del_fhwm_max**2)]

			energy_settings_plot += [energy_setting]

	ax11.legend()
	ax12.legend()

	ax11.set_ylim([-500, 8500])
	ax12.set_ylim([-500, 8500])

	fig.subplots_adjust(
		top=0.92,
		bottom=0.07,
		left=0.05,
		right=0.95,
		hspace=0.08,
		wspace=0.08
	)
	

	ax11.tick_params(labelright=False)
	ax12.tick_params(labelleft=True)
	
	plt.savefig(output_folder + 'LNT_WAVE_ENA_PERFORMANCE_FOV_ALL', dpi=400)
	# plt.close(fig)

	fig = plt.figure(dpi=100, figsize=(6, 4))
	fig.suptitle("LNT - FOV and Angular resolution")
	
	ax21 = fig.add_subplot(211)
	ax22 = fig.add_subplot(212)


	fig.subplots_adjust(
		top=0.860,
		bottom=0.120,
		left=0.090,
		right=0.935,
		hspace=0.22,
		wspace=0.070
	)

	ax21.set_ylabel("Azimuth [deg]")
	ax22.set_xlabel("Energy setting [eV]")
	ax22.set_ylabel("Elevation [deg]")

	ax21.semilogx(energy_settings_plot, fov_az, "-r", label="FOV")
	ax21.semilogx(energy_settings_plot, resolution_az, "-b", label="Ang. resolution")
	ax21.errorbar(energy_settings_plot, fov_az, fov_az_uncertainty, marker='s', color="red")
	ax21.errorbar(energy_settings_plot, resolution_az, fov_az_uncertainty, marker='s', color="blue")

	ax22.semilogx(energy_settings_plot, fov_el, "-r", label="FOV")
	ax22.semilogx(energy_settings_plot, resolution_el, "-b", label="Ang. resolution")
	ax22.errorbar(energy_settings_plot, fov_el, fov_el_uncertainty, marker='s', color="red")
	ax22.errorbar(energy_settings_plot, resolution_el, fov_el_uncertainty, marker='s', color="blue")

	ax21.tick_params(labelbottom=False)
	ax22.tick_params(labeltop=True)

	ax21.legend()
	ax22.legend()

	plt.savefig(output_folder + 'LNT_WAVE_ENA_PERFORMANCE_FOV_AND_ANG_RES_ALL', dpi=400)
	# plt.close(fig)

	plt.show()

	return np.array(fov_az), np.array(fov_el), np.array(resolution_az), np.array(resolution_el)

def secondary_peaks(logfile_path, energy, ion_path):
	""" Analyses the secondary peaks found in low energy settings. """
	M=19
	l=os.listdir(logfile_path)
	sort_nicely(l)
	success=dict()
   
	
	for i in range(len(energy)):
		subenergy = np.linspace(0.25*energy[i], 2.5*energy[i], M)

		# Take the coincidence hits curve and fit it into a gaussian curve, giving the mean, variance and FWHM
		coinc_hits=[]
		
		for se in range(len(subenergy)):
			subhit_coin=0
			file='hits_KE_' + str(math.floor(energy[i])) + "_" + str(math.floor(subenergy[se]))+'.txt'
			subhits=read_tempfile(logfile_path+file)
			subhit_coin=subhit_coin+subhits[9]
	
			coinc_hits.append(subhit_coin)
			
		max_ = argrelextrema(np.asarray(coinc_hits),np.greater)
		min_ = argrelextrema(np.asarray(coinc_hits),np.less)
		
		# Creates a dictionary with the indexes of all of the succesful particles for each energy setting
		for j in range(len(min_[0])):
			start_suffix='KE_' + str(math.floor(energy[i])) + "_" + str(math.floor(subenergy[min_[0][j]])) + '_start.csv'
			tof_suffix='KE_' + str(math.floor(energy[i])) + "_" + str(math.floor(subenergy[min_[0][j]])) + '_tof.csv'
			for file in l:
				if file.endswith(tof_suffix):
					header_tof, data_tof =simion.io.read_logfile(logfile_path+file)
					succ_n=[]
					for o in range(len(data_tof)):
						succ_n.append(data_tof[o,0])
					start_file=file.replace(tof_suffix,start_suffix)
					success[start_file]=succ_n
			
			for file in l:
				if file.endswith(start_suffix):
					header, data = simion.io.read_logfile(logfile_path+file)
					data_=np.zeros((len(success[file]),23))
					m=0
					K=0
					for n in success[file]:
						for k in list(range(K,len(data))):
							if data[k,0] == n:
								data_[m,:] = data[k,:]
								m = m + 1
								K = k + 1
								break

			ion_name = 'valley_KE_' + str(math.floor(energy[i])) + "_" + str(math.floor(subenergy[min_[0][j]]))
			data2ion(header, data_, ion_path, ion_name, subenergy[min_[0][j]])