import subprocess

import sys, os
import numpy as np
from random import uniform
import csv
import os.path
from tqdm import tqdm

import simion

def runfly(project_path, particle_src, iob_name, n_particles, lua_path="test", win_os=False, show_simion_ui=False, fly=True):
    if win_os is True: # Windows Code
        raise ValueError("Windows is not supported yet.")
        # project_path = "/mnt/c/" + project_path
        # simion_exe_path = "/mnt/c/" + simion_exe_path

        # start_command = simion_exe_path  + f' --default-num-particles={n_particles+5} --noprompt'

        # run_command = start_command + ' fly' + ' --retain-trajectories=0' + ' --restore-potentials=0' + ' --particles=' + particle_src + ' ' + iob_name + " 2>&1 | sed -e '/status2,/d'"
        # run_command = start_command + ' fly' + ' --retain-trajectories=0' + ' --restore-potentials=0' + ' --particles=' + project_path + particle_src + ' ' + project_path + iob_name #Windows
        
        # run_simion = os.system('start cmd /c "G: & cd "\\My Drive\\IRF Internship\\SIMION\SIMION-8.1" & ' + run_command) #windows
    else: # Ubuntu Code            
        start_command = f"'{simion.__simion_path__}' --noprompt --default-num-particles=9999999 --num-threads=8 "

        if show_simion_ui is False:
            start_command += "--nogui "

        run_command = start_command
        if fly is True:
            run_command += 'fly' + \
            ' --retain-trajectories=0' + \
            ' --restore-potentials=0' + \
            ' --particles=' + particle_src + ' ' + \
            iob_name + \
            " 2>&1 | sed -e '/status2,/d'" # Command from Martin to hide annoying console prints...
        else:
            run_command += iob_name

    print("Starting SIMION: ", run_command)
    run_simion = subprocess.Popen(run_command, cwd=project_path, shell=True)

    return run_simion


def initial_conditions(e_min, e_max, x_cs, y_cs, z_cs, x_e, y_e, z_e, z_start, mass, charge, n_particles, filename, tol=1, delim=';', ion_cache_file="init_ions.npy"):
    """ Creates the random initial conditions for n particles. """
    with open(filename, 'w+') as f:
        writer = csv.writer(f, delimiter=delim)

        header = ["Ion N", "Mass[amu]", "Charge[e]", "X Ion Start[mm]", "Y Ion Start[mm]", "Z Ion Start[mm]", "Azm Ion Start[deg]", "Elv Ion Start[deg]", "KE Ion Start[eV]"]
        writer.writerow(header)

        if not os.path.exists(ion_cache_file):
            raise Exception(f"The specified ion cache file ({ion_cache_file}) does not exist. Please run 'generate_ions.py'.")

        data = np.load(ion_cache_file)
        data = data[:n_particles]

        pbar = tqdm(total=n_particles)
        pbar.set_description("Importing particles")

        for i in range(n_particles):
            writer.writerow(data[i])
            pbar.update(1)

        pbar.close()