from subprocess import run
import sys, os
import numpy as np
from random import uniform
import csv
import os.path
import tqdm
import pickle

def write_pickle(filename, data):
    with open(filename + '.pickle', 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

def read_pickle(filename, ):
    with open(filename + '.pickle', 'rb') as file:
        data = pickle.load(file)
    return data

def write_lua(file_in, file_out, energy_settings, initial_energies, step_id, w1, w2a, w2b, lens, project_name, log_file, output_folder, hitfile_path):
    """    
    This function overwrite specific lines in the .lua script linked to a SIMION
    project. These lines are adjustables parameters (that can be modified in SIMION)
    or local variables. 

    Parameters
    ----------
    file : str
        Relative path to the input .lua script.
    file_out : str
        Relative path to the output .lua script.
    adjustable_params : packed list of variables
        Variables to be overwritten in the .lua script.

    Returns
    -------
    None.

    """
    with open(file_in, 'r') as f:
        content = f.readlines()
        f.close()

    with open(file_out, 'w+') as f:
        pbar = tqdm.tqdm(total=len(content))
        pbar.set_description("Writing simion lua workbench file")

        for i, line in enumerate(content):
            if 'case[1] =' in line:
                f.writelines(f'    case[1] ="{output_folder}/{log_file}/{project_name}_KE_"..string.format(math.floor(sim_id+1)).."_"..string.format(math.floor(energy_settings[energy_setting_id])).."_"..string.format(math.floor(initial_energies[energy_setting_id][initial_energy_id])).."_init"\n')
            
            elif 'case[2] =' in line:
                f.writelines(f'    case[2] ="{output_folder}/{log_file}/{project_name}_KE_"..string.format(math.floor(sim_id+1)).."_"..string.format(math.floor(energy_settings[energy_setting_id])).."_"..string.format(math.floor(initial_energies[energy_setting_id][initial_energy_id])).."_start"\n')
            
            elif 'case[3] =' in line:
                f.writelines(f'    case[3] ="{output_folder}/{log_file}/{project_name}_KE_"..string.format(math.floor(sim_id+1)).."_"..string.format(math.floor(energy_settings[energy_setting_id])).."_"..string.format(math.floor(initial_energies[energy_setting_id][initial_energy_id])).."_stop"\n')
            
            elif 'case[4] =' in line:
                f.writelines(f'    case[4] ="{output_folder}/{log_file}/{project_name}_KE_"..string.format(math.floor(sim_id+1)).."_"..string.format(math.floor(energy_settings[energy_setting_id])).."_"..string.format(math.floor(initial_energies[energy_setting_id][initial_energy_id])).."_tof"\n')
            
            elif 'case[5] =' in line:
                f.writelines(f'    case[5] ="{output_folder}/{log_file}/{project_name}_KE_"..string.format(math.floor(sim_id+1)).."_"..string.format(math.floor(energy_settings[energy_setting_id])).."_"..string.format(math.floor(initial_energies[energy_setting_id][initial_energy_id])).."_cs"\n')
            
            elif 'case[6] =' in line:
                f.writelines(f'    case[6] ="{output_folder}/{log_file}/{project_name}_KE_"..string.format(math.floor(sim_id+1)).."_"..string.format(math.floor(energy_settings[energy_setting_id])).."_"..string.format(math.floor(initial_energies[energy_setting_id][initial_energy_id])).."_hits"\n')
            
            elif 'tempfilename =' in line:
                f.writelines(f'    tempfilename = "{output_folder}/{hitfile_path}/hits_KE_"..string.format(math.floor(sim_id+1)).."_"..string.format(math.floor(energy_settings[energy_setting_id])).."_"..string.format(math.floor(initial_energies[energy_setting_id][initial_energy_id]))..".txt"\n')
            
            elif 'successful_ions_file =' in line:
                f.writelines(f'    successful_ions_file = "{output_folder}/{log_file}/{project_name}_successful_ions_KE_"..string.format(math.floor(sim_id+1)).."_"..string.format(math.floor(energy_settings[energy_setting_id])).."_"..string.format(math.floor(initial_energies[energy_setting_id][initial_energy_id]))..".ion"\n')
            
            elif 'PL[4] =' in line:
                f.writelines(f'    PL[4] = {{}}\n')
                for energy_setting_id in range(len(energy_settings)):
                    f.writelines(f'    PL[4][{energy_setting_id}] = {w1[energy_setting_id]}\n')

            elif 'PL[5] =' in line:
                f.writelines(f'    PL[5] = {{}}\n')
                for energy_setting_id in range(len(energy_settings)):
                    f.writelines(f'    PL[5][{energy_setting_id}] = {w2b[energy_setting_id]}\n')

            elif 'PL[6] =' in line:
                f.writelines(f'    PL[6] = {{}}\n')
                for energy_setting_id in range(len(energy_settings)):
                    f.writelines(f'    PL[6][{energy_setting_id}] = {w2a[energy_setting_id]}\n')

            elif 'PL[7] =' in line:
                f.writelines(f'    PL[7] = {{}}\n') 
                for energy_setting_id in range(len(energy_settings)):
                    f.writelines(f'    PL[7][{energy_setting_id}] = {lens[energy_setting_id]}\n')   
            
            elif "local initial_energies =" in line:
                f.writelines('local initial_energies = {}\n')
                for energy_setting_id in range(len(energy_settings)):
                    f.writelines(f'initial_energies[{energy_setting_id}] = {{}}\n')
                    for initial_energy_id in range(len(initial_energies[energy_setting_id])):
                        f.writelines(f'initial_energies[{energy_setting_id}][{initial_energy_id}] = {initial_energies[energy_setting_id][initial_energy_id]}\n')
            
            elif 'local energy_settings =' in line:
                f.writelines(f'local energy_settings = {{}}\n')
                for energy_setting_id in range(len(energy_settings)):
                    f.writelines(f'    energy_settings[{energy_setting_id}] = {energy_settings[energy_setting_id]}\n')

            elif "local sim_id = " in line:
                f.writelines(f"local sim_id = {step_id}\n")

            else:
                f.writelines(line)

            pbar.update(1)

        pbar.close()
        f.close()

def data2ion(header, data_, path, filename, energy):
    """ Creates an ion file based on a data array. """
    mass = 1.0
    charge = 0.0

    pbar = tqdm.tqdm(total=len(data_))
    pbar.set_description("Generating ion file from data")

    with open(path + filename + '.ion', 'w') as f:
        writer = csv.writer(f)

        # Definition of vx  vy vz ranges based on el and az relative to z axis (0,-90)
        for i in range(len(data_)):
            data = []

            data.append(0)
            data.append(mass)
            data.append(charge)
            data.append(data_[i, header['X Ion Init[mm]']])
            data.append(data_[i, header['Y Ion Init[mm]']])
            data.append(data_[i, header['Z Ion Init[mm]']])
            data.append(data_[i, header['Azm Ion Start[deg]']])
            data.append(data_[i, header['Elv Ion Start[deg]']])
            data.append(energy)
            data.append(1.0)
            data.append(1)

            writer.writerow(data)
            del data

            pbar.update(1)

        pbar.close()
        f.close()

def read_tempfile(tmp_file, delim=','):
    """ Reads a .txt tempoary file. """
    with open(tmp_file, 'r') as tmp:
        content = tmp.readline()
        data = []

        for i in content.split(delim):
            data.append(float(i))   

        tmp.close()

    return data

def write_tempfile(data, tempfile, delim=','):
    """ Writes data on a .txt file. """
    try:
        with open(tempfile,'w') as tmp:
            text = ''
            for i in range(len(data)):
                if i != len(data)-1:
                    text = text + str(int(data[i])) + delim
                else:
                    text = text + str(int(data[i]))
            tmp.write(text)
    except:
        print(f'{tempfile}: No .txt to read')

def read_datfile(datfile, delim=','):
    """ Reads .dat file provided by Manabu. """
    with open(datfile, 'r') as df:
        content = df.readlines()[3:]
        data = []

        for line in content:
            data.append(np.array([float(i) for i in line.split(delim)[:-1]]))

        data = np.asarray(data)
        df.close()

    return data

def read_logfile(logfile, delim=';', cache_as_npy=True):   
    """
    This function reads the logfile generated by SIMION and writes a numpy matrix
    out of it.

    Parameters
    ----------
    logfile : string
        relative path and name of the log file.

    Returns
    -------
    header : dictionary
        Dictionary giving the list index for one variable name..
    data : np.array()
        Data matrix.
    delim : str
        Expect csv file, so the delimiter is ';' by default 
    """
    # expect csv, so delimiter is ;
    filename = logfile.strip(".csv")

    if not os.path.exists(filename + "_data.npy"):
        with open(logfile, 'r') as f:
            # Seperating the columns headers
            head = (f.readline().rstrip()).replace(delim + ' ', delim).split(delim)
            header = dict([(head[i], i) for i in range(len(head))])
            
            # read data from file
            data = []

            for line in f.readlines():
                data.append(np.array([float(i) for i in line.split(delim)[:]]))

            data = np.asarray(data)
            f.close()

        if cache_as_npy is True:
            write_pickle(filename + "_header", header)
            np.save(filename + "_data.npy", data, allow_pickle=True)
    else:
        header = read_pickle(filename + "_header")
        data = np.load(filename + "_data.npy", allow_pickle=True)
    
    return header, data
 
def read_initfile(logfile, delim=';'):   
    """
    This function reads the logfile generated by SIMION and writes a numpy matrix
    out of it.

    Parameters
    ----------
    logfile : string
        relative path and name of the log file.

    Returns
    -------
    header : dictionary
        Dictionary giving the list index for one variable name..
    data : np.array()
        Data matrix.
    delim : str
        Expect csv file, so the delimiter is ';' by default 
    """
    with open(logfile, 'r') as f:
        # Seperating the columns headers
        head = (f.readline().rstrip()).replace(delim + ' ', delim).split(delim)
        header = dict([(head[i], i) for i in range(len(head))])
        
        # read data from file
        data = []
        for line in f.readlines():
            data.append(np.array([float(i) for i in line.split(delim)[:-1]]))

        data = np.asarray(data)
        f.close()

    return header, data
 
def log2ion(path, logfile, ionfile, e, initial_particle, final_particle, delim=','):
    """ Creates a ion file based on a .csv logfile. """
    header, arr = read_initfile(path + logfile)

    #Use the initial positions of the succesful ENAs and change it to the new workbench coordinates
    with open(path + ionfile + '.ion', 'w') as f:
        writer = csv.writer(f, delimiter=delim)

        pbar = tqdm.tqdm(total=(final_particle-initial_particle))
        pbar.set_description("Generating ion file")
        for i in range(len(arr))[initial_particle-1:final_particle]:            
            data=[]

            data.append(0)
            data.append(arr[i,header['Mass[amu]']])
            data.append(arr[i,header['Charge[e]']])
            data.append(arr[i,header['X Ion Start[mm]']])
            data.append(arr[i,header['Y Ion Start[mm]']])
            data.append(arr[i,header['Z Ion Start[mm]']])
            data.append(arr[i,header['Azm Ion Start[deg]']])
            data.append(arr[i,header['Elv Ion Start[deg]']])
            data.append(e)
            data.append(1.0)
            data.append(1)

            writer.writerow(data)
            del data

            pbar.update(1)

        pbar.close()
        f.close()

def extract_n_first_ions(ion_in_filename, ion_out_filename, n_particles=10000):
    ion_in_file = open(ion_in_filename, "r")
    if ion_in_file is None:
        raise Exception(f"{ion_in_file=} could not be found.")

    ion_out_file = open(ion_out_filename, "w+")
    for i, line in enumerate(ion_in_file.readlines()):
        if i >= n_particles:
            break # all ions have been copied
        else:
            ion_out_file.write(line)

    ion_out_file.close()
    ion_in_file.close()