## Imports
import numpy as np
from mpi4py import MPI

import sys
import os
sys.path.append('/home/astro/magnan/Repository_Stage_3A')
import catalogue_tools as cat
sys.path.append("/home/astro/magnan")
from AbacusCosmos import InputFile
os.chdir('/home/astro/magnan')

print("All imports successful")

## Setting up the MPI
print("Starting to set up the MPI")
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_proc = comm.Get_size() # Should be equal to 41

i = int(rank)

print("MPI set up")

## creating the data set
print("starting to work on creating the data set")

print("starting to work on simulation " + str(i))

# getting the basepath
path = '/hpcstorage/zhaoc/BOSS_PATCHY_MOCKS/PATCHY_CMASS/box1/'

# getting the list of files
list_files = os.listdir(path)

# getting the file
file = list_files[i]
my_file = os.path.join(path, file)

# creating a catalogue object
pa = cat.Catalogue_Patchy(basePath = my_file)
print("simulation " + str(i) + " : catalogue created")

# gettting the data
pa.initialise_data()
print("simulation " + str(i) + " : data acquired")

# computing the histogram
pa.compute_MST_histogram()
print("simulation " + str(i) + " : histogram computed")

"""" saving the full statistics """
target = "/home/astro/magnan/Repository_Stage_3A/Full_MST_stats_Patchy/MST_stats_Catalogue_"

X_d = pa.MST_histogram['x_d']
Y_d = pa.MST_histogram['y_d']
np.savetxt(str(target) + str(i) + "_X_d", X_d)
np.savetxt(str(target) + str(i) + "_Y_d", Y_d)

X_l = pa.MST_histogram['x_l']
Y_l = pa.MST_histogram['y_l']
np.savetxt(str(target) + str(i) + "_X_l", X_l)
np.savetxt(str(target) + str(i) + "_Y_l", Y_l)

X_b = pa.MST_histogram['x_b']
Y_b = pa.MST_histogram['y_b']
np.savetxt(str(target) + str(i) + "_X_b", X_b)
np.savetxt(str(target) + str(i) + "_Y_b", Y_b)

X_s = pa.MST_histogram['x_s']
Y_s = pa.MST_histogram['y_s']
np.savetxt(str(target) + str(i) + "_X_s", X_s)
np.savetxt(str(target) + str(i) + "_Y_s", Y_s)

print("data set fully created")
