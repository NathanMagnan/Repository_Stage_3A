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
if (i < 10):
    number_str = str(0) + str(i)
elif (i < 40):
    number_str = str(i)
else:
    number_str = 'planck'
path = '/hpcstorage/magnan/Abacus_1100/AbacusCosmos_1100box_products/AbacusCosmos_1100box_'
path += number_str
path += '_products/AbacusCosmos_1100box_'
path += number_str
path += '_rockstar_halos/z0.300'

# creating a catalogue object
ab = cat.Catalogue_Abacus(basePath = path)
print("simulation " + str(i) + " : catalogue created")

# gettting the data
ab.initialise_data()
print("simulation " + str(i) + " : data acquired")
print(" density : " + str(np.shape(ab.CM)[0] / (1100**3)))

# computing the histogram
ab.compute_MST_histogram(jacknife = False)
print("simulation " + str(i) + " : histogram computed")

"""" saving the full statistics """
target = "/home/astro/magnan/Repository_Stage_3A/Full_MST_stats_Abacus_1100/MST_stats_Catalogue_"

X_d = ab.MST_histogram['x_d']
Y_d = ab.MST_histogram['y_d']
np.savetxt(str(target) + str(i) + "_X_d", X_d)
np.savetxt(str(target) + str(i) + "_Y_d", Y_d)

X_l = ab.MST_histogram['x_l']
Y_l = ab.MST_histogram['y_l']
np.savetxt(str(target) + str(i) + "_X_l", X_l)
np.savetxt(str(target) + str(i) + "_Y_l", Y_l)

X_b = ab.MST_histogram['x_b']
Y_b = ab.MST_histogram['y_b']
np.savetxt(str(target) + str(i) + "_X_b", X_b)
np.savetxt(str(target) + str(i) + "_Y_b", Y_b)

X_s = ab.MST_histogram['x_s']
Y_s = ab.MST_histogram['y_s']
np.savetxt(str(target) + str(i) + "_X_s", X_s)
np.savetxt(str(target) + str(i) + "_Y_s", Y_s)

print("data set fully created")
