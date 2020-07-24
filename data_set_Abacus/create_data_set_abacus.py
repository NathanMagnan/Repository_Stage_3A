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
n_proc = comm.Get_size() # Should be equal to 61

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
path = '/home/astro/magnan/Abacus/AbacusCosmos_720box_products/AbacusCosmos_720box_'
path += number_str
path += '_products/AbacusCosmos_720box_'
path += number_str
path += '_rockstar_halos/z0.100'
if (i > 40):
	path = '/home/astro/magnan/Abacus/AbacusCosmos_720box_planck_products/AbacusCosmos_720box_planck_00-'
	path += str(i - 41)
	path += '_products/AbacusCosmos_720box_planck_00-'
	path += str(i - 41)
	path += '_rockstar_halos/z0.100'

# creating a catalogue object
ab = cat.Catalogue_Abacus(basePath = path)
print("simulation " + str(i) + " : catalogue created")

# gettting the data
ab.initialise_data()
print("simulation " + str(i) + " : data acquired")

# computing the histogram
ab.compute_MST_histogram(jacknife = False)
print("simulation " + str(i) + " : histogram computed")

"""" saving the full statistics """
target = "/home/astro/magnan/Repository_Stage_3A/Full_MST_stats_Abacus/MST_stats_Catalogue_"

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

"""" saving the shortened statistics """
target = "/home/astro/magnan/Repository_Stage_3A/data_set_Abacus/data_set_Abacus"

# selecting the important points
X_d = ab.MST_histogram['x_d'] / 6 # we normalize every histogram to have a typical scale of 1
Y_d = np.log10(ab.MST_histogram['y_d'])

X_l_long = np.log10(ab.MST_histogram['x_l']) # we normalize every histogram to have a typical scale of 1
Y_l_long = np.log10(ab.MST_histogram['y_l'])
X_l = X_l_long[4::10]
Y_l = Y_l_long[4::10]

X_b_long = np.log10(ab.MST_histogram['x_b']) # we normalize every histogram to have a typical scale of 1
Y_b_long = np.log10(ab.MST_histogram['y_b'])
X_b = X_b_long[4::10]
Y_b = Y_b_long[4::10]

X_s_long = ab.MST_histogram['x_s'] # already normalized between 0 and 1
Y_s_long = np.log10(ab.MST_histogram['y_s'])
X_s = X_s_long[2::5]
Y_s = Y_s_long[2::5]

# reading the simulation parameters
input_path = path + '/header'
param = InputFile.InputFile(fn = input_path)

h0 = (param.get('H0') - 60)  / (75 - 60) # we normalize every parameter to use the whole range from 0 to 1
w0 = (param.get('w0') - (-1.40)) / ((-0.60) - (-1.40)) # we normalize every parameter to use the whole range from 0 to 1
ns = (param.get('ns') - 0.920) / (0.995 - 0.920) # we normalize every parameter to use the whole range from 0 to 1
sigma8 = (param.get('sigma_8') - 0.64) / (1.04 - 0.64) # we normalize every parameter to use the whole range from 0 to 1
omegaM = (param.get('Omega_M') - 0.250) / (0.375 - 0.250) # we normalize every parameter to use the whole range from 0 to 1
print("simulation " + str(i) + " : parameters read")

# creating the set of new points
X_data_new_d = np.reshape([[h0, w0, ns, sigma8, omegaM, x_d, 0] for x_d in X_d], (6, 7))
X_data_new_l = np.reshape([[h0, w0, ns, sigma8, omegaM, x_l, 1] for x_l in X_l], (10, 7))
X_data_new_b = np.reshape([[h0, w0, ns, sigma8, omegaM, x_b, 2] for x_b in X_b], (10, 7))
X_data_new_s = np.reshape([[h0, w0, ns, sigma8, omegaM, x_s, 3] for x_s in X_s], (10, 7))
X_data_new = np.concatenate((X_data_new_d, X_data_new_l, X_data_new_b, X_data_new_s), 0)

Y_data_new_d = np.reshape([[y_d] for y_d in Y_d], (6, 1))
Y_data_new_l = np.reshape([[y_l] for y_l in Y_l], (10, 1))
Y_data_new_b = np.reshape([[y_b] for y_b in Y_b], (10, 1))
Y_data_new_s = np.reshape([[y_s] for y_s in Y_s], (10, 1))
Y_data_new = np.concatenate((Y_data_new_d, Y_data_new_l, Y_data_new_b, Y_data_new_s), 0)

# saving the new points
np.savetxt(str(target) + "_" + str(i) + "_X_data", X_data_new)
np.savetxt(str(target) + "_" + str(i) + "_Y_data", Y_data_new)

print("work done on simulation " + str(i))