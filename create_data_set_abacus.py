""" tested, very long but compiles """
## Imports
import numpy as np

import sys
import os
sys.path.append('/home/astro/magnan/Repository_Stage_3A')
import catalogue_tools as cat
sys.path.append("/home/astro/magnan")
from AbacusCosmos import InputFile
os.chdir('/home/astro/magnan')

print("All imports successful")

## creating the data set
print("starting to work on creating the data set")
X_data = []
Y_data = []

for i in range(40):
    print("starting to work on simulation " + str(i))
    
    # getting the basepath
    if (i < 10):
        number_str = str(0) + str(i)
    elif (i<40):
        number_str = str(i)
    else:
        number_str = 'planck'
    path = '/home/astro/magnan/Abacus/AbacusCosmos_720box_products/AbacusCosmos_720box_'
    path += number_str
    path += '_products/AbacusCosmos_720box_'
    path += number_str
    path += '_rockstar_halos/z0.100'
    
    # creating a catalogue object
    ab = cat.Catalogue_Abacus(basePath = path)
    print("simulation " + str(i) + " : catalogue created")
    
    # gettting the data
    ab.initialise_data()
    print("simulation " + str(i) + " : data acquired")
    
    # computing the histogram
    ab.compute_MST_histogram()
    print("simulation " + str(i) + " : histogram computed")
    
    # reading the histogram
    X_d = ab.MST_histogram['x_d']
    Y_d = ab.MST_histogram['y_d'] / np.max(ab.MST_histogram['y_d']) # we normalize every histogram to 1
    
    X_l_long = ab.MST_histogram['x_l']
    Y_l_long = ab.MST_histogram['y_l'] / np.max(ab.MST_histogram['y_l']) # we normalize every histogram to 1
    X_l = X_l_long[4::10]
    Y_l = Y_l_long[4::10] # we only keep 10 points per histogram to avoid having huge redundant dataset
    
    X_b_long = ab.MST_histogram['x_b']
    Y_b_long = ab.MST_histogram['y_b'] / np.max(ab.MST_histogram['y_b']) # we normalize every histogram to 1
    X_b = X_b_long[4::10]
    Y_b = Y_b_long[4::10] # we only keep 10 points per histogram to avoid having huge redundant dataset
    
    X_s_long = ab.MST_histogram['x_s']
    Y_s_long = ab.MST_histogram['y_s'] / np.max(ab.MST_histogram['y_s']) # we normalize every histogram to 1
    X_s = X_s_long[2::5]
    Y_s = Y_s_long[2::5] # we only keep 10 points per histogram to avoid having huge redundant dataset
    print("simulation " + str(i) + " : histogram read")
    
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
    print("simulation " + str(i) + " : new data points created")
    
    # adding the new points
    if (i == 0):
        X_data = X_data_new
        Y_data = Y_data_new
    else:
        X_data = np.concatenate((X_data, X_data_new), 0)
        Y_data = np.concatenate((Y_data, Y_data_new), 0)
    print("simulation " + str(i) + " : new data points added to the data set")

print("data set fully created")

## saving the data set
print("starting to work on saving the data set")

target = "/home/astro/magnan/Repository_Stage_3A/data_set_Abacus"

np.savetxt(str(target) + "_X_data", X_data)
np.savetxt(str(target) + "_Y_data", Y_data)

print("data fully saved")
print("Data set ready !")