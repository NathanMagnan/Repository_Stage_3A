## Imports
import numpy as np
import math as m
import GPy as GPy
import emcee
import sys
import os
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)

#sys.path.append('/home/astro/magnan/Repository_Stage_3A')
sys.path.append('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A')
import GP_tools as GP
#os.chdir('/home/astro/magnan')
os.chdir('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A')

print("All imports successful")

## Importing the whole Abacus data
print("starting to load the data")

#target = "/home/astro/magnan/Repository_Stage_3A/Full_MST_stats_Abacus/MST_stats_Catalogue_"
target = "C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Full_MST_stats_Abacus/MST_stats_Catalogue_"

dict = {'X_d' : [], 'Y_d' : [], 'Y_d_std' : [], 'X_l' : [], 'Y_l' : [], 'Y_l_std' : [], 'X_b' : [], 'Y_b' : [], 'Y_b_std' : [], 'X_s' : [], 'Y_s' : [], 'Y_s_std' : []}

for i in range(61):    
    X_d_a = np.loadtxt(str(target) + str(i) + "_X_d")
    Y_d_a = np.loadtxt(str(target) + str(i) + "_Y_d")
    Y_d_std_a = np.loadtxt(str(target) + str(i) + "_Y_d_std")
    dict['X_d'].append(X_d_a)
    dict['Y_d'].append(Y_d_a)
    dict['Y_d_std'].append(Y_d_std_a)
    
    X_l_a = np.loadtxt(str(target) + str(i) + "_X_l")
    Y_l_a = np.loadtxt(str(target) + str(i) + "_Y_l")
    Y_l_std_a = np.loadtxt(str(target) + str(i) + "_Y_l_std")
    dict['X_l'].append(X_l_a)
    dict['Y_l'].append(Y_l_a)
    dict['Y_l_std'].append(Y_l_std_a)
    
    X_b_a = np.loadtxt(str(target) + str(i) + "_X_b")
    Y_b_a = np.loadtxt(str(target) + str(i) + "_Y_b")
    Y_b_std_a = np.loadtxt(str(target) + str(i) + "_Y_b_std")
    dict['X_b'].append(X_b_a)
    dict['Y_b'].append(Y_b_a)
    dict['Y_b_std'].append(Y_b_std_a)
    
    X_s_a = np.loadtxt(str(target) + str(i) + "_X_s")
    Y_s_a = np.loadtxt(str(target) + str(i) + "_Y_s")
    Y_s_std_a = np.loadtxt(str(target) + str(i) + "_Y_s_std")
    dict['X_s'].append(X_s_a)
    dict['Y_s'].append(Y_s_a)
    dict['Y_s_std'].append(Y_s_std_a)

print("data fully loaded")

## Importing the data
print("Connexion successfull")
print("starting to load the data")

#target = "/home/astro/magnan/Repository_Stage_3A/data_set_Abacus/data_set_Abacus"
target = 'C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/data_set_Abacus/data_set_Abacus_'

"""
d = 0->4
l = 9->15
b = 17->25
s = 26->35
"""

n_points_per_simulation_complete = 36
n_simulations = 40
n_fiducial = 21

X_d = None
Y_d = None
X_l = None
Y_l = None
X_b = None
Y_b = None
X_s = None
Y_s = None

for i in range(n_fiducial + n_simulations):
    X_data_new = np.loadtxt(fname = str(target) + str(i) + "_X_data") # numpy array with fields h0, w0, ns, sigma8, omegaM, d/l/b/s
    Y_data_new = np.loadtxt( str(target) + str(i) + "_Y_data") # numpy array with field Nd/l/b/s
    
    for j in range(n_points_per_simulation_complete): # there are infinite values because of the log normalisation
        Y_data_new[j] = max(Y_data_new[j], 0)
    
    if i == 0:
        X_d = X_data_new[0 : 4, 0 : 6]
        Y_d = Y_data_new[0 : 4]
        X_l = X_data_new[10 : 15, 0 : 6]
        Y_l = Y_data_new[10 : 15]
        X_b = X_data_new[19 : 24, 0 : 6]
        Y_b = Y_data_new[19 : 24]
        X_s = X_data_new[28 : 33, 0 : 6]
        Y_s = Y_data_new[28 : 33]
    else:
        X_d = np.concatenate((X_data_new[0 : 4, 0:6], X_d))
        Y_d = np.concatenate((Y_data_new[0 : 4], Y_d))
        X_l = np.concatenate((X_data_new[10 : 15, 0:6], X_l))
        Y_l = np.concatenate((Y_data_new[10 : 15], Y_l))
        X_b = np.concatenate((X_data_new[19 : 24, 0:6], X_b))
        Y_b = np.concatenate((Y_data_new[19 : 24], Y_b))
        X_s = np.concatenate((X_data_new[28 : 33, 0:6], X_s))
        Y_s = np.concatenate((Y_data_new[28 : 33], Y_s))

X_d_planck = X_d[:(n_fiducial) * 4]
X_l_planck = X_l[:(n_fiducial) * 5]
X_b_planck = X_b[:(n_fiducial) * 5]
X_s_planck = X_s[:(n_fiducial) * 5]

X_d_data = X_d[(n_fiducial) * 4:]
Y_d_data = Y_d[(n_fiducial) * 4:]
X_l_data = X_l[(n_fiducial) * 5:]
Y_l_data = Y_l[(n_fiducial) * 5:]
X_b_data = X_b[(n_fiducial) * 5:]
Y_b_data = Y_b[(n_fiducial) * 5:]
X_s_data = X_s[(n_fiducial) * 5:]
Y_s_data = Y_s[(n_fiducial) * 5:]
Y_d_data = np.reshape(Y_d_data, (n_simulations * 4, 1))
Y_l_data = np.reshape(Y_l_data, (n_simulations * 5, 1))
Y_b_data = np.reshape(Y_b_data, (n_simulations * 5, 1))
Y_s_data = np.reshape(Y_s_data, (n_simulations * 5, 1))

X_d_planck_expected = X_d_data[-4:, 5]
Y_d_planck_expected = np.asarray([0 for i in range(np.shape(X_d_data[0:4, 5])[0])])
Y_d_std_planck_expected = np.asarray([0 for i in range(np.shape(X_d_data[0:4, 5])[0])])
for i in range(n_fiducial):
    New = []
    for x1 in X_d_planck_expected:
        min = 10
        j_min = 0
        for j in range(np.shape(dict['X_d'][i])[0]):
            x2 = dict['X_d'][i][j] / 6
            if (abs(x1 - x2) < min):
                min = abs(x1 - x2)
                j_min = j
        New.append(dict['Y_d'][i + 40][j_min])

    New = np.asarray(New)
    
    Mean_old = Y_d_planck_expected.copy()
    Std_old = Y_d_std_planck_expected.copy()
    
    Mean_new = (i * Mean_old + New) / (i + 1)
    Std_new = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_new**2)
    
    Y_d_planck_expected = Mean_new.copy()
    Y_d_std_planck_expected = Std_new.copy()
Y_d_std_planck_expected = Y_d_std_planck_expected / (np.log(10) * Y_d_planck_expected)
Y_d_planck_expected = np.log10(Y_d_planck_expected)
for i in range(4):
    X_d_planck[i, 5] = X_d_planck_expected[i]
X_d_planck = X_d_planck[0:4]

X_l_planck_expected = X_l_data[-5:, 5]
Y_l_planck_expected = np.asarray([0 for i in range(np.shape(X_l_data[-5:, 5])[0])])
Y_l_std_planck_expected = np.asarray([0 for i in range(np.shape(X_l_data[-5:, 5])[0])])
for i in range(n_fiducial):
    New = []
    for x1 in 10**X_l_planck_expected:
        min = 10
        j_min = 0
        for j in range(np.shape(dict['X_l'][n_simulations + i])[0]):
            x2 = dict['X_l'][n_simulations + i][j]
            if (abs(x1 - x2) < min):
                min = abs(x1 - x2)
                j_min = j
        try:
            New.append((dict['Y_l'][i + 40][j_min - 1] + dict['Y_l'][i + 40][j_min] + dict['Y_l'][i + 40][j_min + 1]) / 3)
        except:
            New.append(dict['Y_l'][i + 40][j_min])

    New = np.asarray(New)
    
    Mean_old = Y_l_planck_expected.copy()
    Std_old = Y_l_std_planck_expected.copy()
    
    Mean_new = (i * Mean_old + New) / (i + 1)
    Std_new = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_new**2)
    
    Y_l_planck_expected = Mean_new.copy()
    Y_l_std_planck_expected = Std_new.copy()
Y_l_std_planck_expected = Y_l_std_planck_expected / (np.log(10) * Y_l_planck_expected)
Y_l_planck_expected = np.log10(Y_l_planck_expected)
for i in range(5):
    X_l_planck[i, 5] = X_l_planck_expected[i]
X_l_planck = X_l_planck[0:5]

X_b_planck_expected = X_b_data[-5:, 5]
Y_b_planck_expected = np.asarray([0 for i in range(np.shape(X_b_data[-5:, 5])[0])])
Y_b_std_planck_expected = np.asarray([0 for i in range(np.shape(X_b_data[-5:, 5])[0])])
for i in range(n_fiducial):
    New = []
    for x1 in 10**X_b_planck_expected:
        min = 10
        j_min = 0
        for j in range(np.shape(dict['X_b'][n_simulations + i])[0]):
            x2 = dict['X_b'][n_simulations + i][j]
            if (abs(x1 - x2) < min):
                min = abs(x1 - x2)
                j_min = j
        try:
            New.append((dict['Y_b'][i + 40][j_min - 1] + dict['Y_b'][i + 40][j_min] + dict['Y_b'][i + 40][j_min + 1]) / 3)
        except:
            New.append(dict['Y_b'][i + 40][j_min])

    New = np.asarray(New)
    
    Mean_old = Y_b_planck_expected.copy()
    Std_old = Y_b_std_planck_expected.copy()
    
    Mean_new = (i * Mean_old + New) / (i + 1)
    Std_new = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_new**2)
    
    Y_b_planck_expected = Mean_new.copy()
    Y_b_std_planck_expected = Std_new.copy()
Y_b_std_planck_expected = Y_b_std_planck_expected / (np.log(10) * Y_b_planck_expected)
Y_b_planck_expected = np.log10(Y_b_planck_expected)
for i in range(5):
    X_b_planck[i, 5] = X_b_planck_expected[i]
X_b_planck = X_b_planck[0:5]

X_s_planck_expected = X_s_data[-5:, 5]
Y_s_planck_expected = np.asarray([0 for i in range(np.shape(X_s_data[-5:, 5])[0])])
Y_s_std_planck_expected = np.asarray([0 for i in range(np.shape(X_s_data[-5:, 5])[0])])
for i in range(n_fiducial):
    New = []
    for x1 in X_s_planck_expected:
        min = 10
        j_min = 0
        for j in range(np.shape(dict['X_s'][n_simulations + i])[0]):
            x2 = dict['X_s'][n_simulations + i][j]
            if (abs(x1 - x2) < min):
                min = abs(x1 - x2)
                j_min = j
        try:
            New.append((dict['Y_s'][i + 40][j_min - 1] + dict['Y_s'][i + 40][j_min] + dict['Y_s'][i + 40][j_min + 1]) / 3)
        except:
            New.append(dict['Y_s'][i + 40][j_min])

    New = np.asarray(New)
    
    Mean_old = Y_s_planck_expected.copy()
    Std_old = Y_s_std_planck_expected.copy()
    
    Mean_new = (i * Mean_old + New) / (i + 1)
    Std_new = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_new**2)
    
    Y_s_planck_expected = Mean_new.copy()
    Y_s_std_planck_expected = Std_new.copy()
Y_s_std_planck_expected = Y_s_std_planck_expected / (np.log(10) * Y_s_planck_expected)
Y_s_planck_expected = np.log10(Y_s_planck_expected)
for i in range(5):
    X_s_planck[i, 5] = X_s_planck_expected[i]
X_s_planck = X_s_planck[0:5]

print("data loaded")

## Setting up the GPs
print("starting to define the Gps")

gp = GP.GP(X = [X_d_data, X_l_data, X_b_data, X_s_data], Y = [Y_d_data, Y_l_data, Y_b_data, Y_s_data], N_points_per_simu = [4, 5, 5, 5], Noise = [None, None, None, None], type_kernel = "Separated")

print("models defined")

## Optimising the hyperparameters - Gradient Descent
print("Starting to optimise the hyperparameters")

gp.optimize_models(optimizer = 'lbfgsb')

print("Hyperparameters optimised")

gp.print_models()

## Defining the prior
print("starting to define the prior")

def prior(X_01):
    X = [0, 0, 0, 0, 0]
    X[0] = X_01[0] * (75 - 60) + 60
    X[1] = X_01[1] * ((-0.60) - (-1.40)) + (-1.40)
    X[2] = X_01[2] * (0.995 - 0.920) + 0.920
    X[3] = X_01[3] * (1.04 - 0.64) + 0.64
    X[4] = X_01[4] * (0.375 - 0.250) + 0.250
    return X

print("prior defined")

## Defining the expectation
print("starting to define the expectation")

X_d_abacus = dict['X_d'][0]
Mean_d_abacus = np.asarray([0 for k in range(np.shape(dict['X_d'][0])[0])])
Std_d_abacus = np.asarray([0 for k in range(np.shape(dict['X_d'][0])[0])])
for k in range(n_simulations):
    New = []
    for x1 in X_d_abacus:
        min = 10
        l_min = 0
        for l in range(np.shape(X_d_abacus)[0]):
            x2 = dict['X_d'][k][l]
            if (abs(x1 - x2) < min):
                min = abs(x1 - x2)
                l_min = l
        
        New.append(dict['Y_d'][k][l_min])
    New = np.asarray(New)
    
    Mean_old = Mean_d_abacus.copy()
    Std_old = Std_d_abacus.copy()
    
    Mean_new = (k * Mean_old + New) / (k + 1)
    Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + New**2) / (k + 1) - Mean_new**2)
    
    Mean_d_abacus = Mean_new.copy()
    Std_d_abacus = Std_new.copy()

Mean_d_fidu = np.asarray([0 for k in range(np.shape(dict['X_d'][0])[0])])
Std_d_fidu = np.asarray([0 for k in range(np.shape(dict['X_d'][0])[0])])
for k in range(n_fiducial):
    New = []
    for x1 in X_d_abacus:
        min = 10
        l_min = 0
        for l in range(np.shape(X_d_abacus)[0]):
            x2 = dict['X_d'][k + n_simulations][l]
            if (abs(x1 - x2) < min):
                min = abs(x1 - x2)
                l_min = l
        
        New.append(dict['Y_d'][k + n_simulations][l_min])
    New = np.asarray(New)
        
    Mean_old = Mean_d_fidu.copy()
    Std_old = Std_d_fidu.copy()
    
    Mean_new = (k * Mean_old + New) / (k + 1)
    Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + New**2) / (k + 1) - Mean_new**2)
    
    Mean_d_fidu = Mean_new.copy()
    Std_d_fidu = Std_new.copy()

X_l_abacus = dict['X_l'][0]
Mean_l_abacus = np.asarray([0 for k in range(np.shape(dict['X_l'][0])[0])])
Std_l_abacus = np.asarray([0 for k in range(np.shape(dict['X_l'][0])[0])])
for k in range(n_simulations):
    New = []
    for x1 in X_l_abacus:
        min = 10
        l_min = 0
        for l in range(np.shape(X_l_abacus)[0]):
            x2 = dict['X_l'][k][l]
            if (abs(x1 - x2) < min):
                min = abs(x1 - x2)
                l_min = l
        try:
            New.append((dict['Y_l'][k][l_min - 1] + dict['Y_l'][k][l_min] + dict['Y_l'][k][l_min + 1]) / 3)
        except:
            New.append(dict['Y_l'][k][l_min])
    New = np.asarray(New)
    
    Mean_old = Mean_l_abacus.copy()
    Std_old = Std_l_abacus.copy()
    
    Mean_new = (k * Mean_old + New) / (k + 1)
    Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + New**2) / (k + 1) - Mean_new**2)
    
    Mean_l_abacus = Mean_new.copy()
    Std_l_abacus = Std_new.copy()

Mean_l_fidu = np.asarray([0 for k in range(np.shape(dict['X_l'][0])[0])])
Std_l_fidu = np.asarray([0 for k in range(np.shape(dict['X_l'][0])[0])])
for k in range(n_fiducial):
    New = []
    for x1 in X_l_abacus:
        min = 10
        l_min = 0
        for l in range(np.shape(X_l_abacus)[0]):
            x2 = dict['X_l'][k + n_simulations][l]
            if (abs(x1 - x2) < min):
                min = abs(x1 - x2)
                l_min = l
        try:
            New.append((dict['Y_l'][k + n_simulations][l_min - 1] + dict['Y_l'][k + n_simulations][l_min] + dict['Y_l'][k + n_simulations][l_min + 1]) / 3)
        except:
            New.append(dict['Y_l'][k + n_simulations][l_min])
    New = np.asarray(New)
        
    Mean_old = Mean_l_fidu.copy()
    Std_old = Std_l_fidu.copy()
    
    Mean_new = (k * Mean_old + New) / (k + 1)
    Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + New**2) / (k + 1) - Mean_new**2)
    
    Mean_l_fidu = Mean_new.copy()
    Std_l_fidu = Std_new.copy()

X_b_abacus = dict['X_b'][0]
Mean_b_abacus = np.asarray([0 for k in range(np.shape(dict['X_b'][0])[0])])
Std_b_abacus = np.asarray([0 for k in range(np.shape(dict['X_b'][0])[0])])
for k in range(n_simulations):
    New = []
    for x1 in X_b_abacus:
        min = 10
        l_min = 0
        for l in range(np.shape(X_b_abacus)[0]):
            x2 = dict['X_b'][k][l]
            if (abs(x1 - x2) < min):
                min = abs(x1 - x2)
                l_min = l
        try:
            New.append((dict['Y_b'][k][l_min - 1] + dict['Y_b'][k][l_min] + dict['Y_b'][k][l_min + 1]) / 3)
        except:
            New.append(dict['Y_b'][k][l_min])
    New = np.asarray(New)
    
    Mean_old = Mean_b_abacus.copy()
    Std_old = Std_b_abacus.copy()
    
    Mean_new = (k * Mean_old + New) / (k + 1)
    Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + New**2) / (k + 1) - Mean_new**2)
    
    Mean_b_abacus = Mean_new.copy()
    Std_b_abacus = Std_new.copy()

Mean_b_fidu = np.asarray([0 for k in range(np.shape(dict['X_b'][0])[0])])
Std_b_fidu = np.asarray([0 for k in range(np.shape(dict['X_b'][0])[0])])
for k in range(n_fiducial):
    New = []
    for x1 in X_b_abacus:
        min = 10
        l_min = 0
        for l in range(np.shape(X_b_abacus)[0]):
            x2 = dict['X_b'][k + n_simulations][l]
            if (abs(x1 - x2) < min):
                min = abs(x1 - x2)
                l_min = l
        try:
            New.append((dict['Y_b'][k + n_simulations][l_min - 1] + dict['Y_b'][k + n_simulations][l_min] + dict['Y_b'][k + n_simulations][l_min + 1]) / 3)
        except:
            New.append(dict['Y_b'][k + n_simulations][l_min])
    New = np.asarray(New)
        
    Mean_old = Mean_b_fidu.copy()
    Std_old = Std_b_fidu.copy()
    
    Mean_new = (k * Mean_old + New) / (k + 1)
    Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + New**2) / (k + 1) - Mean_new**2)
    
    Mean_b_fidu = Mean_new.copy()
    Std_b_fidu = Std_new.copy()

X_s_abacus = dict['X_s'][0]
Mean_s_abacus = np.asarray([0 for k in range(np.shape(dict['X_s'][0])[0])])
Std_s_abacus = np.asarray([0 for k in range(np.shape(dict['X_s'][0])[0])])
for k in range(n_simulations):
    New = []
    for x1 in X_s_abacus:
        min = 10
        l_min = 0
        for l in range(np.shape(X_s_abacus)[0]):
            x2 = dict['X_s'][k][l]
            if (abs(x1 - x2) < min):
                min = abs(x1 - x2)
                l_min = l
        try:
            New.append((dict['Y_s'][k][l_min - 1] + dict['Y_s'][k][l_min] + dict['Y_s'][k][l_min + 1]) / 3)
        except:
            New.append(dict['Y_s'][k][l_min])
    New = np.asarray(New)
    
    Mean_old = Mean_s_abacus.copy()
    Std_old = Std_s_abacus.copy()
    
    Mean_new = (k * Mean_old + New) / (k + 1)
    Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + New**2) / (k + 1) - Mean_new**2)
    
    Mean_s_abacus = Mean_new.copy()
    Std_s_abacus = Std_new.copy()

Mean_s_fidu = np.asarray([0 for k in range(np.shape(dict['X_s'][0])[0])])
Std_s_fidu = np.asarray([0 for k in range(np.shape(dict['X_s'][0])[0])])
for k in range(n_fiducial):
    New = []
    for x1 in X_s_abacus:
        min = 10
        l_min = 0
        for l in range(np.shape(X_s_abacus)[0]):
            x2 = dict['X_s'][k + n_simulations][l]
            if (abs(x1 - x2) < min):
                min = abs(x1 - x2)
                l_min = l
        try:
            New.append((dict['Y_s'][k + n_simulations][l_min - 1] + dict['Y_s'][k + n_simulations][l_min] + dict['Y_s'][k + n_simulations][l_min + 1]) / 3)
        except:
            New.append(dict['Y_s'][k + n_simulations][l_min])
    New = np.asarray(New)
        
    Mean_old = Mean_s_fidu.copy()
    Std_old = Std_s_fidu.copy()
    
    Mean_new = (k * Mean_old + New) / (k + 1)
    Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + New**2) / (k + 1) - Mean_new**2)
    
    Mean_s_fidu = Mean_new.copy()
    Std_s_fidu = Std_new.copy()

print("expectation defined")

## Defining the log-likelihood
print("Starting to define the log-likelihood")

Boundaries = [[60, 75], [-1.40, -0.60], [0.920, 0.995], [0.64, 1.04], [0.250, 0.375]]

def chi2(X):    
    # Making boundaries
    if ((X[0] > Boundaries[0][1]) or (X[0] < Boundaries[0][0])):
        return(- m.inf)
    if ((X[1] > Boundaries[1][1]) or (X[1] < Boundaries[1][0])):
        return(- m.inf)
    if ((X[2] > Boundaries[2][1]) or (X[2] < Boundaries[2][0])):
        return(- m.inf)
    if ((X[3] > Boundaries[3][1]) or (X[3] < Boundaries[3][0])):
        return(- m.inf)
    if ((X[4] > Boundaries[4][1]) or (X[4] < Boundaries[4][0])):
        return(- m.inf)
    
    # Reading the parameters
    h0 = (X[0] - 60) / (75 - 60)
    w0 = (X[1] - (-1.40)) / ((-0.60) - (-1.40))
    ns = (X[2] - 0.920) / (0.995 - 0.920)
    sigma8 = (X[3] - 0.64) / (1.04 - 0.64)
    omegaM = (X[4] - 0.250) / (0.375 - 0.250)
    X_new = np.asarray([h0, w0, ns, sigma8, omegaM])
    
    # Making the prediction
    X_new = np.reshape(X_new, (1, 5))
    X_predicted, Y_predicted, Cov = gp.compute_prediction(X_new)
    
    # giving the right shape to the predicted value
    Y_d_predicted = Y_predicted[0][0]
    Y_l_predicted = Y_predicted[0][1]
    Y_b_predicted = Y_predicted[0][2]
    Y_s_predicted = Y_predicted[0][3]
    Y_predicted = []
    Y_predicted.append([Y_d_predicted, Y_l_predicted, Y_b_predicted, Y_s_predicted])
    
    # searching for the expected value
    X_d_predicted = X_predicted[0][0][:, 5]
    X_l_predicted = X_predicted[0][1][:, 5]
    X_b_predicted = X_predicted[0][2][:, 5]
    X_s_predicted = X_predicted[0][3][:, 5]
    
    Y_d_expected = np.asarray([0 for i in range(np.shape(X_d_predicted)[0])])
    for k in range(np.shape(Y_d_expected)[0]):
        l_min = 0
        min = 10
        x1 = X_d_predicted[k]
        for l in range(np.shape(X_d_abacus)[0]):
            x2 = X_d_abacus[l] / 6
            if (abs(x1 - x2) < min):
                min = abs(x1 - x2)
                l_min = l
        Y_d_expected[k] = Mean_d_fidu[l_min]
    Y_d_expected = np.log10(np.asarray(Y_d_expected))
    
    Y_l_expected = np.asarray([0 for i in range(np.shape(X_l_predicted)[0])])
    for k in range(np.shape(Y_l_expected)[0]):
        l_min = 0
        min = 10
        x1 = X_l_predicted[k]
        for l in range(np.shape(X_l_abacus)[0]):
            x2 = np.log10(X_l_abacus[l])
            if (abs(x1 - x2) < min):
                min = abs(x1 - x2)
                l_min = l
        Y_l_expected[k] = Mean_l_fidu[l_min]
    Y_l_expected = np.log10(np.asarray(Y_l_expected))
                
    Y_b_expected = np.asarray([0 for i in range(np.shape(X_b_predicted)[0])])
    for k in range(np.shape(Y_b_expected)[0]):
        l_min = 0
        min = 10
        x1 = X_b_predicted[k]
        for l in range(np.shape(X_b_abacus)[0]):
            x2 = np.log10(X_b_abacus[l])
            if (abs(x1 - x2) < min):
                min = abs(x1 - x2)
                l_min = l
        Y_b_expected[k] = Mean_b_fidu[l_min]
    Y_b_expected = np.log10(np.asarray(Y_b_expected))
    
    Y_s_expected = np.asarray([0 for i in range(np.shape(X_s_predicted)[0])])
    for k in range(np.shape(Y_s_expected)[0]):
        l_min = 0
        min = 10
        x1 = X_s_predicted[k]
        for l in range(np.shape(X_s_abacus)[0]):
            x2 = X_s_abacus[l]
            if (abs(x1 - x2) < min):
                min = abs(x1 - x2)
                l_min = l
        Y_s_expected[k] = Mean_s_fidu[l_min]
    Y_s_expected = np.log10(np.asarray(Y_s_expected))
    
    # Giving the right shape to the expected value
    Y_d_expected = np.reshape(Y_d_expected, (4, 1))
    Y_l_expected = np.reshape(Y_l_expected, (5, 1))
    Y_b_expected = np.reshape(Y_b_expected, (5, 1))
    Y_s_expected = np.reshape(Y_s_expected, (5, 1))
    Y_expected = []
    Y_expected.append([Y_d_expected, Y_l_expected, Y_b_expected, Y_s_expected])
    
    # Defining the noises
    Noise_predicted = Cov
    Noise_expected = [[Y_d_std_planck_expected, Y_l_std_planck_expected, Y_b_std_planck_expected, Y_s_std_planck_expected]]
    
    # Computing the likelihood
    chi2 = gp.likelihood_chi2(Y_observation = Y_expected, Noise_observation = Noise_expected, Y_model = Y_predicted, Noise_model = Noise_predicted)
    
    if (m.isnan(chi2)):
        print(X)
        return(- m.inf)
    
    # returning the log-likelihood or chi_2
    return(-0.5 * chi2)

print("Likelihood defined")

## Defining the problem
print("Starting to define the problem")

n_dims = 5
n_walkers = 32

#my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')
my_path = os.path.abspath('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/EMCEE/')
my_file = 'Figure_12_5'
my_file = os.path.join(my_path, my_file)
backend = emcee.backends.HDFBackend(my_file)
backend.reset(n_walkers, n_dims)

A = np.random.rand(n_walkers, n_dims)
Initial_guess = np.asarray([prior(A[i]) for i in range(n_walkers)])

sampler = emcee.EnsembleSampler(n_walkers, n_dims, chi2, args=[], backend=backend)

print("Problem defined")

## Running the MCMC method
print("Starting the MCMC method")

starting_state = state = sampler.run_mcmc(Initial_guess, 1000, progress = True) # burning the first steps
sampler.reset()

print("burning complete")

final_state = sampler.run_mcmc(starting_state, 50000, progress = True)

print("MCMC analysis done")

## Plotting
print("Starting to plot the results")

Coordinates_limits = [[60, 75], [-1.40, -0.60], [0.920, 0.995], [0.64, 1.04], [0.250, 0.375]]
Expected_values_01 = X_d_planck[0, 0:5]
Expected_values = prior(Expected_values_01)
Parameters = ['H_{0}', 'w_{0}', 'n_{s}', '\sigma_{8}', '\Omega_{M}']

fig, axes = plt.subplots(nrows = 5, ncols =5, figsize = (15, 15))
plt.subplots_adjust(hspace = 0.3, wspace = 0.4)

samples = sampler.get_chain(flat=True)

for i in range(5):
    for j in range(5):
        subplot = axes[i][j]
        
        if (i < j):
            subplot.axis('off')
        
        elif (i > j):
            if (i == 4):
                subplot.set_xlabel("$" + Parameters[j] + "$")
            if (j == 0):
                subplot.set_ylabel("$" + Parameters[i] + "$")
            subplot.hist2d(x = samples[:, j], y = samples[:, i], bins = 100, cmap = "Greys")
            subplot.set_xlim(left = Coordinates_limits[j][0], right = Coordinates_limits[j][1])
            subplot.set_ylim(bottom = Coordinates_limits[i][0], top = Coordinates_limits[i][1])
            subplot.axvline(x = Expected_values[j], color = 'black', linestyle = '--')
            subplot.axhline(y = Expected_values[i], color = 'black', linestyle = '--')
        
        else:
            subplot.set_xlim(left = Coordinates_limits[j][0], right = Coordinates_limits[j][1])
            if (i == 4):
                subplot.set_xlabel("$" + Parameters[j] + "$")
            subplot.set_ylabel("$p(" + Parameters[i] + ")$")
            subplot.hist(samples[:, i], 100, color = "black", histtype = "step")
            subplot.axvline(x = Expected_values[j], color = 'black', linestyle = '--')
            subplot.set_yticklabels([])

plt.suptitle("Posterior distribution (Abacus)")

#my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')
my_path = os.path.abspath('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Figures')
my_file = 'Figure_12_EMCEE_5'
my_file = os.path.join(my_path, my_file)
plt.savefig(my_file)
plt.show()

print("Results saved and plotted")

## Plotting 2
import corner
Labels = ['$H_{0}$', '$w_{0}$', '$n_{s}$', '$\sigma_{8}$', '$\Omega_{M}$']
Expected_values_01 = X_d_planck[0, 0:5]
Truths = prior(Expected_values_01)

flat_samples = sampler.get_chain(discard = 0, thin = 2, flat=True)

corner.corner(flat_samples, labels = Labels, truths = Truths, plot_datapoints = False, fill_contours = True)
plt.suptitle("Posterior distribution (Abacus)")

#my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')
my_path = os.path.abspath('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Figures')
my_file = 'Figure_12_EMCEE_corner_5'
my_file = os.path.join(my_path, my_file)
plt.savefig(my_file)
plt.show()