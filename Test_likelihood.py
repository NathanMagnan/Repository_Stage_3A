## Imports
import numpy as np
import math as m
import GPy as GPy
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)

import sys
import os
sys.path.append('/home/astro/magnan/Repository_Stage_3A')
#sys.path.append('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A')
import GP_tools as GP
os.chdir('/home/astro/magnan')
#os.chdir('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A')

print("All imports successful")

## Importing the data
print("Connexion successfull")
print("starting to load the data")

target = "/home/astro/magnan/Repository_Stage_3A/data_set_Abacus/data_set_Abacus"
#target = 'C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/data_set_Abacus/data_set_Abacus'

"""
d = 0->5
l = 9->15
b = 17->25
s = 26->35
"""

n_points_per_simulation_complete = 36
n_simulations = 40

X_data_complete = np.loadtxt(str(target) + "_X_data_all") # numpy array with fields h0, w0, ns, sigma8, omegaM, ds -- 6 point per simu
Y_data_complete = np.loadtxt(fname = str(target) + "_Y_data_all") # numpy array with field Nd

for i in range(n_simulations * n_points_per_simulation_complete): # there are infinite values because of the log normalisation
    Y_data_complete[i] = max(Y_data_complete[i], 0)

X_d = None
Y_d = None
X_l = None
Y_l = None
X_b = None
Y_b = None
X_s = None
Y_s = None
for i in range(n_simulations + 1):
    if i == 0:
        X_d = X_data_complete[0 : 5, 0 : 6]
        Y_d = Y_data_complete[0 : 5]
        X_l = X_data_complete[10 : 15, 0 : 6]
        Y_l = Y_data_complete[10 : 15]
        X_b = X_data_complete[19 : 24, 0 : 6]
        Y_b = Y_data_complete[19 : 24]
        X_s = X_data_complete[28 : 33, 0 : 6]
        Y_s = Y_data_complete[28 : 33]
    else:
        X_d = np.concatenate((X_data_complete[i * n_points_per_simulation_complete + 0 : i * n_points_per_simulation_complete + 5, 0:6], X_d))
        Y_d = np.concatenate((Y_data_complete[i * n_points_per_simulation_complete + 0 : i * n_points_per_simulation_complete + 5], Y_d))
        X_l = np.concatenate((X_data_complete[i * n_points_per_simulation_complete + 10 : i * n_points_per_simulation_complete + 15, 0:6], X_l))
        Y_l = np.concatenate((Y_data_complete[i * n_points_per_simulation_complete + 10 : i * n_points_per_simulation_complete + 15], Y_l))
        X_b = np.concatenate((X_data_complete[i * n_points_per_simulation_complete + 19 : i * n_points_per_simulation_complete + 24, 0:6], X_b))
        Y_b = np.concatenate((Y_data_complete[i * n_points_per_simulation_complete + 19 : i * n_points_per_simulation_complete + 24], Y_b))
        X_s = np.concatenate((X_data_complete[i * n_points_per_simulation_complete + 28 : i * n_points_per_simulation_complete + 33, 0:6], X_s))
        Y_s = np.concatenate((Y_data_complete[i * n_points_per_simulation_complete + 28 : i * n_points_per_simulation_complete + 33], Y_s))
noise_data = 0

X_d_planck = X_d[(n_simulations) * 5 : (n_simulations + 1) * 5]
Y_d_planck = Y_d[(n_simulations) * 5 : (n_simulations + 1) * 5]
X_l_planck = X_l[(n_simulations) * 5 : (n_simulations + 1) * 5]
Y_l_planck = Y_l[(n_simulations) * 5 : (n_simulations + 1) * 5]
X_b_planck = X_b[(n_simulations) * 5 : (n_simulations + 1) * 5]
Y_b_planck = Y_b[(n_simulations) * 5 : (n_simulations + 1) * 5]
X_s_planck = X_s[(n_simulations) * 5 : (n_simulations + 1) * 5]
Y_s_planck = Y_s[(n_simulations) * 5 : (n_simulations + 1) * 5]
Y_d_planck_expected = np.reshape(Y_d_planck, (5, 1))
Y_b_planck_expected = np.reshape(Y_b_planck, (5, 1))
Y_l_planck_expected = np.reshape(Y_l_planck, (5, 1))
Y_s_planck_expected = np.reshape(Y_s_planck, (5, 1))

print(X_d_planck[0])

X_d_data = X_d[0 : (n_simulations) * 5]
Y_d_data = Y_d[0 : (n_simulations) * 5]
X_l_data = X_l[0 : (n_simulations) * 5]
Y_l_data = Y_l[0 : (n_simulations) * 5]
X_b_data = X_b[0 : (n_simulations) * 5]
Y_b_data = Y_b[0 : (n_simulations) * 5]
X_s_data = X_s[0 : (n_simulations) * 5]
Y_s_data = Y_s[0 : (n_simulations) * 5]
Y_d_data = np.reshape(Y_d_data, (n_simulations * 5, 1))
Y_b_data = np.reshape(Y_b_data, (n_simulations * 5, 1))
Y_l_data = np.reshape(Y_l_data, (n_simulations * 5, 1))
Y_s_data = np.reshape(Y_s_data, (n_simulations * 5, 1))

print("data loaded")

## Loading the complete Abacus data for analysis
print("starting to load the data")

target = "/home/astro/magnan/Repository_Stage_3A/Full_MST_stats_Abacus/MST_stats_Catalogue_"
#target = "C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Full_MST_stats_Abacus/MST_stats_Catalogue_"

dict = {'X_d' : [], 'Y_d' : [], 'X_l' : [], 'Y_l' : [], 'X_b' : [], 'Y_b' : [], 'X_s' : [], 'Y_s' : []}

for i in range(41):    
    X_d_a = np.loadtxt(str(target) + str(i) + "_X_d")
    Y_d_a = np.loadtxt(str(target) + str(i) + "_Y_d")
    dict['X_d'].append(X_d_a)
    dict['Y_d'].append(Y_d_a)
    
    X_l_a = np.loadtxt(str(target) + str(i) + "_X_l")
    Y_l_a = np.loadtxt(str(target) + str(i) + "_Y_l")
    dict['X_l'].append(X_l_a)
    dict['Y_l'].append(Y_l_a)
    
    X_b_a = np.loadtxt(str(target) + str(i) + "_X_b")
    Y_b_a = np.loadtxt(str(target) + str(i) + "_Y_b")
    dict['X_b'].append(X_b_a)
    dict['Y_b'].append(Y_b_a)
    
    X_s_a = np.loadtxt(str(target) + str(i) + "_X_s")
    Y_s_a = np.loadtxt(str(target) + str(i) + "_Y_s")
    dict['X_s'].append(X_s_a)
    dict['Y_s'].append(Y_s_a)

print("data fully loaded")

## Setting up the GPs
print("starting to define the Gps")

gp = GP.GP(X = [X_d_data, X_l_data, X_b_data, X_s_data], Y = [Y_d_data, Y_l_data, Y_b_data, Y_s_data], N_points_per_simu = [5, 5, 5, 5], Noise = [None, None, None, None], type_kernel = "Separated")

print("models defined")

## Optimising the hyperparameters - Gradient Descent
print("Starting to optimise the hyperparameters")

gp.optimize_models(optimizer = 'lbfgsb')

print("Hyperparameters optimised")

gp.print_models()

## Defining the prior
print("starting to define the prior")

def prior(cube, ndims, nparams):
    cube[0] = cube[0] * (75 - 60) + 60
    cube[1] = cube[1] * ((-0.60) - (-1.40)) + (-1.40)
    cube[2] = cube[2] * (0.995 - 0.920) + 0.920
    cube[3] = cube[3] * (1.04 - 0.64) + 0.64
    cube[4] = cube[4] * (0.375 - 0.250) + 0.250
    return cube

print("prior defined")

## Defining the log-likelihood
print("Starting to define the log-likelihood")

def loglikelihood(cube,ndims, nparams):
    # Reading the parameters
    h0 = (cube[0] - 60) / (75 - 60)
    w0 = (cube[1] - (-1.40)) / ((-0.60) - (-1.40))
    ns = (cube[2] - 0.920) / (0.995 - 0.920)
    sigma8 = (cube[3] - 0.64) / (1.04 - 0.64)
    omegaM = (cube[4] - 0.250) / (0.375 - 0.250)
    X_new = np.asarray([h0, w0, ns, sigma8, omegaM])
    print(X_new)
    
    # Making the prediction
    X_new = np.reshape(X_new, (1, 5))
    X_predicted, Y_predicted, Cov = gp.compute_prediction(X_new)
    
    # searching for the expected value
    X_d_predicted = X_predicted[0][0]
    X_l_predicted = X_predicted[0][1]
    X_b_predicted = X_predicted[0][2]
    X_s_predicted = X_predicted[0][3]
    
    Y_d_expected = []
    Y_l_expected = []
    Y_b_expected = []
    Y_s_expected = []
    
    for k in range(5):
        min_d = 1
        l_min_d = 0
        min_l = 1
        l_min_l = 0
        min_b = 1
        l_min_b = 0
        min_s = 1
        l_min_s = 0
        
        xd = X_d_predicted[:, 5][k]
        xl = X_l_predicted[:, 5][k]
        xb = X_b_predicted[:, 5][k]
        xs = X_s_predicted[:, 5][k]
        
        for l in range(len(dict['X_d'][40])):
            x = dict['X_d'][40][l] / 6
            dist = abs(x - xd)
            if (dist < min_d):
                l_min_d = l
                min_d = dist
        for l in range(len(dict['X_l'][40])):
            x = np.log10(dict['X_l'][40][l])
            dist = abs(x - xl)
            if (dist < min_l):
                l_min_l = l
                min_l = dist
        for l in range(len(dict['X_b'][40])):
            x = np.log10(dict['X_b'][40][l])
            dist = abs(x - xb)
            if (dist < min_b):
                l_min_b = l
                min_b = dist
        for l in range(len(dict['X_s'][40])):
            x = dict['X_s'][40][l]
            dist = abs(x - xs)
            if (dist < min_s):
                l_min_s = l
                min_s = dist
        
        Y_d_expected.append(np.log10(dict['Y_d'][40][l_min_d]))
        Y_l_expected.append(np.log10(dict['Y_l'][40][l_min_l]))
        Y_b_expected.append(np.log10(dict['Y_b'][40][l_min_b]))
        Y_s_expected.append(np.log10(dict['Y_s'][40][l_min_s]))

    Y_d_expected = np.reshape(Y_d_expected, (5, 1))
    Y_l_expected = np.reshape(Y_l_expected, (5, 1))
    Y_b_expected = np.reshape(Y_b_expected, (5, 1))
    Y_s_expected = np.reshape(Y_s_expected, (5, 1))
    Y_expected = []
    Y_expected.append([Y_d_expected, Y_l_expected, Y_b_expected, Y_s_expected])
    
    # Defining the noises
    Noise_predicted = Cov
    Noise_expected = [None]
    
    # Computing the likelihood
    chi_2 = gp.likelihood_chi2(Y_observation = Y_predicted, Noise_observation = Noise_predicted, Y_model = Y_expected, Noise_model = Noise_expected)
    
    # returning the log-likelihood or chi_2
    return(chi_2)

print("Likelihood defined")

## Plotting
print("starting to plot")

my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')
#my_path = os.path.abspath('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Figures')
my_file = 'test_likelihood_function'
my_file = os.path.join(my_path, my_file)

X = np.linspace(60, 75, 100)
Cube = [[x, 0.326 * ((-0.60) - (-1.40)) + (-1.40), 0.13338 * (0.995 - 0.920) + 0.920, 0.53456 * (1.04 - 0.64) + 0.64, 0.41348 * (0.375 - 0.250) + 0.250] for x in X]
Y = [loglikelihood(cube, 5, 5) for cube in Cube]

figure = plt.figure()
ax = figure.gca()

ax.set_title("$\chi_{2}$ as a function of $H_{0}$")
ax.set_xlabel("$H_{0}$")
ax.set_ylabel("$\chi_{2}$")

ax.plot(X, Y, 'k')
ax.axvline(x = 0.59666 * (75 - 60) + 60)

plt.savefig(my_file)
plt.show()

print("plot done")


