## Imports
import numpy as np
import math as m
import GPy as GPy
import pymultinest as pmn
import json as json
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

def prior(cube):
    cube[0] = cube[0] * (75 - 60) + 60
    cube[1] = cube[1] * ((-0.60) - (-1.40)) + (-1.40)
    cube[2] = cube[2] * (0.995 - 0.920) + 0.920
    cube[3] = cube[3] * (1.04 - 0.64) + 0.64
    cube[4] = cube[4] * (0.375 - 0.250) + 0.250
    return cube

print("prior defined")

## Defining the log-likelihood
print("Starting to define the log-likelihood")

def loglikelihoodTest(cube):
	s = 1 / (2 * m.pi * 0.15**2)**0.5 * np.exp(- (cube[0] - 68)**2 / 0.15**2)
	s += 1 / (2 * m.pi * 0.08**2)**0.5 * np.exp(- (cube[1] + 1)**2 / 0.08**2)
	s += 1 / (2 * m.pi * 0.008**2)**0.5 * np.exp(- (cube[2] - 0.960)**2 / 0.008**2)
	s += 1 / (2 * m.pi * 0.04**2)**0.5 * np.exp(- (cube[3] - 0.80)**2 / 0.04**2)
	s += 1 / (2 * m.pi * 0.01**2)**0.5 * np.exp(- (cube[4] - 0.310)**2 / 0.01**2)
	return np.log(s)

def loglikelihood(cube):
    # Making the prediction
    X_new = np.reshape(cube, (1, 5))
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
    Noise_observation = [None]
    
    # Computing the likelihood
    likelihood = gp.likelihood_chi2(Y_observation = Y_predicted, Noise_observation = Noise_predicted, Y_model = Y_expected, Noise_model = Noise_observation)
    
    # returning the log-likelihood
    return np.log(likelihood)

print("Likelihood defined")

## Defining the problem
print("Starting to define the problem")

parameters = ['$H_{0}$', '$w_{0}$', '$n_{s}$', '$\sigma_{8}$', '$\Omega_{M}$']
n_dims = 5
n_params = 5

print("Problem defined")

## Running the MCMC method
print("Starting the MCMC method")

my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/MCMC')
#my_path = os.path.abspath('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/MCMC')
target = 'Abacus_chi2'
target = os.path.join(my_path, target)

result = pmn.solve(LogLikelihood = loglikelihoodTest, Prior = prior, n_dims = n_params, resume = False, outputfiles_basename = target, sampling_efficiency = 1, evidence_tolerance = 0.0005)

json.dump(parameters, open(target + 'params.json', 'w')) # save parameter names

print("MCMC analysis done")

## Plotting the results
print("Starting to plot the results")

os.system('python3' + ' ' + '/home/astro/magnan/PyMultiNest/multinest_marginals_fancy.py' + ' ' + target)

#a = pmn.Analyzer(n_params = n_params, outputfiles_basename = target)
#s = a.get_stats()
#
#json.dump(s, open(target + 'stats.json', 'w'), indent=4)
#
#data = a.get_data()
#i = data[:,1].argsort()[::-1]
#samples = data[i,2:]
#weights = data[i,0]
#loglike = data[i,1]
#Z = s['global evidence']
#logvol = np.log(weights) + 0.5 * loglike + Z
#logvol = logvol - logvol.max()
#results = dict(samples=samples, weights=weights, logvol=logvol)
#
#marginals.multinest_marginal_fancy.traceplots(results = result, labels = parameters, show_titles = True)
#plt.show()
#plt.close()
#marginals.multinest_marginal_fancy.cornerplot(results = result, labels = parameters, show_titles = True)
#plt.show()
#plt.close()
#marginals.multinest_marginal_fancy.cornerpoints(results = result, labels = parameters, show_titles = True)
#plt.show()
#plt.close()

print("Results plotted")