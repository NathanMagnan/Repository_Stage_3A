## Imports
import numpy as np
import math as m
import pymultinest as pmn
import json as json
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)

import sys
import os

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
	s = 1 / (2 * m.pi * 15**2)**0.5 * np.exp(- (cube[0] - 68)**2 / 1.5**2)
	s *= 1 / (2 * m.pi * 0.8**2)**0.5 * np.exp(- (cube[1] + 1)**2 / 0.08**2)
	s *= 1 / (2 * m.pi * 0.08**2)**0.5 * np.exp(- (cube[2] - 0.960)**2 / 0.008**2)
	s *= 1 / (2 * m.pi * 0.4**2)**0.5 * np.exp(- (cube[3] - 0.80)**2 / 0.04**2)
	s *= 1 / (2 * m.pi * 0.1**2)**0.5 * np.exp(- (cube[4] - 0.310)**2 / 0.01**2)
	return np.log(s)

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
target = 'Test'
target = os.path.join(my_path, target)

result = pmn.solve(LogLikelihood = loglikelihoodTest, Prior = prior, n_dims = n_params, resume = False, outputfiles_basename = target, sampling_efficiency = 1, evidence_tolerance = 10**(-0), n_live_points = 500)

json.dump(parameters, open(target + 'params.json', 'w')) # save parameter names

print("MCMC analysis done")

## Plotting the results
print("Starting to plot the results")

os.system('python3' + ' ' + '/home/astro/magnan/PyMultiNest/multinest_marginals_fancy.py' + ' ' + target)
