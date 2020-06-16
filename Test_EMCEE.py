## Imports
import numpy as np
import math as m
import emcee
import sys
import os
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)

print("All imports successful !")

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

## Definging the log-likelihood
print("Starting to define the log-likelihood")

def loglikelihoodTest(X):
    s = 1 / (2 * m.pi * 15**2)**0.5 * np.exp(- (X[0] - 68)**2 / 15**2)
    s *= 1 / (2 * m.pi * 0.8**2)**0.5 * np.exp(- (X[1] + 1)**2 / 0.8**2)
    s *= 1 / (2 * m.pi * 0.08**2)**0.5 * np.exp(- (X[2] - 0.960)**2 / 0.08**2)
    s *= 1 / (2 * m.pi * 0.4**2)**0.5 * np.exp(- (X[3] - 0.80)**2 / 0.4**2)
    s *= 1 / (2 * m.pi * 0.1**2)**0.5 * np.exp(- (X[4] - 0.310)**2 / 0.1**2)
    return np.log(s)

print("Log-likelihood defined")

## Defining the problem
print("Starting to define the problem")

n_dims = 5
n_walkers = 32

A = np.random.rand(n_walkers, n_dims)
Initial_guess = np.asarray([prior(A[i]) for i in range(n_walkers)])

sampler = emcee.EnsembleSampler(n_walkers, n_dims, loglikelihoodTest, args=[])

print("Problem defined")

## Running the MCMC method
print("Starting the MCMC method")

starting_state = state = sampler.run_mcmc(Initial_guess, 1000) # burning the first steps
sampler.reset()

final_state = sampler.run_mcmc(starting_state, 50000)

print("MCMC analysis done")

## Plotting
print("Starting to plot the results")

Coordinates_limits = [[60, 75], [-1.40, -0.60], [0.920, 0.995], [0.64, 1.04], [0.250, 0.375]]
Expected_values = [68, -1, 0.960, 0.80, 0.310]
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

plt.suptitle("Test of the EMCEE package")

#my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')
my_path = os.path.abspath('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Figures')
my_file = 'Test_EMCEE'
my_file = os.path.join(my_path, my_file)
plt.savefig(my_file)
plt.show()

print("Results saved and plotted")