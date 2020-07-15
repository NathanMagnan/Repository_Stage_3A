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
import GP_tools_simple as GP
#os.chdir('/home/astro/magnan')
os.chdir('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A')

print("All imports successful")

## Choosing the statistics
stat = 'l'

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

X = None
Y = None
Y_std = None

for i in range(41):
    X_data_new = np.loadtxt(fname = str(target) + str(i) + "_X_data") # numpy array with fields h0, w0, ns, sigma8, omegaM, d/l/b/s
    Y_data_new = np.loadtxt( str(target) + str(i) + "_Y_data") # numpy array with field Nd/l/b/s
    Y_std_new = 5 * np.genfromtxt(str(target) + str(i) + "_Y_std", filling_values = 0) / np.log(10) # numpy array with field sigma Nd/l/b/s
    
    for j in range(n_points_per_simulation_complete): # there are infinite values because of the log normalisation
        Y_data_new[j] = max(Y_data_new[j], 0)
    
    if i == 0:
        if (stat == 'd'):
            X = X_data_new[0 : 4, 0 : 6]
            Y = Y_data_new[0 : 4]
            Y_std = Y_std_new[0 : 4]
        if (stat == 'l'):
            X = X_data_new[10 : 15, 0 : 6]
            Y = Y_data_new[10 : 15]
            Y_std = Y_std_new[10 : 15]
        if (stat == 'b'):
            X = X_data_new[19 : 24, 0 : 6]
            Y = Y_data_new[19 : 24]
            Y_std = Y_std_new[19 : 24]
        if (stat == 's'):
            X = X_data_new[28 : 33, 0 : 6]
            Y = Y_data_new[28 : 33]
            Y_std = Y_std_new[28 : 33]
    else:
        if (stat == 'd'):
            X = np.concatenate((X_data_new[0 : 4, 0:6], X))
            Y = np.concatenate((Y_data_new[0 : 4], Y))
            Y_std = np.concatenate((Y_std_new[0 : 4], Y_std))
        if (stat == 'l'):
            X = np.concatenate((X_data_new[10 : 15, 0:6], X))
            Y = np.concatenate((Y_data_new[10 : 15], Y))
            Y_std = np.concatenate((Y_std_new[10 : 15], Y_std))
        if (stat == 'b'):
            X = np.concatenate((X_data_new[19 : 24, 0:6], X))
            Y = np.concatenate((Y_data_new[19 : 24], Y))
            Y_std = np.concatenate((Y_std_new[19 : 24], Y_std))
        if (stat == 's'):
            X = np.concatenate((X_data_new[28 : 33, 0:6], X))
            Y = np.concatenate((Y_data_new[28 : 33], Y))
            Y_std = np.concatenate((Y_std_new[28 : 33], Y_std))

if (stat == 'd'):
    n = 4
else:
    n = 5

X_planck = X[(n_simulations) * n : (n_simulations + 1) * n]
Y_planck = Y[(n_simulations) * n : (n_simulations + 1) * n]
Y_std_planck = Y_std[(n_simulations) * n : (n_simulations + 1) * n]
Y_planck_expected = np.reshape(Y_planck, (n, 1))
Y_std_planck_expected = np.reshape(Y_std_planck, (n, 1))

X_data = X[0 : (n_simulations) * n]
Y_data = Y[0 : (n_simulations) * n]
Y_std_data = Y_std[0 : (n_simulations) * n]
Y_data = np.reshape(Y_data, (n_simulations * n, 1))
Y_std_data = np.reshape(Y_std_data, (n_simulations * n, 1))

print("data loaded")

    ## Loading the complete Abacus data for analysis
print("starting to load the data")

#target = "/home/astro/magnan/Repository_Stage_3A/Full_MST_stats_Abacus/MST_stats_Catalogue_"
target = "C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Full_MST_stats_Abacus/MST_stats_Catalogue_"

dict = {'X' : [], 'Y' : [], 'Y_std' : []}

for i in range(41):   
    if (stat == 'd'):
        X_a = np.loadtxt(str(target) + str(i) + "_X_d") / 6
        Y_a = np.log10(np.loadtxt(str(target) + str(i) + "_Y_d"))
        Y_std_a = np.loadtxt(str(target) + str(i) + "_Y_d_std") / (np.loadtxt(str(target) + str(i) + "_Y_d") * np.log(10))
        dict['X'].append(X_a)
        dict['Y'].append(Y_a)
        dict['Y_std'].append(Y_std_a)
    if (stat == 'l'):
        X_a = np.log10(np.loadtxt(str(target) + str(i) + "_X_l"))
        Y_a = np.log10(np.loadtxt(str(target) + str(i) + "_Y_l"))
        Y_std_a = np.loadtxt(str(target) + str(i) + "_Y_l_std") / (np.loadtxt(str(target) + str(i) + "_Y_l") * np.log(10))
        dict['X'].append(X_a)
        dict['Y'].append(Y_a)
        dict['Y_std'].append(Y_std_a)
    if (stat == 'b'):
        X_a = np.log10(np.loadtxt(str(target) + str(i) + "_X_b"))
        Y_a = np.log10(np.loadtxt(str(target) + str(i) + "_Y_b"))
        Y_std_a = np.loadtxt(str(target) + str(i) + "_Y_b_std") / (np.loadtxt(str(target) + str(i) + "_Y_b") * np.log(10))
        dict['X'].append(X_a)
        dict['Y'].append(Y_a)
        dict['Y_std'].append(Y_std_a)
    if (stat == 's'):
        X_a = np.loadtxt(str(target) + str(i) + "_X_s")
        Y_a = np.log10(np.loadtxt(str(target) + str(i) + "_Y_s"))
        Y_std_a = np.loadtxt(str(target) + str(i) + "_Y_s_std") / (np.loadtxt(str(target) + str(i) + "_Y_s") * np.log(10))
        dict['X'].append(X_a)
        dict['Y'].append(Y_a)
        dict['Y_std'].append(Y_std_a)

print("data fully loaded")

## Setting up the GPs
print("starting to define the Gp")

#gp = GP.GP(X = X_data, Y = Y_data, n_points_per_simu = n, Noise = Y_std_data)
gp = GP.GP(X = X_data, Y = Y_data, n_points_per_simu = n, Noise = None)

print("model defined")

## Optimising the hyperparameters - Gradient Descent
print("Starting to optimise the hyperparameters")

gp.optimize_model(optimizer = 'lbfgsb')
print("Hyperparameters optimised")

# gp.change_model_to_heteroscedatic(Noise = Y_std_data)
# print("model changed to heteroscedatic")

gp.print_model()

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
    
    # searching for the expected value
    X_predicted = X_predicted[0]
    Y_predicted = Y_predicted[0]
    Cov_predicted = np.sqrt(np.reshape(np.diag(Cov[0]), (n, 1)))
    
    Y_expected = []
    Noise_expected = []
    
    for k in range(n):
        min = 1
        l_min = 0
        
        x1 = X_predicted[:, 5][k]
        
        for l in range(len(dict['X'][40])):
            x2 = dict['X'][40][l]
            dist = abs(x1 - x2)
            if (dist < min):
                l_min = l
                min = dist
        
        Y_expected.append(dict['Y'][40][l_min])
        Noise_expected.append(dict['Y_std'][40][l_min])

    Y_expected = np.reshape(Y_expected, (n, 1))
    Noise_expected = np.reshape(Y_expected, (n, 1))
    
    # Defining the noises
    Noise_predicted = Cov
    Noise_expected = [np.diagonal(Noise_expected)]
    
    # Computing the likelihood
    ms = gp.likelihood_ms(Y_model = Y_predicted, Noise_model = Noise_predicted, Y_observation = Y_expected, Noise_observation = Noise_expected)
    
    if (m.isnan(ms)):
        print(X)
        return(- m.inf)
    
    # returning the log-likelihood or chi_2
    #return(-0.5 * ms)
    return(- 500 * ms)

print("Likelihood defined")

## Test
# print("starting to test the likelihood function")
# 
# Basis_01 = X_planck[0, 0:5]
# 
# X = [i / 100 for i in range(100)]
# Y = []
# for x in X:
#     xp = Basis_01.copy()
#     xp[4] = x
#     y = chi2(prior(xp))
#     Y.append(y)
# 
# plt.plot(X, Y, 'k')
# plt.show()
# 
# print("likelihood function tested")
""" It seems that the likelihood function works absolutely fine, its maximum are in the right place for every statistics, but it gives large variation only for the l stat """

## Defining the problem
print("Starting to define the problem")

n_dims = 5
n_walkers = 32

#my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')
my_path = os.path.abspath('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/EMCEE/')
my_file = 'Figure_12_' + stat + '_3'
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
Expected_values_01 = X_planck[0, 0:5]
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

plt.suptitle("Posterior distribution (Abacus " + stat + ")")

#my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')
my_path = os.path.abspath('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Figures')
my_file = 'Figure_12_EMCEE_' + stat + '-500ms'
my_file = os.path.join(my_path, my_file)
plt.savefig(my_file)
plt.show()

print("Results saved and plotted")

## Plotting 2
import corner
Labels = ['$H_{0}$', '$w_{0}$', '$n_{s}$', '$\sigma_{8}$', '$\Omega_{M}$']
Expected_values_01 = X_planck[0, 0:5]
Truths = prior(Expected_values_01)

flat_samples = sampler.get_chain(discard = 0, thin = 2, flat=True)

corner.corner(flat_samples, labels = Labels, truths = Truths, plot_datapoints = False, fill_contours = True)
plt.suptitle("Posterior distribution (Abacus)")

#my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')
my_path = os.path.abspath('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Figures')
my_file = 'Figure_12_EMCEE_corner_' + stat + '-500ms'
my_file = os.path.join(my_path, my_file)
plt.savefig(my_file)
plt.show()