## Imports
import numpy as np
import math as m
import GPy as GPy
import emcee
import sys
import os
import matplotlib

import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)

#sys.path.append('/home/astro/magnan/Repository_Stage_3A')
sys.path.append('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A')
import GP_tools_simple as GP
#os.chdir('/home/astro/magnan')
os.chdir('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A')

print("All imports successful")

## Importing the whole Abacus data
print("starting to load the data")

#target = "/home/astro/magnan/Repository_Stage_3A/Full_MST_stats_Abacus/MST_stats_Catalogue_"
target = "C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Full_MST_stats_Abacus/MST_stats_Catalogue_"

dict = {'X_l' : [], 'Y_l' : [], 'Y_l_std' : []}

for i in range(61):    
    X_l_a = np.loadtxt(str(target) + str(i) + "_X_l")
    Y_l_a = np.loadtxt(str(target) + str(i) + "_Y_l")
    Y_l_std_a = np.loadtxt(str(target) + str(i) + "_Y_l_std")
    dict['X_l'].append(X_l_a)
    dict['Y_l'].append(Y_l_a)
    dict['Y_l_std'].append(Y_l_std_a)

print("data fully loaded")

## Importing the data
print("Connexion successfull")
print("starting to load the data")

#target = "/home/astro/magnan/Repository_Stage_3A/data_set_Abacus/data_set_Abacus_"
target = 'C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/data_set_Abacus/data_set_Abacus_'

"""
l = 10->15
"""

n_points_per_simulation_complete = 36
n_simulations = 40
n_fiducial = 21

X_l = None
Y_l = None

for i in range(n_fiducial + n_simulations):
    X_data_new = np.loadtxt(fname = str(target) + str(i) + "_X_data") # numpy array with fields h0, w0, ns, sigma8, omegaM, d/l/b/s
    Y_data_new = np.loadtxt( str(target) + str(i) + "_Y_data") # numpy array with field Nd/l/b/s
    
    for j in range(n_points_per_simulation_complete): # there are infinite values because of the log normalisation
        Y_data_new[j] = max(Y_data_new[j], 0)
    
    if i == 0:
        X_l = X_data_new[10 : 15, 0 : 6]
        Y_l = Y_data_new[10 : 15]
    else:
        X_l = np.concatenate((X_data_new[10 : 15, 0:6], X_l))
        Y_l = np.concatenate((Y_data_new[10 : 15], Y_l))

X_l_planck = X_l[:(n_fiducial) * 5]

X_l_data = X_l[(n_fiducial) * 5:]
Y_l_data = Y_l[(n_fiducial) * 5:]
Y_l_data = np.reshape(Y_l_data, (n_simulations * 5, 1))

print("data loaded")

## Setting up the GPs
print("starting to define the Gps")

gp_l = GP.GP(X = X_l_data, Y = Y_l_data, n_points_per_simu = 5, Noise = None)

print("models defined")

## Optimising the hyperparameters - Gradient Descent
print("Starting to optimise the hyperparameters")

gp_l.optimize_model(optimizer = 'lbfgsb')

print("Hyperparameters optimised")

gp_l.print_model()

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

X_l_abacus = dict['X_l'][0]
Mean_l_abacus = np.asarray([0 for k in range(np.shape(dict['X_l'][0])[0])])
Std_l_abacus = np.asarray([0 for k in range(np.shape(dict['X_l'][0])[0])])
for k in range(n_simulations):
    New = []
    for x1 in X_l_abacus:
        min = 10
        l_min = 0
        for l in range(np.shape(dict['X_l'][k])[0]):
            x2 = dict['X_l'][k][l]
            if (abs(x1 - x2) < min):
                min = abs(x1 - x2)
                l_min = l
        
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
        for l in range(np.shape(dict['X_l'][k + n_simulations])[0]):
            x2 = dict['X_l'][k + n_simulations][l]
            if (abs(x1 - x2) < min):
                min = abs(x1 - x2)
                l_min = l
        
        New.append(dict['Y_l'][k + n_simulations][l_min])
    New = np.asarray(New)
        
    Mean_old = Mean_l_fidu.copy()
    Std_old = Std_l_fidu.copy()
    
    Mean_new = (k * Mean_old + New) / (k + 1)
    Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + New**2) / (k + 1) - Mean_new**2)
    
    Mean_l_fidu = Mean_new.copy()
    Std_l_fidu = Std_new.copy()

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
    X_l_predicted, Y_l_predicted, Cov_l = gp_l.compute_prediction(X_new)
    
    # giving the right shape to the predicted value
    Y_l_predicted = [Y_l_predicted[0]]
    
    # searching for the expected value
    X_l_predicted = X_l_predicted[0][:, 5]
    
    Y_l_expected = np.asarray([0 for i in range(np.shape(X_l_predicted)[0])])
    Y_l_std_expected = np.asarray([0 for i in range(np.shape(X_l_predicted)[0])])
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
        Y_l_std_expected[k] = Std_l_fidu[l_min]
    Y_l_expected = np.asarray(Y_l_expected)
    Y_l_std_expected = np.asarray(Y_l_std_expected) / (np.log(10) * Y_l_expected)
    Y_l_expected = np.log10(Y_l_expected)
    
    # Giving the right shape to the expected value
    Y_l_expected = [np.reshape(Y_l_expected, (5, 1))]
    
    # Defining the noises
    Noise_predicted_l = Cov_l
    Noise_expected_l = [Y_l_std_expected]
    
    # Computing the likelihood
    chi2_l = gp_l.likelihood_chi2(Y_observation = Y_l_expected, Noise_observation = Noise_expected_l, Y_model = Y_l_predicted, Noise_model = Noise_predicted_l)
    
    if (m.isnan(chi2_l)):
        print("chi2_l is NaN")
        print(X)
        return(- m.inf)
    
    # combining the 2 statistics
    chi2 = chi2_l
    
    # returning the log-likelihood or chi_2
    return(-0.5 * chi2)

print("Likelihood defined")

## Defining the problem
print("Starting to define the problem")

n_dims = 5
n_walkers = 32

#my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/EMCEE/')
my_path = os.path.abspath('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/EMCEE/')
my_file = 'Figure_12_l'
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
Expected_values_01 = X_l_planck[0, 0:5]
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
my_file = 'Figure_12_EMCEE_l'
my_file = os.path.join(my_path, my_file)
plt.savefig(my_file)
plt.show()

print("Results saved and plotted")

## Plotting 2
import corner
Labels = ['$H_{0}$', '$w_{0}$', '$n_{s}$', '$\sigma_{8}$', '$\Omega_{M}$']
Expected_values_01 = X_l_planck[0, 0:5]
Truths = prior(Expected_values_01)

flat_samples = sampler.get_chain(discard = 0, thin = 2, flat=True)

corner.corner(flat_samples, labels = Labels, truths = Truths, plot_datapoints = False, fill_contours = True)
plt.suptitle("Posterior distribution (Abacus)")

#my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')
my_path = os.path.abspath('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Figures')
my_file = 'Figure_12_EMCEE_corner_l'
my_file = os.path.join(my_path, my_file)
plt.savefig(my_file)
plt.show()