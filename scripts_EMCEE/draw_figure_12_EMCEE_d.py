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

## Importing the whole Abacus data
print("starting to load the data")

#target = "/home/astro/magnan/Repository_Stage_3A/Full_MST_stats_Abacus/MST_stats_Catalogue_"
target = "C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Full_MST_stats_Abacus/MST_stats_Catalogue_"

dict = {'X_d' : [], 'Y_d' : [], 'Y_d_std' : []}

for i in range(61):    
    X_d_a = np.loadtxt(str(target) + str(i) + "_X_d")
    Y_d_a = np.loadtxt(str(target) + str(i) + "_Y_d")
    Y_d_std_a = np.loadtxt(str(target) + str(i) + "_Y_d_std")
    dict['X_d'].append(X_d_a)
    dict['Y_d'].append(Y_d_a)
    dict['Y_d_std'].append(Y_d_std_a)

print("data fully loaded")

## Importing the data
print("Connexion successfull")
print("starting to load the data")

#target = "/home/astro/magnan/Repository_Stage_3A/data_set_Abacus/data_set_Abacus"
target = 'C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/data_set_Abacus/data_set_Abacus_'

"""
d = 0->4
"""

n_points_per_simulation_complete = 36
n_simulations = 40
n_fiducial = 21

X_d = None
Y_d = None

for i in range(n_fiducial + n_simulations):
    X_data_new = np.loadtxt(fname = str(target) + str(i) + "_X_data") # numpy array with fields h0, w0, ns, sigma8, omegaM, d/l/b/s
    Y_data_new = np.loadtxt( str(target) + str(i) + "_Y_data") # numpy array with field Nd/l/b/s
    
    for j in range(n_points_per_simulation_complete): # there are infinite values because of the log normalisation
        Y_data_new[j] = max(Y_data_new[j], 0)
    
    if i == 0:
        X_d = X_data_new[0 : 4, 0 : 6]
        Y_d = Y_data_new[0 : 4]
    else:
        X_d = np.concatenate((X_data_new[0 : 4, 0:6], X_d))
        Y_d = np.concatenate((Y_data_new[0 : 4], Y_d))

X_d_planck = X_d[:(n_fiducial) * 4]

X_d_data = X_d[(n_fiducial) * 4:]
Y_d_data = Y_d[(n_fiducial) * 4:]
Y_d_data = np.reshape(Y_d_data, (n_simulations * 4, 1))

print("data loaded")

## Setting up the GPs
print("starting to define the Gps")

gp_d = GP.GP(X = X_d_data, Y = Y_d_data, n_points_per_simu = 4, Noise = None)

print("models defined")

## Optimising the hyperparameters - Gradient Descent
print("Starting to optimise the hyperparameters")

gp_d.optimize_model(optimizer = 'lbfgsb')

print("Hyperparameters optimised")

gp_d.print_model()

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
        for l in range(np.shape(dict['X_d'][k])[0]):
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
        for l in range(np.shape(dict['X_d'][k + n_simulations])[0]):
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
    X_d_predicted, Y_d_predicted, Cov_d = gp_d.compute_prediction(X_new)
    
    # giving the right shape to the predicted value
    Y_d_predicted = [Y_d_predicted[0]]
    
    # searching for the expected value
    X_d_predicted = X_d_predicted[0][:, 5]
    
    Y_d_expected = np.asarray([0 for i in range(np.shape(X_d_predicted)[0])])
    Y_d_std_expected = np.asarray([0 for i in range(np.shape(X_d_predicted)[0])])
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
        Y_d_std_expected[k] = Std_d_fidu[l_min]
    Y_d_expected = np.asarray(Y_d_expected)
    Y_d_std_expected = np.asarray(Y_d_std_expected) / (np.log(10) * Y_d_expected)
    Y_d_expected = np.log10(Y_d_expected)
    
    # Giving the right shape to the expected value
    Y_d_expected = [np.reshape(Y_d_expected, (4, 1))]
    
    # Defining the noises
    Noise_predicted_d = Cov_d
    Noise_expected_d = [Y_d_std_expected]
    
    # Computing the likelihood
    chi2_d = gp_d.likelihood_chi2(Y_observation = Y_d_expected, Noise_observation = Noise_expected_d, Y_model = Y_d_predicted, Noise_model = Noise_predicted_d)
    
    if (m.isnan(chi2_d)):
        print("chi2_d is NaN")
        print(X)
        return(- m.inf)
    
    # combining the 2 statistics
    chi2 = (4 * chi2_d + 5 * chi2_s) / (4 + 5)
    
    # returning the log-likelihood or chi_2
    return(-0.5 * chi2)

print("Likelihood defined")

## Defining the problem
print("Starting to define the problem")

n_dims = 5
n_walkers = 32

#my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')
my_path = os.path.abspath('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/EMCEE/')
my_file = 'Figure_12_d'
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
my_file = 'Figure_12_EMCEE_d'
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
my_file = 'Figure_12_EMCEE_corner_d'
my_file = os.path.join(my_path, my_file)
plt.savefig(my_file)
plt.show()