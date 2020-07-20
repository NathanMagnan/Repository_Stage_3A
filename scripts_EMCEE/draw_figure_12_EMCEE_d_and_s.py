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

dict = {'X_d' : [], 'Y_d' : [], 'Y_d_std' : [], 'X_l' : [], 'Y_l' : [], 'Y_l_std' : [], 'X_b' : [], 'Y_b' : [], 'Y_b_std' : [], 'X_s' : [], 'Y_s' : [], 'Y_s_std' : []}

for i in range(61):    
    X_d_a = np.loadtxt(str(target) + str(i) + "_X_d")
    Y_d_a = np.loadtxt(str(target) + str(i) + "_Y_d")
    Y_d_std_a = np.loadtxt(str(target) + str(i) + "_Y_d_std")
    dict['X_d'].append(X_d_a)
    dict['Y_d'].append(Y_d_a)
    dict['Y_d_std'].append(Y_d_std_a)
    
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
s = 28->33
"""

n_points_per_simulation_complete = 36
n_simulations = 40
n_fiducial = 21

X_d = None
Y_d = None
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
        X_s = X_data_new[28 : 33, 0 : 6]
        Y_s = Y_data_new[28 : 33]
    else:
        X_d = np.concatenate((X_data_new[0 : 4, 0:6], X_d))
        Y_d = np.concatenate((Y_data_new[0 : 4], Y_d))
        X_s = np.concatenate((X_data_new[28 : 33, 0:6], X_s))
        Y_s = np.concatenate((Y_data_new[28 : 33], Y_s))

X_d_planck = X_d[:(n_fiducial) * 4]
X_s_planck = X_s[:(n_fiducial) * 5]

X_d_data = X_d[(n_fiducial) * 4:]
Y_d_data = Y_d[(n_fiducial) * 4:]
X_s_data = X_s[(n_fiducial) * 5:]
Y_s_data = Y_s[(n_fiducial) * 5:]
Y_d_data = np.reshape(Y_d_data, (n_simulations * 4, 1))
Y_s_data = np.reshape(Y_s_data, (n_simulations * 5, 1))

print("data loaded")

## Setting up the GPs
print("starting to define the Gps")

gp_d = GP.GP(X = X_d_data, Y = Y_d_data, n_points_per_simu = 4, Noise = None)
gp_s = GP.GP(X = X_s_data, Y = Y_s_data, n_points_per_simu = 5, Noise = None)

print("models defined")

## Optimising the hyperparameters - Gradient Descent
print("Starting to optimise the hyperparameters")

gp_d.optimize_model(optimizer = 'lbfgsb')
gp_s.optimize_model(optimizer = 'lbfgsb')

print("Hyperparameters optimised")

gp_d.print_model()
gp_s.print_model()

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
    X_d_predicted, Y_d_predicted, Cov_d = gp_d.compute_prediction(X_new)
    X_s_predicted, Y_s_predicted, Cov_s = gp_s.compute_prediction(X_new)
    
    # giving the right shape to the predicted value
    Y_d_predicted = [Y_d_predicted[0]]
    Y_s_predicted = [Y_s_predicted[0]]
    
    # searching for the expected value and the noise
    X_d_predicted = X_d_predicted[0][:, 5]
    
    Fiducial_data = np.asarray([[0 for j in range(n_fiducial)] for i in range(np.shape(X_d_predicted)[0])])
    for i in range(np.shape(X_d_predicted)[0]):
        x1 = X_d_predicted[i]
        for j in range(n_fiducial):
            l_min = 0
            min = 10
            for l in range(np.shape(dict['X_d'][n_simulations + j])[0]):
                x2 = dict['X_d'][n_simulations + j][l] / 6
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            Fiducial_data[i][j] = dict['Y_d'][n_simulations + j][l_min]
    
    Fiducial_data = Fiducial_data
    Y_d_expected = np.mean(Fiducial_data, axis = 1)
    Noise_d_expected = np.cov(Fiducial_data)
    Y_d_std_expected = np.sqrt(np.diagonal(Noise_d_expected))
    
    X_s_predicted = X_s_predicted[0][:, 5]
    
    Fiducial_data = np.asarray([[0 for j in range(n_fiducial)] for i in range(np.shape(X_s_predicted)[0])])
    for i in range(np.shape(X_s_predicted)[0]):
        x1 = X_s_predicted[i]
        for j in range(n_fiducial):
            l_min = 0
            min = 10
            for l in range(np.shape(dict['X_s'][n_simulations + j])[0]):
                x2 = dict['X_s'][n_simulations + j][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            try:
                Fiducial_data[i][j] = (dict['Y_s'][n_simulations + j][l_min - 1] + dict['Y_s'][n_simulations + j][l_min] + dict['Y_s'][n_simulations + j][l_min + 1]) / 3
            except:
                Fiducial_data[i][j] = dict['Y_s'][n_simulations + j][l_min]
    
    Fiducial_data = Fiducial_data
    Y_s_expected = np.mean(Fiducial_data, axis = 1)
    Noise_s_expected = np.cov(Fiducial_data)
    Y_s_std_expected = np.sqrt(np.diagonal(Noise_s_expected))
    
    # correction to account for the logarithm transformation
    for i in range(np.shape(X_d_predicted)[0]):
        Y_d_std_expected[i] = Y_d_std_expected[i] / (np.log(10) * Y_d_expected[i])
    
    for i in range(np.shape(X_d_predicted)[0]):
        for j in range(np.shape(X_d_predicted)[0]):
            Noise_d_expected[i, j] = Noise_d_expected[i, j] / (np.log(10)**2 * Y_d_expected[i] * Y_d_expected[j])
    
    for i in range(np.shape(X_d_predicted)[0]):
        Y_d_expected[i] = np.log10(Y_d_expected[i])
    
    for i in range(np.shape(X_s_predicted)[0]):
        Y_s_std_expected[i] = Y_s_std_expected[i] / (np.log(10) * Y_s_expected[i])
    
    for i in range(np.shape(X_s_predicted)[0]):
        for j in range(np.shape(X_s_predicted)[0]):
            Noise_s_expected[i, j] = Noise_s_expected[i, j] / (np.log(10)**2 * Y_s_expected[i] * Y_s_expected[j])
    
    for i in range(np.shape(X_s_predicted)[0]):
        Y_s_expected[i] = np.log10(Y_s_expected[i])
    
    # Giving the right shape to the expected value
    Y_d_expected = [np.reshape(Y_d_expected, (4, 1))]
    Y_s_expected = [np.reshape(Y_s_expected, (5, 1))]
    
    # Defining the noises
    Noise_predicted_d = Cov_d
    Noise_expected_d_diagonal = [Y_d_std_expected]
    Noise_expected_d_matrix = [Noise_d_expected]
    
    Noise_predicted_s = Cov_s
    Noise_expected_s_diagonal = [Y_s_std_expected]
    Noise_expected_s_matrix = [Noise_s_expected]
    
    # Computing the likelihood
    #chi2_d = gp_d.likelihood_chi2_bd(Y_observation = Y_d_expected, Noise_observation = Noise_expected_d_diagonal, Y_model = Y_d_predicted, Noise_model = Noise_predicted_d)
    # chi2_d = gp_d.likelihood_chi2_ad(Y_observation = Y_d_expected, Noise_observation = Noise_expected_d_diagonal, Y_model = Y_d_predicted, Noise_model = Noise_predicted_d, N = 21)
    chi2_d = gp_d.likelihood_chi2_bm(Y_observation = Y_d_expected, Noise_observation = Noise_expected_d_matrix, Y_model = Y_d_predicted, Noise_model = Noise_predicted_d) 
    
    # chi2_s = gp_s.likelihood_chi2_bd(Y_observation = Y_s_expected, Noise_observation = Noise_expected_s_diagonal, Y_model = Y_s_predicted, Noise_model = Noise_predicted_s)
    # chi2_s = gp_s.likelihood_chi2_ad(Y_observation = Y_s_expected, Noise_observation = Noise_expected_s_diagonal, Y_model = Y_s_predicted, Noise_model = Noise_predicted_s, N = 21)
    chi2_s = gp_s.likelihood_chi2_bm(Y_observation = Y_s_expected, Noise_observation = Noise_expected_s_matrix, Y_model = Y_s_predicted, Noise_model = Noise_predicted_s) 
    
    if (m.isnan(chi2_d)):
        print("chi2_d is NaN")
        print(X)
        return(- m.inf)
    
    if (m.isnan(chi2_s)):
        print("chi2_s is Nan")
        print(X)
        return(- m.inf)
    
    # combining the 2 statistics
    chi2 = chi2_d + chi2_s
    
    # returning the log-likelihood or chi_2
    return(-0.5 * chi2)

print("Likelihood defined")

## Defining the problem
print("Starting to define the problem")

n_dims = 5
n_walkers = 32

#my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')
my_path = os.path.abspath('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/EMCEE/')
my_file = 'Figure_12_d_and_s'
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
my_file = 'Figure_12_EMCEE_bm_ds'
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
my_file = 'Figure_12_EMCEE_corner_bm_ds'
my_file = os.path.join(my_path, my_file)
plt.savefig(my_file)
plt.show()