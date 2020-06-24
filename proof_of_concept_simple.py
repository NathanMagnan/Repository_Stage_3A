## Imports
import numpy as np
import GPy as GPy
import os
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)

import sys
import os
#sys.path.append('/home/astro/magnan/Repository_Stage_3A')
sys.path.append('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A')
import GP_tools_simple as GP
#os.chdir('/home/astro/magnan')
os.chdir('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A')

print("All imports successful")

## Choosing the statistics
stat = 'b'

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

## Setting up the GPs
print("starting to define the Gp")

gp = GP.GP(X = X_data, Y = Y_data, n_points_per_simu = n, Noise = Y_std_data)
#gp = GP.GP(X = X_data, Y = Y_data, n_points_per_simu = n, Noise = None)

print("model defined")

## Optimising the hyperparameters - Gradient Descent
print("Starting to optimise the hyperparameters")

gp.optimize_model(optimizer = 'lbfgsb')
print("Hyperparameters optimised")

# gp.change_model_to_heteroscedatic(Noise = Y_std_data)
# print("model changed to heteroscedatic")

gp.print_model()

## Making a prediction
print("Starting to make a prediction")

X_Planck = np.reshape(X_planck[0, 0:5], (1, 5))
X_planck_predicted, Y_planck_predicted, Cov = gp.compute_prediction(X_Planck)

X_planck_predicted = X_planck_predicted[0]
Y_planck_predicted = Y_planck_predicted[0]
Cov_predicted = np.sqrt(np.reshape(np.diag(Cov[0]), (n, 1)))

print("Prediction done")

rms = gp.test_rms(X_test = X_planck, Y_test =  Y_planck_expected)
print("RMS Planck : " + str(rms))

#Y_planck_expected_2 = np.reshape(np.array([4.02997508, 4.32685058, 4.4974139, 4.38601435, 3.52911742]), (5, 1))
chi2 = gp.test_chi2(X_test = X_planck, Y_test = Y_planck_expected, Noise_test = Y_std_planck_expected)
print("Chi 2 Planck : " + str(chi2))

## Loading the whole Abacus data for plotting
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

## Plot
print("Starting to plot the MST stats")

fig, axes = plt.subplots(nrows = 2, ncols = 1, figsize = (8, 8))
plt.subplots_adjust(hspace = 0.3, wspace = 0.4)

for i in range(2):
    subplot = axes[i]
    
    subplot.set_xlabel('$' + stat + '$')
    if ((stat == 'l') or (stat == 'b')):
        subplot.set_xscale('log')
    subplot.set_xlim(X_planck_predicted[:, 5][0], X_planck_predicted[:, 5][-1])
    
    Mean = 10**(dict['Y'][0])
    Std = np.asarray([0 for k in range(np.shape(dict['Y'][0])[0])])
    for k in range(1, 41):
        Mean_old = Mean.copy()
        Std_old = Std.copy()
        
        Mean_new = (k * Mean_old + 10**(dict['Y'][k])) / (k + 1)
        Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + 10**(2 * dict['Y'][k])) / (k + 1) - Mean_new**2)
        
        Mean = Mean_new.copy()
        Std = Std_new.copy()
    
    if (i == 0):
        subplot.set_ylabel('$N_{' + stat + '}$')
        subplot.set_yscale('log')
        
        subplot.fill_between(x = dict['X'][40], y1 = Mean - Std, y2 = Mean + Std, color = 'g', alpha = 0.2, label = 'Abacus range')
        subplot.plot(X_planck_predicted[:, 5], 10**Y_planck_predicted, 'g', label = "Prediction")
        subplot.plot(dict['X'][40], 10**(dict['Y'][40]), 'g--', label = "Expectation")
        subplot.errorbar(dict['X'][40], 10**(dict['Y'][40]), yerr = np.log(10) * dict['Y_std'][40] * 10**(dict['Y'][40]), fmt = 'none', ecolor = 'green')
        subplot.errorbar(X_planck_predicted[:, 5], 10**Y_planck_predicted, yerr = np.log(10) * Cov_predicted * 10**Y_planck_predicted, fmt = 'none', ecolor = 'green')
        
        subplot.legend()
        
    else:
        subplot.set_ylabel('$\Delta N_{' + stat + '} / <N_{' + stat + '}>$')
        
        M = []
        
        for k in range(n):
            min = 1
            l_min = 0
            
            x1 = X_planck_predicted[:, 5][k]
            
            for l in range(len(dict['X'][40])):
                x2 = dict['X'][40][l]
                dist = abs(x2 - x1)
                if (dist < min):
                    l_min = l
                    min = dist
        
            M.append(Mean[l_min])
        M = np.reshape(np.array(M), (n, 1))
        
        subplot.fill_between(x = dict['X'][40], y1 = - Std / Mean, y2 = Std / Mean, color = 'g', alpha = 0.2, label = "Abacus range")
        subplot.plot(X_planck_predicted[:, 5], (10**Y_planck_predicted - M) / M, 'g', label = "Prediction")
        subplot.plot(dict['X'][40], (10**(dict['Y'][40]) - Mean) / Mean, 'g--', label = "Expectation")
        subplot.errorbar(dict['X'][40], (10**(dict['Y'][40]) - Mean) / Mean, yerr = dict['Y_std'][40] * 10**(dict['Y'][40]) / Mean, fmt = 'none', ecolor = 'green')
        subplot.errorbar(X_planck_predicted[:, 5], (10**Y_planck_predicted - M) / M, yerr = Cov_predicted * 10**(Y_planck_predicted) / M, fmt = 'none', ecolor = 'green')
        
        subplot.legend()

plt.suptitle("Comparison between prediction and expectation - Separated Heteroscedatic GP on $" + stat + "$")
plt.show()

#my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')
my_path = os.path.abspath('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Figures')
my_file = 'Comparison_prediction_expectation_separated'
my_file = os.path.join(my_path, my_file)
#plt.savefig(my_file)

print("results plotted and saved")