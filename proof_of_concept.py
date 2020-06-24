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
import GP_tools as GP
#os.chdir('/home/astro/magnan')
os.chdir('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A')

print("All imports successful")

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

X_d = None
Y_d = None
Y_d_std = None
X_l = None
Y_l = None
Y_l_std = None
X_b = None
Y_b = None
Y_b_std = None
X_s = None
Y_s = None
Y_s_std = None

for i in range(41):
    X_data_new = np.loadtxt(fname = str(target) + str(i) + "_X_data") # numpy array with fields h0, w0, ns, sigma8, omegaM, d/l/b/s
    Y_data_new = np.loadtxt( str(target) + str(i) + "_Y_data") # numpy array with field Nd/l/b/s
    Y_std_new = 5 * np.genfromtxt(str(target) + str(i) + "_Y_std", filling_values = 0) / np.log(10) # numpy array with field sigma Nd/l/b/s
    
    for j in range(n_points_per_simulation_complete): # there are infinite values because of the log normalisation
        Y_data_new[j] = max(Y_data_new[j], 0)
    
    if i == 0:
        X_d = X_data_new[0 : 4, 0 : 6]
        Y_d = Y_data_new[0 : 4]
        Y_d_std = Y_std_new[0 : 4]
        X_l = X_data_new[10 : 15, 0 : 6]
        Y_l = Y_data_new[10 : 15]
        Y_l_std = Y_std_new[10 : 15]
        X_b = X_data_new[19 : 24, 0 : 6]
        Y_b = Y_data_new[19 : 24]
        Y_b_std = Y_std_new[19 : 24]
        X_s = X_data_new[28 : 33, 0 : 6]
        Y_s = Y_data_new[28 : 33]
        Y_s_std = Y_std_new[28 : 33]
    else:
        X_d = np.concatenate((X_data_new[0 : 4, 0:6], X_d))
        Y_d = np.concatenate((Y_data_new[0 : 4], Y_d))
        Y_d_std = np.concatenate((Y_std_new[0 : 4], Y_d_std))
        X_l = np.concatenate((X_data_new[10 : 15, 0:6], X_l))
        Y_l = np.concatenate((Y_data_new[10 : 15], Y_l))
        Y_l_std = np.concatenate((Y_std_new[10 : 15], Y_l_std))
        X_b = np.concatenate((X_data_new[19 : 24, 0:6], X_b))
        Y_b = np.concatenate((Y_data_new[19 : 24], Y_b))
        Y_b_std = np.concatenate((Y_std_new[19 : 24], Y_b_std))
        X_s = np.concatenate((X_data_new[28 : 33, 0:6], X_s))
        Y_s = np.concatenate((Y_data_new[28 : 33], Y_s))
        Y_s_std = np.concatenate((Y_std_new[28 : 33], Y_s_std))

X_d_planck = X_d[(n_simulations) * 4 : (n_simulations + 1) * 4]
Y_d_planck = Y_d[(n_simulations) * 4 : (n_simulations + 1) * 4]
Y_d_std_planck = Y_d_std[(n_simulations) * 4 : (n_simulations + 1) * 4]
X_l_planck = X_l[(n_simulations) * 5 : (n_simulations + 1) * 5]
Y_l_planck = Y_l[(n_simulations) * 5 : (n_simulations + 1) * 5]
Y_l_std_planck = Y_l_std[(n_simulations) * 5 : (n_simulations + 1) * 5]
X_b_planck = X_b[(n_simulations) * 5 : (n_simulations + 1) * 5]
Y_b_planck = Y_b[(n_simulations) * 5 : (n_simulations + 1) * 5]
Y_b_std_planck = Y_b_std[(n_simulations) * 5 : (n_simulations + 1) * 5]
X_s_planck = X_s[(n_simulations) * 5 : (n_simulations + 1) * 5]
Y_s_planck = Y_s[(n_simulations) * 5 : (n_simulations + 1) * 5]
Y_s_std_planck = Y_s_std[(n_simulations) * 5 : (n_simulations + 1) * 5]
Y_d_planck_expected = np.reshape(Y_d_planck, (4, 1))
Y_l_planck_expected = np.reshape(Y_l_planck, (5, 1))
Y_b_planck_expected = np.reshape(Y_b_planck, (5, 1))
Y_s_planck_expected = np.reshape(Y_s_planck, (5, 1))
Y_d_std_planck_expected = np.reshape(Y_d_std_planck, (4, 1))
Y_l_std_planck_expected = np.reshape(Y_l_std_planck, (5, 1))
Y_b_std_planck_expected = np.reshape(Y_b_std_planck, (5, 1))
Y_s_std_planck_expected = np.reshape(Y_s_std_planck, (5, 1))

X_d_data = X_d[0 : (n_simulations) * 4]
Y_d_data = Y_d[0 : (n_simulations) * 4]
Y_d_std_data = Y_d_std[0 : (n_simulations) * 4]
X_l_data = X_l[0 : (n_simulations) * 5]
Y_l_data = Y_l[0 : (n_simulations) * 5]
Y_l_std_data = Y_l_std[0 : (n_simulations) * 5]
X_b_data = X_b[0 : (n_simulations) * 5]
Y_b_data = Y_b[0 : (n_simulations) * 5]
Y_b_std_data = Y_b_std[0 : (n_simulations) * 5]
X_s_data = X_s[0 : (n_simulations) * 5]
Y_s_data = Y_s[0 : (n_simulations) * 5]
Y_s_std_data = Y_s_std[0 : (n_simulations) * 5]
Y_d_data = np.reshape(Y_d_data, (n_simulations * 4, 1))
Y_l_data = np.reshape(Y_l_data, (n_simulations * 5, 1))
Y_b_data = np.reshape(Y_b_data, (n_simulations * 5, 1))
Y_s_data = np.reshape(Y_s_data, (n_simulations * 5, 1))
Y_d_std_data = np.reshape(Y_d_std_data, (n_simulations * 4, 1))
Y_l_std_data = np.reshape(Y_l_std_data, (n_simulations * 5, 1))
Y_b_std_data = np.reshape(Y_b_std_data, (n_simulations * 5, 1))
Y_s_std_data = np.reshape(Y_s_std_data, (n_simulations * 5, 1))

print("data loaded")

## Setting up the GPs
print("starting to define the Gps")

#gp = GP.GP(X = [X_d_data, X_l_data, X_b_data, X_s_data], Y = [Y_d_data, Y_l_data, Y_b_data, Y_s_data], N_points_per_simu = [4, 5, 5, 5], Noise = [Y_d_std_data, Y_l_std_data, Y_b_std_data, Y_s_std_data], type_kernel = "Separated")
gp = GP.GP(X = [X_d_data, X_l_data, X_b_data, X_s_data], Y = [Y_d_data, Y_l_data, Y_b_data, Y_s_data], N_points_per_simu = [4, 5, 5, 5], Noise = [None, None, None, None], type_kernel = "Separated")

print("models defined")

## Optimising the hyperparameters - Gradient Descent
print("Starting to optimise the hyperparameters")

gp.optimize_models(optimizer = 'lbfgsb')
print("Hyperparameters optimised")

#gp.change_models_to_heteroscedatic(Noise = [Y_d_std_data, Y_l_std_data, Y_b_std_data, Y_s_std_data])
#print("model changed to heteroscedatic")

gp.print_models()

## Making a prediction
print("Starting to make a prediction")

X_planck = np.reshape(X_d_planck[0, 0:5], (1, 5))
X_planck_predicted, Y_planck_predicted, Cov = gp.compute_prediction(X_planck)

X_d_planck_predicted = X_planck_predicted[0][0]
X_l_planck_predicted = X_planck_predicted[0][1]
X_b_planck_predicted = X_planck_predicted[0][2]
X_s_planck_predicted = X_planck_predicted[0][3]

Y_d_planck_predicted = Y_planck_predicted[0][0]
Y_l_planck_predicted = Y_planck_predicted[0][1]
Y_b_planck_predicted = Y_planck_predicted[0][2]
Y_s_planck_predicted = Y_planck_predicted[0][3]

Cov_d_predicted = np.sqrt(np.reshape(np.diag(Cov[0][0]), (4, 1)))
Cov_l_predicted = np.sqrt(np.reshape(np.diag(Cov[0][1]), (5, 1)))
Cov_b_predicted = np.sqrt(np.reshape(np.diag(Cov[0][2]), (5, 1)))
Cov_s_predicted = np.sqrt(np.reshape(np.diag(Cov[0][3]), (5, 1)))

print("Prediction done")
rms = gp.test_rms(X_test = [X_d_planck, X_l_planck, X_b_planck, X_s_planck], Y_test = [Y_d_planck_expected, Y_l_planck_expected, Y_b_planck_expected, Y_s_planck_predicted])
print("RMS Planck : " + str(rms))
chi2 = gp.test_chi2(X_test = [X_d_planck, X_l_planck, X_b_planck, X_s_planck], Y_test = [Y_d_planck_expected, Y_l_planck_expected, Y_b_planck_expected, Y_s_planck_predicted], Noise_test = [Y_d_std_planck_expected, Y_l_std_planck_expected, Y_b_std_planck_expected, Y_s_std_planck_expected])
print("Chi 2 Planck : " + str(chi2))

## Loading the whole Abacus data for plotting
print("starting to load the data")

#target = "/home/astro/magnan/Repository_Stage_3A/Full_MST_stats_Abacus/MST_stats_Catalogue_"
target = "C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Full_MST_stats_Abacus/MST_stats_Catalogue_"

dict = {'X_d' : [], 'Y_d' : [], 'Y_d_std' : [], 'X_l' : [], 'Y_l' : [], 'Y_l_std' : [], 'X_b' : [], 'Y_b' : [], 'Y_b_std' : [], 'X_s' : [], 'Y_s' : [], 'Y_s_std' : []}

for i in range(41):    
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

## Plot
print("Starting to plot the MST stats")

fig, axes = plt.subplots(nrows = 2, ncols = 4, figsize = (20, 10))
plt.subplots_adjust(hspace = 0.3, wspace = 0.4)

for j in range(4):
    for i in range(2):
        subplot = axes[i][j]
        
        if (j == 0):
            subplot.set_xlabel('$d$')
            
            Mean = dict['Y_d'][0]
            Std = np.asarray([0 for k in range(np.shape(dict['Y_d'][0])[0])])
            for k in range(1, 41):
                Mean_old = Mean.copy()
                Std_old = Std.copy()
                
                Mean_new = (k * Mean_old + dict['Y_d'][k]) / (k + 1)
                Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + dict['Y_d'][k]**2) / (k + 1) - Mean_new**2)
                
                Mean = Mean_new.copy()
                Std = Std_new.copy()
            
            if (i == 0):
                subplot.set_ylabel('$N_{d}$')
                subplot.set_yscale('log')
                subplot.set_xlim(6 * X_d_planck_predicted[:, 5][0], 6 * X_d_planck_predicted[:, 5][-1])
                
                subplot.fill_between(x = dict['X_d'][40], y1 = Mean - Std, y2 = Mean + Std, color = 'b', alpha = 0.2, label = 'Abacus range')
                subplot.plot(6 * X_d_planck_predicted[:, 5], 10**Y_d_planck_predicted, 'b', label = "Prediction")
                subplot.errorbar(6 * X_d_planck_predicted[:, 5], 10**Y_d_planck_predicted, yerr = np.log(10) * Cov_d_predicted * 10**Y_d_planck_predicted, fmt = 'none', ecolor = 'blue')
                subplot.plot(dict['X_d'][40], dict['Y_d'][40], 'b--', label = "Expectation")
                subplot.errorbar(dict['X_d'][40], dict['Y_d'][40], yerr = dict['Y_d_std'][40], fmt = 'none', ecolor = 'blue')
                
                subplot.legend()
                
            else:
                subplot.set_ylabel('$\Delta N_{d} / <N_{d}>$')
                subplot.set_xlim(6 * X_d_planck_predicted[:, 5][0], 6 * X_d_planck_predicted[:, 5][-1])
                
                M = []
                for k in range(4):
                    min = 1
                    l_min = 0
                    
                    x1 = X_d_planck_predicted[:, 5][k]
                    
                    for l in range(len(dict['X_d'][40])):
                        x2 = dict['X_d'][40][l] / 6
                        dist = abs(x2 - x1)
                        if (dist < min):
                            l_min = l
                            min = dist

                    M.append(Mean[l_min])

                M = np.reshape(np.array(M), (4, 1))
                
                subplot.fill_between(x = dict['X_d'][40], y1 = - Std / Mean, y2 = Std / Mean, color = 'b', alpha = 0.2, label = "Abacus range")
                subplot.plot(6 * X_d_planck_predicted[:, 5], (10**Y_d_planck_predicted - M) / M, 'b', label = "Prediction")
                subplot.errorbar(6 * X_d_planck_predicted[:, 5], (10**Y_d_planck_predicted - M) / M, yerr = np.log(10) * Cov_d_predicted / M * 10**Y_d_planck_predicted, fmt = 'none', ecolor = 'blue')
                subplot.plot(dict['X_d'][40], (dict['Y_d'][40] - Mean) / Mean, 'b--', label = "Expectation")
                subplot.errorbar(dict['X_d'][40], (dict['Y_d'][40] - Mean) / Mean, yerr = dict['Y_d_std'][40] / Mean, fmt = 'none', ecolor = 'blue')
                
                subplot.legend()
        
        elif (j == 1):
            subplot.set_xlabel('$l$')
            subplot.set_xscale('log')
            
            Mean = dict['Y_l'][0]
            Std = np.asarray([0 for k in range(np.shape(dict['Y_l'][0])[0])])
            for k in range(1, 41):
                Mean_old = Mean.copy()
                Std_old = Std.copy()
                
                Mean_new = (k * Mean_old + dict['Y_l'][k]) / (k + 1)
                Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + dict['Y_l'][k]**2) / (k + 1) - Mean_new**2)
                
                Mean = Mean_new.copy()
                Std = Std_new.copy()
            
            if (i == 0):
                subplot.set_ylabel('$N_{l}$')
                subplot.set_yscale('log')
                subplot.set_xlim(10**(X_l_planck_predicted[:, 5][0]), 10**(X_l_planck_predicted[:, 5][-1]))
                
                subplot.fill_between(x = dict['X_l'][40], y1 = Mean - Std, y2 = Mean + Std, color = 'g', alpha = 0.2, label = 'Abacus range')
                subplot.plot(10**X_l_planck_predicted[:, 5], 10**Y_l_planck_predicted, 'g', label = "Prediction")
                subplot.errorbar(10**X_l_planck_predicted[:, 5], 10**Y_l_planck_predicted, yerr = np.log(10) * Cov_l_predicted * 10**Y_l_planck_predicted, fmt = 'none', ecolor = 'green')
                subplot.plot(dict['X_l'][40], dict['Y_l'][40], 'g--', label = "Expectation")
                subplot.errorbar(dict['X_l'][40], dict['Y_l'][40], yerr = dict['Y_l_std'][40], fmt = 'none', ecolor = 'green')
                
                subplot.legend()
                
            else:
                subplot.set_ylabel('$\Delta N_{l} / <N_{l}>$')
                subplot.set_xlim(10**(X_l_planck_predicted[:, 5][0]), 10**(X_l_planck_predicted[:, 5][-1]))
                
                M = []
                for k in range(5):
                    min = 1
                    l_min = 0
                    
                    x1 = X_l_planck_predicted[:, 5][k]
                    
                    for l in range(len(dict['X_l'][40])):
                        x2 = np.log10(dict['X_l'][40][l])
                        dist = abs(x2 - x1)
                        if (dist < min):
                            l_min = l
                            min = dist

                    M.append(Mean[l_min])

                M = np.reshape(np.array(M), (5, 1))
                
                subplot.fill_between(x = dict['X_l'][40], y1 = - Std / Mean, y2 = Std / Mean, color = 'g', alpha = 0.2, label = "Abacus range")
                subplot.plot(10**X_l_planck_predicted[:, 5], (10**Y_l_planck_predicted - M) / M, 'g', label = "Prediction")
                subplot.errorbar(10**X_l_planck_predicted[:, 5], (10**Y_l_planck_predicted - M) / M, yerr = np.log(10) * Cov_l_predicted / M * 10**Y_l_planck_predicted, fmt = 'none', ecolor = 'green')
                subplot.plot(dict['X_l'][40], (dict['Y_l'][40] - Mean) / Mean, 'g--', label = "Expectation")
                subplot.errorbar(dict['X_l'][40], (dict['Y_l'][40] - Mean) / Mean, yerr = dict['Y_l_std'][40] / Mean, fmt = 'none', ecolor = 'green')
                
                subplot.legend()
                
        elif (j == 2):
            subplot.set_xlabel('$b$')
            subplot.set_xscale('log')
            
            Mean = dict['Y_b'][0]
            Std = np.asarray([0 for k in range(np.shape(dict['Y_b'][0])[0])])
            for k in range(1, 41):
                Mean_old = Mean.copy()
                Std_old = Std.copy()
                
                Mean_new = (k * Mean_old + dict['Y_b'][k]) / (k + 1)
                Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + dict['Y_b'][k]**2) / (k + 1) - Mean_new**2)
                
                Mean = Mean_new.copy()
                Std = Std_new.copy()
            
            if (i == 0):
                subplot.set_ylabel('$N_{b}$')
                subplot.set_yscale('log')
                subplot.set_xlim(10**(X_b_planck_predicted[:, 5][0]), 10**(X_b_planck_predicted[:, 5][-1]))
                
                subplot.fill_between(x = dict['X_b'][40], y1 = Mean - Std, y2 = Mean + Std, color = 'r', alpha = 0.2, label = 'Abacus range')
                subplot.plot(10**X_b_planck_predicted[:, 5], 10**Y_b_planck_predicted, 'r', label = "Prediction")
                subplot.errorbar(10**X_b_planck_predicted[:, 5], 10**Y_b_planck_predicted, yerr = np.log(10) * Cov_b_predicted * 10**Y_b_planck_predicted, fmt = 'none', ecolor = 'red')
                subplot.plot(dict['X_b'][40], dict['Y_b'][40], 'r--', label = "Expectation")
                subplot.errorbar(dict['X_b'][40], dict['Y_b'][40], yerr = dict['Y_b_std'][40], fmt = 'none', ecolor = 'red')
                
                subplot.legend()
                
            else:
                subplot.set_ylabel('$\Delta N_{b} / <N_{b}>$')
                subplot.set_xlim(10**(X_b_planck_predicted[:, 5][0]), 10**(X_b_planck_predicted[:, 5][-1]))
                
                M = []
                for k in range(5):
                    min = 1
                    l_min = 0
                    
                    x1 = X_b_planck_predicted[:, 5][k]
                    
                    for l in range(len(dict['X_b'][40])):
                        x2 = np.log10(dict['X_b'][40][l])
                        dist = abs(x2 - x1)
                        if (dist < min):
                            l_min = l
                            min = dist

                    M.append(Mean[l_min])

                M = np.reshape(np.array(M), (5, 1))
                
                subplot.fill_between(x = dict['X_b'][40], y1 = - Std / Mean, y2 = Std / Mean, color = 'r', alpha = 0.2, label = "Abacus range")
                subplot.plot(10**X_b_planck_predicted[:, 5], (10**Y_b_planck_predicted - M) / M, 'r', label = "Prediction")
                subplot.errorbar(10**X_b_planck_predicted[:, 5], (10**Y_b_planck_predicted - M) / M, yerr = np.log(10) * Cov_b_predicted / M * 10**Y_b_planck_predicted, fmt = 'none', ecolor = 'red')
                subplot.plot(dict['X_b'][40], (dict['Y_b'][40] - Mean) / Mean, 'r--', label = "Expectation")
                subplot.errorbar(dict['X_b'][40], (dict['Y_b'][40] - Mean) / Mean, yerr = dict['Y_b_std'][40] / Mean, fmt = 'none', ecolor = 'red')
                
                subplot.legend()
                
        else:
            subplot.set_xlabel('$s$')
            
            Mean = dict['Y_s'][0]
            Std = np.asarray([0 for k in range(np.shape(dict['Y_s'][0])[0])])
            for k in range(1, 41):
                Mean_old = Mean.copy()
                Std_old = Std.copy()
                
                Mean_new = (k * Mean_old + dict['Y_s'][k]) / (k + 1)
                Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + dict['Y_s'][k]**2) / (k + 1) - Mean_new**2)
                
                Mean = Mean_new.copy()
                Std = Std_new.copy()
            
            if (i == 0):
                subplot.set_ylabel('$N_{s}$')
                subplot.set_yscale('log')
                subplot.set_xlim(X_s_planck_predicted[:, 5][0], X_s_planck_predicted[:, 5][-1])
                
                subplot.fill_between(x = dict['X_s'][40], y1 = Mean - Std, y2 = Mean + Std, color = 'y', alpha = 0.2, label = 'Abacus range')
                subplot.plot(X_s_planck_predicted[:, 5], 10**Y_s_planck_predicted, 'y', label = "Prediction")
                subplot.errorbar(X_s_planck_predicted[:, 5], 10**Y_s_planck_predicted, yerr = np.log(10) * Cov_s_predicted * 10**Y_s_planck_predicted, fmt = 'none', ecolor = 'y')
                subplot.plot(dict['X_s'][40], dict['Y_s'][40], 'y--', label = "Expectation")
                subplot.errorbar(dict['X_s'][40], dict['Y_s'][40], yerr = dict['Y_s_std'][40], fmt = 'none', ecolor = 'y')
                
                subplot.legend()
                
            else:
                subplot.set_ylabel('$\Delta N_{s} / <N_{s}>$')
                subplot.set_xlim(X_s_planck_predicted[:, 5][0], X_s_planck_predicted[:, 5][-1])
                
                M = []
                for k in range(5):
                    min = 1
                    l_min = 0
                    
                    x1 = X_s_planck_predicted[:, 5][k]
                    
                    for l in range(len(dict['X_s'][40])):
                        x2 = dict['X_s'][40][l]
                        dist = abs(x2 - x1)
                        if (dist < min):
                            l_min = l
                            min = dist

                    M.append(Mean[l_min])

                M = np.reshape(np.array(M), (5, 1))
                
                subplot.fill_between(x = dict['X_s'][40], y1 = - Std / Mean, y2 = Std / Mean, color = 'y', alpha = 0.2, label = "Abacus range")
                subplot.plot(X_s_planck_predicted[:, 5], (10**Y_s_planck_predicted - M) / M, 'y', label = "Prediction")
                subplot.errorbar(X_s_planck_predicted[:, 5], (10**Y_s_planck_predicted - M) / M, yerr = np.log(10) * Cov_s_predicted / M * 10**Y_s_planck_predicted, fmt = 'none', ecolor = 'y')
                subplot.plot(dict['X_s'][40], (dict['Y_s'][40] - Mean) / Mean, 'y--', label = "Expectation")
                subplot.errorbar(dict['X_s'][40], (dict['Y_s'][40] - Mean) / Mean, yerr = dict['Y_s_std'][40] / Mean, fmt = 'none', ecolor = 'y')
                
                subplot.legend()

plt.suptitle("Comparison between prediction and expectation - Separated Heteroscedatic GPs")
plt.show()

#my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')
my_path = os.path.abspath('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Figures')
my_file = 'Comparison_prediction_expectation_separated'
my_file = os.path.join(my_path, my_file)
#plt.savefig(my_file)

print("results plotted and saved")