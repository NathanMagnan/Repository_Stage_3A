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
n_fiducial = 21

X_d = None
Y_d = None
X_l = None
Y_l = None
X_b = None
Y_b = None
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
        X_l = X_data_new[10 : 15, 0 : 6]
        Y_l = Y_data_new[10 : 15]
        X_b = X_data_new[19 : 24, 0 : 6]
        Y_b = Y_data_new[19 : 24]
        X_s = X_data_new[28 : 33, 0 : 6]
        Y_s = Y_data_new[28 : 33]
    else:
        X_d = np.concatenate((X_data_new[0 : 4, 0:6], X_d))
        Y_d = np.concatenate((Y_data_new[0 : 4], Y_d))
        X_l = np.concatenate((X_data_new[10 : 15, 0:6], X_l))
        Y_l = np.concatenate((Y_data_new[10 : 15], Y_l))
        X_b = np.concatenate((X_data_new[19 : 24, 0:6], X_b))
        Y_b = np.concatenate((Y_data_new[19 : 24], Y_b))
        X_s = np.concatenate((X_data_new[28 : 33, 0:6], X_s))
        Y_s = np.concatenate((Y_data_new[28 : 33], Y_s))

X_d_planck = X_d[:(n_fiducial) * 4]
X_l_planck = X_l[:(n_fiducial) * 5]
X_b_planck = X_b[:(n_fiducial) * 5]
X_s_planck = X_s[:(n_fiducial) * 5]

X_d_data = X_d[(n_fiducial) * 4:]
Y_d_data = Y_d[(n_fiducial) * 4:]
X_l_data = X_l[(n_fiducial) * 5:]
Y_l_data = Y_l[(n_fiducial) * 5:]
X_b_data = X_b[(n_fiducial) * 5:]
Y_b_data = Y_b[(n_fiducial) * 5:]
X_s_data = X_s[(n_fiducial) * 5:]
Y_s_data = Y_s[(n_fiducial) * 5:]
Y_d_data = np.reshape(Y_d_data, (n_simulations * 4, 1))
Y_l_data = np.reshape(Y_l_data, (n_simulations * 5, 1))
Y_b_data = np.reshape(Y_b_data, (n_simulations * 5, 1))
Y_s_data = np.reshape(Y_s_data, (n_simulations * 5, 1))

X_d_planck_expected = X_d_data[-4:, 5]
Y_d_planck_expected = np.asarray([0 for i in range(np.shape(X_d_data[0:4, 5])[0])])
Y_d_std_planck_expected = np.asarray([0 for i in range(np.shape(X_d_data[0:4, 5])[0])])
for i in range(n_fiducial):
    New = []
    for x1 in X_d_planck_expected:
        min = 10
        j_min = 0
        for j in range(np.shape(dict['X_d'][i])[0]):
            x2 = dict['X_d'][i][j] / 6
            if (abs(x1 - x2) < min):
                min = abs(x1 - x2)
                j_min = j
        New.append(dict['Y_d'][i + 40][j_min])

    New = np.asarray(New)
    
    Mean_old = Y_d_planck_expected.copy()
    Std_old = Y_d_std_planck_expected.copy()
    
    Mean_new = (i * Mean_old + New) / (i + 1)
    Std_new = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_new**2)
    
    Y_d_planck_expected = Mean_new.copy()
    Y_d_std_planck_expected = Std_new.copy()
Y_d_std_planck_expected = Y_d_std_planck_expected / (np.log(10) * Y_d_planck_expected)
Y_d_planck_expected = np.log10(Y_d_planck_expected)
for i in range(4):
    X_d_planck[i, 5] = X_d_planck_expected[i]
X_d_planck = X_d_planck[0:4]

X_l_planck_expected = X_l_data[-5:, 5]
Y_l_planck_expected = np.asarray([0 for i in range(np.shape(X_l_data[-5:, 5])[0])])
Y_l_std_planck_expected = np.asarray([0 for i in range(np.shape(X_l_data[-5:, 5])[0])])
for i in range(n_fiducial):
    New = []
    for x1 in 10**X_l_planck_expected:
        min = 10
        j_min = 0
        for j in range(np.shape(dict['X_l'][n_simulations + i])[0]):
            x2 = dict['X_l'][n_simulations + i][j]
            if (abs(x1 - x2) < min):
                min = abs(x1 - x2)
                j_min = j
        try:
            New.append((dict['Y_l'][i + 40][j_min - 1] + dict['Y_l'][i + 40][j_min] + dict['Y_l'][i + 40][j_min + 1]) / 3)
        except:
            New.append(dict['Y_l'][i + 40][j_min])

    New = np.asarray(New)
    
    Mean_old = Y_l_planck_expected.copy()
    Std_old = Y_l_std_planck_expected.copy()
    
    Mean_new = (i * Mean_old + New) / (i + 1)
    Std_new = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_new**2)
    
    Y_l_planck_expected = Mean_new.copy()
    Y_l_std_planck_expected = Std_new.copy()
Y_l_std_planck_expected = Y_l_std_planck_expected / (np.log(10) * Y_l_planck_expected)
Y_l_planck_expected = np.log10(Y_l_planck_expected)
for i in range(5):
    X_l_planck[i, 5] = X_l_planck_expected[i]
X_l_planck = X_l_planck[0:5]

X_b_planck_expected = X_b_data[-5:, 5]
Y_b_planck_expected = np.asarray([0 for i in range(np.shape(X_b_data[-5:, 5])[0])])
Y_b_std_planck_expected = np.asarray([0 for i in range(np.shape(X_b_data[-5:, 5])[0])])
for i in range(n_fiducial):
    New = []
    for x1 in 10**X_b_planck_expected:
        min = 10
        j_min = 0
        for j in range(np.shape(dict['X_b'][n_simulations + i])[0]):
            x2 = dict['X_b'][n_simulations + i][j]
            if (abs(x1 - x2) < min):
                min = abs(x1 - x2)
                j_min = j
        try:
            New.append((dict['Y_b'][i + 40][j_min - 1] + dict['Y_b'][i + 40][j_min] + dict['Y_b'][i + 40][j_min + 1]) / 3)
        except:
            New.append(dict['Y_b'][i + 40][j_min])

    New = np.asarray(New)
    
    Mean_old = Y_b_planck_expected.copy()
    Std_old = Y_b_std_planck_expected.copy()
    
    Mean_new = (i * Mean_old + New) / (i + 1)
    Std_new = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_new**2)
    
    Y_b_planck_expected = Mean_new.copy()
    Y_b_std_planck_expected = Std_new.copy()
Y_b_std_planck_expected = Y_b_std_planck_expected / (np.log(10) * Y_b_planck_expected)
Y_b_planck_expected = np.log10(Y_b_planck_expected)
for i in range(5):
    X_b_planck[i, 5] = X_b_planck_expected[i]
X_b_planck = X_b_planck[0:5]

X_s_planck_expected = X_s_data[-5:, 5]
Y_s_planck_expected = np.asarray([0 for i in range(np.shape(X_s_data[-5:, 5])[0])])
Y_s_std_planck_expected = np.asarray([0 for i in range(np.shape(X_s_data[-5:, 5])[0])])
for i in range(n_fiducial):
    New = []
    for x1 in X_s_planck_expected:
        min = 10
        j_min = 0
        for j in range(np.shape(dict['X_s'][n_simulations + i])[0]):
            x2 = dict['X_s'][n_simulations + i][j]
            if (abs(x1 - x2) < min):
                min = abs(x1 - x2)
                j_min = j
        try:
            New.append((dict['Y_s'][i + 40][j_min - 1] + dict['Y_s'][i + 40][j_min] + dict['Y_s'][i + 40][j_min + 1]) / 3)
        except:
            New.append(dict['Y_s'][i + 40][j_min])

    New = np.asarray(New)
    
    Mean_old = Y_s_planck_expected.copy()
    Std_old = Y_s_std_planck_expected.copy()
    
    Mean_new = (i * Mean_old + New) / (i + 1)
    Std_new = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_new**2)
    
    Y_s_planck_expected = Mean_new.copy()
    Y_s_std_planck_expected = Std_new.copy()
Y_s_std_planck_expected = Y_s_std_planck_expected / (np.log(10) * Y_s_planck_expected)
Y_s_planck_expected = np.log10(Y_s_planck_expected)
for i in range(5):
    X_s_planck[i, 5] = X_s_planck_expected[i]
X_s_planck = X_s_planck[0:5]

print("data loaded")

## Setting up the GPs
print("starting to define the Gps")

gp = GP.GP(X = [X_d_data, X_l_data, X_b_data, X_s_data], Y = [Y_d_data, Y_l_data, Y_b_data, Y_s_data], N_points_per_simu = [4, 5, 5, 5], Noise = [None, None, None, None], type_kernel = "Separated")

print("models defined")

## Optimising the hyperparameters - Gradient Descent
print("Starting to optimise the hyperparameters")

gp.optimize_models(optimizer = 'lbfgsb')
print("Hyperparameters optimised")

gp.print_models()

## Making a prediction
print("Starting to make a prediction")

X_planck = np.reshape(X_d_planck[0, 0:5], (1, 5))
X_planck_predicted, Y_planck_predicted, Cov = gp.compute_prediction(X_planck)

X_d_planck_predicted = X_planck_predicted[0][0][:, 5]
X_l_planck_predicted = X_planck_predicted[0][1][:, 5]
X_b_planck_predicted = X_planck_predicted[0][2][:, 5]
X_s_planck_predicted = X_planck_predicted[0][3][:, 5]

Y_d_planck_predicted = np.reshape(Y_planck_predicted[0][0], (4,))
Y_l_planck_predicted = np.reshape(Y_planck_predicted[0][1], (5,))
Y_b_planck_predicted = np.reshape(Y_planck_predicted[0][2], (5,))
Y_s_planck_predicted = np.reshape(Y_planck_predicted[0][3], (5,))

Y_d_std_planck_predicted = np.sqrt(np.reshape(np.diag(Cov[0][0]), (1, 4)))
Y_l_std_planck_predicted = np.sqrt(np.reshape(np.diag(Cov[0][1]), (1, 5)))
Y_b_std_planck_predicted = np.sqrt(np.reshape(np.diag(Cov[0][2]), (1, 5)))
Y_s_std_planck_predicted = np.sqrt(np.reshape(np.diag(Cov[0][3]), (1, 5)))

print("Prediction done")
rms = gp.test_rms(X_test = [X_d_planck, X_l_planck, X_b_planck, X_s_planck], Y_test = [Y_d_planck_expected, Y_l_planck_expected, Y_b_planck_expected, Y_s_planck_expected])
print("RMS Planck : " + str(rms))
chi2 = gp.test_chi2(X_test = [X_d_planck, X_l_planck, X_b_planck, X_s_planck], Y_test = [Y_d_planck_expected, Y_l_planck_expected, Y_b_planck_expected, Y_s_planck_expected], Noise_test = [Y_d_std_planck_expected, Y_l_std_planck_expected, Y_b_std_planck_expected, Y_s_std_planck_expected])
print("Chi 2 Planck : " + str(chi2))

## Plotting
print("Starting to plot the MST stats")

fig, axes = plt.subplots(nrows = 2, ncols = 4, figsize = (20, 10))
plt.subplots_adjust(hspace = 0.3, wspace = 0.4)

for j in range(4):
    for i in range(2):
        subplot = axes[i][j]
        
        if (j == 0):
            subplot.set_xlabel('$d$')
            subplot.set_xlim(1, 4)
            
            X_abacus = dict['X_d'][0]
            Mean_abacus = np.asarray([0 for k in range(np.shape(dict['X_d'][0])[0])])
            Std_abacus = np.asarray([0 for k in range(np.shape(dict['X_d'][0])[0])])
            for k in range(n_simulations):
                New = []
                for x1 in X_abacus:
                    min = 10
                    l_min = 0
                    for l in range(np.shape(X_abacus)[0]):
                        x2 = dict['X_d'][k][l]
                        if (abs(x1 - x2) < min):
                            min = abs(x1 - x2)
                            l_min = l
                    
                    New.append(dict['Y_d'][k][l_min])
                New = np.asarray(New)
                
                Mean_old = Mean_abacus.copy()
                Std_old = Std_abacus.copy()
                
                Mean_new = (k * Mean_old + New) / (k + 1)
                Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + New**2) / (k + 1) - Mean_new**2)
                
                Mean_abacus = Mean_new.copy()
                Std_abacus = Std_new.copy()
            
            Mean_fidu = np.asarray([0 for k in range(np.shape(dict['X_d'][0])[0])])
            Std_fidu = np.asarray([0 for k in range(np.shape(dict['X_d'][0])[0])])
            for k in range(n_fiducial):
                New = []
                for x1 in X_abacus:
                    min = 10
                    l_min = 0
                    for l in range(np.shape(X_abacus)[0]):
                        x2 = dict['X_d'][k + n_simulations][l]
                        if (abs(x1 - x2) < min):
                            min = abs(x1 - x2)
                            l_min = l
                    
                    New.append(dict['Y_d'][k + n_simulations][l_min])
                New = np.asarray(New)
                    
                Mean_old = Mean_fidu.copy()
                Std_old = Std_fidu.copy()
                
                Mean_new = (k * Mean_old + New) / (k + 1)
                Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + New**2) / (k + 1) - Mean_new**2)
                
                Mean_fidu = Mean_new.copy()
                Std_fidu = Std_new.copy()
            
            Mean_abacus_reduced = np.asarray([0 for i in range(np.shape(X_d_planck_expected)[0])])
            for k in range(np.shape(Mean_abacus_reduced)[0]):
                l_min = 0
                min = 10
                x1 = X_d_planck_expected[k]
                for l in range(np.shape(X_abacus)[0]):
                    x2 = X_abacus[l] / 6
                    if (abs(x1 - x2) < min):
                        min = abs(x1 - x2)
                        l_min = l
                Mean_abacus_reduced[k] = Mean_abacus[l_min]
            Mean_abacus_reduced = np.log10(np.asarray(Mean_abacus_reduced))
            
            Mean_abacus_reduced_2 = np.asarray([0 for i in range(np.shape(X_d_planck_predicted)[0])])
            for k in range(np.shape(Mean_abacus_reduced)[0]):
                l_min = 0
                min = 10
                x1 = X_d_planck_predicted[k]
                for l in range(np.shape(X_abacus)[0]):
                    x2 = X_abacus[l] / 6
                    if (abs(x1 - x2) < min):
                        min = abs(x1 - x2)
                        l_min = l
                Mean_abacus_reduced_2[k] = Mean_abacus[l_min]
            Mean_abacus_reduced_2 = np.log10(np.asarray(Mean_abacus_reduced_2))
            
            if (i == 0):
                subplot.set_ylabel('$N_{d}$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**4, 10**6)
                
                subplot.fill_between(x = X_abacus, y1 = Mean_abacus - Std_abacus, y2 = Mean_abacus + Std_abacus, color = 'b', alpha = 0.2, label = 'Abacus range')
                subplot.fill_between(x = X_abacus, y1 = Mean_fidu - Std_fidu, y2 = Mean_fidu + Std_fidu, color = 'b', alpha = 0.6, label = 'Fiducial range')
                subplot.plot(6 * X_d_planck_expected, 10**(Y_d_planck_expected), 'b', label = "Expectation")
                subplot.plot(6 * X_d_planck_predicted, 10**(Y_d_planck_predicted), 'b--', label = 'Prediction')
                
                subplot.legend()
                
            else:
                subplot.set_ylabel('$\Delta N_{d} / <N_{d}>$')
                subplot.set_ylim(-0.1, 0.1)
                
                subplot.fill_between(x = X_abacus, y1 = - Std_abacus / Mean_abacus, y2 = Std_abacus / Mean_abacus, color = 'b', alpha = 0.2, label = "Abacus range")
                subplot.fill_between(x = X_abacus, y1 = (Mean_fidu - Std_fidu - Mean_abacus) / Mean_abacus, y2 = (Mean_fidu + Std_fidu - Mean_abacus) / Mean_abacus, color = 'b', alpha = 0.6, label = "Fiducial range")
                subplot.plot(6 * X_d_planck_expected, (10**(Y_d_planck_expected) - 10**(Mean_abacus_reduced)) / 10**(Mean_abacus_reduced), 'b', label = "Expectation")
                subplot.plot(6 * X_d_planck_predicted, (10**(Y_d_planck_predicted) - 10**(Mean_abacus_reduced_2)) / 10**(Mean_abacus_reduced_2), 'b--', label = 'Prediction')
                
                subplot.legend()
        
        elif (j == 1):
            subplot.set_xlabel('$l$')
            subplot.set_xscale('log')
            subplot.set_xlim(1, 10)
            
            X_abacus = dict['X_l'][0]
            Mean_abacus = np.asarray([0 for k in range(np.shape(dict['X_l'][0])[0])])
            Std_abacus = np.asarray([0 for k in range(np.shape(dict['X_l'][0])[0])])
            for k in range(n_simulations):
                New = []
                for x1 in X_abacus:
                    min = 10
                    l_min = 0
                    for l in range(np.shape(X_abacus)[0]):
                        x2 = dict['X_l'][k][l]
                        if (abs(x1 - x2) < min):
                            min = abs(x1 - x2)
                            l_min = l
                    try:
                        New.append((dict['Y_l'][k][l_min - 1] + dict['Y_l'][k][l_min] + dict['Y_l'][k][l_min + 1]) / 3)
                    except:
                        New.append(dict['Y_l'][k][l_min])
                New = np.asarray(New)
                
                Mean_old = Mean_abacus.copy()
                Std_old = Std_abacus.copy()
                
                Mean_new = (k * Mean_old + New) / (k + 1)
                Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + New**2) / (k + 1) - Mean_new**2)
                
                Mean_abacus = Mean_new.copy()
                Std_abacus = Std_new.copy()
            
            Mean_fidu = np.asarray([0 for k in range(np.shape(dict['X_l'][0])[0])])
            Std_fidu = np.asarray([0 for k in range(np.shape(dict['X_l'][0])[0])])
            for k in range(n_fiducial):
                New = []
                for x1 in X_abacus:
                    min = 10
                    l_min = 0
                    for l in range(np.shape(X_abacus)[0]):
                        x2 = dict['X_l'][k + n_simulations][l]
                        if (abs(x1 - x2) < min):
                            min = abs(x1 - x2)
                            l_min = l
                    try:
                        New.append((dict['Y_l'][k + n_simulations][l_min - 1] + dict['Y_l'][k + n_simulations][l_min] + dict['Y_l'][k + n_simulations][l_min + 1]) / 3)
                    except:
                        New.append(dict['Y_l'][k + n_simulations][l_min])
                New = np.asarray(New)
                    
                Mean_old = Mean_fidu.copy()
                Std_old = Std_fidu.copy()
                
                Mean_new = (k * Mean_old + New) / (k + 1)
                Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + New**2) / (k + 1) - Mean_new**2)
                
                Mean_fidu = Mean_new.copy()
                Std_fidu = Std_new.copy()
            
            Mean_abacus_reduced = np.asarray([0 for i in range(np.shape(X_l_planck_expected)[0])])
            for k in range(np.shape(Mean_abacus_reduced)[0]):
                l_min = 0
                min = 10
                x1 = X_l_planck_expected[k]
                for l in range(np.shape(X_abacus)[0]):
                    x2 = np.log10(X_abacus[l])
                    if (abs(x1 - x2) < min):
                        min = abs(x1 - x2)
                        l_min = l
                Mean_abacus_reduced[k] = Mean_abacus[l_min]
            Mean_abacus_reduced = np.log10(np.asarray(Mean_abacus_reduced))
            
            Mean_abacus_reduced_2 = np.asarray([0 for i in range(np.shape(X_l_planck_predicted)[0])])
            for k in range(np.shape(Mean_abacus_reduced)[0]):
                l_min = 0
                min = 10
                x1 = X_l_planck_predicted[k]
                for l in range(np.shape(X_abacus)[0]):
                    x2 = np.log10(X_abacus[l])
                    if (abs(x1 - x2) < min):
                        min = abs(x1 - x2)
                        l_min = l
                Mean_abacus_reduced_2[k] = Mean_abacus[l_min]
            Mean_abacus_reduced_2 = np.log10(np.asarray(Mean_abacus_reduced_2))
            
            if (i == 0):
                subplot.set_ylabel('$N_{l}$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**3, 10**5)
                
                subplot.fill_between(x = X_abacus, y1 = Mean_abacus - Std_abacus, y2 = Mean_abacus + Std_abacus, color = 'g', alpha = 0.2, label = 'Abacus range')
                subplot.fill_between(x = X_abacus, y1 = Mean_fidu - Std_fidu, y2 = Mean_fidu + Std_fidu, color = 'g', alpha = 0.6, label = 'Fiducial range')
                subplot.plot(10**X_l_planck_expected, 10**(Y_l_planck_expected), 'g', label = "Expectation")
                subplot.plot(10**X_l_planck_predicted, 10**(Y_l_planck_predicted), 'g--', label = 'Prediction')
                
                subplot.legend()
                
            else:
                subplot.set_ylabel('$\Delta N_{l} / <N_{l}>$')
                subplot.set_ylim(-0.3, 0.3)
                
                subplot.fill_between(x = X_abacus, y1 = - Std_abacus / Mean_abacus, y2 = Std_abacus / Mean_abacus, color = 'g', alpha = 0.2, label = "Abacus range")
                subplot.fill_between(x = X_abacus, y1 = (Mean_fidu - Std_fidu - Mean_abacus) / Mean_abacus, y2 = (Mean_fidu + Std_fidu - Mean_abacus) / Mean_abacus, color = 'g', alpha = 0.6, label = "Fiducial range")
                subplot.plot(10**X_l_planck_expected, (10**(Y_l_planck_expected) - 10**(Mean_abacus_reduced)) / 10**(Mean_abacus_reduced), 'g', label = "Expectation")
                subplot.plot(10**X_l_planck_predicted, (10**(Y_l_planck_predicted) - 10**(Mean_abacus_reduced_2)) / 10**(Mean_abacus_reduced_2), 'g--', label = 'Prediction')
                
                subplot.legend()
                
        elif (j == 2):
            subplot.set_xlabel('$b$')
            subplot.set_xscale('log')
            subplot.set_xlim(3, 30)
            
            X_abacus = dict['X_b'][0]
            Mean_abacus = np.asarray([0 for k in range(np.shape(dict['X_b'][0])[0])])
            Std_abacus = np.asarray([0 for k in range(np.shape(dict['X_b'][0])[0])])
            for k in range(n_simulations):
                New = []
                for x1 in X_abacus:
                    min = 10
                    l_min = 0
                    for l in range(np.shape(X_abacus)[0]):
                        x2 = dict['X_b'][k][l]
                        if (abs(x1 - x2) < min):
                            min = abs(x1 - x2)
                            l_min = l
                    try:
                        New.append((dict['Y_b'][k][l_min - 1] + dict['Y_b'][k][l_min] + dict['Y_b'][k][l_min + 1]) / 3)
                    except:
                        New.append(dict['Y_b'][k][l_min])
                New = np.asarray(New)
                
                Mean_old = Mean_abacus.copy()
                Std_old = Std_abacus.copy()
                
                Mean_new = (k * Mean_old + New) / (k + 1)
                Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + New**2) / (k + 1) - Mean_new**2)
                
                Mean_abacus = Mean_new.copy()
                Std_abacus = Std_new.copy()
            
            Mean_fidu = np.asarray([0 for k in range(np.shape(dict['X_b'][0])[0])])
            Std_fidu = np.asarray([0 for k in range(np.shape(dict['X_b'][0])[0])])
            for k in range(n_fiducial):
                New = []
                for x1 in X_abacus:
                    min = 10
                    l_min = 0
                    for l in range(np.shape(X_abacus)[0]):
                        x2 = dict['X_b'][k + n_simulations][l]
                        if (abs(x1 - x2) < min):
                            min = abs(x1 - x2)
                            l_min = l
                    try:
                        New.append((dict['Y_b'][k + n_simulations][l_min - 1] + dict['Y_b'][k + n_simulations][l_min] + dict['Y_b'][k + n_simulations][l_min + 1]) / 3)
                    except:
                        New.append(dict['Y_b'][k + n_simulations][l_min])
                New = np.asarray(New)
                    
                Mean_old = Mean_fidu.copy()
                Std_old = Std_fidu.copy()
                
                Mean_new = (k * Mean_old + New) / (k + 1)
                Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + New**2) / (k + 1) - Mean_new**2)
                
                Mean_fidu = Mean_new.copy()
                Std_fidu = Std_new.copy()
            
            Mean_abacus_reduced = np.asarray([0 for i in range(np.shape(X_b_planck_expected)[0])])
            for k in range(np.shape(Mean_abacus_reduced)[0]):
                l_min = 0
                min = 10
                x1 = X_b_planck_expected[k]
                for l in range(np.shape(X_abacus)[0]):
                    x2 = np.log10(X_abacus[l])
                    if (abs(x1 - x2) < min):
                        min = abs(x1 - x2)
                        l_min = l
                Mean_abacus_reduced[k] = Mean_abacus[l_min]
            Mean_abacus_reduced = np.log10(np.asarray(Mean_abacus_reduced))
            
            Mean_abacus_reduced_2 = np.asarray([0 for i in range(np.shape(X_b_planck_predicted)[0])])
            for k in range(np.shape(Mean_abacus_reduced)[0]):
                l_min = 0
                min = 10
                x1 = X_b_planck_predicted[k]
                for l in range(np.shape(X_abacus)[0]):
                    x2 = np.log10(X_abacus[l])
                    if (abs(x1 - x2) < min):
                        min = abs(x1 - x2)
                        l_min = l
                Mean_abacus_reduced_2[k] = Mean_abacus[l_min]
            Mean_abacus_reduced_2 = np.log10(np.asarray(Mean_abacus_reduced_2))
            
            if (i == 0):
                subplot.set_ylabel('$N_{b}$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**3, 10**4)
                
                subplot.fill_between(x = X_abacus, y1 = Mean_abacus - Std_abacus, y2 = Mean_abacus + Std_abacus, color = 'r', alpha = 0.2, label = 'Abacus range')
                subplot.fill_between(x = X_abacus, y1 = Mean_fidu - Std_fidu, y2 = Mean_fidu + Std_fidu, color = 'r', alpha = 0.6, label = 'Fiducial range')
                subplot.plot(10**X_b_planck_expected, 10**(Y_b_planck_expected), 'r', label = "Expectation")
                subplot.plot(10**X_b_planck_predicted, 10**(Y_b_planck_predicted), 'r--', label = 'Prediction')
                
                subplot.legend()
                
            else:
                subplot.set_ylabel('$\Delta N_{b} / <N_{b}>$')
                subplot.set_ylim(-0.3, 0.3)
                
                subplot.fill_between(x = X_abacus, y1 = - Std_abacus / Mean_abacus, y2 = Std_abacus / Mean_abacus, color = 'r', alpha = 0.2, label = "Abacus range")
                subplot.fill_between(x = X_abacus, y1 = (Mean_fidu - Std_fidu - Mean_abacus) / Mean_abacus, y2 = (Mean_fidu + Std_fidu - Mean_abacus) / Mean_abacus, color = 'r', alpha = 0.6, label = "Fiducial range")
                subplot.plot(10**X_b_planck_expected, (10**(Y_b_planck_expected) - 10**(Mean_abacus_reduced)) / 10**(Mean_abacus_reduced), 'r', label = "Expectation")
                subplot.plot(10**X_b_planck_predicted, (10**(Y_b_planck_predicted) - 10**(Mean_abacus_reduced_2)) / 10**(Mean_abacus_reduced_2), 'r--', label = 'Prediction')
                
                subplot.legend()
                
        else:
            subplot.set_xlabel('$s$')
            subplot.set_xlim(0.2, 0.7)
            
            X_abacus = dict['X_s'][0]
            Mean_abacus = np.asarray([0 for k in range(np.shape(dict['X_s'][0])[0])])
            Std_abacus = np.asarray([0 for k in range(np.shape(dict['X_s'][0])[0])])
            for k in range(n_simulations):
                New = []
                for x1 in X_abacus:
                    min = 10
                    l_min = 0
                    for l in range(np.shape(X_abacus)[0]):
                        x2 = dict['X_s'][k][l]
                        if (abs(x1 - x2) < min):
                            min = abs(x1 - x2)
                            l_min = l
                    try:
                        New.append((dict['Y_s'][k][l_min - 1] + dict['Y_s'][k][l_min] + dict['Y_s'][k][l_min + 1]) / 3)
                    except:
                        New.append(dict['Y_s'][k][l_min])
                New = np.asarray(New)
                
                Mean_old = Mean_abacus.copy()
                Std_old = Std_abacus.copy()
                
                Mean_new = (k * Mean_old + New) / (k + 1)
                Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + New**2) / (k + 1) - Mean_new**2)
                
                Mean_abacus = Mean_new.copy()
                Std_abacus = Std_new.copy()
            
            Mean_fidu = np.asarray([0 for k in range(np.shape(dict['X_s'][0])[0])])
            Std_fidu = np.asarray([0 for k in range(np.shape(dict['X_s'][0])[0])])
            for k in range(n_fiducial):
                New = []
                for x1 in X_abacus:
                    min = 10
                    l_min = 0
                    for l in range(np.shape(X_abacus)[0]):
                        x2 = dict['X_s'][k + n_simulations][l]
                        if (abs(x1 - x2) < min):
                            min = abs(x1 - x2)
                            l_min = l
                    try:
                        New.append((dict['Y_s'][k + n_simulations][l_min - 1] + dict['Y_s'][k + n_simulations][l_min] + dict['Y_s'][k + n_simulations][l_min + 1]) / 3)
                    except:
                        New.append(dict['Y_s'][k + n_simulations][l_min])
                New = np.asarray(New)
                    
                Mean_old = Mean_fidu.copy()
                Std_old = Std_fidu.copy()
                
                Mean_new = (k * Mean_old + New) / (k + 1)
                Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + New**2) / (k + 1) - Mean_new**2)
                
                Mean_fidu = Mean_new.copy()
                Std_fidu = Std_new.copy()
            
            Mean_abacus_reduced = np.asarray([0 for i in range(np.shape(X_s_planck_expected)[0])])
            for k in range(np.shape(Mean_abacus_reduced)[0]):
                l_min = 0
                min = 10
                x1 = X_s_planck_expected[k]
                for l in range(np.shape(X_abacus)[0]):
                    x2 = X_abacus[l]
                    if (abs(x1 - x2) < min):
                        min = abs(x1 - x2)
                        l_min = l
                Mean_abacus_reduced[k] = Mean_abacus[l_min]
            Mean_abacus_reduced = np.log10(np.asarray(Mean_abacus_reduced))
            
            Mean_abacus_reduced_2 = np.asarray([0 for i in range(np.shape(X_s_planck_predicted)[0])])
            for k in range(np.shape(Mean_abacus_reduced)[0]):
                l_min = 0
                min = 10
                x1 = X_s_planck_predicted[k]
                for l in range(np.shape(X_abacus)[0]):
                    x2 = X_abacus[l]
                    if (abs(x1 - x2) < min):
                        min = abs(x1 - x2)
                        l_min = l
                Mean_abacus_reduced_2[k] = Mean_abacus[l_min]
            Mean_abacus_reduced_2 = np.log10(np.asarray(Mean_abacus_reduced_2))
            
            if (i == 0):
                subplot.set_ylabel('$N_{s}$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**3, 1.1 * 10**4)
                
                subplot.fill_between(x = X_abacus, y1 = Mean_abacus - Std_abacus, y2 = Mean_abacus + Std_abacus, color = 'y', alpha = 0.2, label = 'Abacus range')
                subplot.fill_between(x = X_abacus, y1 = Mean_fidu - Std_fidu, y2 = Mean_fidu + Std_fidu, color = 'y', alpha = 0.6, label = 'Fiducial range')
                subplot.plot(X_s_planck_expected, 10**(Y_s_planck_expected), 'y', label = "Expectation")
                subplot.plot(X_s_planck_predicted, 10**(Y_s_planck_predicted), 'y--', label = 'Prediction')
                
                subplot.legend()
                
            else:
                subplot.set_ylabel('$\Delta N_{s} / <N_{s}>$')
                subplot.set_ylim(-0.1, 0.1)
                
                subplot.fill_between(x = X_abacus, y1 = - Std_abacus / Mean_abacus, y2 = Std_abacus / Mean_abacus, color = 'y', alpha = 0.2, label = "Abacus range")
                subplot.fill_between(x = X_abacus, y1 = (Mean_fidu - Std_fidu - Mean_abacus) / Mean_abacus, y2 = (Mean_fidu + Std_fidu - Mean_abacus) / Mean_abacus, color = 'y', alpha = 0.6, label = "Fiducial range")
                subplot.plot(X_s_planck_expected, (10**(Y_s_planck_expected) - 10**(Mean_abacus_reduced)) / 10**(Mean_abacus_reduced), 'y', label = "Expectation")
                subplot.plot(X_s_planck_predicted, (10**(Y_s_planck_predicted) - 10**(Mean_abacus_reduced_2)) / 10**(Mean_abacus_reduced_2), 'y--', label = 'Prediction')
                
                subplot.legend()

plt.suptitle("Proof of performance : GP")
plt.show()