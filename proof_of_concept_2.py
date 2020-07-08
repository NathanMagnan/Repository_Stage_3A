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
target = 'C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/data_set_Abacus_2/data_set_Abacus_'

"""
d = 0->4
l = 4->16
b = 16->27
s = 27->32
"""

n_points_per_simulation_complete = 32
n_simulations = 41
n_patchy = 0

X_d = None
Y_d = None
X_l = None
Y_l = None
X_b = None
Y_b = None
X_s = None
Y_s = None

for i in range(n_simulations + n_patchy + 1):
    X_data_new = np.loadtxt(fname = str(target) + str(i) + "_X_data") # numpy array with fields h0, w0, ns, sigma8, omegaM, d/l/b/s
    Y_data_new = np.loadtxt( str(target) + str(i) + "_Y_data") # numpy array with field Nd/l/b/s
    
    if (i == 0):
        X_d = X_data_new[0 : 4, 0 : 6]
        Y_d = Y_data_new[0 : 4]
        X_l = X_data_new[4 : 16, 0 : 6]
        Y_l = Y_data_new[4 : 16]
        X_b = X_data_new[16 : 27, 0 : 6]
        Y_b = Y_data_new[16 : 27]
        X_s = X_data_new[27 : 32, 0 : 6]
        Y_s = Y_data_new[27 : 32]
    elif (i == n_simulations + 0):
        X_d = np.concatenate((X_data_new[0 : 4, 0:6], X_d))
        Y_d = np.concatenate((Y_data_new[0 : 4], Y_d))
        X_l = np.concatenate((X_data_new[4 : 13, 0:6], X_l))
        Y_l = np.concatenate((Y_data_new[4 : 13], Y_l))
        X_b = np.concatenate((X_data_new[13 : 21, 0:6], X_b))
        Y_b = np.concatenate((Y_data_new[13 : 21], Y_b))
        X_s = np.concatenate((X_data_new[21 : 26, 0:6], X_s))
        Y_s = np.concatenate((Y_data_new[21 : 26], Y_s))
    else:
        X_d = np.concatenate((X_data_new[0 : 4, 0:6], X_d))
        Y_d = np.concatenate((Y_data_new[0 : 4], Y_d))
        X_l = np.concatenate((X_data_new[4 : 16, 0:6], X_l))
        Y_l = np.concatenate((Y_data_new[4 : 16], Y_l))
        X_b = np.concatenate((X_data_new[16 : 27, 0:6], X_b))
        Y_b = np.concatenate((Y_data_new[16 : 27], Y_b))
        X_s = np.concatenate((X_data_new[27 : 32, 0:6], X_s))
        Y_s = np.concatenate((Y_data_new[27 : 32], Y_s))

Parameters_BigMD = np.reshape(X_d[: 4][0][0:5], (1, 5))
print(Parameters_BigMD)

X_d_BigMD = X_d[:4]
Y_d_BigMD = Y_d[:4]
X_l_BigMD = X_l[:9]
Y_l_BigMD = Y_l[:9]
X_b_BigMD = X_b[:8]
Y_b_BigMD = Y_b[:8]
X_s_BigMD = X_s[:5]
Y_s_BigMD = Y_s[:5]
Y_d_BigMD = np.reshape(Y_d_BigMD, (4, 1))
Y_l_BigMD = np.reshape(Y_l_BigMD, (9, 1))
Y_b_BigMD = np.reshape(Y_b_BigMD, (8, 1))
Y_s_BigMD = np.reshape(Y_s_BigMD, ( 5, 1))

X_d_data = X_d[4:]
Y_d_data = Y_d[4:]
X_l_data = X_l[9:]
Y_l_data = Y_l[9:]
X_b_data = X_b[8:]
Y_b_data = Y_b[8:]
X_s_data = X_s[5:]
Y_s_data = Y_s[5:]
Y_d_data = np.reshape(Y_d_data, (n_simulations * 4, 1))
Y_l_data = np.reshape(Y_l_data, (n_simulations * 12, 1))
Y_b_data = np.reshape(Y_b_data, (n_simulations * 11, 1))
Y_s_data = np.reshape(Y_s_data, (n_simulations * 5, 1))

print("data loaded")

## Loading the whole Abacus data for plotting
print("starting to load the data")

#target = "/home/astro/magnan/Repository_Stage_3A/Full_MST_stats_Abacus/MST_stats_Catalogue_"
target = "C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Full_MST_stats_Abacus_2/MST_stats_Catalogue_"

dict = {'X_d' : [], 'Y_d' : [], 'X_l' : [], 'Y_l' : [], 'X_b' : [], 'Y_b' : [], 'X_s' : [], 'Y_s' : []}

for i in range(n_simulations + n_patchy + 1):    
    X_d_a = np.loadtxt(str(target) + str(i) + "_X_d")
    Y_d_a = np.loadtxt(str(target) + str(i) + "_Y_d")
    dict['X_d'].append(X_d_a)
    dict['Y_d'].append(Y_d_a)
    
    X_l_a = np.loadtxt(str(target) + str(i) + "_X_l")
    Y_l_a = np.loadtxt(str(target) + str(i) + "_Y_l")
    dict['X_l'].append(X_l_a)
    dict['Y_l'].append(Y_l_a)
    
    X_b_a = np.loadtxt(str(target) + str(i) + "_X_b")
    Y_b_a = np.loadtxt(str(target) + str(i) + "_Y_b")
    dict['X_b'].append(X_b_a)
    dict['Y_b'].append(Y_b_a)
    
    X_s_a = np.loadtxt(str(target) + str(i) + "_X_s")
    Y_s_a = np.loadtxt(str(target) + str(i) + "_Y_s")
    dict['X_s'].append(X_s_a)
    dict['Y_s'].append(Y_s_a)

print("data fully loaded")

## Setting up the GPs
print("starting to define the Gps")

gp = GP.GP(X = [X_d_data, X_l_data, X_b_data, X_s_data], Y = [Y_d_data, Y_l_data, Y_b_data, Y_s_data], N_points_per_simu = [4, 12, 11, 5], Noise = [None, None, None, None], type_kernel = "Separated")

print("models defined")

## Optimising the hyperparameters - Gradient Descent
print("Starting to optimise the hyperparameters")

gp.optimize_models(optimizer = 'lbfgsb')
print("Hyperparameters optimised")

gp.print_models()

## Making a prediction
print("Starting to make a prediction")

X_BigMD_predicted, Y_BigMD_predicted, Cov = gp.compute_prediction(Parameters_BigMD)

X_d_BigMD_predicted = X_BigMD_predicted[0][0][:, 5]
X_l_BigMD_predicted = X_BigMD_predicted[0][1][:, 5]
X_b_BigMD_predicted = X_BigMD_predicted[0][2][:, 5]
X_s_BigMD_predicted = X_BigMD_predicted[0][3][:, 5]

Y_d_BigMD_predicted = np.reshape(Y_BigMD_predicted[0][0], (4,))
Y_l_BigMD_predicted = np.reshape(Y_BigMD_predicted[0][1], (12,))
Y_b_BigMD_predicted = np.reshape(Y_BigMD_predicted[0][2], (11,))
Y_s_BigMD_predicted = np.reshape(Y_BigMD_predicted[0][3], (5,))

Y_d_std_BigMD_predicted = np.sqrt(np.reshape(np.diag(Cov[0][0]), (1, 4)))
Y_l_std_BigMD_predicted = np.sqrt(np.reshape(np.diag(Cov[0][1]), (1, 12)))
Y_b_std_BigMD_predicted = np.sqrt(np.reshape(np.diag(Cov[0][2]), (1, 11)))
Y_s_std_BigMD_predicted = np.sqrt(np.reshape(np.diag(Cov[0][3]), (1, 5)))

print("Prediction done")

## Testing the quality of the predictions
print("Starting to test the quality of the predictions")
rms = gp.test_rms(X_test = [X_d_BigMD, X_l_BigMD, X_b_BigMD, X_s_BigMD], Y_test = [Y_d_BigMD, Y_l_BigMD, Y_b_BigMD, Y_s_BigMD])
print("RMS Planck : " + str(rms))

## Plot
print("Starting to plot the MST stats")

fig, axes = plt.subplots(nrows = 2, ncols = 4, figsize = (20, 10))
plt.subplots_adjust(hspace = 0.3, wspace = 0.4)

for j in range(4):
    for i in range(2):
        subplot = axes[i][j]
        
        if (j == 0):
            subplot.set_xlabel('$l$')
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
            
            Mean_BigMD = []
            for x1 in dict['X_d'][n_simulations + 0]:
                min = 10
                l_min = 0
                for l in range(np.shape(X_abacus)[0]):
                    x2 = X_abacus[l]
                    if (abs(x1 - x2) < min):
                        min = abs(x1 - x2)
                        l_min = l
                
                Mean_BigMD.append(Mean_abacus[l_min])
            Mean_BigMD = np.asarray(Mean_BigMD)
            
            Mean_GP = []
            for x1 in X_d_BigMD_predicted:
                min = 10
                l_min = 0
                for l in range(np.shape(X_abacus)[0]):
                    x2 = X_abacus[l] / 6
                    if (abs(x1 - x2) < min):
                        min = abs(x1 - x2)
                        l_min = l
                
                Mean_GP.append(Mean_abacus[l_min])
            Mean_GP = np.asarray(Mean_GP)
            
            if (i == 0):
                subplot.set_ylabel('$N_{d}$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**(-1), 10**0)
                
                subplot.fill_between(x = X_abacus, y1 = Mean_abacus - Std_abacus, y2 = Mean_abacus + Std_abacus, color = 'b', alpha = 0.2, label = 'Abacus range')
                subplot.plot(dict['X_d'][n_simulations + 0], dict['Y_d'][n_simulations + 0], 'b', label = "BigMD")
                subplot.plot(6 * X_d_BigMD_predicted, 10**Y_d_BigMD_predicted, 'b--', label = 'Prediction')
                
                subplot.legend()
                
            else:
                subplot.set_ylabel('$\Delta N_{d} / <N_{d}>$')
                subplot.set_ylim(-0.1, 0.1)
                
                subplot.fill_between(x = X_abacus, y1 = - Std_abacus / Mean_abacus, y2 = Std_abacus / Mean_abacus, color = 'b', alpha = 0.2, label = "Abacus range")
                subplot.plot(dict['X_d'][n_simulations + 0], (dict['Y_d'][n_simulations + 0] - Mean_BigMD) / Mean_BigMD, 'b', label = "BigMD")
                subplot.plot(6 * X_d_BigMD_predicted, (10**Y_d_BigMD_predicted - Mean_GP) / Mean_GP, 'b--', label = 'Prediction')
                
                subplot.legend()
        
        elif (j == 1):
            subplot.set_xlabel('$l$')
            subplot.set_xscale('log')
            subplot.set_xlim(2, 11)
            
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
            
            Mean_BigMD = []
            for x1 in dict['X_l'][n_simulations + 0]:
                min = 10
                l_min = 0
                for l in range(np.shape(X_abacus)[0]):
                    x2 = X_abacus[l]
                    if (abs(x1 - x2) < min):
                        min = abs(x1 - x2)
                        l_min = l
                
                Mean_BigMD.append(Mean_abacus[l_min])
            Mean_BigMD = np.asarray(Mean_BigMD)
            
            Mean_GP = []
            for x1 in X_l_BigMD_predicted:
                min = 10
                l_min = 0
                for l in range(np.shape(X_abacus)[0]):
                    x2 = np.log10(X_abacus[l])
                    if (abs(x1 - x2) < min):
                        min = abs(x1 - x2)
                        l_min = l
                
                Mean_GP.append(Mean_abacus[l_min])
            Mean_GP = np.asarray(Mean_GP)
            
            if (i == 0):
                subplot.set_ylabel('$N_{l}$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**(-1), 10**0)
                
                subplot.fill_between(x = X_abacus, y1 = Mean_abacus - Std_abacus, y2 = Mean_abacus + Std_abacus, color = 'g', alpha = 0.2, label = 'Abacus range')
                subplot.plot(dict['X_l'][n_simulations + 0], dict['Y_l'][n_simulations + 0], 'g', label = "BigMD")
                subplot.plot(10**X_l_BigMD_predicted, 10**Y_l_BigMD_predicted, 'g--', label = 'Prediction')
                
                subplot.legend()
                
            else:
                subplot.set_ylabel('$\Delta N_{l} / <N_{l}>$')
                subplot.set_ylim(-0.3, 0.3)
                
                subplot.fill_between(x = X_abacus, y1 = - Std_abacus / Mean_abacus, y2 = Std_abacus / Mean_abacus, color = 'g', alpha = 0.2, label = "Abacus range")
                subplot.plot(dict['X_l'][n_simulations + 0], (dict['Y_l'][n_simulations + 0] - Mean_BigMD) / Mean_BigMD, 'g', label = "BigMD")
                subplot.plot(10**X_l_BigMD_predicted, (10**Y_l_BigMD_predicted - Mean_GP) / Mean_GP, 'g--', label = 'Prediction')
                
                subplot.legend()
                
        elif (j == 2):
            subplot.set_xlabel('$b$')
            subplot.set_xscale('log')
            subplot.set_xlim(5, 100)
            
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
            
            Mean_BigMD = []
            for x1 in dict['X_b'][n_simulations + 0]:
                min = 10
                l_min = 0
                for l in range(np.shape(X_abacus)[0]):
                    x2 = X_abacus[l]
                    if (abs(x1 - x2) < min):
                        min = abs(x1 - x2)
                        l_min = l
                
                Mean_BigMD.append(Mean_abacus[l_min])
            Mean_BigMD = np.asarray(Mean_BigMD)
            
            Mean_GP = []
            for x1 in X_b_BigMD_predicted:
                min = 10
                l_min = 0
                for l in range(np.shape(X_abacus)[0]):
                    x2 = np.log10(X_abacus[l])
                    if (abs(x1 - x2) < min):
                        min = abs(x1 - x2)
                        l_min = l
                
                Mean_GP.append(Mean_abacus[l_min])
            Mean_GP = np.asarray(Mean_GP)
            
            if (i == 0):
                subplot.set_ylabel('$N_{b}$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**(-1), 10**0)
                
                #subplot.fill_between(x = X_abacus, y1 = Mean_abacus - Std_abacus, y2 = Mean_abacus + Std_abacus, color = 'r', alpha = 0.2, label = 'Abacus range')
                subplot.plot(X_abacus, Mean_abacus, color = 'k', label = 'Abacus Mean')
                subplot.plot(dict['X_b'][n_simulations + 0], dict['Y_b'][n_simulations + 0], 'r', label = "BigMD")
                #subplot.plot(10**X_b_BigMD_predicted, 10**Y_b_BigMD_predicted, 'r--', label = 'Prediction')
                
                subplot.legend()
                
            else:
                subplot.set_ylabel('$\Delta N_{b} / <N_{b}>$')
                subplot.set_ylim(-0.3, 0.3)
                
                subplot.fill_between(x = X_abacus, y1 = - Std_abacus / Mean_abacus, y2 = Std_abacus / Mean_abacus, color = 'r', alpha = 0.2, label = "Abacus range")
                subplot.plot(dict['X_b'][n_simulations + 0], (dict['Y_b'][n_simulations + 0] - Mean_BigMD) / Mean_BigMD, 'r', label = "BigMD")
                subplot.plot(10**X_b_BigMD_predicted, (10**Y_b_BigMD_predicted - Mean_GP) / Mean_GP, 'r--', label = 'Prediction')
                
                subplot.legend()
                
        elif (j == 3):
            subplot.set_xlabel('$s$')
            subplot.set_xlim(0.3, 0.8)
            
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
            
            Mean_BigMD = []
            for x1 in dict['X_s'][n_simulations + 0]:
                min = 10
                l_min = 0
                for l in range(np.shape(X_abacus)[0]):
                    x2 = X_abacus[l]
                    if (abs(x1 - x2) < min):
                        min = abs(x1 - x2)
                        l_min = l
                
                Mean_BigMD.append(Mean_abacus[l_min])
            Mean_BigMD = np.asarray(Mean_BigMD)
            
            Mean_GP = []
            for x1 in X_s_BigMD_predicted:
                min = 10
                l_min = 0
                for l in range(np.shape(X_abacus)[0]):
                    x2 = X_abacus[l]
                    if (abs(x1 - x2) < min):
                        min = abs(x1 - x2)
                        l_min = l
                
                Mean_GP.append(Mean_abacus[l_min])
            Mean_GP = np.asarray(Mean_GP)
            
            if (i == 0):
                subplot.set_ylabel('$N_{s}$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**(-1), 10**0)
                
                subplot.fill_between(x = X_abacus, y1 = Mean_abacus - Std_abacus, y2 = Mean_abacus + Std_abacus, color = 'y', alpha = 0.2, label = 'Abacus range')
                subplot.plot(dict['X_s'][n_simulations + 0], dict['Y_s'][n_simulations + 0], 'y', label = "BigMD")
                subplot.plot(X_s_BigMD_predicted, 10**Y_s_BigMD_predicted, 'y--', label = 'Prediction')
                
                subplot.legend()
                
            else:
                subplot.set_ylabel('$\Delta N_{s} / <N_{s}>$')
                subplot.set_ylim(-0.1, 0.1)
                
                subplot.fill_between(x = X_abacus, y1 = - Std_abacus / Mean_abacus, y2 = Std_abacus / Mean_abacus, color = 'y', alpha = 0.2, label = "Abacus range")
                subplot.plot(dict['X_s'][n_simulations + 0], (dict['Y_s'][n_simulations + 0] - Mean_BigMD) / Mean_BigMD, 'y', label = "BigMD")
                subplot.plot(X_s_BigMD_predicted, (10**Y_s_BigMD_predicted - Mean_GP) / Mean_GP, 'y--', label = 'Prediction')
                
                subplot.legend()

plt.suptitle("Proof of performance : GP")
plt.show()

#my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')
my_path = os.path.abspath('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Figures')
my_file = 'Proof_of_performance_BigMD'
my_file = os.path.join(my_path, my_file)
#plt.savefig(my_file)

print("results plotted and saved")

""" 
Possible explanation : I reduce the size of the Abacus simulation, reducing the number in each bin. Which :
1) makes the curves more noisy
2) makes the GP much less precise

If true :
1) BigMD curves shouldn't be that noisy
"""