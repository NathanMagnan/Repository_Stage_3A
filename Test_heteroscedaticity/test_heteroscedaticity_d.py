## Imports
import numpy as np
import GPy as GPy
import os
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)

import sys
import os
#sys.path.append('/home/astro/magnan/Repository_stage_3A')
sys.path.append('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_stage_3A')
import GP_tools_simple as GP
#os.chdir('/home/astro/magnan')
os.chdir('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_stage_3A')

print("All imports successful")

## Importing the training data
print("Connexion successfull")
print("starting to load the data")

#target = "/home/astro/magnan/Repository_stage_3A/data_det_Abacus/data_set_Abacus"
target = 'C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_stage_3A/data_set_Abacus/data_set_Abacus_'

"""
d = 0->4
"""

n_points_per_simulation_complete = 36
n_simulations = 40
n_fiducial = 0

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

X_d_data = X_d[(n_fiducial) * 4:]
Y_d_data = Y_d[(n_fiducial) * 4:]
Y_d_data = np.reshape(Y_d_data, (n_simulations * 4, 1))

print("data loaded")

## Importing the noise data
print("starting to load the data")

#target = "/home/astro/magnan/Repository_stage_3A/Full_MST_stats_Abacus/MST_stats_Catalogue_"
target = "C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_stage_3A/Full_MST_stats_Abacus/MST_stats_Catalogue_"

dict = {'X_d' : [], 'Y_d' : []}
n_fiducial = 21

for i in range(n_fiducial + n_simulations):     
    X_d_a = np.loadtxt(str(target) + str(i) + "_X_d")
    Y_d_a = np.loadtxt(str(target) + str(i) + "_Y_d")
    dict['X_d'].append(X_d_a)
    dict['Y_d'].append(Y_d_a)

print("data fully loaded")

## Finding the noise on each training point
print("Starting to compute the noise")

X_d_abacus = X_d_data[0 : 4, 5]
Mean_d_fidu = np.asarray([0 for k in range(np.shape(X_d_abacus)[0])])
Std_d_fidu = np.asarray([0 for k in range(np.shape(X_d_abacus)[0])])
for k in range(n_fiducial):
    New = []
    for x1 in X_d_abacus:
        min = 10
        l_min = 0
        for l in range(np.shape(dict['X_d'][k + n_simulations])[0]):
            x2 = dict['X_d'][k + n_simulations][l] / 6
            if (abs(x1 - x2) < min):
                min = abs(x1 - x2)
                l_min = l
        try:
            New.append((dict['Y_d'][k + n_simulations][l_min - 1] + dict['Y_d'][k + n_simulations][l_min] + dict['Y_d'][k + n_simulations][l_min + 1]) / 3)
        except:
            New.append(dict['Y_d'][k + n_simulations][l_min])
    New = np.asarray(New)
        
    Mean_old = Mean_d_fidu.copy()
    Std_old = Std_d_fidu.copy()
    
    Mean_new = (k * Mean_old + New) / (k + 1)
    Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + New**2) / (k + 1) - Mean_new**2)
    
    Mean_d_fidu = Mean_new.copy()
    Std_d_fidu = Std_new.copy()

Std_d_fidu = Std_d_fidu / (np.log(10) * Mean_d_fidu) # renormalizing because y-axis is log-scaled

print("Noise computed")

## Setting up the data and test groups
print("Starting to make the data and test groups")

n_groups = 40

List_groups = []
for i in range(n_groups):
   start_group = ((i * n_simulations) // n_groups)
   end_group =  (((i + 1) * n_simulations) // n_groups)
   
   X_d_test_a = X_d_data[start_group * 4 : end_group * 4]
   X_d_data_a = np.concatenate((X_d_data[0 : start_group * 4], X_d_data[end_group * 4 :]), 0)
   Y_d_test_a = Y_d_data[start_group * 4 : end_group * 4]
   Y_d_data_a = np.concatenate((Y_d_data[0 : start_group * 4], Y_d_data[end_group * 4 :]), 0)
   Y_d_std_test_a = np.reshape(np.array([Std_d_fidu for j in range(start_group, end_group)]), (-1, 1))
   Y_d_std_data_a = np.reshape(np.array([Std_d_fidu for j in range(0, n_groups - (end_group - start_group))]), (-1, 1))
   
   List_groups.append((X_d_data_a, Y_d_data_a, Y_d_std_data_a, X_d_test_a, Y_d_test_a, Y_d_std_test_a))

print("data and test groups defined")

## Evaluating the GP's errors
print("starting to evaluate the performances")

Error_homoscedatic = [0 for j in range(n_groups)]
Error_heteroscedatic = [0 for j in range(n_groups)]

for j in range(n_groups):
    print(" group " + str(j))
    
    # getting the right data and test groups
    X_data, Y_data, Y_std_data, X_test, Y_test, Y_std_test = List_groups[j]
    
    # creating the gaussian process and optimizing it
    gp_homoscedatic = GP.GP(X = X_data, Y = Y_data, n_points_per_simu = 4, Noise = None, make_covariance_matrix = False)
    gp_homoscedatic.optimize_model(optimizer = 'lbfgsb')
    gp_heteroscedatic = GP.GP(X = X_data, Y = Y_data, n_points_per_simu = 4, Noise = Y_std_data, make_covariance_matrix = False)
    gp_heteroscedatic.optimize_model(optimizer = 'lbfgsb')
    
    # getting the errors
    error_homoscedatic = gp_homoscedatic.compute_chi2_test(X_test, Y_test, Y_std_test)
    error_heteroscedatic = gp_heteroscedatic.compute_chi2_test(X_test, Y_test, Y_std_test)
    
    # adding the errors to the lists
    Error_homoscedatic[j] = error_homoscedatic
    Error_heteroscedatic[j] = error_heteroscedatic
    
    print("work done on group " + str(j))
    
print("performances successfully evaluated")

## Plotting the performance
print("starting to plot the results")

plt.title("$k$-fold cross validation of heteroscedaticity : $d$")
plt.ylabel("Performance (a.u.)")
plt.xlabel("$\# group$")
plt.yscale('log')
plt.plot([i for i in range(n_groups)], Error_homoscedatic, 'b', label = 'Standard GP')
plt.plot([i for i in range(n_groups)], Error_heteroscedatic, 'r', label = 'Heteroscedatic GP')
plt.legend()

#my_path = os.path.abspath('/home/astro/magnan/Repository_stage_3A/Figures')
my_path = os.path.abspath('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_stage_3A/Test_heteroscedaticity/')
my_file = 'Test_heteroscedaticity_d'
my_file = os.path.join(my_path, my_file)
plt.savefig(my_file)
plt.show()

print("results plotted and saved")