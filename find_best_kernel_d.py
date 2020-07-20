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
   
   List_groups.append((X_d_data_a, Y_d_data_a, X_d_test_a, Y_d_test_a))

print("data and test groups defined")

## Setting up the kernels to study
print("starting to define the kernels")

Kernels = []
Kernel_names = []

# RBF
kernel = GPy.kern.RBF(6, active_dims = [0, 1, 2, 3, 4, 5])
Kernels.append(kernel)
Kernel_names.append('RBF')

# RBF
kernel = GPy.kern.RBF(6, active_dims = [0, 1, 2, 3, 4, 5], ARD = True)
Kernels.append(kernel)
Kernel_names.append('RBF anisotropic')

# Exponential
kernel = GPy.kern.Exponential(6, active_dims = [0, 1, 2, 3, 4, 5])
Kernels.append(kernel)
Kernel_names.append('Exponential')

# Matern 3/2
kernel = GPy.kern.Matern32(6, active_dims = [0, 1, 2, 3, 4, 5])
Kernels.append(kernel)
Kernel_names.append('Matern32')

# Matern 3/2 ARD
kernel = GPy.kern.Matern32(6, active_dims = [0, 1, 2, 3, 4, 5], ARD = True)
Kernels.append(kernel)
Kernel_names.append('Matern32 anisotropic')

# Matern 5/2
kernel = GPy.kern.Matern52(6, active_dims = [0, 1, 2, 3, 4, 5])
Kernels.append(kernel)
Kernel_names.append('Matern52')

# Matern 5/2 ARD
kernel = GPy.kern.Matern52(6, active_dims = [0, 1, 2, 3, 4, 5], ARD = True)
Kernels.append(kernel)
Kernel_names.append('Matern52 anisotropic')

print("kernels defined")

## Evaluating the kernels' errors
print("starting to evaluate the performances")

Errors = [[] for i in range(len(Kernel_names))]

for i in range(len(Kernel_names)):
    print("starting to work on kernel " + str(i))
    
    for j in range(n_groups):
        print("kernel " + str(i) + " group " + str(j))
        
        # getting the right kernel
        kernel = Kernels[i].copy()
        
        # getting the right data and test groups
        X_data, Y_data, X_test, Y_test = List_groups[j]
        
        # creating the gaussian process and optimizing it
        gp = GP.GP(X = X_data, Y = Y_data, n_points_per_simu = 4, Noise = None, make_covariance_matrix = False)
        gp.change_kernel(new_kernel = kernel, make_covariance_matrix = False)
        gp.optimize_model(optimizer = 'lbfgsb')
        
        # getting the errors
        errors = gp.compute_error_test(X_test, Y_test)
        
        # adding the errors to the lsit
        Errors[i].append(errors)
        
        print("work done on group " + str(j))
    
    print("work done on kernel " + str(i))
    
print("performances successfully evaluated")

## Printing the covariance matrices of the kernels
print("starting to calculate the covariance matrices")

for i in range(len(Kernel_names)):
    print("Covariance matrix for statistic d and kernel " + Kernel_names[i] + " :")
    errors = np.asarray(Errors[i]).T
    
    cov_new = [[0 for i in range(np.shape(errors)[0])] for j in range(np.shape(errors)[0])]
    for i in range(np.shape(errors)[0]):
        for j in range(np.shape(errors)[0]):
            sum = 0
            for k in range(np.shape(errors)[1]):
                sum += errors[i, k] * errors[j, k]
            cov_new[i][j] = sum / (np.shape(errors)[1] - 1)
    cov_new = np.asarray(cov_new)
    
    print(cov_new)

print("Covariance matrices printed")

## Plotting the performance
print("starting to plot the results")

RMS = []
Std = []

for i in range(len(Kernel_names)):
    errors = np.asarray(Errors[i])
    rms = np.sqrt(np.mean(errors**2))
    std = np.sqrt(np.mean(abs(errors**2 - rms**2)))
    RMS.append(rms)
    Std.append(std)

plt.title("$k$-fold cross validation : $d$")
plt.ylabel("Performance (a.u.)")
plt.yscale('log')
plt.errorbar(Kernel_names, RMS, Std, fmt = 'ko', ecolor = 'k')

#my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')
my_path = os.path.abspath('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Figures')
my_file = 'Find_best_kernel_d'
my_file = os.path.join(my_path, my_file)
#plt.savefig(my_file)
plt.show()

print("results plotted and saved")