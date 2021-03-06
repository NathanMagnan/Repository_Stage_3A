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

print("data loaded")

## Setting up the data and test groups
print("Starting to make the data and test groups")

n_groups = 10

List_groups = []
for i in range(n_groups):
   start_group = ((i * n_simulations) // n_groups)
   end_group =  (((i + 1) * n_simulations) // n_groups)
   
   X_d_test_a = X_d_data[start_group * 4 : end_group * 4]
   X_d_data_a = np.concatenate((X_d_data[0 : start_group * 4], X_d_data[end_group * 4 :]), 0)
   Y_d_test_a = Y_d_data[start_group * 4 : end_group * 4]
   Y_d_data_a = np.concatenate((Y_d_data[0 : start_group * 4], Y_d_data[end_group * 4 :]), 0)
   
   X_l_test_a = X_l_data[start_group * 5 : end_group * 5]
   X_l_data_a = np.concatenate((X_l_data[0 : start_group * 5], X_l_data[end_group * 5 :]), 0)
   Y_l_test_a = Y_l_data[start_group * 5 : end_group * 5]
   Y_l_data_a = np.concatenate((Y_l_data[0 : start_group * 5], Y_l_data[end_group * 5 :]), 0)
   
   X_b_test_a = X_b_data[start_group * 5 : end_group * 5]
   X_b_data_a = np.concatenate((X_b_data[0 : start_group * 5], X_b_data[end_group * 5 :]), 0)
   Y_b_test_a = Y_b_data[start_group * 5 : end_group * 5]
   Y_b_data_a = np.concatenate((Y_b_data[0 : start_group * 5], Y_b_data[end_group * 5 :]), 0)
   
   X_s_test_a = X_s_data[start_group * 5 : end_group * 5]
   X_s_data_a = np.concatenate((X_s_data[0 : start_group * 5], X_s_data[end_group * 5 :]), 0)
   Y_s_test_a = Y_s_data[start_group * 5 : end_group * 5]
   Y_s_data_a = np.concatenate((Y_s_data[0 : start_group * 5], Y_s_data[end_group * 5 :]), 0)
   
   List_groups.append(([X_d_data_a, X_l_data_a, X_b_data_a, X_s_data_a], [Y_d_data_a, Y_l_data_a, Y_b_data_a, Y_s_data_a], [X_d_test_a, X_l_test_a, X_b_test_a, X_s_test_a], [Y_d_test_a, Y_l_test_a, Y_b_test_a, Y_s_test_a]))

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

# Matern 5/2
kernel = GPy.kern.Matern52(6, active_dims = [0, 1, 2, 3, 4, 5])
Kernels.append(kernel)
Kernel_names.append('Matern52')

# Matern 5/2 ARD
kernel = GPy.kern.Matern52(6, active_dims = [0, 1, 2, 3, 4, 5], ARD = True)
Kernels.append(kernel)
Kernel_names.append('Matern52 anisotropic')

print("kernels defined")

## Evaluating the kernels performances
print("starting to evaluate the performances")

Mean = [[], [], [], []]
Std = [[], [], [], []]

for i in range(4):
    print("starting to work on MST stat" + str(i))
    
    for j in range(len(Kernels)):
        print("starting to work on kernel " + str(j))
        
        mean = 0
        std = 0
        
        for k in range(n_groups):
            kernel = Kernels[j].copy()
            print("kernel " + str(j) + " group " + str(k))
            
            # getting the right data and test groups
            X_data, Y_data, X_test, Y_test = List_groups[k]
            
            # creating the gaussian process and optimizing it
            gp = GP.GP(X = X_data, Y = Y_data, N_points_per_simu = [4, 5, 5, 5], Noise = [None, None, None, None], type_kernel = "Separated")
            gp.change_kernels(Stats = [i], New_kernels = [kernel])
            gp.optimize_models(optimizer = 'lbfgsb')
            
            # getting the performance of the gaussian process
            performance_new = gp.test_rms(X_test = X_test, Y_test = Y_test)
            print(performance_new)
            
            mean_old = mean
            std_old = std
            mean_new = (k * mean_old + performance_new) / (k + 1)
            std_new = np.sqrt((k * (std_old**2 + mean_old**2) + performance_new**2) / (k + 1) - mean_new**2)
            mean = mean_new
            std = std_new
        
        Mean[i].append(mean)
        Std[i].append(std)
        print("work done on kernel " + str(j))
    
print("performances successfully evaluated")
print(Mean)
print(Std)
print(Kernel_names)

## Plotting the results
print("starting to plot the results")

plt.title("$k$-fold cross validation : $d$")
plt.ylabel("Performance (a.u.)")
plt.errorbar(Kernel_names, Mean[0], Std[0], fmt = 'ko', ecolor = 'k')

#my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')
my_path = os.path.abspath('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Figures')
my_file = 'Find_best_kernel_coregionalized'
my_file = os.path.join(my_path, my_file)
#plt.savefig(my_file)
plt.show()

print("results plotted and saved")