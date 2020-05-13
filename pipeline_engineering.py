## Imports
import numpy as np
import GPy as GPy # To study non-trivial kernels

import sys
import os
sys.path.append('/home/astro/magnan/Repository_Stage_3A')
import GP_tools as GP
# import data_set
os.chdir('/home/astro/magnan')

import matplotlib.pyplot as plt

## Importing the data
""" I thnink we should compute the data in another script, and find a way to save it. Here, we only load what's interesting i.e. X_data and Y_data """

## Setting up the kernels to study
""" Here we make a np.array(?, 1) named Kernels, containing all the kernels we want to study """

## Setting up the groups
n_groups = 30 # quite arbitrary
n_points = np.shape(X_data)[0]

List_groups = []
for i in range(n_groups):
    start_group = i * n_points // n_groups
    end_group = (i + 1) * n_points // n_groups
    
    X_test_group = X_data[start_group:end_group]
    X_data_group = np.concatenate((X_data[0:start_group], X_data[end_group:]))
    Y_test_group = Y_data[start_group:end_group]
    Y_data_group = np.concatenate((Y_data[0:start_group], Y_data[end_group:]))
    
    List_groups.append((X_data_group, Y_data_group, X_test_group, Y_test_group))

## Evaluating the kernels performances
Performances = []
dim_y = 3
Noise_std = np.identity(n = dim_y) """" To Be Defined """

for kernel in Kernels:
    mean = 0
    std = 0
    
    for i in range(n_groups):
        X_data_group, Y_data_group, X_test_group, Y_test_group = List_groups[i]
        gp = GP(X_data_group, Y_data_group, Noise_std = Noise_std)
        gp.initialise_kernel(kernel = kernel)
        gp.initialise_model()
        gp.optimize_model()
        
        performance_new = gp.compute_performance_on_tests(X_test = X_test_group, Y_test = Y_test_group, Noise_std_test = Noise_std)
        
        mean_old = mean
        std_old = std
        mean_new = (i * mean_old + performance_new) / (i + 1)
        std_new = np.sqrt((i * (std_old**2 + mean_old**2) + performance_new**2) / (i + 1) - mean_new**2)
        mean = mean_new
        std = std_new
    
    Performances.append((mean, std))
        

## Choosing the best kernel
n_kernels = len(Kernels)

fig = plt.figure()

figure.set_title("Performances of the different kernels")
figure.set_xlabel("Kernels")
figure.set_ylabel("Performance (arbitrary unit)")

for i in range(n_kernels):
    figure.errorbarbar(x = [Kernel_names[i]], y = Peformances[i][0], yerr = Performances[i][1], fmt = "o")

plt.show(block = True)