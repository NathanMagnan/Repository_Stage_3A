""" need to modify GP_tools """
## Imports
import numpy as np
import GPy as GPy
import matplotlib.pyplot as plt

import sys
import os
sys.path.append('/home/astro/magnan/Repository_Stage_3A')
import GP_tools_d as GP
os.chdir('/home/astro/magnan')

print("All imports successful")

## Importing the data
print("Connexion successfull")
print("starting to load the data")

target = "/home/astro/magnan/Repository_Stage_3A/data_set_Abacus/data_set_Abacus"

X_data = np.loadtxt(str(target) + "_X_data_d") # numpy array with fields h0, w0, ns, sigma8, omegaM, ds -- 6 point per simu
Y_data = np.loadtxt(str(target) + "_Y_data_d") # numpy array with field Nd

X_data = X_data[0:240] # we leave the planck simulation out
Y_data = Y_data[0:240]

print("data loaded")

## Setting up the kernels to study
print("starting to define the kernels")

Kernels = []
Kernel_names = []

# RBF
kernel = GPy.kern.RBF(6, active_dims = [0, 1, 2, 3, 4, 5])
Kernels.append(kernel)
Kernel_names.append('RBF isotropic')

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

# RBF
kernel = GPy.kern.RBF(6, active_dims = [0, 1, 2, 3, 4, 5], ARD = True)
Kernels.append(kernel)
Kernel_names.append('RBF anisotropic')

# RBF
kernel = GPy.kern.RBF(6, active_dims = [0, 1, 2, 3, 4, 5], ARD = True)
Kernels.append(kernel)
Kernel_names.append('RBF anisotropic bounded')

# RBF
kernel = GPy.kern.RBF(6, active_dims = [0, 1, 2, 3, 4, 5], ARD = True)
Kernels.append(kernel)
Kernel_names.append('RBF anisotropic with prior')

# RBF
kernel = GPy.kern.RBF(6, active_dims = [0, 1, 2, 3, 4, 5], ARD = True)
Kernels.append(kernel)
Kernel_names.append('RBF anisotropic sgc')

print("kernels defined")

## Setting up the data and test groups
n_points_per_simulation = 6
n_simulations = 40
n_groups = 10

List_groups = []
for i in range(n_groups):
   start_group = (i * n_simulations // n_groups) * n_points_per_simulation
   end_group =  ((i + 1) * n_simulations // n_groups) * n_points_per_simulation
   
   X_test_group = X_data[start_group:end_group]
   X_data_group = np.concatenate((X_data[0:start_group], X_data[end_group:]), 0)
   Y_test_group = Y_data[start_group:end_group]
   Y_data_group = np.concatenate((Y_data[0:start_group], Y_data[end_group:]), 0)
   
   List_groups.append((X_data_group, Y_data_group, X_test_group, Y_test_group))

print("data and test groups defined")

## Evaluating the kernels performances
print("starting to evaluate the performances")

Performances = []

for k in range(len(Kernels)):
    print("starting to work on kernel " + str(k))
    
    kernel = Kernels[k]
    mean = 0
    std = 0
    
    for i in range(n_groups):
        print("kernel " + str(k) + " group " + str(i))
        
        # getting the right data and test groups
        X_data_group, Y_data_group, X_test_group, Y_test_group = List_groups[i]
        
        # modelling the noise
        noise_data = 0.01
        noise_test = 0.01
        
        # creating the gaussian process and optimizing it
        gp = GP.GP(X_data_group, Y_data_group, kernel = kernel, noise_data = noise_data)
        gp.initialise_model()
        
        if (k <= 4):
            gp.optimize_model()
        if (k == 5):
            gp.model.rbf.lengthscale.constrain_bounded(0, 3)
            gp.optimize_model()
        if (k == 6):
            gp.model.rbf.lengthscale.set_prior(GPy.priors.Gamma(1, 1))
            gp.optimize_model()
        if (k == 7):
            gp.optimize_model('scg')
        
        # printing results of optimisation
        if (k >= 4):
            print(gp.model.rbf.variance)
            print(gp.model.rbf.lengthscale)
            print(gp.model.Gaussian_noise.variance)
        
        # getting the performance of the gaussian process
        performance_new = gp.compute_performance_on_tests(X_test = X_test_group, Y_test = Y_test_group, noise_test = noise_test)
        
        mean_old = mean
        std_old = std
        mean_new = (i * mean_old + performance_new) / (i + 1)
        std_new = np.sqrt((i * (std_old**2 + mean_old**2) + performance_new**2) / (i + 1) - mean_new**2)
        mean = mean_new
        std = std_new
    
    Performances.append((mean, std))
    print("work done on kernel " + str(k))

print("performances successfully evaluated")
print(Performances)
print(Kernel_names)