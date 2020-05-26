## Imports
import numpy as np
import GPy as GPy
import matplotlib.pyplot as plt

import sys
import os
sys.path.append('/home/astro/magnan/Repository_Stage_3A')
import GP_tools as GP
os.chdir('/home/astro/magnan')

print("All imports successful")

## Importing the data
print("Connexion successfull")
print("starting to load the data")

target = "/home/astro/magnan/Repository_Stage_3A/data_set_Abacus"

X_data = np.loadtxt(str(target) + "_X_data") # numpy array with fields h0, w0, ns, sigma8, omegaM, (d, l, b or s), i -- 36 points per simu
Y_data = np.loadtxt(str(target) + "_Y_data") # numpy array with fields either Nd, Nl, Nb or Ns depending on the corresponding x

X_data = X_data[0:1440] # we leave the planck simulation out
Y_data = Y_data[0:1440]

print("data loaded")

## Setting up the kernels to study
print("starting to define the kernels")

Kernels = []
Kernels_input = []
Kernel_names = []

""" pretty sure the lengthscale on axis 6 (d, l, b, s) will pose problem... """

# RBF
kernel_input = GPy.kern.RBF(6, active_dims = [0, 1, 2, 3, 4, 5]) # the column 6 will be dealt with by the coregionalization
Kernels_input.append(kernel_input)
Kernel_names.append('RBF isotropic')

# RBF
kernel_input = GPy.kern.RBF(6, active_dims = [0, 1, 2, 3, 4, 5], ARD = True) # the column 6 will be dealt with by the coregionalization
Kernels_input.append(kernel_input)
Kernel_names.append('RBF anisotropic')

# Exponential
kernel_input = GPy.kern.Exponential(6, active_dims = [0, 1, 2, 3, 4, 5]) # the column 6 will be dealt with by the coregionalization
Kernels_input.append(kernel_input)
Kernel_names.append('Exponential')

# Matern 3/2
kernel_input = GPy.kern.Matern32(6, active_dims = [0, 1, 2, 3, 4, 5]) # the column 6 will be dealt with by the coregionalization
Kernels_input.append(kernel_input)
Kernel_names.append('Matern32')

# Matern 5/2
kernel_input = GPy.kern.Matern52(6, active_dims = [0, 1, 2, 3, 4, 5]) # the column 6 will be dealt with by the coregionalization
Kernels_input.append(kernel_input)
Kernel_names.append('Matern52')

for kernel_input in Kernels_input:
   kernel_output = GPy.kern.Coregionalize(input_dim = 1, output_dim = 4, rank = 4) # rank 4 since there are 4 outputs
   kernel = kernel_input**kernel_output
   Kernels.append(kernel)

print("kernels defined")

## Setting up the data and test groups
n_points_per_simulation = 36
n_groups = 10
n_simulations = np.shape(X_data)[0] // n_points_per_simulation

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
       gp.optimize_model()
       
       # printing results of optimisation
       if (k == 1):
          print(gp.model.mul.rbf.variance)
          print(gp.model.mul.rbf.lengthscale)
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