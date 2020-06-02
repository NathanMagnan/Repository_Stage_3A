## Imports
import numpy as np
import GPy as GPy
import cma as cma
import matplotlib.pyplot as plt

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
target = 'C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/data_set_Abacus/data_set_Abacus'

"""
d = 0->5
l = 3->9
b = 1->9
s = 0->9
"""

n_points_per_simulation_complete = 6
n_points_per_simulation = 5
n_simulations = 39

X_data_complete = np.loadtxt(str(target) + "_X_data_d") # numpy array with fields h0, w0, ns, sigma8, omegaM, ds -- 6 point per simu
Y_data_complete = np.loadtxt(str(target) + "_Y_data_d") # numpy array with field Nd

X_data = None
Y_data = None
for i in range(n_simulations + 2):
    if i == 0:
        X_data = X_data_complete[:][0 : 5]
        Y_data = Y_data_complete[0 : 5]
    else:
        X_data = np.concatenate((X_data_complete[:][i * n_points_per_simulation_complete + 0 : i * n_points_per_simulation_complete + 5], X_data))
        Y_data = np.concatenate((Y_data_complete[i * n_points_per_simulation_complete + 0 : i * n_points_per_simulation_complete + 5], Y_data))
noise_data = 0

X_planck = X_data[(n_simulations + 1) * n_points_per_simulation : (n_simulations + 2) * n_points_per_simulation]
Y_planck = Y_data[(n_simulations + 1) * n_points_per_simulation : (n_simulations + 2) * n_points_per_simulation]

X_other = X_data[0 : n_points_per_simulation]
Y_other = Y_data[0 : n_points_per_simulation]

X_data = X_data[n_points_per_simulation : (n_simulations + 1) * n_points_per_simulation]
Y_data = Y_data[n_points_per_simulation : (n_simulations + 1) * n_points_per_simulation]
noise_data = 0

print("data loaded")

## Setting up the kernel to study
print("starting to define the kernel")

# RBF
kernel = GPy.kern.RBF(6, active_dims = [0, 1, 2, 3, 4, 5], ARD = True)
kernel.lengthscale = [0.1, 1, 2, 7, 3, 1]
Kernel_name = 'RBF anisotropic'

print("kernel defined")

gp = GP.GP(X_data = X_data, Y_data = Y_data, kernel = kernel, noise_data = noise_data, n_points_per_simulation = n_points_per_simulation)
gp.initialise_model()

## Optimising the hyperparameters - CMA
# print("Starting to optimise the hyperparameters")
# 
# h0, w0, ns, sigma8, omegaM = X_planck[0, 0:5]
# Planck_parameters = np.reshape([h0, w0, ns, sigma8, omegaM], (1, 5))
# 
# def loss_function(X): # X is a 1 dimensional numpy array of size 7
#     var, l0, l1, l2, l3, l4, l5 = X
#     gp.model.rbf.variance = abs(var) # to constrain the hyperparameters to positive values
#     gp.model.rbf.lengthscale[0] = abs(l0)
#     gp.model.rbf.lengthscale[1] = abs(l1)
#     gp.model.rbf.lengthscale[2] = abs(l2)
#     gp.model.rbf.lengthscale[3] = abs(l3)
#     gp.model.rbf.lengthscale[4] = abs(l4)
#     gp.model.rbf.lengthscale[5] = abs(l5)
#     
#     obj = gp.model.objective_function()
#     
#     #print("Obj : " + str(obj))
#     return(obj)
# 
# x0 = np.asarray([1, 1, 1, 1, 1, 1, 1]) # initial point for the research # we could use a lambda function to make random restarts
# sigma0 = 1 #should be 1/4th of the search domain
# cma.fmin(objective_function = loss_function, x0 = x0, sigma0 = sigma0, options={'maxfevals': 10**(4)})
# 
# print("Hyperparameters optimised")
# print(gp.model.rbf.variance)
# print(gp.model.rbf.lengthscale)
# print(gp.model.Gaussian_noise.variance)
# 
# obj = gp.model.objective_function() 
# print("Obj : " + str(obj))
# 
# Y_predicted, Cov = gp.compute_prediction(Planck_parameters)
# rms = gp.RMS(Y_model = Y_planck, Y_observation = Y_predicted)
# print("RMS Planck : " + str(rms))

## Optimising the hyperparameters - CMA
# print("Starting to optimise the hyperparameters")
# 
# h0, w0, ns, sigma8, omegaM = X_planck[0, 0:5]
# Planck_parameters = np.reshape([h0, w0, ns, sigma8, omegaM], (1, 5))
# 
# gp.model.optimize_restarts(num_restarts = 50, optimizer = 'lbfgsb', gtol = 10**(-15))
# 
# print("Hyperparameters optimised")
# print(gp.model.rbf.variance)
# print(gp.model.rbf.lengthscale)
# print(gp.model.Gaussian_noise.variance)
# 
# obj = gp.model.objective_function() 
# print("Obj : " + str(obj))
# 
# Y_predicted, Cov = gp.compute_prediction(Planck_parameters)
# rms = gp.RMS(Y_model = Y_planck, Y_observation = Y_predicted)
# print("RMS Planck : " + str(rms))

## Drawing the histograms
print("Starting to plot the histograms")

figure = plt.figure()
ax = figure.gca()

ax.set_title("Comparison between expected and predicted histogram ($d$)")
ax.set_xlabel("$d$")
ax.set_ylabel("$\log{N_{d}}$")

D = X_planck[:, 5]
#ax.errorbar(x = D, y = Y_predicted, yerr = [np.sqrt(Cov[i][i]) for i in range(6)], color = 'r', fmt = 'o', label = "Prediction")
ax.plot(D, Y_predicted, 'ro', label = "Prediction Planck")
ax.errorbar(x = D, y = Y_planck, yerr = [noise_data for i in range(n_points_per_simulation)], color = 'g', fmt = 'o', label = "Expectation Planck")
ax.errorbar(x = D, y = Y_other, yerr = [noise_data for i in range(n_points_per_simulation)], color = 'b', fmt = 'o', label = "Expectation 39")
ax.legend()

#my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')
my_path = os.path.abspath('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Figures')
my_file = 'CMA_optimisation_of_marginal_likelihood_d'
my_file = os.path.join(my_path, my_file)
#plt.savefig(my_file)
plt.show()

print("results plotted and saved")