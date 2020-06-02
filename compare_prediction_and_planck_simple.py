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

n_points_per_simulation_complete = 10
n_points_per_simulation = 9
n_simulations = 39

X_data_complete = np.loadtxt(str(target) + "_X_data_s") # numpy array with fields h0, w0, ns, sigma8, omegaM, ds -- 6 point per simu
Y_data_complete = np.loadtxt(str(target) + "_Y_data_s") # numpy array with field Nd

X_data = None
Y_data = None
for i in range(n_simulations + 2):
    if i == 0:
        X_data = X_data_complete[:][0 : 9]
        Y_data = Y_data_complete[0 : 9]
    else:
        X_data = np.concatenate((X_data_complete[:][i * n_points_per_simulation_complete + 0 : i * n_points_per_simulation_complete + 9], X_data))
        Y_data = np.concatenate((Y_data_complete[i * n_points_per_simulation_complete + 0 : i * n_points_per_simulation_complete + 9], Y_data))
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

## Finding the best optimiser
#print("looking for the best optimiser")

#gp.model.optimize('lbfgsb')
#gp.model.optimize('tnc')
#gp.model.optimize('scg')
#gp.model.optimize('simplex')
#gp.model.optimize('adadelta') # doesn't work
#gp.model.optimize('rprop') # doesn't work
#gp.model.optimize('adam') # doesn't work
# Opts = ['bfgs', 'lbfgsb', 'tnc', 'scg', 'simplex']
# Chi_2 = []
# for opt in Opts:
#     print(opt)
#     gp.model.optimize_restarts(num_restarts = 10, optimizer = opt, xtol = 10**(-6), ftol = 10**(-6), gtol = 10**(-6))
# 
#     h0, w0, ns, sigma8, omegaM = X_planck[5, 0:5]
#     Planck_parameters = np.reshape([h0, w0, ns, sigma8, omegaM], (1, 5))
#     Y_predicted, Cov = gp.compute_prediction(Planck_parameters)
#     
#     chi_2 = GP.chi_2(Y_model = Y_planck, noise_model = noise_data, Y_observation = Y_predicted, Noise_observations = Cov)
#     Chi_2.append(chi_2)
# 
# print(Chi_2)
# print("The best optimiser seems to be lbfgsb")

## Optimising the hyperparameters manually
print("Starting to optimise the hyperparameters")


h0, w0, ns, sigma8, omegaM = X_planck[0, 0:5]
Planck_parameters = np.reshape([h0, w0, ns, sigma8, omegaM], (1, 5))
Y_predicted, Cov = gp.compute_prediction(Planck_parameters)

def loss_function(X): # X is a 1 dimensional numpy array of size 7
    var, l0, l1, l2, l3, l4, l5 = X
    gp.model.rbf.variance = abs(var) # to constrain the hyperparameters to positive values
    gp.model.rbf.lengthscale[0] = abs(l0)
    gp.model.rbf.lengthscale[1] = abs(l1)
    gp.model.rbf.lengthscale[2] = abs(l2)
    gp.model.rbf.lengthscale[3] = abs(l3)
    gp.model.rbf.lengthscale[4] = abs(l4)
    gp.model.rbf.lengthscale[5] = abs(l5)
    
    Y_predicted, Cov = gp.compute_prediction(Planck_parameters)
    
    rms = gp.RMS(Y_model = Y_planck, Y_observation = Y_predicted)
    print("RMS : " + str(rms))
    return(rms)

x0 = np.asarray([1, 1, 1, 1, 1, 1, 1]) # initial point for the research # could use a lambda function to make random restarts
sigma0 = 1 #should be 1/4th of the search domain
cma.fmin(objective_function = loss_function, x0 = x0, sigma0 = sigma0, options={'maxfevals': 10**(4)})

print("Hyperparameters optimised")
print(gp.model.rbf.variance)
print(gp.model.rbf.lengthscale)
print(gp.model.Gaussian_noise.variance)

Y_predicted, Cov = gp.compute_prediction(Planck_parameters)
rms = gp.RMS(Y_model = Y_planck, Y_observation = Y_predicted)
print("RMS Planck : " + str(rms))

h0_2, w0_2, ns_2, sigma8_2, omegaM_2 = X_other[0, 0:5]
Other_parameters = np.reshape([h0_2, w0_2, ns_2, sigma8_2, omegaM_2], (1, 5))
Y_predicted_2, Cov_2 = gp.compute_prediction(Other_parameters)
rms_2 = gp.RMS(Y_model = Y_other, Y_observation = Y_predicted_2)
print("RMS Other : " + str(rms_2))

## Optimizing the hyperparameters properly
# print("starting to optimize the hyperparameters")
# 
# gp.model.optimize_restarts(num_restarts = 10, optimizer = 'lbfgsb', gtol = 10**(-15))
# 
# h0, w0, ns, sigma8, omegaM = X_planck[0, 0:5]
# Planck_parameters = np.reshape([h0, w0, ns, sigma8, omegaM], (1, 5))
# Y_predicted, Cov = gp.compute_prediction(Planck_parameters)
# rms = gp.RMS(Y_model = Y_planck, Y_observation = Y_predicted)
# print("RMS Planck : " + str(rms))
# 
# h0_2, w0_2, ns_2, sigma8_2, omegaM_2 = X_other[0, 0:5]
# Other_parameters = np.reshape([h0_2, w0_2, ns_2, sigma8_2, omegaM_2], (1, 5))
# Y_predicted_2, Cov_2 = gp.compute_prediction(Other_parameters)
# rms_2 = gp.RMS(Y_model = Y_other, Y_observation = Y_predicted_2)
# print("RMS 39 : " + str(rms_2))
# 
# print("hyperparameters optimized")
# print(gp.model.rbf.variance)
# print(gp.model.rbf.lengthscale)
# print(gp.model.Gaussian_noise.variance)

## Drawing the histograms
print("Starting to plot the histograms")

figure = plt.figure()
ax = figure.gca()

ax.set_title("Comparison between expected and predicted histogram ($l$)")
ax.set_xlabel("$l$")
ax.set_ylabel("$\log{N_{l}}$")

D = X_planck[:, 5]
#ax.errorbar(x = D, y = Y_predicted, yerr = [np.sqrt(Cov[i][i]) for i in range(6)], color = 'r', fmt = 'o', label = "Prediction")
ax.plot(D, Y_predicted, 'ro', label = "Prediction Planck")
ax.plot(D, Y_predicted_2, 'ko', label = "Prediction 39")
ax.errorbar(x = D, y = Y_planck, yerr = [noise_data for i in range(n_points_per_simulation)], color = 'g', fmt = 'o', label = "Expectation Planck")
ax.errorbar(x = D, y = Y_other, yerr = [noise_data for i in range(n_points_per_simulation)], color = 'b', fmt = 'o', label = "Expectation 39")
ax.legend()

#my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')
my_path = os.path.abspath('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Figures')
my_file = 'Comparison_between_prediction_and_expectation_l'
my_file = os.path.join(my_path, my_file)
#plt.savefig(my_file)
plt.show()

print("results plotted and saved")