## Imports
import numpy as np
import GPy as GPy
from matplotlib import use
use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)

import sys
import os
sys.path.append('/home/astro/magnan/Repository_Stage_3A')
import GP_tools as GP
os.chdir('/home/astro/magnan')

print("All imports successful")

## Getting the data set
print("starting to load the data")

target = "/home/astro/magnan/Repository_Stage_3A/data_set_Abacus"

X_data = np.loadtxt(str(target) + "_X_data") # numpy array with fields h0, w0, ns, sigma8, omegaM, (d, l, b or s), i -- 36 points per simu
Y_data = np.loadtxt(str(target) + "_Y_data") # numpy array with fields either Nd, Nl, Nb or Ns depending on the corresponding x

print("data loaded")

## Dividing the data set
print("starting to divide the data set")
noise = 0.01

X_planck = X_data[1440:1476]
Y_planck = Y_data[1440:1476]

h0_i, w0_i, ns_i, sigma8_i, omegaM_i = X_planck[35, 0:5]
h0_p = h0_i  *  (75 - 60) + 60 # we had normalized every parameter between 0 and 1 when creating the data set
w0_p = w0_i  *  ((-0.60) - (-1.40)) + (-1.40) # we had normalized every parameter between 0 and 1 when creating the data set
ns_p = ns_i  *  (0.995 - 0.920) + 0.920 # we had normalized every parameter between 0 and 1 when creating the data set
sigma8_p = sigma8_i  *  (1.04 - 0.64) + 0.64 # we had normalized every parameter between 0 and 1 when creating the data set
omegaM_p = omegaM_i  *  (0.375 - 0.250) + 0.250 # we had normalized every parameter between 0 and 1 when creating the data set
Planck_parameters = [h0_p, w0_p, ns_p, sigma8_p, omegaM_p]

X_training = X_data[0:1440]
Y_training = Y_data[0:1440]

print("data set divided")
print(np.shape(X_training))
print(np.shape(Y_training))
print(np.shape(X_planck))
print(np.shape(Y_planck))

## Defining the pipeline
print("starting to define the pipeline")

kernel_input = GPy.kern.RBF(6, active_dims = [0, 1, 2, 3, 4, 5], ARD = True) # the column 6 will be dealt with by the coregionalization
kernel_output = GPy.kern.Coregionalize(input_dim = 1, output_dim = 4, rank = 4) # rank 4 since there are 4 outputs
kernel = kernel_input ** kernel_output
kernel_name = "RBF anisotropic"

gp = GP.GP(X_training, Y_training, kernel = kernel, noise_data = noise)
gp.initialise_model()
gp.model.mul.rbf.lengthscale.constrain_bounded(0, 3)
print(gp.model)

print("pipeline defined")

## Training the pipeline
print("starting to train the pipeline")

gp.optimize_model()

print("pipeline trained")
print(gp.model.mul.rbf.variance)
print(gp.model.mul.rbf.lengthscale)
print(gp.model.mul.coregion.W)
print(gp.model.mul.coregion.kappa)
print(gp.model.mul.coregion.B)
print(gp.model.Gaussian_noise.variance)

## Defining the grids
print("starting to define the grids")

# Coordinates
H0 = np.linspace(60, 75, 100)
W0 = np.linspace((-1.40), (-0.60), 100)
Ns = np.linspace(0.920, 0.995, 100)
Sigma8 = np.linspace(0.64, 1.04, 100)
OmegaM = np.linspace(0.250, 0.375, 100)
Coordinates = [H0, W0, Ns, Sigma8, OmegaM]
Coordinates_names = ['$H_{0}$', '$w_{0}$', '$n_{s}$', '$\sigma_{8}$', '$\Omega_{M}$', '$\chi_{2}$']
Coordinates_units = ['$km / s / Mpc$', 'a.u.', 'a.u.', 'a.u.', 'a.u.', 'a.u.']

# Grids
Grids = {'name_x_axis' : [], 'name_y_axis' : [], 'unit_x_axis' : [], 'unit_y_axis' : [], 'x_axis' : [], 'y_axis' : [], 'grid' : [], 'chi2_grid' : [], 'grid_white' : [], 'grid_blue' : []}

for i in range(5):
    for j in range(i + 1):
        Grids['name_x_axis'].append(Coordinates_names[i])
        Grids['unit_x_axis'].append(Coordinates_units[i])
        Grids['x_axis'].append(Coordinates[i])
        
        if (j == i):
            Grids['name_y_axis'].append(Coordinates_names[5])
            Grids['unit_y_axis'].append(Coordinates_units[5])
            Grids['y_axis'].append([0])
        else:
            Grids['name_y_axis'].append(Coordinates_names[j])
            Grids['unit_y_axis'].append(Coordinates_units[j])
            Grids['y_axis'].append(Coordinates[j])
        
        X_axis = Grids['x_axis'][-1]
        Y_axis = Grids['y_axis'][-1]
        
        if (j != i):
            Grid = [[[0, 0, 0, 0, 0] for y in Y_axis] for x in X_axis]
            for a in range(len(X_axis)):
                for b in range(len(Y_axis)):
                    Grid[a][b][i] = X_axis[a]
                    Grid[a][b][j] = Y_axis[b]
                    for c in range(5):
                        if ((c != i) and (c != j)):
                            Grid[a][b][c] = Planck_parameters[c]
            Grid = np.asarray(Grid)
            Grid = np.reshape(Grid, (len(X_axis) * len(Y_axis), 5))
            
        else:
            Grid = [[0, 0, 0, 0, 0] for x in X_axis]
            for a in range(len(X_axis)):
                Grid[a][i] = X_axis[a]
                for b in range(5):
                    if (b != i):
                        Grid[a][b] = Planck_parameters[b]
        
        Grids['grid'].append(Grid)

print("grids defined")

## Making the predictions
print("starting to make predictions")

n_grids = len(Grids['name_x_axis'])

for i in range(n_grids):
	print("starting to work on grid " + str(i))
	n_points = np.shape(Grids['grid'][i])[0]
	
	Chi2_grid = np.zeros(shape = (n_points, 1))
	for a in range(n_points):
		if (a % 1000 == 0):
			print(100 * a / n_points)
		chi2 = 0
		h0_i, w0_i, ns_i, sigma8_i, omegaM_i = Grids['grid'][i][a]
		h0 = (h0_i - 60)  / (75 - 60) # we normalize every parameter to use the whole range from 0 to 1
		w0 = (w0_i - (-1.40)) / ((-0.60) - (-1.40)) # we normalize every parameter to use the whole range from 0 to 1
		ns = (ns_i - 0.920) / (0.995 - 0.920) # we normalize every parameter to use the whole range from 0 to 1
		sigma8 = (sigma8_i - 0.64) / (1.04 - 0.64) # we normalize every parameter to use the whole range from 0 to 1
		omegaM = (omegaM_i - 0.250) / (0.375 - 0.250) # we normalize every parameter to use the whole range from 0 to 1
		X_new = np.asarray([h0, w0, ns, sigma8, omegaM])
		X_new = np.reshape(X_new, (1, 5))
		
		Y_predicted, Cov = gp.compute_prediction(X_new = X_new)
		
		chi2 = GP.chi_2(Y_model = Y_planck, noise_model = noise, Y_observation = Y_predicted, Noise_observations = Cov)
		Chi2_grid[a] = chi2
	
	Grids['chi2_grid'].append(Chi2_grid)

print("predictions made")

## Finding the points with chi2 < 1.96
print("starting to look for the points with chi2 < 1.96")

for i in range(n_grids):
    X_axis = Grids['x_axis'][i]
    Y_axis = Grids['y_axis'][i]
    
    Chi2_grid = Grids['chi2_grid'][i]
    
    Grid_white = [] # will hold the points of the grid that have chi2 > 1.96
    Grid_blue = [] # will hold the points of the grid that have chi2 < 1.96
    
    n_x, n_y = np.shape(X_axis)[0], np.shape(Y_axis)[0]
    
    if (n_y > 1):
        for a in range(n_x):
            for b in range(n_y):
                n_point = a * n_y + b
                if (Chi2_grid[n_point] > 1.96):
                    Grid_white.append((X_axis[a], Y_axis[b]))
                else:
                    Grid_blue.append((X_axis[a], Y_axis[b]))
    else:
        for a in range(n_x):
            n_point = a
            if (Chi2_grid[n_point] > 1.96):
                Grid_white.append((X_axis[a], 1 / Chi2_grid[n_point][0]))
            else:
                Grid_blue.append((X_axis[a], 1 / Chi2_grid[n_point][0]))
    
    Grid_white = np.asarray(Grid_white)
    Grid_blue = np.asarray(Grid_blue)
    
    
    Grids['grid_white'].append(Grid_white)
    Grids['grid_blue'].append(Grid_blue)

print("points with chi2 > 1.96 found")

## Drawing
print("starting to plot the results")

K = [[0, 0, 0, 0, 0], [1, 2, 0, 0, 0], [3, 4, 5, 0, 0], [6, 7, 8, 9, 0], [10, 11, 12, 13, 14]]
target = "/home/astro/magnan/Repository_Stage_3A/data_figure_12/figure_12"

for i in range(5):
	for j in range(i+1):
		k = K[i][j]
		if (j == i):
			if (Grids['grid_white'][k] != []):
				np.savetxt(str(target) + "_grid_" + str(k) + "_X_white", Grids['grid_white'][k][:, 0])
				np.savetxt(str(target) + "_grid_" + str(k) + "_Y_white", Grids['grid_white'][k][:, 1])
			if (Grids['grid_blue'][k] != []):
				np.savetxt(str(target) + "_grid_" + str(k) + "_X_blue", Grids['grid_blue'][k][:, 0])
				np.savetxt(str(target) + "_grid_" + str(k) + "_Y_blue", Grids['grid_blue'][k][:, 1])
		else:
			if (Grids['grid_white'][k] != []):
				np.savetxt(str(target) + "_grid_" + str(k) + "_X_white", Grids['grid_white'][k][:, 0])
				np.savetxt(str(target) + "_grid_" + str(k) + "_Y_white", Grids['grid_white'][k][:, 1])
			if (Grids['grid_blue'][k] != []):
				np.savetxt(str(target) + "_grid_" + str(k) + "_X_blue", Grids['grid_blue'][k][:, 0])
				np.savetxt(str(target) + "_grid_" + str(k) + "_Y_blue", Grids['grid_blue'][k][:, 1])

print("results saved")