## Imports
import numpy as np
import GPy as GPy
import os
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)

print("All imports successful")

## Getting the data set
print("starting to load the data")

target = "/home/astro/magnan/Repository_Stage_3A/data_set_Abacus"

X_data = np.loadtxt(str(target) + "_X_data") # numpy array with fields h0, w0, ns, sigma8, omegaM, (d, l, b or s), i -- 36 points per simu
Y_data = np.loadtxt(str(target) + "_Y_data") # numpy array with fields either Nd, Nl, Nb or Ns depending on the corresponding x

print("data loaded")

## Dividing the data set
print("starting to get the Planck parameters")
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

print("Planck parameters acquired")
print(Planck_parameters)

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
Grids = {'name_x_axis' : [], 'name_y_axis' : [], 'unit_x_axis' : [], 'unit_y_axis' : [], 'x_axis' : [], 'y_axis' : [], 'grid' : []}

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

## Drawing
print("starting to plot the results")

K = [[0, 0, 0, 0, 0], [1, 2, 0, 0, 0], [3, 4, 5, 0, 0], [6, 7, 8, 9, 0], [10, 11, 12, 13, 14]]
Coordinates_limits = [[60, 75], [-1.40, -0.60], [0.920, 0.995], [0.64, 1.04], [0.250, 0.375], [0, 1]]
target = "/home/astro/magnan/Repository_Stage_3A/data_figure_12/figure_12"

fig, axes = plt.subplots(nrows = 5, ncols =5, figsize = (20, 20))
plt.subplots_adjust(hspace = 0.3, wspace = 0.4)

for i in range(5):
	for j in range(i+1):
		subplot = axes[i][j]
		k = K[i][j]
		
		try:
			Grid_white_X = np.loadtxt(str(target) + "_grid_" + str(k) + "_X_white")
		except:
			Grid_white_X = []
		try:
			Grid_white_Y = np.loadtxt(str(target) + "_grid_" + str(k) + "_Y_white")
		except:
			Grid_white_Y = []
		try:
			Grid_blue_X = np.loadtxt(str(target) + "_grid_" + str(k) + "_X_blue")
		except:
			Grid_blue_X = []
		try:
			Grid_blue_Y = np.loadtxt(str(target) + "_grid_" + str(k) + "_Y_blue")
		except:
			Grid_blue_Y = []
		
		if (j == i):
			if (i == 4):
				subplot.set_xlabel(Grids['name_x_axis'][k] + ' [' + Grids['unit_x_axis'][k] + ']')
			subplot.set_ylabel(Grids['name_y_axis'][k] + ' [' + Grids['unit_y_axis'][k] + ']')
			subplot.set_xlim(left = Coordinates_limits[i][0], right = Coordinates_limits[i][1])
			subplot.set_ylim(bottom = Coordinates_limits[5][0], top = Coordinates_limits[5][1])
			if (Grid_white_X != []):
				subplot.plot(Grid_white_X, Grid_white_Y, 'r')
			if (Grid_blue_X != []):
				subplot.plot(Grid_blue_X, Grid_blue_Y, 'g')
			subplot.axvline(x = Planck_parameters[i], color = 'black', linestyle = '--')
		else:
			if (i == 4):
				subplot.set_xlabel(Grids['name_y_axis'][k] + ' [' + Grids['unit_y_axis'][k] + ']')
			if (j == 0):
				subplot.set_ylabel(Grids['name_x_axis'][k] + ' [' + Grids['unit_x_axis'][k] + ']')
			subplot.set_xlim(left = Coordinates_limits[j][0], right = Coordinates_limits[j][1])
			subplot.set_ylim(bottom = Coordinates_limits[i][0], top = Coordinates_limits[i][1])
			if (Grid_white_X != []):
				subplot.scatter(Grid_white_X, Grid_white_Y, s = 1, c = 'red')
			if (Grid_blue_X != []):
				subplot.scatter(Grid_blue_X, Grid_blue_Y, s = 1, c = 'green')
			subplot.axvline(x = Planck_parameters[j], color = 'black', linestyle = '--')
			subplot.axhline(y = Planck_parameters[i], color = 'black', linestyle = '--')
			
	for j in range(i+1, 5):
		subplot = axes[i][j]
		subplot.axis('off')

plt.suptitle("Sensitivity of the MST to cosmological parameters")
print("results plotted")

print("starting to save the results")
my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')
my_file = 'Sensitivity_MST_Abacus'
plt.savefig(os.path.join(my_path, my_file))
print("results saved")