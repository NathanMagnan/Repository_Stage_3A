## Imports
import numpy as np
import GPy as GPy
import cma as cma
import os
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)

import sys
import os
sys.path.append('/home/astro/magnan/Repository_Stage_3A')
#sys.path.append('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A')
import GP_tools as GP
os.chdir('/home/astro/magnan')
#os.chdir('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A')

print("All imports successful")

## Importing the data
print("Connexion successfull")
print("starting to load the data")

target = "/home/astro/magnan/Repository_Stage_3A/data_set_Abacus/data_set_Abacus"
#target = 'C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/data_set_Abacus/data_set_Abacus'

"""
d = 0->5
l = 9->15
b = 17->25
s = 26->35
"""

n_points_per_simulation_complete = 36
n_simulations = 40

X_data_complete = np.loadtxt(str(target) + "_X_data_all") # numpy array with fields h0, w0, ns, sigma8, omegaM, ds -- 6 point per simu
Y_data_complete = np.loadtxt(fname = str(target) + "_Y_data_all") # numpy array with field Nd

for i in range(n_simulations * n_points_per_simulation_complete): # there are infinite values because of the log normalisation
    Y_data_complete[i] = max(Y_data_complete[i], 0)

X_d = None
Y_d = None
X_l = None
Y_l = None
X_b = None
Y_b = None
X_s = None
Y_s = None
for i in range(n_simulations + 1):
    if i == 0:
        X_d = X_data_complete[0 : 5, 0 : 6]
        Y_d = Y_data_complete[0 : 5]
        X_l = X_data_complete[10 : 15, 0 : 6]
        Y_l = Y_data_complete[10 : 15]
        X_b = X_data_complete[19 : 24, 0 : 6]
        Y_b = Y_data_complete[19 : 24]
        X_s = X_data_complete[28 : 33, 0 : 6]
        Y_s = Y_data_complete[28 : 33]
    else:
        X_d = np.concatenate((X_data_complete[i * n_points_per_simulation_complete + 0 : i * n_points_per_simulation_complete + 5, 0:6], X_d))
        Y_d = np.concatenate((Y_data_complete[i * n_points_per_simulation_complete + 0 : i * n_points_per_simulation_complete + 5], Y_d))
        X_l = np.concatenate((X_data_complete[i * n_points_per_simulation_complete + 10 : i * n_points_per_simulation_complete + 15, 0:6], X_l))
        Y_l = np.concatenate((Y_data_complete[i * n_points_per_simulation_complete + 10 : i * n_points_per_simulation_complete + 15], Y_l))
        X_b = np.concatenate((X_data_complete[i * n_points_per_simulation_complete + 19 : i * n_points_per_simulation_complete + 24, 0:6], X_b))
        Y_b = np.concatenate((Y_data_complete[i * n_points_per_simulation_complete + 19 : i * n_points_per_simulation_complete + 24], Y_b))
        X_s = np.concatenate((X_data_complete[i * n_points_per_simulation_complete + 28 : i * n_points_per_simulation_complete + 33, 0:6], X_s))
        Y_s = np.concatenate((Y_data_complete[i * n_points_per_simulation_complete + 28 : i * n_points_per_simulation_complete + 33], Y_s))
noise_data = 0

X_d_planck = X_d[(n_simulations) * 5 : (n_simulations + 1) * 5]
Y_d_planck = Y_d[(n_simulations) * 5 : (n_simulations + 1) * 5]
X_l_planck = X_l[(n_simulations) * 5 : (n_simulations + 1) * 5]
Y_l_planck = Y_l[(n_simulations) * 5 : (n_simulations + 1) * 5]
X_b_planck = X_b[(n_simulations) * 5 : (n_simulations + 1) * 5]
Y_b_planck = Y_b[(n_simulations) * 5 : (n_simulations + 1) * 5]
X_s_planck = X_s[(n_simulations) * 5 : (n_simulations + 1) * 5]
Y_s_planck = Y_s[(n_simulations) * 5 : (n_simulations + 1) * 5]
Y_d_planck_expected = np.reshape(Y_d_planck, (5, 1))
Y_b_planck_expected = np.reshape(Y_b_planck, (5, 1))
Y_l_planck_expected = np.reshape(Y_l_planck, (5, 1))
Y_s_planck_expected = np.reshape(Y_s_planck, (5, 1))

X_d_data = X_d[0 : (n_simulations) * 5]
Y_d_data = Y_d[0 : (n_simulations) * 5]
X_l_data = X_l[0 : (n_simulations) * 5]
Y_l_data = Y_l[0 : (n_simulations) * 5]
X_b_data = X_b[0 : (n_simulations) * 5]
Y_b_data = Y_b[0 : (n_simulations) * 5]
X_s_data = X_s[0 : (n_simulations) * 5]
Y_s_data = Y_s[0 : (n_simulations) * 5]
Y_d_data = np.reshape(Y_d_data, (n_simulations * 5, 1))
Y_b_data = np.reshape(Y_b_data, (n_simulations * 5, 1))
Y_l_data = np.reshape(Y_l_data, (n_simulations * 5, 1))
Y_s_data = np.reshape(Y_s_data, (n_simulations * 5, 1))

print("data loaded")

## Setting up the GPs
print("starting to define the Gps")

gp = GP.GP(X = [X_d_data, X_l_data, X_b_data, X_s_data], Y = [Y_d_data, Y_l_data, Y_b_data, Y_s_data], N_points_per_simu = [5, 5, 5, 5], Noise = [None, None, None, None], type_kernel = "Separated")

print("models defined")

## Optimising the hyperparameters - Gradient Descent
print("Starting to optimise the hyperparameters")

gp.optimize_models(optimizer = 'lbfgsb')

print("Hyperparameters optimised")

gp.print_models()

## Defining the grids
print("starting to define the grids")

# Planck parameters
h0_i, w0_i, ns_i, sigma8_i, omegaM_i = X_d_planck[0][0 : 5]
h0_p = h0_i * (75 - 60) + 60 # to get the non-normalized value of the planck parameters
w0_p = w0_i * ((-0.60) - (-1.40)) + (-1.40)
ns_p = ns_i * (0.995 - 0.920) + 0.920
sigma8_p = sigma8_i * (1.04 - 0.64) + 0.64
omegaM_p = omegaM_i * (0.375 - 0.250) + 0.250
Planck_parameters = np.array([h0_p, w0_p, ns_p, sigma8_p, omegaM_p])

# Coordinates
H0 = np.linspace(60, 75, 50)
W0 = np.linspace((-1.40), (-0.60), 50)
Ns = np.linspace(0.920, 0.995, 50)
Sigma8 = np.linspace(0.64, 1.04, 50)
OmegaM = np.linspace(0.250, 0.375, 50)
Coordinates = [H0, W0, Ns, Sigma8, OmegaM]
Coordinates_names = ['$H_{0}$', '$w_{0}$', '$n_{s}$', '$\sigma_{8}$', '$\Omega_{M}$', '$\chi_{2}$']
Coordinates_units = ['$km / s / Mpc$', 'a.u.', 'a.u.', 'a.u.', 'a.u.', 'a.u.']

# Grids
Grids = {'name_x_axis' : [], 'name_y_axis' : [], 'unit_x_axis' : [], 'unit_y_axis' : [], 'x_axis' : [], 'y_axis' : [], 'grid' : [], 'chi2_grid' : [], 'grid_green' : [], 'grid_orange' : [], 'grid_red' : []}

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
            Grid = np.asarray(Grid)
            Grid = np.reshape(Grid, (len(X_axis), 5))
        
        Grids['grid'].append(Grid)

print("grids defined")

## Loading the whole Abacus data for analyzing the predictions
print("starting to load the data")

target = "/home/astro/magnan/Repository_Stage_3A/Full_MST_stats_Abacus/MST_stats_Catalogue_"
#target = "C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Full_MST_stats_Abacus/MST_stats_Catalogue_"

dict = {'X_d' : [], 'Y_d' : [], 'X_l' : [], 'Y_l' : [], 'X_b' : [], 'Y_b' : [], 'X_s' : [], 'Y_s' : []}

for i in range(41):    
    X_d_a = np.loadtxt(str(target) + str(i) + "_X_d")
    Y_d_a = np.loadtxt(str(target) + str(i) + "_Y_d")
    dict['X_d'].append(X_d_a)
    dict['Y_d'].append(Y_d_a)
    
    X_l_a = np.loadtxt(str(target) + str(i) + "_X_l")
    Y_l_a = np.loadtxt(str(target) + str(i) + "_Y_l")
    dict['X_l'].append(X_l_a)
    dict['Y_l'].append(Y_l_a)
    
    X_b_a = np.loadtxt(str(target) + str(i) + "_X_b")
    Y_b_a = np.loadtxt(str(target) + str(i) + "_Y_b")
    dict['X_b'].append(X_b_a)
    dict['Y_b'].append(Y_b_a)
    
    X_s_a = np.loadtxt(str(target) + str(i) + "_X_s")
    Y_s_a = np.loadtxt(str(target) + str(i) + "_Y_s")
    dict['X_s'].append(X_s_a)
    dict['Y_s'].append(Y_s_a)

print("data fully loaded")

## Making the predictions
print("starting to make predictions")

n_grids = len(Grids['name_x_axis'])

for i in range(n_grids):
    print("starting to work on grid " + str(i))
    n_points = np.shape(Grids['grid'][i])[0]
    
    Chi2_grid = np.zeros(shape = (n_points, 1))
    for j in range(n_points):
        if (j % 1000 == 0):
            print(100 * j / n_points)
        
        chi2 = 0
        
        # finding the cosmological parameters on which to make a prediction
        h0_i, w0_i, ns_i, sigma8_i, omegaM_i = Grids['grid'][i][j]
        h0 = (h0_i - 60)  / (75 - 60) # we normalize every parameter to use the whole range from 0 to 1
        w0 = (w0_i - (-1.40)) / ((-0.60) - (-1.40))
        ns = (ns_i - 0.920) / (0.995 - 0.920)
        sigma8 = (sigma8_i - 0.64) / (1.04 - 0.64)
        omegaM = (omegaM_i - 0.250) / (0.375 - 0.250)
        X_new = np.asarray([h0, w0, ns, sigma8, omegaM])
        X_new = np.reshape(X_new, (1, 5))
        
        # making the prediction
        X_predicted, Y_predicted, Cov = gp.compute_prediction(X_new)
        
        # searching for the expected value
        X_d_predicted = X_predicted[0][0]
        X_l_predicted = X_predicted[0][1]
        X_b_predicted = X_predicted[0][2]
        X_s_predicted = X_predicted[0][3]
        
        Y_d_expected = []
        Y_l_expected = []
        Y_b_expected = []
        Y_s_expected = []
        
        for k in range(5):
            min_d = 1
            l_min_d = 0
            min_l = 1
            l_min_l = 0
            min_b = 1
            l_min_b = 0
            min_s = 1
            l_min_s = 0
            
            xd = X_d_predicted[:, 5][k]
            xl = X_l_predicted[:, 5][k]
            xb = X_b_predicted[:, 5][k]
            xs = X_s_predicted[:, 5][k]
            
            for l in range(len(dict['X_d'][40])):
                x = dict['X_d'][40][l] / 6
                dist = abs(x - xd)
                if (dist < min_d):
                    l_min_d = l
                    min_d = dist
            for l in range(len(dict['X_l'][40])):
                x = np.log10(dict['X_l'][40][l])
                dist = abs(x - xl)
                if (dist < min_l):
                    l_min_l = l
                    min_l = dist
            for l in range(len(dict['X_b'][40])):
                x = np.log10(dict['X_b'][40][l])
                dist = abs(x - xb)
                if (dist < min_b):
                    l_min_b = l
                    min_b = dist
            for l in range(len(dict['X_s'][40])):
                x = dict['X_s'][40][l]
                dist = abs(x - xs)
                if (dist < min_s):
                    l_min_s = l
                    min_s = dist
            
            Y_d_expected.append(np.log10(dict['Y_d'][40][l_min_d]))
            Y_l_expected.append(np.log10(dict['Y_l'][40][l_min_l]))
            Y_b_expected.append(np.log10(dict['Y_b'][40][l_min_b]))
            Y_s_expected.append(np.log10(dict['Y_s'][40][l_min_s]))
    
        Y_d_expected = np.reshape(Y_d_expected, (5, 1))
        Y_l_expected = np.reshape(Y_l_expected, (5, 1))
        Y_b_expected = np.reshape(Y_b_expected, (5, 1))
        Y_s_expected = np.reshape(Y_s_expected, (5, 1))
        Y_expected = []
        Y_expected.append([Y_d_expected, Y_l_expected, Y_b_expected, Y_s_expected])
                
        chi2 = gp.likelihood_chi2(Y_model = Y_expected, Noise_model = [None], Y_observation = Y_predicted, Noise_observation = Cov)
        Chi2_grid[j] = chi2
    
    Grids['chi2_grid'].append(Chi2_grid)

print("predictions made")

## Finding the points with chi2 < 1.96
print("starting to make the bins on chi2")

for i in range(n_grids):
    X_axis = Grids['x_axis'][i]
    Y_axis = Grids['y_axis'][i]
    
    Chi2_grid = Grids['chi2_grid'][i]
    
    Grid_green = [] # will hold the points of the grid that have chi2 > 1.96
    Grid_orange = [] # will hold the points of the grid that have 1.96 < chi2 < 
    Grid_red = [] # will hold the points of the grid that have chi2 < 1.96
    
    n_x, n_y = np.shape(X_axis)[0], np.shape(Y_axis)[0]
    
    if (n_y > 1):
        for a in range(n_x):
            for b in range(n_y):
                n_point = a * n_y + b
                if (Chi2_grid[n_point] > 2.58):
                    Grid_green.append((X_axis[a], Y_axis[b]))
                elif ((Chi2_grid[n_point] > 1.96) and (Chi2_grid[n_point] < 2.58)):
                    Grid_orange.append((X_axis[a], Y_axis[b]))
                else:
                    Grid_red.append((X_axis[a], Y_axis[b]))
    else:
        for a in range(n_x):
            n_point = a
            #if (Chi2_grid[n_point] > 2.58):
            Grid_green.append((X_axis[a], 1 / Chi2_grid[n_point][0]))
            #elif ((Chi2_grid[n_point] > 1.96) and (Chi2_grid[n_point] < 2.58)):
            if (Chi2_grid[n_point] < 2.58):
                Grid_orange.append((X_axis[a], 1 / Chi2_grid[n_point][0]))
            #else:
            if (Chi2_grid[n_point] < 1.96):
                Grid_red.append((X_axis[a], 1 / Chi2_grid[n_point][0]))
    
    Grid_green = np.asarray(Grid_green)
    Grid_orange = np.asarray(Grid_orange)
    Grid_red = np.asarray(Grid_red)
    
    
    Grids['grid_green'].append(Grid_green)
    Grids['grid_orange'].append(Grid_orange)
    Grids['grid_red'].append(Grid_red)

print("bins on chi2 made")

## Drawing
print("starting to plot the results")

K = [[0, 0, 0, 0, 0], [1, 2, 0, 0, 0], [3, 4, 5, 0, 0], [6, 7, 8, 9, 0], [10, 11, 12, 13, 14]]
Coordinates_limits = [[60, 75], [-1.40, -0.60], [0.920, 0.995], [0.64, 1.04], [0.250, 0.375], [0, 1]]

fig, axes = plt.subplots(nrows = 5, ncols =5, figsize = (20, 20))
plt.subplots_adjust(hspace = 0.3, wspace = 0.4)

for i in range(5):
    for j in range(i+1):
        subplot = axes[i][j]
        k = K[i][j]
        
        Grid_green_X = []
        Grid_green_Y = []
        test_green = False
        Grid_orange_X = []
        Grid_orange_Y = []
        test_orange = False
        Grid_red_X = []
        Grid_red_Y = []
        test_red = False
        if (np.shape(Grids['grid_green'][k])[0] != 0):
            test_green = True
            Grid_green_X = Grids['grid_green'][k][:, 0]
            Grid_green_Y = Grids['grid_green'][k][:, 1]
        if (np.shape(Grids['grid_orange'][k])[0] != 0):
            test_orange = True
            Grid_orange_X = Grids['grid_orange'][k][:, 0]
            Grid_orange_Y = Grids['grid_orange'][k][:, 1]
        if (np.shape(Grids['grid_red'][k])[0] != 0):
            test_red = True
            Grid_red_X = Grids['grid_red'][k][:, 0]
            Grid_red_Y = Grids['grid_red'][k][:, 1]
        
        if (j == i):
            if (i == 4):
                subplot.set_xlabel(Grids['name_x_axis'][k] + ' [' + Grids['unit_x_axis'][k] + ']')
            subplot.set_ylabel(Grids['name_y_axis'][k] + ' [' + Grids['unit_y_axis'][k] + ']')
            subplot.set_xlim(left = Coordinates_limits[i][0], right = Coordinates_limits[i][1])
            subplot.set_ylim(bottom = Coordinates_limits[5][0], top = Coordinates_limits[5][1])
            if (test_green):
                subplot.plot(Grid_green_X, Grid_green_Y, 'green')
            if (test_orange):
                subplot.plot(Grid_orange_X, Grid_orange_Y, 'orange')
            if (test_red):
                subplot.plot(Grid_red_X, Grid_red_Y, 'red')
            subplot.axvline(x = Planck_parameters[i], color = 'black', linestyle = '--')
        else:
            if (i == 4):
                subplot.set_xlabel(Grids['name_y_axis'][k] + ' [' + Grids['unit_y_axis'][k] + ']')
            if (j == 0):
                subplot.set_ylabel(Grids['name_x_axis'][k] + ' [' + Grids['unit_x_axis'][k] + ']')
            subplot.set_xlim(left = Coordinates_limits[j][0], right = Coordinates_limits[j][1])
            subplot.set_ylim(bottom = Coordinates_limits[i][0], top = Coordinates_limits[i][1])
            if (test_green):
                subplot.scatter(Grid_green_Y, Grid_green_X, s = 1, c = 'green')
            if (test_orange):
                subplot.scatter(Grid_orange_Y, Grid_orange_X, s = 1, c = 'orange')
            if (test_red):
                subplot.scatter(Grid_red_Y, Grid_red_X, s = 1, c = 'red')
            subplot.axvline(x = Planck_parameters[j], color = 'black', linestyle = '--')
            subplot.axhline(y = Planck_parameters[i], color = 'black', linestyle = '--')
            
    for j in range(i+1, 5):
        subplot = axes[i][j]
        subplot.axis('off')

plt.suptitle("Sensitivity of the MST to cosmological parameters")

print("starting to save the results")
my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')
#my_path = os.path.abspath('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Figures')
my_file = 'Chi_2_test_Abacus_separated'
my_file = os.path.join(my_path, my_file)
#plt.savefig(my_file)
plt.show()

print("results plotted and saved")