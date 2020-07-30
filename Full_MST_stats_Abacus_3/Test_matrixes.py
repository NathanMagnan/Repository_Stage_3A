## Imports
import numpy as np
import pickle
import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)

#sys.path.append('/home/astro/magnan/Repository_Stage_3A')
sys.path.append('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A')
import GP_tools_simple as GP
#os.chdir('/home/astro/magnan')
os.chdir('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A')

print("all imports sucessfull")

## Loading the jackniffed histograms
print("starting to load the jackknifed histograms")

Histograms_by_simulations = [[] for i in range(21)]

for n_simu in range(21):
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # getting the number of the box
                n_box = 16 * i + 4 * j + k + 1
                
                # reading the histograms
                path = r'C:\Users\Nathan\Documents\D - X\C - Stages\Stage 3A\Repository_Stage_3A\Full_MST_stats_Abacus_3'
                file = 'MST_stats_Simulation_' + str(n_simu) + '_Box_' + str(n_box) + '.pkl'
                my_file = os.path.join(path, file)
                
                f = open(my_file, "rb")
                New_histogram = pickle.load(f)
                f.close()
                
                # adding the histograms to the right data sets
                Histograms_by_simulations[n_simu].append(New_histogram)

print("jackkniffed histograms loaded")

## Loading the full histograms
print("print starting to load the full histograms")

target = "C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Full_MST_stats_Abacus/MST_stats_Catalogue_"

Histograms_full = [{'X_d' : 0, 'Y_d' : 0, 'X_l' : 0, 'Y_l' : 0, 'X_b' : 0, 'Y_b' : 0, 'X_s' : 0, 'Y_s' : 0} for i in range(21)]

for i in range(21): 
    X_d = np.loadtxt(str(target) + str(i + 40) + "_X_d")
    Y_d = np.loadtxt(str(target) + str(i + 40) + "_Y_d")
    Histograms_full[i]['x_d'] = X_d
    Histograms_full[i]['y_d'] = Y_d
    
    X_l = np.loadtxt(str(target) + str(i + 40) + "_X_l")
    Y_l = np.loadtxt(str(target) + str(i + 40) + "_Y_l")
    Histograms_full[i]['x_l'] = X_l
    Histograms_full[i]['y_l'] = Y_l
    
    X_b = np.loadtxt(str(target) + str(i + 40) + "_X_b")
    Y_b = np.loadtxt(str(target) + str(i + 40) + "_Y_b")
    Histograms_full[i]['x_b'] = X_b
    Histograms_full[i]['y_b'] = Y_b
    
    X_s = np.loadtxt(str(target) + str(i + 40) + "_X_s")
    Y_s = np.loadtxt(str(target) + str(i + 40) + "_Y_s")
    Histograms_full[i]['x_s'] = X_s
    Histograms_full[i]['y_s'] = Y_s

print("full histograms loaded")

## Loading the GP's data
print("starting to load the GP's 'data")

#target = "/home/astro/magnan/Repository_Stage_3A/data_set_Abacus/data_set_Abacus"
target = 'C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/data_set_Abacus/data_set_Abacus_'

"""
d = 0->4
s = 28->33
"""

n_points_per_simulation_complete = 36
n_simulations = 40
n_fiducial = 21

X_d = None
Y_d = None
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
        X_s = X_data_new[28 : 33, 0 : 6]
        Y_s = Y_data_new[28 : 33]
    else:
        X_d = np.concatenate((X_data_new[0 : 4, 0:6], X_d))
        Y_d = np.concatenate((Y_data_new[0 : 4], Y_d))
        X_s = np.concatenate((X_data_new[28 : 33, 0:6], X_s))
        Y_s = np.concatenate((Y_data_new[28 : 33], Y_s))

X_d_planck = X_d[:(n_fiducial) * 4]
X_s_planck = X_s[:(n_fiducial) * 5]

X_d_data = X_d[(n_fiducial) * 4:]
Y_d_data = Y_d[(n_fiducial) * 4:]
X_s_data = X_s[(n_fiducial) * 5:]
Y_s_data = Y_s[(n_fiducial) * 5:]
Y_d_data = np.reshape(Y_d_data, (n_simulations * 4, 1))
Y_s_data = np.reshape(Y_s_data, (n_simulations * 5, 1))

print("GP data loaded")

## Setting up the data and test groups for the GP
print("Starting to make the data and test groups")

n_groups = 40

List_groups_d = []
List_groups_s = []

for i in range(n_groups):
   start_group = ((i * n_simulations) // n_groups)
   end_group =  (((i + 1) * n_simulations) // n_groups)
   
   X_d_test_a = X_d_data[start_group * 4 : end_group * 4]
   X_d_data_a = np.concatenate((X_d_data[0 : start_group * 4], X_d_data[end_group * 4 :]), 0)
   Y_d_test_a = Y_d_data[start_group * 4 : end_group * 4]
   Y_d_data_a = np.concatenate((Y_d_data[0 : start_group * 4], Y_d_data[end_group * 4 :]), 0)
   
   List_groups_d.append((X_d_data_a, Y_d_data_a, X_d_test_a, Y_d_test_a))
   
   X_s_test_a = X_s_data[start_group * 5 : end_group * 5]
   X_s_data_a = np.concatenate((X_s_data[0 : start_group * 5], X_s_data[end_group * 5 :]), 0)
   Y_s_test_a = Y_s_data[start_group * 5 : end_group * 5]
   Y_s_data_a = np.concatenate((Y_s_data[0 : start_group * 5], Y_s_data[end_group * 5 :]), 0)
   
   List_groups_s.append((X_s_data_a, Y_s_data_a, X_s_test_a, Y_s_test_a))

print("data and test groups defined")

## Choosing a common x axis

X_d = Histograms_full[0]['x_d']
X_l = Histograms_full[0]['x_l']
X_b = Histograms_full[0]['x_b']
X_s = Histograms_full[0]['x_s']

## Moving every histogram to this x axis
print("Starting to reposition every histogram")

# Repositionnings in Histograms_by_simulation
for i in range(21):
    for j in range(64):
        
        # d
        New_d = [0 for x1 in X_d]
        for k in range(np.shape(X_d)[0]):
            x1 = X_d[k]
            min = 10
            l_min = 0
            for l in range(np.shape(Histograms_by_simulations[i][j]['x_d'])[0]):
                x2 = Histograms_by_simulations[i][j]['x_d'][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            New_d[k] = Histograms_by_simulations[i][j]['y_d'][l_min]
        Histograms_by_simulations[i][j]['y_d'] = np.asarray(New_d)
        
        # l
        New_l = [0 for x1 in X_l]
        for k in range(np.shape(X_l)[0]):
            x1 = X_l[k]
            min = 10
            l_min = 0
            for l in range(np.shape(Histograms_by_simulations[i][j]['x_l'])[0]):
                x2 = Histograms_by_simulations[i][j]['x_l'][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            try:
                if (x1 > Histograms_by_simulations[i][j]['x_l'][l_min]):
                    x2a = Histograms_by_simulations[i][j]['x_l'][l_min]
                    x2b = Histograms_by_simulations[i][j]['x_l'][l_min+1]
                    a = (x2b - x1) / (x2b - x2a)
                    b = (x1 - x2a) / (x2b - x2a)
                    New_l[k] = a * Histograms_by_simulations[i][j]['y_l'][l_min] + b * Histograms_by_simulations[i][j]['y_l'][l_min + 1]
                else:
                    x2a = Histograms_by_simulations[i][j]['x_l'][l_min - 1]
                    x2b = Histograms_by_simulations[i][j]['x_l'][l_min]
                    a = (x2b - x1) / (x2b - x2a)
                    b = (x1 - x2a) / (x2b - x2a)
                    New_l[k] = a * Histograms_by_simulations[i][j]['y_l'][l_min - 1] + b * Histograms_by_simulations[i][j]['y_l'][l_min]
            except:
                New_l[k] = Histograms_by_simulations[i][j]['y_l'][l_min]
        Histograms_by_simulations[i][j]['y_l'] = np.asarray(New_l)
        
        # b
        New_b = [0 for x1 in X_b]
        for k in range(np.shape(X_b)[0]):
            x1 = X_b[k]
            min = 10
            l_min = 0
            for l in range(np.shape(Histograms_by_simulations[i][j]['x_b'])[0]):
                x2 = Histograms_by_simulations[i][j]['x_b'][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            try:
                if (x1 > Histograms_by_simulations[i][j]['x_b'][l_min]):
                    x2a = Histograms_by_simulations[i][j]['x_b'][l_min]
                    x2b = Histograms_by_simulations[i][j]['x_b'][l_min+1]
                    a = (x2b - x1) / (x2b - x2a)
                    b = (x1 - x2a) / (x2b - x2a)
                    New_b[k] = a * Histograms_by_simulations[i][j]['y_b'][l_min] + b * Histograms_by_simulations[i][j]['y_b'][l_min + 1]
                else:
                    x2a = Histograms_by_simulations[i][j]['x_b'][l_min - 1]
                    x2b = Histograms_by_simulations[i][j]['x_b'][l_min]
                    a = (x2b - x1) / (x2b - x2a)
                    b = (x1 - x2a) / (x2b - x2a)
                    New_b[k] = a * Histograms_by_simulations[i][j]['y_b'][l_min - 1] + b * Histograms_by_simulations[i][j]['y_b'][l_min]
            except:
                New_b[k] = Histograms_by_simulations[i][j]['y_b'][l_min]
        Histograms_by_simulations[i][j]['y_b'] = np.asarray(New_b)
        
        # s
        New_s = [0 for x1 in X_s]
        for k in range(np.shape(X_s)[0]):
            x1 = X_s[k]
            min = 10
            l_min = 0
            for l in range(np.shape(Histograms_by_simulations[i][j]['x_s'])[0]):
                x2 = Histograms_by_simulations[i][j]['x_s'][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            try:
                if (x1 > Histograms_by_simulations[i][j]['x_s'][l_min]):
                    x2a = Histograms_by_simulations[i][j]['x_s'][l_min]
                    x2b = Histograms_by_simulations[i][j]['x_s'][l_min+1]
                    a = (x2b - x1) / (x2b - x2a)
                    b = (x1 - x2a) / (x2b - x2a)
                    New_s[k] = a * Histograms_by_simulations[i][j]['y_s'][l_min] + b * Histograms_by_simulations[i][j]['y_s'][l_min + 1]
                else:
                    x2a = Histograms_by_simulations[i][j]['x_s'][l_min - 1]
                    x2b = Histograms_by_simulations[i][j]['x_s'][l_min]
                    a = (x2b - x1) / (x2b - x2a)
                    b = (x1 - x2a) / (x2b - x2a)
                    New_s[k] = a * Histograms_by_simulations[i][j]['y_s'][l_min - 1] + b * Histograms_by_simulations[i][j]['y_s'][l_min]
            except:
                New_s[k] = Histograms_by_simulations[i][j]['y_s'][l_min]
        Histograms_by_simulations[i][j]['y_s'] = np.asarray(New_s)

# Repositionning the full histograms
for i in range(21):
    
    # d
    New_d = [0 for x1 in X_d]
    for k in range(np.shape(X_d)[0]):
        x1 = X_d[k]
        min = 10
        l_min = 0
        for l in range(np.shape(Histograms_full[i]['x_d'])[0]):
            x2 = Histograms_full[i]['x_d'][l]
            if (abs(x1 - x2) < min):
                min = abs(x1 - x2)
                l_min = l
        New_d[k] = Histograms_full[i]['y_d'][l_min]
    Histograms_full[i]['y_d'] = np.asarray(New_d)
    
    # l
    New_l = [0 for x1 in X_l]
    for k in range(np.shape(X_l)[0]):
        x1 = X_l[k]
        min = 10
        l_min = 0
        for l in range(np.shape(Histograms_full[i]['x_l'])[0]):
            x2 = Histograms_full[i]['x_l'][l]
            if (abs(x1 - x2) < min):
                min = abs(x1 - x2)
                l_min = l
        try:
            if (x1 > Histograms_full[i]['x_l'][l_min]):
                x2a = Histograms_full[i]['x_l'][l_min]
                x2b = Histograms_full[i]['x_l'][l_min+1]
                a = (x2b - x1) / (x2b - x2a)
                b = (x1 - x2a) / (x2b - x2a)
                New_l[k] = a * Histograms_full[i]['y_l'][l_min] + b * Histograms_full[i]['y_l'][l_min + 1]
            else:
                x2a = Histograms_full[i]['x_l'][l_min - 1]
                x2b = Histograms_full[i]['x_l'][l_min]
                a = (x2b - x1) / (x2b - x2a)
                b = (x1 - x2a) / (x2b - x2a)
                New_l[k] = a * Histograms_full[i]['y_l'][l_min - 1] + b * Histograms_full[i]['y_l'][l_min]
        except:
            New_l[k] = Histograms_full[i]['y_l'][l_min]
    Histograms_full[i]['y_l'] = np.asarray(New_l)
    
    # b
    New_b = [0 for x1 in X_b]
    for k in range(np.shape(X_b)[0]):
        x1 = X_b[k]
        min = 10
        l_min = 0
        for l in range(np.shape(Histograms_full[i]['x_b'])[0]):
            x2 = Histograms_full[i]['x_b'][l]
            if (abs(x1 - x2) < min):
                min = abs(x1 - x2)
                l_min = l
        try:
            if (x1 > Histograms_full[i]['x_b'][l_min]):
                x2a = Histograms_full[i]['x_b'][l_min]
                x2b = Histograms_full[i]['x_b'][l_min+1]
                a = (x2b - x1) / (x2b - x2a)
                b = (x1 - x2a) / (x2b - x2a)
                New_b[k] = a * Histograms_full[i]['y_b'][l_min] + b * Histograms_full[i]['y_b'][l_min + 1]
            else:
                x2a = Histograms_full[i]['x_b'][l_min - 1]
                x2b = Histograms_full[i]['x_b'][l_min]
                a = (x2b - x1) / (x2b - x2a)
                b = (x1 - x2a) / (x2b - x2a)
                New_b[k] = a * Histograms_full[i]['y_b'][l_min - 1] + b * Histograms_full[i]['y_b'][l_min]
        except:
            New_b[k] = Histograms_full[i]['y_b'][l_min]
    Histograms_full[i]['y_b'] = np.asarray(New_b)
    
    # s
    New_s = [0 for x1 in X_s]
    for k in range(np.shape(X_s)[0]):
        x1 = X_s[k]
        min = 10
        l_min = 0
        for l in range(np.shape(Histograms_full[i]['x_s'])[0]):
            x2 = Histograms_full[i]['x_s'][l]
            if (abs(x1 - x2) < min):
                min = abs(x1 - x2)
                l_min = l
        try:
            if (x1 > Histograms_full[i]['x_s'][l_min]):
                x2a = Histograms_full[i]['x_s'][l_min]
                x2b = Histograms_full[i]['x_s'][l_min+1]
                a = (x2b - x1) / (x2b - x2a)
                b = (x1 - x2a) / (x2b - x2a)
                New_s[k] = a * Histograms_full[i]['y_s'][l_min] + b * Histograms_full[i]['y_s'][l_min + 1]
            else:
                x2a = Histograms_full[i]['x_s'][l_min - 1]
                x2b = Histograms_full[i]['x_s'][l_min]
                a = (x2b - x1) / (x2b - x2a)
                b = (x1 - x2a) / (x2b - x2a)
                New_s[k] = a * Histograms_full[i]['y_s'][l_min - 1] + b * Histograms_full[i]['y_s'][l_min]
        except:
            New_s[k] = Histograms_full[i]['y_s'][l_min]
    Histograms_full[i]['y_s'] = np.asarray(New_s)

print("Histograms repositionned")

## Finding the old noise matrix (DS only)
print('Starting to work on the old noise matrix')

# Concatenating d and s
Vectors = []
for i in range(21):
        New_d = Histograms_full[i]['y_d']
        New_s = Histograms_full[i]['y_s']
        New = np.concatenate((New_d, New_s), axis = 0)
        Vectors.append(New)
Vectors = np.asarray(Vectors).T

# Getting the covariance matrix
Cov_old = np.cov(Vectors, bias = False)

# Getting the mean, which will be more usefull later
Mean_old = np.mean(Vectors, axis = 1)

print('old noise matrix computed')

## Finding the jackknife noise matrix (DS only)
print('Starting to work on the jackknife noise matrix')

# Concatenating d and s
Vectors_by_simu = []
for i in range(21):
    Vectors = []
    for j in range(64):
        New_d = Histograms_by_simulations[i][j]['y_d']
        New_s = Histograms_by_simulations[i][j]['y_s']
        New = np.concatenate((New_d, New_s), axis = 0)
        Vectors.append(New)
    Vectors = np.asarray(Vectors).T
    Vectors_by_simu.append(Vectors)

# Getting the 21 covariance matrix
Covs_by_simu = []
for i in range(21):
    Vectors = Vectors_by_simu[i]
    Cov = np.cov(Vectors, bias = True)
    Covs_by_simu.append(Cov)
Covs_by_simu = np.asarray(Covs_by_simu)

#unbiasing the matrices
Covs_by_simu = 63 * Covs_by_simu

# Taking the average of the matrices
Cov_jack = np.mean(Covs_by_simu, axis = 0)

print('jackknife noise matrix computed')

## Plotting the full noise matrices
print("starting to plot the full matrices")

fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 5))

cmap = 'viridis'
norm = mpl.colors.LogNorm()

ax1 = axes[0]
ax1.set_title("21 simulations noise matrix")
ax1.set_xlabel("bin numbers : $d$ if $<6$, $s$ otherwise")
ax1.set_ylabel("bin numbers : $d$ if $<6$, $s$ otherwise")
ax1.axhline(y = 6, color = 'black', linestyle = '--')
ax1.axvline(x = 6, color = 'black', linestyle = '--')
im = ax1.matshow(np.abs(Cov_old), cmap = cmap, norm = norm, origin = 'upper')

ax2 = axes[1]
ax2.set_title("21 simulations + (64/1) Jackknife noise matrix")
ax2.set_xlabel("bin numbers : $d$ if $<6$, $s$ otherwise")
ax2.set_ylabel("bin numbers : $d$ if $<6$, $s$ otherwise")
ax2.axhline(y = 6, color = 'black', linestyle = '--')
ax2.axvline(x = 6, color = 'black', linestyle = '--')
im = ax2.matshow(np.abs(Cov_jack), cmap = cmap, norm = norm, origin = 'upper')

cbar = ax2.figure.colorbar(im, ax = ax2)
cbar.ax.set_ylabel("Coefficients")

plt.suptitle("Comparison between 2 noise estimators")
plt.show()

print("full matrices plotted")

## Finding the GP's covariance matrix
print("looking for the GP covariance matrix")

Errors = [0 for j in range(n_groups)]

for j in range(n_groups):
    print(" group " + str(j))
    
    # getting the right data and test groups
    X_d_data, Y_d_data, X_d_test, Y_d_test = List_groups_d[j]
    X_s_data, Y_s_data, X_s_test, Y_s_test = List_groups_s[j]
    
    # creating the gaussian processes and optimizing them
    gp_d = GP.GP(X = X_d_data, Y = Y_d_data, n_points_per_simu = 4, Noise = None, make_covariance_matrix = False)
    gp_d.optimize_model(optimizer = 'lbfgsb')
    gp_s = GP.GP(X = X_s_data, Y = Y_s_data, n_points_per_simu = 5, Noise = None, make_covariance_matrix = False)
    gp_s.optimize_model(optimizer = 'lbfgsb')
    
    # getting the errors
    error_d = gp_d.compute_error_test(X_d_test, Y_d_test)
    error_s = gp_s.compute_error_test(X_s_test, Y_s_test)
    error_ds = np.concatenate((error_d, error_s))
    
    # adding the errors to the lists
    Errors[j] = error_ds
    
    print("work done on group " + str(j))

Errors = np.asarray(Errors)
Cov_GP = np.cov(Errors.T)
    
print("performances successfully evaluated")

# we need to renormalize it because it was computed from log10-normalized statistics
Indexes_to_keep = [0, 1, 2, 3, 18, 23, 28, 33, 38]
Means = np.reshape(Mean_old[Indexes_to_keep], (9,1))
Normalization_matrix = Means.T * Means
Normalization_matrix = np.log(10)**2 * Normalization_matrix

Cov_GP_small = Normalization_matrix * Cov_GP

print("GP matrix found")

## Cutting the noise matrices
print("starting to cut the noise matrices")

Indexes_to_keep = [0, 1, 2, 3, 18, 23, 28, 33, 38]
Cov_old_small = Cov_old[np.ix_(Indexes_to_keep,Indexes_to_keep)]
Cov_jack_small = Cov_jack[np.ix_(Indexes_to_keep,Indexes_to_keep)]

print("noise matrices cut")

## Plotting the small matrices
print("starting to plot the old matrix")

fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 5))

cmap = 'viridis'
norm = mpl.colors.LogNorm()

# ax1 = axes[0]
# ax1.set_title("Old noise matrix")
# ax1.set_xlabel("bin numbers : $d$ if $<4$, $s$ otherwise")
# ax1.set_ylabel("bin numbers : $d$ if $<4$, $s$ otherwise")
# ax1.axhline(y = 3.5, color = 'black', linestyle = '--')
# ax1.axvline(x = 3.5, color = 'black', linestyle = '--')
# im = ax1.matshow(np.abs(Cov_old_small), cmap = cmap, norm = norm, origin = 'upper')

ax2 = axes[0]
ax2.set_title("Jackknife noise matrix")
ax2.set_xlabel("bin numbers : $d$ if $<4$, $s$ otherwise")
ax2.set_ylabel("bin numbers : $d$ if $<4$, $s$ otherwise")
ax2.axhline(y = 3.5, color = 'black', linestyle = '--')
ax2.axvline(x = 3.5, color = 'black', linestyle = '--')
im = ax2.matshow(np.abs(Cov_jack_small), cmap = cmap, norm = norm, origin = 'upper')

ax3 = axes[1]
ax3.set_title("GP covariance matrix")
ax3.set_xlabel("bin numbers : $d$ if $<4$, $s$ otherwise")
ax3.set_ylabel("bin numbers : $d$ if $<4$, $s$ otherwise")
ax3.axhline(y = 3.5, color = 'black', linestyle = '--')
ax3.axvline(x = 3.5, color = 'black', linestyle = '--')
im = ax3.matshow(np.abs(Cov_GP_small), cmap = cmap, norm = norm, origin = 'upper')

cbar = ax3.figure.colorbar(im, ax = ax3)
cbar.ax.set_ylabel("Coefficients")

plt.suptitle("Comparison between covariance matrices")
plt.show()

print("old matrix plotted")

## Plotting the ratio of the 2 matrices
print("starting to plot the ratio matrix")

fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (6, 5))

cmap = 'viridis'
norm = mpl.colors.LogNorm(vmin = 10**(-1), vmax = 10**(1))

ax1 = axes
ax1.set_title("Ratio of the GP noise to the observational noise")
ax1.set_xlabel("bin numbers : $d$ if $<4$, $s$ otherwise")
ax1.set_ylabel("bin numbers : $d$ if $<4$, $s$ otherwise")
ax1.axhline(y = 3.5, color = 'black', linestyle = '--')
ax1.axvline(x = 3.5, color = 'black', linestyle = '--')
im = ax1.matshow(np.abs(Cov_GP_small / Cov_jack_small), cmap = cmap, norm = norm, origin = 'upper')

cbar = ax1.figure.colorbar(im, ax = ax1)
cbar.ax.set_ylabel("Coefficients")

plt.show()

print("ratio plotted")

## Saving the 2 important matrices
print("Starting to save the matrices")

my_path = r'C:\Users\Nathan\Documents\D - X\C - Stages\Stage 3A\Repository_Stage_3A\Full_MST_stats_Abacus_3'
my_file = 'Jackknife_noise_matrix.pkl'
my_file = os.path.join(my_path, my_file)

f = open(my_file, "wb")
pickle.dump(Cov_jack_small * Normalization_matrix**(-1), f)
f.close()

my_path = r'C:\Users\Nathan\Documents\D - X\C - Stages\Stage 3A\Repository_Stage_3A\Full_MST_stats_Abacus_3'
my_file = 'GP_error_matrix.pkl'
my_file = os.path.join(my_path, my_file)

f = open(my_file, "wb")
pickle.dump(Cov_GP, f)
f.close()

print("matrices saved")