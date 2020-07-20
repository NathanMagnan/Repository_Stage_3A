## Imports
import numpy as np
import GPy as GPy
import os
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)

print("All imports successful")

## Importing Abacus 720
print("starting to load the data")

#target = "/home/astro/magnan/Repository_Stage_3A/Full_MST_stats_Abacus/MST_stats_Catalogue_"
target = "C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Full_MST_stats_Abacus/MST_stats_Catalogue_"

dict_720 = {'X_d' : [], 'Y_d' : [], 'X_l' : [], 'Y_l' : [], 'X_b' : [], 'Y_b' : [], 'X_s' : [], 'Y_s' : []}

for i in range(41):    
    X_d_a = np.loadtxt(str(target) + str(i) + "_X_d")
    Y_d_a = np.loadtxt(str(target) + str(i) + "_Y_d")
    dict_720['X_d'].append(X_d_a)
    dict_720['Y_d'].append(Y_d_a)
    
    X_l_a = np.loadtxt(str(target) + str(i) + "_X_l")
    Y_l_a = np.loadtxt(str(target) + str(i) + "_Y_l")
    dict_720['X_l'].append(X_l_a)
    dict_720['Y_l'].append(Y_l_a)
    
    X_b_a = np.loadtxt(str(target) + str(i) + "_X_b")
    Y_b_a = np.loadtxt(str(target) + str(i) + "_Y_b")
    dict_720['X_b'].append(X_b_a)
    dict_720['Y_b'].append(Y_b_a)
    
    X_s_a = np.loadtxt(str(target) + str(i) + "_X_s")
    Y_s_a = np.loadtxt(str(target) + str(i) + "_Y_s")
    dict_720['X_s'].append(X_s_a)
    dict_720['Y_s'].append(Y_s_a)

print("data fully loaded")

## Importing Abacus 1100
print("starting to load the data")

#target = "/home/astro/magnan/Repository_Stage_3A/Full_MST_stats_Abacus/MST_stats_Catalogue_"
target = "C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Full_MST_stats_Abacus_1100/MST_stats_Catalogue_"

dict_1100 = {'X_d' : [], 'Y_d' : [], 'X_l' : [], 'Y_l' : [], 'X_b' : [], 'Y_b' : [], 'X_s' : [], 'Y_s' : []}

for i in range(41):    
    X_d_a = np.loadtxt(str(target) + str(i) + "_X_d")
    Y_d_a = np.loadtxt(str(target) + str(i) + "_Y_d")
    dict_1100['X_d'].append(X_d_a)
    dict_1100['Y_d'].append(Y_d_a)
    
    X_l_a = np.loadtxt(str(target) + str(i) + "_X_l")
    Y_l_a = np.loadtxt(str(target) + str(i) + "_Y_l")
    dict_1100['X_l'].append(X_l_a)
    dict_1100['Y_l'].append(Y_l_a)
    
    X_b_a = np.loadtxt(str(target) + str(i) + "_X_b")
    Y_b_a = np.loadtxt(str(target) + str(i) + "_Y_b")
    dict_1100['X_b'].append(X_b_a)
    dict_1100['Y_b'].append(Y_b_a)
    
    X_s_a = np.loadtxt(str(target) + str(i) + "_X_s")
    Y_s_a = np.loadtxt(str(target) + str(i) + "_Y_s")
    dict_1100['X_s'].append(X_s_a)
    dict_1100['Y_s'].append(Y_s_a)

print("data fully loaded")

## Plotting
print("starting to plot")

fig, axes = plt.subplots(nrows = 2, ncols = 4, figsize = (20, 10))
plt.subplots_adjust(hspace = 0.3, wspace = 0.4)

for j in range(4):
    for i in range(2):
        subplot = axes[i][j]
        
        if (j == 0):
            subplot.set_xlabel('$d$')
            subplot.set_xlim(1, 6)
            
            X_Mean_720 = dict_720['X_d'][0]
            Mean_720 = np.asarray([0 for k in range(np.shape(dict_720['X_d'][0])[0])])
            Std_720 = np.asarray([0 for k in range(np.shape(dict_720['X_d'][0])[0])])
            for k in range(n_simulations):
                New = []
                for x1 in X_Mean_720:
                    min = 10
                    l_min = 0
                    for l in range(np.shape(dict_720['X_d'][k])[0]):
                        x2 = dict_720['X_d'][k][l]
                        if (abs(x1 - x2) < min):
                            min = abs(x1 - x2)
                            l_min = l
                    
                    New.append(dict_720['Y_d'][k][l_min])
                New = np.asarray(New)
                
                Mean_old = Mean_720.copy()
                Std_old = Std_720.copy()
                
                Mean_new = (k * Mean_old + New) / (k + 1)
                Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + New**2) / (k + 1) - Mean_new**2)
                
                Mean_720 = Mean_new.copy()
                Std_720 = Std_new.copy()
            
            if (i == 0):
                subplot.set_ylabel('$N_{d}$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**4, 10**6)
                
                subplot.fill_between(x = X_Mean_720, y1 = Mean_720 - Std_720, y2 = Mean_720 + Std_720, color = 'b', alpha = 0.2, label = 'Abacus 720 range')
                subplot.plot(dict_720['X_d'][0], dict_720['Y_d'][0], 'b', label = "Abacus 720")
                subplot.plot(dict_1100['X_d'][0], dict_1100['Y_d'][0], 'b--', label = "Abacus 1100")
                
                subplot.legend()
                
            else:
                subplot.set_ylabel('$\Delta N_{d} / <N_{d}>$')
                subplot.set_ylim(-0.1, 0.1)
                
                subplot.fill_between(x = X_Mean_720, y1 = - Std_720 / Mean_720, y2 = Std_720 / Mean_720, color = 'b', alpha = 0.2, label = "Abacus 720 range")
                
                subplot.legend()

plt.show()