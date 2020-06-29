## Imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)

import sys
import os

print("All imports successful")

## Importing the data
print("starting to load the data")

#target = "/home/astro/magnan/Repository_Stage_3A/Full_MST_stats_Abacus/MST_stats_Catalogue_"
target = "C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Full_MST_stats_Abacus/MST_stats_Catalogue_"

dict = {'X_d' : [], 'Y_d' : [], 'Y_d_std' : [], 'X_l' : [], 'Y_l' : [], 'Y_l_std' : [], 'X_b' : [], 'Y_b' : [], 'Y_b_std' : [], 'X_s' : [], 'Y_s' : [], 'Y_s_std' : []}

for i in range(61):    
    X_d_a = np.loadtxt(str(target) + str(i) + "_X_d")
    Y_d_a = np.loadtxt(str(target) + str(i) + "_Y_d")
    Y_d_std_a = np.loadtxt(str(target) + str(i) + "_Y_d_std")
    dict['X_d'].append(X_d_a)
    dict['Y_d'].append(Y_d_a)
    dict['Y_d_std'].append(Y_d_std_a)
    
    X_l_a = np.loadtxt(str(target) + str(i) + "_X_l")
    Y_l_a = np.loadtxt(str(target) + str(i) + "_Y_l")
    Y_l_std_a = np.loadtxt(str(target) + str(i) + "_Y_l_std")
    dict['X_l'].append(X_l_a)
    dict['Y_l'].append(Y_l_a)
    dict['Y_l_std'].append(Y_l_std_a)
    
    X_b_a = np.loadtxt(str(target) + str(i) + "_X_b")
    Y_b_a = np.loadtxt(str(target) + str(i) + "_Y_b")
    Y_b_std_a = np.loadtxt(str(target) + str(i) + "_Y_b_std")
    dict['X_b'].append(X_b_a)
    dict['Y_b'].append(Y_b_a)
    dict['Y_b_std'].append(Y_b_std_a)
    
    X_s_a = np.loadtxt(str(target) + str(i) + "_X_s")
    Y_s_a = np.loadtxt(str(target) + str(i) + "_Y_s")
    Y_s_std_a = np.loadtxt(str(target) + str(i) + "_Y_s_std")
    dict['X_s'].append(X_s_a)
    dict['Y_s'].append(Y_s_a)
    dict['Y_s_std'].append(Y_s_std_a)

print("data fully loaded")

## Plotting
print("Starting to plot the MST stats")

fig, axes = plt.subplots(nrows = 2, ncols = 4, figsize = (20, 10))
plt.subplots_adjust(hspace = 0.3, wspace = 0.4)

for j in range(4):
    for i in range(2):
        subplot = axes[i][j]
        
        if (j == 0):
            subplot.set_xlabel('$d$')
            subplot.set_xlim(1, 4)
            
            X_abacus = dict['X_d'][0]
            Mean_abacus = np.asarray([0 for k in range(np.shape(dict['X_d'][0])[0])])
            Std_abacus = np.asarray([0 for k in range(np.shape(dict['X_d'][0])[0])])
            for k in range(41):
                New = []
                for x1 in X_abacus:
                    min = 1
                    l_min = 0
                    for l in range(np.shape(X_abacus)[0]):
                        x2 = dict['X_d'][k][l]
                        if (abs(x1 - x2) < min):
                            min = abs(x1 - x2)
                            l_min = l
                    
                    New.append(dict['Y_d'][k][l_min])
                New = np.asarray(New)
                
                Mean_old = Mean_abacus.copy()
                Std_old = Std_abacus.copy()
                
                Mean_new = (k * Mean_old + New) / (k + 1)
                Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + New**2) / (k + 1) - Mean_new**2)
                
                Mean_abacus = Mean_new.copy()
                Std_abacus = Std_new.copy()
            
            Mean_fidu = np.asarray([0 for k in range(np.shape(dict['X_d'][0])[0])])
            Std_fidu = np.asarray([0 for k in range(np.shape(dict['X_d'][0])[0])])
            for k in range(20):
                New = []
                for x1 in X_abacus:
                    min = 1
                    l_min = 0
                    for l in range(np.shape(X_abacus)[0]):
                        x2 = dict['X_d'][k + 41][l]
                        if (abs(x1 - x2) < min):
                            min = abs(x1 - x2)
                            l_min = l
                    
                    New.append(dict['Y_d'][k + 41][l_min])
                New = np.asarray(New)
                    
                Mean_old = Mean_fidu.copy()
                Std_old = Std_fidu.copy()
                
                Mean_new = (k * Mean_old + New) / (k + 1)
                Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + New**2) / (k + 1) - Mean_new**2)
                
                Mean_fidu = Mean_new.copy()
                Std_fidu = Std_new.copy()
            
            if (i == 0):
                subplot.set_ylabel('$N_{d}$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**4, 10**6)
                
                subplot.fill_between(x = X_abacus, y1 = Mean_abacus - Std_abacus, y2 = Mean_abacus + Std_abacus, color = 'b', alpha = 0.2, label = 'Abacus range')
                subplot.fill_between(x = X_abacus, y1 = Mean_fidu - Std_fidu, y2 = Mean_fidu + Std_fidu, color = 'b', alpha = 0.6, label = 'Fiducial range')
                
                subplot.legend()
                
            else:
                subplot.set_ylabel('$\Delta N_{d} / <N_{d}>$')
                subplot.set_ylim(-0.3, 0.3)
                
                subplot.fill_between(x = X_abacus, y1 = - Std_abacus / Mean_abacus, y2 = Std_abacus / Mean_abacus, color = 'b', alpha = 0.2, label = "Abacus range")
                subplot.fill_between(x = X_abacus, y1 = (Mean_fidu - Std_fidu - Mean_abacus) / Mean_abacus, y2 = (Mean_fidu + Std_fidu - Mean_abacus) / Mean_abacus, color = 'b', alpha = 0.6, label = "Fiducial range")
                
                subplot.legend()
        
        elif (j == 1):
            subplot.set_xlabel('$l$')
            subplot.set_xscale('log')
            subplot.set_xlim(1, 10)
            
            X_abacus = dict['X_l'][0]
            Mean_abacus = np.asarray([0 for k in range(np.shape(dict['X_l'][0])[0])])
            Std_abacus = np.asarray([0 for k in range(np.shape(dict['X_l'][0])[0])])
            for k in range(41):
                New = []
                for x1 in X_abacus:
                    min = 1
                    l_min = 0
                    for l in range(np.shape(X_abacus)[0]):
                        x2 = dict['X_l'][k][l]
                        if (abs(x1 - x2) < min):
                            min = abs(x1 - x2)
                            l_min = l
                    try:
                        New.append((dict['Y_l'][k][l_min - 1] + dict['Y_l'][k][l_min] + dict['Y_l'][k][l_min + 1]) / 3)
                    except:
                        New.append(dict['Y_l'][k][l_min])
                New = np.asarray(New)
                
                Mean_old = Mean_abacus.copy()
                Std_old = Std_abacus.copy()
                
                Mean_new = (k * Mean_old + New) / (k + 1)
                Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + New**2) / (k + 1) - Mean_new**2)
                
                Mean_abacus = Mean_new.copy()
                Std_abacus = Std_new.copy()
            
            Mean_fidu = np.asarray([0 for k in range(np.shape(dict['X_l'][0])[0])])
            Std_fidu = np.asarray([0 for k in range(np.shape(dict['X_l'][0])[0])])
            for k in range(20):
                New = []
                for x1 in X_abacus:
                    min = 1
                    l_min = 0
                    for l in range(np.shape(X_abacus)[0]):
                        x2 = dict['X_l'][k + 41][l]
                        if (abs(x1 - x2) < min):
                            min = abs(x1 - x2)
                            l_min = l
                    try:
                        New.append((dict['Y_l'][k + 41][l_min - 1] + dict['Y_l'][k + 41][l_min] + dict['Y_l'][k + 41][l_min + 1]) / 3)
                    except:
                        New.append(dict['Y_l'][k + 41][l_min])
                New = np.asarray(New)
                    
                Mean_old = Mean_fidu.copy()
                Std_old = Std_fidu.copy()
                
                Mean_new = (k * Mean_old + New) / (k + 1)
                Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + New**2) / (k + 1) - Mean_new**2)
                
                Mean_fidu = Mean_new.copy()
                Std_fidu = Std_new.copy()
            
            if (i == 0):
                subplot.set_ylabel('$N_{l}$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**3, 10**5)
                
                subplot.fill_between(x = X_abacus, y1 = Mean_abacus - Std_abacus, y2 = Mean_abacus + Std_abacus, color = 'g', alpha = 0.2, label = 'Abacus range')
                subplot.fill_between(x = X_abacus, y1 = Mean_fidu - Std_fidu, y2 = Mean_fidu + Std_fidu, color = 'g', alpha = 0.6, label = 'Fiducial range')
                
                subplot.legend()
                
            else:
                subplot.set_ylabel('$\Delta N_{l} / <N_{l}>$')
                subplot.set_ylim(-0.3, 0.3)
                
                subplot.fill_between(x = X_abacus, y1 = - Std_abacus / Mean_abacus, y2 = Std_abacus / Mean_abacus, color = 'g', alpha = 0.2, label = "Abacus range")
                subplot.fill_between(x = X_abacus, y1 = (Mean_fidu - Std_fidu - Mean_abacus) / Mean_abacus, y2 = (Mean_fidu + Std_fidu - Mean_abacus) / Mean_abacus, color = 'g', alpha = 0.6, label = "Fiducial range")
                
                subplot.legend()
                
                subplot.legend()
                
        elif (j == 2):
            subplot.set_xlabel('$b$')
            subplot.set_xscale('log')
            subplot.set_xlim(5, 50)
            
            X_abacus = dict['X_b'][0]
            Mean_abacus = np.asarray([0 for k in range(np.shape(dict['X_b'][0])[0])])
            Std_abacus = np.asarray([0 for k in range(np.shape(dict['X_b'][0])[0])])
            for k in range(41):
                New = []
                for x1 in X_abacus:
                    min = 1
                    l_min = 0
                    for l in range(np.shape(X_abacus)[0]):
                        x2 = dict['X_b'][k][l]
                        if (abs(x1 - x2) < min):
                            min = abs(x1 - x2)
                            l_min = l
                    try:
                        New.append((dict['Y_b'][k][l_min - 1] + dict['Y_b'][k][l_min] + dict['Y_b'][k][l_min + 1]) / 3)
                    except:
                        New.append(dict['Y_b'][k][l_min])
                New = np.asarray(New)
                
                Mean_old = Mean_abacus.copy()
                Std_old = Std_abacus.copy()
                
                Mean_new = (k * Mean_old + New) / (k + 1)
                Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + New**2) / (k + 1) - Mean_new**2)
                
                Mean_abacus = Mean_new.copy()
                Std_abacus = Std_new.copy()
            
            Mean_fidu = np.asarray([0 for k in range(np.shape(dict['X_b'][0])[0])])
            Std_fidu = np.asarray([0 for k in range(np.shape(dict['X_b'][0])[0])])
            for k in range(20):
                New = []
                for x1 in X_abacus:
                    min = 1
                    l_min = 0
                    for l in range(np.shape(X_abacus)[0]):
                        x2 = dict['X_b'][k + 41][l]
                        if (abs(x1 - x2) < min):
                            min = abs(x1 - x2)
                            l_min = l
                    try:
                        New.append((dict['Y_b'][k + 41][l_min - 1] + dict['Y_b'][k + 41][l_min] + dict['Y_b'][k + 41][l_min + 1]) / 3)
                    except:
                        New.append(dict['Y_b'][k + 41][l_min])
                New = np.asarray(New)
                    
                Mean_old = Mean_fidu.copy()
                Std_old = Std_fidu.copy()
                
                Mean_new = (k * Mean_old + New) / (k + 1)
                Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + New**2) / (k + 1) - Mean_new**2)
                
                Mean_fidu = Mean_new.copy()
                Std_fidu = Std_new.copy()
            
            if (i == 0):
                subplot.set_ylabel('$N_{b}$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**1, 10**4)
                
                subplot.fill_between(x = X_abacus, y1 = Mean_abacus - Std_abacus, y2 = Mean_abacus + Std_abacus, color = 'r', alpha = 0.2, label = 'Abacus range')
                subplot.fill_between(x = X_abacus, y1 = Mean_fidu - Std_fidu, y2 = Mean_fidu + Std_fidu, color = 'r', alpha = 0.6, label = 'Fiducial range')
                
                subplot.legend()
                
            else:
                subplot.set_ylabel('$\Delta N_{b} / <N_{b}>$')
                subplot.set_ylim(-0.3, 0.3)
                
                subplot.fill_between(x = X_abacus, y1 = - Std_abacus / Mean_abacus, y2 = Std_abacus / Mean_abacus, color = 'r', alpha = 0.2, label = "Abacus range")
                subplot.fill_between(x = X_abacus, y1 = (Mean_fidu - Std_fidu - Mean_abacus) / Mean_abacus, y2 = (Mean_fidu + Std_fidu - Mean_abacus) / Mean_abacus, color = 'r', alpha = 0.6, label = "Fiducial range")
                
                subplot.legend()
                
        else:
            subplot.set_xlabel('$s$')
            subplot.set_xlim(0.3, 0.7)
            
            X_abacus = dict['X_s'][0]
            Mean_abacus = np.asarray([0 for k in range(np.shape(dict['X_s'][0])[0])])
            Std_abacus = np.asarray([0 for k in range(np.shape(dict['X_s'][0])[0])])
            for k in range(41):
                New = []
                for x1 in X_abacus:
                    min = 1
                    l_min = 0
                    for l in range(np.shape(X_abacus)[0]):
                        x2 = dict['X_s'][k][l]
                        if (abs(x1 - x2) < min):
                            min = abs(x1 - x2)
                            l_min = l
                    try:
                        New.append((dict['Y_s'][k][l_min - 1] + dict['Y_s'][k][l_min] + dict['Y_s'][k][l_min + 1]) / 3)
                    except:
                        New.append(dict['Y_s'][k][l_min])
                New = np.asarray(New)
                
                Mean_old = Mean_abacus.copy()
                Std_old = Std_abacus.copy()
                
                Mean_new = (k * Mean_old + New) / (k + 1)
                Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + New**2) / (k + 1) - Mean_new**2)
                
                Mean_abacus = Mean_new.copy()
                Std_abacus = Std_new.copy()
            
            Mean_fidu = np.asarray([0 for k in range(np.shape(dict['X_s'][0])[0])])
            Std_fidu = np.asarray([0 for k in range(np.shape(dict['X_s'][0])[0])])
            for k in range(20):
                New = []
                for x1 in X_abacus:
                    min = 1
                    l_min = 0
                    for l in range(np.shape(X_abacus)[0]):
                        x2 = dict['X_s'][k + 41][l]
                        if (abs(x1 - x2) < min):
                            min = abs(x1 - x2)
                            l_min = l
                    try:
                        New.append((dict['Y_s'][k + 41][l_min - 1] + dict['Y_s'][k + 41][l_min] + dict['Y_s'][k + 41][l_min + 1]) / 3)
                    except:
                        New.append(dict['Y_s'][k + 41][l_min])
                New = np.asarray(New)
                    
                Mean_old = Mean_fidu.copy()
                Std_old = Std_fidu.copy()
                
                Mean_new = (k * Mean_old + New) / (k + 1)
                Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + New**2) / (k + 1) - Mean_new**2)
                
                Mean_fidu = Mean_new.copy()
                Std_fidu = Std_new.copy()
            
            if (i == 0):
                subplot.set_ylabel('$N_{s}$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**3, 10**4)
                
                subplot.fill_between(x = X_abacus, y1 = Mean_abacus - Std_abacus, y2 = Mean_abacus + Std_abacus, color = 'y', alpha = 0.2, label = 'Abacus range')
                subplot.fill_between(x = X_abacus, y1 = Mean_fidu - Std_fidu, y2 = Mean_fidu + Std_fidu, color = 'y', alpha = 0.6, label = 'Fiducial range')
                
                subplot.legend()
                
            else:
                subplot.set_ylabel('$\Delta N_{s} / <N_{s}>$')
                subplot.set_ylim(-0.3, 0.3)
                
                subplot.fill_between(x = X_abacus, y1 = - Std_abacus / Mean_abacus, y2 = Std_abacus / Mean_abacus, color = 'y', alpha = 0.2, label = "Abacus range")
                subplot.fill_between(x = X_abacus, y1 = (Mean_fidu - Std_fidu - Mean_abacus) / Mean_abacus, y2 = (Mean_fidu + Std_fidu - Mean_abacus) / Mean_abacus, color = 'y', alpha = 0.6, label = "Fiducial range")
                
                subplot.legend()

plt.suptitle("Error in Abacus simulations")
plt.show()