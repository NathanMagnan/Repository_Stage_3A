## Imports
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)

print("All imports successful")

## Loading the data
print("starting to load the data")

#target = "/home/astro/magnan/Repository_Stage_3A/Full_MST_stats_Abacus/MST_stats_Catalogue_"
target = "C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Full_MST_stats_Abacus/MST_stats_Catalogue_"

dict = {'X_d' : [], 'Y_d' : [], 'X_l' : [], 'Y_l' : [], 'X_b' : [], 'Y_b' : [], 'X_s' : [], 'Y_s' : []}

for i in range(41):    
    X_d = np.loadtxt(str(target) + str(i) + "_X_d")
    Y_d = np.loadtxt(str(target) + str(i) + "_Y_d")
    dict['X_d'].append(X_d)
    dict['Y_d'].append(Y_d)
    
    X_l = np.loadtxt(str(target) + str(i) + "_X_l")
    Y_l = np.loadtxt(str(target) + str(i) + "_Y_l")
    dict['X_l'].append(X_l)
    dict['Y_l'].append(Y_l)
    
    X_b = np.loadtxt(str(target) + str(i) + "_X_b")
    Y_b = np.loadtxt(str(target) + str(i) + "_Y_b")
    dict['X_b'].append(X_b)
    dict['Y_b'].append(Y_b)
    
    X_s = np.loadtxt(str(target) + str(i) + "_X_s")
    Y_s = np.loadtxt(str(target) + str(i) + "_Y_s")
    dict['X_s'].append(X_s)
    dict['Y_s'].append(Y_s)

print("data fully loaded")

## Plot
print("Starting to plot the MST stats")

fig, axes = plt.subplots(nrows = 2, ncols = 4, figsize = (20, 10))
plt.subplots_adjust(hspace = 0.3, wspace = 0.4)

for j in range(4):
    for i in range(2):
        subplot = axes[i][j]
        
        if (j == 0):
            subplot.set_xlabel('$d$')
            if (i == 0):
                subplot.set_ylabel('$N_{d}$')
                subplot.set_yscale('log')
                
                Mean = dict['Y_d'][0]
                Std = np.asarray([0 for k in range(np.shape(dict['Y_d'][0])[0])])
                for k in range(1, 41):
                    Mean_old = Mean.copy()
                    Std_old = Std.copy()
                    
                    Mean_new = (k * Mean_old + dict['Y_d'][k]) / (k + 1)
                    Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + dict['Y_d'][k]**2) / (k + 1) - Mean_new**2)
                    
                    Mean = Mean_new.copy()
                    Std = Std_new.copy()
                
                subplot.fill_between(x = dict['X_d'][0], y1 = Mean - 3 * Std, y2 = Mean + 3 * Std, color = 'b', alpha = 0.2)
                subplot.errorbar(x = dict['X_d'][0], y = Mean, yerr = 3 * Std, fmt = 'o', markersize = 0, ecolor = 'b')
                subplot.plot(dict['X_d'][0], Mean, 'b')
                
            else:
                subplot.set_ylabel('$\Delta N_{d} / \sqrt{<N_{d}}>$')
                subplot.set_yscale('log')
                subplot.set_ylim(0.1, 50)
                
                Mean = dict['Y_d'][0]
                Zeros = np.asarray([0 for k in range(np.shape(dict['Y_d'][0])[0])])
                Std = Zeros
                for k in range(1, 41):
                    Mean_old = Mean.copy()
                    Std_old = Std.copy()
                    
                    Mean_new = (k * Mean_old + dict['Y_d'][k]) / (k + 1)
                    Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + dict['Y_d'][k]**2) / (k + 1) - Mean_new**2)
                    
                    Mean = Mean_new.copy()
                    Std = Std_new.copy()
                
                subplot.fill_between(x = dict['X_d'][0], y1 = Zeros, y2 = Std / np.sqrt(Mean), color = 'b', alpha = 0.2)
                subplot.errorbar(x = dict['X_d'][0], y = Zeros, yerr = Std / np.sqrt(Mean), fmt = 'o', markersize = 0, ecolor = 'b')
                subplot.plot(dict['X_d'][0], Zeros, 'b')
        
        elif (j == 1):
            subplot.set_xlabel('$l$')
            subplot.set_xscale('log')
            if (i == 0):
                subplot.set_ylabel('$N_{l}$')
                subplot.set_yscale('log')
                
                Mean = dict['Y_l'][0]
                Std = np.asarray([0 for k in range(np.shape(dict['Y_l'][0])[0])])
                for k in range(1, 41):
                    Mean_old = Mean.copy()
                    Std_old = Std.copy()
                    
                    Mean_new = (k * Mean_old + dict['Y_l'][k]) / (k + 1)
                    Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + dict['Y_l'][k]**2) / (k + 1) - Mean_new**2)
                    
                    Mean = Mean_new.copy()
                    Std = Std_new.copy()
                
                subplot.fill_between(x = dict['X_l'][0], y1 = Mean - 3 * Std, y2 = Mean + 3 * Std, color = 'g', alpha = 0.2)
                subplot.errorbar(x = dict['X_l'][0][::5], y = Mean[::5], yerr = 3 * Std[::5], fmt = 'o', markersize = 0, ecolor = 'g')
                subplot.plot(dict['X_l'][0], Mean, 'g')
                
            else:
                subplot.set_ylabel('$\Delta N_{l} / \sqrt{<N_{l}>}$')
                subplot.set_yscale('log')
                subplot.set_ylim(0.05, 50)
                
                Mean = dict['Y_l'][0]
                Zeros = np.asarray([0 for k in range(np.shape(dict['Y_l'][0])[0])])
                Std = Zeros
                for k in range(1, 41):
                    Mean_old = Mean.copy()
                    Std_old = Std.copy()
                    
                    Mean_new = (k * Mean_old + dict['Y_l'][k]) / (k + 1)
                    Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + dict['Y_l'][k]**2) / (k + 1) - Mean_new**2)
                    
                    Mean = Mean_new.copy()
                    Std = Std_new.copy()
                
                subplot.fill_between(x = dict['X_l'][0], y1 = Zeros, y2 = Std / np.sqrt(Mean), color = 'g', alpha = 0.2)
                subplot.errorbar(x = dict['X_l'][0][::5], y = Zeros[::5], yerr = (Std / np.sqrt(Mean))[::5], fmt = 'o', markersize = 0, ecolor = 'g')
                subplot.plot(dict['X_l'][0], Zeros, 'g')
                
        elif (j == 2):
            subplot.set_xlabel('$b$')
            subplot.set_xscale('log')
            if (i == 0):
                subplot.set_ylabel('$N_{b}$')
                subplot.set_yscale('log')
                
                Mean = dict['Y_b'][0]
                Std = np.asarray([0 for k in range(np.shape(dict['Y_b'][0])[0])])
                for k in range(1, 41):
                    Mean_old = Mean.copy()
                    Std_old = Std.copy()
                    
                    Mean_new = (k * Mean_old + dict['Y_b'][k]) / (k + 1)
                    Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + dict['Y_b'][k]**2) / (k + 1) - Mean_new**2)
                    
                    Mean = Mean_new.copy()
                    Std = Std_new.copy()
                
                subplot.fill_between(x = dict['X_b'][0], y1 = Mean - 3 * Std, y2 = Mean + 3 * Std, color = 'r', alpha = 0.2)
                subplot.errorbar(x = dict['X_b'][0][::5], y = Mean[::5], yerr = 3 * Std[::5], fmt = 'o', markersize = 0, ecolor = 'r')
                subplot.plot(dict['X_b'][0], Mean, 'r')
                
            else:
                subplot.set_ylabel('$\Delta N_{b} / \sqrt{<N_{b}>}$')
                subplot.set_yscale('log')
                subplot.set_ylim(0.05, 50)
                
                Mean = dict['Y_b'][0]
                Zeros = np.asarray([0 for k in range(np.shape(dict['Y_b'][0])[0])])
                Std = Zeros
                for k in range(1, 41):
                    Mean_old = Mean.copy()
                    Std_old = Std.copy()
                    
                    Mean_new = (k * Mean_old + dict['Y_b'][k]) / (k + 1)
                    Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + dict['Y_b'][k]**2) / (k + 1) - Mean_new**2)
                    
                    Mean = Mean_new.copy()
                    Std = Std_new.copy()
                
                subplot.fill_between(x = dict['X_b'][0], y1 = Zeros, y2 = Std / np.sqrt(Mean), color = 'r', alpha = 0.2)
                subplot.errorbar(x = dict['X_b'][0][::5], y = Zeros[::5], yerr = (Std / np.sqrt(Mean))[::5], fmt = 'o', markersize = 0, ecolor = 'r')
                subplot.plot(dict['X_b'][0], Zeros, 'r')
                
        else:
            subplot.set_xlabel('$s$')
            if (i == 0):
                subplot.set_ylabel('$N_{s}$')
                subplot.set_yscale('log')
                
                Mean = dict['Y_s'][0]
                Std = np.asarray([0 for k in range(np.shape(dict['Y_s'][0])[0])])
                for k in range(1, 41):
                    Mean_old = Mean.copy()
                    Std_old = Std.copy()
                    
                    Mean_new = (k * Mean_old + dict['Y_s'][k]) / (k + 1)
                    Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + dict['Y_s'][k]**2) / (k + 1) - Mean_new**2)
                    
                    Mean = Mean_new.copy()
                    Std = Std_new.copy()
                
                subplot.fill_between(x = dict['X_s'][0], y1 = Mean - 3 * Std, y2 = Mean + 3 * Std, color = 'y', alpha = 0.2)
                subplot.errorbar(x = dict['X_s'][0], y = Mean, yerr = 3 * Std, fmt = 'o', markersize = 0, ecolor = 'y')
                subplot.plot(dict['X_s'][0], Mean, 'y')
                
            else:
                subplot.set_ylabel('$\Delta N_{s} / \sqrt{<N_{s}>}$')
                subplot.set_yscale('log')
                subplot.set_ylim(0.05, 50)
                
                Mean = dict['Y_s'][0]
                Zeros = np.asarray([0 for k in range(np.shape(dict['Y_s'][0])[0])])
                Std = Zeros
                for k in range(1, 41):
                    Mean_old = Mean.copy()
                    Std_old = Std.copy()
                    
                    Mean_new = (k * Mean_old + dict['Y_s'][k]) / (k + 1)
                    Std_new = np.sqrt((k * (Std_old**2 + Mean_old**2) + dict['Y_s'][k]**2) / (k + 1) - Mean_new**2)
                    
                    Mean = Mean_new.copy()
                    Std = Std_new.copy()
                
                subplot.fill_between(x = dict['X_s'][0], y1 = Zeros, y2 = Std / np.sqrt(Mean), color = 'y', alpha = 0.2)
                subplot.errorbar(x = dict['X_s'][0], y = Zeros, yerr = Std / np.sqrt(Mean), fmt = 'o', markersize = 0, ecolor = 'y')
                subplot.plot(dict['X_s'][0], Zeros, 'y')

plt.suptitle("Comparison of all the Abacus MSTs")
print("results plotted")

print("starting to save the results")
#my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Figures')
my_path = os.path.abspath('C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Figures')
my_file = 'Comparison_Abacus_MSTs.png'
plt.savefig(os.path.join(my_path, my_file))
print("results saved")

plt.show()