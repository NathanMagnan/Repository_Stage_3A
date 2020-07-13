## Imports
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)

print("All imports successful")

## Importing the data
print("starting to load the data")

#target = "/home/astro/magnan/Repository_Stage_3A/Full_MST_stats_Abacus/MST_stats_Catalogue_"
target = "C:/Users/Nathan/Documents/D - X/C - Stages/Stage 3A/Repository_Stage_3A/Test_masks/"

dict = {'X_d' : [], 'Y_d' : [], 'Y_d_std' : [], 'X_l' : [], 'Y_l' : [], 'Y_l_std' : [], 'X_b' : [], 'Y_b' : [], 'Y_b_std' : [], 'X_s' : [], 'Y_s' : [], 'Y_s_std' : []}

Strs = ['X_d', 'Y_d', 'Y_d_std', 'X_l', 'Y_l', 'Y_l_std', 'X_b', 'Y_b', 'Y_b_std', 'X_s', 'Y_s', 'Y_s_std']

## Difference between the random catalogue and Abacus
# Labels = ['Abacus', 'random']
# Linestyles = ['b', 'b--']
# 
# # Abacus catalogue
# X_d = np.loadtxt(str(target) + 'Full_Abacus' + "_X_d")
# Y_d = np.loadtxt(str(target) + 'Full_Abacus' + "_Y_d")
# Y_d_std = np.array([0 for y_d in Y_d])
# dict['X_d'].append(X_d)
# dict['Y_d'].append(Y_d)
# dict['Y_d_std'].append(Y_d_std)
# 
# X_l = np.loadtxt(str(target) + 'Full_Abacus' + "_X_l")
# Y_l = np.loadtxt(str(target) + 'Full_Abacus' + "_Y_l")
# Y_l_std = np.array([0 for y_l in Y_l])
# dict['X_l'].append(X_l)
# dict['Y_l'].append(Y_l)
# dict['Y_l_std'].append(Y_l_std)
# 
# X_b = np.loadtxt(str(target) + 'Full_Abacus' + "_X_b")
# Y_b = np.loadtxt(str(target) + 'Full_Abacus' + "_Y_b")
# Y_b_std = np.array([0 for y_b in Y_b])
# dict['X_b'].append(X_b)
# dict['Y_b'].append(Y_b)
# dict['Y_b_std'].append(Y_b_std)
# 
# X_s = np.loadtxt(str(target) + 'Full_Abacus' + "_X_s")
# Y_s = np.loadtxt(str(target) + 'Full_Abacus' + "_Y_s")
# Y_s_std = np.array([0 for y_s in Y_s])
# dict['X_s'].append(X_s)
# dict['Y_s'].append(Y_s)
# dict['Y_s_std'].append(Y_s_std)
# 
# # Random catalogue
# X_d = np.loadtxt(str(target) + 'Full_Random' + "_X_d")
# Y_d = np.loadtxt(str(target) + 'Full_Random' + "_Y_d")
# Y_d_std = np.array([0 for y_d in Y_d])
# dict['X_d'].append(X_d)
# dict['Y_d'].append(Y_d)
# dict['Y_d_std'].append(Y_d_std)
# 
# X_l = np.loadtxt(str(target) + 'Full_Random' + "_X_l")
# Y_l = np.loadtxt(str(target) + 'Full_Random' + "_Y_l")
# Y_l_std = np.array([0 for y_l in Y_l])
# dict['X_l'].append(X_l)
# dict['Y_l'].append(Y_l)
# dict['Y_l_std'].append(Y_l_std)
# 
# X_b = np.loadtxt(str(target) + 'Full_Random' + "_X_b")
# Y_b = np.loadtxt(str(target) + 'Full_Random' + "_Y_b")
# Y_b_std = np.array([0 for y_b in Y_b])
# dict['X_b'].append(X_b)
# dict['Y_b'].append(Y_b)
# dict['Y_b_std'].append(Y_b_std)
# 
# X_s = np.loadtxt(str(target) + 'Full_Random' + "_X_s")
# Y_s = np.loadtxt(str(target) + 'Full_Random' + "_Y_s")
# Y_b_std = np.array([0 for y_b in Y_b])
# dict['X_s'].append(X_s)
# dict['Y_s'].append(Y_s)
# dict['Y_s_std'].append(Y_s_std)

## Effect of the masks on Abacus
# Labels = ['center', 'face', 'edge', 'corner']
# Linestyles = ['b', 'g', 'r', 'y']
#
# for k in range(4):
#     X_d = np.loadtxt(str(target) + 'Masked_abacus_' + Labels[k] + "_X_d")
#     Y_d = np.loadtxt(str(target) + 'Masked_abacus_' + Labels[k] + "_Y_d")
#     Y_d_std = np.loadtxt(str(target) + 'Masked_abacus_' + Labels[k] + "_Y_d_std")
#     dict['X_d'].append(X_d)
#     dict['Y_d'].append(Y_d)
#     dict['Y_d_std'].append(Y_d_std)
#     
#     X_l = np.loadtxt(str(target) + 'Masked_abacus_' + Labels[k] + "_X_l")
#     Y_l = np.loadtxt(str(target) + 'Masked_abacus_' + Labels[k] + "_Y_l")
#     Y_l_std = np.loadtxt(str(target) + 'Masked_abacus_' + Labels[k] + "_Y_l_std")
#     dict['X_l'].append(X_l)
#     dict['Y_l'].append(Y_l)
#     dict['Y_l_std'].append(Y_l_std)
#     
#     X_b = np.loadtxt(str(target) + 'Masked_abacus_' + Labels[k] + "_X_b")
#     Y_b = np.loadtxt(str(target) + 'Masked_abacus_' + Labels[k] + "_Y_b")
#     Y_b_std = np.loadtxt(str(target) + 'Masked_abacus_' + Labels[k] + "_Y_b_std")
#     dict['X_b'].append(X_b)
#     dict['Y_b'].append(Y_b)
#     dict['Y_b_std'].append(Y_b_std)
#     
#     X_s = np.loadtxt(str(target) + 'Masked_abacus_' + Labels[k] + "_X_s")
#     Y_s = np.loadtxt(str(target) + 'Masked_abacus_' + Labels[k] + "_Y_s")
#     Y_s_std = np.loadtxt(str(target) + 'Masked_abacus_' + Labels[k] + "_Y_s_std")
#     dict['X_s'].append(X_s)
#     dict['Y_s'].append(Y_s)
#     dict['Y_s_std'].append(Y_s_std)
# 
# print("data fully loaded")

## Effect of the masks on the random catalogue
# Labels = ['center', 'face', 'edge', 'corner']
# Linestyles = ['b', 'g', 'r', 'y']
# 
# for k in range(4):
#     X_d = np.loadtxt(str(target) + 'Masked_random_' + Labels[k] + "_X_d")
#     Y_d = np.loadtxt(str(target) + 'Masked_random_' + Labels[k] + "_Y_d")
#     Y_d_std = np.loadtxt(str(target) + 'Masked_random_' + Labels[k] + "_Y_d_std")
#     dict['X_d'].append(X_d)
#     dict['Y_d'].append(Y_d)
#     dict['Y_d_std'].append(Y_d_std)
#     
#     X_l = np.loadtxt(str(target) + 'Masked_random_' + Labels[k] + "_X_l")
#     Y_l = np.loadtxt(str(target) + 'Masked_random_' + Labels[k] + "_Y_l")
#     Y_l_std = np.loadtxt(str(target) + 'Masked_random_' + Labels[k] + "_Y_l_std")
#     dict['X_l'].append(X_l)
#     dict['Y_l'].append(Y_l)
#     dict['Y_l_std'].append(Y_l_std)
#     
#     X_b = np.loadtxt(str(target) + 'Masked_random_' + Labels[k] + "_X_b")
#     Y_b = np.loadtxt(str(target) + 'Masked_random_' + Labels[k] + "_Y_b")
#     Y_b_std = np.loadtxt(str(target) + 'Masked_random_' + Labels[k] + "_Y_b_std")
#     dict['X_b'].append(X_b)
#     dict['Y_b'].append(Y_b)
#     dict['Y_b_std'].append(Y_b_std)
#     
#     X_s = np.loadtxt(str(target) + 'Masked_random_' + Labels[k] + "_X_s")
#     Y_s = np.loadtxt(str(target) + 'Masked_random_' + Labels[k] + "_Y_s")
#     Y_s_std = np.loadtxt(str(target) + 'Masked_random_' + Labels[k] + "_Y_s_std")
#     dict['X_s'].append(X_s)
#     dict['Y_s'].append(Y_s)
#     dict['Y_s_std'].append(Y_s_std)
# 
# print("data fully loaded")

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
            
            Mean = []
            
            for k in range(len(Labels)):
                Mean_new = []
                for x1 in dict['X_d'][k]:
                    min = 10
                    l_min = 0
                    for l in range(np.shape(dict['X_d'][0])[0]):
                        x2 = dict['X_d'][0][l]
                        if (abs(x1 - x2) < min):
                            l_min = l
                            min = abs(x1 - x2)
                    Mean_new.append(dict['Y_d'][0][l_min])
                Mean_new = np.array(Mean_new)
                Mean.append(Mean_new)
            
            if (i == 0):
                subplot.set_ylabel('$N_{d}$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**4, 10**6)
                
                for k in range(len(Labels)):
                    subplot.plot(dict['X_d'][k], dict['Y_d'][k], Linestyles[k], label = Labels[k])
                    subplot.fill_between(x = dict['X_d'][k], y1 = dict['Y_d'][k] + dict['Y_d_std'][k], y2 = dict['Y_d'][k] - dict['Y_d_std'][k], color = Linestyles[k], alpha = 0.2)
                
                subplot.legend()
            
            elif (i == 1):
                subplot.set_ylabel('$\Delta N_{d} / N_{d}$')
                subplot.set_ylim(-0.01, 0.01)
                
                for k in range(len(Labels)):
                    subplot.plot(dict['X_d'][k], (dict['Y_d'][k] - Mean[k]) / Mean[k], Linestyles[k], label = Labels[k])
                    subplot.fill_between(x = dict['X_d'][k], y1 = (dict['Y_d'][k] + dict['Y_d_std'][k] - Mean[k]) / Mean[k], y2 = (dict['Y_d'][k] - dict['Y_d_std'][k] - Mean[k]) / Mean[k], color = Linestyles[k], alpha = 0.2)
                
                subplot.legend()
        
        elif (j == 1):
            subplot.set_xlabel('$l$')
            subplot.set_xscale('log')
            subplot.set_xlim(0.5, 20)
            
            Mean = []
            
            for k in range(len(Labels)):
                Mean_new = []
                for x1 in dict['X_l'][k]:
                    min = 10
                    l_min = 0
                    for l in range(np.shape(dict['X_l'][0])[0]):
                        x2 = dict['X_l'][0][l]
                        if (abs(x1 - x2) < min):
                            l_min = l
                            min = abs(x1 - x2)
                    Mean_new.append(dict['Y_l'][0][l_min])
                Mean_new = np.array(Mean_new)
                Mean.append(Mean_new)
            
            if (i == 0):
                subplot.set_ylabel('$N_{l}$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**3, 10**5)
                
                for k in range(len(Labels)):
                    subplot.plot(dict['X_l'][k], dict['Y_l'][k], Linestyles[k], label = Labels[k])
                    subplot.fill_between(x = dict['X_l'][k], y1 = dict['Y_l'][k] + dict['Y_l_std'][k], y2 = dict['Y_l'][k] - dict['Y_l_std'][k], color = Linestyles[k], alpha = 0.2)
                
                subplot.legend()
            
            elif (i == 1):
                subplot.set_ylabel('$\Delta N_{l} / N_{l}$')
                subplot.set_ylim(-0.01, 0.01)
                
                for k in range(len(Labels)):
                    subplot.plot(dict['X_l'][k], (dict['Y_l'][k] - Mean[k]) / Mean[k], Linestyles[k], label = Labels[k])
                    subplot.fill_between(x = dict['X_l'][k], y1 = (dict['Y_l'][k] + dict['Y_l_std'][k] - Mean[k]) / Mean[k], y2 = (dict['Y_l'][k] - dict['Y_l_std'][k] - Mean[k]) / Mean[k], color = Linestyles[k], alpha = 0.2)
                
                subplot.legend()
            
        elif (j == 2):
            subplot.set_xlabel('$b$')
            subplot.set_xscale('log')
            subplot.set_xlim(2, 50)
            
            Mean = []
            
            for k in range(len(Labels)):
                Mean_new = []
                for x1 in dict['X_b'][k]:
                    min = 10
                    l_min = 0
                    for l in range(np.shape(dict['X_b'][0])[0]):
                        x2 = dict['X_b'][0][l]
                        if (abs(x1 - x2) < min):
                            l_min = l
                            min = abs(x1 - x2)
                    Mean_new.append(dict['Y_b'][0][l_min])
                Mean_new = np.array(Mean_new)
                Mean.append(Mean_new)
            
            if (i == 0):
                subplot.set_ylabel('$N_{b}$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**3, 2 * 10**4)
                
                for k in range(len(Labels)):
                    subplot.plot(dict['X_b'][k], dict['Y_b'][k], Linestyles[k], label = Labels[k])
                    subplot.fill_between(x = dict['X_b'][k], y1 = dict['Y_b'][k] + dict['Y_b_std'][k], y2 = dict['Y_b'][k] - dict['Y_b_std'][k], color = Linestyles[k], alpha = 0.2)
                
                subplot.legend()
            
            elif (i == 1):
                subplot.set_ylabel('$\Delta N_{b} / N_{b}$')
                subplot.set_ylim(-0.01, 0.01)
                
                for k in range(len(Labels)):
                    subplot.plot(dict['X_b'][k], (dict['Y_b'][k] - Mean[k]) / Mean[k], Linestyles[k], label = Labels[k])
                    subplot.fill_between(x = dict['X_b'][k], y1 = (dict['Y_b'][k] + dict['Y_b_std'][k] - Mean[k]) / Mean[k], y2 = (dict['Y_b'][k] - dict['Y_b_std'][k] - Mean[k]) / Mean[k], color = Linestyles[k], alpha = 0.2)
                
                subplot.legend()
        
        else:
            subplot.set_xlabel('$s$')
            subplot.set_xlim(0.2, 0.8)
            
            Mean = []
            
            for k in range(len(Labels)):
                Mean_new = []
                for x1 in dict['X_s'][k]:
                    min = 10
                    l_min = 0
                    for l in range(np.shape(dict['X_s'][0])[0]):
                        x2 = dict['X_s'][0][l]
                        if (abs(x1 - x2) < min):
                            l_min = l
                            min = abs(x1 - x2)
                    Mean_new.append(dict['Y_s'][0][l_min])
                Mean_new = np.array(Mean_new)
                Mean.append(Mean_new)
            
            if (i == 0):
                subplot.set_ylabel('$N_{s}$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**3, 2 * 10**4)
                
                for k in range(len(Labels)):
                    subplot.plot(dict['X_s'][k], dict['Y_s'][k], Linestyles[k], label = Labels[k])
                    subplot.fill_between(x = dict['X_s'][k], y1 = dict['Y_s'][k] + dict['Y_s_std'][k], y2 = dict['Y_s'][k] - dict['Y_s_std'][k], color = Linestyles[k], alpha = 0.2)
                
                subplot.legend()
            
            elif (i == 1):
                subplot.set_ylabel('$\Delta N_{s} / N_{s}$')
                subplot.set_ylim(-0.01, 0.01)
                
                for k in range(len(Labels)):
                    subplot.plot(dict['X_s'][k], (dict['Y_s'][k] - Mean[k]) / Mean[k], Linestyles[k], label = Labels[k])
                    subplot.fill_between(x = dict['X_s'][k], y1 = (dict['Y_s'][k] + dict['Y_s_std'][k] - Mean[k]) / Mean[k], y2 = (dict['Y_s'][k] - dict['Y_s_std'][k] - Mean[k]) / Mean[k], color = Linestyles[k], alpha = 0.2)
                
                subplot.legend()

plt.suptitle("Effect of the masks on the random catalogue")
plt.show()