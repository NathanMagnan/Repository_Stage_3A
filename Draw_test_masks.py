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

dict = {'X_d' : [], 'Y_d' : [], 'X_l' : [], 'Y_l' : [], 'X_b' : [], 'Y_b' : [], 'X_s' : [], 'Y_s' : []}
   
X_d = np.loadtxt(str(target) + 'Full_Abacus' + "_X_d")
Y_d = np.loadtxt(str(target) + 'Full_Abacus' + "_Y_d")
dict['X_d'].append(X_d)
dict['Y_d'].append(Y_d)

X_l = np.loadtxt(str(target) + 'Full_Abacus' + "_X_l")
Y_l = np.loadtxt(str(target) + 'Full_Abacus' + "_Y_l")
dict['X_l'].append(X_l)
dict['Y_l'].append(Y_l)

X_b = np.loadtxt(str(target) + 'Full_Abacus' + "_X_b")
Y_b = np.loadtxt(str(target) + 'Full_Abacus' + "_Y_b")
dict['X_b'].append(X_b)
dict['Y_b'].append(Y_b)

X_s = np.loadtxt(str(target) + 'Full_Abacus' + "_X_s")
Y_s = np.loadtxt(str(target) + 'Full_Abacus' + "_Y_s")
dict['X_s'].append(X_s)
dict['Y_s'].append(Y_s)

X_d = np.loadtxt(str(target) + 'Masked_Abacus' + "_X_d")
Y_d = np.loadtxt(str(target) + 'Masked_Abacus' + "_Y_d")
dict['X_d'].append(X_d)
dict['Y_d'].append(Y_d)

X_l = np.loadtxt(str(target) + 'Masked_Abacus' + "_X_l")
Y_l = np.loadtxt(str(target) + 'Masked_Abacus' + "_Y_l")
dict['X_l'].append(X_l)
dict['Y_l'].append(Y_l)

X_b = np.loadtxt(str(target) + 'Masked_Abacus' + "_X_b")
Y_b = np.loadtxt(str(target) + 'Masked_Abacus' + "_Y_b")
dict['X_b'].append(X_b)
dict['Y_b'].append(Y_b)

X_s = np.loadtxt(str(target) + 'Masked_Abacus' + "_X_s")
Y_s = np.loadtxt(str(target) + 'Masked_Abacus' + "_Y_s")
dict['X_s'].append(X_s)
dict['Y_s'].append(Y_s)

X_d = np.loadtxt(str(target) + 'Full_Random' + "_X_d")
Y_d = np.loadtxt(str(target) + 'Full_Random' + "_Y_d")
dict['X_d'].append(X_d)
dict['Y_d'].append(Y_d)

X_l = np.loadtxt(str(target) + 'Full_Random' + "_X_l")
Y_l = np.loadtxt(str(target) + 'Full_Random' + "_Y_l")
dict['X_l'].append(X_l)
dict['Y_l'].append(Y_l)

X_b = np.loadtxt(str(target) + 'Full_Random' + "_X_b")
Y_b = np.loadtxt(str(target) + 'Full_Random' + "_Y_b")
dict['X_b'].append(X_b)
dict['Y_b'].append(Y_b)

X_s = np.loadtxt(str(target) + 'Full_Random' + "_X_s")
Y_s = np.loadtxt(str(target) + 'Full_Random' + "_Y_s")
dict['X_s'].append(X_s)
dict['Y_s'].append(Y_s)

X_d = np.loadtxt(str(target) + 'Masked_Random' + "_X_d")
Y_d = np.loadtxt(str(target) + 'Masked_Random' + "_Y_d")
dict['X_d'].append(X_d)
dict['Y_d'].append(Y_d)

X_l = np.loadtxt(str(target) + 'Masked_Random' + "_X_l")
Y_l = np.loadtxt(str(target) + 'Masked_Random' + "_Y_l")
dict['X_l'].append(X_l)
dict['Y_l'].append(Y_l)

X_b = np.loadtxt(str(target) + 'Masked_Random' + "_X_b")
Y_b = np.loadtxt(str(target) + 'Masked_Random' + "_Y_b")
dict['X_b'].append(X_b)
dict['Y_b'].append(Y_b)

X_s = np.loadtxt(str(target) + 'Masked_Random' + "_X_s")
Y_s = np.loadtxt(str(target) + 'Masked_Random' + "_Y_s")
dict['X_s'].append(X_s)
dict['Y_s'].append(Y_s)

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
            
            Mean_abacus_masked = []
            for x1 in dict['X_d'][1]:
                min = 10
                k_min = 0
                for k in range(np.shape(dict['X_d'][0])[0]):
                    x2 = dict['X_d'][0][k]
                    if (abs(x1 - x2) < min):
                        k_min = k
                        min = abs(x1 - x2)
                Mean_abacus_masked.append(dict['Y_d'][0][k_min])
            Mean_abacus_masked = np.array(Mean_abacus_masked)
            
            if (i == 0):
                subplot.set_ylabel('$N_{d}$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**4, 10**6)
                
                subplot.plot(dict['X_d'][0], dict['Y_d'][0], 'b', label = 'Abacus')
                subplot.plot(dict['X_d'][1], dict['Y_d'][1], 'b--', label = 'Abacus + Mask')
                
                subplot.legend()
            if (i == 1):
                subplot.set_ylabel('$\Delta N_{d} / N_{d}$')
                subplot.set_ylim(-0.3, 0.3)
                
                subplot.plot(dict['X_d'][0], [0 for x in dict['X_d'][0]], 'b', label = 'Abacus')
                subplot.plot(dict['X_d'][0], (dict['Y_d'][1] - Mean_abacus_masked) / Mean_abacus_masked, 'b--', label = 'Abacus + Mask')
                
                subplot.legend()
        elif (j == 1):
            subplot.set_xlabel('$l$')
            subplot.set_xscale('log')
            
            Mean_abacus_masked = []
            for x1 in dict['X_l'][1]:
                min = 10
                k_min = 0
                for k in range(np.shape(dict['X_l'][0])[0]):
                    x2 = dict['X_l'][0][k]
                    if (abs(x1 - x2) < min):
                        k_min = k
                        min = abs(x1 - x2)
                Mean_abacus_masked.append(dict['Y_l'][0][k_min])
            Mean_abacus_masked = np.array(Mean_abacus_masked)
            
            if (i == 0):
                subplot.set_ylabel('$N_{l}$')
                subplot.set_yscale('log')
                #subplot.set_ylim(10**4, 10**6)
                
                subplot.plot(dict['X_l'][0], dict['Y_l'][0], 'g', label = 'Abacus')
                subplot.plot(dict['X_l'][1], dict['Y_l'][1], 'g--', label = 'Abacus + Mask')
                
                subplot.legend()
            if (i == 1):
                subplot.set_ylabel('$\Delta N_{l} / N_{l}$')
                subplot.set_ylim(-0.3, 0.3)
                
                subplot.plot(dict['X_l'][0], [0 for x in dict['X_l'][0]], 'g', label = 'Abacus')
                subplot.plot(dict['X_l'][0], (dict['Y_l'][1] - Mean_abacus_masked) / Mean_abacus_masked, 'g--', label = 'Abacus + Mask')
                
                subplot.legend()
        elif (j == 2):
            subplot.set_xlabel('$b$')
            subplot.set_xscale('log')
            
            Mean_abacus_masked = []
            for x1 in dict['X_b'][1]:
                min = 10
                k_min = 0
                for k in range(np.shape(dict['X_b'][0])[0]):
                    x2 = dict['X_b'][0][k]
                    if (abs(x1 - x2) < min):
                        k_min = k
                        min = abs(x1 - x2)
                Mean_abacus_masked.append(dict['Y_b'][0][k_min])
            Mean_abacus_masked = np.array(Mean_abacus_masked)
            
            if (i == 0):
                subplot.set_ylabel('$N_{b}$')
                subplot.set_yscale('log')
                #subplot.set_ylim(10**4, 10**6)
                
                subplot.plot(dict['X_b'][0], dict['Y_b'][0], 'r', label = 'Abacus')
                subplot.plot(dict['X_b'][1], dict['Y_b'][1], 'r--', label = 'Abacus + Mask')
                
                subplot.legend()
            if (i == 1):
                subplot.set_ylabel('$\Delta N_{b} / N_{b}$')
                subplot.set_ylim(-0.3, 0.3)
                
                subplot.plot(dict['X_b'][0], [0 for x in dict['X_b'][0]], 'r', label = 'Abacus')
                subplot.plot(dict['X_b'][0], (dict['Y_b'][1] - Mean_abacus_masked) / Mean_abacus_masked, 'r--', label = 'Abacus + Mask')
                
                subplot.legend()
        else:
            subplot.set_xlabel('$s$')
            
            Mean_abacus_masked = []
            for x1 in dict['X_s'][1]:
                min = 10
                k_min = 0
                for k in range(np.shape(dict['X_s'][0])[0]):
                    x2 = dict['X_s'][0][k]
                    if (abs(x1 - x2) < min):
                        k_min = k
                        min = abs(x1 - x2)
                Mean_abacus_masked.append(dict['Y_s'][0][k_min])
            Mean_abacus_masked = np.array(Mean_abacus_masked)
            
            if (i == 0):
                subplot.set_ylabel('$N_{s}$')
                subplot.set_yscale('log')
                #subplot.set_ylim(10**4, 10**6)
                
                subplot.plot(dict['X_s'][0], dict['Y_s'][0], 'y', label = 'Abacus')
                subplot.plot(dict['X_s'][1], dict['Y_s'][1], 'y--', label = 'Abacus + Mask')
                
                subplot.legend()
            if (i == 1):
                subplot.set_ylabel('$\Delta N_{s} / N_{s}$')
                subplot.set_ylim(-0.3, 0.3)
                
                subplot.plot(dict['X_s'][0], [0 for x in dict['X_s'][0]], 'y', label = 'Abacus')
                subplot.plot(dict['X_s'][0], (dict['Y_s'][1] - Mean_abacus_masked) / Mean_abacus_masked, 'y--', label = 'Abacus + Mask')
                
                subplot.legend()

plt.suptitle("Effect of a spherical mask ($r = 100$ Mpc / $h$) on an Abacus simulation")
plt.show()