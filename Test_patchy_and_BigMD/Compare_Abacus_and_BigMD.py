## Imports
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)

print("all imports sucessfull")

## Loading BigMD
print("print starting to load BigMD")

path = r'C:\Users\Nathan\Documents\D - X\C - Stages\Stage 3A\Repository_Stage_3A\Full_MST_stats_Patchy'
file = 'BigMD' + '.pkl'
my_file = os.path.join(path, file)

f = open(my_file, "rb")
BigMD = pickle.load(f)
f.close()

print("BigMD loaded")

## Loading Abacus
print("print starting to load Abacus")

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

print("Abacus loaded")

## Choosing a common x axis

X_d = Histograms_full[0]['x_d']
X_l = Histograms_full[0]['x_l']
X_b = Histograms_full[0]['x_b']
X_s = Histograms_full[0]['x_s']

## Moving every histogram to this x axis
print("Starting to reposition every histogram")

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

## Finding the Std and the mean from Abacus
print("starting to compute for Abacus")

Mean_basic_d = np.asarray([0 for x1 in X_d])
Std_basic_d = np.asarray([0 for x1 in X_d])
for i in range(21):
    New = Histograms_full[i]['y_d']
    Mean_old = Mean_basic_d.copy()
    Std_old = Std_basic_d.copy()
    
    Mean_basic_d = (i * Mean_old + New) / (i + 1)
    Std_basic_d = (i * (Std_old + Mean_old**2) + New**2) / (i + 1) - Mean_basic_d**2
Std_basic_d = (21 / 20) * Std_basic_d

Mean_basic_l = np.asarray([0 for x1 in X_l])
Std_basic_l = np.asarray([0 for x1 in X_l])
for i in range(21):
    New = Histograms_full[i]['y_l']
    Mean_old = Mean_basic_l.copy()
    Std_old = Std_basic_l.copy()
    
    Mean_basic_l = (i * Mean_old + New) / (i + 1)
    Std_basic_l = (i * (Std_old + Mean_old**2) + New**2) / (i + 1) - Mean_basic_l**2
Std_basic_l = (21 / 20) * Std_basic_l

Mean_basic_b = np.asarray([0 for x1 in X_b])
Std_basic_b = np.asarray([0 for x1 in X_b])
for i in range(21):
    New = Histograms_full[i]['y_b']
    Mean_old = Mean_basic_b.copy()
    Std_old = Std_basic_b.copy()
    
    Mean_basic_b = (i * Mean_old + New) / (i + 1)
    Std_basic_b = (i * (Std_old + Mean_old**2) + New**2) / (i + 1) - Mean_basic_b**2
Std_basic_b = (21 / 20) * Std_basic_b

Mean_basic_s = np.asarray([0 for x1 in X_s])
Std_basic_s = np.asarray([0 for x1 in X_s])
for i in range(21):
    New = Histograms_full[i]['y_s']
    Mean_old = Mean_basic_s.copy()
    Std_old = Std_basic_s.copy()
    
    Mean_basic_s = (i * Mean_old + New) / (i + 1)
    Std_basic_s = (i * (Std_old + Mean_old**2) + New**2) / (i + 1) - Mean_basic_s**2
Std_basic_s = (21 / 20) * Std_basic_s

print("Abacus ready")

## Finding the y-normalization factors
print("starting to research the y-normalization factors")

sum_abacus_d = np.sum(Mean_basic_d)
sum_bigmd_d = np.sum(BigMD['y_d'])
factor_y_d = sum_abacus_d / sum_bigmd_d

sum_abacus_l = np.sum(Mean_basic_l[50:])
sum_bigmd_l = np.sum(BigMD['y_l'][50:])
factor_y_l = sum_abacus_l / sum_bigmd_l

sum_abacus_b = np.sum(Mean_basic_b[50:])
sum_bigmd_b = np.sum(BigMD['y_b'][50:])
factor_y_b = sum_abacus_b / sum_bigmd_b

sum_abacus_s = np.sum(Mean_basic_s)
sum_bigmd_s = np.sum(BigMD['y_s'])
factor_y_s = sum_abacus_s / sum_bigmd_s

print("y-normalization found")
print(factor_y_d, factor_y_l, factor_y_b, factor_y_s)

## Finding the x-normalization factors
print("starting to research the y-normalization factors")

factor_x_d = 1
factor_x_l = 1
factor_x_b = 1
factor_x_s = 1

print("x-normalization_found")
print(factor_x_d, factor_x_l, factor_x_b, factor_x_s)

## Plotting the results
print("starting to plot")

fig, axes = plt.subplots(nrows = 1, ncols = 4, figsize = (20, 5))
plt.subplots_adjust(hspace = 0.3, wspace = 0.4)

for a in range(4):
        subplot = axes[a]
        
        if (a == 0):
            subplot.set_xlabel('$d$')
            subplot.set_xlim(1, 4)

            subplot.set_ylabel('$N_{d}$')
            subplot.set_yscale('log')
            subplot.set_ylim(10**(4), 10**(7))
            
            subplot.plot(X_d, Mean_basic_d, color = 'b', label = 'Abacus range')
            subplot.fill_between(x = X_d, y1 = (Mean_basic_d - np.sqrt(Std_basic_d)), y2 = (Mean_basic_d + np.sqrt(Std_basic_d)), color = 'b', alpha = 0.4)
            subplot.plot(BigMD['x_d'] * factor_x_d, BigMD['y_d'] * factor_y_d, color = 'k', label = 'BigMD')
            
            subplot.legend()
        
        elif (a == 1):
            subplot.set_xlabel('$l$')
            subplot.set_xscale('log')
            #subplot.set_xlim(10**(-2), 18)

            subplot.set_ylabel('$N_{l}$')
            subplot.set_yscale('log')
            #subplot.set_ylim(10**(2), 10**(5))
            
            subplot.plot(X_l, Mean_basic_l, color = 'g', label = 'Abacus range')
            subplot.fill_between(x = X_l, y1 = (Mean_basic_l - np.sqrt(Std_basic_l)), y2 = (Mean_basic_l + np.sqrt(Std_basic_l)), color = 'g', alpha = 0.4)
            subplot.plot(BigMD['x_l'] * factor_x_l, BigMD['y_l'] * factor_y_l, color = 'k', label = 'BigMD')
            
            subplot.legend()
        
        elif (a == 2):
            subplot.set_xlabel('$b$')
            subplot.set_xscale('log')
            #subplot.set_xlim(10**(-1), 80)

            subplot.set_ylabel('$N_{b}$')
            subplot.set_yscale('log')
            #subplot.set_ylim(10**(2), 10**(5))
            
            subplot.plot(X_b, Mean_basic_b, color = 'r', label = 'Abacus range')
            subplot.fill_between(x = X_b, y1 = (Mean_basic_b - np.sqrt(Std_basic_b)), y2 = (Mean_basic_b + np.sqrt(Std_basic_b)), color = 'r', alpha = 0.4)
            subplot.plot(BigMD['x_b'] * factor_x_b, BigMD['y_b'] * factor_y_b, color = 'k', label = 'BigMD')
            
            subplot.legend()
        
        else:
            subplot.set_xlabel('$s$')
            subplot.set_xlim(0, 1)

            subplot.set_ylabel('$N_{s}$')
            subplot.set_yscale('log')
            subplot.set_ylim(10**(2), 10**(5))
            
            subplot.plot(X_s, Mean_basic_s, color = 'y', label = 'Abacus range')
            subplot.fill_between(x = X_s, y1 = (Mean_basic_s - np.sqrt(Std_basic_s)), y2 = (Mean_basic_s + np.sqrt(Std_basic_s)), color = 'y', alpha = 0.4)
            subplot.plot(BigMD['x_s'] * factor_x_s, BigMD['y_s'] * factor_y_d, color = 'k', label = 'BigMD')
            
            subplot.legend()

plt.suptitle("Comparison between Abacus and BigMD")
plt.show()