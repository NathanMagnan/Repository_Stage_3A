## Imports
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)

print("all imports sucessfull")

## loading the histograms
print("starting to load the histograms")

Histograms_by_positions = [[] for i in range(4)]

for n_simu in range(21):
    for i in range(4):
        for j in range(4):
            for k in range(4):
                # getting the number of the box
                n_box = 16 * i + 4 * j + k + 1
                
                # finding the position of the box
                m = 0
                if ((i == 0) or (i == 3)):
                    m += 1
                if ((j == 0) or (j == 3)):
                    m += 1
                if ((k == 0) or (k == 3)):
                    m += 1
                
                # reading the histograms
                path = r'C:\Users\Nathan\Documents\D - X\C - Stages\Stage 3A\Repository_Stage_3A\Full_MST_stats_Abacus_3'
                file = 'MST_stats_Simulation_' + str(n_simu) + '_Box_' + str(n_box) + '.pkl'
                my_file = os.path.join(path, file)
                
                f = open(my_file, "rb")
                New_histogram = pickle.load(f)
                f.close()
                
                # adding the histograms to the right data sets
                Histograms_by_positions[m].append(New_histogram)

print("histograms loaded")

## Finding the mean and std for each positions
print("starting to analyse the effect of position")

# choosing a common x axis
X_d = Histograms_by_positions[0][0]['x_d']
X_l = Histograms_by_positions[0][0]['x_l']
X_b = Histograms_by_positions[0][0]['x_b']
X_s = Histograms_by_positions[0][0]['x_s']

Mean_d = [[] for i in range(4)]
Std_d = [[] for i in range(4)]
for i in range(4):
    
    Mean_d[i] = np.asarray([0 for x1 in X_d])
    Std_d[i] = np.asarray([0 for x1 in X_d])
    
    for j in range(np.shape(Histograms_by_positions[i])[0]):
        New = [0 for x1 in X_d]
        
        for k in range(np.shape(X_d)[0]):
            x1 = X_d[k]
            min = 10
            l_min = 0
            for l in range(np.shape(Histograms_by_positions[i][j]['x_d'])[0]):
                x2 = Histograms_by_positions[i][j]['x_d'][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            New[k] = Histograms_by_positions[i][j]['y_d'][l_min]
        New = np.asarray(New)
        
        Mean_old = Mean_d[i].copy()
        Std_old = Std_d[i].copy()
        
        Mean_d[i] = (j * Mean_old + New) / (j + 1)
        Std_d[i] = np.sqrt((j * (Std_old**2 + Mean_old**2) + New**2) / (j + 1) - Mean_d[i]**2)

Mean_l = [[] for i in range(4)]
Std_l = [[] for i in range(4)]
for i in range(4):
    
    Mean_l[i] = np.asarray([0 for x1 in X_l])
    Std_l[i] = np.asarray([0 for x1 in X_l])
    
    for j in range(np.shape(Histograms_by_positions[i])[0]):
        New = [0 for x1 in X_l]
        
        for k in range(np.shape(X_l)[0]):
            x1 = X_l[k]
            min = 10
            l_min = 0
            for l in range(np.shape(Histograms_by_positions[i][j]['x_l'])[0]):
                x2 = Histograms_by_positions[i][j]['x_l'][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            New[k] = Histograms_by_positions[i][j]['y_l'][l_min]
        New = np.asarray(New)
        
        Mean_old = Mean_l[i].copy()
        Std_old = Std_l[i].copy()
        
        Mean_l[i] = (j * Mean_old + New) / (j + 1)
        Std_l[i] = np.sqrt((j * (Std_old**2 + Mean_old**2) + New**2) / (j + 1) - Mean_l[i]**2)

Mean_b = [[] for i in range(4)]
Std_b = [[] for i in range(4)]
for i in range(4):
    
    Mean_b[i] = np.asarray([0 for x1 in X_b])
    Std_b[i] = np.asarray([0 for x1 in X_b])
    
    for j in range(np.shape(Histograms_by_positions[i])[0]):
        New = [0 for x1 in X_b]
        
        for k in range(np.shape(X_b)[0]):
            x1 = X_b[k]
            min = 10
            l_min = 0
            for l in range(np.shape(Histograms_by_positions[i][j]['x_b'])[0]):
                x2 = Histograms_by_positions[i][j]['x_b'][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            New[k] = Histograms_by_positions[i][j]['y_b'][l_min]
        New = np.asarray(New)
        
        Mean_old = Mean_b[i].copy()
        Std_old = Std_b[i].copy()
        
        Mean_b[i] = (j * Mean_old + New) / (j + 1)
        Std_b[i] = np.sqrt((j * (Std_old**2 + Mean_old**2) + New**2) / (j + 1) - Mean_b[i]**2)

Mean_s = [[] for i in range(4)]
Std_s = [[] for i in range(4)]
for i in range(4):
    
    Mean_s[i] = np.asarray([0 for x1 in X_s])
    Std_s[i] = np.asarray([0 for x1 in X_s])
    
    for j in range(np.shape(Histograms_by_positions[i])[0]):
        New = [0 for x1 in X_s]
        
        for k in range(np.shape(X_s)[0]):
            x1 = X_s[k]
            min = 10
            l_min = 0
            for l in range(np.shape(Histograms_by_positions[i][j]['x_s'])[0]):
                x2 = Histograms_by_positions[i][j]['x_s'][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            New[k] = Histograms_by_positions[i][j]['y_s'][l_min]
        New = np.asarray(New)
        
        Mean_old = Mean_s[i].copy()
        Std_old = Std_s[i].copy()
        
        Mean_s[i] = (j * Mean_old + New) / (j + 1)
        Std_s[i] = np.sqrt((j * (Std_old**2 + Mean_old**2) + New**2) / (j + 1) - Mean_s[i]**2)

print("Effects of position analyzed")

## Plotting the effects of position
print("starting to plot the effect of position")

Labels = ['center', 'face', 'edge', 'corner']
Colors = ['b', 'y', 'g', 'r']

fig, axes = plt.subplots(nrows = 2, ncols = 4, figsize = (20, 10))
plt.subplots_adjust(hspace = 0.3, wspace = 0.4)

for a in range(4):
    for b in range(2):
        subplot = axes[b][a]
        
        if (a == 0):
            subplot.set_xlabel('$d$')
            #subplot.set_xlim(1, 6)
            
            if (b == 0):
                subplot.set_ylabel('$N_{d}$')
                subplot.set_yscale('log')
                #subplot.set_ylim(10**4, 10**6)
                
                for i in range(4):
                    subplot.fill_between(x = X_d, y1 = Mean_d[i] - Std_d[i], y2 = Mean_d[i] + Std_d[i], color = Colors[i], alpha = 0.2, label = Labels[i])
                
                subplot.legend()
                
            else:
                subplot.set_ylabel('$\Delta N_{d} / <N_{d}>$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**-5, 10**0)
                
                for i in range(4):
                    subplot.fill_between(x = X_d, y1 = (Mean_d[i] - Mean_d[0] - Std_d[i]) / Mean_d[0], y2 = (Mean_d[i] - Mean_d[0] + Std_d[i]) / Mean_d[0], color = Colors[i], alpha = 0.2, label = Labels[i])
                    subplot.plot(X_d, np.abs(Mean_d[i] - Mean_d[0]) / Mean_d[0], color = Colors[i])
                
                subplot.legend()
            
        elif (a == 1):
            subplot.set_xlabel('$l$')
            subplot.set_xscale('log')
            
            if (b == 0):
                subplot.set_ylabel('$N_{l}$')
                subplot.set_yscale('log')
                #subplot.set_ylim(10**4, 10**6)
                
                for i in range(4):
                    subplot.fill_between(x = X_l, y1 = Mean_l[i] - Std_l[i], y2 = Mean_l[i] + Std_l[i], color = Colors[i], alpha = 0.2, label = Labels[i])
                
                subplot.legend()
                
            else:
                subplot.set_ylabel('$\Delta N_{l} / <N_{l}>$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**-5, 10**0)
                
                for i in range(4):
                    subplot.fill_between(x = X_l, y1 = (Mean_l[i] - Mean_l[0] - Std_l[i]) / Mean_l[0], y2 = (Mean_l[i] - Mean_l[0] + Std_l[i]) / Mean_l[0], color = Colors[i], alpha = 0.2, label = Labels[i])
                    subplot.plot(X_l, np.abs(Mean_l[i] - Mean_l[0]) / Mean_l[0], color = Colors[i])
                
                subplot.legend()
        
        elif (a == 2):
            subplot.set_xlabel('$b$')
            subplot.set_xscale('log')
            
            if (b == 0):
                subplot.set_ylabel('$N_{b}$')
                subplot.set_yscale('log')
                #subplot.set_ylim(10**4, 10**6)
                
                for i in range(4):
                    subplot.fill_between(x = X_b, y1 = Mean_b[i] - Std_b[i], y2 = Mean_b[i] + Std_b[i], color = Colors[i], alpha = 0.2, label = Labels[i])
                
                subplot.legend()
                
            else:
                subplot.set_ylabel('$\Delta N_{b} / <N_{b}>$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**-5, 10**0)
                
                for i in range(4):
                    subplot.fill_between(x = X_b, y1 = (Mean_b[i] - Mean_b[0] - Std_b[i]) / Mean_b[0], y2 = (Mean_b[i] - Mean_b[0] + Std_b[i]) / Mean_b[0], color = Colors[i], alpha = 0.2, label = Labels[i])
                    subplot.plot(X_b, np.abs(Mean_b[i] - Mean_b[0]) / Mean_b[0], color = Colors[i])
                
                subplot.legend()
        
        else:
            subplot.set_xlabel('$s$')
            
            if (b == 0):
                subplot.set_ylabel('$N_{s}$')
                subplot.set_yscale('log')
                #subplot.set_ylim(10**4, 10**6)
                
                for i in range(4):
                    subplot.fill_between(x = X_s, y1 = Mean_s[i] - Std_s[i], y2 = Mean_s[i] + Std_s[i], color = Colors[i], alpha = 0.2, label = Labels[i])
                
                subplot.legend()
                
            else:
                subplot.set_ylabel('$\Delta N_{s} / <N_{s}>$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**-5, 10**0)
                
                for i in range(4):
                    subplot.fill_between(x = X_s, y1 = (Mean_s[i] - Mean_s[0] - Std_s[i]) / Mean_s[0], y2 = (Mean_s[i] - Mean_s[0] + Std_s[i]) / Mean_s[0], color = Colors[i], alpha = 0.2, label = Labels[i])
                    subplot.plot(X_s, np.abs(Mean_s[i] - Mean_s[0]) / Mean_s[0], color = Colors[i])
                
                subplot.legend()

plt.suptitle("Effect of removing boxes on the MST statistics")
plt.show()