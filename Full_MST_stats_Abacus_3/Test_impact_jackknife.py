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

Histograms_by_simulations = [[] for i in range(21)]
Histograms = []

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
                Histograms.append(New_histogram)

print("histograms loaded")

## Finding the mean and std for each simulation
print("starting to analyse")

# choosing a common x axis
X_d = Histograms_by_positions[0][0]['x_d']
X_l = Histograms_by_positions[0][0]['x_l']
X_b = Histograms_by_positions[0][0]['x_b']
X_s = Histograms_by_positions[0][0]['x_s']

# finding the mean and std of each simulation
Mean_d = [[] for i in range(21)]
Std_d = [[] for i in range(21)]
for i in range(21):
    
    Mean_d[i] = np.asarray([0 for x1 in X_d])
    Std_d[i] = np.asarray([0 for x1 in X_d])
    
    for j in range(64):
        New = [0 for x1 in X_d]
        
        for k in range(np.shape(X_d)[0]):
            x1 = X_d[k]
            min = 10
            l_min = 0
            for l in range(np.shape(Histograms_by_simulations[i][j]['x_d'])[0]):
                x2 = Histograms_by_simulations[i][j]['x_d'][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            New[k] = Histograms_by_simulations[i][j]['y_d'][l_min]
        New = np.asarray(New)
        
        Mean_old = Mean_d[i].copy()
        Std_old = Std_d[i].copy()
        
        Mean_d[i] = (j * Mean_old + New) / (j + 1)
        Std_d[i] = np.sqrt((j * (Std_old**2 + Mean_old**2) + New**2) / (j + 1) - Mean_d[i]**2)

Mean_l = [[] for i in range(21)]
Std_l = [[] for i in range(21)]
for i in range(21):
    
    Mean_l[i] = np.asarray([0 for x1 in X_l])
    Std_l[i] = np.asarray([0 for x1 in X_l])
    
    for j in range(64):
        New = [0 for x1 in X_l]
        
        for k in range(np.shape(X_l)[0]):
            x1 = X_l[k]
            min = 10
            l_min = 0
            for l in range(np.shape(Histograms_by_simulations[i][j]['x_l'])[0]):
                x2 = Histograms_by_simulations[i][j]['x_l'][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            New[k] = Histograms_by_simulations[i][j]['y_l'][l_min]
        New = np.asarray(New)
        
        Mean_old = Mean_l[i].copy()
        Std_old = Std_l[i].copy()
        
        Mean_l[i] = (j * Mean_old + New) / (j + 1)
        Std_l[i] = np.sqrt((j * (Std_old**2 + Mean_old**2) + New**2) / (j + 1) - Mean_l[i]**2)

Mean_b = [[] for i in range(21)]
Std_b = [[] for i in range(21)]
for i in range(21):
    
    Mean_b[i] = np.asarray([0 for x1 in X_b])
    Std_b[i] = np.asarray([0 for x1 in X_b])
    
    for j in range(64):
        New = [0 for x1 in X_b]
        
        for k in range(np.shape(X_b)[0]):
            x1 = X_b[k]
            min = 10
            l_min = 0
            for l in range(np.shape(Histograms_by_simulations[i][j]['x_b'])[0]):
                x2 = Histograms_by_simulations[i][j]['x_b'][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            New[k] = Histograms_by_simulations[i][j]['y_b'][l_min]
        New = np.asarray(New)
        
        Mean_old = Mean_b[i].copy()
        Std_old = Std_b[i].copy()
        
        Mean_b[i] = (j * Mean_old + New) / (j + 1)
        Std_b[i] = np.sqrt((j * (Std_old**2 + Mean_old**2) + New**2) / (j + 1) - Mean_b[i]**2)

Mean_s = [[] for i in range(21)]
Std_s = [[] for i in range(21)]
for i in range(21):
    
    Mean_s[i] = np.asarray([0 for x1 in X_s])
    Std_s[i] = np.asarray([0 for x1 in X_s])
    
    for j in range(64):
        New = [0 for x1 in X_s]
        
        for k in range(np.shape(X_s)[0]):
            x1 = X_s[k]
            min = 10
            l_min = 0
            for l in range(np.shape(Histograms_by_simulations[i][j]['x_s'])[0]):
                x2 = Histograms_by_simulations[i][j]['x_s'][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            New[k] = Histograms_by_simulations[i][j]['y_s'][l_min]
        New = np.asarray(New)
        
        Mean_old = Mean_s[i].copy()
        Std_old = Std_s[i].copy()
        
        Mean_s[i] = (j * Mean_old + New) / (j + 1)
        Std_s[i] = np.sqrt((j * (Std_old**2 + Mean_old**2) + New**2) / (j + 1) - Mean_s[i]**2)

print("first part of analysis done")

## Finding the mean of the stds and the std of the means
print("starting second part of the analysis")

Mean_simu_d = np.asarray([0 for x1 in X_d])
Std_simu_d = np.asarray([0 for x1 in X_d])
Std_jackknife_d = np.asarray([0 for x1 in X_d])
for i in range(21):
    New_mean = Mean_d[i]
    New_std = Std_d[i]**2
    
    Mean_simu_old = Mean_simu_d.copy()
    Std_simu_old = Std_simu_d.copy()
    Mean_simu_d = (i * Mean_simu_old + New_mean) / (i + 1)
    Std_simu_d = np.sqrt((i * (Std_simu_old**2 + Mean_simu_old**2) + New_mean**2) / (i + 1) - Mean_simu_d**2)
    
    Std_jackknife_d = (i * Std_jackknife_d + New_std) / (i + 1)
Std_jackknife_d = np.sqrt(Std_jackknife_d)

Mean_simu_l = np.asarray([0 for x1 in X_l])
Std_simu_l = np.asarray([0 for x1 in X_l])
Std_jackknife_l = np.asarray([0 for x1 in X_l])
for i in range(21):
    New_mean = Mean_l[i]
    New_std = Std_l[i]**2
    
    Mean_simu_old = Mean_simu_l.copy()
    Std_simu_old = Std_simu_l.copy()
    Mean_simu_l = (i * Mean_simu_old + New_mean) / (i + 1)
    Std_simu_l = np.sqrt((i * (Std_simu_old**2 + Mean_simu_old**2) + New_mean**2) / (i + 1) - Mean_simu_l**2)
    
    Std_jackknife_l = (i * Std_jackknife_l + New_std) / (i + 1)
Std_jackknife_l = np.sqrt(Std_jackknife_l)

Mean_simu_b = np.asarray([0 for x1 in X_b])
Std_simu_b = np.asarray([0 for x1 in X_b])
Std_jackknife_b = np.asarray([0 for x1 in X_b])
for i in range(21):
    New_mean = Mean_b[i]
    New_std = Std_b[i]**2
    
    Mean_simu_old = Mean_simu_b.copy()
    Std_simu_old = Std_simu_b.copy()
    Mean_simu_b = (i * Mean_simu_old + New_mean) / (i + 1)
    Std_simu_b = np.sqrt((i * (Std_simu_old**2 + Mean_simu_old**2) + New_mean**2) / (i + 1) - Mean_simu_b**2)
    
    Std_jackknife_b = (i * Std_jackknife_b + New_std) / (i + 1)
Std_jackknife_b = np.sqrt(Std_jackknife_b)

Mean_simu_s = np.asarray([0 for x1 in X_s])
Std_simu_s = np.asarray([0 for x1 in X_s])
Std_jackknife_s = np.asarray([0 for x1 in X_s])
for i in range(21):
    New_mean = Mean_s[i]
    New_std = Std_s[i]**2
    
    Mean_simu_old = Mean_simu_s.copy()
    Std_simu_old = Std_simu_s.copy()
    Mean_simu_s = (i * Mean_simu_old + New_mean) / (i + 1)
    Std_simu_s = np.sqrt((i * (Std_simu_old**2 + Mean_simu_old**2) + New_mean**2) / (i + 1) - Mean_simu_s**2)
    
    Std_jackknife_s = (i * Std_jackknife_s + New_std) / (i + 1)
Std_jackknife_s = np.sqrt(Std_jackknife_s)

print("analysis finished")

## Finding the overall std
print("starting to compute the overall std")

Mean_total_d = np.asarray([0 for x1 in X_d])
Std_total_d = np.asarray([0 for x1 in X_d])
for i in range(np.shape(Histograms)[0]):
        New = [0 for x1 in X_d]
        
        for k in range(np.shape(X_d)[0]):
            x1 = X_d[k]
            min = 10
            l_min = 0
            for l in range(np.shape(Histograms[i]['x_d'])[0]):
                x2 = Histograms[i]['x_d'][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            New[k] = Histograms[i]['y_d'][l_min]
        New = np.asarray(New)
        
        Mean_old = Mean_total_d.copy()
        Std_old = Std_total_d.copy()
        
        Mean_total_d = (i * Mean_old + New) / (i + 1)
        Std_total_d = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_total_d**2)

Mean_total_l = np.asarray([0 for x1 in X_l])
Std_total_l = np.asarray([0 for x1 in X_l])
for i in range(np.shape(Histograms)[0]):
        New = [0 for x1 in X_l]
        
        for k in range(np.shape(X_l)[0]):
            x1 = X_l[k]
            min = 10
            l_min = 0
            for l in range(np.shape(Histograms[i]['x_l'])[0]):
                x2 = Histograms[i]['x_l'][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            New[k] = Histograms[i]['y_l'][l_min]
        New = np.asarray(New)
        
        Mean_old = Mean_total_l.copy()
        Std_old = Std_total_l.copy()
        
        Mean_total_l = (i * Mean_old + New) / (i + 1)
        Std_total_l = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_total_l**2)

Mean_total_b = np.asarray([0 for x1 in X_b])
Std_total_b = np.asarray([0 for x1 in X_b])
for i in range(np.shape(Histograms)[0]):
        New = [0 for x1 in X_b]
        
        for k in range(np.shape(X_b)[0]):
            x1 = X_b[k]
            min = 10
            l_min = 0
            for l in range(np.shape(Histograms[i]['x_b'])[0]):
                x2 = Histograms[i]['x_b'][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            New[k] = Histograms[i]['y_b'][l_min]
        New = np.asarray(New)
        
        Mean_old = Mean_total_b.copy()
        Std_old = Std_total_b.copy()
        
        Mean_total_b = (i * Mean_old + New) / (i + 1)
        Std_total_b = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_total_b**2)

Mean_total_s = np.asarray([0 for x1 in X_s])
Std_total_s = np.asarray([0 for x1 in X_s])
for i in range(np.shape(Histograms)[0]):
        New = [0 for x1 in X_s]
        
        for k in range(np.shape(X_s)[0]):
            x1 = X_s[k]
            min = 10
            l_min = 0
            for l in range(np.shape(Histograms[i]['x_s'])[0]):
                x2 = Histograms[i]['x_s'][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            New[k] = Histograms[i]['y_s'][l_min]
        New = np.asarray(New)
        
        Mean_old = Mean_total_s.copy()
        Std_old = Std_total_s.copy()
        
        Mean_total_s = (i * Mean_old + New) / (i + 1)
        Std_total_s = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_total_s**2)

print("overall std computed")

## Plotting the results
print("starting to plot")

fig, axes = plt.subplots(nrows = 1, ncols = 4, figsize = (20, 5))
plt.subplots_adjust(hspace = 0.3, wspace = 0.4)

for a in range(4):
        subplot = axes[a]
        
        if (a == 0):
            subplot.set_xlabel('$d$')

            subplot.set_ylabel('$\sigma_{d} / <N_{d}>$')
            subplot.set_yscale('log')
            subplot.set_ylim(10**(-4), 10**0)
            
            subplot.plot(X_d, Std_jackknife_d / Mean_total_d, color = 'b', label = 'Jackknife')
            subplot.plot(X_d, Std_simu_d / Mean_total_d, color = 'r', label = 'Simulation')
            subplot.plot(X_d, Std_total_d / Mean_total_d, color = 'k', label = 'Total')
            
            subplot.legend()
        
        elif (a == 1):
            subplot.set_xlabel('$l$')
            subplot.set_xscale('log')

            subplot.set_ylabel('$\sigma_{l} / <N_{l}>$')
            subplot.set_yscale('log')
            subplot.set_ylim(10**(-4), 10**0)
            
            subplot.plot(X_l, Std_jackknife_l / Mean_total_l, color = 'b', label = 'Jackknife')
            subplot.plot(X_l, Std_simu_l / Mean_total_l, color = 'r', label = 'Simulation')
            subplot.plot(X_l, Std_total_l / Mean_total_l, color = 'k', label = 'Total')
            
            subplot.legend()
        
        elif (a == 2):
            subplot.set_xlabel('$b$')
            subplot.set_xscale('log')

            subplot.set_ylabel('$\sigma_{b} / <N_{b}>$')
            subplot.set_yscale('log')
            subplot.set_ylim(10**(-4), 10**0)
            
            subplot.plot(X_b, Std_jackknife_b / Mean_total_b, color = 'b', label = 'Jackknife')
            subplot.plot(X_b, Std_simu_b / Mean_total_b, color = 'r', label = 'Simulation')
            subplot.plot(X_b, Std_total_b / Mean_total_b, color = 'k', label = 'Total')
            
            subplot.legend()
        
        else:
            subplot.set_xlabel('$s$')

            subplot.set_ylabel('$\sigma_{s} / <N_{s}>$')
            subplot.set_yscale('log')
            subplot.set_ylim(10**(-4), 10**0)
            
            subplot.plot(X_s, Std_jackknife_s / Mean_total_s, color = 'b', label = 'Jackknife')
            subplot.plot(X_s, Std_simu_s / Mean_total_s, color = 'r', label = 'Simulation')
            subplot.plot(X_s, Std_total_s / Mean_total_s, color = 'k', label = 'Total')
            
            subplot.legend()

plt.suptitle("Relative effects of jackknife vs. change of simulation")
plt.show()