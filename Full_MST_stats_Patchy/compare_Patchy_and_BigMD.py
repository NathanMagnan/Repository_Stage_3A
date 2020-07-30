## Imports
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)

print("all imports sucessfull")

## Loading the Patchy histograms
print("starting to load the Patchy histograms")

Histograms = []

for n_simu in range(100):
    # reading the histograms
    path = r'C:\Users\Nathan\Documents\D - X\C - Stages\Stage 3A\Repository_Stage_3A\Full_MST_stats_Patchy'
    file = 'Simulation' + str(n_simu) + '.pkl'
    my_file = os.path.join(path, file)
    
    f = open(my_file, "rb")
    New_histogram = pickle.load(f)
    f.close()
    
    # adding the histograms to the right data sets
    Histograms.append(New_histogram)

print("histograms loaded")

## Finding the mean and std for Patchy
print("starting to analyse")

# choosing a common x axis
X_d = Histograms[0]['x_d']
X_l = Histograms[0]['x_l']
X_b = Histograms[0]['x_b']
X_s = Histograms[0]['x_s']

# finding the mean and std
Mean_d = np.asarray([0 for x1 in X_d])
Std_d = np.asarray([0 for x1 in X_d])
for i in range(100):
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
        
        Mean_old = Mean_d.copy()
        Std_old = Std_d.copy()
        
        Mean_d = (i * Mean_old + New) / (i + 1)
        Std_d = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_d**2)
Std_d = (100 / 99) * Std_d # unbiasing the estimator

Mean_l = np.asarray([0 for x1 in X_l])
Std_l = np.asarray([0 for x1 in X_l])
for i in range(100):
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
            x2 = Histograms[i]['x_l'][l_min]
            if (x2 > x1):
                try:
                    New[k] = (Histograms[i]['y_l'][l_min] + Histograms[i]['y_l'][l_min - 1]) / 2
                except:
                    New[k] = Histograms[i]['y_l'][l_min]
            else:
                try:
                    New[k] = (Histograms[i]['y_l'][l_min] + Histograms[i]['y_l'][l_min + 1]) / 2
                except:
                    New[k] = Histograms[i]['y_l'][l_min]
        New = np.asarray(New)
        
        Mean_old = Mean_l.copy()
        Std_old = Std_l.copy()
        
        Mean_l = (i * Mean_old + New) / (i + 1)
        Std_l = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_l**2)
Std_l = (100 / 99) * Std_l

Mean_b = np.asarray([0 for x1 in X_b])
Std_b = np.asarray([0 for x1 in X_b])
for i in range(100):
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
            x2 = Histograms[i]['x_b'][l_min]
            if (x2 > x1):
                try:
                    New[k] = (Histograms[i]['y_b'][l_min] + Histograms[i]['y_b'][l_min - 1]) / 2
                except:
                    New[k] = Histograms[i]['y_b'][l_min]
            else:
                try:
                    New[k] = (Histograms[i]['y_b'][l_min] + Histograms[i]['y_b'][l_min + 1]) / 2
                except:
                    New[k] = Histograms[i]['y_b'][l_min]
        New = np.asarray(New)
        
        Mean_old = Mean_b.copy()
        Std_old = Std_b.copy()
        
        Mean_b = (i * Mean_old + New) / (i + 1)
        Std_b = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_b**2)
Std_b = (100 / 99) * Std_b

Mean_s = np.asarray([0 for x1 in X_s])
Std_s = np.asarray([0 for x1 in X_s])
for i in range(100):
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
        
        Mean_old = Mean_s.copy()
        Std_old = Std_s.copy()
        
        Mean_s = (i * Mean_old + New) / (i + 1)
        Std_s = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_s**2)
Std_s = (100 / 99) * Std_s

print("Patchy analyzed")

## Loading the BigMD Histograms
print("Starting to load the BigMD histograms")

path = r'C:\Users\Nathan\Documents\D - X\C - Stages\Stage 3A\Repository_Stage_3A\Full_MST_stats_Patchy'
file = 'BigMD' + '.pkl'
my_file = os.path.join(path, file)

f = open(my_file, "rb")
BigMD = pickle.load(f)
f.close()

print("BigMD histograms loaded")

## Adapting the x-axis of BigMD
print("Starting to match the x-axis of BigMD to that of Patchy")

New_d = [0 for x1 in X_d]
for k in range(np.shape(X_d)[0]):
    x1 = X_d[k]
    min = 10
    l_min = 0
    for l in range(np.shape(BigMD['x_d'])[0]):
        x2 = BigMD['x_d'][l]
        if (abs(x1 - x2) < min):
            min = abs(x1 - x2)
            l_min = l
    New_d[k] = BigMD['y_d'][l_min]
BigMD['y_d'] = np.asarray(New_d)

New_l = [0 for x1 in X_l]
for k in range(np.shape(X_l)[0]):
    x1 = X_l[k]
    min = 10
    l_min = 0
    for l in range(np.shape(BigMD['x_l'])[0]):
        x2 = BigMD['x_l'][l]
        if (abs(x1 - x2) < min):
            min = abs(x1 - x2)
            l_min = l
    x2 = BigMD['x_l'][l_min]
    if (x2 > x1):
        try:
            New_l[k] = (BigMD['y_l'][l_min] + BigMD['y_l'][l_min - 1]) / 2
        except:
            New_l[k] = BigMD['y_l'][l_min]
    else:
        try:
            New_l[k] = (BigMD['y_l'][l_min] + BigMD['y_l'][l_min + 1]) / 2
        except:
            New_l[k] = BigMD['y_l'][l_min]
BigMD_Y_l = np.asarray(New_l)

New_b = [0 for x1 in X_b]
for k in range(np.shape(X_b)[0]):
    x1 = X_b[k]
    min = 10
    l_min = 0
    for l in range(np.shape(BigMD['x_b'])[0]):
        x2 = BigMD['x_b'][l]
        if (abs(x1 - x2) < min):
            min = abs(x1 - x2)
            l_min = l
    x2 = BigMD['x_b'][l_min]
    if (x2 > x1):
        try:
            New_b[k] = (BigMD['y_b'][l_min] + BigMD['y_b'][l_min - 1]) / 2
        except:
            New_b[k] = BigMD['y_b'][l_min]
    else:
        try:
            New_b[k] = (BigMD['y_b'][l_min] + BigMD['y_b'][l_min + 1]) / 2
        except:
            New_b[k] = BigMD['y_b'][l_min]
BigMD_Y_b = np.asarray(New_b)

New_s = [0 for x1 in X_s]
for k in range(np.shape(X_s)[0]):
    x1 = X_s[k]
    min = 10
    l_min = 0
    for l in range(np.shape(BigMD['x_s'])[0]):
        x2 = BigMD['x_s'][l]
        if (abs(x1 - x2) < min):
            min = abs(x1 - x2)
            l_min = l
    try:
        New_s[k] = (BigMD['y_s'][l_min - 1] + BigMD['y_s'][l_min] + BigMD['y_s'][l_min + 1]) / 3
    except:
        New_s[k] = BigMD['y_s'][l_min]
BigMD['y_s'] = np.asarray(New_s)

print("x-axis of BigMD adapted")

## Plotting the results
print("starting to plot")

fig, axes = plt.subplots(nrows = 2, ncols = 4, figsize = (20, 10))
plt.subplots_adjust(hspace = 0.3, wspace = 0.4)

for a in range(4):
    for b in range(2):
        subplot = axes[b][a]
        
        if (a == 0):
            subplot.set_xlabel('$d$')
            
            if (b == 0):
                subplot.set_ylabel('$N_{d}$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**0, 10**7)
                
                subplot.fill_between(x = X_d, y1 = Mean_d - Std_d, y2 = Mean_d + Std_d, color = 'b', alpha = 0.4, label = 'Patchy')
                subplot.plot(X_d, BigMD['y_d'], color = 'b', label = 'BigMD')
                
                subplot.legend()
            
            else:
                subplot.set_ylabel('$\Delta N_{d} / <N_{d}>$')
                subplot.set_ylim(-0.5, 0.5)
                
                subplot.fill_between(x = X_d, y1 = - Std_d / Mean_d, y2 = Std_d / Mean_d, color = 'b', alpha = 0.4, label = 'Patchy')
                subplot.plot(X_d, (BigMD['y_d'] - Mean_d) / Mean_d, color = 'b', label = 'BigMD')
                
                subplot.legend()
        
        elif (a == 1):
            subplot.set_xlabel('$l$')
            subplot.set_xscale('log')
            
            if (b == 0):
                subplot.set_ylabel('$N_{l}$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**0, 10**7)
                
                subplot.fill_between(x = X_l, y1 = Mean_l - Std_l, y2 = Mean_l + Std_l, color = 'g', alpha = 0.4, label = 'Patchy')
                subplot.plot(BigMD['x_l'], BigMD['y_l'], color = 'g', label = 'BigMD')
                
                subplot.legend()
            
            else:
                subplot.set_ylabel('$\Delta N_{l} / <N_{l}>$')
                subplot.set_ylim(-0.5, 0.5)
                
                subplot.fill_between(x = X_l, y1 = - Std_l / Mean_l, y2 = Std_l / Mean_l, color = 'g', alpha = 0.4, label = 'Patchy')
                subplot.plot(X_l, (BigMD_Y_l - Mean_l) / Mean_l, color = 'g', label = 'BigMD')
                
                subplot.legend()
        
        elif (a == 2):
            subplot.set_xlabel('$b$')
            subplot.set_xscale('log')
            
            if (b == 0):
                subplot.set_ylabel('$N_{b}$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**0, 10**7)
                
                subplot.fill_between(x = X_b, y1 = Mean_b - Std_b, y2 = Mean_b + Std_b, color = 'r', alpha = 0.4, label = 'Patchy')
                subplot.plot(BigMD['x_b'], BigMD['y_b'], color = 'r', label = 'BigMD')
                
                subplot.legend()
            
            else:
                subplot.set_ylabel('$\Delta N_{b} / <N_{b}>$')
                subplot.set_ylim(-0.5, 0.5)
                
                subplot.fill_between(x = X_b, y1 = - Std_b / Mean_b, y2 = Std_b / Mean_b, color = 'r', alpha = 0.4, label = 'Patchy')
                subplot.plot(X_b, (BigMD_Y_b - Mean_b) / Mean_b, color = 'r', label = 'BigMD')
                
                subplot.legend()
        
        else:
            subplot.set_xlabel('$s$')
            
            if (b == 0):
                subplot.set_ylabel('$N_{s}$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**0, 10**7)
                
                subplot.fill_between(x = X_s, y1 = Mean_s - Std_s, y2 = Mean_s + Std_s, color = 'y', alpha = 0.4, label = 'Patchy')
                subplot.plot(X_s, BigMD['y_s'], color = 'y', label = 'BigMD')
                
                subplot.legend()
            else:
                subplot.set_ylabel('$\Delta N_{s} / <N_{s}>$')
                subplot.set_ylim(-0.5, 0.5)
                
                subplot.fill_between(x = X_s, y1 = - Std_s / Mean_s, y2 = Std_s / Mean_s, color = 'y', alpha = 0.4, label = 'Patchy')
                subplot.plot(X_s, (BigMD['y_s'] - Mean_s) / Mean_s, color = 'y', label = 'BigMD')
                
                subplot.legend()

plt.suptitle("Comparison between fast mocks and a N-body simulation")
plt.show()