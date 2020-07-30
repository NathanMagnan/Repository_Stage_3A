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

## Loading the BigMD Histograms
print("Starting to load the BigMD histograms")

path = r'C:\Users\Nathan\Documents\D - X\C - Stages\Stage 3A\Repository_Stage_3A\Full_MST_stats_Patchy'
file = 'BigMD' + '.pkl'
my_file = os.path.join(path, file)

f = open(my_file, "rb")
BigMD = pickle.load(f)
f.close()

print("BigMD histograms loaded")

## Choosing a common x-axis

X_d = BigMD['x_d']
X_l = BigMD['x_l']
X_b = BigMD['x_b']
X_s = BigMD['x_s']

## Adapting the x-axis
print("starting to match the x-axis")

# adapting the x-axis for Patchy
Patchy_d = []
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
        
        Patchy_d.append(np.asarray(New))
        

Patchy_l = []
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
            try:
                if (x1 > Histograms[i]['x_l'][l_min]):
                    x2a = Histograms[i]['x_l'][l_min]
                    x2b = Histograms[i]['x_l'][l_min+1]
                    a = (x2b - x1) / (x2b - x2a)
                    b = (x1 - x2a) / (x2b - x2a)
                    New[k] = a * Histograms[i]['y_l'][l_min] + b * Histograms[i]['y_l'][l_min + 1]
                else:
                    x2a = Histograms[i]['x_l'][l_min - 1]
                    x2b = Histograms[i]['x_l'][l_min]
                    a = (x2b - x1) / (x2b - x2a)
                    b = (x1 - x2a) / (x2b - x2a)
                    New[k] = a * Histograms[i]['y_l'][l_min - 1] + b * Histograms[i]['y_l'][l_min]
            except:
                New[k] = Histograms[i]['y_l'][l_min]
        
        Patchy_l.append(np.asarray(New))

Patchy_b = []
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
            try:
                if (x1 > Histograms[i]['x_b'][l_min]):
                    x2a = Histograms[i]['x_b'][l_min]
                    x2b = Histograms[i]['x_b'][l_min+1]
                    a = (x2b - x1) / (x2b - x2a)
                    b = (x1 - x2a) / (x2b - x2a)
                    New[k] = a * Histograms[i]['y_b'][l_min] + b * Histograms[i]['y_b'][l_min + 1]
                else:
                    x2a = Histograms[i]['x_b'][l_min - 1]
                    x2b = Histograms[i]['x_b'][l_min]
                    a = (x2b - x1) / (x2b - x2a)
                    b = (x1 - x2a) / (x2b - x2a)
                    New[k] = a * Histograms[i]['y_b'][l_min - 1] + b * Histograms[i]['y_b'][l_min]
            except:
                New[k] = Histograms[i]['y_b'][l_min]
        Patchy_b.append(np.asarray(New))

Patchy_s = []
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
        Patchy_s.append(np.asarray(New))

# adapting the x-axis for BigMD

BigMD_d = [0 for x1 in X_d]
for k in range(np.shape(X_d)[0]):
    x1 = X_d[k]
    min = 10
    l_min = 0
    for l in range(np.shape(BigMD['x_d'])[0]):
        x2 = BigMD['x_d'][l]
        if (abs(x1 - x2) < min):
            min = abs(x1 - x2)
            l_min = l
    BigMD_d[k] = BigMD['y_d'][l_min]
BigMD_d = np.asarray(BigMD_d)

BigMD_l = [0 for x1 in X_l]
for k in range(np.shape(X_l)[0]):
    x1 = X_l[k]
    min = 10
    l_min = 0
    for l in range(np.shape(BigMD['x_l'])[0]):
        x2 = BigMD['x_l'][l]
        if (abs(x1 - x2) < min):
            min = abs(x1 - x2)
            l_min = l
    try:
        if (x1 > BigMD['x_l'][l_min]):
            x2a = BigMD['x_l'][l_min]
            x2b = BigMD['x_l'][l_min+1]
            a = (x2b - x1) / (x2b - x2a)
            b = (x1 - x2a) / (x2b - x2a)
            BigMD_l[k] = a * BigMD['y_l'][l_min] + b * BigMD['y_l'][l_min + 1]
        else:
            x2a = BigMD['x_l'][l_min - 1]
            x2b = BigMD['x_l'][l_min]
            a = (x2b - x1) / (x2b - x2a)
            b = (x1 - x2a) / (x2b - x2a)
            BigMD_l[k] = a * BigMD['y_l'][l_min - 1] + b * BigMD['y_l'][l_min]
    except:
        BigMD_l[k] = BigMD['y_l'][l_min]
BigMD_l = np.asarray(BigMD_l)

BigMD_b = [0 for x1 in X_b]
for k in range(np.shape(X_b)[0]):
    x1 = X_b[k]
    min = 10
    l_min = 0
    for l in range(np.shape(BigMD['x_b'])[0]):
        x2 = BigMD['x_b'][l]
        if (abs(x1 - x2) < min):
            min = abs(x1 - x2)
            l_min = l
    try:
        if (x1 > BigMD['x_b'][l_min]):
            x2a = BigMD['x_b'][l_min]
            x2b = BigMD['x_b'][l_min+1]
            a = (x2b - x1) / (x2b - x2a)
            b = (x1 - x2a) / (x2b - x2a)
            BigMD_b[k] = a * BigMD['y_b'][l_min] + b * BigMD['y_b'][l_min + 1]
        else:
            x2a = BigMD['x_b'][l_min - 1]
            x2b = BigMD['x_b'][l_min]
            a = (x2b - x1) / (x2b - x2a)
            b = (x1 - x2a) / (x2b - x2a)
            BigMD_b[k] = a * BigMD['y_b'][l_min - 1] + b * BigMD['y_b'][l_min]
    except:
        BigMD_b[k] = BigMD['y_b'][l_min]
BigMD_b = np.asarray(BigMD_b)

BigMD_s = [0 for x1 in X_s]
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
        if (x1 > BigMD['x_s'][l_min]):
            x2a = BigMD['x_s'][l_min]
            x2b = BigMD['x_s'][l_min+1]
            a = (x2b - x1) / (x2b - x2a)
            b = (x1 - x2a) / (x2b - x2a)
            BigMD_s[k] = a * BigMD['y_s'][l_min] + b * BigMD['y_s'][l_min + 1]
        else:
            x2a = BigMD['x_s'][l_min - 1]
            x2b = BigMD['x_s'][l_min]
            a = (x2b - x1) / (x2b - x2a)
            b = (x1 - x2a) / (x2b - x2a)
            BigMD_s[k] = a * BigMD['y_s'][l_min - 1] + b * BigMD['y_s'][l_min]
    except:
        BigMD_s[k] = BigMD['y_s'][l_min]
BigMD_s = np.asarray(BigMD_s)

print("x-axis adapted")

# ## Dividing by 2 the number of bins
# print("starting to divide by 2 the number of bins for l, b and s")
# 
# # l
# New_X_l = []
# New_BigMD_l = []
# New_Patchy_l = [np.array([0.0 for i in range(25)]) for j in range(100)]
# for i in range(25):
#     New_X_l.append((X_l[4*i] + X_l[4*i + 1] + X_l[4*i + 2] + X_l[4*i + 3]) / 4)
#     New_BigMD_l.append(BigMD_l[4*i] + BigMD_l[4*i + 1] + BigMD_l[4*i + 2] + BigMD_l[4*i + 3])
#     
#     for j in range(100):
#         New_Patchy_l[j][i] = Patchy_l[j][4*i] + Patchy_l[j][4*i + 1] + Patchy_l[j][4*i + 2] + Patchy_l[j][4*i + 2]
# X_l = np.asarray(New_X_l)
# BigMD_l = np.asarray(New_BigMD_l)
# Patchy_l = New_Patchy_l.copy()
# 
# # b
# New_X_b = []
# New_BigMD_b = []
# New_Patchy_b = [np.array([0.0 for i in range(25)]) for j in range(100)]
# for i in range(25):
#     New_X_b.append((X_b[4*i] + X_b[4*i + 1] + X_b[4*i + 2] + X_b[4*i + 3]) / 4)
#     New_BigMD_b.append(BigMD_b[4*i] + BigMD_b[4*i + 1] + BigMD_b[4*i + 2] + BigMD_b[4*i + 3])
#     
#     for j in range(100):
#         New_Patchy_b[j][i] = Patchy_b[j][4*i] + Patchy_b[j][4*i + 1] + Patchy_b[j][4*i + 2] + Patchy_b[j][4*i + 2]
# X_b = np.asarray(New_X_b)
# BigMD_b = np.asarray(New_BigMD_b)
# Patchy_b = New_Patchy_b.copy()
# 
# # s
# New_X_s = []
# New_BigMD_s = []
# New_Patchy_s = [np.array([0.0 for i in range(25)]) for j in range(100)]
# for i in range(25):
#     New_X_s.append((X_s[2*i] + X_s[2*i + 1]) / 2)
#     New_BigMD_s.append(BigMD_s[2*i] + BigMD_s[2*i + 1])
#     
#     for j in range(100):
#         New_Patchy_s[j][i] = Patchy_s[j][2*i] + Patchy_s[j][2*i + 1]
# X_s = np.asarray(New_X_s)
# BigMD_s = np.asarray(New_BigMD_s)
# Patchy_s = New_Patchy_s.copy()
# 
# print("number of bins divided by 2")

## Computing the mean and std for Patchy
print("starting to compute the mean and std for patchy")

Mean_d = np.asarray([0 for x_d in X_d])
Std_d = np.asarray([0 for x_d in X_d])
for i in range(100):
    New = Patchy_d[i]
    
    Mean_old = Mean_d.copy()
    Std_old = Std_d.copy()
    
    Mean_d = (i * Mean_old + New) / (i + 1)
    Std_d = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_d**2)

Mean_l = np.array([0 for x_l in X_l])
Std_l = np.array([0 for x_l in X_l])
for i in range(100):
    New = Patchy_l[i]
    
    Mean_old = Mean_l.copy()
    Std_old = Std_l.copy()
    
    Mean_l = (i * Mean_old + New) / (i + 1)
    Std_l = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_l**2)

Mean_b = np.asarray([0 for x_b in X_b])
Std_b = np.asarray([0 for x_b in X_b])
for i in range(100):
    New = Patchy_b[i]
    
    Mean_old = Mean_b.copy()
    Std_old = Std_b.copy()
    
    Mean_b = (i * Mean_old + New) / (i + 1)
    Std_b = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_b**2)

Mean_s = np.asarray([0 for x_s in X_s])
Std_s = np.asarray([0 for x_s in X_s])
for i in range(100):
    New = Patchy_s[i]
    
    Mean_old = Mean_s.copy()
    Std_old = Std_s.copy()
    
    Mean_s = (i * Mean_old + New) / (i + 1)
    Std_s = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_s**2)

print("Patchy mean and std computed")

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
                subplot.plot(X_d, BigMD_d, color = 'b', label = 'BigMD')
                
                subplot.legend(loc = 'upper right')
            
            else:
                subplot.set_ylabel('$N_{d} / <N_{d}>$')
                subplot.set_ylim(0, 2)
                
                subplot.fill_between(x = X_d, y1 = (Mean_d - Std_d) / Mean_d, y2 = (Mean_d + Std_d) / Mean_d, color = 'b', alpha = 0.8, label = 'Patchy')
                subplot.fill_between(x = X_d, y1 = (Mean_d - 3*Std_d) / Mean_d, y2 = (Mean_d + 3*Std_d) / Mean_d, color = 'b', alpha = 0.4)
                subplot.plot(X_d, BigMD_d / Mean_d, color = 'b', label = 'BigMD')
                
                subplot.legend(loc = 'upper right')
        
        elif (a == 1):
            subplot.set_xlabel('$l$')
            subplot.set_xscale('log')
            
            if (b == 0):
                subplot.set_ylabel('$N_{l}$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**0, 10**7)
                
                subplot.fill_between(x = X_l, y1 = Mean_l - Std_l, y2 = Mean_l + Std_l, color = 'g', alpha = 0.4, label = 'Patchy')
                subplot.plot(X_l, BigMD_l, color = 'g', label = 'BigMD')
                
                subplot.legend(loc = 'upper right')
            
            else:
                subplot.set_ylabel('$N_{l} / <N_{l}>$')
                subplot.set_ylim(0, 2)
                
                subplot.fill_between(x = X_l, y1 = (Mean_l - Std_l) / Mean_l, y2 = (Mean_l + Std_l) / Mean_l, color = 'g', alpha = 0.8, label = 'Patchy')
                subplot.fill_between(x = X_l, y1 = (Mean_l - 3*Std_l) / Mean_l, y2 = (Mean_l + 3*Std_l) / Mean_l, color = 'g', alpha = 0.4)
                subplot.plot(X_l, BigMD_l / Mean_l, color = 'g', label = 'BigMD')
                
                subplot.legend(loc = 'upper right')
        
        elif (a == 2):
            subplot.set_xlabel('$b$')
            subplot.set_xscale('log')
            subplot.set_xlim(10**(-1), 10**(2))
            
            if (b == 0):
                subplot.set_ylabel('$N_{b}$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**0, 10**7)
                
                subplot.fill_between(x = X_b, y1 = Mean_b - Std_b, y2 = Mean_b + Std_b, color = 'r', alpha = 0.4, label = 'Patchy')
                subplot.plot(X_b, BigMD_b, color = 'r', label = 'BigMD')
                
                subplot.legend(loc = 'upper right')
            
            else:
                subplot.set_ylabel('$N_{b} / <N_{b}>$')
                subplot.set_ylim(0, 2)
                
                subplot.fill_between(x = X_b, y1 = (Mean_b - Std_b) / Mean_b, y2 = (Mean_b + Std_b) / Mean_b, color = 'r', alpha = 0.8, label = 'Patchy')
                subplot.fill_between(x = X_b, y1 = (Mean_b - 3*Std_b) / Mean_b, y2 = (Mean_b + 3*Std_b) / Mean_b, color = 'r', alpha = 0.4)
                subplot.plot(X_b, BigMD_b / Mean_b, color = 'r', label = 'BigMD')
                
                subplot.legend(loc = 'upper right')
        
        else:
            subplot.set_xlabel('$s$')
            
            if (b == 0):
                subplot.set_ylabel('$N_{s}$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**0, 10**7)
                
                subplot.fill_between(x = X_s, y1 = Mean_s - Std_s, y2 = Mean_s + Std_s, color = 'y', alpha = 0.4, label = 'Patchy')
                subplot.plot(X_s, BigMD_s, color = 'y', label = 'BigMD')
                
                subplot.legend(loc = 'upper right')
            
            else:
                subplot.set_ylabel('$N_{s} / <N_{s}>$')
                subplot.set_ylim(0, 2)
                
                subplot.fill_between(x = X_s, y1 = (Mean_s - Std_s) / Mean_s, y2 = (Mean_s + Std_s) / Mean_s, color = 'y', alpha = 0.8, label = 'Patchy')
                subplot.fill_between(x = X_s, y1 = (Mean_s - 3*Std_s) / Mean_s, y2 = (Mean_s + 3*Std_s) / Mean_s, color = 'y', alpha = 0.4)
                subplot.plot(X_s, BigMD_s / Mean_s, color = 'y', label = 'BigMD')
                
                subplot.legend(loc = 'upper right')

plt.suptitle("Comparison between fast mocks and a N-body simulation")
plt.show()