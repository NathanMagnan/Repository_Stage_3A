## Imports
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)

print("all imports sucessfull")

## Loading the jackniffed histograms
print("starting to load the jackknifed histograms")

Histograms_by_simulations = [[] for i in range(21)]

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

print("jackkniffed histograms loaded")

## Loading the full histograms
print("print starting to load the full histograms")

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

print("full histograms loaded")

## Choosing a common x axis

X_d = Histograms_full[0]['x_d']
X_l = Histograms_full[0]['x_l']
X_b = Histograms_full[0]['x_b']
X_s = Histograms_full[0]['x_s']

## Moving every histogram to this x axis
print("Starting to reposition every histogram")

# repositionning in Histograms_by_simulation
for i in range(21):
    for j in range(64):
        
        # d
        New_d = [0 for x1 in X_d]
        for k in range(np.shape(X_d)[0]):
            x1 = X_d[k]
            min = 10
            l_min = 0
            for l in range(np.shape(Histograms_by_simulations[i][j]['x_d'])[0]):
                x2 = Histograms_by_simulations[i][j]['x_d'][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            New_d[k] = Histograms_by_simulations[i][j]['y_d'][l_min]
        Histograms_by_simulations[i][j]['y_d'] = np.asarray(New_d)
        
        # l
        New_l = [0 for x1 in X_l]
        for k in range(np.shape(X_l)[0]):
            x1 = X_l[k]
            min = 10
            l_min = 0
            for l in range(np.shape(Histograms_by_simulations[i][j]['x_l'])[0]):
                x2 = Histograms_by_simulations[i][j]['x_l'][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            try:
                if (x1 > Histograms_by_simulations[i][j]['x_l'][l_min]):
                    x2a = Histograms_by_simulations[i][j]['x_l'][l_min]
                    x2b = Histograms_by_simulations[i][j]['x_l'][l_min+1]
                    a = (x2b - x1) / (x2b - x2a)
                    b = (x1 - x2a) / (x2b - x2a)
                    New_l[k] = a * Histograms_by_simulations[i][j]['y_l'][l_min] + b * Histograms_by_simulations[i][j]['y_l'][l_min + 1]
                else:
                    x2a = Histograms_by_simulations[i][j]['x_l'][l_min - 1]
                    x2b = Histograms_by_simulations[i][j]['x_l'][l_min]
                    a = (x2b - x1) / (x2b - x2a)
                    b = (x1 - x2a) / (x2b - x2a)
                    New_l[k] = a * Histograms_by_simulations[i][j]['y_l'][l_min - 1] + b * Histograms_by_simulations[i][j]['y_l'][l_min]
            except:
                New_l[k] = Histograms_by_simulations[i][j]['y_l'][l_min]
        Histograms_by_simulations[i][j]['y_l'] = np.asarray(New_l)
        
        # b
        New_b = [0 for x1 in X_b]
        for k in range(np.shape(X_b)[0]):
            x1 = X_b[k]
            min = 10
            l_min = 0
            for l in range(np.shape(Histograms_by_simulations[i][j]['x_b'])[0]):
                x2 = Histograms_by_simulations[i][j]['x_b'][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            try:
                if (x1 > Histograms_by_simulations[i][j]['x_b'][l_min]):
                    x2a = Histograms_by_simulations[i][j]['x_b'][l_min]
                    x2b = Histograms_by_simulations[i][j]['x_b'][l_min+1]
                    a = (x2b - x1) / (x2b - x2a)
                    b = (x1 - x2a) / (x2b - x2a)
                    New_b[k] = a * Histograms_by_simulations[i][j]['y_b'][l_min] + b * Histograms_by_simulations[i][j]['y_b'][l_min + 1]
                else:
                    x2a = Histograms_by_simulations[i][j]['x_b'][l_min - 1]
                    x2b = Histograms_by_simulations[i][j]['x_b'][l_min]
                    a = (x2b - x1) / (x2b - x2a)
                    b = (x1 - x2a) / (x2b - x2a)
                    New_b[k] = a * Histograms_by_simulations[i][j]['y_b'][l_min - 1] + b * Histograms_by_simulations[i][j]['y_b'][l_min]
            except:
                New_b[k] = Histograms_by_simulations[i][j]['y_b'][l_min]
        Histograms_by_simulations[i][j]['y_b'] = np.asarray(New_b)
        
        # s
        New_s = [0 for x1 in X_s]
        for k in range(np.shape(X_s)[0]):
            x1 = X_s[k]
            min = 10
            l_min = 0
            for l in range(np.shape(Histograms_by_simulations[i][j]['x_s'])[0]):
                x2 = Histograms_by_simulations[i][j]['x_s'][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            try:
                if (x1 > Histograms_by_simulations[i][j]['x_s'][l_min]):
                    x2a = Histograms_by_simulations[i][j]['x_s'][l_min]
                    x2b = Histograms_by_simulations[i][j]['x_s'][l_min+1]
                    a = (x2b - x1) / (x2b - x2a)
                    b = (x1 - x2a) / (x2b - x2a)
                    New_s[k] = a * Histograms_by_simulations[i][j]['y_s'][l_min] + b * Histograms_by_simulations[i][j]['y_s'][l_min + 1]
                else:
                    x2a = Histograms_by_simulations[i][j]['x_s'][l_min - 1]
                    x2b = Histograms_by_simulations[i][j]['x_s'][l_min]
                    a = (x2b - x1) / (x2b - x2a)
                    b = (x1 - x2a) / (x2b - x2a)
                    New_s[k] = a * Histograms_by_simulations[i][j]['y_s'][l_min - 1] + b * Histograms_by_simulations[i][j]['y_s'][l_min]
            except:
                New_s[k] = Histograms_by_simulations[i][j]['y_s'][l_min]
        Histograms_by_simulations[i][j]['y_s'] = np.asarray(New_s)

# creating the jackknived histograms
""" it more efficient to create these here with only a copy"""
Histograms_jackknived = []

for i in range(21):
    for j in range(64):
        Histograms_jackknived.append(Histograms_by_simulations[i][j].copy())

# Repositionning the full histograms
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

## Finding the Std and the mean from the old estimators
print("starting to compute the old estimators")

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

print("Old estimators computed")

## Finding the Stds for formula 1
print("starting to work on formula 1")

# finding the mean and std of each simulation
Mean_d = [[] for i in range(21)]
Std_d = [[] for i in range(21)]
for i in range(21):
    
    Mean_d[i] = np.asarray([0 for x1 in X_d])
    Std_d[i] = np.asarray([0 for x1 in X_d])
    
    for j in range(64):
        New = Histograms_by_simulations[i][j]['y_d']
        Mean_old = Mean_d[i].copy()
        Std_old = Std_d[i].copy()
        
        Mean_d[i] = (j * Mean_old + New) / (j + 1)
        Std_d[i] = (j * (Std_old + Mean_old**2) + New**2) / (j + 1) - Mean_d[i]**2
    Std_d[i] = 63 * Std_d[i]

Mean_l = [[] for i in range(21)]
Std_l = [[] for i in range(21)]
for i in range(21):
    
    Mean_l[i] = np.asarray([0 for x1 in X_l])
    Std_l[i] = np.asarray([0 for x1 in X_l])
    
    for j in range(64):
        New = Histograms_by_simulations[i][j]['y_l']
        Mean_old = Mean_l[i].copy()
        Std_old = Std_l[i].copy()
        
        Mean_l[i] = (j * Mean_old + New) / (j + 1)
        Std_l[i] = (j * (Std_old + Mean_old**2) + New**2) / (j + 1) - Mean_l[i]**2
    Std_l[i] = 63 * Std_l[i]

Mean_b = [[] for i in range(21)]
Std_b = [[] for i in range(21)]
for i in range(21):
    
    Mean_b[i] = np.asarray([0 for x1 in X_b])
    Std_b[i] = np.asarray([0 for x1 in X_b])
    
    for j in range(64):
        New = Histograms_by_simulations[i][j]['y_b']
        Mean_old = Mean_b[i].copy()
        Std_old = Std_b[i].copy()
        
        Mean_b[i] = (j * Mean_old + New) / (j + 1)
        Std_b[i] = (j * (Std_old + Mean_old**2) + New**2) / (j + 1) - Mean_b[i]**2
    Std_b[i] = 63 * Std_b[i]

Mean_s = [[] for i in range(21)]
Std_s = [[] for i in range(21)]
for i in range(21):
    
    Mean_s[i] = np.asarray([0 for x1 in X_s])
    Std_s[i] = np.asarray([0 for x1 in X_s])
    
    for j in range(64):
        New = Histograms_by_simulations[i][j]['y_s']
        Mean_old = Mean_s[i].copy()
        Std_old = Std_s[i].copy()
        
        Mean_s[i] = (j * Mean_old + New) / (j + 1)
        Std_s[i] = (j * (Std_old + Mean_old**2) + New**2) / (j + 1) - Mean_s[i]**2
    Std_s[i] = 63 * Std_s[i]

print("first part of analysis done")
print("starting second part")

# computing the mean of the stds and the std of the means
Mean_simu_d = np.asarray([0 for x1 in X_d])
Std_simu_d = np.asarray([0 for x1 in X_d])
Std_jackknife_d = np.asarray([0 for x1 in X_d])
for i in range(21):
    New_mean = Mean_d[i]
    New_std = Std_d[i]
    
    Mean_simu_old = Mean_simu_d.copy()
    Std_simu_old = Std_simu_d.copy()
    Mean_simu_d = (i * Mean_simu_old + New_mean) / (i + 1)
    Std_simu_d = (i * (Std_simu_old + Mean_simu_old**2) + New_mean**2) / (i + 1) - Mean_simu_d**2
    
    Std_jackknife_d = (i * Std_jackknife_d + New_std) / (i + 1)
Std_simu_d = (21 / 20) * Std_simu_d

Mean_simu_l = np.asarray([0 for x1 in X_l])
Std_simu_l = np.asarray([0 for x1 in X_l])
Std_jackknife_l = np.asarray([0 for x1 in X_l])
for i in range(21):
    New_mean = Mean_l[i]
    New_std = Std_l[i]
    
    Mean_simu_old = Mean_simu_l.copy()
    Std_simu_old = Std_simu_l.copy()
    Mean_simu_l = (i * Mean_simu_old + New_mean) / (i + 1)
    Std_simu_l = (i * (Std_simu_old + Mean_simu_old**2) + New_mean**2) / (i + 1) - Mean_simu_l**2
    
    Std_jackknife_l = (i * Std_jackknife_l + New_std) / (i + 1)
Std_simu_l = (21 / 20) * Std_simu_l

Mean_simu_b = np.asarray([0 for x1 in X_b])
Std_simu_b = np.asarray([0 for x1 in X_b])
Std_jackknife_b = np.asarray([0 for x1 in X_b])
for i in range(21):
    New_mean = Mean_b[i]
    New_std = Std_b[i]
    
    Mean_simu_old = Mean_simu_b.copy()
    Std_simu_old = Std_simu_b.copy()
    Mean_simu_b = (i * Mean_simu_old + New_mean) / (i + 1)
    Std_simu_b = (i * (Std_simu_old + Mean_simu_old**2) + New_mean**2) / (i + 1) - Mean_simu_b**2
    
    Std_jackknife_b = (i * Std_jackknife_b + New_std) / (i + 1)
Std_simu_b = (21 / 20) * Std_simu_b

Mean_simu_s = np.asarray([0 for x1 in X_s])
Std_simu_s = np.asarray([0 for x1 in X_s])
Std_jackknife_s = np.asarray([0 for x1 in X_s])
for i in range(21):
    New_mean = Mean_s[i]
    New_std = Std_s[i]
    
    Mean_simu_old = Mean_simu_s.copy()
    Std_simu_old = Std_simu_s.copy()
    Mean_simu_s = (i * Mean_simu_old + New_mean) / (i + 1)
    Std_simu_s = (i * (Std_simu_old + Mean_simu_old**2) + New_mean**2) / (i + 1) - Mean_simu_s**2
    
    Std_jackknife_s = (i * Std_jackknife_s + New_std) / (i + 1)
Std_simu_s = (21 / 20) * Std_simu_s

Std_1_d = Std_simu_d + Std_jackknife_d
Std_1_l = Std_simu_l + Std_jackknife_l
Std_1_b = Std_simu_b + Std_jackknife_b
Std_1_s = Std_simu_s + Std_jackknife_s

print("Formula 1 ready")

## Finding the Std for Formula 2
print("starting to work on formula 2")

Mean_2_d = np.asarray([0 for x1 in X_d])
Std_2_d = np.asarray([0 for x1 in X_d])
for i in range(np.shape(Histograms_jackknived)[0]):
    New = Histograms_jackknived[i]['y_d']
    
    Mean_old = Mean_2_d.copy()
    Std_old = Std_2_d.copy()
    
    Mean_2_d = (i * Mean_old + New) / (i + 1)
    Std_2_d = (i * (Std_old + Mean_old**2) + New**2) / (i + 1) - Mean_2_d**2
Std_2_d = 63 * Std_2_d

Mean_2_l = np.asarray([0 for x1 in X_l])
Std_2_l = np.asarray([0 for x1 in X_l])
for i in range(np.shape(Histograms_jackknived)[0]):
    New = Histograms_jackknived[i]['y_l']
    
    Mean_old = Mean_2_l.copy()
    Std_old = Std_2_l.copy()
    
    Mean_2_l = (i * Mean_old + New) / (i + 1)
    Std_2_l = (i * (Std_old + Mean_old**2) + New**2) / (i + 1) - Mean_2_l**2
Std_2_l = 63 * Std_2_l

Mean_2_b = np.asarray([0 for x1 in X_b])
Std_2_b = np.asarray([0 for x1 in X_b])
for i in range(np.shape(Histograms_jackknived)[0]):
    New = Histograms_jackknived[i]['y_b']
    
    Mean_old = Mean_2_b.copy()
    Std_old = Std_2_b.copy()
    
    Mean_2_b = (i * Mean_old + New) / (i + 1)
    Std_2_b = (i * (Std_old + Mean_old**2) + New**2) / (i + 1) - Mean_2_b**2
Std_2_b = 63 * Std_2_b

Mean_2_s = np.asarray([0 for x1 in X_s])
Std_2_s = np.asarray([0 for x1 in X_s])
for i in range(np.shape(Histograms_jackknived)[0]):
    New = Histograms_jackknived[i]['y_s']
    
    Mean_old = Mean_2_s.copy()
    Std_old = Std_2_s.copy()
    
    Mean_2_s = (i * Mean_old + New) / (i + 1)
    Std_2_s = (i * (Std_old + Mean_old**2) + New**2) / (i + 1) - Mean_2_s**2
Std_2_s = 63 * Std_2_s

print("Formula 2 ready")

## Finding the Std for formula 3
print("starting to work on formula 3")

Mean_3_d = np.asarray([0 for x1 in X_d])
Std_3_d = np.asarray([0 for x1 in X_d])
for i in range(21):
    for j in range(64):
        n = j + i*64
        
        New = Histograms_by_simulations[i][j]['y_d']
        for k in range(21):
            if (k != i):
                New += Histograms_full[k]['y_d'].astype(int)
        
        Mean_old = Mean_3_d.copy()
        Std_old = Std_3_d.copy()
        
        Mean_3_d = (n * Mean_old + New) / (n + 1)
        Std_3_d = (n * (Std_old + Mean_old**2) + New**2) / (n + 1) - Mean_3_d**2
Std_3_d = (64 * 21 - 1) * Std_3_d

Mean_3_l = np.asarray([0 for x1 in X_l])
Std_3_l = np.asarray([0 for x1 in X_l])
for i in range(21):
    for j in range(64):
        n = j + i*64
        
        New = Histograms_by_simulations[i][j]['y_l']
        for k in range(21):
            if (k != i):
                New += Histograms_full[k]['y_l'].astype(int)
        
        Mean_old = Mean_3_l.copy()
        Std_old = Std_3_l.copy()
        
        Mean_3_l = (n * Mean_old + New) / (n + 1)
        Std_3_l = (n * (Std_old + Mean_old**2) + New**2) / (n + 1) - Mean_3_l**2
Std_3_l = (64 * 21 - 1) * Std_3_l

Mean_3_b = np.asarray([0 for x1 in X_b])
Std_3_b = np.asarray([0 for x1 in X_b])
for i in range(21):
    for j in range(64):
        n = j + i*64
        
        New = Histograms_by_simulations[i][j]['y_b']
        for k in range(21):
            if (k != i):
                New += Histograms_full[k]['y_b'].astype(int)
        
        Mean_old = Mean_3_b.copy()
        Std_old = Std_3_b.copy()
        
        Mean_3_b = (n * Mean_old + New) / (n + 1)
        Std_3_b = (n * (Std_old + Mean_old**2) + New**2) / (n + 1) - Mean_3_b**2
Std_3_b = (64 * 21 - 1) * Std_3_b

Mean_3_s = np.asarray([0 for x1 in X_s])
Std_3_s = np.asarray([0 for x1 in X_s])
for i in range(21):
    for j in range(64):
        n = j + i*64
        
        New = Histograms_by_simulations[i][j]['y_s']
        for k in range(21):
            if (k != i):
                New += Histograms_full[k]['y_s'].astype(int)
        
        Mean_old = Mean_3_s.copy()
        Std_old = Std_3_s.copy()
        
        Mean_3_s = (n * Mean_old + New) / (n + 1)
        Std_3_s = (n * (Std_old + Mean_old**2) + New**2) / (n + 1) - Mean_3_s**2
Std_3_s = (64 * 21 - 1) * Std_3_s

print("formula 3 ready")

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
            subplot.set_ylim(10**(-3))
            
            subplot.plot(X_d, np.sqrt(Std_basic_d) / Mean_basic_d, color = 'k', label = "Old estimator")
            subplot.plot(X_d, np.sqrt(Std_jackknife_d) / Mean_basic_d, 'k--', label = 'Jackknife')
            subplot.plot(X_d, np.sqrt(Std_1_d) / Mean_basic_d, color = 'g', label = 'Formula 1')
            subplot.plot(X_d, np.sqrt(Std_2_d) / Mean_basic_d, color = 'b', label = 'Formula 2')
            subplot.plot(X_d, np.sqrt(Std_3_d) / Mean_basic_d, color = 'r', label = "Formula 3")
            subplot.plot(X_d, np.sqrt(Std_3_d / (21**2)) / Mean_basic_d, color = 'y', label = "Formula 4")
            subplot.plot(X_d, np.sqrt(Std_2_d / 63) / Mean_basic_d, color = 'orange', label = 'Formula 5')
            
            subplot.legend()
        
        elif (a == 1):
            subplot.set_xlabel('$l$')
            subplot.set_xscale('log')

            subplot.set_ylabel('$\sigma_{l} / <N_{l}>$')
            subplot.set_yscale('log')
            subplot.set_ylim(10**(-3))
            
            subplot.plot(X_l, np.sqrt(Std_basic_l) / Mean_basic_l, color = 'k', label = "Old estimator")
            subplot.plot(X_l, np.sqrt(Std_jackknife_l) / Mean_basic_l, 'k--', label = 'Jackknife')
            subplot.plot(X_l, np.sqrt(Std_1_l) / Mean_basic_l, color = 'g', label = 'Formula 1')
            subplot.plot(X_l, np.sqrt(Std_2_l) / Mean_basic_l, color = 'b', label = 'Formula 2')
            subplot.plot(X_l, np.sqrt(Std_3_l) / Mean_basic_l, color = 'r', label = "Formula 3")
            subplot.plot(X_l, np.sqrt(Std_3_l / (21**2)) / Mean_basic_l, color = 'y', label = "Formula 4")
            subplot.plot(X_l, np.sqrt(Std_2_l / 63) / Mean_basic_l, color = 'orange', label = 'Formula 5')
            
            subplot.legend(loc = 'upper left')
        
        elif (a == 2):
            subplot.set_xlabel('$b$')
            subplot.set_xscale('log')

            subplot.set_ylabel('$\sigma_{b} / <N_{b}>$')
            subplot.set_yscale('log')
            subplot.set_ylim(10**(-3))
            
            subplot.plot(X_b, np.sqrt(Std_basic_b) / Mean_basic_b, color = 'k', label = "Old estimator")
            subplot.plot(X_b, np.sqrt(Std_jackknife_b) / Mean_basic_b, 'k--', label = 'Jackknife')
            subplot.plot(X_b, np.sqrt(Std_1_b) / Mean_basic_b, color = 'g', label = 'Formula 1')
            subplot.plot(X_b, np.sqrt(Std_2_b) / Mean_basic_b, color = 'b', label = 'Formula 2')
            subplot.plot(X_b, np.sqrt(Std_3_b) / Mean_basic_b, color = 'r', label = "Formula 3")
            subplot.plot(X_b, np.sqrt(Std_3_b / (21**2)) / Mean_basic_b, color = 'y', label = "Formula 4")
            subplot.plot(X_b, np.sqrt(Std_2_b / 63) / Mean_basic_b, color = 'orange', label = 'Formula 5')
            
            subplot.legend(loc = 'upper left')
        
        else:
            subplot.set_xlabel('$s$')

            subplot.set_ylabel('$\sigma_{s} / <N_{s}>$')
            subplot.set_yscale('log')
            subplot.set_ylim(10**(-3))
            
            subplot.plot(X_s, np.sqrt(Std_basic_s) / Mean_basic_s, color = 'k', label = "Old estimator")
            subplot.plot(X_s, np.sqrt(Std_jackknife_s) / Mean_basic_s, 'k--', label = 'Jackknife')
            subplot.plot(X_s, np.sqrt(Std_1_s) / Mean_basic_s, color = 'g', label = 'Formula 1')
            subplot.plot(X_s, np.sqrt(Std_2_s) / Mean_basic_s, color = 'b', label = 'Formula 2')
            subplot.plot(X_s, np.sqrt(Std_3_s) / Mean_basic_s, color = 'r', label = "Formula 3")
            subplot.plot(X_s, np.sqrt(Std_3_s / (21**2)) / Mean_basic_s, color = 'y', label = "Formula 4")
            subplot.plot(X_s, np.sqrt(Std_2_s / 63) / Mean_basic_s, color = 'orange', label = 'Formula 5')
            
            subplot.legend()

plt.suptitle("Comparison of different formulas for estimating the noise")
plt.show()