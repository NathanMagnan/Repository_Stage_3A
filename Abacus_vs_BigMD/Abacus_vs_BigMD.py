## Imports
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex = True)

print("all imports sucessfull")

## Loading the Mass_cut histograms
print("starting to load the mass cut histograms")

Histograms_mass_cut = []

for n_simu in range(22):
    # reading the histograms
    path = r'C:\Users\Nathan\Documents\D - X\C - Stages\Stage 3A\Repository_Stage_3A\Abacus_vs_BigMD'
    file = 'MST_Mass_cut_' + str(n_simu) + '.pkl'
    my_file = os.path.join(path, file)
    
    f = open(my_file, "rb")
    New_histogram = pickle.load(f)
    f.close()
    
    # adding the histograms to the right data sets
    Histograms_mass_cut.append(New_histogram)

print("Mass cut histograms loaded")

## Loading the Random_cut histograms
print("starting to load the random cut histograms")

Histograms_random_cut = []

for n_simu in range(22):
    # reading the histograms
    path = r'C:\Users\Nathan\Documents\D - X\C - Stages\Stage 3A\Repository_Stage_3A\Abacus_vs_BigMD'
    file = 'MST_random_cut' + str(n_simu) + '.pkl'
    my_file = os.path.join(path, file)
    
    f = open(my_file, "rb")
    New_histogram = pickle.load(f)
    f.close()
    
    # adding the histograms to the right data sets
    Histograms_random_cut.append(New_histogram)

print("random cut histograms loaded")

## Loading the Full histograms
print("starting to load the full histograms")

Histograms_full = []

for n_simu in range(22):
    # reading the histograms
    path = r'C:\Users\Nathan\Documents\D - X\C - Stages\Stage 3A\Repository_Stage_3A\Abacus_vs_BigMD'
    file = 'MST_full' + str(n_simu) + '.pkl'
    my_file = os.path.join(path, file)
    
    f = open(my_file, "rb")
    New_histogram = pickle.load(f)
    f.close()
    
    # adding the histograms to the right data sets
    Histograms_full.append(New_histogram)

print("full histograms loaded")

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

# adapting the x-axis for mass cut
Mass_cut_d = []
for i in range(22):
        New = [0 for x1 in X_d]
        for k in range(np.shape(X_d)[0]):
            x1 = X_d[k]
            min = 10
            l_min = 0
            for l in range(np.shape(Histograms_mass_cut[i]['x_d'])[0]):
                x2 = Histograms_mass_cut[i]['x_d'][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            New[k] = Histograms_mass_cut[i]['y_d'][l_min]
        
        Mass_cut_d.append(np.asarray(New))
        

Mass_cut_l = []
for i in range(22):
        New = [0 for x1 in X_l]
        for k in range(np.shape(X_l)[0]):
            x1 = X_l[k]
            min = 10
            l_min = 0
            for l in range(np.shape(Histograms_mass_cut[i]['x_l'])[0]):
                x2 = Histograms_mass_cut[i]['x_l'][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            try:
                if (x1 > Histograms_mass_cut[i]['x_l'][l_min]):
                    x2a = Histograms_mass_cut[i]['x_l'][l_min]
                    x2b = Histograms_mass_cut[i]['x_l'][l_min+1]
                    a = (x2b - x1) / (x2b - x2a)
                    b = (x1 - x2a) / (x2b - x2a)
                    New[k] = a * Histograms_mass_cut[i]['y_l'][l_min] + b * Histograms_mass_cut[i]['y_l'][l_min + 1]
                else:
                    x2a = Histograms_mass_cut[i]['x_l'][l_min - 1]
                    x2b = Histograms_mass_cut[i]['x_l'][l_min]
                    a = (x2b - x1) / (x2b - x2a)
                    b = (x1 - x2a) / (x2b - x2a)
                    New[k] = a * Histograms_mass_cut[i]['y_l'][l_min - 1] + b * Histograms_mass_cut[i]['y_l'][l_min]
            except:
                New[k] = Histograms_mass_cut[i]['y_l'][l_min]
        
        Mass_cut_l.append(np.asarray(New))

Mass_cut_b = []
for i in range(22):
        New = [0 for x1 in X_b]
        for k in range(np.shape(X_b)[0]):
            x1 = X_b[k]
            min = 10
            l_min = 0
            for l in range(np.shape(Histograms_mass_cut[i]['x_b'])[0]):
                x2 = Histograms_mass_cut[i]['x_b'][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            try:
                if (x1 > Histograms_mass_cut[i]['x_b'][l_min]):
                    x2a = Histograms_mass_cut[i]['x_b'][l_min]
                    x2b = Histograms_mass_cut[i]['x_b'][l_min+1]
                    a = (x2b - x1) / (x2b - x2a)
                    b = (x1 - x2a) / (x2b - x2a)
                    New[k] = a * Histograms_mass_cut[i]['y_b'][l_min] + b * Histograms_mass_cut[i]['y_b'][l_min + 1]
                else:
                    x2a = Histograms_mass_cut[i]['x_b'][l_min - 1]
                    x2b = Histograms_mass_cut[i]['x_b'][l_min]
                    a = (x2b - x1) / (x2b - x2a)
                    b = (x1 - x2a) / (x2b - x2a)
                    New[k] = a * Histograms_mass_cut[i]['y_b'][l_min - 1] + b * Histograms_mass_cut[i]['y_b'][l_min]
            except:
                New[k] = Histograms_mass_cut[i]['y_b'][l_min]
        Mass_cut_b.append(np.asarray(New))

Mass_cut_s = []
for i in range(22):
        New = [0 for x1 in X_s]
        for k in range(np.shape(X_s)[0]):
            x1 = X_s[k]
            min = 10
            l_min = 0
            for l in range(np.shape(Histograms_mass_cut[i]['x_s'])[0]):
                x2 = Histograms_mass_cut[i]['x_s'][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            New[k] = Histograms_mass_cut[i]['y_s'][l_min]
        Mass_cut_s.append(np.asarray(New))

# adapting the x-axis for random cut
Random_cut_d = []
for i in range(22):
        New = [0 for x1 in X_d]
        for k in range(np.shape(X_d)[0]):
            x1 = X_d[k]
            min = 10
            l_min = 0
            for l in range(np.shape(Histograms_random_cut[i]['x_d'])[0]):
                x2 = Histograms_random_cut[i]['x_d'][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            New[k] = Histograms_random_cut[i]['y_d'][l_min]
        
        Random_cut_d.append(np.asarray(New))
        

Random_cut_l = []
for i in range(22):
        New = [0 for x1 in X_l]
        for k in range(np.shape(X_l)[0]):
            x1 = X_l[k]
            min = 10
            l_min = 0
            for l in range(np.shape(Histograms_random_cut[i]['x_l'])[0]):
                x2 = Histograms_random_cut[i]['x_l'][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            try:
                if (x1 > Histograms_random_cut[i]['x_l'][l_min]):
                    x2a = Histograms_random_cut[i]['x_l'][l_min]
                    x2b = Histograms_random_cut[i]['x_l'][l_min+1]
                    a = (x2b - x1) / (x2b - x2a)
                    b = (x1 - x2a) / (x2b - x2a)
                    New[k] = a * Histograms_random_cut[i]['y_l'][l_min] + b * Histograms_random_cut[i]['y_l'][l_min + 1]
                else:
                    x2a = Histograms_random_cut[i]['x_l'][l_min - 1]
                    x2b = Histograms_random_cut[i]['x_l'][l_min]
                    a = (x2b - x1) / (x2b - x2a)
                    b = (x1 - x2a) / (x2b - x2a)
                    New[k] = a * Histograms_random_cut[i]['y_l'][l_min - 1] + b * Histograms_random_cut[i]['y_l'][l_min]
            except:
                New[k] = Histograms_random_cut[i]['y_l'][l_min]
        
        Random_cut_l.append(np.asarray(New))

Random_cut_b = []
for i in range(22):
        New = [0 for x1 in X_b]
        for k in range(np.shape(X_b)[0]):
            x1 = X_b[k]
            min = 10
            l_min = 0
            for l in range(np.shape(Histograms_random_cut[i]['x_b'])[0]):
                x2 = Histograms_random_cut[i]['x_b'][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            try:
                if (x1 > Histograms_random_cut[i]['x_b'][l_min]):
                    x2a = Histograms_random_cut[i]['x_b'][l_min]
                    x2b = Histograms_random_cut[i]['x_b'][l_min+1]
                    a = (x2b - x1) / (x2b - x2a)
                    b = (x1 - x2a) / (x2b - x2a)
                    New[k] = a * Histograms_random_cut[i]['y_b'][l_min] + b * Histograms_random_cut[i]['y_b'][l_min + 1]
                else:
                    x2a = Histograms_random_cut[i]['x_b'][l_min - 1]
                    x2b = Histograms_random_cut[i]['x_b'][l_min]
                    a = (x2b - x1) / (x2b - x2a)
                    b = (x1 - x2a) / (x2b - x2a)
                    New[k] = a * Histograms_random_cut[i]['y_b'][l_min - 1] + b * Histograms_random_cut[i]['y_b'][l_min]
            except:
                New[k] = Histograms_random_cut[i]['y_b'][l_min]
        Random_cut_b.append(np.asarray(New))

Random_cut_s = []
for i in range(22):
        New = [0 for x1 in X_s]
        for k in range(np.shape(X_s)[0]):
            x1 = X_s[k]
            min = 10
            l_min = 0
            for l in range(np.shape(Histograms_random_cut[i]['x_s'])[0]):
                x2 = Histograms_random_cut[i]['x_s'][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            New[k] = Histograms_random_cut[i]['y_s'][l_min]
        Random_cut_s.append(np.asarray(New))

# adapting the x-axis for the full histograms
Full_d = []
for i in range(22):
        New = [0 for x1 in X_d]
        for k in range(np.shape(X_d)[0]):
            x1 = X_d[k]
            min = 10
            l_min = 0
            for l in range(np.shape(Histograms_full[i]['x_d'])[0]):
                x2 = Histograms_full[i]['x_d'][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            New[k] = Histograms_full[i]['y_d'][l_min]
        Full_d.append(np.asarray(New))
        

Full_l = []
for i in range(22):
        New = [0 for x1 in X_l]
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
                    New[k] = a * Histograms_full[i]['y_l'][l_min] + b * Histograms_full[i]['y_l'][l_min + 1]
                else:
                    x2a = Histograms_full[i]['x_l'][l_min - 1]
                    x2b = Histograms_full[i]['x_l'][l_min]
                    a = (x2b - x1) / (x2b - x2a)
                    b = (x1 - x2a) / (x2b - x2a)
                    New[k] = a * Histograms_full[i]['y_l'][l_min - 1] + b * Histograms_full[i]['y_l'][l_min]
            except:
                New[k] = Histograms_full[i]['y_l'][l_min]
        
        Full_l.append(np.asarray(New))

Full_b = []
for i in range(22):
        New = [0 for x1 in X_b]
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
                    New[k] = a * Histograms_full[i]['y_b'][l_min] + b * Histograms_full[i]['y_b'][l_min + 1]
                else:
                    x2a = Histograms_full[i]['x_b'][l_min - 1]
                    x2b = Histograms_full[i]['x_b'][l_min]
                    a = (x2b - x1) / (x2b - x2a)
                    b = (x1 - x2a) / (x2b - x2a)
                    New[k] = a * Histograms_full[i]['y_b'][l_min - 1] + b * Histograms_full[i]['y_b'][l_min]
            except:
                New[k] = Histograms_full[i]['y_b'][l_min]
        Full_b.append(np.asarray(New))

Full_s = []
for i in range(22):
        New = [0 for x1 in X_s]
        for k in range(np.shape(X_s)[0]):
            x1 = X_s[k]
            min = 10
            l_min = 0
            for l in range(np.shape(Histograms_full[i]['x_s'])[0]):
                x2 = Histograms_full[i]['x_s'][l]
                if (abs(x1 - x2) < min):
                    min = abs(x1 - x2)
                    l_min = l
            New[k] = Histograms_full[i]['y_s'][l_min]
        Full_s.append(np.asarray(New))

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

## Computing the Mean and Stds
print("starting to compute the means and stds")

# for mass cuts
Mean_mc_d = np.asarray([0 for x_d in X_d])
Std_mc_d = np.asarray([0 for x_d in X_d])
for i in range(21):
    New = Mass_cut_d[i]
    
    Mean_old = Mean_mc_d.copy()
    Std_old = Std_mc_d.copy()
    
    Mean_mc_d = (i * Mean_old + New) / (i + 1)
    Std_mc_d = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_mc_d**2)

Mean_mc_l = np.array([0 for x_l in X_l])
Std_mc_l = np.array([0 for x_l in X_l])
for i in range(21):
    New = Mass_cut_l[i]
    
    Mean_old = Mean_mc_l.copy()
    Std_old = Std_mc_l.copy()
    
    Mean_mc_l = (i * Mean_old + New) / (i + 1)
    Std_mc_l = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_mc_l**2)

Mean_mc_b = np.asarray([0 for x_b in X_b])
Std_mc_b = np.asarray([0 for x_b in X_b])
for i in range(21):
    New = Mass_cut_b[i]
    
    Mean_old = Mean_mc_b.copy()
    Std_old = Std_mc_b.copy()
    
    Mean_mc_b = (i * Mean_old + New) / (i + 1)
    Std_mc_b = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_mc_b**2)

Mean_mc_s = np.asarray([0 for x_s in X_s])
Std_mc_s = np.asarray([0 for x_s in X_s])
for i in range(21):
    New = Mass_cut_s[i]
    
    Mean_old = Mean_mc_s.copy()
    Std_old = Std_mc_s.copy()
    
    Mean_mc_s = (i * Mean_old + New) / (i + 1)
    Std_mc_s = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_mc_s**2)

# for random cuts
Mean_rc_d = np.asarray([0 for x_d in X_d])
Std_rc_d = np.asarray([0 for x_d in X_d])
for i in range(21):
    New = Random_cut_d[i]
    
    Mean_old = Mean_rc_d.copy()
    Std_old = Std_rc_d.copy()
    
    Mean_rc_d = (i * Mean_old + New) / (i + 1)
    Std_rc_d = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_rc_d**2)

Mean_rc_l = np.array([0 for x_l in X_l])
Std_rc_l = np.array([0 for x_l in X_l])
for i in range(21):
    New = Random_cut_l[i]
    
    Mean_old = Mean_rc_l.copy()
    Std_old = Std_rc_l.copy()
    
    Mean_rc_l = (i * Mean_old + New) / (i + 1)
    Std_rc_l = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_rc_l**2)

Mean_rc_b = np.asarray([0 for x_b in X_b])
Std_rc_b = np.asarray([0 for x_b in X_b])
for i in range(21):
    New = Random_cut_b[i]
    
    Mean_old = Mean_rc_b.copy()
    Std_old = Std_rc_b.copy()
    
    Mean_rc_b = (i * Mean_old + New) / (i + 1)
    Std_rc_b = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_rc_b**2)

Mean_rc_s = np.asarray([0 for x_s in X_s])
Std_rc_s = np.asarray([0 for x_s in X_s])
for i in range(21):
    New = Random_cut_s[i]
    
    Mean_old = Mean_rc_s.copy()
    Std_old = Std_rc_s.copy()
    
    Mean_rc_s = (i * Mean_old + New) / (i + 1)
    Std_rc_s = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_rc_s**2)

# for full
Mean_f_d = np.asarray([0 for x_d in X_d])
Std_f_d = np.asarray([0 for x_d in X_d])
for i in range(21):
    New = Full_d[i]
    
    Mean_old = Mean_f_d.copy()
    Std_old = Std_f_d.copy()
    
    Mean_f_d = (i * Mean_old + New) / (i + 1)
    Std_f_d = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_f_d**2)

Mean_f_l = np.array([0 for x_l in X_l])
Std_f_l = np.array([0 for x_l in X_l])
for i in range(21):
    New = Full_l[i]
    
    Mean_old = Mean_f_l.copy()
    Std_old = Std_f_l.copy()
    
    Mean_f_l = (i * Mean_old + New) / (i + 1)
    Std_f_l = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_f_l**2)

Mean_f_b = np.asarray([0 for x_b in X_b])
Std_f_b = np.asarray([0 for x_b in X_b])
for i in range(21):
    New = Full_b[i]
    
    Mean_old = Mean_f_b.copy()
    Std_old = Std_f_b.copy()
    
    Mean_f_b = (i * Mean_old + New) / (i + 1)
    Std_f_b = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_f_b**2)

Mean_f_s = np.asarray([0 for x_s in X_s])
Std_f_s = np.asarray([0 for x_s in X_s])
for i in range(21):
    New = Full_s[i]
    
    Mean_old = Mean_f_s.copy()
    Std_old = Std_f_s.copy()
    
    Mean_f_s = (i * Mean_old + New) / (i + 1)
    Std_f_s = np.sqrt((i * (Std_old**2 + Mean_old**2) + New**2) / (i + 1) - Mean_f_s**2)

print("Means and std computed")

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
                
                subplot.fill_between(x = X_d, y1 = Mass_cut_d[21] - Std_mc_d, y2 = Mass_cut_d[21] + Std_mc_d, color = 'b', alpha = 0.4, label = 'Mass cut')
                subplot.fill_between(x = X_d, y1 = Random_cut_d[21] - Std_rc_d, y2 = Random_cut_d[21] + Std_rc_d, color = 'g', alpha = 0.4, label = 'Random cut')
                subplot.fill_between(x = X_d, y1 = Full_d[21] - Std_f_d, y2 = Full_d[21] + Std_f_d, color = 'r', alpha = 0.4, label = 'Full distribution')
                subplot.plot(X_d, BigMD_d * (720 / 2500)**3, color = 'y', label = 'BigMD')
                
                subplot.legend(loc = 'upper right')
            
            else:
                subplot.set_ylabel('$N_{d} / <N_{d}>$')
                subplot.set_ylim(0, 2)
                
                subplot.fill_between(x = X_d, y1 = (Mass_cut_d[21] - 3 * Std_mc_d) / (BigMD_d * (720 / 2500)**3), y2 = (Mass_cut_d[21] + 3 * Std_mc_d) / (BigMD_d * (720 / 2500)**3), color = 'b', alpha = 0.4, label = 'Mass cut')
                subplot.fill_between(x = X_d, y1 = (Random_cut_d[21] - 3 * Std_rc_d ) / (BigMD_d * (720 / 2500)**3), y2 = (Random_cut_d[21] + 3 * Std_rc_d) / (BigMD_d * (720 / 2500)**3), color = 'g', alpha = 0.4, label = 'Random cut')
                subplot.plot(X_d, BigMD_d / BigMD_d, color = 'y', label = 'BigMD')
                
                subplot.legend(loc = 'upper right')
        
        elif (a == 1):
            subplot.set_xlabel('$l$')
            subplot.set_xscale('log')
            subplot.set_xlim(5 * 10**(-2), 25)
            
            if (b == 0):
                subplot.set_ylabel('$N_{l}$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**0, 10**7)
                
                subplot.fill_between(x = X_l, y1 = Mass_cut_l[21] - Std_mc_l, y2 = Mass_cut_l[21] + Std_mc_l, color = 'b', alpha = 0.4, label = 'Mass cut')
                subplot.fill_between(x = X_l, y1 = Random_cut_l[21] - Std_rc_l, y2 = Random_cut_l[21] + Std_rc_l, color = 'g', alpha = 0.4, label = 'Random cut')
                subplot.fill_between(x = X_l, y1 = Full_l[21] - Std_f_l, y2 = Full_l[21] + Std_f_l, color = 'r', alpha = 0.4, label = 'Full distribution')
                subplot.plot(X_l, BigMD_l * (720 / 2500)**3, color = 'y', label = 'BigMD')
                
                subplot.legend(loc = 'upper right')
            
            else:
                subplot.set_ylabel('$N_{l} / <N_{l}>$')
                subplot.set_ylim(0, 2)
                
                subplot.fill_between(x = X_l, y1 = (Mass_cut_l[21] - 3 * Std_mc_l) / (BigMD_l * (720 / 2500)**3), y2 = (Mass_cut_l[21] + 3 * Std_mc_l) / (BigMD_l * (720 / 2500)**3), color = 'b', alpha = 0.4, label = 'Mass cut')
                subplot.fill_between(x = X_l, y1 = (Random_cut_l[21] - 3 * Std_rc_l ) / (BigMD_l * (720 / 2500)**3), y2 = (Random_cut_l[21] + 3 * Std_rc_l) / (BigMD_l * (720 / 2500)**3), color = 'g', alpha = 0.4, label = 'Random cut')
                subplot.plot(X_l, BigMD_l / BigMD_l, color = 'y', label = 'BigMD')
                
                subplot.legend(loc = 'upper right')
        
        elif (a == 2):
            subplot.set_xlabel('$b$')
            subplot.set_xscale('log')
            subplot.set_xlim(10**(-1), 75)
            
            if (b == 0):
                subplot.set_ylabel('$N_{b}$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**0, 10**7)
                
                subplot.fill_between(x = X_b, y1 = Mass_cut_b[21] - Std_mc_b, y2 = Mass_cut_b[21] + Std_mc_b, color = 'b', alpha = 0.4, label = 'Mass cut')
                subplot.fill_between(x = X_b, y1 = Random_cut_b[21] - Std_rc_b, y2 = Random_cut_b[21] + Std_rc_b, color = 'g', alpha = 0.4, label = 'Random cut')
                subplot.fill_between(x = X_b, y1 = Full_b[21] - Std_f_b, y2 = Full_b[21] + Std_f_b, color = 'r', alpha = 0.4, label = 'Full distribution')
                subplot.plot(X_b, BigMD_b * (720 / 2500)**3, color = 'y', label = 'BigMD')
                
                subplot.legend(loc = 'upper right')
            
            else:
                subplot.set_ylabel('$N_{b} / <N_{b}>$')
                subplot.set_ylim(0, 2)
                
                subplot.fill_between(x = X_b, y1 = (Mass_cut_b[21] - 3 * Std_mc_b) / (BigMD_b * (720 / 2500)**3), y2 = (Mass_cut_b[21] + 3 * Std_mc_b) / (BigMD_b * (720 / 2500)**3), color = 'b', alpha = 0.4, label = 'Mass cut')
                subplot.fill_between(x = X_b, y1 = (Random_cut_b[21] - 3 * Std_rc_b) / (BigMD_b * (720 / 2500)**3), y2 = (Random_cut_b[21] + 3 * Std_rc_b) / (BigMD_b * (720 / 2500)**3), color = 'g', alpha = 0.4, label = 'Random cut')
                subplot.plot(X_b, BigMD_b / BigMD_b, color = 'y', label = 'BigMD')
                
                subplot.legend(loc = 'upper right')
        
        else:
            subplot.set_xlabel('$s$')
            
            if (b == 0):
                subplot.set_ylabel('$N_{s}$')
                subplot.set_yscale('log')
                subplot.set_ylim(10**0, 10**7)
                
                subplot.fill_between(x = X_s, y1 = Mass_cut_s[21] - Std_mc_s, y2 = Mass_cut_s[21] + Std_mc_s, color = 'b', alpha = 0.4, label = 'Mass cut')
                subplot.fill_between(x = X_s, y1 = Random_cut_s[21] - Std_rc_s, y2 = Random_cut_s[21] + Std_rc_s, color = 'g', alpha = 0.4, label = 'Random cut')
                subplot.fill_between(x = X_s, y1 = Full_s[21] - Std_f_s, y2 = Full_s[21] + Std_f_s, color = 'r', alpha = 0.4, label = 'Full distribution')
                subplot.plot(X_s, BigMD_s * (720 / 2500)**3, color = 'y', label = 'BigMD')
                
                subplot.legend(loc = 'upper right')
            
            else:
                subplot.set_ylabel('$N_{s} / <N_{s}>$')
                subplot.set_ylim(0, 2)
                
                subplot.fill_between(x = X_s, y1 = (Mass_cut_s[21] - 3 * Std_mc_s) / (BigMD_s * (720 / 2500)**3), y2 = (Mass_cut_s[21] + 3 * Std_mc_s) / (BigMD_s * (720 / 2500)**3), color = 'b', alpha = 0.4, label = 'Mass cut')
                subplot.fill_between(x = X_s, y1 = (Random_cut_s[21] - 3 * Std_rc_s) / (BigMD_s * (720 / 2500)**3), y2 = (Random_cut_s[21] + 3 * Std_rc_s) / (BigMD_s * (720 / 2500)**3), color = 'g', alpha = 0.4, label = 'Random cut')
                subplot.plot(X_s, BigMD_s / BigMD_s, color = 'y', label = 'BigMD')
                
                subplot.legend(loc = 'upper right')

plt.suptitle("Comparison between Abacus and BigMD")
plt.show()