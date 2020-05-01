import numpy as np
import treecorr as treecorr
import mistree as mist

import sys
import os
sys.path.append('/home/astro/magnan')
import illustris_python as il
os.chdir('/home/astro/magnan')

def get_data(basePath, simulation_number):
    Fields = ['SubhaloCM', 'SubhaloMass', 'SubhaloMassType']
    Subhalos = il.groupcat.loadSubhalos(basePath, simulation_number, fields = Fields)
    box_size = 75.0 # To Be Automatised
    h = 0.704 # To Be Automatised
    
    Subhalos_unbiased = {'count' : 0, 'haloCM' : []}
    for i in range(Subhalos['count']):
        if ((Subhalos['SubhaloMassType'][i][4] >= (0.01 / h)) and (Subhalos['SubhaloMassType'][i][1] * 0.63 > Subhalos['SubhaloMassType'][i][4])):
            Subhalos_unbiased['count'] += 1
            Subhalos_unbiased['haloCM'].append(Subhalos['SubhaloCM'][i])
    
    return(Subhalos_unbiased['count'], box_size, Subhalos_unbiased['haloCM'], h)

def get_2PCF(input_data, bin_min, bin_max, n_bin, box_size):
    Bins = np.logspace(np.log10(bin_min), np.log10(bin_max), n_bin)
    
    X = input_data[:-1, 0]
    Y = input_data[:-1, 1]
    Z = input_data[:-1, 2]
    
    data = treecorr.Catalog(x = X, y = Y, z = Z)
    dd = treecorr.NNCorrelation(min_sep = bin_min, max_sep = bin_max, nbins = n_bin)
    dd.process(data)
    
    List_xi = []
    for i in range(50):
        pos_uniform = np.random.rand(len(X), 3)
        X_uniform = pos_uniform[:-1, 0] * box_size
        Y_uniform = pos_uniform[:-1, 1] * box_size
        Z_uniform = pos_uniform[:-1, 2] * box_size
        uniform_distribution = treecorr.Catalog(x = X_uniform, y = Y_uniform, z = Z_uniform)
        uu = treecorr.NNCorrelation(min_sep = bin_min, max_sep = bin_max, nbins = n_bin)
        uu.process(uniform_distribution)
    
        xi, varxi = dd.calculateXi(uu) #The 2PCF compare the data distribution to an uniform distribution
        List_xi.append(xi)
    
    Mean_xi = [0 for r in Bins]
    Std_xi = [0 for r in Bins]
    for i in range(50):
        for k in range(len(Bins)):
            mean_old = Mean_xi[k]
            std_old = Std_xi[k]
            x_new = List_xi[i][k]
            mean_new = (i * mean_old + x_new) / (i + 1)
            std_new = np.sqrt((i * (std_old**2 + mean_old**2) + x_new**2) / (i + 1) - mean_new**2)
            Mean_xi[k] = mean_new
            Std_xi[k] = std_new
    Mean_xi = np.asarray(Mean_xi)
    Std_xi = np.asarray(Std_xi)
    
    return(Bins, Mean_xi, Std_xi)

def get_MST_histrogram(MST):
    d, l, b, s, l_index, b_index = MST.get_stats(include_index = True)
    histogram = mist.HistMST()
    histogram.setup(usenorm = False, uselog = True)
    
    return(histogram.get_hist(d, l, b, s))