import numpy as np
import treecorr as treecorr
import mistree as mist

import sys
import os
sys.path.append("/home/astro/magnan")
from AbacusCosmos import Halos
from AbacusCosmos import InputFile
os.chdir('/home/astro/magnan')

def get_data(dataPath, inputPath):
    cat = Halos.make_catalog_from_dir(dirname = dataPath, load_subsamples = False, load_pids = False)
    param = InputFile.InputFile(fn = inputPath)
    halos = cat.halos
    box_size = param.get('BoxSize')
    h = param.get('H0') / 100
    
    particle_mass = param.get('ParticleMassMsun')
    lower_limit_mass = particle_mass * 100
    halos_unbiased = {'count' : 0, 'haloCM' : [], 'haloM' : []}
    for i in range(np.shape(halos)[0]):
        if ((halos['m'][i] >= lower_limit_mass) and (halos['parent_id'][i] == -1)):
            halos_unbiased['count'] += 1
            halos_unbiased['haloCM'].append(halos['pos'][i])
            halos_unbiased["haloM"].append(halos['m'][i])
    
    return(halos_unbiased['count'], box_size, np.asarray(halos_unbiased['haloCM']), h, np.asarray(halos_unbiased['haloM']))

def get_2PCF(input_data, bin_min, bin_max, n_bin_2PCF, box_size):
    Bins = np.logspace(np.log10(bin_min), np.log10(bin_max), n_bin_2PCF)
    
    X = input_data[:, 0]
    Y = input_data[:, 1]
    Z = input_data[:, 2]
    
    data = treecorr.Catalog(x = X, y = Y, z = Z)
    dd = treecorr.NNCorrelation(min_sep = bin_min, max_sep = bin_max, nbins = n_bin_2PCF)
    dd.process(data)
    
    List_xi = []
    for i in range(50):
        pos_uniform = np.random.rand(len(X), 3)
        X_uniform = pos_uniform[:-1, 0] * box_size
        Y_uniform = pos_uniform[:-1, 1] * box_size
        Z_uniform = pos_uniform[:-1, 2] * box_size
        uniform_distribution = treecorr.Catalog(x = X_uniform, y = Y_uniform, z = Z_uniform)
        uu = treecorr.NNCorrelation(min_sep = bin_min, max_sep = bin_max, nbins = n_bin_2PCF)
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

def get_MST_histogram(MST, jacknife = False, CM = None):
	 histogram = mist.HistMST()
	 histogram.setup(usenorm = False, uselog = True)
	 d, l, b, s, l_index, b_index = MST.get_stats(include_index=True)
	 
	 if (jacknife == False):
	 	return(histogram.get_hist(d, l, b, s))
	 	
	 else:
	 	histogram.start_group()
	 	
	 	for i in range(4):
	 		for j in range(4):
	 			for k in range(4):
	 				# finding the limits of the small box to exclude
	 				lim_inf_x, lim_sup_x = 720 / 4 * i, 720 / 4 * (i + 1)
	 				lim_inf_y, lim_sup_y = 720 / 4 * j, 720 / 4 * (j + 1)
	 				lim_inf_z, lim_sup_z = 720 / 4 * k, 720 / 4 * (k + 1)
	 				
	 				# finding the index of the nodes in the small box
	 				Indexes_to_remove = []
	 				n_nodes = np.shape(CM)[0]
	 				for m in range(n_nodes):
	 					if ((CM[m, 0] > lim_inf_x) and (CM[m, 0] < lim_sup_x)):
	 						if ((CM[m, 1] > lim_inf_y) and (CM[m, 1] < lim_sup_y)):
	 							if ((CM[m, 2] > lim_inf_z) and (CM[m, 2] < lim_sup_z)):
	 								Indexes_to_remove.append(m)
	 				
	 				# constructing d_reduced
	 				d_reduced = d.copy()
	 				d_reduced = d_reduced.tolist()
	 				for m in Indexes_to_remove:
	 					del d_reduced[m]
	 				d_reduced = np.asarray(d_reduced)
	 				
	 				# constructing l_reduced
	 				l_reduced = l.copy()
	 				l_reduced = l_reduced.tolist()
	 				for m in range(np.shape(l)[0]):
	 					if ((l_index[0, m] in Indexes_to_remove) or (l_index[1, m] in Indexes_to_remove)):
	 						del l_reduced[m]
	 				l_reduced = np.asarray(l_reduced)
	 				
	 				# constructing b_reduced and s_reduced
	 				b_reduced = b.copy()
	 				b_reduced = b_reduced.tolist()
	 				s_reduced = s.copy()
	 				s_reduced = s_reduced.tolist()
	 				for m in range(len(b_index)):
	 					branch = b_index[m]
	 					for index in branch:
	 						if (index in Indexes_to_remove):
	 							del b_reduced[m]
	 							del s_reduced[m]
	 							break
	 				b_reduced = np.asarray(b_reduced)
	 				s_reduced = np.asarray(s_reduced)
	 				
	 				_hist = histogram.get_hist(d_reduced, l_reduced, b_reduced, s_reduced)
	 		
	 	return(histogram.end_group())