## Importing the libraries
import numpy as np
import mistree as mist
import pickle

import sys
import os
sys.path.append('/home/astro/magnan/Repository_Stage_3A')
import catalogue_tools as cat
sys.path.append("/home/astro/magnan")
from AbacusCosmos import InputFile
os.chdir('/home/astro/magnan')

print("All imports successful")

## Importing the data
# getting the basepath
number_str = '00'

path = '/home/astro/magnan/Abacus/AbacusCosmos_720box_products/AbacusCosmos_720box_'
path += number_str
path += '_products/AbacusCosmos_720box_'
path += number_str
path += '_rockstar_halos/z0.100'

# creating a catalogue object
ab = cat.Catalogue_Abacus(basePath = path)

# gettting the data
ab.initialise_data()
CM = ab.CM
print("data acquired")

# getting the MST
ab.compute_MST()
d, l, b, s, l_index, b_index = ab.MST.get_stats(include_index=True)
print("MST acquired")

# creating the data set
print("starting to work on creating the data set")
Catalogues = []
MST_dicts = [0, 0, 0, 0]

for m in range(4):
    histogram = mist.HistMST()
    histogram.setup(usenorm = False, uselog = True)
    histogram.start_group()
    Catalogues.append(histogram)

for i in range(4):
    for j in range(4):
        for k in range(4):
            
            print("starting to work on subsample " + str(i*16 + j*4 + k + 1))
            
            # determining if center, face, side or corner :
            m = 0
            if ((i == 0) or (i == 3)):
                m += 1
            if ((j == 0) or (j == 3)):
                m += 1
            if ((k == 0) or (k == 3)):
                m += 1
            
            # getting rid of the points in the small cube (i,j,k)
            lim_inf_x, lim_sup_x = 720 / 4 * i, 720 / 4 * (i + 1)
            lim_inf_y, lim_sup_y = 720 / 4 * j, 720 / 4 * (j + 1)
            lim_inf_z, lim_sup_z = 720 / 4 * k, 720 / 4 * (k + 1)
            
            # finding the index of the nodes in the small box
            def toRemove(cm):
            	if ((cm[0] > lim_inf_x) and (cm[0] < lim_sup_x)):
            		if ((cm[1] > lim_inf_y) and (cm[1] < lim_sup_y)):
            			if ((cm[2] > lim_inf_z) and (cm[2] < lim_sup_z)):
            				return(True)
            	return(False)
            
            # constructing d_reduced
            d_reduced = d.copy()
            d_reduced = d_reduced.tolist()
            for n in range(np.shape(d)[0] - 1, -1, -1): # to avoid problems with list indexes between d and d_reduced that changes size
            	cm = CM[n]
            	if (toRemove(cm)):
            		del d_reduced[n]
            d_reduced = np.asarray(d_reduced)
            print("d_reduced computed")
            
            # constructing l_reduced
            l_reduced = l.copy()
            l_reduced = l_reduced.tolist()
            for n in range(np.shape(l)[0] - 1, -1, -1): # to avoid problems with list indexes between l and l_reduced that changes size
            	cm1 = CM[l_index[0, n]]
            	cm2 = CM[l_index[1, n]]
            	if (toRemove(cm1) or toRemove(cm2)):
            		del l_reduced[n]
            l_reduced = np.asarray(l_reduced)
            print("l_reduced computed")
            
            # constructing b_reduced and s_reduced
            b_reduced = b.copy()
            b_reduced = b_reduced.tolist()
            s_reduced = s.copy()
            s_reduced = s_reduced.tolist()
            for n in range(np.shape(b)[0] - 1, -1, -1): # to avoid problems with list indexes between b and b_reduced that changes size
            	test = False
            	branch = b_index[n]
            	for index in branch:
            		cm = CM[index]
            		if toRemove(cm):
            			test = True
            	if test:
            		del b_reduced[n]
            		del s_reduced[n]
            b_reduced = np.asarray(b_reduced)
            s_reduced = np.asarray(s_reduced)
            print("b and s_reduced computed")
            
            # saving the catalogue
            _hist = Catalogues[m].get_hist(d_reduced, l_reduced, b_reduced, s_reduced)
            print("results saved to the right catalogue")
            print("box treated.")

for m in range(4):
    MST_dicts[m] = Catalogues[m].end_group()

print("data set fully created")

## Saving the statistics
print("Starting to save the results")

Labels = ['center', 'face', 'edge', 'corner']

for m in range(4):
	my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Test_jacknife/')
	my_file = 'Test_3_' + Labels[m] + '.pkl'
	my_file = os.path.join(my_path, my_file)
	
	f = open(my_file, "wb")
	pickle.dump(MST_dicts[m], f)
	f.close()

print("Results saved")

## getting the histogram
#print("starting to compute the histogram")
#ab.compute_MST_histogram(jacknife = True)
#print("histogram computed")
#
## saving the data
#print("starting to save the data")
#target = "/home/astro/magnan/Repository_Stage_3A/Test_jacknife/Test_compilation"
#
#X_d = ab.MST_histogram['x_d']
#Y_d = ab.MST_histogram['y_d']
#Y_d_std = ab.MST_histogram['y_d_std']
#np.savetxt(str(target) + "_X_d", X_d)
#np.savetxt(str(target) + "_Y_d", Y_d)
#np.savetxt(str(target) + "_Y_d_std", Y_d_std)
#
#X_l = ab.MST_histogram['x_l']
#Y_l = ab.MST_histogram['y_l']
#Y_l_std = ab.MST_histogram['y_l_std']
#np.savetxt(str(target) + "_X_l", X_l)
#np.savetxt(str(target) + "_Y_l", Y_l)
#np.savetxt(str(target) + "_Y_l_std", Y_l_std)
#
#X_b = ab.MST_histogram['x_b']
#Y_b = ab.MST_histogram['y_b']
#Y_b_std = ab.MST_histogram['y_b_std']
#np.savetxt(str(target) + "_X_b", X_b)
#np.savetxt(str(target) + "_Y_b", Y_b)
#np.savetxt(str(target) + "_Y_b_std", Y_b_std)
#
#X_s = ab.MST_histogram['x_s']
#Y_s = ab.MST_histogram['y_s']
#Y_s_std = ab.MST_histogram['y_s_std']
#np.savetxt(str(target) + "_X_s", X_s)
#np.savetxt(str(target) + "_Y_s", Y_s)
#np.savetxt(str(target) + "_Y_s_std", Y_s_std)
#
#print("data saved")

