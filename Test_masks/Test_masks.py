## Imports
import numpy as np
import mistree as mist
import pandas
import math as math
import pickle

import sys
import os
sys.path.append('/home/astro/magnan/Repository_Stage_3A')
import catalogue_tools as cat
sys.path.append("/home/astro/magnan")
from AbacusCosmos import InputFile
os.chdir('/home/astro/magnan')

print("All imports successful")

## MST Abacus
print("starting to work on the abacus simulation")

number_str = '00'
path = '/home/astro/magnan/Abacus/AbacusCosmos_720box_products/AbacusCosmos_720box_'
path += number_str
path += '_products/AbacusCosmos_720box_'
path += number_str
path += '_rockstar_halos/z0.100'

ab = cat.Catalogue_Abacus(basePath = path)
ab.initialise_data()

n_abacus = int(np.shape(ab.CM)[0])
CM_abacus = ab.CM

X = CM_abacus[:, 0]
Y = CM_abacus[:, 1]
Z = CM_abacus[:, 2]

print(np.shape(X))

MST = mist.GetMST(x = X, y = Y, z = Z)
MST_histogram = mist.HistMST()
MST_histogram.setup(usenorm = False, uselog = True)
d_abacus, l_abacus, b_abacus, s_abacus, l_index_abacus, b_index_abacus = MST.get_stats(include_index=True)
MST_histogram = MST_histogram.get_hist(d_abacus, l_abacus, b_abacus, s_abacus)

target = "/home/astro/magnan/Repository_Stage_3A/Test_masks/Full_Abacus"

np.savetxt(str(target) + "_X_d", MST_histogram['x_d'])
np.savetxt(str(target) + "_Y_d", MST_histogram['y_d'])
np.savetxt(str(target) + "_X_l", MST_histogram['x_l'])
np.savetxt(str(target) + "_Y_l", MST_histogram['y_l'])
np.savetxt(str(target) + "_X_b", MST_histogram['x_b'])
np.savetxt(str(target) + "_Y_b", MST_histogram['y_b'])
np.savetxt(str(target) + "_X_s", MST_histogram['x_s'])
np.savetxt(str(target) + "_Y_s", MST_histogram['y_s'])

print("Abacus simulation extracted")

## MST Random
print("starting to work on the random catalogue")

X = np.random.random(n_abacus) * 720
Y = np.random.random(n_abacus) * 720
Z = np.random.random(n_abacus) * 720

CM_random = np.array([[X[i], Y[i], Z[i]] for i in range(n_abacus)])

MST = mist.GetMST(x = X, y = Y, z = Z)
MST_histogram = mist.HistMST()
MST_histogram.setup(usenorm = False, uselog = True)
d_random, l_random, b_random, s_random, l_index_random, b_index_random = MST.get_stats(include_index=True)
MST_histogram = MST_histogram.get_hist(d_random, l_random, b_random, s_random)

target = "/home/astro/magnan/Repository_Stage_3A/Test_masks/Full_Random"

np.savetxt(str(target) + "_X_d", MST_histogram['x_d'])
np.savetxt(str(target) + "_Y_d", MST_histogram['y_d'])
np.savetxt(str(target) + "_X_l", MST_histogram['x_l'])
np.savetxt(str(target) + "_Y_l", MST_histogram['y_l'])
np.savetxt(str(target) + "_X_b", MST_histogram['x_b'])
np.savetxt(str(target) + "_Y_b", MST_histogram['y_b'])
np.savetxt(str(target) + "_X_s", MST_histogram['x_s'])
np.savetxt(str(target) + "_Y_s", MST_histogram['y_s'])

print("Random catalogue constructed")

## Treating the masked catalogues
print("starting to work on the masked catalogues")

MST_hists_abacus = []
MST_hists_random = []

X_d_estimator = [np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0])]
X_l_estimator = [np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0])]
X_b_estimator = [np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0])]
X_s_estimator = [np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0])]
Y_d_estimator = [np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0])]
Y_l_estimator = [np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0])]
Y_b_estimator = [np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0])]
Y_s_estimator = [np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0])]
Y_d_std_estimator = [np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0])]
Y_l_std_estimator = [np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0])]
Y_b_std_estimator = [np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0])]
Y_s_std_estimator = [np.array([0, 0]), np.array([0, 0]), np.array([0, 0]), np.array([0, 0])]

for m in range(4):
	histogram = mist.HistMST()
	histogram.setup(usenorm = False, uselog = True)
	histogram.start_group()
	MST_hists_abacus.append(histogram)
    
	histogram = mist.HistMST()
	histogram.setup(usenorm = False, uselog = True)
	histogram.start_group()
	MST_hists_random.append(histogram)

for i in range(4):
	for j in range(4):
		for k in range(4):
				number = i*16 + j*4 + k
				print("starting to work on subsample " + str(number + 1))
            
				# determining if center, face, side or corner :
				m = 0
				if ((i == 0) or (i == 3)):
					m += 1
				if ((j == 0) or (j == 3)):
					m += 1
				if ((k == 0) or (k == 3)):
					m += 1
            
				# determining the limits of the small box
				lim_inf_x, lim_sup_x = (720 / 4) * i, (720 / 4) * (i + 1)
				lim_inf_y, lim_sup_y = (720 / 4) * j, (720 / 4) * (j + 1)
				lim_inf_z, lim_sup_z = (720 / 4) * k, (720 / 4) * (k + 1)
            
				# finding nodes in the small box
				def toRemove(cm):
					if ((cm[0] > lim_inf_x) and (cm[0] < lim_sup_x)):
						if ((cm[1] > lim_inf_y) and (cm[1] < lim_sup_y)):
								if ((cm[2] > lim_inf_z) and (cm[2] < lim_sup_z)):
									return(True)
					return(False)
                
				# making the smaller catalogue
				X_abacus_reduced, Y_abacus_reduced, Z_abacus_reduced = [], [], []
				X_random_reduced, Y_random_reduced, Z_random_reduced = [], [], []
				for n in range(n_abacus):
					if not (toRemove(CM_abacus[n])):
						X_abacus_reduced.append(CM_abacus[n, 0])
						Y_abacus_reduced.append(CM_abacus[n, 1])
						Z_abacus_reduced.append(CM_abacus[n, 2])
					if not (toRemove(CM_random[n])):
						X_random_reduced.append(CM_random[n, 0])
						Y_random_reduced.append(CM_random[n, 1])
						Z_random_reduced.append(CM_random[n, 2])
				X_abacus_reduced = np.asarray(X_abacus_reduced)
				Y_abacus_reduced = np.asarray(Y_abacus_reduced)
				Z_abacus_reduced = np.asarray(Z_abacus_reduced)
				X_random_reduced = np.asarray(X_random_reduced)
				Y_random_reduced = np.asarray(Y_random_reduced)
				Z_random_reduced = np.asarray(Z_random_reduced)
				
				print(np.shape(X_abacus_reduced))
            
				# computing the smaller MST
				MST = mist.GetMST(x = X_abacus_reduced, y = Y_abacus_reduced, z = Z_abacus_reduced)
				d_abacus_reduced, l_abacus_reduced, b_abacus_reduced, s_abacus_reduced, l_index_abacus_reduced, b_index_abacus_reduced = MST.get_stats(include_index=True)
				MST = mist.GetMST(x = X_random_reduced, y = Y_random_reduced, z = Z_random_reduced)
				d_random_reduced, l_random_reduced, b_random_reduced, s_random_reduced, l_index_random_reduced, b_index_random_reduced = MST.get_stats(include_index=True)
            
				# getting and storing the MST stats
				_hist = MST_hists_abacus[m].get_hist(d_abacus_reduced, l_abacus_reduced, b_abacus_reduced, s_abacus_reduced)
				_hist = MST_hists_random[m].get_hist(d_random_reduced, l_random_reduced, b_random_reduced, s_random_reduced)
            
				# getting the MST stats in prevision of the correction
				hist = mist.HistMST()
				hist.setup(usenorm = False, uselog = True)
				hist_abacus = hist.get_hist(d_abacus_reduced, l_abacus_reduced, b_abacus_reduced, s_abacus_reduced)
				hist = mist.HistMST()
				hist.setup(usenorm = False, uselog = True)
				hist_random = hist.get_hist(d_random_reduced, l_random_reduced, b_random_reduced, s_random_reduced)
            
				# computing the correction estimator for this small box
				if (number == 0): # we define Xs only for the first small box, and keep these for the others
						for mm in range(4):
								X_d_estimator[mm] = hist_abacus['x_d'].copy()
								X_l_estimator[mm] = hist_abacus['x_l'].copy()
								X_b_estimator[mm] = hist_abacus['x_b'].copy()
								X_s_estimator[mm] = hist_abacus['x_s'].copy()
            
				Y_d_estimator_new = np.array([0.0 for n in range(np.shape(X_d_estimator[m])[0])]) # we'll these with the values of the estimator
				Y_l_estimator_new = np.array([0.0 for n in range(np.shape(X_l_estimator[m])[0])])
				Y_b_estimator_new = np.array([0.0 for n in range(np.shape(X_b_estimator[m])[0])])
				Y_s_estimator_new = np.array([0.0 for n in range(np.shape(X_s_estimator[m])[0])])
            
				for n1 in range(np.shape(X_d_estimator[m])[0]): # before computing the estimator, we have to match the X to the one from the first box
					x1 = X_d_estimator[m][n1]
					
					min = 10
					n_min_random = 0
					for n2 in range(np.shape(hist_random['x_d'])[0]):
						x2 = hist_random['x_d'][n2]
						if (abs(x1 - x2) < min):
								min = abs(x1 - x2)
								n_min_random = n2
					
					min = 10
					n_min_abacus = 0
					for n2 in range(np.shape(hist_abacus['x_d'])[0]):
						x2 = hist_abacus['x_d'][n2]
						if (abs(x1 - x2) < min):
								min = abs(x1 - x2)
								n_min_abacus = n2
					
					Y_d_estimator_new[n1] = hist_abacus['y_d'][n_min_abacus] / hist_random['y_d'][n_min_random] - 1
            
				for n1 in range(np.shape(X_l_estimator[m])[0]):
					x1 = X_l_estimator[m][n1]
					
					min = 10
					n_min_random = 0
					for n2 in range(np.shape(hist_random['x_l'])[0]):
						x2 = hist_random['x_l'][n2]
						if (abs(x1 - x2) < min):
								min = abs(x1 - x2)
								n_min_random = n2
					
					min = 10
					n_min_abacus = 0
					for n2 in range(np.shape(hist_abacus['x_l'])[0]):
						x2 = hist_abacus['x_l'][n2]
						if (abs(x1 - x2) < min):
								min = abs(x1 - x2)
								n_min_abacus = n2
					
					try: # we try a smoothing for l, b and s because the values of x can be quite different from one box to another
						Y_l_estimator_new[n1] = ((hist_abacus['y_l'][n_min_abacus - 1] / hist_random['y_l'][n_min_random - 1] - 1) + (hist_abacus['y_l'][n_min_abacus] / hist_random['y_l'][n_min_random] - 1) + (hist_abacus['y_l'][n_min_abacus + 1] / hist_random['y_l'][n_min_random + 1] - 1)) / 3
					except:
						try:
							Y_l_estimator_new[n1] = hist_abacus['y_l'][n_min_abacus] / hist_random['y_l'][n_min_random] - 1
						except:
							Y_l_estimator_new[n1] = + math.inf
            
				for n1 in range(np.shape(X_b_estimator[m])[0]):
					x1 = X_b_estimator[m][n1]
					
					min = 10
					n_min_random = 0
					for n2 in range(np.shape(hist_random['x_b'])[0]):
						x2 = hist_random['x_b'][n2]
						if (abs(x1 - x2) < min):
								min = abs(x1 - x2)
								n_min_random = n2
					
					min = 10
					n_min_abacus = 0
					for n2 in range(np.shape(hist_abacus['x_b'])[0]):
						x2 = hist_abacus['x_b'][n2]
						if (abs(x1 - x2) < min):
								min = abs(x1 - x2)
								n_min_abacus = n2
					
					try:
						Y_b_estimator_new[n1] = ((hist_abacus['y_b'][n_min_abacus - 1] / hist_random['y_b'][n_min_random - 1] - 1) + (hist_abacus['y_b'][n_min_abacus] / hist_random['y_b'][n_min_random] - 1) + (hist_abacus['y_b'][n_min_abacus + 1] / hist_random['y_b'][n_min_random + 1] - 1)) / 3
					except:
						try:
							Y_b_estimator_new[n1] = hist_abacus['y_b'][n_min_abacus] / hist_random['y_b'][n_min_random] - 1
						except:
							Y_b_estimator_new[n1] = + math.inf
            
				for n1 in range(np.shape(X_s_estimator[m])[0]):
					x1 = X_s_estimator[m][n1]
					
					min = 10
					n_min_random = 0
					for n2 in range(np.shape(hist_random['x_s'])[0]):
						x2 = hist_random['x_s'][n2]
						if (abs(x1 - x2) < min):
								min = abs(x1 - x2)
								n_min_random = n2
					
					min = 10
					n_min_abacus = 0
					for n2 in range(np.shape(hist_abacus['x_s'])[0]):
						x2 = hist_abacus['x_s'][n2]
						if (abs(x1 - x2) < min):
								min = abs(x1 - x2)
								n_min_abacus = n2
					
					try:
						Y_s_estimator_new[n1] = ((hist_abacus['y_s'][n_min_abacus - 1] / hist_random['y_s'][n_min_random - 1] - 1) + (hist_abacus['y_s'][n_min_abacus] / hist_random['y_s'][n_min_random] - 1) + (hist_abacus['y_s'][n_min_abacus + 1] / hist_random['y_s'][n_min_random + 1] - 1)) / 3
					except:
						try:
							Y_s_estimator_new[n1] = hist_abacus['y_s'][n_min_abacus] / hist_random['y_s'][n_min_random] - 1
						except:
							Y_s_estimator_new[n1] = + math.inf
            
				# updating the mean and std of the estimator
				if (np.shape(Y_d_estimator[m])[0] == 2):
					Y_d_estimator[m] = Y_d_estimator_new.copy()
					Y_d_std_estimator[m] = np.array([0.0 for n in range(np.shape(X_d_estimator[m])[0])])
				else:
					Y_d_estimator_old = Y_d_estimator[m].copy()
					Y_d_std_estimator_old = Y_d_std_estimator[m].copy()
                
					Y_d_estimator[m] = (number * Y_d_estimator_old + Y_d_estimator_new.copy()) / (number + 1)
					Y_d_std_estimator[m] = np.sqrt((number * (Y_d_estimator_old**2 + Y_d_std_estimator_old**2) + Y_d_estimator_new.copy()**2) / (number + 1) - Y_d_estimator[m].copy())
            
				if (np.shape(Y_l_estimator[m])[0] == 2):
					Y_l_estimator[m] = Y_l_estimator_new.copy()
					Y_l_std_estimator[m] = np.array([0.0 for n in range(np.shape(X_l_estimator[m])[0])])
				else:
					Y_l_estimator_old = Y_l_estimator[m].copy()
					Y_l_std_estimator_old = Y_l_std_estimator[m].copy()
                
					Y_l_estimator[m] = (number * Y_l_estimator_old + Y_l_estimator_new.copy()) / (number + 1)
					Y_l_std_estimator[m] = np.sqrt((number * (Y_l_estimator_old**2 + Y_l_std_estimator_old**2) + Y_l_estimator_new.copy()**2) / (number + 1) - Y_l_estimator[m].copy())
            
				if (np.shape(Y_b_estimator[m])[0] == 2):
					Y_b_estimator[m] = Y_b_estimator_new.copy()
					Y_b_std_estimator[m] = np.array([0.0 for n in range(np.shape(X_b_estimator[m])[0])])
				else:
					Y_b_estimator_old = Y_b_estimator[m].copy()
					Y_b_std_estimator_old = Y_b_std_estimator[m].copy()
                
					Y_b_estimator[m] = (number * Y_b_estimator_old + Y_b_estimator_new.copy()) / (number + 1)
					Y_b_std_estimator[m] = np.sqrt((number * (Y_b_estimator_old**2 + Y_b_std_estimator_old**2) + Y_b_estimator_new.copy()**2) / (number + 1) - Y_b_estimator[m].copy())
            
				if (np.shape(Y_s_estimator[m])[0] == 2):
					Y_s_estimator[m] = Y_s_estimator_new.copy()
					Y_s_std_estimator[m] = np.array([0.0 for n in range(np.shape(X_s_estimator[m])[0])])
				else:
					Y_s_estimator_old = Y_s_estimator[m].copy()
					Y_s_std_estimator_old = Y_s_std_estimator[m].copy()
                
					Y_s_estimator[m] = (number * Y_s_estimator_old + Y_s_estimator_new.copy()) / (number + 1)
					Y_s_std_estimator[m] = np.sqrt((number * (Y_s_estimator_old**2 + Y_s_std_estimator_old**2) + Y_s_estimator_new.copy()**2) / (number + 1) - Y_s_estimator[m].copy())
            
				print("box treated.")

# closing the grouped histograms
for m in range(4):
	MST_hists_abacus[m] = MST_hists_abacus[m].end_group()
	MST_hists_random[m] = MST_hists_random[m].end_group()

## Saving the statistics
print("Starting to save the results")
labels = ['center', 'face', 'edge', 'corner']

# Abacus Masked
for m in range(4):
	my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Test_masks/')
	my_file = 'Masked_abacus_' + labels[m] + '.pkl'
	my_file = os.path.join(my_path, my_file)
	
	f = open(my_file, "wb")
	pickle.dump(MST_hists_abacus[m], f)
	f.close()

# Random Masked
for m in range(4):
	my_path = os.path.abspath('/home/astro/magnan/Repository_Stage_3A/Test_masks/')
	my_file = 'Masked_random_' + labels[m] + '.pkl'
	my_file = os.path.join(my_path, my_file)
	
	f = open(my_file, "wb")
	pickle.dump(MST_hists_random[m], f)
	f.close()

# Estimator
target = "/home/astro/magnan/Repository_Stage_3A/Test_masks/estimator_"

for m in range(4):
	np.savetxt(str(target) + labels[m] + "_X_d", X_d_estimator[m])
	np.savetxt(str(target) + labels[m] + "_Y_d", Y_d_estimator[m])
	np.savetxt(str(target) + labels[m] + "_Y_d_std", Y_d_std_estimator[m])
	np.savetxt(str(target) + labels[m] + "_X_l", X_l_estimator[m])
	np.savetxt(str(target) + labels[m] + "_Y_l", Y_l_estimator[m])
	np.savetxt(str(target) + labels[m] + "_Y_l_std", Y_l_std_estimator[m])
	np.savetxt(str(target) + labels[m] + "_X_b", X_b_estimator[m])
	np.savetxt(str(target) + labels[m] + "_Y_b", Y_b_estimator[m])
	np.savetxt(str(target) + labels[m] + "_Y_b_std", Y_b_std_estimator[m])
	np.savetxt(str(target) + labels[m] + "_X_s", X_s_estimator[m])
	np.savetxt(str(target) + labels[m] + "_Y_s", Y_s_estimator[m])
	np.savetxt(str(target) + labels[m] + "_Y_s_std", Y_s_std_estimator[m])
    